#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random, torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.nn import functional as F
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler

# 分布式训练初始化
def setup_distributed():
    # 使用 argparse 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1))
    args, _ = parser.parse_known_args()  # 忽略未知参数
    if args.local_rank == -1:
        return -1  # 单GPU模式
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return args.local_rank

# 3. 数据读取和预处理
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_conversation(conversations):
    """将对话转换为Qwen格式"""
    formatted = []
    for msg in conversations:
        role = "user" if msg["from"] == "human" else "assistant"
        formatted.append({"role": role, "content": msg["value"]})
    return formatted

# 4. 自定义数据整理器 (修复版)
class RewardDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = {
            "chosen_input_ids": [f["chosen_input_ids"] for f in features],
            "chosen_attention_mask": [f["chosen_attention_mask"] for f in features],
            "rejected_input_ids": [f["rejected_input_ids"] for f in features],
            "rejected_attention_mask": [f["rejected_attention_mask"] for f in features]
        }
        
        # 分别处理chosen和rejected
        chosen_batch = super().__call__([
            {"input_ids": ids, "attention_mask": mask} 
            for ids, mask in zip(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        ])
        
        rejected_batch = super().__call__([
            {"input_ids": ids, "attention_mask": mask} 
            for ids, mask in zip(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        ])
        
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"]
        }

# 5. 自定义Trainer实现Pairwise Loss
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 前向传播计算chosen/rejected分数
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )
        
        # 获取分数logits
        chosen_scores = chosen_outputs.logits
        rejected_scores = rejected_outputs.logits
        
        # 计算Pairwise Ranking Loss
        loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
        
        if return_outputs:
            return loss, {
                "chosen_scores": chosen_scores,
                "rejected_scores": rejected_scores
            }
        return loss

def main():
    # 初始化分布式训练
    local_rank = setup_distributed()
    
    # 1. 训练参数设置
    model_name = "/home/kas/kas_workspace/model/Reward/Qwen3-4B/"
    file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/rm_infinity/score_dpo_train.jsonl"  # 测试集路径
    save_path = "/home/kas/kas_workspace/share/chenkai/playground/output/cls_prompt/qwen3_rewared_3epoch_ddp_0704"  # 保存路径
    logging_dir = save_path + "/tensorboard/"

    num_train_epochs = 3
    per_device_train_batch_size = 2  # 每个GPU的批处理大小
    gradient_accumulation = 4        # 梯度累积步数
    per_device_eval_batch_size = 4
    warmup_steps = 30
    weight_decay = 0.01
    logging_steps = 10
    lr = 5e-5
    lr_scheduler_type = "cosine"
    max_length = 4096

    # 2. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 只在主进程打印信息
    if local_rank in [-1, 0]:
        print(f"Loading model on rank {local_rank}...")

    # Reward Model使用回归头
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,
        #attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # 将模型移到当前设备
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")
    model = model.to(device)

    # 读取数据
    if local_rank in [-1, 0]:
        print("Loading dataset...")
    data = read_jsonl(file_path)
    random.shuffle(data)
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # 数据预处理函数
    def preprocess_function(examples):
        batch = {"chosen_input_ids": [], "chosen_attention_mask": [], 
                "rejected_input_ids": [], "rejected_attention_mask": []}
        
        for conv, chosen, rejected in zip(examples['conversations'], 
                                        examples['chosen'], 
                                        examples['rejected']):
            # 构建完整对话
            full_conv = format_conversation(conv)
            
            # Chosen路径
            chosen_conv = full_conv + [{"role": "assistant", "content": chosen["value"]}]
            chosen_enc = tokenizer.apply_chat_template(
                chosen_conv,
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            # Rejected路径
            rejected_conv = full_conv + [{"role": "assistant", "content": rejected["value"]}]
            rejected_enc = tokenizer.apply_chat_template(
                rejected_conv,
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            batch["chosen_input_ids"].append(chosen_enc)
            batch["chosen_attention_mask"].append([1] * len(chosen_enc))
            batch["rejected_input_ids"].append(rejected_enc)
            batch["rejected_attention_mask"].append([1] * len(rejected_enc))
        
        return batch

    # 预处理数据集
    encoded_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=32
    )

    # 数据整理器
    data_collator = RewardDataCollator(tokenizer=tokenizer)

    # 6. 训练配置 (分布式优化)
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        bf16=True,  # H100支持bfloat16
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        optim="adamw_torch",
        gradient_checkpointing=False,  # 减少显存占用
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # 分布式优化
        ddp_backend="nccl",
        fp16=False,  # 禁用fp16，使用bf16
        local_rank=local_rank  # 传递本地rank,
    )

    # 7. 创建Trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
    )

    # 8. 训练模型
    if local_rank in [-1, 0]:
        print(f"可见GPU数量: {torch.cuda.device_count()}")
        print(f"当前使用GPU: {torch.cuda.get_device_name()}")
        print("开始训练Reward Model...")
    
    trainer.train()

    # 9. 保存模型 (只在主进程保存)
    if local_rank in [-1, 0]:
        print("训练完成，保存模型...")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")

if __name__ == "__main__":
    main()

"""
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    rm_train_gpu.py
"""