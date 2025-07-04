#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random, torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.nn import functional as F
import numpy as np

# 1. 训练参数设置
model_name = "/home/kas/kas_workspace/model/Reward/Qwen3-4B/"  # 模型名或本地路径
file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/rm_infinity/score_dpo_train.jsonl"  # 测试集路径
#file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/rm_infinity/train_sample.jsonl"  # 测试集路径
save_path = "/home/kas/kas_workspace/share/chenkai/playground/output/cls_prompt/qwen3_rewared_3epoch_0702"  # 保存路径
logging_dir = save_path + "/tensorboard/"  # 日志目录

num_train_epochs = 3
per_device_train_batch_size = 32  # 降低批处理大小（RM需要更多显存）
gradient_accumulation = 2        # 增加梯度累积
per_device_eval_batch_size = 4
warmup_steps = 25
weight_decay = 0.01
logging_steps = 1
lr = 2e-5  # 更小的学习率
lr_scheduler_type = "cosine"
max_length = 2048  # 最大长度

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

# Reward Model使用回归头 (num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=1,  # 关键修改：输出单个分数值
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id

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

def preprocess_function(examples):
    """处理单个样本生成chosen/rejected对"""
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
        
        # 添加到批次
        batch["chosen_input_ids"].append(chosen_enc)
        batch["chosen_attention_mask"].append([1] * len(chosen_enc))
        batch["rejected_input_ids"].append(rejected_enc)
        batch["rejected_attention_mask"].append([1] * len(rejected_enc))
    
    return batch

# 读取数据
data = read_jsonl(file_path)
random.shuffle(data)

# 转换为Dataset
dataset = Dataset.from_list(data)

# 拆分训练集/验证集
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# 预处理数据集
encoded_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    batch_size=32
)

# 4. 自定义数据整理器
class RewardDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": []
        }
        
        for feature in features:
            for key in ["chosen", "rejected"]:
                batch[f"{key}_input_ids"].append(feature[f"{key}_input_ids"])
                batch[f"{key}_attention_mask"].append(feature[f"{key}_attention_mask"])
        
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

data_collator = RewardDataCollator(tokenizer=tokenizer)

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

# 6. 训练配置
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
    bf16=True,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    optim="adamw_torch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
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
print("开始训练Reward Model...")
trainer.train()

# 9. 保存模型
print("训练完成，保存模型...")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"模型已保存到: {save_path}")
