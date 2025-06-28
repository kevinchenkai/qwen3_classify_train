#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random, torch
from datasets import Dataset
from transformers import DataCollatorWithPadding

# 1.训练参数设置
model_name = "/home/kas/kas_workspace/model/Reward/Qwen3-4B/"  # 模型配置路径
file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/cls_prompt/cls_train.jsonl"  # 定义训练集路径
save_path = "/home/kas/kas_workspace/share/chenkai/playground/output/cls_prompt/qwen3_cls_3epoch_scratch"  # 定义保存路径
logging_dir = save_path + "/tensorboard/"  # 日志目录

num_train_epochs = 3
per_device_train_batch_size = 16  # 提高批处理大小（根据GPU内存调整）
gradient_accumulation = 2  # 梯度累积步数（根据GPU内存调整）
per_device_eval_batch_size = 8  # 提高评估批处理大小
warmup_steps = 25
weight_decay = 0.01
logging_steps = 1              # 减少日志频率
lr = 1e-4  # 学习率
lr_scheduler_type = "cosine"

# 2.创建标签到索引的映射
label_to_id = {
    "世界知识问答": 0, "开放域问答": 1, "常识推理": 2, "逻辑推理": 3,
    "COT推理": 4, "代码": 5, "数学": 6, "角色扮演": 7, "翻译": 8,
    "阅读理解": 9, "信息抽取": 10, "文本改写": 11, "文本摘要": 12,
    "文本纠错": 13, "文本分类": 14, "意图识别": 15, "文本写作": 16, "创意设计": 17, "其他类别": 18
}
num_labels = len(label_to_id)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 关键修改：从配置初始化模型（随机权重）
config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels
config.attn_implementation = "flash_attention_2"
config.pad_token_id = 151643  # 确保与分词器一致

# 从配置创建模型（随机初始化权重）
model = AutoModelForSequenceClassification.from_config(config)
model = model.to(torch.bfloat16)  # 设置为bfloat16精度
print("模型已成功初始化：随机权重（From Scratch）")

# 3.读取训练jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = read_jsonl(file_path)
random.shuffle(data)

# 将文本标签转换为数值标签
for example in data:
    example['label'] = label_to_id[example['label']]

# 检查标签范围
for example in data:
    assert 0 <= example['label'] < len(label_to_id), f"Label out of range: {example['label']}"    

# 将数据转换为datasets库的Dataset对象
dataset = Dataset.from_list(data)

# 将数据集拆分为训练集和验证集
dataset = dataset.train_test_split(test_size=0.05, seed=42)  # 5%作为验证集

# 定义一个函数来处理数据集中的文本
def preprocess_function(examples):
    # 仅返回字典格式的分词结果，不转换为张量
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding=False,  # 重要：不要在此处填充
        max_length=2048  # 建议设置最大长度
    )

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=save_path,                           # 输出目录
    num_train_epochs=num_train_epochs,              # 训练的epoch数
    per_device_train_batch_size=per_device_train_batch_size,    # 每个设备的训练batch size
    per_device_eval_batch_size=per_device_eval_batch_size,      # 每个设备的评估batch size
    warmup_steps=warmup_steps,                  # 预热步数
    weight_decay=weight_decay,                  # 权重衰减
    logging_dir=logging_dir,                      # 日志目录
    logging_steps=logging_steps,
    report_to="tensorboard",                    # 使用TensorBoard记录日志
    save_strategy="steps",                      # 每个epoch保存一次检查点
    save_steps=2000,                              # 每100步保存一次检查点
    save_total_limit=3,                         # 最多保存3个检查点，旧的会被删除
    bf16=True,                                  # 启用BF16（针对H100优化）
    gradient_accumulation_steps=gradient_accumulation,   # 梯度累积（可选，根据显存调整）
    learning_rate=lr,                          # 设置学习率（根据模型调整）
    lr_scheduler_type=lr_scheduler_type,       # 使用余弦学习率调度器
    optim="adamw_torch",                        # 使用Torch实现的AdamW
    load_best_model_at_end=False,                # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",          # 根据验证损失选择最佳模型
    greater_is_better=False,                    # 损失越小越好
    gradient_checkpointing=True,  # 启用梯度检查点
    # 从零训练建议调整的参数
    ddp_find_unused_parameters=False,           # 避免分布式训练问题
    remove_unused_columns=False,                # 保留所有列
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    data_collator=data_collator,  # 关键：添加数据整理器
)

# 开始训练
trainer.train()
print(f"训练完成，正在保存模型....")

# 训练结束后保存最终模型和最佳模型
trainer.save_state()
trainer.save_model(output_dir=save_path)
tokenizer.save_pretrained(save_path)
print(f"模型已保存到 {save_path}")
