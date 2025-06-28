#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random, torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score

# 1. 自定义多任务模型
class MultiTaskQwenModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        config = base_model.config
        
        # 创建三个独立的分类头，并指定与基础模型相同的数据类型
        self.clarity_head = torch.nn.Linear(config.hidden_size, 5).to(dtype=base_model.dtype)  # 5个类别
        self.complexity_head = torch.nn.Linear(config.hidden_size, 5).to(dtype=base_model.dtype)
        self.quality_head = torch.nn.Linear(config.hidden_size, 5).to(dtype=base_model.dtype)
        
        # 添加梯度检查点支持
        self.gradient_checkpointing = False
        
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # 添加梯度检查点支持
        if self.gradient_checkpointing and self.training:
            outputs = torch.utils.checkpoint.checkpoint(
                self.model,
                input_ids,
                attention_mask,
                output_hidden_states=True,
                use_reentrant=False
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # 获取[CLS]位置的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0]  # 使用第一个token作为分类表示
        
        # 确保分类头使用正确的数据类型
        pooled_output = pooled_output.to(dtype=self.clarity_head.weight.dtype)
        
        # 通过三个分类头
        clarity_logits = self.clarity_head(pooled_output)
        complexity_logits = self.complexity_head(pooled_output)
        quality_logits = self.quality_head(pooled_output)
        
        loss = None
        if labels is not None:
            # 确保标签使用正确的数据类型（通常是long）
            labels = labels.to(dtype=torch.long)
            
            # 拆分三个任务的标签
            clarity_labels = labels[:, 0]
            complexity_labels = labels[:, 1]
            quality_labels = labels[:, 2]
            
            # 计算三个任务的交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss()
            loss_clarity = loss_fct(clarity_logits, clarity_labels)
            loss_complexity = loss_fct(complexity_logits, complexity_labels)
            loss_quality = loss_fct(quality_logits, quality_labels)
            
            # 加权平均损失
            loss = (loss_clarity + loss_complexity + loss_quality) / 3.0
        
        return {
            "loss": loss,
            "clarity_logits": clarity_logits,
            "complexity_logits": complexity_logits,
            "quality_logits": quality_logits
        }
    
    # 添加梯度检查点启用方法
    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True
        self.model.gradient_checkpointing_enable(**kwargs)
    
    # 添加梯度检查点禁用方法
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.model.gradient_checkpointing_disable()

# 2. 训练参数设置
model_name = "/home/kas/kas_workspace/model/Reward/Qwen3-4B/"
file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/cls_mlabels/rm_labels_train.jsonl"
save_path = "/home/kas/kas_workspace/share/chenkai/playground/output/multi_prompt/qwen3_multitask_3epoch"
logging_dir = save_path + "/tensorboard/"

num_train_epochs = 3
per_device_train_batch_size = 16
gradient_accumulation = 2
per_device_eval_batch_size = 8
warmup_steps = 25
weight_decay = 0.01
logging_steps = 1
lr = 2e-5
lr_scheduler_type = "cosine"

# 3. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=19,  # 原始模型标签数量
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
base_model.config.pad_token_id = 151643

# 使用自定义模型
model = MultiTaskQwenModel(base_model)

# 4. 读取JSONL数据
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 确保标签格式正确
            if 'label' in item and isinstance(item['label'], list) and len(item['label']) == 3:
                data.append({
                    'text': item['text'],
                    'label': item['label']
                })
            else:
                print(f"警告: 跳过无效标签格式的样本: {item}")
    return data

data = read_jsonl(file_path)
random.shuffle(data)

print(f"加载了 {len(data)} 个训练样本")

# 5. 创建数据集
dataset = Dataset.from_list(data)

# 拆分训练集和验证集
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# 6. 数据预处理
def preprocess_function(examples):
    # 文本分词
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=2048
    )
    
    # 处理三维标签
    labels = examples['label']
    
    # 返回包含标签的结果
    return {
        **tokenized,
        "labels": labels  # 直接存储标签列表
    }

# 应用预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 打印数据集结构以验证
print("数据集特征:", encoded_dataset['train'].features)

# 7. 自定义数据整理器
class MultiLabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # 首先处理输入数据
        batch = super().__call__(features)
        
        # 收集标签
        labels = []
        for feature in features:
            if 'labels' in feature:
                labels.append(feature['labels'])
            else:
                # 如果缺少标签，使用默认值 [0, 0, 0]
                print(f"警告: 样本缺少标签，使用默认值 [0,0,0]")
                labels.append([0, 0, 0])
        
        # 转换为张量
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return batch

data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

# 8. 自定义评估指标计算
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 确保我们有三个预测输出
    if not isinstance(predictions, tuple) or len(predictions) != 3:
        # 如果不是元组或长度不对，尝试从单个数组提取
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3 and predictions.shape[1] == 3:
            clarity_preds = np.argmax(predictions[:, 0], axis=1)
            complexity_preds = np.argmax(predictions[:, 1], axis=1)
            quality_preds = np.argmax(predictions[:, 2], axis=1)
        else:
            print(f"错误: 预测格式无效 - {type(predictions)}")
            return {}
    else:
        clarity_preds = np.argmax(predictions[0], axis=1)
        complexity_preds = np.argmax(predictions[1], axis=1)
        quality_preds = np.argmax(predictions[2], axis=1)
    
    clarity_labels = labels[:, 0]
    complexity_labels = labels[:, 1]
    quality_labels = labels[:, 2]
    
    # 计算三个任务的准确率
    clarity_acc = accuracy_score(clarity_labels, clarity_preds)
    complexity_acc = accuracy_score(complexity_labels, complexity_preds)
    quality_acc = accuracy_score(quality_labels, quality_preds)
    
    # 平均准确率
    avg_acc = (clarity_acc + complexity_acc + quality_acc) / 3.0
    
    return {
        "clarity_accuracy": clarity_acc,
        "complexity_accuracy": complexity_acc,
        "quality_accuracy": quality_acc,
        "average_accuracy": avg_acc
    }

# 9. 配置训练参数
training_args = TrainingArguments(
    output_dir=save_path,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    #eval_strategy="steps",
    #eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    gradient_accumulation_steps=gradient_accumulation,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    optim="adamw_torch",
    metric_for_best_model="average_accuracy",
    greater_is_better=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
)

# 10. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 11. 开始训练
print("开始训练...")
trainer.train()

# 12. 保存最终模型
print(f"训练完成，正在保存模型到 {save_path}...")
trainer.save_state()
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("模型保存完成！")
