#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoConfig,AutoTokenizer,AutoModel
from transformers import PreTrainedModel,Qwen3PreTrainedModel,Qwen3Model
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import json, random, torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple
import os

# 1. 自定义多任务模型 - 适配Qwen3结构
class MultiTaskQwenModel(Qwen3PreTrainedModel):
    config_class = AutoConfig
    supports_gradient_checkpointing = True  # 添加对梯度检查点的支持
    
    def __init__(self, config, base_model=None):
        super().__init__(config)
        
        if base_model is None:
            self.model = Qwen3Model(config)
        else:
            self.model = base_model

        # 创建三个独立的分类头
        dtype = torch.bfloat16
        self.accuracy_head = torch.nn.Linear(config.hidden_size, 10).to(dtype)
        self.satisfaction_head = torch.nn.Linear(config.hidden_size, 10).to(dtype)
        self.coherence_head = torch.nn.Linear(config.hidden_size, 10).to(dtype)        

        # 初始化分类头
        self._init_weights(self.accuracy_head)
        self._init_weights(self.satisfaction_head)
        self._init_weights(self.coherence_head)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> dict:
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一个隐藏状态
        #hidden_states = outputs.hidden_states[-1]
        hidden_states = outputs.last_hidden_state
        
        # 确定最后一个非填充token的位置
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 更安全的索引方式
        seq_lengths = attention_mask.sum(dim=1) - 1
        row_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled_output = hidden_states[row_indices, seq_lengths]     

        # 通过三个分类头
        accuracy_logits = self.accuracy_head(pooled_output)
        satisfaction_logits = self.satisfaction_head(pooled_output)
        coherence_logits = self.coherence_head(pooled_output)
        
        loss = None
        if labels is not None:
            # 拆分三个任务的标签
            accuracy_labels = labels[:, 0]
            satisfaction_labels = labels[:, 1]
            coherence_labels = labels[:, 2]
            
            # 计算三个任务的交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss_clarity = loss_fct(accuracy_logits, accuracy_labels)
            loss_complexity = loss_fct(satisfaction_logits, satisfaction_labels)
            loss_quality = loss_fct(coherence_logits, coherence_labels)
            
            # 加权平均损失
            loss = (loss_clarity + loss_complexity + loss_quality) / 3.0
        
        return {
            "loss": loss,
            "accuracy_logits": accuracy_logits,
            "satisfaction_logits": satisfaction_logits,
            "coherence_logits": coherence_logits
        }
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

# 2. 训练参数设置
model_name = "/home/kas/kas_workspace/model/Qwen3/Qwen3-4B"  # 使用较小模型便于演示
file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/score_labels/score_model_train.jsonl"
save_path = "/home/kas/kas_workspace/share/chenkai/playground/output/score_labels/qwen3_3score_3epoch_0703"
logging_dir = save_path + "/tensorboard/"

# 训练参数
num_train_epochs = 3
per_device_train_batch_size = 16
gradient_accumulation_steps = 2
per_device_eval_batch_size = 8
warmup_steps = 50
weight_decay = 0.01
logging_steps = 1
lr = 2e-5
lr_scheduler_type = "cosine"

# 3. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

# 加载基础模型
base_model = Qwen3Model.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# 使用自定义模型
config = base_model.config
config.pad_token_id = tokenizer.pad_token_id  # 确保配置中有pad_token_id
model = MultiTaskQwenModel(config, base_model=base_model)

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
        max_length=512,
        padding=False
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
                labels.append([0, 0, 0])
        
        # 转换为张量
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch

data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

# 8. 自定义评估指标计算
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # 确保我们有三个预测输出
    if isinstance(predictions, tuple) and len(predictions) == 3:
        accuracy_preds, satisfaction_preds, coherence_preds = predictions
    else:
        # 处理预测格式
        accuracy_preds = predictions[0]
        satisfaction_preds = predictions[1]
        coherence_preds = predictions[2]
    
    # 获取预测类别
    accuracy_preds = np.argmax(accuracy_preds, axis=1)
    satisfaction_preds = np.argmax(satisfaction_preds, axis=1)
    coherence_preds = np.argmax(coherence_preds, axis=1)
    
    # 提取真实标签
    accuracy_labels = labels[:, 0]
    satisfaction_labels = labels[:, 1]
    coherence_labels = labels[:, 2]
    
    # 计算三个任务的准确率
    accuracy_acc = accuracy_score(accuracy_labels, accuracy_preds)
    satisfaction_acc = accuracy_score(satisfaction_labels, satisfaction_preds)
    coherence_acc = accuracy_score(coherence_labels, coherence_preds)
    
    # 计算F1分数
    accuracy_f1 = f1_score(accuracy_labels, accuracy_preds, average='weighted')
    satisfaction_f1 = f1_score(satisfaction_labels, satisfaction_preds, average='weighted')
    coherence_f1 = f1_score(coherence_labels, coherence_preds, average='weighted')
    
    # 平均准确率和F1
    avg_acc = (accuracy_acc + satisfaction_acc + coherence_acc) / 3.0
    avg_f1 = (accuracy_f1 + satisfaction_f1 + coherence_f1) / 3.0
    
    return {
        "accuracy_accuracy": accuracy_acc,
        "satisfaction_accuracy": satisfaction_acc,
        "coherence_accuracy": coherence_acc,
        "accuracy_f1": accuracy_f1,
        "satisfaction_f1": satisfaction_f1,
        "coherence_f1": coherence_f1,
        "average_accuracy": avg_acc,
        "average_f1": avg_f1
    }

# 9. 配置训练参数
training_args = TrainingArguments(
    output_dir=save_path,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    #eval_strategy="None",  # 评估策略
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    optim="adamw_torch",
    metric_for_best_model="average_accuracy",
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
model.save_pretrained(save_path)  # 使用PreTrainedModel的保存方法
tokenizer.save_pretrained(save_path)
print("模型保存完成！")
