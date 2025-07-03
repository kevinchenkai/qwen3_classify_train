#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoConfig, AutoTokenizer, Qwen3PreTrainedModel, Qwen3Model
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file
from typing import Optional, Tuple

# 定义自定义模型结构（与推理代码相同）
class MultiTaskQwenModel(Qwen3PreTrainedModel):
    config_class = AutoConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config, base_model=None):
        super().__init__(config)
        
        if base_model is None:
            self.model = Qwen3Model(config)
        else:
            self.model = base_model

        dtype = torch.bfloat16
        self.accuracy_head = torch.nn.Linear(config.hidden_size, 10).to(dtype)
        self.satisfaction_head = torch.nn.Linear(config.hidden_size, 10).to(dtype)
        self.coherence_head = torch.nn.Linear(config.hidden_size, 10).to(dtype)        

        self._init_weights(self.accuracy_head)
        self._init_weights(self.satisfaction_head)
        self._init_weights(self.coherence_head)
        
    def _init_weights(self, module):
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
        **kwargs
    ) -> dict:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        seq_lengths = attention_mask.sum(dim=1) - 1
        row_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled_output = hidden_states[row_indices, seq_lengths]     

        accuracy_logits = self.accuracy_head(pooled_output)
        satisfaction_logits = self.satisfaction_head(pooled_output)
        coherence_logits = self.coherence_head(pooled_output)
        
        return {
            "accuracy_logits": accuracy_logits,
            "satisfaction_logits": satisfaction_logits,
            "coherence_logits": coherence_logits
        }

# 加载模型和分词器
test_file_path = "/home/kas/kas_workspace/share/chenkai/playground/dataset/score_labels/score_model_test.jsonl"
model_path = "/home/kas/kas_workspace/share/chenkai/playground/output/score_labels/qwen3_3score_3epoch_0703"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MultiTaskQwenModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("模型加载完成，模型结构如下:")
print(model)
#print(f"模型已加载到 {device} 设备")

# 推理函数（稍作修改用于评测）
def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    accuracy_pred = torch.argmax(outputs["accuracy_logits"], dim=1).item()
    satisfaction_pred = torch.argmax(outputs["satisfaction_logits"], dim=1).item()
    coherence_pred = torch.argmax(outputs["coherence_logits"], dim=1).item()
    
    return [accuracy_pred, satisfaction_pred, coherence_pred]

# 评测函数
def evaluate(test_file):
    # 加载测试数据
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 存储预测结果和真实标签
    all_preds = []
    all_labels = []
    
    # 逐个样本预测
    for sample in tqdm(samples, desc="评测进度"):
        text = sample["text"]
        label = sample["label"]  # 真实标签 [accuracy, satisfaction, coherence]
        
        try:
            pred = predict(text)  # 模型预测 [accuracy, satisfaction, coherence]
            all_preds.append(pred)
            all_labels.append(label)
        except Exception as e:
            print(f"处理样本时出错: {text[:100]}... 错误: {str(e)}")
            continue

        #print(f"样本预测: {pred}, 真实标签: {label}")
    
    # 转换为numpy数组便于计算
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算各项准确率
    total = len(all_labels)
    """
    accuracy_correct = np.sum(all_preds[:, 0] == all_labels[:, 0])
    satisfaction_correct = np.sum(all_preds[:, 1] == all_labels[:, 1])
    coherence_correct = np.sum(all_preds[:, 2] == all_labels[:, 2])
    all_correct = np.sum(np.all(all_preds == all_labels, axis=1))
    """
    accuracy_correct = np.sum(np.abs(all_preds[:, 0] - all_labels[:, 0]) <= 1)
    satisfaction_correct = np.sum(np.abs(all_preds[:, 1] - all_labels[:, 1]) <= 1)
    coherence_correct = np.sum(np.abs(all_preds[:, 2] - all_labels[:, 2]) <= 1)

    # 总体计数（同时满足三个维度条件）
    diff = np.abs(all_preds - all_labels)
    all_correct = np.sum(np.all(diff <= 1, axis=1))    
    
    accuracy_acc = accuracy_correct / total
    satisfaction_acc = satisfaction_correct / total
    coherence_acc = coherence_correct / total
    overall_acc = all_correct / total
    
    # 打印结果
    print("\n" + "="*50)
    print(f"评测结果 (样本总数: {total})")
    print("="*50)
    print(f"准确度 (accuracy) 准确率: {accuracy_acc:.4f} ({accuracy_correct}/{total})")
    print(f"满意度 (satisfaction) 准确率: {satisfaction_acc:.4f} ({satisfaction_correct}/{total})")
    print(f"连贯度 (coherence) 准确率: {coherence_acc:.4f} ({coherence_correct}/{total})")
    print("-"*50)
    print(f"整体准确率 (三项全对): {overall_acc:.4f} ({all_correct}/{total})")
    print("="*50)
    
    # 返回结果
    return {
        "accuracy_accuracy": accuracy_acc,
        "satisfaction_accuracy": satisfaction_acc,
        "coherence_accuracy": coherence_acc,
        "overall_accuracy": overall_acc,
        "total_samples": total
    }

if __name__ == "__main__":
    # 替换为您的测试文件路径    
    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在 - {test_file_path}")
    else:
        results = evaluate(test_file_path)
