#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import json

# 1. 导入自定义模型结构（必须与训练代码一致）
class MultiTaskQwenModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        config = base_model.config
        
        # 创建三个独立的分类头
        self.clarity_head = torch.nn.Linear(config.hidden_size, 5).to(dtype=base_model.dtype)
        self.complexity_head = torch.nn.Linear(config.hidden_size, 5).to(dtype=base_model.dtype)
        self.quality_head = torch.nn.Linear(config.hidden_size, 5).to(dtype=base_model.dtype)
        
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0]
        pooled_output = pooled_output.to(dtype=self.clarity_head.weight.dtype)
        
        clarity_logits = self.clarity_head(pooled_output)
        complexity_logits = self.complexity_head(pooled_output)
        quality_logits = self.quality_head(pooled_output)
        
        return clarity_logits, complexity_logits, quality_logits

# 2. 加载模型和分词器
model_path = "/home/kas/kas_workspace/share/chenkai/playground/output/multi_prompt/qwen3_multitask_3epoch"


print(f"加载模型和分词器，路径: {model_path}")
# 加载基础模型
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
print("基础模型加载成功！")
# 创建自定义模型结构
model = MultiTaskQwenModel(base_model)
model = model.from_pretrained(model_path)
#model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
model.eval().cuda()  # 使用GPU推理
print("自定义模型加载成功！")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 推理函数
def predict(text):
    # 预处理输入
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to("cuda")
    
    # 执行推理
    with torch.no_grad():
        clarity_logits, complexity_logits, quality_logits = model(**inputs)
    
    # 获取预测结果
    clarity_pred = torch.argmax(clarity_logits, dim=1).item()
    complexity_pred = torch.argmax(complexity_logits, dim=1).item()
    quality_pred = torch.argmax(quality_logits, dim=1).item()
    
    return {
        "clarity": clarity_pred,
        "complexity": complexity_pred,
        "quality": quality_pred
    }

# 4. 使用示例
if __name__ == "__main__":
    sample_text = "这是一个需要分类的示例文本..."
    
    # 单条文本推理
    result = predict(sample_text)
    print("单条文本推理结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 批量推理示例
    batch_texts = [
        "请您扮演《原神》游戏中的珊瑚宫心海来讲几句台词。\n我的心在久久酝酿之后，终于在此刻绽放。这是我的信念，也是我的使命。",
        "我们团队制作的一份报告中，提到到2025年预计全球新能源汽车的市场销量将达到2000万辆。这个数据看起来很有前景，但我想知道，这个预测数据基于哪些关键因素？",
        "提供一些设置灵感的一些乡村生活场景，这些有助于表达人物特征"
    ]
    
    print("\n批量推理结果:")
    for text in batch_texts:
        result = predict(text)
        print(f"文本: {text[:20]}... | 结果: {result}")