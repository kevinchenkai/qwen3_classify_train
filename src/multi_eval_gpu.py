#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
from safetensors.torch import load_file
from transformers import AutoConfig, PreTrainedModel

# 1. 定义与训练时相同的自定义模型结构
class MultiTaskQwenModel(PreTrainedModel):
    config_class = AutoConfig
    
    def __init__(self, config, base_model=None):
        super().__init__(config)
        
        if base_model is None:
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            self.model = base_model
        
        # 创建三个独立的分类头
        dtype = next(self.model.parameters()).dtype
        self.clarity_head = torch.nn.Linear(config.hidden_size, 5).to(dtype)
        self.complexity_head = torch.nn.Linear(config.hidden_size, 5).to(dtype)
        self.quality_head = torch.nn.Linear(config.hidden_size, 5).to(dtype)
        
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取[CLS]位置的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0]
        
        # 通过三个分类头
        clarity_logits = self.clarity_head(pooled_output)
        complexity_logits = self.complexity_head(pooled_output)
        quality_logits = self.quality_head(pooled_output)
        
        return {
            "clarity_logits": clarity_logits,
            "complexity_logits": complexity_logits,
            "quality_logits": quality_logits
        }

# 2. 加载模型和分词器
model_path = "/home/kas/kas_workspace/share/chenkai/playground/output/multi_prompt/qwen3_multitask_3epoch"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载配置
config = AutoConfig.from_pretrained(model_path)

# 加载基础模型
print(f"正在加载模型: {model_path}...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    config=config,
    attn_implementation="flash_attention_2",  # 如果设备支持FlashAttention
    torch_dtype=torch.bfloat16                # 与训练相同的精度
)

# 构建自定义模型
print("正在构建多任务模型...")
model = MultiTaskQwenModel(config, base_model=base_model)

# 加载分片的安全张量权重
def load_sharded_safetensors(model, model_path):
    # 查找所有分片文件
    shard_files = [f for f in os.listdir(model_path) 
                   if f.startswith("model-") and f.endswith(".safetensors")]
    shard_files.sort()  # 确保按顺序加载
    
    # 加载并合并状态字典
    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        state_dict.update(load_file(shard_path, device="cpu"))
    
    # 加载到模型
    model.load_state_dict(state_dict)
    return model
model = load_sharded_safetensors(model, model_path)
model.eval()  # 切换到评估模式

print("模型加载完成，模型结构如下:")
print(model)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# 3. 推理函数
def predict(text):
    # 文本编码
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    # GPU加速 (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    # 执行推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 处理输出结果
    clarity_pred = torch.argmax(outputs["clarity_logits"], dim=1).item()
    complexity_pred = torch.argmax(outputs["complexity_logits"], dim=1).item()
    quality_pred = torch.argmax(outputs["quality_logits"], dim=1).item()
    
    return {
        "text": text,
        "clarity": clarity_pred,
        "complexity": complexity_pred,
        "quality": quality_pred
    }

# 4. 使用示例
if __name__ == "__main__":
    test_texts = [
        "请您扮演《原神》游戏中的珊瑚宫心海来讲几句台词。\n我的心在久久酝酿之后，终于在此刻绽放。这是我的信念，也是我的使命。",
        "我们团队制作的一份报告中，提到到2025年预计全球新能源汽车的市场销量将达到2000万辆。这个数据看起来很有前景，但我想知道，这个预测数据基于哪些关键因素？",
        "提供一些设置灵感的一些乡村生活场景，这些有助于表达人物特征",
        "翻译成现代文：\n由畦塍中南行七里，复涉冈而南，见有鼓吹东去者，执途人问之，乃捕尉勒部过此也。",
        "请简要介绍一下JavaScript的事件模型，并以代码块形式展示一个简单的示例。",
        "在《红楼梦》中，林黛玉的性格特点是什么？请用简洁的语言概括。",
        "你好",
        "请总结以下我国法律中关于行政处罚的规定，并归纳其中重要的条款；如果行政处罚需经过多部门协商议定，请特别指出。",
    ]
    
    results = []
    for text in test_texts:
        result = predict(text)
        results.append(result)
        print(f"文本: {text}")
        print(f"清晰度: {result['clarity']}, 复杂度: {result['complexity']}, 质量: {result['quality']}")
        print("-" * 50)
    