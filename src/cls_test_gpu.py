#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm

# 配置参数
TEST_FILE = "/home/kas/kas_workspace/share/chenkai/playground/dataset/cls_prompt/cls_test.jsonl"
MODEL_PATH = "/home/kas/kas_workspace/share/chenkai/playground/output/cls_prompt/qwen3_cls_3epoch"
BATCH_SIZE = 4  # 根据GPU显存调整批次大小

# 标签映射定义
LABEL_MAP = {
    "世界知识问答": 0, "开放域问答": 1, "常识推理": 2, "逻辑推理": 3,
    "COT推理": 4, "代码": 5, "数学": 6, "角色扮演": 7, "翻译": 8,
    "阅读理解": 9, "信息抽取": 10, "文本改写": 11, "文本摘要": 12,
    "文本纠错": 13, "文本分类": 14, "意图识别": 15, "文本写作": 16, 
    "创意设计": 17, "其他类别": 18
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}  # 创建反向映射

def load_test_data(file_path):
    """从jsonl文件加载测试数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 处理可能的多行文本数据
            if isinstance(item['text'], list):
                item['text'] = '\n'.join(item['text'])
            data.append((item['text'], item['label']))
    return data

def main():
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    # 加载测试数据
    test_data = load_test_data(TEST_FILE)
    texts, true_labels = zip(*test_data)
    
    print(f"Loaded {len(texts)} test samples from {TEST_FILE}")
    
    # 分批处理预测
    predictions, batch_texts = [], []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="预测进度"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        predictions.extend([ID_TO_LABEL[p] for p in batch_preds])
        batch_texts.extend(batch)
    
    # 计算并显示准确率
    correct_count = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct_count / len(true_labels)

    # 打印详细结果（可选）
    for i, (text, true_label) in enumerate(zip(batch_texts, true_labels)):
        pred_label = predictions[i]
        status = "✓" if pred_label == true_label else "✗"
        print(f"[{status}] 样本 {i+1}:\n  文本: {text[:50]}...\n  真实: {true_label}\n  预测: {pred_label}\n")

    print("\n" + "="*50)
    print(f"测试结果 (样本数: {len(true_labels)})")
    print(f"正确预测: {correct_count}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()