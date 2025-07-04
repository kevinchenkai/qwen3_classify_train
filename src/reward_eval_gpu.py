import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_path, device="cuda:0"):
    """加载奖励模型及其分词器"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1
        )
    except RuntimeError:  # 如果flash attention不支持则回退
        print("Flash attention not available, using default attention")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            num_labels=1
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_dataset(file_path):
    """从JSONL文件加载测试数据集"""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data["conversations"][0]['value']
            chosen = data["chosen"]['value']
            rejected = data["rejected"]['value']
            dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return dataset

def format_conversation(tokenizer, prompt, response):
    """将对话格式化为模型输入"""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False)

def compute_reward(model, tokenizer, prompt, response, device):
    """计算单个响应的奖励分数"""
    formatted_text = format_conversation(tokenizer, prompt, response)
    inputs = tokenizer(formatted_text, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).logits[0][0].item()

def evaluate_reward_model(model, tokenizer, dataset, threshold=1.0, device="cuda:0"):
    """评估奖励模型的准确率"""
    correct = 0
    total = len(dataset)
    
    for item in dataset:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        score_chosen = compute_reward(model, tokenizer, prompt, chosen, device)
        score_rejected = compute_reward(model, tokenizer, prompt, rejected, device)
        score_diff = score_chosen - score_rejected
        
        if score_diff > threshold:
            correct += 1
        
        # 打印单条结果（前80字符）
        print(f"Prompt: {prompt[:80]}...")
        print(f"Chosen Score: {score_chosen:.4f}, Rejected Score: {score_rejected:.4f}, "
              f"Difference: {score_diff:.4f}, Correct: {score_diff > threshold}")
        print("-" * 80)
    
    accuracy = correct / total
    print("=" * 80)
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2%}")
    return accuracy

if __name__ == "__main__":

    #infinity preference dataset
    #DATASET_PATH = "/home/kas/kas_workspace/share/chenkai/playground/dataset/rm_infinity/infinity_reward.test.jsonl"
    #MODEL_PATH = "/home/kas/kas_workspace/output/Qwen3_reward_model_4B_0625"

    # GPT-4 preference dataset
    #MODEL_PATH = "/home/kas/kas_workspace/model/Reward/Skywork-Reward-Llama-3.1-8B" # skywork baseline，85%
    #MODEL_PATH = "/home/kas/kas_workspace/output/Skywork_Reward_Model_8B_dpo_0704" # skywork ft,97%
    MODEL_PATH = "/home/kas/kas_workspace/output/Qwen3_Reward_Model_4B_dpo_0703/"  # Qwen3-ft, 98%
    #MODEL_PATH = "/home/kas/kas_workspace/share/chenkai/playground/output/cls_prompt/qwen3_rewared_3epoch_0704"
  
    DATASET_PATH = "/home/kas/kas_workspace/share/chenkai/playground/dataset/rm_infinity/score_dpo_test.jsonl"
    
    DEVICE = "cuda:0"
    
    # 加载模型和数据集
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DEVICE)
    dataset = load_dataset(DATASET_PATH)
    
    # 评估模型
    print(f"Evaluating model: {MODEL_PATH}\n" + "=" * 80)
    print(f"Model architecture:\n{model}\n" + "=" * 80)
    evaluate_reward_model(model, tokenizer, dataset, threshold=1.0, device=DEVICE)
