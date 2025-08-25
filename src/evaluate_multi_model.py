# -*- coding: utf-8 -*-
import http.client
import time
import json
import argparse
import asyncio
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from typing import AsyncIterator
from openai import AsyncOpenAI, OpenAI

import pandas as pd

def build_prompt(prompt, model_a_output, model_b_output):
    eval_prompt = f"""
## 角色：模型输出评估专家  
你负责从多维度对比两个模型（Model_A 和 Model_B）对同一用户问题的回答质量。
请严格按以下步骤分析，并仅输出 JSON 格式结果，不包含任何额外文本或解释。

## 评估步骤
1. **提取原始问题**  
   - 用户输入

2. **维度打分（每项满分10分）**  
   - **准确度**：信息是否基于可信事实和数据，且无错误（例如，科学原理是否正确）。
   - **满足度**：回答是否全面覆盖用户问题需求，避免遗漏关键点（例如，是否解释原理的核心）。
   - **连贯度**：回答是否逻辑一致，各部分无矛盾（例如，语句间是否流畅衔接）。 

3. **输出结构化结果**  
   - **Result 字段**：比较平均分：  
     - 平均分相同， 输出 S
     - Model_A 平均分更高，输出 A  
     - Model_B 平均分更高，输出 B
   - **输出格式**：严格使用以下 JSON 结构： 
   ```json
    {{
    "Model_A": {{"正确性": <score>, "满足度": <score>, "连贯度": <score>, "平均分": <score>}},
    "Model_B": {{"正确性": <score>, "满足度": <score>, "连贯度": <score>, "平均分": <score>}},
    "Result": "A/B/S"
    }}
   ```

## 待评估内容
---
**用户输入**：
{prompt}

**Model_A 输出**： 
{model_a_output}

**Model_B 输出**： 
{model_b_output}
---
"""
    return eval_prompt

def _json_postprocess(res_json_str: str):
        """处理json结果"""
        if "：" in res_json_str:
            res_json_str = res_json_str.replace("：", ":")
        if "</think>" in res_json_str:
            res_json_str = res_json_str.split("</think>")[1].strip()
        if "```json" in res_json_str:
            res_json_str = (
                res_json_str.replace("\n", "").split("```json")[1].split("```")[0]
            )
        extracted_json = json.loads(res_json_str)

        return extracted_json

async def gpt_aysnc_infer(user_prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 4378acwf356zbgjys55yenz67az2yqq4',
        'Host': 'api.ai.ksyun.com'
    }
    conn = http.client.HTTPSConnection("api.ai.ksyun.com")
    #conn = http.client.HTTPConnection("api-internal.ai.ksyun.com")  # 使用HTTP连接
    payload = json.dumps({
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],
        "stream": False,
        "temperature": 0.7
    })
    max_retries = 5  # 最大重试次数
    for attempt in range(max_retries):
        try:
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read().decode('utf-8')
            conn.close()

            # 将响应内容解析为JSON
            chat_response = json.loads(data)
            return chat_response['choices'][0]['message']['content']
        except Exception as e:
            print(f"请求失败(尝试 {attempt+1}/{max_retries}): {str(data)}")
            time.sleep(3)  # 等待2秒后重试
    return json.dumps({"error": 1}, ensure_ascii=False)

async def infer(user_prompt: str) -> str:
    """异步模型推理"""
    return await gpt_aysnc_infer(user_prompt)

async def process_batch(batch: List[dict], out_file: str):
    """处理一批数据"""
    tasks = []
    for json_obj in batch:
        prompt =  json_obj['prompt'][0]['content']
        Model_A = json_obj['Model_A']
        Model_B = json_obj['Model_B']
        user_prompt = build_prompt(prompt, Model_A, Model_B)
        tasks.append(infer(user_prompt))

    results = await asyncio.gather(*tasks)
    res_list = []
    
    with open(out_file, 'a', encoding='utf-8') as fw:
        for json_data, resp in zip(batch, results):
            try:
                response = _json_postprocess(resp)
                new_data = {
                    "result": response,
                    "prompt": json_obj['prompt'][0]['content'],
                    "Model_A": json_data['Model_A'],
                    "Model_B": json_data['Model_B']
                }
                fw.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"JSON错误: {e}，resp: {resp}")
                continue

async def async_batch_iterator(items: List[Any], batch_size: int) -> AsyncIterator[List[Any]]:
    """异步批次迭代器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def model_predict_async(intput_file: str, output_file: str, batch_size: int):
    """异步模型推理主函数"""
    lines = load_prompts_jsonl(intput_file)
    
    # 创建进度条
    from tqdm import tqdm
    pbar = tqdm(total=len(lines), desc="蒸馏进度", ncols=80)
    
    async for batch in async_batch_iterator(lines, batch_size):
        batch_results = await process_batch(batch, output_file)
        pbar.update(len(batch))  # 更新进度条
    
    pbar.close()  # 关闭进度条

def load_prompts_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]
        return lines

def model_predict(intput_file: str, output_file: str, batch_size: int):
    """模型推理的同步包装器"""
    asyncio.run(model_predict_async(intput_file, output_file, batch_size))

def evaluate(output_file: str):
    # 初始化统计字典
    model_counts = {'good': 0, 'same': 0, 'bad': 0}
    
    # 读取JSONL文件
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                result = data.get("result").get("Result", "").upper()
                
                # 根据Result字段更新统计
                if result == "A":
                    model_counts['good'] += 1
                elif result == "B":
                    model_counts['bad'] += 1
                elif result == "S":
                    model_counts['same'] += 1
            except json.JSONDecodeError:
                print(f"警告：跳过无效的JSON行: {line}")
                continue
    
    print("模型评估结果统计：")
    print(f"Model_A 优于 Model_B: {model_counts['good']}")
    print(f"Model_B 优于 Model_A: {model_counts['bad']}")
    print(f"Model_A 和 Model_B 相同: {model_counts['same']}")
    # 返回统计结果
    return model_counts

def main():
    """蒸馏主流程"""
    parser = argparse.ArgumentParser(description='请求 GPT')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='蒸馏数据文件路径')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='输出结果文件路径')    
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='批量推理大小')
    parser.add_argument('--local_file', '-l', type=bool, default=False, help='是否使用本地已推理完成的文件')    
    args = parser.parse_args()
    
    # 模型推理
    if args.local_file:
        evaluate(args.output_file)
    else:
        model_predict(args.input_file, args.output_file, args.batch_size)
        evaluate(args.output_file)

if __name__ == '__main__':
    main() 
    
# python evaluate_multi_model.py -i step2_result.jsonl -o multi_result.jsonl -b 2
