#!/usr/bin/env python
# coding: utf-8

# # 1 环境设置

# In[1]:


import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from modelscope import snapshot_download


# # 2推理

# In[2]:


# 数据导入
df = pd.read_json('autodl-tmp/Dataset/新闻文本json文件/2014.json')
print(df.head(10))
print(f"文件总行数（含表头）: {len(df)}")


# In[ ]:


#!/usr/bin/env python
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

# 配置参数
MODEL_PATH = '/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct/'
LORA_PATH = 'lora'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Prompt模板 ==================
PROMPT_TEMPLATE = """请分析以下新闻的情感倾向，并严格按照要求输出单一数字：
- 中性输出0
- 积极输出1
- 消极输出2

新闻内容：{text}
情感分析结果："""  # 注意末尾无空格

# ================== 生成配置 ==================
def get_generation_config(tokenizer):
    return GenerationConfig(
        max_new_tokens=3,              # 最大生成长度
        num_beams=1,                   # 减少计算量
        do_sample=False,               # 确定性输出
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # 使用eos作为pad
        forced_decoder_ids=[
            [2, tokenizer.convert_tokens_to_ids("0")],  # 强制首token为数字
            [2, tokenizer.convert_tokens_to_ids("1")],
            [2, tokenizer.convert_tokens_to_ids("2")]
        ]
    )

# ================== 情感提取函数 ==================
def extract_sentiment(text): 
    # 严格匹配最后一个数字
    match = re.search(r'([0-2])(?!.*[0-2])', text.strip())
    return int(match.group(1)) if match else -1

# ================== 数据加载器 ==================
class TextDataset(Dataset):
    def __init__(self, texts):
        # 清洗数据：仅保留有效字符串
        self.texts = [
            str(t) for t in texts 
            if pd.notna(t) and isinstance(t, (str, bytes))
        ]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# ================== 批量推理 ==================
def batch_inference(model, tokenizer, batch_texts):
    # 应用Prompt模板
    formatted_texts = [PROMPT_TEMPLATE.format(text=t) for t in batch_texts]
    
    # Tokenize
    inputs = tokenizer(
        formatted_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        pad_to_multiple_of=8
    ).to(DEVICE)
    
    # 生成结果
    outputs = model.generate(
        **inputs,
        generation_config=get_generation_config(tokenizer)
    )
    
    # 解码并提取情感
    decoded = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],  # 只取新生成部分
        skip_special_tokens=True
    )
    return [extract_sentiment(d) for d in decoded]

# ================== 主流程 ==================
if __name__ == "__main__":
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    padding_side='left'  # 强制左侧填充
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
    model.eval()
    
    # 加载数据
    df = pd.read_json('autodl-tmp/Dataset/新闻文本json文件/2014.json')
    #df = df.head(10000)
    print(f"原始数据量: {len(df)}")
    
    # 创建数据集
    dataset = TextDataset(df['input'].tolist())
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False)
    
    # 执行推理
    results = []
    for batch in tqdm(dataloader, desc="情感分析"):
        results.extend(batch_inference(model, tokenizer, batch))
    
    # 保存结果
    df = df.iloc[:len(results)]  # 对齐数据
    df['sentiment'] = results
    df = df[df['sentiment'] != -1]  # 过滤无效结果
    
    print("\n情感分布统计:")
    print(df['sentiment'].value_counts().sort_index())
    df.to_csv("sentiment_results_2014.csv", index=False)
df


# In[ ]:




