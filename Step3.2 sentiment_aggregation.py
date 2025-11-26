#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # 1 环境设置
import json
import os
import re
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from modelscope import snapshot_download

# # 2 批量处理函数
def process_year(year):
    print(f"\n正在处理 {year} 年数据...")
    
    # 1. 读取情感分析结果并替换2为-1
    sentiment_file = f'sentiment_results_{year}.csv'
    df_sentiment = pd.read_csv(sentiment_file)
    df_sentiment['sentiment'] = df_sentiment['sentiment'].replace(2, -1)
    
    # 2. 读取原始新闻数据并合并情感分析结果
    news_file = f'autodl-tmp/Dataset/新闻文本合并文件/{year}.csv'
    df_news = pd.read_csv(news_file)
    df_news["sentiment"] = df_sentiment["sentiment"]
    
    # 3. 计算净字数
    def clean_text_length(text):
        if pd.isna(text):
            return 0
        cleaned = re.sub(r'[^\w\u4e00-\u9fa5]', '', str(text))
        return len(cleaned)
    
    df_news["NewsContent_WordCount"] = df_news["NewsContent"].apply(clean_text_length)
    
    # 4. 计算相同title下的字数占比
    def calculate_ratio(group):
        total = group["NewsContent_WordCount"].sum()
        group["WordCount_Ratio"] = (group["NewsContent_WordCount"] / total) if total > 0 else 0
        return group
    df_news = df_news.groupby("Title", group_keys=False).apply(calculate_ratio)
    
    # 5. 计算加权sentiment
    df_news["Weighted_Sentiment"] = df_news["sentiment"] * df_news["WordCount_Ratio"]
    
    # 6. 按照title聚合sentiment
    final_result = (
        df_news
        .groupby(['Title', 'Symbol', 'DeclareDate'], as_index=False)
        .agg({
            'NewsContent_WordCount': 'sum',
            'Weighted_Sentiment': 'sum'
        })
        .rename(columns={
            'Weighted_Sentiment': 'sentiment',
            'NewsContent_WordCount': 'total_word_count'
        })
        .round(4)
    )
    
    # 7. 按照Symbol和DeclareDate分组聚合sentiment
    final_aggregated = (
        final_result
        .groupby(['Symbol', 'DeclareDate'], as_index=False)
        .agg({
            'sentiment': 'sum',
            'total_word_count': 'sum'
        })
        .round(4)
    )
    
    # 8. 保存结果
    # 保存打标后的原始数据
    labeled_path = f'autodl-tmp/Dataset/完成打标的数据/{year}.csv'
    df_news.drop(columns=['NewsContent']).to_csv(labeled_path, index=False)
    
    # 保存聚合结果
    aggregated_path = f'autodl-tmp/Dataset/aggregated_sentiment/{year}_aggregated.csv'
    final_aggregated.to_csv(aggregated_path, index=False)
    
    print(f"{year} 年数据处理完成，结果已保存至 {labeled_path} 和 {aggregated_path}")

# # 3 主循环
def main():
    # 确保输出目录存在
    os.makedirs('autodl-tmp/Dataset/完成打标的数据', exist_ok=True)
    os.makedirs('autodl-tmp/Dataset/aggregated_sentiment', exist_ok=True)
    
    # 处理2014-2023年数据
    for year in range(2014, 2024):
        try:
            process_year(year)
        except Exception as e:
            print(f"处理 {year} 年数据时出错: {str(e)}")
            continue

if __name__ == "__main__":
    main()


# In[2]:


df = pd.read_csv('autodl-tmp/Dataset/aggregated_sentiment/2023_aggregated.csv')
df


# In[ ]:




