#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv  
import json


# In[2]:


df = pd.read_csv(
    'autodl-tmp/Dataset/新闻文本合并文件/2019.csv', 
    encoding='utf-8', 
    encoding_errors='ignore',
    quoting=csv.QUOTE_ALL, 
    on_bad_lines='skip',
)

df


# In[3]:


df['instruction'] = '请为以下文本进行情感打分。'
df['input'] = df['NewsContent']


# In[4]:


data_to_export = df[['instruction', 'input']].to_dict(orient='records')


# In[5]:


with open('autodl-tmp/Dataset/新闻文本json文件/2019.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_to_export, json_file, ensure_ascii=False, indent=4)

