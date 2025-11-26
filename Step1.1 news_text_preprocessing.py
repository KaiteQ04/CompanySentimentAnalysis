#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import chardet


# # 数据读取合并

# ## 三张表数据读取、筛选指定列

# In[2]:


news_file_path = "autodl-tmp/Dataset/新闻证券关联表2012-2015.csv"
news_file_path2 = "autodl-tmp/Dataset/证券行业关联表.csv"
news_file_path3 = "autodl-tmp/Dataset/新闻文本源文件/2014.csv"


news_security_df = pd.read_csv(news_file_path, dtype={'Symbol': str})
industry_security_df = pd.read_csv(news_file_path2, dtype={'Symbol': str})
title_fullnews_df = pd.read_csv(news_file_path3, dtype={'Symbol': str} ,
            on_bad_lines='skip'
                               )

# 查看格式
print("\n新闻证券关联表")
print(news_security_df.tail(3))

print("\n证券行业关联表")
print(industry_security_df.tail(3))

print("\n2014")
print(title_fullnews_df.head(3))


# In[3]:


print(title_fullnews_df.tail(3))


# In[4]:


news_security_df = news_security_df[news_security_df['SecurityType']=='A股'][[ 'DeclareDate', 'Title', 'Symbol']]
news_security_df.head(10)


# In[5]:


industry_security_df = industry_security_df[[ 'IndustryName2', 'IndustryCode2','Symbol']]
industry_security_df.head(10)


# In[6]:


title_fullnews_df = title_fullnews_df[[ 'Title', 'NewsContent']]
title_fullnews_df.head(10)


# ## 三张表数据合并

# In[7]:


merged_df1 = pd.merge(news_security_df, title_fullnews_df, on='Title', how='inner')
merged_df1.head(10)


# In[8]:


merged_df2 = pd.merge(merged_df1, industry_security_df, on='Symbol', how='inner')
merged_df2.drop_duplicates(subset=['Title', 'Symbol'], keep='first', inplace=True)
merged_df2.head(10)


# In[9]:


print(len(merged_df2['Symbol'])) #三表同时匹配的记录数量
print(len(merged_df1['Symbol'])) #新闻证券表和新闻正文表匹配的记录数量


# ## 满足Bert要求的512个token最大长度
# 中文文段一般需要控制在512字长度以内（当然后续还会做截断统一长度，但是想尽可能多地利用文本信息）

# In[10]:


# 确保 'NewsContent' 列没有 NaN 值
merged_df2 = merged_df2.dropna(subset=['NewsContent'])

# 用空格分割 'NewsContent' 列
split_paragraphs = merged_df2['NewsContent'].str.split('\s+')

# 计算每个分割后的列表的长度
lengths = split_paragraphs.str.len()

# 检查长度是否匹配
if len(merged_df2['DeclareDate']) == len(lengths):
    # 创建新的 DataFrame
    merged_df3 = pd.DataFrame({
        'DeclareDate': merged_df2['DeclareDate'].repeat(lengths),
        'Title': merged_df2['Title'].repeat(lengths),
        'Symbol': merged_df2['Symbol'].repeat(lengths),
        'DeclareDate': merged_df2['DeclareDate'].repeat(lengths),
        'IndustryName2': merged_df2['IndustryName2'].repeat(lengths),
        'IndustryCode2': merged_df2['IndustryCode2'].repeat(lengths),
        'NewsContent': [paragraph for paragraphs in split_paragraphs for paragraph in paragraphs]
    })

    # 重置索引
    merged_df3.reset_index(drop=True, inplace=True)
else:
    print("长度不匹配，请检查数据。")



# In[11]:


merged_df3.head(10)


# In[12]:


#验证分割以后的段落长度是不是大多数在512字以内
# 计算每个 'NewsContent' 的长度
lengths = merged_df3['NewsContent'].str.len()

# 计算长度小于 512 的文本的数量
less_than_512 = lengths[lengths < 512].count()

# 计算总文本的数量
total_texts = lengths.count()

# 计算比例
proportion_less_than_512 = less_than_512 / total_texts

# 打印结果
print(f"长度小于 512 的文本的比例是：{proportion_less_than_512:.2%}")



# In[13]:


merged_df3['Symbol'] = merged_df3['Symbol'].astype(str)
merged_df3.to_csv('autodl-tmp/Dataset/新闻文本合并文件/2014.csv', index=False)
merged_df3.head(10)
print(f"新闻文本合并文件总行数（含表头）: {len(merged_df3)}")


# In[ ]:




