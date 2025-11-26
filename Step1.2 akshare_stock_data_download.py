#!/usr/bin/env python
# coding: utf-8

# In[4]:


import akshare as ak
import pandas as pd


# In[7]:


# 获取所有A股股票代码
stock_list = ak.stock_zh_a_spot_em()
# 过滤掉ST和ST*股票，并创建一个副本（排除高风险股票）
non_st_stocks = stock_list[~stock_list['名称'].str.contains('ST')].copy()
non_st_stocks.rename(columns={'代码':'symbol', '名称':'name'}, inplace=True)
symbol_list=non_st_stocks['symbol'].tolist()

# 初始化df
all_data = pd.DataFrame()
for symbol in symbol_list:
    try:
        # 获取指定股票代码的历史行情数据
        stock_data = ak.stock_zh_a_hist(
            symbol=symbol, 
            period="daily", 
            start_date="20140101", 
            end_date="20231231"
        )
        # 将股票代码添加到DataFrame
        stock_data['Symbol'] = symbol
        # 将数据追加到all_data DataFrame
        all_data = pd.concat([all_data, stock_data], ignore_index=True)
    except KeyError:
        # 如果遇到KeyError，打印股票代码并跳过
        print(f"股票代码 {symbol} 无法获取数据，已跳过。")
        continue


all_data.to_csv('autodl-tmp/Dataset/个股量价数据/2014-2023.csv', index=False, encoding='utf-8-sig')


# In[8]:


all_data


# In[ ]:




