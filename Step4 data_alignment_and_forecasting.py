#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入需要用到的库
get_ipython().system('pip install seaborn')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install statsmodels')
get_ipython().system('pip install lightgbm')
import pandas as pd
import scutquant2.scutquant as q
from scutquant2 import alpha, report, executor


# # 🔥 hsy 数据结构

# ## 读取股价历史量价数据
# 
# 

# In[2]:


import pandas as pd

read_path = "autodl-tmp/Dataset/个股量价数据/2014-2023.csv"

# 读取CSV文件，并指定Symbol列的数据类型为字符串
df = pd.read_csv(read_path, dtype={'Symbol': str})
df['Symbol'] = df['Symbol'].astype(str)  # 确保Symbol列是字符串类型
print(df['日期'].min())  # 打印df1中日期的最小值


# 打印前20行数据
print(df.head(3))


# In[3]:


df = df.rename(columns={'Symbol':'instrument','日期': 'datetime', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount', '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'change', '换手率': 'turnover'})

df.head(10)
df = df[df["datetime"] < "2023-12-31"]
df


# In[4]:


df.set_index(["datetime", "instrument"], inplace=True)
# df["turnover_20std"] = df['turnover'].groupby(level='instrument').rolling(window=20).std().droplevel(0)
# df['turnover_20mean'] = df['turnover'].groupby(level='instrument').rolling(window=20).mean().droplevel(0)
# df['turnover_std/mean'] =df["turnover_20std"]/df["turnover_20mean"]


# In[5]:


df.dropna(inplace=True)
df.head(3)


# In[6]:


# 构造目标值, 并按照-10% 和10%截断(因为有涨跌停限制)
df["label"] = df["pct_chg"].groupby("instrument").shift(-2) / 100
df = df.dropna()
df["label"].clip(-0.1, 0.1, inplace=True) 
df["label"].describe()


# In[7]:


kwargs = {
    "data": df,
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
    "amplitude": "amplitude",
    "turnover": "turnover",
    "groupby": "instrument"
    
}

X = alpha.qlib158(df)
X = pd.concat([X, df[["label"]]], axis=1)
X = X.dropna()
X.reset_index(inplace=True)
X


# ## 读取新闻标题情感指数

# ### 整合文件

# In[8]:


import pandas as pd
import os


base_dir = "autodl-tmp/Dataset/aggregated_sentiment"  # 基础目录
file_pattern = "{}_aggregated.csv"  # 文件名模式（年份占位符）
years = range(2014, 2024)  # 2014到2023
file_paths = [os.path.join(base_dir, file_pattern.format(year)) for year in years]

# 3. 读取所有CSV文件并合并
dfs = []  # 存储所有DataFrame的列表
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path, dtype={'Symbol': str}) 
        df = df.rename(columns={'Symbol': 'instrument'})
        df['instrument'] = df['instrument'].astype(str)
        dfs.append(df)
    except FileNotFoundError:
        print(f"警告：文件不存在，已跳过: {file_path}")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

# 4. 合并所有DataFrame并命名为df_news
if dfs:
    df_news = pd.concat(dfs, ignore_index=True)
    # 确保'instrument'列是字符串类型（双重保险）
    df_news['instrument'] = df_news['instrument'].astype(str)
else:
    raise ValueError("没有找到任何可合并的文件！")

# 5. 列名调整
df_news.drop(columns=["total_word_count"], inplace=True)
df_news = df_news[["instrument","DeclareDate", "sentiment"]].rename(columns={
    "DeclareDate": "datetime",
    "sentiment": "sentiment_index"
})
df_news.head(10)

# 6. 打印合并后的结果
print("合并后的df_news形状:", df_news.shape)
print("\n前5行数据:")
df_news


# ### 新闻数据和股票量价数据对齐
# 

# In[9]:


import pandas as pd

# 假设df_news是一个pandas DataFrame，包含'instrument', 'datetime'和'sentiment_index'列
# 首先，确保'datetime'列的类型为datetime
df_news['datetime'] = pd.to_datetime(df_news['datetime'])

# 创建一个新的DataFrame，包含所有'instrument'和从'2014-01-01'到'2023-12-31'的日期
unique_instruments = df_news['instrument'].unique()
all_dates = pd.date_range(start='2014-01-01', end='2023-12-31')

# 使用笛卡尔积创建所有可能的'instrument'和'date'组合
full_df = pd.MultiIndex.from_product([unique_instruments, all_dates], names=['instrument', 'datetime']).to_frame(index=False)

# 确保full_df的'datetime'列是datetime类型
full_df['datetime'] = pd.to_datetime(full_df['datetime'])

# 合并df_news到full_df，保留所有full_df的行，并将缺失的'sentiment_index'值填充为0
df_news = pd.merge(full_df, df_news, on=['instrument', 'datetime'], how='left').fillna({'sentiment_index': 0})

# 打印更新后的DataFrame的前20行
df_news.head(3)



# In[10]:


X = X.reset_index() 
# 确保Symbol列是字符串类型
df_news['instrument'] = df_news['instrument'].astype(str)
X['instrument'] = X['instrument'].astype(str)

# 确保datetime列是datetime类型
df_news['datetime'] = pd.to_datetime(df_news['datetime'])
X['datetime'] = pd.to_datetime(X['datetime'])

#筛选df和df_news两个dataframe中instrument相同且datetime相同的行，然后把X中的数据合并到df_news中
merged_df = pd.merge(df_news, X, on=['instrument', 'datetime'], how='inner')
merged_df


# In[11]:


print(max(merged_df["sentiment_index"]))


# In[12]:


del X
del df_news


# In[13]:


merged_df['instrument'] = merged_df['instrument'].astype(str)
merged_df.set_index(["datetime", "instrument"], inplace=True)
print(merged_df.head(10))


# # 🔥hjl 机器学习预测因子

# ## 模型预测

# In[14]:


merged_df.dropna(inplace=True)


# In[15]:


import pandas as pd

split_params = {
    "test_start_date": "2022-01-01",  # 测试集的开始日期，此日期之后的数据归入测试集
    "split_method": "split_by_date", 
    "split_kwargs": {
        "train_end_date": "2021-01-01",  # 训练集的结束日期，此日期之前的数据用于训练
        "valid_end_date": "2022-01-01"   # 验证集的结束日期，介于train_end_date和此日期之间的数据用于验证
    }
}


# 按split_params拆分出训练集、验证集和测试集, 并在截面上进行标准化
result = q.auto_process(merged_df, "label", groupby="instrument", split_params=split_params, select=False, orth=False, clip=5)

X_train, y_train, X_valid, y_valid = result["X_train"], result["y_train"], result["X_valid"], result["y_valid"]
X_test, y_test, ymean, ystd = result["X_test"], result["y_test"], result["ymean"], result["ystd"]


# In[16]:


# 继续后续的处理
X_train, y_train, X_valid, y_valid
X_test, y_test, ymean, ystd


# In[17]:


# 选择一个因子, 并对其可视化 (此步骤非必要)
report.single_factor_ana(X_train["sentiment_index"])


# In[18]:


# 拟合模型
model = q.auto_lgbm(X_train, y_train, X_valid, y_valid, early_stopping=50)

# 使用模型进行预测
pred = model.predict(X_test)
pred = pd.DataFrame(pred, columns=["predict"], index=X_test.index)

# 为防止数据泄露，使用滞后2阶的mean和std还原预测值
pred.rename_axis(['datetime', 'instrument'], inplace=True)

pred["predict"] += ymean.groupby("datetime").shift(2).fillna(0.0002)
pred["predict"] *= ystd.groupby("datetime").shift(2).fillna(0.0189)

# 输出预测结果的描述性统计信息
pred.describe()


# In[19]:


pred.head(10)


# # 🔥wxx 投资应用

# In[20]:


# 使用IC评判拟合优度
ic, icir, rank_ic, rank_icir = q.ic_ana(pred, y_test, groupby='datetime')
print('ic=', ic, 'icir=', icir, 'rank_ic=', rank_ic, 'rank_icir=', rank_icir)
# 使用pearson相关系数评判拟合优度
r = q.pearson_corr(pred["predict"], y_test)
r


# In[21]:


merged_df.head(10)


# In[22]:


pred.head()


# In[23]:


df_test = merged_df[merged_df.index.isin(pred.index)]
df_test.head(10)


# In[24]:


# 以下为回测的数据格式转换部分
# 将预测值处理成回测需要的格式(需要知道预测值"predict", 交易价格price和交易量volume)这边的data包含datetime和close
# df_test = merged_df[merged_df.index.isin(pred.index)]
"""
    :param predict: pd.DataFrame, 预测值, 应包括"predict"
    :param data: pd.DataFrame, 提供时间和价格信息
    :param price: str, data中表示价格的列名
    :param volume: str, data中表示成交量的列名
    :param real_ret: pd.Series, 真实收益率
    :return: pd.DataFrame
    """
data_ = df_test.copy()
data_.head()


# In[25]:


pred.columns = ["predict"]
index = pred.index
data1 = data_[data_.index.isin(index)]
data1 = data1.reset_index()
data1 = data1.set_index(pred.index.names).sort_index()
data1.head()


# In[26]:


pred.head()


# In[27]:


# 将 'y_test' 转换为 DataFrame 并设置列名称为 'R'
y_test = y_test.to_frame(name='R')
y_test.head()


# In[28]:


data1.head()


# In[29]:


# 确保 'pred' 的索引是唯一的
# 使用 merge 方法合并 'pred' 和 'data1'，基于它们共有的索引
pred.index.names = ["datetime", "instrument"]
pred = pred.merge(data1[['close', 'volume']], left_index=True, right_index=True, how='left')


pred.rename(columns={'close': 'price'}, inplace=True)
pred.index.names = ["time", "code"]
y_test.index.names = ["time", "code"]
pred["price"] = pred['price'].groupby(["code"]).shift(-1) # 指令是T时生成的, 但是T+1执行, 所以是shift(-1)
pred = pred.merge(y_test[['R']], left_index=True, right_index=True, how='left')  # 本来就是T+2对T+1的收益率, 因此不用前移
pred.head(10)


# In[ ]:


pred = pred[~pred.index.duplicated(keep='first')]
pred.sort_index()


# In[ ]:


y_test = y_test[~y_test.index.duplicated(keep='first')]
y_test.sort_index()
y_test.dropna(inplace=True)
y_test_series = y_test.squeeze()
y_test_series


# In[ ]:


pred.dropna(inplace=True)
pred


# In[ ]:


# 以下为回测的策略执行部分
# 将预测值处理成回测需要的格式(需要知道预测值"predict", 交易价格price和交易量volume)这边的data包含datetime和close
# df_test = merged_df[merged_df.index.isin(pred.index)]
# pred = executor.prepare(pred, df_test, price='close', volume='volume', real_ret=y_test)

pred.dropna(inplace=True)
# 如果出现明显的分层, 则说明因子有良好的选股能力
report.group_return_ana(pred, y_test_series)

backtest = {
    "generator": {
        "mode": "generate"
    },
    "strategy": {
        "class": "TopKStrategy",  # 做多预测值前20%股票, 做空后20%的股票. 自动平仓
        "kwargs": {
            "k": 0.2,
            "auto_offset": False,
            "offset_freq": 2, # 应为delta_t + 1, 例如目标值是close_-2 / close_-1 - 1, 则delta_t = 1
            "buy_only": False,  # =True时，只做多不做空(在A股做空有一定的难度)
            "short_volume": 500, # 融券做空的数量
            "risk_degree": 0.95,  # 将风险度控制在这个数，如果超过了就按比例减持股票直到风险度小于等于它为止
            "unit": None,  # 由于数据已经是以手为单位, 故无需二次处理
            "max_volume": 0.05  # 手数随可用资金而改变，最大不会超过股票当天成交量的1%(例如T+1时下单，下单手数不会超过T时成交量的1%)
        }
    },
    "account": None, # 使用默认账户, 即初始资金为1亿, 无底仓 (注意策略容量！)
    "trade_params": {
        "cost_buy": 0.0015,  # 佣金加上印花税
        "cost_sell": 0.0015,
        "min_cost": 5,
    }
}
generator, strategy, account, trade_params = backtest["generator"], backtest["strategy"], backtest["account"], backtest["trade_params"]
exe = executor.Executor(generator, strategy, account, trade_params)
exe.execute(data=pred, verbose=0)  # verbose=1时，按时间输出买卖指令

# 报告回测结果
user_account, benchmark = exe.user_account, exe.benchmark
report.report_all(user_account, benchmark, freq=1, time=exe.time,rf=0.01 )  # 无风险收益率默认为年化3%，可以通过参数rf调节


# In[ ]:




