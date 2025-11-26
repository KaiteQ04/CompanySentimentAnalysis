#!/usr/bin/env python
# coding: utf-8

# # 1 环境设置

# In[1]:


import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import numpy as np
import json
import random


# # 2 模型准备

# ## 2.1 模型下载

# In[2]:


import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct/', device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
model.dtype


# ## 2.2 微调数据集准备

# In[5]:


df = pd.read_json('autodl-tmp/Dataset/NewsContent.json')
#ds = Dataset.from_pandas(df)
df


# In[10]:


instruction = df.iloc[:, 0].astype(str).tolist()
input = df.iloc[:, 1].astype(str).tolist()
output = df.iloc[:, 2].astype(str).tolist()

training_dataset = Dataset.from_dict({
    "instruction": instruction,
    "input": input,
    "output": output
})


# ## 2.3 tokenize

# In[11]:


tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct/', use_fast=False, trust_remote_code=True)


# In[12]:


def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是金融领域的专家,请为以下文本进行情感打分。打分标准：看涨（积极）打1分，看跌（消极）打-1分，中立（中性）打0分。我会输入一段新闻文本，请输出你的情感打分。输出格式：根据情感分析在1、-1和0中输出一个数字，不要有其他的文字和符号。。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# In[13]:


training_tokenized = training_dataset.map(process_func)
training_tokenized = training_tokenized.remove_columns(['instruction', 'input', 'output'])
training_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
training_tokenized


# ## 2.4 微调函数准备

# In[14]:


from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model, config)# 保存 lora/adapter_config.json


# In[15]:


args = TrainingArguments(
    output_dir="./output/Qwen2.5_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=10,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)


# # 3 微调

# In[16]:


# 创建 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=training_tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# 开始训练
trainer.train()

# Save the fine-tuned model
model.save_pretrained("lora")


# # 4 合并模型

# In[17]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = '/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct/'
lora_path = 'lora' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "生猪表现良好，我要买入"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "你是金融领域的专家,请为以下文本进行情感打分。打分标准：看涨（积极）打1分，看跌（消极）打-1分，中立（中性）打0分。我会输入一段新闻文本，请输出你的情感打分。输出格式：根据情感分析在1、-1和0中输出一个数字，不要有其他的文字和符号。"},{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')


gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# # 5 模型调用

# ## 5.1 数据

# In[18]:


import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.nn.functional import softmax
import numpy as np
from torch.nn.functional import softmax
# 检查是否有可用的 GPU，如果没有，将使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[19]:


# 将JSON文件转换为CSV文件
df = pd.read_json('autodl-tmp/Dataset/NewsContent.json')
#ds = Dataset.from_pandas(df)
df.head(10)


# ## 5.2 推理

# In[30]:


# 模型路径设定
model_path = '/root/autodl-tmp/qwen/Qwen2.5-0.5B-Instruct/'
lora_path = 'lora'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    pad_token='<|endoftext|>'
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 预处理允许的数字token
digit_tokens = []
for c in ['0', '1', '2']:
    if tokenizer.added_tokens_encoder.get(c):
        digit_tokens.append(tokenizer.added_tokens_encoder[c])
    else:
        digit_tokens.append(tokenizer.convert_tokens_to_ids(c))

def get_sentiment_score(prompt):
    try:
        prompt = str(prompt).strip()
        messages = [
            {
                "role": "system",
                "content": "你是一个金融情感分析工具，必须严格遵守以下规则：\n"
                           "1. 只能输出以下三个数字：\n"
                           "   - 1 (积极)\n"
                           "   - 2 (消极)\n"
                           "   - 0 (中性)\n"
                           "2. 禁止任何解释性文字\n"
                           "3. 示例输出：1"
            },
            {
                "role": "user",
                "content": f"分析以下文本，输出情感分数（0/1/2）：\n{prompt}"
            }
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": 3,
            "do_sample": False,
            "num_beams": 1,
            "temperature": 0.01,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            raw_output = tokenizer.decode(
                output_tokens,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            ).strip()
            
            # 清洗输出逻辑（适配新评分标准）
            cleaned = raw_output.replace(" ", "").replace("\n", "")
            
            # 优先检测最高优先级数字
            if '2' in cleaned:  # 消极
                return 2
            elif '1' in cleaned:  # 积极
                return 1
            else:  # 中性或无效输出
                return 0
                
    except Exception as e:
        print(f"处理异常: {prompt[:50]}... | 错误: {str(e)}")
        return 0  # 异常时默认中性

# 应用情感分析
df['sentiment'] = df['input'].apply(get_sentiment_score)

# 结果验证
print("\n情感分布统计:")
print(df['sentiment'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

print("\n样本验证（前10条）:")
print(df[['input', 'output', 'sentiment']].head(10))


# ## 5.3 计算准确率

# In[32]:


df['equal'] = df['output'] == df['sentiment']
equal_count = df['equal'].sum()
total_count = len(df)
accuracy = equal_count / total_count

print(f"准确率: {accuracy:.2%}")

