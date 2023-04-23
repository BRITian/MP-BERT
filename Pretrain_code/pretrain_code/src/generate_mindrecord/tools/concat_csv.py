import pandas as pd
import numpy as np
#读取文件
df1 = pd.read_csv("/data1/bert/PMT/pmt_seqfaslenc04clstr/train_0.csv")
df1["origin_data"] = "train"
print(sum(list(df1["label"]))/len(df1))
df2 = pd.read_csv("/data1/bert/PMT/pmt_seqfaslenc04clstr/val_0.csv")
df2["origin_data"] = "val"
print(sum(list(df2["label"]))/len(df2))
#合并
df = pd.concat([df1,df2])
df.drop_duplicates()  #数据去重
#保存合并后的文件
df.to_csv("/data1/bert/PMT/pmt_seqfaslenc04clstr/train_val_0.csv",encoding = 'utf-8')
