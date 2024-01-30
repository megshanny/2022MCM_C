import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF


data1 = pd.read_csv('./LBMA-GOLD.csv')
rowname1 = 'USD (PM)'
data2 = pd.read_csv('./BCHAIN-MKPRU.csv')
rowname2 = 'Value'
data1 = data1.dropna(how='any')
data1['DayFromBegin'] = pd.to_datetime(data1['Date'],format='%m/%d/%y').map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data1['RowNumber'] = range(0, len(data1))
data2 = data2.dropna(how='any')
data2['DayFromBegin'] = pd.to_datetime(data2['Date'],format='%m/%d/%y').map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data2['RowNumber'] = range(0, len(data2))


choose_warning_list = []

Day = 30
data_length = 50
pre_length = 10
fee1 = 0.01
fee2 = 0.02

name_dict = {0:'money', 1:'gold', 2:'bitcoin'}
choose_dict = {'gold': 0, 'bitcoin': 0, 'money': 0}
last_choose = 0
money_history = [1000,1000]
money_time = [0,Day]



# ADF检验, 平稳性检验，大于0.05不稳定
print('ADF检验结果：',ADF(data1[rowname1])[1])

data1diff = data1[rowname1]
data1diff = data1diff.diff().dropna()
print('一阶差分ADF检验结果：',ADF(data1diff)[1])

# 白噪声检验
print('白噪声检验：',acorr_ljungbox(data1diff,lags=1))


# # BIC矩阵确定pq的值
# # 定阶
# d = 1
# # p = 0
# # q = 0
# pmax = 6
# qmax = 6
# bic_matric = []
# data1_pre = data1[rowname1].to_numpy()
# for p in range(pmax+1):
#     tmp = []
#     for q in range(qmax+1):
#         try:
#             tmp.append(ARIMA(data1_pre,order=(p,d,q)).fit().bic)
#         except:
#             tmp.append(None)
#     bic_matric.append(tmp)
# bic_matric = pd.DataFrame(bic_matric)
# print(bic_matric)

# p,q = bic_matric.stack().idxmin()

# print(p)
# print(q)