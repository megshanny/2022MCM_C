import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
from scipy.optimize import leastsq

data_dict = {}
with open('./LSTM-all-1.txt', 'r') as file:
    current_day = None
    current_data = None
    for line in file:
        if 'Day:' in line:
            current_day = int(line.split()[1])
            data_dict[current_day] = {}
        elif 'data' in line:
            current_data = line.split(':')[0].strip()
            data_dict[current_day][current_data] = {}
            if 'price' in line:
                data_values = [float(value) for value in line.split('[')[1].split(']')[0].split()]
            elif 'date' in line:
                data_values = [int(value) for value in line.split('[')[1].split(']')[0].split()]
            data_dict[current_day][current_data] = data_values

data1 = pd.read_csv('.\LBMA-GOLD.csv')
data1 = data1.dropna(how='any')
data1 ['Date'] = pd.to_datetime(data1['Date'],format='%m/%d/%y')
data1['DayFromBegin'] = data1['Date'].map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data2 = pd.read_csv('.\BCHAIN-MKPRU.csv')
data2 = data2.dropna(how='any')
data2 ['Date'] = pd.to_datetime(data2['Date'],format='%m/%d/%y')
data2['DayFromBegin'] = data2['Date'].map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
x1 = []
x2 = []
y1 = []
y2 = []

for key, value in data_dict.items():
    try:
        if(value['data1_predict_date'] not in x1):
            x1.append(value['data1_predict_date'][0])
            y1.append(value['data1_predict_price'][0])
        if(value['data2_predict_date'] not in x2):
            x2.append(value['data2_predict_date'][0])
            y2.append(value['data2_predict_price'][0])
    except:
        print(value)

import pandas as pd

# 假设你的数据框名为 df，其中有一个名为 'days_from_base_date' 的列，表示与基准日期的天数差
base_date = '2016-9-11'
x1 = pd.to_timedelta(x1, unit='D')
x1 = pd.to_datetime(base_date) + x1 
x2 = pd.to_timedelta(x2, unit='D')
x2 = pd.to_datetime(base_date) + x2
print(data2['Date'])
print(data2['DayFromBegin'])


plt.figure(figsize=(12, 10))
plt.subplot(211)
plt.plot(x1, y1, label='predict')
plt.plot(data1['Date'], data1['USD (PM)'], label = 'real')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Gold Price')
plt.subplot(212)
plt.plot(x2, y2, label='predict')
plt.plot(data2['Date'], data2['Value'], label = 'real')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Bitcoin Price')
plt.subplots_adjust(hspace=0.3)
plt.show()
