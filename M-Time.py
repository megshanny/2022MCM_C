import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.ar_model import AutoReg
from random import random

data1 = pd.read_csv('.\LBMA-GOLD.csv')
rowname1 = 'USD (PM)'
data2 = pd.read_csv('.\BCHAIN-MKPRU.csv')
rowname2 = 'Value'
data1 = data1.dropna(how='any')
data1['DayFromBegin'] = pd.to_datetime(data1['Date'],format='%m/%d/%y').map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data1['RowNumber'] = range(0, len(data1))
data2 = data2.dropna(how='any')
data2['DayFromBegin'] = pd.to_datetime(data2['Date'],format='%m/%d/%y').map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data2['RowNumber'] = range(0, len(data2))

money_history = [1000]
money_time = [0]

# 147581.39881543102 20
# 93585.83220056961 23 
# 75794.70674612132 24


Day = 5
pre_length = 20
last_choice = 'money'
change_dict = {'gold': 0, 'bitcoin': 0, 'money': 0}

fee1 = 0.001
fee2 = 0.002

change_times = 0

while(Day < data1['DayFromBegin'].max()):
    data1_slice = data1[data1['DayFromBegin'] <= Day]
    data2_slice = data2[data2['DayFromBegin'] <= Day]
    if(pre_length <= data1_slice.__len__()):
        data1_slice = data1_slice.iloc[-pre_length:]
    else:
        data1_slice = data1_slice.iloc[0:]
    if(pre_length <= data2_slice.__len__()):
        data2_slice = data2_slice.iloc[-pre_length:]
    else:
        data2_slice = data2_slice.iloc[0:]
    data1_price = data1_slice[rowname1].to_numpy()
    data2_price = data2_slice[rowname2].to_numpy()
    
    data1_test_price = np.diff(data1_price)
    data2_test_price = np.diff(data2_price)


    model = AutoReg(data1_test_price, lags=1)
    model_fit = model.fit()
    data1_predict = model_fit.predict(len(data1_test_price), len(data1_test_price)) 
    data1_rate = data1_predict / data1_price[-1] + 1

    model = AutoReg(data2_test_price, lags=1)
    model_fit = model.fit()
    data2_predict = model_fit.predict(len(data2_test_price), len(data2_test_price))
    data2_rate = data2_predict / data2_price[-1] + 1

    data1_period = data1.iloc[data1_slice['RowNumber'].max() + 1]['DayFromBegin'] - data1_slice['DayFromBegin'].max()
    data2_period = data2.iloc[data2_slice['RowNumber'].max() + 1]['DayFromBegin'] - data2_slice['DayFromBegin'].max()
    if(data1_slice['DayFromBegin'].max()!=Day):
        data1_rate = 0
    if(data2_slice['DayFromBegin'].max()!=Day):
        data2_rate = 0

    true_data1_rate = data1.iloc[data1_slice['RowNumber'].max() + 1][rowname1] / data1_price[-1]
    true_data2_rate = data2.iloc[data2_slice['RowNumber'].max() + 1][rowname2] / data2_price[-1]

    # if (黄金预测**天数)
    # if(pow(data1_rate, data2_period) - fee1 > pow(data2_rate, data1_period) - fee2):
    #     if(data1_rate - fee1 > 1):
    #         Day += data1_period
    #         money_history.append(money_history[-1] * (true_data1_rate - fee1))
    #         money_time.append(Day)
    #         new_choice = 'gold'
    #     else:
    #         Day += 1
    #         money_history.append(money_history[-1])
    #         money_time.append(Day)
    #         new_choice = 'money'
    # else:
    #     if(data2_rate - fee2 > 1):
    #         Day += data2_period
    #         money_history.append(money_history[-1] * (true_data2_rate - fee2))
    #         money_time.append(Day)
    #         new_choice = 'bitcoin'
    #     else:
    #         Day += 1
    #         money_history.append(money_history[-1])
    #         money_time.append(Day)
    #         new_choice = 'money'
    # if(new_choice != last_choice):
    #     # print(last_choice, '->', new_choice, 'on day', Day)
    #     last_choice = new_choice
    #     change_dict[new_choice] += 1

    # 
    max_money = 0
    new_choice = ''
    # change_times = 0
    if last_choice == 'gold':
        money2gold = money_history[-1]*data1_rate
        money2money = money_history[-1]*(1-fee1)
        money2bitcoin = money2money*data2_rate*(1-fee2)

    elif last_choice == 'money':
        money2money = money_history[-1]
        money2gold = money_history[-1]*(1-fee1)*data1_rate
        money2bitcoin = money_history[-1]*(1-fee2)*data2_rate

    else:
        money2bitcoin = money_history[-1]*data2_rate
        money2money = money_history[-1]*(1-fee2)
        money2gold = money2money*data1_rate*(1-fee1)

    max_money = max(money2gold,money2money,money2bitcoin)
    # money_history.append(max_money)
    
    if max_money == money2gold:
        new_choice = 'gold'
    elif max_money == money2bitcoin:
        new_choice = 'bitcoin'
    else:
        new_choice = 'money'

    # if last_choice == new_choice:
    #     real_money = 
    if last_choice == 'money':
        if new_choice == 'money':
            real_money = max_money
        elif new_choice == 'bitcoin':
            real_money = money_history[-1]*(1-fee2)*true_data2_rate
        else:
            real_money = money_history[-1]*(1-fee1)*true_data1_rate

    if last_choice == 'bitcoin':
        if new_choice == 'money':
            real_money = money_history[-1]*(1-fee2)
        elif new_choice == 'gold':
            real_money = money_history[-1]*(1-fee2)*(1-fee1)*true_data1_rate
        else:
            real_money = money_history[-1]*true_data2_rate

    if last_choice == 'gold':
        if new_choice == 'money':
            real_money = money_history[-1]*(1-fee1)
        elif new_choice == 'bitcoin':
            real_money = money_history[-1]*(1-fee1)*(1-fee2)*true_data2_rate
        else:
            real_money = money_history[-1]*true_data1_rate
    money_history.append(real_money)

    if last_choice!= new_choice:
        change_times+=1

    last_choice = new_choice

    if data1_period != 1 and last_choice == 'money':
        Day+=data1_period
        money_time.append(Day)
        continue


    Day += 1
    money_time.append(Day)
    
    

print(change_dict)
print(money_history[-1])
print(change_times)

plt.plot(money_time, money_history)
plt.show()