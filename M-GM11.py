import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
from scipy.optimize import leastsq

def GM_11_pretest(data):
    ago_sequence = np.cumsum(data)
    smooth_ratios = ago_sequence[1:] / ago_sequence[:-1]
    smooth_ratios = smooth_ratios[1:]
    satisfying_ratios = (smooth_ratios >= 1) & (smooth_ratios <= 1.5)
    satisfying_ratio_percentage = np.sum(satisfying_ratios) / len(satisfying_ratios)
    return satisfying_ratio_percentage

def GM_11_minfunc(params, B, Y):
    x0, b = params
    return B.dot(params) - Y

def GM_11_fit(data):
    X1 = data.cumsum()
    X1 = (X1[:-1]+X1[1:])/2.0
    B = np.vstack((-X1, np.ones(len(X1)))).T
    Y = data[1:]
    # p, pcov = leastsq(GM_11_minfunc, [1, 1], args=(B, Y))
    p = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    return p

def GM_11_sum_predicate(x0, p, x):
    a, b = p
    if(x < 0):
        return 0
    return (x0 - b / a) * np.exp(-a * x) + b / a

def GM_11_predicate(x0, p, x):
    return GM_11_sum_predicate(x0, p, x) - GM_11_sum_predicate(x0, p, x - 1)

def GM_11_posttest(data, p):#均方差比，平均相对误差
    x0 = data[0]
    x_pred = np.array([GM_11_predicate(x0, p, i) for i in range(len(data))])
    e = data - x_pred
    S1 = np.array(data).var()
    S2 = np.array(e).var()
    MSER = S2 / S1
    MRE = np.abs(e / data).mean()
    return MSER, MRE

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

Day = 5
pre_length = 4
last_choice = 'money'
change_dict = {'gold': 0, 'bitcoin': 0, 'money': 0}

fee1 = 0.0
fee2 = 0.0

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
    p1 = GM_11_fit(data1_price)
    p2 = GM_11_fit(data2_price)
    data1_predict = GM_11_predicate(data1_price[0], p1, data1_price.__len__())
    data2_predict = GM_11_predicate(data2_price[0], p2, data2_price.__len__())

    # Test1 41829
    data1_rate = data1_predict / data1_price[-1]
    data2_rate = data2_predict / data2_price[-1]

    # Test2 82349
    # weights = np.array([1,1.1,1])
    # data1_rate = (data1_price[-1] / data1_price[-2] * weights[0] +
    #               data1_price[-2] / data1_price[-3] * weights[1] + 
    #               data1_price[-3] / data1_price[-4] * weights[2])
    # data2_rate = (data2_price[-1] / data2_price[-2] * weights[0] +
    #               data2_price[-2] / data2_price[-3] * weights[1] + 
    #               data2_price[-3] / data2_price[-4] * weights[2])
    # data1_rate = data1_rate / weights.sum()
    # data2_rate = data2_rate / weights.sum()

    data1_period = data1.iloc[data1_slice['RowNumber'].max() + 1]['DayFromBegin'] - data1_slice['DayFromBegin'].max()
    data2_period = data2.iloc[data2_slice['RowNumber'].max() + 1]['DayFromBegin'] - data2_slice['DayFromBegin'].max()
    if(data1_slice['DayFromBegin'].max()!=Day):
        data1_rate = 0
    if(data2_slice['DayFromBegin'].max()!=Day):
        data2_rate = 0

    true_data1_rate = data1.iloc[data1_slice['RowNumber'].max() + 1][rowname1] / data1_price[-1]
    true_data2_rate = data2.iloc[data2_slice['RowNumber'].max() + 1][rowname2] / data2_price[-1]
    if(pow(data1_rate, data2_period) - fee1 > pow(data2_rate, data1_period) - fee2):
        if(data1_rate - fee1 > 1):
            Day += data1_period
            money_history.append(money_history[-1] * (true_data1_rate - fee1))
            money_time.append(Day)
            new_choice = 'gold'
        else:
            Day += 1
            money_history.append(money_history[-1])
            money_time.append(Day)
            new_choice = 'money'
    else:
        if(data2_rate - fee2 > 1):
            Day += data2_period
            money_history.append(money_history[-1] * (true_data2_rate - fee2))
            money_time.append(Day)
            new_choice = 'bitcoin'
        else:
            Day += 1
            money_history.append(money_history[-1])
            money_time.append(Day)
            new_choice = 'money'
    if(new_choice != last_choice):
        print(last_choice, '->', new_choice, 'on day', Day)
        last_choice = new_choice
        change_dict[new_choice] += 1

print(change_dict)
print(money_history[-1])

plt.plot(money_time, money_history)
plt.show()