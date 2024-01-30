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
    p, pcov = leastsq(GM_11_minfunc, [1, 1], args=(B, Y))
    return p

def GM_11_sum_predicate(x0, a, b, x):
    if(x < 0):
        return 0
    return (x0 - b / a) * np.exp(-a * x) + b / a

def GM_11_predicate(x0, a, b, x):
    return GM_11_sum_predicate(x0, a, b, x) - GM_11_sum_predicate(x0, a, b, x - 1)

def GM_11_posttest(data, p):#均方差比，平均相对误差
    a, b = p
    x0 = data[0]
    x_pred = np.array([GM_11_predicate(x0, a, b, i) for i in range(len(data))])
    e = data - x_pred
    S1 = np.array(data).var()
    S2 = np.array(e).var()
    MSER = S2 / S1
    MRE = np.abs(e / data).mean()
    return MSER, MRE

data = pd.read_csv('D:\Files\code\Python\MCM\workspace\LBMA-GOLD.csv')
data = data.dropna(how='any')
data['DayFromBegin'] = pd.to_datetime(data['Date'],format='%m/%d/%y').map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data['RowNumber'] = range(0, len(data))
print(data.tail())

pre_start = 3
pre_length = 40
pre_duration = 7

pre_data = []
pre_data_day = []
MSER_SUM = 0
count = 0
error = np.zeros(pre_duration).astype(np.float64)

for i in range(pre_start, data['RowNumber'].max()+2):
    data_start = i - pre_length
    data_end = i
    if(data_start < 0):
        data_start = 0
    data_slice = data[data_start:data_end]['USD (PM)'].to_numpy()
    a,b = GM_11_fit(data_slice)
    MSER_SUM += GM_11_posttest(data_slice, [a, b])[0]
    pre_data_temp = []
    pre_data_day_temp = []
    for j in range(0, pre_duration):
        if(data_end + 1 + j > data['RowNumber'].max()):
            break
        pre_data_temp.append(data_slice[-1])
        pre_data_day_temp.append(data_end + 1 + j)
    if pre_data_temp.__len__() != 0:
        pre_data.append(pre_data_temp)
        pre_data_day.append(pre_data_day_temp)
        data_true = data[(data['RowNumber'] >= pre_data_day_temp[0]) & (data['RowNumber'] <= pre_data_day_temp[-1])]['USD (PM)'].to_numpy()
        error_temp = np.abs(pre_data_temp - data_true) / data_true
        error_new_temp = np.zeros(pre_duration).astype(np.float64)
        error_new_temp[0:error_temp.__len__()] += error_temp
        error += error_new_temp
    count += 1
    if(i % 100 == 0):
        print(i, '/', data['RowNumber'].max())
error = error / count
print(error)
print('Mean Error:', error.mean())
print('Mean MSER:', MSER_SUM / count)

from matplotlib.widgets import Slider
#绘图，以黑色显示真实数据，以红色显示预测数据，添加拖拽条用于显示从那一天开始预测得到的数据
fig = plt.figure()
ax = fig.add_subplot()
plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gold Price Prediction')
plt.plot(data['Date'], data['USD (PM)'], color='black')
line, = ax.plot(pre_data_day[0], pre_data[0], color='red')
axcolor = 'lightgoldenrodyellow'
axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
spos = Slider(axpos, 'Pos', 0, data['RowNumber'].max()- pre_start + 1, valstep= 1, valinit=0)
def update(val):
    pos = spos.val
    line.set_xdata(pre_data_day[pos])
    line.set_ydata(pre_data[pos])
    fig.canvas.draw_idle()
    data_predict = pre_data[pos]
    data_true = data[(data['RowNumber'] >= pre_data_day[pos][0]) & (data['RowNumber'] <= pre_data_day[pos][-1])]['USD (PM)'].to_numpy()
    error = np.abs(data_predict - data_true) / data_true
    print('Mean Error:', error.mean(), 'Day:', pre_data_day[pos][0], 'to', pre_data_day[pos][-1])
spos.on_changed(update)
plt.show()
