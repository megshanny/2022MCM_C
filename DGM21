import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
from scipy.optimize import leastsq

def DGM_21_minfunc(params, B, Y):
    x0, b = params
    return B.dot(params) - Y

def DGM_21_fit(data):
    B = np.vstack((-data[1:], np.ones(len(data)-1))).T
    Y = data[1:] - data[:-1]
    # p, pcov = leastsq(DGM_21_minfunc, [1, 1], args=(B, Y))
    p = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    return p

def DGM_21_sum_predicate(x0, p, x):
    a, b = p
    if(x < 0):
        return 0
    r1 = (b / a / a - x0 / a) * np.exp(-a * x)
    r2 = b / a * (x + 1)
    r3 = ( x0 - b / a) * (1 + a) / a
    return r1 + r2 + r3

def DGM_21_predicate(x0, p, x):
    return DGM_21_sum_predicate(x0, p, x) - DGM_21_sum_predicate(x0, p, x - 1)

def DGM_21_posttest(data, p):#均方差比，平均相对误差
    x0 = data[0]
    x_pred = np.array([DGM_21_predicate(x0, p, i) for i in range(len(data))])
    e = data - x_pred
    S1 = np.array(data).var()
    S2 = np.array(e).var()
    MSER = S2 / S1
    MRE = np.abs(e / data).mean()
    if(MSER > 10000000000):
        print(x_pred)
        print(data)
        print(MSER)
        exit()
    return MSER, MRE

data = pd.read_csv('D:\Files\code\Python\MCM\workspace\LBMA-GOLD.csv')
data = data.dropna(how='any')
data['DayFromBegin'] = pd.to_datetime(data['Date'],format='%m/%d/%y').map(lambda x: (x - pd.to_datetime('2016-9-11')).days)
data['RowNumber'] = range(0, len(data))
print(data.tail())


pre_start = 3
pre_length = 10
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
    p = DGM_21_fit(data_slice)
    MSER_SUM += DGM_21_posttest(data_slice, p)[0]
    pre_data_temp = []
    pre_data_day_temp = []
    for j in range(0, pre_duration):
        if(data_end + 1 + j > data['RowNumber'].max()):
            break
        pre_data_temp.append(DGM_21_predicate(data_slice[0], p, data_slice.__len__() + j))
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
