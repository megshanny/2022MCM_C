import numpy as np
from sympy import symbols, Function, Eq, diff, dsolve
from numpy.linalg import lstsq
import pandas as pd
import matplotlib.pyplot as plt

t = symbols('t')  # 定义符号变量 t
x = Function('x')(t)  # 定义函数 x(t)

def GM_21_fit(data):
    n = len(data)  # 原始序列元素个数
    x1 = np.cumsum(data)  # 原始序列的累积和
    ax = np.diff(data)  # 对原始序列进行一阶差分
    ax = np.concatenate([[0], ax])  # 在差分结果前面补0，保持序列长度一致
    z1 = np.zeros(n)  # 初始化z序列
    for i in range(1, n):  # 对每一个元素：
        z1[i] = 0.5 * (x1[i] + x1[i - 1])  # 计算相邻元素的平均值
    B = np.column_stack((-data[1:], -z1[1:], np.ones(n - 1)))  # 构建 B，其中包含负的 data 序列元素（从第二个元素开始）、负的 z 序列元素（从第二个元素开始）以及单位列向量
    Y = ax[1:]  # 构建 Y，由一阶差分序列 ax （从第二个元素开始）
    u, _, _, _ = lstsq(B, Y, rcond=None)  # 运用 Numpy 的最小二乘法 lstsq 求解线性方程得到 u    
    a1, a2, b, = u[0], u[1], u[2]  # 从参数 u 提取所需参数 a1，a2 和 b
    diff_eq = Eq(diff(x, t, t) + a1 * diff(x, t) + a2 * x, b)  # 定义微分方程
    ics = {x.subs(t, 0): x1[0], x.subs(t, 5): x1[-1]}  # 设定初始条件为 x(0) = x1[0]，x(5) = x1[5] （注意这里索引是以 0 开始的）
    model = dsolve(diff_eq, ics=ics)  # 解微分方程
    return model

def GM_21_sum_predicate(model, x):
    if(x < 0):
        return 0
    return model.rhs.subs(t, x).evalf()

def GM_21_predicate(model, x):
    return GM_21_sum_predicate(model, x) - GM_21_sum_predicate(model, x - 1)

def GM_21_posttest(data, model):#均方差比，平均相对误差
    x_pred = np.array([GM_21_predicate(model, i) for i in range(len(data))])
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
pre_length = 6
pre_duration = 10

pre_data = []
pre_data_day = []
MSER_SUM = 0
count = 0
error = np.zeros(pre_duration).astype(np.float64)

for i in range(pre_start, data['RowNumber'].max()+2):
    data_start = i - pre_length
    data_end = i
    if(data_start < 0):
        continue
    data_slice = data[data_start:data_end]['USD (PM)'].to_numpy()
    model = GM_21_fit(data_slice)
    MSER_SUM += GM_21_posttest(data_slice, model)[0]
    print(GM_21_posttest(data_slice, model)[0])
    pre_data_temp = []
    pre_data_day_temp = []
    for j in range(0, pre_duration):
        if(data_end + 1 + j > data['RowNumber'].max()):
            break
        pre_data_temp.append(GM_21_predicate(model, data_slice.__len__() + j))
        pre_data_day_temp.append(data_end + 1 + j)
    if pre_data_temp.__len__() != 0:
        pre_data.append(pre_data_temp)
        pre_data_day.append(pre_data_day_temp)
        data_true = data[(data['RowNumber'] >= pre_data_day_temp[0]) & (data['RowNumber'] <= pre_data_day_temp[-1])]['USD (PM)'].to_numpy()
        error_temp = np.abs(pre_data_temp - data_true) / data_true
        error_new_temp = np.zeros(pre_duration).astype(np.float64)
        for k in range(0, error_temp.__len__()):
            error_new_temp[k] = np.float64(error_temp[k])
        error += error_new_temp
    count += 1
    print(i, '/', data['RowNumber'].max())
    if(i == 50):
        break
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
