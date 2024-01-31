import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ARIMA = pd.read_csv('./error_history_ARIMA.csv')
GP = pd.read_csv('./error_history_GP.csv')

data2 = pd.read_csv('.\BCHAIN-MKPRU.csv')
data2 = data2.dropna(how='any')
data2['Date'] = pd.to_datetime(data2['Date'],format='%m/%d/%y')
# 使用 iloc 方法去掉前50行
df_after_drop = data2.iloc[50:]
# print(df_after_drop['Date'])

errorA1 = ARIMA['error1']
errorA2 = ARIMA['error2']

errorD1 = GP['error1']
errorD2 = GP['error2']

#-----------------画饼--------------------
# 数据
data = [1073, 702]
labels = ['A', 'B']

# 绘制饼状图
plt.figure(figsize=(6, 6))
sns.set_palette("pastel")  # 设置颜色风格

plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140)

# 添加标题
plt.title('Pie Chart: A vs B')

# 显示图表
plt.show()
