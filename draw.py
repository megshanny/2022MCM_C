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

# plt.plot(data1['DayFromBegin'],data1[rowname1])
# plt.show()
plt.plot(data2['DayFromBegin'],data2[rowname2])
plt.show()