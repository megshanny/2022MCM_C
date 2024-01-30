import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

choose_warning_list = []
# 参数：
# gold_pre: data1的预测价格(不含当日)，data_length数组
# bitcoin_date: data1_pre的对应日期，data_length数组
# gold_pre: data2的预测价格(不含当日)，data_length数组
# bitcoin_date: data2_pre的对应日期，data_length数组
# gold_fee: data1的手续费(0-1)
# bitcoin_fee: data2的手续费(0-1)
# begin_date:开始日期
# end_date:结束日期(不会在这一天进行操作)
# begin_type:开始资产类型(0-2, 0:money, 1:gold, 2:bitcoin)
# calculate:是否计算收益
# begin_money:开始资产金额(仅在calculate为True时有效,否则固定为1000)
# 返回值：
# type: 0-2, 0:money, 1:gold, 2:bitcoin, 表示第一天选择的资产类型
# money: 最终收益，如果calculate为False，固定返回1000
# 全局变量：
# data1: 金价数据
# data2: 比特币价
# choose_warning_list: 用于记录警告信息，防止重复输出
def choose(gold_pre, gold_date, bitcoin_pre, bitcoin_date, gold_fee, bitcoin_fee, begin_date, end_date, begin_type, calculate = False, begin_money = 1000):
    def mymax(p0, p1, p2, prior = 0):
        p = [p0, p1, p2]
        max = p[prior]
        pos = prior
        if(p0 > max):
            max = p0
            pos = 0
        if(p1 > max):
            max = p1
            pos = 1
        if(p2 > max):
            max = p2
            pos = 2
        return max, pos

    if(gold_pre.__len__() != gold_date.__len__()):
        print('Error: gold_pre and gold_date have different length')
        print(gold_pre.__len__(), gold_date.__len__())
        return begin_type, 1000.0
    if(bitcoin_pre.__len__() != bitcoin_date.__len__()):
        print('Error: bitcoin_pre and bitcoin_date have different length')
        print(bitcoin_pre.__len__(), bitcoin_date.__len__())
        return begin_type, 1000.0
    if(end_date <= begin_date):
        print('Error: end_date <= begin_date')
        return begin_type, 1000.0
    if(gold_fee < 0 or gold_fee > 1 or bitcoin_fee < 0 or bitcoin_fee > 1):
        if('fee_warn' not in choose_warning_list):
            print('Warning: gold_fee or bitcoin_fee is not in [0,1]')
            print('gold_fee, bitcoin_fee:', gold_fee, bitcoin_fee)
            choose_warning_list.append('fee_warn')
    if(begin_type < 0 or begin_type > 2):
        print('Error: begin_type is not in [0,2]')
        print('begin_type:', begin_type)
        return begin_type, 1000.0
    if(begin_money <= 0):
        print('Warning: begin_money < 0, set to 1000')
        print('begin_money:', begin_money)
        begin_money = 1000
    
    now_gold = data1[data1['DayFromBegin'] <= begin_date]['USD (PM)'].to_numpy()
    if(now_gold.__len__() == 0):
        now_gold_price = 1
    else:
        now_gold_price = now_gold[-1]
    now_bitcoin_price = data2[data2['DayFromBegin'] <= begin_date]['Value'].to_numpy()[-1]
    gold_pre = np.insert(gold_pre, 0, now_gold_price)
    bitcoin_pre = np.insert(bitcoin_pre, 0, now_bitcoin_price)
    gold_date = np.insert(gold_date, 0, begin_date)
    bitcoin_date = np.insert(bitcoin_date, 0, begin_date)

    start_dict = {}
    start_dict[begin_date] = [0.0,0.0,0.0]
    if(calculate):
        start_dict[begin_date][begin_type] = begin_money
    else:
        start_dict[begin_date][begin_type] = 1000.0
    start_dict_fromtype = {}
    end_dict = {}
    k1 = 1 - gold_fee
    k2 = 1 - bitcoin_fee
    for day in range(begin_date, end_date):
        today_gold_open = day in gold_date
        end_dict[day] = [0.0,0.0,0.0]
        start_dict_fromtype[day+1] = [0,0,0]
        if(today_gold_open):
            end_dict[day][0], start_dict_fromtype[day+1][0] = mymax(start_dict[day][0], start_dict[day][1] * k1, start_dict[day][2] * k2, prior = 0)
            end_dict[day][1], start_dict_fromtype[day+1][1] = mymax(start_dict[day][0] * k1, start_dict[day][1], start_dict[day][2] * k2 * k1, prior = 1)
            end_dict[day][2], start_dict_fromtype[day+1][2] = mymax(start_dict[day][0] * k2, start_dict[day][1] * k2 * k1, start_dict[day][2], prior = 2)
        else:
            end_dict[day][0], start_dict_fromtype[day+1][0] = mymax(start_dict[day][0], -1, start_dict[day][2] * k2, prior = 0)
            end_dict[day][1], start_dict_fromtype[day+1][1] = mymax(-1, start_dict[day][1], -1, prior = 1)
            end_dict[day][2], start_dict_fromtype[day+1][2] = mymax(start_dict[day][0] * k2, -1, start_dict[day][2], prior = 2)
        try:
            today_gold_price = gold_pre[gold_date <= day][-1]
            next_gold_price = gold_pre[gold_date <= day + 1][-1]
            today_bitcoin_price = bitcoin_pre[bitcoin_date <= day][-1]
            next_bitcoin_price = bitcoin_pre[bitcoin_date <= day + 1][-1]
        except:
            print('Error: lack of data on day', day)
            return begin_type, 1000.0
        start_dict[day+1] = [0.0,0.0,0.0]
        start_dict[day+1][0]=end_dict[day][0]
        start_dict[day+1][1]=end_dict[day][1] / today_gold_price * next_gold_price
        start_dict[day+1][2]=end_dict[day][2] / today_bitcoin_price * next_bitcoin_price

    now_time = end_date
    _, now_choice = mymax(start_dict[now_time][0], start_dict[now_time][1], start_dict[now_time][2])
    choices = [now_choice]
    while(now_time != begin_date):
        now_choice = start_dict_fromtype[now_time][now_choice]
        now_time -= 1
        choices.append(now_choice)
    choices.reverse()
    if(begin_type != choices[0]):
        print('Error: Unown error in calculating choice type')
        return begin_type, 1000.0

    money = 1000.0
    if(calculate):
        money = begin_money
        today_bitcoin_price_true = data2[data2['DayFromBegin'] <= begin_date]['Value'].to_numpy()[-1]
        today_gold_price_true = gold_pre[gold_date <= begin_date][-1]
        next_gold_price_true = data1[data1['DayFromBegin'] <= begin_date + 1]['USD (PM)'].to_numpy()[-1]
        next_bitcoin_price_true = data2[data2['DayFromBegin'] <= begin_date + 1]['Value'].to_numpy()[-1]
        if(begin_type != choices[1]):
            if(begin_type == 1 or choices[1] == 1):
                money *= k1
            if(begin_type == 2 or choices[1] == 2):
                money *= k2
        if(choices[1] == 1):
            money = money / today_gold_price_true * next_gold_price_true
        if(choices[1] == 2):
            money = money / today_bitcoin_price_true * next_bitcoin_price_true

    return choices[1], money

Day_Begin = 1000
Day_End = 1200
data_length =2000
pre_length = 5
fee1 = 0.01
fee2 = 0.02
picture = False

from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error



import sys
if len(sys.argv) >= 1:
    i = 1
    while(i < len(sys.argv)):
        if(sys.argv[i] == '--DayBegin'):
            Day_Begin = int(sys.argv[i+1])
            i += 2
        elif(sys.argv[i] == '--DayEnd'):
            Day_End = int(sys.argv[i+1])
            i += 2
else:
    print("no argument, use default value")

Day = Day_Begin

file_name = 'LSTM-' + str(Day_Begin) + '-' + str(Day_End) +'.txt'
with open(file_name, 'w') as f:
    def predict(data_price):
        data_diff = np.diff(data_price)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(data_diff.reshape(-1, 1))
        data_scaled = scaler.transform(data_diff.reshape(-1, 1))
        X = data_scaled[:-1].reshape(-1, 1, 1)
        y = data_scaled[1:]
        model = Sequential()
        model.add(LSTM(4,batch_input_shape=(1,1,1),stateful=True)) #神经元个数为4
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        i = 1
        last_loss = 0
        while i <= 300:
            history = model.fit(X, y, batch_size=1, verbose=1, shuffle=False)
            loss = history.history['loss'][0]
            model.reset_states()
            if(i >= 5 and abs(loss - last_loss) < 0.00003):
                break
            last_loss = loss
            i += 1
        print('Epoch:', i, file = f)
        for i in range(0,pre_length):
            data_predict = model.predict(data_scaled[-1:].reshape(-1,1,1),batch_size=1)
            data_scaled = np.append(data_scaled, data_predict)
        data_predict_scaled = data_diff[-pre_length:]
        data_predict_diff = scaler.inverse_transform(data_predict_scaled.reshape(-1, 1))
        data_predict = np.cumsum(data_predict_diff) + data_price[-1]
        return data_predict

    name_dict = {0:'money', 1:'gold', 2:'bitcoin'}
    choose_dict = {'gold': 0, 'bitcoin': 0, 'money': 0}
    last_choose = 0
    money_history = [1000,1000]
    money_time = [0,Day]

    while(Day < Day_End + 1):
        try:
            print('Day:', Day, file = f)
            data1_slice = data1[data1['DayFromBegin'] <= Day]
            if(data_length <= data1_slice.__len__()):
                data1_slice = data1_slice.iloc[-data_length:]
            else:
                data1_slice = data1_slice.iloc[0:]
            data1_price = data1_slice[rowname1].to_numpy()

            data2_slice = data2[data2['DayFromBegin'] <= Day]
            if(data_length <= data2_slice.__len__()):
                data2_slice = data2_slice.iloc[-data_length:]
            else:
                data2_slice = data2_slice.iloc[0:]
            data2_price = data2_slice[rowname2].to_numpy()
            
            data1_predict_price = predict(data1_price)
            data2_predict_price = predict(data2_price)

            data1_predict_date = data1[data1['DayFromBegin'] > Day]['DayFromBegin'].to_numpy()
            if(data1_predict_date.__len__() < pre_length):
                data1_predict_price = data1_predict_price[0:data1_predict_date.__len__()]
            else:
                data1_predict_date = data1_predict_date[0:pre_length]
            data2_predict_date = data2[data2['DayFromBegin'] > Day]['DayFromBegin'].to_numpy()
            if(data2_predict_date.__len__() < pre_length):
                data2_predict_price = data2_predict_price[0:data2_predict_date.__len__()]
            else:
                data2_predict_date = data2_predict_date[0:pre_length]
            print('data1_predict_price:', data1_predict_price, file = f)
            print('data1_predict_date:', data1_predict_date, file = f)
            print('data2_predict_price:', data2_predict_price, file = f)
            print('data2_predict_date:', data2_predict_date, file = f)
            print('Success', file = f)
            choose_type, money = choose(data1_predict_price, data1_predict_date, data2_predict_price, data2_predict_date, fee1, fee2, Day, Day + pre_length, last_choose, calculate = True, begin_money = money_history[-1])
            choose_dict[name_dict[choose_type]] += 1
            money_history.append(money)
            money_time.append(Day)
            last_choose = choose_type
        except:
            print('Error', file = f)
        Day += 1
        # if(Day % 20 == 0):
        #     print(Day, '/', data1['DayFromBegin'].max())

    if(picture):
        plt.plot(money_time, money_history)
        plt.show()