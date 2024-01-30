import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

error_history1 = []
error_history2 = []
data1 = pd.read_csv('D:\Files\code\Python\MCM\workspace\LBMA-GOLD.csv')
rowname1 = 'USD (PM)'
data2 = pd.read_csv('D:\Files\code\Python\MCM\workspace\BCHAIN-MKPRU.csv')
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

Day = 50
pre_length = 5
fee1 = 0.01
fee2 = 0.02
picture = True
file_name = 'LSTM-50-2000-30-1.txt'

import sys
if len(sys.argv) >= 1:
    i = 1
    while(i < len(sys.argv)):
        if(sys.argv[i] == '--day'):
            Day = int(sys.argv[i+1])
            print('Day set to', Day)
            i += 2
        elif(sys.argv[i] == '--pre_length'):
            pre_length = int(sys.argv[i+1])
            print('pre_length set to', pre_length)
            i += 2
        elif(sys.argv[i] == '--fee1'):
            fee1 = float(sys.argv[i+1])
            print('fee1 set to', fee1)
            i += 2
        elif(sys.argv[i] == '--fee2'):
            fee2 = float(sys.argv[i+1])
            print('fee2 set to', fee2)
            i += 2
        elif(sys.argv[i] == '--hide'):
            picture = False
            i += 1
        elif(sys.argv[i] == '-f1'):
            temp1 = float(sys.argv[i+1])
        elif(sys.argv[i] == '-f2'):
            temp2 = float(sys.argv[i+1])
        elif(sys.argv[i] == '-i1'):
            temp1 = int(sys.argv[i+1])
        elif(sys.argv[i] == '-i2'):
            temp2 = int(sys.argv[i+1])
else:
    print("no argument, use default value")

data_dict = {}
with open('./workspace/file/'+file_name, 'r') as file:
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

name_dict = {0:'money', 1:'gold', 2:'bitcoin'}
choose_dict = {'gold': 0, 'bitcoin': 0, 'money': 0}
last_choose = 0
money_history = [1000,1000]
money_time = [0,Day]

while(Day < data1['DayFromBegin'].max()):
    data1_predict_date = data_dict[Day]['data1_predict_date']
    data1_predict_price = data_dict[Day]['data1_predict_price']
    data2_predict_date = data_dict[Day]['data2_predict_date']
    data2_predict_price = data_dict[Day]['data2_predict_price']
    true_data1_price = data1[data1['DayFromBegin'] > Day]['USD (PM)'].to_numpy()[0]
    true_data2_price = data2[data2['DayFromBegin'] > Day]['Value'].to_numpy()[0]
    data1_price = data1[data1['DayFromBegin'] <= Day]['USD (PM)'].to_numpy()[-1]
    data2_price = data2[data2['DayFromBegin'] <= Day]['Value'].to_numpy()[-1]
    error1 = data1_predict_price[0] - true_data1_price
    error2 = data2_predict_price[0] - true_data2_price
    if(true_data1_price - data1_price <=0):
        error1 = -error1
    if(true_data2_price - data2_price <=0):
        error2 = -error2
    error_history1.append(error1)
    error_history2.append(error2)
    if(data1_predict_date.__len__() > pre_length):
        data1_predict_date = data1_predict_date[:pre_length]
        data1_predict_price = data1_predict_price[:pre_length]
    if(data2_predict_date.__len__() > pre_length):
        data2_predict_date = data2_predict_date[:pre_length]
        data2_predict_price = data2_predict_price[:pre_length]
    choose_type, money = choose(data1_predict_price, data1_predict_date, data2_predict_price, data2_predict_date, fee1, fee2, Day, Day + pre_length, last_choose, calculate = True, begin_money = money_history[-1])
    choose_dict[name_dict[choose_type]] += 1
    money_history.append(money)
    money_time.append(Day)
    last_choose = choose_type
    Day += 1
    # if(Day % 20 == 0):
    #     print(Day, '/', data1['DayFromBegin'].max())

print(choose_dict)
print(money_history[-1])

if(picture):
    plt.subplot(3,1,1)
    plt.plot(error_history1)
    plt.subplot(3,1,2)
    plt.plot(error_history2)
    plt.subplot(3,1,3)
    plt.plot(money_time, money_history)
    plt.show()