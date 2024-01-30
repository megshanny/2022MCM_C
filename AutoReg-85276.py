import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

error_history1 = []
error_history2 = []
error_check1 = []
error_check2 = []

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

# data_dict = {}
# with open('./workspace/file/'+'LSTM-50-40-30-1.txt', 'r') as file:
#     current_day = None
#     current_data = None
#     for line in file:
#         if 'Day:' in line:
#             current_day = int(line.split()[1])
#             data_dict[current_day] = {}
#         elif 'data' in line:
#             current_data = line.split(':')[0].strip()
#             data_dict[current_day][current_data] = {}
#             if 'price' in line:
#                 data_values = [float(value) for value in line.split('[')[1].split(']')[0].split()]
#             elif 'date' in line:
#                 data_values = [int(value) for value in line.split('[')[1].split(']')[0].split()]
#             data_dict[current_day][current_data] = data_values

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

def GM_11_predicate(x0, p, x1, x2):
    res = []
    for i in range(x1,x2):
        res.append(GM_11_sum_predicate(x0, p, i) - GM_11_sum_predicate(x0, p, i - 1))
    return np.array(res)

from statsmodels.tsa.ar_model import AutoReg
'''
87342: 0.7*68_5 0.3*21_1
97408:0.618*68_5 0.382*27~29_2
98269:params = [[67,5,0.618,0]  #长期
         ,[68,5,0.618,0]
         ,[69,5,0.618,0]
         ,[27,2,0.382,0]  #短期
         ,[28,2,0.382,0]
         ,[29,2,0.382,0]] 
95945:短期一天
106255:短期三天
'''

'''
0.0001 0.0002:154976 {'gold': 456, 'bitcoin': 1001, 'money': 318}
0.001,0.002:117328 {'gold': 414, 'bitcoin': 1027, 'money': 334} = 
0.005,0.01: 49830  {'gold': 382, 'bitcoin': 1119, 'money': 274} = 
0.009,0.015:43982  {'gold': 173, 'bitcoin': 1075, 'money': 527}
0.009,0.018:50214  {'gold': 173, 'bitcoin': 1099, 'money': 503}
0.099,0.0198:94537 {'gold': 194, 'bitcoin': 1067, 'money': 514}

0.0101, 0.0202:82744 {'gold': 194, 'bitcoin': 1007, 'money': 574}
0.011,0.022:35911 {'gold': 0, 'bitcoin': 1040, 'money': 735}
0.015,0.025:16526 {'gold': 0, 'bitcoin': 980, 'money': 795}
'''
#这里的参数是[数据长度,预测长度,权重，方法]，或[黄金的四个参数,比特币的四个参数]
params = [[67,5,0.618,0]  #长期
        #  ,[68,5,0.618,0]
         ,[69,5,0.618,0]

         ,[27,3,0.382,0]  #短期
        #  ,[28,3,0.382,0]
         ,[29,3,0.382,0]
         ] 

Day = 50
fee1 = 0.01
fee2 = 0.02
picture = True

def predict_0(data_price, length):
    data_need_predict = np.diff(data_price)
    model = AutoReg(data_need_predict,lags=1)
    model_fit = model.fit()
    data_predict = model_fit.predict(len(data_need_predict), len(data_need_predict)+length-1)
    data_predict_price = np.cumsum(data_predict) + data_price[-1]
    return data_predict_price

def predict_1(data_price, length):
    data_need_predict = np.diff(data_price)
    model = AutoReg(data_need_predict,lags=1,trend = 'ct')
    model_fit = model.fit()
    data_predict = model_fit.predict(len(data_need_predict), len(data_need_predict)+length-1)
    data_predict_price = np.cumsum(data_predict) + data_price[-1]
    return data_predict_price

# def predict_2(Day, length, type):
#     if(type == 1):
#         return data_dict[Day]['data1_predict_price'][-length:]
#     if(type == 2):
#         return data_dict[Day]['data2_predict_price'][-length:]

def predict(data_price, length, way, Day, type):
    if(way == 0):
        return predict_0(data_price, length)
    if(way == 1):
        return predict_1(data_price, length)
    # if(way == 2):
    #     return predict_2(Day, length, type)

name_dict = {0:'money', 1:'gold', 2:'bitcoin'}
choose_dict = {'gold': 0, 'bitcoin': 0, 'money': 0}
last_choose = 0
money_history = [1000,1000]
money_time = [0,Day]

while(Day < data1['DayFromBegin'].max()):
    data1_price = data1[data1['DayFromBegin'] <= Day][rowname1].to_numpy()
    data2_price = data2[data2['DayFromBegin'] <= Day][rowname2].to_numpy()
    max_data1_predict_length = 0
    max_data2_predict_length = 0
    for i in range(0, params.__len__()):
        if(params[i][1] > max_data1_predict_length):
            max_data1_predict_length = params[i][1]
        if(params[i].__len__() > 4):
            if(params[i][5] > max_data2_predict_length):
                max_data2_predict_length = params[i][5]
        else:
            if(params[i][1] > max_data2_predict_length):
                max_data2_predict_length = params[i][1]
    data1_predict_price = np.zeros(max_data1_predict_length)
    data2_predict_price = np.zeros(max_data2_predict_length)
    data1_predict_weight = np.zeros(max_data1_predict_length)
    data2_predict_weight = np.zeros(max_data2_predict_length)
    for i in range(0, params.__len__()):
        data_length_1 = params[i][0]
        pre_length_1 = params[i][1]
        weight1 = params[i][2]
        way1 = params[i][3]
        if(params[i].__len__() > 4):
            data_length_2 = params[i][4]
            pre_length_2 = params[i][5]
            weight2 = params[i][6]
            way2 = params[i][7]
        else:
            data_length_2 = data_length_1
            pre_length_2 = pre_length_1
            weight2 = weight1
            way2 = way1
        if(data_length_1 > data1_price.__len__()):
            data_length_1 = data1_price.__len__()
        if(data_length_2 > data1_price.__len__()):
            data_length_2 = data1_price.__len__()
        data1_predict_once = predict(data1_price[-data_length_1:], pre_length_1, way1, Day, 1)
        data2_predict_once = predict(data2_price[-data_length_2:], pre_length_2, way2, Day, 2)
        for i in range(0, pre_length_1):
            data1_predict_price[i] += data1_predict_once[i] * weight1
            data1_predict_weight[i] += weight1
        for i in range(0, pre_length_2):
            data2_predict_price[i] += data2_predict_once[i] * weight2
            data2_predict_weight[i] += weight2
    data1_predict_price = data1_predict_price / data1_predict_weight
    data2_predict_price = data2_predict_price / data2_predict_weight
    true_data1_price = data1[data1['DayFromBegin'] > Day]['USD (PM)'].to_numpy()[0]
    true_data2_price = data2[data2['DayFromBegin'] > Day]['Value'].to_numpy()[0]
    error1 = (data1_predict_price[0] - true_data1_price) / true_data1_price
    error2 = (data2_predict_price[0] - true_data2_price) / true_data2_price
    if(true_data1_price - data1_price[-1] <=0):
        error1 = -error1
    if(true_data2_price - data2_price[-1] <=0):
        error2 = -error2
    error_history1.append(error1)
    error_history2.append(error2)
    data1_predict_date = data1[data1['DayFromBegin'] > Day]['DayFromBegin'].to_numpy()
    if(data1_predict_date.__len__() < max_data1_predict_length):
        data1_predict_price = data1_predict_price[0:data1_predict_date.__len__()]
    else:
        data1_predict_date = data1_predict_date[0:max_data1_predict_length]
    data2_predict_date = data2[data2['DayFromBegin'] > Day]['DayFromBegin'].to_numpy()
    if(data2_predict_date.__len__() < max_data2_predict_length):
        data2_predict_price = data2_predict_price[0:data2_predict_date.__len__()]
    else:
        data2_predict_date = data2_predict_date[0:max_data2_predict_length]
    pre_length = min(max_data1_predict_length, max_data2_predict_length)
    choose_type, money = choose(data1_predict_price, data1_predict_date, data2_predict_price, data2_predict_date, fee1, fee2, Day, Day + pre_length, last_choose, calculate = True, begin_money = money_history[-1])
    choose_dict[name_dict[choose_type]] += 1
    money_history.append(money)
    money_time.append(Day)
    last_choose = choose_type
    Day += 1

    if(Day % 50 == 0):
        print(Day, '/', data1['DayFromBegin'].max())

print(choose_dict)
print(money_history[-1])

if(picture):
    # plt.figure(figsize=(8,15))
    # plt.subplot(5,1,1)
    # plt.plot(error_history1)
    # plt.subplot(5,1,2)
    # plt.plot(error_history2)
    # plt.subplot(5,1,3)
    # plt.plot(error_check1)
    # plt.subplot(5,1,4)
    # plt.plot(error_check2)
    # plt.subplot(5,1,5)
    plt.plot()
    plt.plot(money_time, money_history)
    plt.show()