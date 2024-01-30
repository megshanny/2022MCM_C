data_dict = {}
with open('./workspace/file/LSTM-50-2000-30.txt', 'r') as file:
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

print(data_dict[1820]['data1_predict_price'][1])
