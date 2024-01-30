#本文件的功能，打开一个文件，在这个文件中如果任何一行首行是空格，就将这一行于上一行合并，将处理后的文件保存到另一个文件中
last_day =49
with open('./LSTM-all.txt', 'r') as file:
    with open('./LSTM-all-1.txt', 'w') as file1:
        for line in file:
            if line[0] == ' ':
                file1.write(' ' + line.strip())
            else:
                file1.write('\n' + line.strip())
