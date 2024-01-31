import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Button

money_time = [1,2,3,4,5,6,7,8,9,10]
money_history = [3,7,8,2,19,0,9,10,11,10]


base_date = '2016-9-11'
x1 = pd.to_timedelta(money_time, unit='D')
x1 = pd.to_datetime(base_date) + x1 
plt.figure(dpi=200)
sns.set_style('whitegrid')       # 图片风格
sns.set(font='Times New Roman')  # 图片全局字体

sns.lineplot(x=x1,y=money_history)
plt.xticks(fontweight='bold')  # 横坐标值加粗
plt.yticks(fontweight='bold')  # 纵坐标值加粗

plt.title('Daily Investment Worth Increase under Model-based Decision',fontweight = 'bold')
plt.xlabel('Date')
plt.ylabel('Investment Worth($)')
# plt.show()

# plt.savefig("test.png",dpi=300,fontsize = (6,4))
buttonax = plt.axes([0.1, 0.9, 0.1, 0.1])
button = Button(buttonax, ' ')
def button_press_event(event):
    plt.delaxes(ax=buttonax)
    plt.draw()
    plt.savefig('test.png',dpi = 300)
button.on_clicked(button_press_event)
plt.show()
# plt.savefig("test.svg", dpi=300,format="svg") 保存为矢量图