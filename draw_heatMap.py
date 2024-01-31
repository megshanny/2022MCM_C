import seaborn as sns
import matplotlib.pyplot as plt

# 数据
errors = [
          [0.028156484736807014, 0.03498940522025023],
          [0.006302700392062883,0.007751788940166315 ]
        ]

# 使用Seaborn的heatmap绘制
sns.set(font='Times New Roman')
sns.heatmap(errors, cmap="GnBu", annot=True, fmt=".5f", yticklabels=['Gold', 'Bitcoin'], xticklabels=['ARIMA', 'GP'])
plt.rcParams.update({'font.size': 14})
# 添加标签和标题
plt.ylabel('Mean Relative Error')
# plt.ylabel('Model')
# plt.title('Comparison of autoreg and GM(1,1) Errors')

# 显示图表
plt.show()
