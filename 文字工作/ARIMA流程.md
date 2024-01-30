# ARIMA流程

## 数据预处理：

1. 空数据，查阅了网站，发现网站上只有开盘，没有收盘，所以按照休市处理了。

2. 使用ADF 判断出来比特币和黄金价格的不平稳性。

   ```
   p value>0.05: Accepts the Null Hypothesis (H0), the data has a unit root and is non-stationary.
   p value>0.05 ：接受随机假设（H 0），数据具有单位根并且是非平稳的。
   
   p value<=0.05: Rejects the Null Hypothesis (H0), the data is stationary. The more negative it is, the stronger the rejection of the hypothesis that there is a unit root at some level of confidence.
   p value<=0.05 ：重新假设H0，数据是平稳的。它越是负的，就越强烈地拒绝在某种置信水平下存在单位根的假设。
   ```

   > 从论文里摘出来的，感觉可以写进去

   1. ADF 值大于0.05，认为数据并不平稳
   2. ADF检验结果： 0.9042384812941663>0.05

2. 对数列进行一阶差分
   1. 一阶差分ADF检验结果： 9.26971142153572e-13
   2. 一阶差分后变得平稳
3. 白噪声检验：
   1. lb_pvalue小于0.05说明是白噪声
   2. 对一阶差分后的数列进行白噪声检验，lb_pvalue = 0.424308，说明不是白噪声

### 模型训练：

2. 训练模型

   在开始时，由于我们没有足够的数据来预测走势，为了保险，我们就先啥也不干。

   从（）天之后，我们开始预测未来的数据并且开始进行投资。

   为了反应数据长期趋势，同时兼顾短期内可能的异常值，我们采用了长期，中期，短期混合的预测模型。

   长期设置为根据过去（）天的预测接下来（）天

   中期设置为根据过去（）天的预测接下来（）天

   短期期设置为根据过去（）天的预测接下来（）天

   同时三者的比例是（*, *, *)

   

