from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import time
#设置中文字体
# plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 生成横纵坐标信息





MemCount='mempool-count.csv'
TxBlockCount='transactionperblock.csv'


txdata = pd.read_csv(TxBlockCount, sep=",")
MemCount=pd.read_csv(MemCount, sep=",")



def ExtractXAndY(Infor):
    valus=Infor.values
    X=valus[:,0]/1000
    Y=valus[:,1]
    TimeX=[]
    for i in range(X.shape[0]):
        time_local = time.localtime(X[i])
        dt = time.strftime("%Y-%m-%d", time_local)
        TimeX.append(dt)
    return X, Y,TimeX







X, Y,TimeX=ExtractXAndY(txdata)

#dates = ['2018/01/01','2018/1/2', '2018/1/03', '2018/01/4','2018/01/5','2018/01/6','2018/01/07','2018/01/08','2019/01/08']
date = [datetime.strptime(d, '%Y-%m-%d').date() for d in TimeX]

# 配置横坐标

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
#设置每隔多少距离一个刻度
plt.xticks(date[::2])
plt.ylabel(u"y值")
plt.xlabel(u"时间(天)")
plt.plot(date, y,label=u"曲线")
plt.legend()
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
plt.show()