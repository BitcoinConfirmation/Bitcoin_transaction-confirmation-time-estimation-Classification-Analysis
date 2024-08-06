from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as ticker
import matplotlib.dates as mdate
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter

# MemCount='mempool-count.csv'
# TxBlockCount='transactionperblock.csv'
#
#
# txdata = pd.read_csv(TxBlockCount, sep=",")
# MemCount=pd.read_csv(MemCount, sep=",")


Txdata=pd.read_csv('Txdata.csv', sep=",")
coloList=['#3B4992','#DE8F44','#01AAD5','#B34745']


DataInfo=Txdata.values
MemTx=DataInfo[:,0]
AveBlock=DataInfo[:,1]

plt.figure(figsize=(10,4))
print('hello')
x_major_locator=MultipleLocator(240)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlabel('Time')
plt.ylabel('Transaction count')
plt.plot(MemTx,linewidth=1,label='Average transactions in the mempool (mean='+str(int(mean(MemTx)))+ ')' )
plt.plot(AveBlock,linestyle=':',linewidth=1.5, label='Average transactions per block (mean='+str(int(mean(AveBlock)))+ ')' )

plt.legend()
plt.savefig("AvetTxperMem.pdf")
plt.show()



TxPerBlock=pd.read_csv('TransactionsPerBlock.csv', sep=",")
DataInfo=TxPerBlock.values
BlockTx=DataInfo[:,1]
plt.figure(figsize=(10,4))
print('hello')
x_major_locator=MultipleLocator(450)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlabel('Time')
plt.ylabel('Average transactions per block')
plt.plot(BlockTx,linewidth=1)
plt.savefig("AvetTxperblock.pdf")
plt.show()


#
#
#
#
#
# plt.figure(figsize=(10,4))
# MemX, MemY, MemTimeX=ExtractXAndY(MemCount)
# x_major_locator=MultipleLocator(250)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
#
# plt.xlabel('Time')
# plt.ylabel('Average transactions in the Mempool')
# plt.plot(MemY)
# plt.savefig("AvetTxMem.pdf")
#
#
# # dates = ['2018/01/01','2018/1/2', '2018/1/03', '2018/01/4','2018/01/5','2018/01/6','2018/01/07','2018/01/08']
# # date = [datetime.strptime(d, '%Y/%m/%d').date() for d in dates]
# # y=[25,18,13,14,12,17,16,15]
# # # 配置横坐标
# # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
# # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# # #设置每隔多少距离一个刻度
# # plt.xticks(date[::2])
# # plt.ylabel("y")
# # plt.xlabel("day")
# # plt.plot(date, y,label="scre")
# # plt.legend()
# # plt.gcf().autofmt_xdate()  # 自动旋转日期标记
# # plt.show()
# #
#
