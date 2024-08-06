
from enum import Enum

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import time
import os 


import sys
sys.path.append("..")
import G_Variables

import matplotlib.pyplot as plt
import joblib

intx=1
outtx=2
vertx=3
sizeintx=4
weightintx=5
receivetimeintx = 6
relayintx=7
lockintx=8
feeintx=9
blockHeightintx = 10
blockindexintx=11
confirmedtimeintx = 12
waitingtimeinx=13
feerateintx = 14
enterBlockintx=15
waitingblockintx=16
#Because of locktime info
validtimeintx=17
validblockintx=18
validwaitingintx=19
#RelatedTo observation time
lastBlockIntervalintx=20## obsertime-latblocktime(obseveBased)
waitedTimeintx=21# obsertime-receivetime
timeToConfirmintx=22# confirm






###tx feature
intx=G_Variables.intx
outtx=G_Variables.outtx
vertx=G_Variables.vertx
sizeintx=G_Variables.sizeintx
weightintx=G_Variables.weightintx
receivetimeintx = G_Variables.receivetimeintx
relayintx=G_Variables.relayintx
lockintx=G_Variables.lockintx
feeintx=G_Variables.feeintx
blockHeightintx = G_Variables.blockHeightintx
confirmedtimeintx = G_Variables.confirmedtimeintx
waitingtimeinx= G_Variables.waitingtimeinx
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx
waitingblockintx=G_Variables.waitingblockintx





 



SHORT_BLOCK_PERIODS = 12
SHORT_SCALE = 1
MED_BLOCK_PERIODS = 24
MED_SCALE = 2
LONG_BLOCK_PERIODS = 1000
LONG_SCALE = 1
OLDEST_ESTIMATE_HISTORY = 6 * 1008


SHORT_DECAY = .962
MED_DECAY = .9952

# LONG_DECAY=SHORT_DECAY
#LONG_DECAY = .99931
LONG_DECAY=0.962
# Require greater than 60% of X feerate transactions to be confirmed within Y/2 blocks*/
HALF_SUCCESS_PCT = .6
#Require greater than 85% of X feerate transactions to be confirmed within Y blocks*/
SUCCESS_PCT = .85
# Require greater than 95% of X feerate transactions to be confirmed within 2 * Y blocks*/
DOUBLE_SUCCESS_PCT = .95

# Require an avg of 0.1 tx in the combined feerate bucket per block to have stat significance */
SUFFICIENT_FEETXS = 0.1
# Require an avg of 0.5 tx when using short decay since there are fewer blocks considered*/
SUFFICIENT_TXS_SHORT = 0.5




MIN_BUCKET_TIME=100
MAX_BUCKET_TIME=600*50
FEE_SPACING = 1.05

training_blocks = G_Variables.training_blocks

TestGroup=16
TestGroup=str(TestGroup)



def MeanConfirmedFee(TestGroup):
    txinblockfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
    temp_count=[0 for x in range(1001)]
    temp_weight=[0 for x in range(1001)]
    txdata = pd.read_csv(txinblockfile, sep=",",header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]
    txcount=np.array(txdata)
    txcount=txcount.shape[0]
    print(txcount)

    txdata=txdata[txdata[feerateintx]>0]
    for i in range(1000):
        r=i+1
        collected_txs=np.array(txdata[(r-1<=txdata[feerateintx]) & (txdata[feerateintx]<r)])
        temp_count[i] = collected_txs.shape[0]
        temp_weight[i]=sum(collected_txs[:,5])
    collected_txs = np.array(txdata[txdata[feerateintx] >=1000])
    temp_count[1000] = collected_txs.shape[0]
    temp_weight[1000] = sum(collected_txs[:, 5])


    return np.array(temp_count),np.array(temp_weight)


temp_count1,temp_weight1=MeanConfirmedFee('16')
print('hello')
temp_count2,temp_weight2=MeanConfirmedFee('17')
print('hello')
temp_count3,temp_weight3=MeanConfirmedFee('18')
print('hello')

overall_count=temp_count1+temp_count2+temp_count3
overall_weight=temp_weight1+temp_weight2+temp_weight3


# 保存x
joblib.dump(overall_count, 'overall_count.pkl')
joblib.dump(overall_weight, 'weight_count.pkl')




scaleIndex=[10,40,160,640,641]


fig = plt.figure(figsize=(9,6))
plt.plot(overall_count)
plt.xlabel('Feerate')
plt.ylabel('Transaction Count')
plt.savefig("FigureForcount.eps")

ListCount=[]
ListCount.append(sum(overall_count[0:10]))
ListCount.append(sum(overall_count[10:40]))
ListCount.append(sum(overall_count[40:160]))
ListCount.append(sum(overall_count[160:640]))
ListCount.append(sum(overall_count[640:]))

joblib.dump(ListCount, 'Pie_tx_count.pkl')

labels =['<10','10~40','40~160','160~640','640+']
X = ListCount
fig = plt.figure(figsize=(9,6))
plt.pie(X, labels=labels, autopct='%1.2f%%')
plt.title("")
#plt.show()
plt.savefig("PieChartForcount.eps")






fig = plt.figure(figsize=(9,6))
plt.plot(overall_weight)
plt.xlabel('Feerate')
plt.ylabel('Transaction Weight')
plt.savefig("FigureForweight.eps")

Listweight=[]
Listweight.append(sum(overall_weight[0:10]))
Listweight.append(sum(overall_weight[10:40]))
Listweight.append(sum(overall_weight[40:160]))
Listweight.append(sum(overall_weight[160:640]))
Listweight.append(sum(overall_weight[640:]))

joblib.dump(Listweight, 'Pie_weight_count.pkl')
labels =['<10','10~40','40~160','160~640','640+']
X = Listweight
fig = plt.figure(figsize=(9,6))
plt.pie(X, labels=labels, autopct='%1.2f%%')
plt.title("")
#plt.show()
plt.savefig("PieChartForweight.eps")

