
from enum import Enum

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import time
import os 
import seaborn as sns
from scipy import stats

import sys
sys.path.append("..")
import G_Variables

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
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



###Seting according to datasets

def GenerateFeerateTimePoints(TestGroup,MaxTimeInterval):
    txinblockfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
    txdata = pd.read_csv(txinblockfile, sep=",",header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]

    dataList = txdata.values.tolist()
    sortedData=sorted(dataList,key=lambda x:x[receivetimeintx])

    sortedFrame=pd.DataFrame(sortedData)
    sortedValues=sortedFrame.values
    receivetimeInfo=sortedValues[:,receivetimeintx].tolist()
    receivetimeInfo.pop()##remove the last value
    receivetimeList=[receivetimeInfo[0]]
    receivetimeList.extend(receivetimeInfo)

    sortedFrame[sortedFrame.shape[1]]=receivetimeList
    sortedFrame[sortedFrame.shape[1]-1]=sortedFrame[receivetimeintx]-sortedFrame[sortedFrame.shape[1]-1]




    intervalInx=sortedFrame.shape[1]-1

    txdata=sortedFrame[sortedFrame[intervalInx]<=MaxTimeInterval]
    txArray=np.array(txdata)
    Points_x=txArray[:,feerateintx]
    Points_y=txArray[:,-1]



    return Points_x,Points_y










MaxTimeInterval=5
Points_x,Points_y=GenerateFeerateTimePoints(TestGroup,MaxTimeInterval)

######Define exponential distribution
def func(x, a, b, c):
    return a * np.exp(-b * x) + c



fig = plt.figure(figsize=(9,6))
plt.scatter(Points_x,Points_y,s=2)
plt.xlabel('Feerate')
plt.ylabel('Interarrival Time')
plt.title("Distribution of Transaction arrival time under feerates (feerate<1000 and intervaltime<80)")
plt.savefig("TxArrivalTimeUnderFeerates"+str(MaxTimeInterval)+".png")
plt.show()

# fig = plt.figure(figsize=(9,6))
# sns.distplot(a=Points_y,bins=MaxTime)
# plt.xlabel('Transaction Time Distribution time<'+str(MaxTime))
# plt.ylabel('Pencentage')
# plt.title("Time distrition  density ")
# plt.savefig("TimeDensity"+str(MaxTime)+".png")
# plt.show()

numOfBins=int(max(Points_y))

res_freq = stats.relfreq(Points_y, numbins=numOfBins,defaultreallimits=(min(Points_y),max(Points_y))) # numbins 是统计一次的间隔(步长)是多大
pdf_value = res_freq.frequency
cdf_value = np.cumsum(res_freq.frequency)
xData = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)

plt.figure()
plt.bar(xData, pdf_value, width=res_freq.binsize,label='original values')

#width=res_freq.binsize
###Fitting the cure for Original Data
X_noneZeroIndex = [i for i, e in enumerate(pdf_value) if e != 0]
X_Label = [xData[i] for i in X_noneZeroIndex]
Y_noneZeroValue = [pdf_value[i] for i in X_noneZeroIndex]

popt, pcov = curve_fit(func, X_Label, Y_noneZeroValue)
y2 = [func(i, popt[0], popt[1], popt[2]) for i in xData]
plt.plot(xData, y2, 'r--',label='simulated values')
plt.ylabel('pdf')
plt.xlabel('Transaction intervalarrival time')
plt.savefig('pdfOfTxInterarrval'+str(MaxTimeInterval)+'.png')
plt.legend()
plt.show()
















# plt.figure()
# plt.plot(x, cdf_value)
# plt.title('cumulative distribution function (5000-0.93) when Time under '+str(MaxTimeInterval))
# plt.savefig('cumulative distribution function when Time under '+str(MaxTimeInterval)+'.png')
# plt.show()



