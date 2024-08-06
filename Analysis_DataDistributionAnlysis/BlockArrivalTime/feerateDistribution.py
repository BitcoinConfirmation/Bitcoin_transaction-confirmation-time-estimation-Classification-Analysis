

import pandas as pd

from scipy import stats
import sys
sys.path.append("..")


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


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




#############Block Features
blockHeightBinx=1
n_txBinx=2
sizeBinx=3
bitsBinx=4
feeBinx=5
verBinx=6
timeBinx=7
intervalBinx=8




 


###Seting according to datasets

def GenerateFeerateTimePoints(TestGroup,MaxTimeInterval):
    txinblockfile= '../../TimetxinBlock621500.csv'
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



Points_x,Points_y=GenerateFeerateTimePoints('16',10000000)

max_value=500
numOfBins=max_value
res_freq = stats.relfreq(Points_x, numbins=numOfBins,defaultreallimits=(min(Points_x),max_value)) # numbins 是统计一次的间隔(步长)是多大
pdf_value = res_freq.frequency
cdf_value = np.cumsum(res_freq.frequency)
xData = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)







plt.figure(figsize=(8,4))
#plt.plot(xData, pdf_value, width=res_freq.binsize,label='original values')
#plt.scatter(xData, pdf_value, s=1,label='original values')
plt.plot(xData, cdf_value)



x0 = xData[99]
y0 = cdf_value[99]

# 画出标注点
plt.scatter(x0, y0, s=50)

plt.annotate('ratio(feerate<=100)=0.969' , xy=(xData[99], cdf_value[99]), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=6,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))



plt.ylabel('pdf')
plt.xlabel('Transaction feerate (Satoshi/Weight)')
plt.savefig('FeerateDistribution' +'.pdf')

plt.show()
















# plt.figure()
# plt.plot(x, cdf_value)
# plt.title('cumulative distribution function (5000-0.93) when Time under '+str(MaxTimeInterval))
# plt.savefig('cumulative distribution function when Time under '+str(MaxTimeInterval)+'.png')
# plt.show()



