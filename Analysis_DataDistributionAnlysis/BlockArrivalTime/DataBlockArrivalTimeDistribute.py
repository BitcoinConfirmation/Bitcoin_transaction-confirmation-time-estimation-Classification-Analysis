

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




blockfile= '../../1000BlockTotal.csv'



Blockdata = pd.read_csv(blockfile, sep=",",header=None)

Blockdata[8]=Blockdata[7].shift(1)

Temp=Blockdata[7]-Blockdata[8]



Temp_Points_y=(Temp.values)[1:]

Points_y=Temp_Points_y[Temp_Points_y>0]


def func(x, a,b,c):
    return a * np.exp(-b * x)+c




def func2(x, a):
    return a * np.exp(-a * x)

# fig = plt.figure(figsize=(9,6))
# plt.scatter(Points_x,Points_y,s=2)
# plt.xlabel('Feerate')
# plt.ylabel('Interarrival Time')
# plt.title("Distribution of Transaction arrival time under feerates (feerate<1000 and intervaltime<80)")
# plt.savefig("TxArrivalTimeUnderFeerates"+str(MaxTimeInterval)+".png")
# plt.show()

# fig = plt.figure(figsize=(9,6))
# sns.distplot(a=Points_y,bins=MaxTime)
# plt.xlabel('Transaction Time Distribution time<'+str(MaxTime))
# plt.ylabel('Pencentage')
# plt.title("Time distrition  density ")
# plt.savefig("TimeDensity"+str(MaxTime)+".png")
# plt.show()
numOfBins=int(max(Points_y)/10)

res_freq = stats.relfreq(Points_y, numbins=numOfBins,defaultreallimits=(min(Points_y),max(Points_y))) # numbins 是统计一次的间隔(步长)是多大
pdf_value = res_freq.frequency
cdf_value = np.cumsum(res_freq.frequency)
xData = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)

plt.figure(figsize=(8,4))
#plt.plot(xData, pdf_value, width=res_freq.binsize,label='original values')
#plt.scatter(xData, pdf_value, s=1,label='original values')
plt.bar(xData, pdf_value, width=res_freq.binsize,label='original values')
#width=res_freq.binsize
###Fitting the cure for Original Data
X_noneZeroIndex = [i for i, e in enumerate(pdf_value) if e != 0]
X_Label = [xData[i] for i in X_noneZeroIndex]
Y_noneZeroValue = [pdf_value[i] for i in X_noneZeroIndex]

popt, pcov = curve_fit(func, X_Label, Y_noneZeroValue)
y2 = [func(i, popt[0],popt[1],popt[2]) for i in xData]

plt.plot(xData, y2, 'r--',label='y=a'+'$\mathregular{e^{-bx}}$'+'+c'+' (a=1.38e-2,b=1.52e-3,c=4.74e-3)')



# popt2, pcov2 = curve_fit(func2, X_Label, Y_noneZeroValue)
# y2_2 = [func2(i, popt2[0]) for i in xData]
#
# #plt.plot(xData, y2_2, '-', color='#FFA500',label='y=a'+'$\mathregular{e^{-ax}}$'+' (a=1.38e-2)')
#

plt.legend()
plt.ylabel('pdf')
plt.xlabel('Block mining time (seconds)')
plt.savefig('BlockMiningTime' +'.pdf')

plt.show()
















# plt.figure()
# plt.plot(x, cdf_value)
# plt.title('cumulative distribution function (5000-0.93) when Time under '+str(MaxTimeInterval))
# plt.savefig('cumulative distribution function when Time under '+str(MaxTimeInterval)+'.png')
# plt.show()



