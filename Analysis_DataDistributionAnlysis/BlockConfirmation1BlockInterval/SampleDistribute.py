
import pandas as pd

from scipy import stats
import sys
sys.path.append("..")


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


int x =1
outt x =2
vert x =3
sizeint x =4
weightint x =5
receivetimeintx = 6
relayint x =7
lockint x =8
feeint x =9
blockHeightintx = 10
blockindexint x =11
confirmedtimeintx = 12
waitingtimein x =13
feerateintx = 14
enterBlockint x =15
waitingblockint x =16
# Because of locktime info
validtimeint x =17
validblockint x =18
validwaitingint x =19
# RelatedTo observation time
lastBlockIntervalint x =2  0## obsertime-latblocktime(obseveBased)
waitedTimeint x =2  1# obsertime-receivetime
timeToConfirmint x =2  2# confirm




#############Block Features
blockHeightBin x =1
n_txBin x =2
sizeBin x =3
bitsBin x =4
feeBin x =5
verBin x =6
timeBin x =7
intervalBin x =8







###Seting according to datasets

def GenerateFeerateTimePoints(TestGroup ,MaxTimeInterval):
    txinblockfile= '../../TimetxinBlock621500.csv'
    txdata = pd.read_csv(txinblockfile, sep="," ,header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]

    dataList = txdata.values.tolist()
    sortedDat a =sorted(dataList ,key=lambda x :x[receivetimeintx])

    sortedFram e =pd.DataFrame(sortedData)
    sortedValue s =sortedFrame.values
    receivetimeInf o =sortedValues[: ,receivetimeintx].tolist()
    receivetimeInfo.pop(  )##remove the last value
    receivetimeLis t =[receivetimeInfo[0]]
    receivetimeList.extend(receivetimeInfo)

    sortedFrame[sortedFrame.shape[1] ] =receivetimeList
    sortedFrame[sortedFrame.shape[1 ] -1 ] =sortedFrame[receivetimeintx ] -sortedFrame[sortedFrame.shape[1 ] -1]




    intervalIn x =sortedFrame.shape[1 ] -1

    txdat a =sortedFrame[sortedFrame[intervalInx ]< =MaxTimeInterval]
    txArra y =np.array(txdata)
    Points_ x =txArray[: ,feerateintx]
    Points_ y =txArray[: ,-1]



    return Points_x ,Points_y



Points_x ,Points_ y =GenerateFeerateTimePoints('16' ,10000000)

max_valu e =500
numOfBin s =max_value
res_freq = stats.relfreq(Points_x, numbins=numOfBins
                         ,defaultreallimits=(min(Points_x) ,max_value)) # numbins 是统计一次的间隔(步长)是多大
pdf_value = res_freq.frequency
cdf_value = np.cumsum(res_freq.frequency)
xData = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)







plt.figure(figsize=(8 ,4))
# plt.plot(xData, pdf_value, width=res_freq.binsize,label='original values')
# plt.scatter(xData, pdf_value, s=1,label='original values')
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



