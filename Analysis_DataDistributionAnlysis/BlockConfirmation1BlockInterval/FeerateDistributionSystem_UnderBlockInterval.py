# 用于粗略--查看统计系统中feerate的分布情况
# GenerateFeerateTimePoints('16',1,200) #blockinteval=1即统计confirmationtime=1block的tx信息
# GenerateFeerateTimePoints('16',-1,200) ###blockinteval=-1即统计所有tx的信息


import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



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




# import sys
# sys.path.append("..")
import G_Variables


###Seting according to datasets

def GenerateFeerateTimePoints(TestGroup,blockInterval,fee_split):
    txinblockfile ='../../'+ getattr(G_Variables, 'txfile_S' + TestGroup)
    txdata = pd.read_csv(txinblockfile, sep=",",header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]


    startfeerate=1


    if blockInterval==-1: # Blockinterval 不限制，及统计所有tx信息
        SlectedTx=txdata[(txdata[feerateintx]<=fee_split)]
        fre_feerate_Above100 = txdata[(txdata[feerateintx] > fee_split)].shape[0]
    else: #统计某个确认区间 block interval的 tx信息
        SlectedTx=txdata[(txdata[waitingblockintx]==blockInterval)&(txdata[feerateintx]<=fee_split)]
        fre_feerate_Above100 =txdata[(txdata[waitingblockintx] == blockInterval) & (txdata[feerateintx] > fee_split)].shape[0]

    feerate_Under100=SlectedTx[feerateintx].values
    interval_bin=[x for x in range(fee_split+1)]
    s_under100 = pd.cut(feerate_Under100, interval_bin)
    fre_feerate_Under100=s_under100.value_counts().values


    Overall=fre_feerate_Under100.tolist()
    Overall.append(fre_feerate_Above100)

    CDF=[sum(Overall[0:x]) for x in range(1,len(Overall)+1)]
    CDF_ratio=[x/CDF[-1] for x in CDF]


    plt.figure(figsize=(8, 4))

    points=int(fee_split)+1
    xData=range(1, points+1)
    plt.plot(xData, CDF_ratio)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))



    for samplepoint in [2,8,26,83]:
        x0 = xData[samplepoint]
        y0 = CDF_ratio[samplepoint]
        plt.scatter(x0, y0, s=10)
        plt.text(x0, y0, (x0, '%1.0f' % (100 * round(y0,2)) + '%'), color='r',fontsize=10)




    plt.ylabel('Ratio')
    plt.xlabel('Transaction feerate')
    plt.savefig('FeerateDistributionSystem_UnderBlockInterval_' +str(blockInterval)+ '.pdf')

    plt.show()

    print(startfeerate)



GenerateFeerateTimePoints('16',1,200)#blockinteval=1即统计confirmationtime=1block的tx信息
GenerateFeerateTimePoints('16',-1,200)###blockinteval=-1即统计所有tx的信息
