
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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch



import sys
sys.path.append("..")
import G_Variables

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


training_blocks = G_Variables.training_blocks







TestGroup=16
TestGroup=str(TestGroup)


MaxTime=2000000






def PossibilityDensityValue():
    txfile = '../' + getattr(G_Variables, 'txfile_S' + TestGroup)
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    txArray = np.array(txcollection)
    Points_y = txArray[:, waitingblockintx]
    res_freq = stats.relfreq(Points_y, defaultreallimits=(1, max(Points_y) + 1),
                             numbins=int(max(Points_y) + 1))  # numbins 直方图使用的箱数
    pdf_value = res_freq.frequency
    return pdf_value


def Classfyining(classes):
    pdf_value = PossibilityDensityValue()
    eachClass = (1- pdf_value[0])/ (classes-1)
    blocksInterval = [0 for _ in range(pdf_value.shape[0])]
    blocksInterval[0]=1
    classLabel = 2
    # Class label Starting from 1
    sum = 0
    for i in range(1,pdf_value.shape[0]):
        sum = sum + pdf_value[i]
        if sum >= eachClass:
            blocksInterval[i]=classLabel
            classLabel = classLabel + 1
            sum = 0
        else:
            blocksInterval[i] = classLabel


    return blocksInterval



BlockInterval=Classfyining(3)













def GenerateFeerateBlockPoints(TestGroup,maxFeerate,MaxTime):
    txinblockfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
    txdata = pd.read_csv(txinblockfile, sep=",",header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]
    txcount=np.array(txdata)
    txcount=txcount.shape[0]
    print(txcount)

    txdata_All=txdata[(txdata[feerateintx]<=maxFeerate)&(txdata[waitingtimeinx]<=MaxTime)]
    txArray=np.array(txdata_All)
    Points_x=txArray[:,feerateintx]
    Points_y=txArray[:,waitingblockintx]


    temptxdata = txdata[txdata[waitingblockintx]==2]
    temptxArray = np.array(temptxdata)
    Points_z=temptxArray[:,waitingtimeinx]



    return Points_x,Points_y,Points_z




Points_x,Points_y,Points_z=GenerateFeerateBlockPoints(TestGroup,1000,MaxTime)


res_freq_pointz = stats.relfreq(Points_z,defaultreallimits=(min(Points_z),max(Points_z)),numbins=int(max(Points_z)-min(Points_z))+1) # numbins 直方图使用的箱数
print(res_freq_pointz.binsize)
pdf_value_pointz = res_freq_pointz.frequency
number_samples=Points_z.shape[0]
freq_value=pdf_value_pointz*number_samples
fig, ax = plt.subplots(1, 1,figsize=(9, 4))
X=[x for x in range(int(min(Points_z)),int(max(Points_z))+1)]
ax.scatter( X,freq_value.tolist(),s=2)
plt.xlabel('Confirmation Time (seconds)')
plt.ylabel('frequency')
plt.savefig("TimeDistributionWhen2block.pdf")
plt.show()


fig = plt.figure(figsize=(9,6))
plt.scatter(Points_x,Points_y,s=2)
plt.xlabel('Feerate')
plt.ylabel('Transaction Time (Block)')
plt.title("Confirmation Time (Block) Under Feerates")
plt.savefig("ConfirmationBlockUnderFeerates.png")
plt.show()



res_freq = stats.relfreq(Points_y,defaultreallimits=(1,max(Points_y)+1),numbins=int(max(Points_y)+1)) # numbins 直方图使用的箱数
print(res_freq.binsize)
pdf_value = res_freq.frequency
cdf_value = np.cumsum(res_freq.frequency)
x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
plt.figure()
plt.bar(x[0:60], pdf_value[0:60], width=res_freq.binsize)
plt.title('block probability density function (maxTime '+str(MaxTime)+')')
plt.savefig('Blockprobabilitydensit function (maxTime '+str(MaxTime)+').png')
plt.show()




fig, ax = plt.subplots(1, 1,figsize=(8,4))
X=[x for x in range(len(pdf_value))]
ax.plot( X,pdf_value.tolist())
plt.xlabel('Confirmation time (blocks)')
plt.ylabel('pdf')
axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.3, 0.2, 1, 1),
                   bbox_transform=ax.transAxes)
axins.plot(X, pdf_value.tolist())

x_axis_data=X
reward_demaddpg5=pdf_value
zone_left = 8
zone_right = len(pdf_value)-1
# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0  # x轴显示范围的扩展比例
y_ratio = 0.05  # y轴显示范围的扩展比例
# X轴的显示范围
xlim0 = x_axis_data[zone_left]-(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
xlim1 = x_axis_data[zone_right]+(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
# Y轴的显示范围
y = np.hstack(reward_demaddpg5[zone_left:zone_right]
               )
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
# 原图中画方框
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0-0.01
ty1 = ylim1+0.01
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"red",linestyle='--')
# 画两条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)
xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)
plt.savefig('BlockDistribution.pdf')






def DrawClassPie(pdf_value, criter_List):
    value_List = []
    label_List = []
    for i in range(len(criter_List)):
        if i == len(criter_List) - 1:
            class_ratio = sum(pdf_value[criter_List[i] - 1:])
            label_class = '>=' + str(criter_List[i])+' blocks'
        else:
            class_ratio = sum(pdf_value[criter_List[i] - 1:criter_List[i + 1] - 1])
            if criter_List[i] == criter_List[i + 1] - 1:
                if criter_List[i] == 1:
                    label_class = str(criter_List[i])+' block'
                else:
                    label_class = str(criter_List[i])+' blocks'
            else:
                label_class = str(criter_List[i]) + '~' + str(criter_List[i + 1] - 1)+' blocks'
        value_List.append(class_ratio)
        label_List.append(label_class)
    plt.figure(figsize=(7,7))
    patches,l_text,p_text=plt.pie(value_List, labels=label_List, autopct='%.1f%%',labeldistance = 1.1,startangle=90)
    # plt.title('block probability density function (maxTime ' + str(MaxTime) + ')')
    for t in p_text:

        t.set_size(12)

    for t in l_text:

        t.set_size(12)
    plt.savefig('PieBlock' + str(len(value_List)) + '.pdf',dpi=200, bbox_inches='tight')
DrawClassPie(pdf_value, [1, 2])
DrawClassPie(pdf_value, [1, 2, 3, 8])
DrawClassPie(pdf_value, [1, 2, 3, 5, 9, 29])
DrawClassPie(pdf_value, [1, 2, 3, 4, 6, 10, 19, 59])












col_name = [str(n) for n in range (1,pdf_value.shape[0]+1)]
plt.figure(figsize=(7,7))
patches,l_text,p_text=plt.pie(pdf_value,labels=col_name,autopct='%.1f%%',labeldistance = 1.1,startangle=90)
for t in p_text:
    t.set_size(12)

for t in l_text:
    t.set_size(12)
plt.title('block probability density function (maxTime '+str(MaxTime)+')')
plt.savefig('PieBlockprobabilitydensit function (maxTime '+str(MaxTime)+').pdf',dpi=200, bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(x, cdf_value)
plt.title('cumulative distribution function (maxTime '+str(MaxTime)+')')
plt.savefig('Blockcumulativedistributionfunction (maxTime '+str(MaxTime)+').png')
plt.show()
