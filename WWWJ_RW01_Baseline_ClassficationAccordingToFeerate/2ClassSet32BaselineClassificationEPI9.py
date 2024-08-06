'''
Created on 4 Jun. 2019

@author: limengzhang
'''
import os
from statistics import mean

import numpy as np

import pandas as pd
from sklearn import metrics
from scipy import stats


import sys
import joblib

sys.path.append("..")
import G_Variables



confirmedtimeintx = G_Variables.confirmedtimeintx
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx


waitingblockintx=G_Variables.waitingblockintx
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
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx
waitingblockintx=G_Variables.waitingblockintx
#Because of locktime info
validtimeintx=G_Variables.validtimeintx
validblockintx=G_Variables.validblockintx
validwaitingintx=G_Variables.validwaitingintx

lastBlockIntervalintx=G_Variables.lastBlockIntervalintx









blockHeightBinx=G_Variables.blockHeightBinx
n_txBinx=G_Variables.n_txBinx
sizeBinx=G_Variables.sizeBinx
bitsBinx=G_Variables.bitsBinx
feeBinx=G_Variables.feeBinx
verBinx=G_Variables.verBinx
timeBinx=G_Variables.timeBinx
intervalBinx=G_Variables.intervalBinx
valid_weightBinx=G_Variables.valid_weightBinx
valid_sizeBinx=G_Variables.valid_sizeBinx
avg_feerateBinx=G_Variables.avg_feerateBinx
avg_waitingBinx=G_Variables.avg_waitingBinx
med_waitingBinx=G_Variables.med_waitingBinx



##########Block new feature
# intervalBinx: interval since last block
#valid_weightBinx: sum of tx weight in block
#valid_sizeBinx: sum of tx weight in block
# avg_feerateBinx= 4*overallfee/valid_weightBinx
# avg_waitingtime: average confirmation time for txs in a block
# med_waitingtime: median confirmation time for txs in a block
totalBlockfile='1000BlockTotal.csv'




##Evidence
##According to feerate distiribution
#### We find that only 0.02% has feerate>=640


feerate_Interval = 0.001

numBins = int(100/feerate_Interval)+1
### Set a flexibile fee limit
##622500>100 10%
MempoolClassification=2
blocks_overall=499
TestGroup=32
TestGroup=str(TestGroup)
addaccount=8
START_EstimateBlock=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables,'total_EstimateBlock_S'+TestGroup)










def MeanConfirmedFee(testgroup):
    txfile = '../' + getattr(G_Variables, 'txfile_S' + str(testgroup))

    temp_count=[0 for x in range(numBins)]
    temp_weight=[0 for x in range(numBins)]
    txdata = pd.read_csv(txfile, sep=",",header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]
    txcount=np.array(txdata)
    txcount=txcount.shape[0]
    print(txcount)

    txdata=txdata[txdata[feerateintx]>0]
    for i in range(numBins-1):
        r=i+1
        # Statistics on Each  feerate interval=0.1
        #feerateInterval=0.1
        collected_txs=np.array(txdata[((r-1)*feerate_Interval<=txdata[feerateintx]) & (txdata[feerateintx]<r*feerate_Interval)])
        temp_count[i] = collected_txs.shape[0]
        temp_weight[i]=sum(collected_txs[:,5])

    collected_txs = np.array(txdata[txdata[feerateintx] >=(numBins-1)*feerate_Interval])
    temp_count[numBins-1] = collected_txs.shape[0]
    temp_weight[numBins-1] = sum(collected_txs[:, 5])
    oveallNum=sum(temp_count)
    count_freq= np.divide(temp_count, oveallNum)
    return np.array(temp_count),np.array(count_freq)













def getHeightTime(blockfile,blockheight):
    blockdata = pd.read_csv(blockfile, sep=",",header=None)
    block_array=np.array(blockdata)
    heighttime = -1
    for i in range(block_array.shape[0]):
        if block_array[i][blockHeightBinx]==blockheight:
            heighttime=block_array[i][timeBinx]
            break
    return heighttime



def PossibilityDensityValue(testgroup):
    txfile = '../' + getattr(G_Variables, 'txfile_S' + str(testgroup))
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    txArray = np.array(txcollection)
    Points_y = txArray[:, waitingblockintx]
    res_freq = stats.relfreq(Points_y, defaultreallimits=(1, max(Points_y) + 1),
                             numbins=int(max(Points_y) + 1))  #
    pdf_value = res_freq.frequency
    return pdf_value


def Classfyining(classes,testgroup):
    pdf_value = PossibilityDensityValue(testgroup)
    eachClass = (1- pdf_value[0])/ (classes-1)
    blocksInterval = [0 for _ in range(pdf_value.shape[0])]
    blocksInterval[0]=0
    classLabel = 1
    # Class label Starting from 1
    sum_val = 0
    for i in range(1,pdf_value.shape[0]):
        sum_val = sum_val + pdf_value[i]
        if sum_val >eachClass+0.00001:
            blocksInterval[i] = classLabel
            classLabel = classLabel + 1
            sum_val = 0
            eachClass=(1-sum(pdf_value[0:i+1]))/(classes-classLabel)
        else:
            blocksInterval[i] = classLabel


    return pdf_value,blocksInterval





def ClassfyiningTxAccordingtoBlock(pdf_value, blocksInterval, tx_freq):
    #Approximation with the ratio sum
    clusters=max(blocksInterval)+1
    Cluster_ratio=[]
    txCluster_ratio=[]
    Feerate_Classifier = []
    for i in range(clusters):
        b_index_start=blocksInterval.index(i)
        if i<clusters-1:
            b_index_end = blocksInterval.index(i + 1)
        else:
            b_index_end = len(blocksInterval)-1
        Cluster_ratio.append(sum(pdf_value[b_index_start:b_index_end]))
#########Get the ratio for each block interval

    end_txIndex=len(tx_freq)-1
    sum_agg_block=0
    for i in range(clusters):
        cluster_ratio=Cluster_ratio[i]
        sum_agg_block = sum_agg_block+cluster_ratio

        for q in range(0,len(tx_freq)):
            if sum(tx_freq[q:])<=sum_agg_block:
                feerate_divi=(q+1)*feerate_Interval
                Feerate_Classifier.append(feerate_divi)
                txCluster_ratio.append(sum(tx_freq[q:])-sum(txCluster_ratio))
                print(' Index='+str(q))
                break

    return Feerate_Classifier,Cluster_ratio,txCluster_ratio


#for start_selection in [621500,622000,622500]:
def CalculateArrayDis(txs_temp, feerate_Classfier):
    dis_vector=[]
    classNum=len(feerate_Classfier)
    for i in range(classNum):
        if i==classNum-1:
            lowerbound=0
        else:
            lowerbound=feerate_Classfier[i]
        if i==0:
            upperbound=100000000
        else:
            upperbound = feerate_Classfier[i-1]
        slected_tx=txs_temp[np.where((txs_temp[:, feerateintx] < upperbound)&(txs_temp[:, feerateintx] >= lowerbound))]
        if slected_tx.shape[0]==0:
            dis_vector.append(0)
        else:
            dis_vector.append(sum(slected_tx[:,feerateintx]))
    return dis_vector


def FindFeerateClass(target, feerateClassfierList):
    lens = len(feerateClassfierList)
    if target <= feerateClassfierList[lens - 1]:
        return lens - 1
    elif target >= feerateClassfierList[0]:
        return 0
    else:
        for i in range(1, lens):
            upperBound = feerateClassfierList[i - 1]
            currentLimit = feerateClassfierList[i]
            if currentLimit <= target < upperBound:
                break

        return i







def txDatasetConstruction(testgroup,startSearchBlock,endSearchBlock,blocksInterval,feerate_Classfier):
    txfile = '../' + getattr(G_Variables, 'txfile_S' + str(testgroup))
    txcollection = pd.read_csv(txfile, sep=",",header=None)
    #pdf_value,blocksInterval=Classfyining(MempoolClassification,TestGroup)
    newcol=txcollection.shape[1]
    txcollection[newcol]=txcollection[waitingblockintx]
    txcollection[newcol]=  txcollection[newcol].apply(lambda x: blocksInterval[int(x-1)])

    Y_Pred=[]
    Y_True=[]



    for h_index in range(startSearchBlock,endSearchBlock):
      txsSelected = txcollection[txcollection[enterBlockintx] ==h_index]
      txsSelected = txsSelected.copy()
      txsArray = np.array(txsSelected)
      for tx in txsArray:
          feerate_tx=tx[feerateintx]
          y_pred=FindFeerateClass(feerate_tx,feerate_Classfier)
          Y_Pred.append(y_pred)



      txOutputArray = txsArray[:, -1]
      ####The last colum is class label
      Y_True.extend(txOutputArray.tolist())

    return np.array(Y_True).reshape(-1),np.array(Y_Pred).reshape(-1)



def get_metrics(y_test, predictions):
    acc=metrics.accuracy_score(y_test,predictions)
    prec_macro=metrics.precision_score(y_test,predictions,average='macro')
    prec_micro = metrics.precision_score(y_test, predictions, average='micro')
    prec_wht=metrics.precision_score(y_test, predictions, average='weighted')
    recall_macro=metrics.recall_score(y_test,predictions,average='macro')
    recall_micro = metrics.recall_score(y_test, predictions, average='micro')
    recall_wht = metrics.recall_score(y_test, predictions, average='weighted')
    f1_macro=metrics.f1_score(y_test,predictions,average='macro')
    f1_micro = metrics.f1_score(y_test, predictions, average='micro')
    f1_wht = metrics.f1_score(y_test, predictions, average='weighted')
    return acc,prec_macro,prec_micro,prec_wht,recall_macro,recall_micro,recall_wht,f1_macro,f1_micro,f1_wht


def get_mean_results(y_test,predictions):

    acc_list = []
    prec_mac_list = []
    prec_mic_list=[]
    prec_wht_list=[]
    rec_mac_list=[]
    rec_mic_list = []
    rec_wht_list = []
    f1_mac_list = []
    f1_mic_list = []
    f1_wht_list = []

    for i in range(1):  # time = repeat time for experiments
        #predictions = train_test_once_lightBGM(X_train, y_train, X_test, y_test,  plst)
        acc,prec_mac,prec_mic,prec_wht,rec_mac,rec_mic,rec_wht,f1_mac,f1_mic,f1_wht = get_metrics(y_test, predictions)

        acc_list.append(acc)
        prec_mac_list.append(prec_mac)
        prec_mic_list.append(prec_mic)
        prec_wht_list.append(prec_wht)

        rec_mac_list.append(rec_mac)
        rec_mic_list.append(rec_mic)
        rec_wht_list.append(rec_wht)


        f1_mac_list.append(f1_mac)
        f1_mic_list.append(f1_mic)
        f1_wht_list.append(f1_wht)


    acc = mean(acc_list)

    prec_mac = mean(prec_mac_list)
    prec_mic = mean(prec_mic_list)
    prec_wht = mean(prec_wht_list)

    rec_mac = mean(rec_mac_list)
    rec_mic = mean(rec_mic_list)
    rec_wht = mean(rec_wht_list)

    f1_mac=mean(f1_mac_list)
    f1_mic=mean(f1_mic_list)
    f1_wht=mean(f1_wht_list)


    return acc,prec_mac,prec_mic,prec_wht,rec_mac,rec_mic,rec_wht,f1_mac,f1_mic,f1_wht









print(TestGroup)
pdf_value,blocksInterval=Classfyining(MempoolClassification,TestGroup)
print(max(blocksInterval)+1)
print(MempoolClassification)
tx_count,tx_freq=MeanConfirmedFee(TestGroup)
feerate_Classfier,BlockingCluster_ratio,txCluster_ratio=ClassfyiningTxAccordingtoBlock(pdf_value,blocksInterval,tx_freq)




CurrentBlock=START_EstimateBlock+addaccount
Y_True,Y_Pred=txDatasetConstruction(TestGroup,CurrentBlock,CurrentBlock + total_EstimateBlock,blocksInterval,feerate_Classfier)

print('TestingSet is '+str(Y_True.shape[0]))
acc, prec_macro, prec_micro, prec_wht, recall_macro, recall_micro, recall_wht, f1_macro, f1_micro, f1_wht = get_mean_results(
    Y_True,Y_Pred)

df_save = pd.DataFrame([[MempoolClassification,
                         acc, prec_macro, prec_micro, prec_wht, recall_macro, recall_micro, recall_wht, f1_macro,
                         f1_micro, f1_wht]])
df_save.to_csv(
    os.path.abspath('..').replace('\\', '/') + '/NeuarlResult/BaselineClassificationResults/'  + 'Set' + str(
        TestGroup) + 'BaselineClassification'  + str(CurrentBlock) + str(MempoolClassification) + '.csv',
    mode='w', encoding='utf-8', index=False, header=False)
print(MempoolClassification, acc, prec_macro, prec_micro, prec_wht, recall_macro, recall_micro, recall_wht, f1_macro, f1_micro,
      f1_wht)






