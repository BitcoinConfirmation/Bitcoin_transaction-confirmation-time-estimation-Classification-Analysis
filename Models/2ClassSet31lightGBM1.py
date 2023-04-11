#!/usr/bin/env python
# coding: utf-8
from scipy import stats
from sklearn import metrics

import pandas as pd
import numpy as np
import os
from statistics import mean
import lightgbm as lgb

# import xgboost as xgb
# from xgboost import plot_importance






import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")
import G_Variables
import os
import time
import random
sleep_time= random.randint(1,6)





###tx feature
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
waitingtimeinx= G_Variables.waitingtimeinx
confirmedtimeintx = G_Variables.confirmedtimeintx
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx
waitingblockintx=G_Variables.waitingblockintx
#Because of locktime info
validtimeintx=G_Variables.validtimeintx
validblockintx=G_Variables.validblockintx
validwaitingintx=G_Variables.validwaitingintx
#RelatedTo observation time
lastBlockIntervalintx=G_Variables.lastBlockIntervalintx## obsertime-latblocktime
waitedTimeintx=G_Variables.waitedTimeintx# obsertime-receivetime
timeToConfirmintx=G_Variables.timeToConfirmintx# confirmtime-obsertime


###block feature
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


training_blocks=G_Variables.training_blocks
lstmunits=G_Variables.lstmunits
lstmtimestamps=G_Variables.lstmtimestamps
layers=G_Variables.layers
prediction_epoch=G_Variables.prediction_epoch*3
bachsize=G_Variables.bachsize
optimizer_model=G_Variables.optimizer_model
dropout_factor=G_Variables.dropout_factor
lossfunction='mse'
target = 'FEL'
Testblocks=45

######
TestGroup=31
TestGroup=str(TestGroup)

START_EstimateBlock=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables,'total_EstimateBlock_S'+TestGroup)
result_path = getattr(G_Variables,'result_path_S'+TestGroup)


blockfile= '../FeerateVector'+getattr(G_Variables,'blockfile_S'+TestGroup)
memfile='../MemFeerateVector'+getattr(G_Variables,'blockfile_S'+TestGroup)
txfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
totalblockfile='../1000BlockTotal.csv'









weightVectorLen=100



addaccount=0

estimators = 100
time = 1
smallest_classes = 4
biggest_classes = 5
classes=2
discretization_list = ['EPI']
discretization=discretization_list[0]
cost_sensitiveList = [False]


















print(training_blocks)





TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx,lastBlockIntervalintx]

# Tx will append a dim presenting the unconfirmed weights of txs with higher feerate trasanctions in the mempool.
TxFeaLens=len(TxFeatureSelection)+1####additional 1 dim for higherfeerate weights

BocFeatureSelection=[n_txBinx,sizeBinx,bitsBinx,intervalBinx,valid_weightBinx,valid_sizeBinx]
vectorFeature=[med_waitingBinx+i+1 for i in range(weightVectorLen)]
BocFeaLens=len(BocFeatureSelection)+weightVectorLen
BocFeatureSelection.extend(vectorFeature)


MemFeatureSelection=[i+1 for i in range(weightVectorLen)]
MemFeaLens=weightVectorLen
#Block will append a vector to illuste the confirmed weignt of each feerate


PredictSelection=[waitingtimeinx]
# it represents the time interval between current time and its final confirmation block time

######BlockScaler
blockcsv = pd.read_csv(blockfile, sep=",", header=None)
blocksfortrain = blockcsv[(blockcsv[blockHeightBinx]>=START_EstimateBlock-training_blocks)&(blockcsv[blockHeightBinx]<START_EstimateBlock+total_EstimateBlock)]
blockdata = np.array(blocksfortrain)
scalerBlock = MinMaxScaler()
blockFeatures = blockdata[:, BocFeatureSelection]
sblockFeatures = scalerBlock.fit_transform(blockFeatures)


######BlockScaler
memcsv = pd.read_csv(memfile, sep=",", header=None)
blockHeightMinx=0
memsfortrain = memcsv[(memcsv[blockHeightMinx]>=START_EstimateBlock-training_blocks)&(memcsv[blockHeightMinx]<START_EstimateBlock+total_EstimateBlock)]
memdata = np.array(memsfortrain)
scalerMem = MinMaxScaler()
memFeatures = memdata[:, MemFeatureSelection]
smemFeatures = scalerMem.fit_transform(memFeatures)


def train_test_once_lightBGM(X_train, y_train, X_test, y_test, plst):
    dtrain=lgb.Dataset(X_train,y_train)
    model=lgb.train(plst,dtrain,num_boost_round=1000)
    predictions_class=model.predict(X_test)

    predictions = np.argmax(predictions_class, axis=1)
    return predictions


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




def get_mean_results(plst, X_train, y_train, X_test, y_test, time):

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

    for i in range(time):  # time = repeat time for experiments
        predictions = train_test_once_lightBGM(X_train, y_train, X_test, y_test,  plst)
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




def constructBlockSeries(blockfile,blockStartHeight,lstmtimestamps):
    blockcsv = pd.read_csv(blockfile, sep=",", header=None)
    blockdata=np.array(blockcsv)
    blockdata_selected=blockdata[np.where((blockdata[:, blockHeightBinx] >blockStartHeight-lstmtimestamps) &
                      (blockdata[:, blockHeightBinx] <blockStartHeight+1))]
    block_series=blockdata_selected[:,BocFeatureSelection]
    sblock_series=scalerBlock.transform(block_series)
    return sblock_series

def constructMemSeries(memfile,blockStartHeight,lstmtimestamps):
    blockcsv = pd.read_csv(memfile, sep=",", header=None)
    blockdata=np.array(blockcsv)
    blockHeightBinx=0
    blockdata_selected=blockdata[np.where((blockdata[:, blockHeightBinx] >blockStartHeight-lstmtimestamps) &
                      (blockdata[:, blockHeightBinx] <blockStartHeight+1))]
    block_series=blockdata_selected[:,MemFeatureSelection]
    sblock_series=scalerMem.transform(block_series)
    return sblock_series



def PossibilityDensityValue():
    txfile = '../' + getattr(G_Variables, 'txfile_S' + TestGroup)
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    txArray = np.array(txcollection)
    Points_y = txArray[:, waitingblockintx]
    res_freq = stats.relfreq(Points_y, defaultreallimits=(1, max(Points_y) + 1),
                             numbins=int(max(Points_y) + 1))
    pdf_value = res_freq.frequency
    return pdf_value



def Classfyining(classes):
    pdf_value=PossibilityDensityValue()
    eachClass = (1- pdf_value[0])/ (classes-1)
    blocksLable = [0 for _ in range(pdf_value.shape[0])]
    blocksLable[0]=0
    classLabel = 1
    # Class label Starting from 1
    sum_val = 0
    for i in range(1,pdf_value.shape[0]):
        sum_val = sum_val + pdf_value[i]
        if sum_val >eachClass+0.00001:
            blocksLable[i] = classLabel
            classLabel = classLabel + 1
            sum_val = 0
            eachClass=(1-sum(pdf_value[0:i+1]))/(classes-classLabel)
        else:
            blocksLable[i] = classLabel


    return blocksLable















def txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,startSearchBlock,endSearchBlock,classes):

  txcollection = pd.read_csv(txfile, sep=",",header=None)
  blocksInterval=Classfyining(classes)
  newcol=txcollection.shape[1]
  txcollection[newcol]=txcollection[waitingblockintx]
  txcollection[newcol]=  txcollection[newcol].apply(lambda x: blocksInterval[int(x-1)])



  txfeatureList=[]
  txOutputList=[]
  blockSeqList=[]
  memSeqList=[]

  for h_index in range(startSearchBlock,endSearchBlock):
      txsSelected = txcollection[txcollection[enterBlockintx] ==h_index]
      txsSelected = txsSelected.copy()
      txsArray = np.array(txsSelected)
      UnconfirmedTxs=txcollection[(txcollection[enterBlockintx] <=h_index)&(txcollection[blockHeightintx] >=h_index)]
      for tx in txsArray:
          feerate=tx[feerateintx]
          weight=tx[weightintx]
          higherUnconftx=np.array(UnconfirmedTxs[UnconfirmedTxs[feerateintx]>=feerate])
          higherWeights=sum(higherUnconftx[:,weightintx])-weight####Minplus selfWeight
          txfeatureay= tx[TxFeatureSelection]
          fea_list=txfeatureay.tolist()
          fea_list.append(higherWeights)
          txfeatureList.append(fea_list)



      txOutputArray = txsArray[:, -1]
      ####The last colum is class label
      txOutputList.extend(txOutputArray.tolist())
      # block_series = constructBlockSeries(blockfile, h_index-1, lstmtimestamps)
      # blockseriesList = [block_series] * txsArray.shape[0]
      # blockSeqList.extend(blockseriesList)
      #
      # mem_series = constructMemSeries(memfile, h_index-1, lstmtimestamps)
      # memseriesList = [mem_series] * txsArray.shape[0]
      # memSeqList.extend(memseriesList)
  return txfeatureList,txOutputList,np.array(blockSeqList),np.array(memSeqList)











    





CurrentBlock=START_EstimateBlock+addaccount

BlockInterval = Classfyining(classes)
RealClasses = max(BlockInterval) + 1
params = {'num_leaves': 60,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class': RealClasses,
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 15,
          'metric': 'multi_logloss',
          "random_state": 2019,
          # 'device': 'gpu'
          }

# plst = list(params.items())

plst = params
def lightGBMTime(txfile, classes, plst, time):

    print('train')
    train_txfeatureList1,train_txOutputList1,strain_blockSeqList1,strain_memSeqList1=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,CurrentBlock-training_blocks,CurrentBlock+total_EstimateBlock,classes)


    #######Scale Features
    trn_X1=np.array(train_txfeatureList1)
    #Normalization
    ss_x = MinMaxScaler()
    strain_txfeatureList1 = ss_x.fit_transform(trn_X1)
    
    train_txfeatureList,train_txOutputList,strain_blockSeqList,strain_memSeqList=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,CurrentBlock-training_blocks,CurrentBlock,classes)
    #######Scale Features
    trn_X=np.array(train_txfeatureList)
    #Normalization
    X_train = ss_x.transform(trn_X)
    y_train=np.array(train_txOutputList).reshape(-1)
    
    test_txfeatureList, test_txOutputList,stest_blockSeqList,stest_memSeqList = txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,CurrentBlock, CurrentBlock + total_EstimateBlock,classes)
    tst_X = np.array(test_txfeatureList)
    X_test = ss_x.transform(tst_X)
    y_test=np.array(test_txOutputList).reshape(-1)





    acc,prec_macro,prec_micro,prec_wht,recall_macro,recall_micro,recall_wht,f1_macro,f1_micro,f1_wht = get_mean_results(plst, X_train, y_train, X_test, y_test, time)










    df_save = pd.DataFrame([[classes, plst, time,
                             acc,prec_macro,prec_micro,prec_wht,recall_macro,recall_micro,recall_wht,f1_macro,f1_micro,f1_wht]])
    df_save.to_csv(os.path.abspath('..').replace('\\','/') + '/NeuarlResult/lightGBMResults/Set'+str(TestGroup)+'lightGBM'+str(CurrentBlock)+str(classes)+'.csv',
                   mode='w', encoding='utf-8', index=False, header=False)
    print(classes, time, acc,prec_macro,prec_micro,prec_wht,recall_macro,recall_micro,recall_wht,f1_macro,f1_micro,f1_wht)




















if __name__ == '__main__':
    
    
    for cost_sensitiveLable in cost_sensitiveList:
    #pool = mp.Pool(processes=1, maxtasksperchild=1)
        lightGBMTime(txfile,  classes, plst, time)
   
