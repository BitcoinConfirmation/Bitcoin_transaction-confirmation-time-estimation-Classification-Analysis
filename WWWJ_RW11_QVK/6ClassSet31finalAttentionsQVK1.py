'''
Created on 4 Jun. 2019

@author: limengzhang
'''
import multiprocessing

import tensorflow as tf
from tensorflow.keras.utils import plot_model,to_categorical
from scipy import stats
from TestFeeClass import IntervalConfig
from tensorflow.keras import layers,Model


from tensorflow.keras.layers import Input, Dense,Dropout,concatenate,Lambda


from keras_self_attention import SeqSelfAttention,SeqWeightedAttention
from TestFeeClass.Self_Attention import Self_Attention
from TestFeeClass.SharedFunction import NpEncoder
from TestFeeClass.SharedFunction import mean_absolute_percentage_error
from TestFeeClass.SharedFunction import calculateinterval
from TestFeeClass.SharedFunction import FindBestModelFromCheckPointByInterval
from math import sqrt
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split
import json
import time
import numpy as np


import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")
import G_Variables_WWWJ
import os
import math
import time
import random
sleep_time= random.randint(1,6)


from sklearn import metrics
from statistics import mean
import pandas as pd




###tx feature
confirmedtimeintx = G_Variables_WWWJ.confirmedtimeintx
feerateintx = G_Variables_WWWJ.feerateintx
enterBlockintx=G_Variables_WWWJ.enterBlockintx
waitingblockintx=G_Variables_WWWJ.waitingblockintx
intx=G_Variables_WWWJ.intx
outtx=G_Variables_WWWJ.outtx
vertx=G_Variables_WWWJ.vertx
sizeintx=G_Variables_WWWJ.sizeintx
weightintx=G_Variables_WWWJ.weightintx
receivetimeintx = G_Variables_WWWJ.receivetimeintx
relayintx=G_Variables_WWWJ.relayintx
lockintx=G_Variables_WWWJ.lockintx
feeintx=G_Variables_WWWJ.feeintx
blockHeightintx = G_Variables_WWWJ.blockHeightintx
waitingtimeinx= G_Variables_WWWJ.waitingtimeinx
confirmedtimeintx = G_Variables_WWWJ.confirmedtimeintx
feerateintx = G_Variables_WWWJ.feerateintx
enterBlockintx=G_Variables_WWWJ.enterBlockintx
waitingblockintx=G_Variables_WWWJ.waitingblockintx
#Because of locktime info
validtimeintx=G_Variables_WWWJ.validtimeintx
validblockintx=G_Variables_WWWJ.validblockintx
validwaitingintx=G_Variables_WWWJ.validwaitingintx
#RelatedTo observation time
lastBlockIntervalintx=G_Variables_WWWJ.lastBlockIntervalintx## obsertime-latblocktime
waitedTimeintx=G_Variables_WWWJ.waitedTimeintx# obsertime-receivetime
timeToConfirmintx=G_Variables_WWWJ.timeToConfirmintx# confirmtime-obsertime


###block feature
blockHeightBinx=G_Variables_WWWJ.blockHeightBinx
n_txBinx=G_Variables_WWWJ.n_txBinx
sizeBinx=G_Variables_WWWJ.sizeBinx
bitsBinx=G_Variables_WWWJ.bitsBinx
feeBinx=G_Variables_WWWJ.feeBinx
verBinx=G_Variables_WWWJ.verBinx
timeBinx=G_Variables_WWWJ.timeBinx
intervalBinx=G_Variables_WWWJ.intervalBinx
valid_weightBinx=G_Variables_WWWJ.valid_weightBinx
valid_sizeBinx=G_Variables_WWWJ.valid_sizeBinx
avg_feerateBinx=G_Variables_WWWJ.avg_feerateBinx
avg_waitingBinx=G_Variables_WWWJ.avg_waitingBinx
med_waitingBinx=G_Variables_WWWJ.med_waitingBinx

###mem feature
blockHeightMeminx=G_Variables_WWWJ.blockHeightMeminx



training_blocks=G_Variables_WWWJ.training_blocks
lstmunits=G_Variables_WWWJ.lstmunits
lstmtimestamps=G_Variables_WWWJ.lstmtimestamps

prediction_epoch=G_Variables_WWWJ.prediction_epoch*3
bachsize=G_Variables_WWWJ.bachsize
optimizer_model=G_Variables_WWWJ.optimizer_model
dropout_factor=G_Variables_WWWJ.dropout_factor

target = ''




#########
#ChangeFeerateInterval from 0.1 to 0.001
#FeerateIntervalLabel='Interval1000'


###Block Distribution
BLOCKSIZE=4000000 # the size (weight) of a block
BucketLen=36



FeerateIntervalLabel='ClassifiedFeerateBin'+str(BucketLen)
#########



#****Classification
clusterNumber=6
#************


TestGroup=31
TestGroup=str(TestGroup)
addaccount=0



START_EstimateBlock=getattr(G_Variables_WWWJ,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables_WWWJ,'total_EstimateBlock_S'+TestGroup)
result_path = getattr(G_Variables_WWWJ,'result_path_S'+TestGroup)
blockfile_sub=FeerateIntervalLabel+'FeerateVector'
memfile_sub=FeerateIntervalLabel+'MemFeerateVector'
blockfile= '../'+blockfile_sub+getattr(G_Variables_WWWJ,'blockfile_S'+TestGroup)
memfile='../'+memfile_sub+getattr(G_Variables_WWWJ,'blockfile_S'+TestGroup)
txfile= '../'+getattr(G_Variables_WWWJ,'txfile_S'+TestGroup)
totalblockfile='../1000BlockTotal.csv'



SelectionModule=FeerateIntervalLabel+'Feerate'+str(clusterNumber)+'ClassTime'


dir_abbv='finalAttentionsQVK'

result_path='.'+result_path
sleep_time= random.randint(1,40)
time.sleep(sleep_time)
dirs= result_path+SelectionModule+dir_abbv
if not os.path.exists(dirs):
    os.makedirs(dirs)
result_path=result_path+SelectionModule+dir_abbv+'/'








TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx,lastBlockIntervalintx]
# Tx will append a dim presenting the unconfirmed weights of txs with higher feerate trasanctions in the mempool.
TxFeaLens=len(TxFeatureSelection)+1####additional 1 dim for higherfeerate weights
#BocFeatureSelection=[n_txBinx,sizeBinx,bitsBinx,intervalBinx,valid_weightBinx,valid_sizeBinx]
BocFeaLens=BucketLen
MemFeaLens=BucketLen

#***Parameter****
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
dropout_factor=0.2
FC_units=[64,48,36,24,18,12,9]
FC_units.append(clusterNumber)
#***************



def QVKModel(FC_units,lstmunits,lstmtimestamps,blockFeatureDim,MemFeatureDim,txFeatureDim):
  np.random.seed(9)
  model2_input1 = Input(shape=(lstmtimestamps, blockFeatureDim), name='model2_input1')
  self_attention_out1 = Self_Attention(lstmunits)(model2_input1)
  self_attention_out1 =Lambda(lambda x:x[:,lstmtimestamps-1,:])(self_attention_out1)

  model2_input2 = Input(shape=(lstmtimestamps, MemFeatureDim), name='model2_input2')
  self_attention_out2 = Self_Attention(lstmunits)(model2_input2)
  self_attention_out2 =Lambda(lambda x:x[:,lstmtimestamps-1,:])(self_attention_out2)

  model2_auxiliary_input = Input(shape=(txFeatureDim,), name='model2_tx_input')
  model2_merged_vector = concatenate([self_attention_out1, self_attention_out2,model2_auxiliary_input], axis=-1)
  model2_merged_vector = Dropout(dropout_factor)(model2_merged_vector)

  layer1_vector = model2_merged_vector
  for lay_inx in range(len(FC_units) - 1):
      layer1_vector = Dense(FC_units[lay_inx], kernel_initializer='uniform', activation='relu',
                                   name='Dense_FC_' + str(lay_inx + 1))(layer1_vector)
      layer1_vector = Dropout(dropout_factor, name='Drop_FC_' + str(lay_inx + 1))(layer1_vector)

  model2_predictions = Dense(FC_units[-1], activation='softmax', name='Dense_FC_' + str(len(FC_units)))(
      layer1_vector)
  model2 = Model(inputs=[model2_input1, model2_input2, model2_auxiliary_input], outputs=model2_predictions)
  model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
  return model2

def constructBlockDistriSeries(blockfile,blockStartHeight,lstmtimestamps,FeaDistrDim):
 # Features only contains transaction distribution in the block.
  blockcsv = pd.read_csv(blockfile, sep=",", header=None)
  blockdata = np.array(blockcsv)
  blockdata_selected=blockdata[np.where((blockdata[:, blockHeightBinx] >= blockStartHeight) &
                      (blockdata[:, blockHeightBinx] <blockStartHeight+lstmtimestamps))]

  blockdata_series=blockdata_selected[:,-FeaDistrDim:]/BLOCKSIZE

  return blockdata_series
def constructMemDistriSeries(memfile, blockStartHeight, lstmtimestamps,FeaDistrDim):
    # Features only contains transaction distribution in the block.
    memcsv = pd.read_csv(memfile, sep=",", header=None)
    memdata=np.array(memcsv)

    memdata_selected = memdata[np.where((memdata[:, blockHeightMeminx] >= blockStartHeight) &(memdata[:, blockHeightMeminx] < blockStartHeight + lstmtimestamps))]
    memdata_series= memdata_selected[:,-FeaDistrDim:] / BLOCKSIZE

    return memdata_series

def PossibilityDensityValue():
    txfile = '../' + getattr(G_Variables_WWWJ, 'txfile_S' + TestGroup)
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
def txDatasetConstruction(txfile, blockfile, memfile, lstmtimestamps, startSearchBlock, endSearchBlock, clusterNumber):
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    blocksInterval = Classfyining(clusterNumber)
    newcol = txcollection.shape[1]
    txcollection[newcol] = txcollection[waitingblockintx]
    txcollection[newcol] = txcollection[newcol].apply(lambda x: blocksInterval[int(x - 1)])

    txfeatureList = []
    txOutputList = []
    blockSeqList = []
    memSeqList = []

    for h_index in range(startSearchBlock, endSearchBlock):
        txsSelected = txcollection[txcollection[enterBlockintx] == h_index]
        txsSelected = txsSelected.copy()
        txsArray = np.array(txsSelected)
        UnconfirmedTxs = txcollection[
            (txcollection[enterBlockintx] <= h_index) & (txcollection[blockHeightintx] >= h_index)]
        for tx in txsArray:
            feerate = tx[feerateintx]
            weight = tx[weightintx]
            higherUnconftx = np.array(UnconfirmedTxs[UnconfirmedTxs[feerateintx] >= feerate])
            higherWeights = sum(higherUnconftx[:, weightintx])
            virtualBlockPos=higherWeights/BLOCKSIZE
            txfeatureay = tx[TxFeatureSelection]
            fea_list = txfeatureay.tolist()
            fea_list.append(virtualBlockPos)
            txfeatureList.append(fea_list)

        txOutputArray = txsArray[:, -1]####The last colum is class label
        txOutputList.extend(txOutputArray.tolist())

        block_series = constructBlockDistriSeries(blockfile, h_index - 1, lstmtimestamps,BucketLen)
        blockseriesList = [block_series] * txsArray.shape[0]
        blockSeqList.extend(blockseriesList)

        mem_series = constructMemDistriSeries(memfile, h_index - 1, lstmtimestamps,BucketLen)
        memseriesList = [mem_series] * txsArray.shape[0]
        memSeqList.extend(memseriesList)
    return txfeatureList, txOutputList, np.array(blockSeqList), np.array(memSeqList)



####Construct Training and Testing dataset
train_txfeatureList1,train_txOutputList1,strain_blockSeqList1,strain_memSeqList1=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,START_EstimateBlock-training_blocks,START_EstimateBlock+total_EstimateBlock,clusterNumber)
#######Scale Features
trn_X1=np.array(train_txfeatureList1)
trn_Y1 = np.array(train_txOutputList1)
#Normalization
ss_x = MinMaxScaler()
strain_txfeatureList1 = ss_x.fit_transform(trn_X1)

lastUpdateHeight=START_EstimateBlock+addaccount
train_txfeatureList,train_txOutputList,strain_blockSeqList,strain_memSeqList=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,lastUpdateHeight-training_blocks,lastUpdateHeight,clusterNumber)
#######Scale Features
trn_X=np.array(train_txfeatureList)
trn_Y=np.array(train_txOutputList)

#Normalization And Transfromation
strain_txfeatureList = ss_x.transform(trn_X)
strain_txOutputList = to_categorical(trn_Y,num_classes=clusterNumber) #convert integers to dummy variables (one hot encoding)


fel_model = QVKModel(FC_units, lstmunits, lstmtimestamps, BocFeaLens,MemFeaLens, TxFeaLens)
plot_model(fel_model, to_file='Class'+str(clusterNumber)+dir_abbv+'.png', show_shapes=True)



filepath_fel = result_path + SelectionModule +str(lastUpdateHeight)+str(clusterNumber)+'.h5'



checkpoint = ModelCheckpoint(
  filepath=filepath_fel,
  monitor='accuracy',
  save_best_only=True,
  verbose=0,
  mode='auto',
  save_weights_only=False,
  period=1)

fel_hist = fel_model.fit([strain_blockSeqList,strain_memSeqList, strain_txfeatureList], strain_txOutputList, epochs=prediction_epoch,
                         batch_size=bachsize, verbose=1,callbacks=[checkpoint])


#***Testing dataset

lastUpdateHeight = START_EstimateBlock + addaccount
test_txfeatureList, test_txOutputList, stest_blockSeqList, stest_memSeqList=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,lastUpdateHeight,lastUpdateHeight+total_EstimateBlock,clusterNumber)
#######Scale Features
tst_X = np.array(test_txfeatureList)
tst_Y = np.array(test_txOutputList)
#Normalization And Transfromation
stest_txfeatureList = ss_x.transform(tst_X)
stest_txOutputList = to_categorical(tst_Y, num_classes=clusterNumber)

#***Test Performance
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
        #predictions = train_test_once_xgBoost(X_train, y_train, X_test, y_test,  plst)
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

Y_hat_original=fel_model.predict([stest_blockSeqList,stest_memSeqList, stest_txfeatureList])
YPred=np.array(np.argmax(Y_hat_original, axis=-1)).reshape(-1)
YTrue=np.array(np.argmax(stest_txOutputList, axis=-1)).reshape(-1)

acc, prec_macro, prec_micro, prec_wht, recall_macro, recall_micro, recall_wht, f1_macro, f1_micro, f1_wht = get_mean_results(
    YTrue, YPred)
df_save = pd.DataFrame([[clusterNumber,
                         acc, prec_macro, prec_micro, prec_wht, recall_macro, recall_micro, recall_wht, f1_macro,
                         f1_micro, f1_wht]])
df_save.to_csv( '../WWWJ_OverallPerformance/'+ SelectionModule + TestGroup + dir_abbv + str(
    START_EstimateBlock + addaccount) + str(clusterNumber) + 'results.csv',
               mode='w', encoding='utf-8', index=False, header=False)






