'''
Created on 4 Jun. 2019

@author: limengzhang
'''
import multiprocessing

from keras.utils import np_utils
from scipy import stats
from TestFeeClass import IntervalConfig
from keras.utils.vis_utils import plot_model
import keras


from keras.layers import Input, Embedding, LSTM,GRU, Dense,Dropout,Bidirectional
from keras.models import Model
from keras.layers.core import Flatten
from keras.models import load_model
from keras_self_attention import SeqSelfAttention,SeqWeightedAttention
from TestFeeClass.Self_Attention import Self_Attention
from TestFeeClass.SharedFunction import NpEncoder
from TestFeeClass.SharedFunction import getLastBlockTime,getHeightTime
from TestFeeClass.SharedFunction import mean_absolute_percentage_error
from TestFeeClass.SharedFunction import calculateinterval
from TestFeeClass.SharedFunction import FindBestModelFromCheckPointByInterval
from math import sqrt
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

from keras.layers import Dense, Lambda, dot, Activation, concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import time
import numpy as np


import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")
import G_Variables
import os
import math
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

######Scanning timeinterval(s)
train_timeinterval=7*60
test_timeinterval=60
######
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

#

#########
#ChangeFeerateInterval from 0.1 to 0.001
FeerateIntervalLabel='Interval1000'
#########



######
TestGroup=31
TestGroup=str(TestGroup)
discretization_list = [ 'EPI']
classes_list = [2]
smallest_classes = 4
biggest_classes = 5

START_EstimateBlock=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables,'total_EstimateBlock_S'+TestGroup)
result_path = getattr(G_Variables,'result_path_S'+TestGroup)

blockfile_sub=FeerateIntervalLabel+str(classes_list[0])+'FeerateVector'
memfile_sub=FeerateIntervalLabel+str(classes_list[0])+'MemFeerateVector'

blockfile= '../'+blockfile_sub+getattr(G_Variables,'blockfile_S'+TestGroup)
memfile='../'+memfile_sub+getattr(G_Variables,'blockfile_S'+TestGroup)
txfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
totalblockfile='../1000BlockTotal.csv'





SelectionModule=FeerateIntervalLabel+'Feerate'+str(classes_list[0])+'ClassTime'
dir_abbv='finalAttentionsWht'
result_path='.'+result_path
sleep_time= random.randint(1,40)
time.sleep(sleep_time)
dirs= result_path+SelectionModule+dir_abbv
if not os.path.exists(dirs):
    os.makedirs(dirs)
result_path=result_path+SelectionModule+dir_abbv+'/'

weightVectorLen=classes_list[0]

addaccount=0









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






def selfAttentionmempoolModel(layers,lstmunits,lstmtimestamps,blockFeatureDim,MemFeatureDim,txFeatureDim,classes):
  np.random.seed(9)

  model2_input1 = Input(shape=(lstmtimestamps, blockFeatureDim), name='model2_input1')
  self_attention_out1 = SeqWeightedAttention()(model2_input1)


  model2_input2 = Input(shape=(lstmtimestamps, MemFeatureDim), name='model2_input2')
  self_attention_out2 =SeqWeightedAttention()(model2_input2)




  model2_auxiliary_input = Input(shape=(txFeatureDim,), name='model2_tx_input')
  model2_merged_vector = keras.layers.concatenate([self_attention_out1,self_attention_out2,model2_auxiliary_input], axis=-1)
  model2_merged_vector = Dropout(dropout_factor)(model2_merged_vector)
  model2_layer1_vector = Dense(layers[0],kernel_initializer='uniform', activation='relu')(model2_merged_vector)
  model2_layer1_vector=Dropout(dropout_factor)(model2_layer1_vector)
  model2_layer2_vector = Dense(layers[1], activation='relu')(model2_layer1_vector)
  model2_layer2_vector=Dropout(dropout_factor)(model2_layer2_vector)
  model2_predictions = Dense(classes, activation='softmax')(model2_layer2_vector)
  model2 = Model(inputs=[model2_input1,model2_input2, model2_auxiliary_input], outputs=model2_predictions)
  model2.compile(loss='categorical_crossentropy', optimizer=optimizer_model,metrics=['accuracy'])
  return model2




























def getHeightTime(totalblockfile,startEsitimateBlock):
    totalblock = pd.read_csv(totalblockfile, sep=",", header=None)
    block=totalblock[totalblock[blockHeightBinx]==startEsitimateBlock]
    HeightTime=block[timeBinx]
    return HeightTime.values[0]


def getLastBlockTime(totalblockfile,obserTime):
    totalblock = pd.read_csv(totalblockfile, sep=",", header=None)
    totalblock_array=np.array(totalblock)
    for blc in range(totalblock_array.shape[0]):
        if totalblock_array[blc][timeBinx]<obserTime and totalblock_array[blc+1][timeBinx]>=obserTime:
            break
    lastBlockHeight=totalblock_array[blc][blockHeightBinx]
    lastBlocktime=totalblock_array[blc][timeBinx]
    return lastBlockHeight,lastBlocktime











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





def txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,startSearchBlock,endSearchBlock,classes, discretization, smallest_classes):
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
      block_series = constructBlockSeries(blockfile, h_index-1, lstmtimestamps)
      blockseriesList = [block_series] * txsArray.shape[0]
      blockSeqList.extend(blockseriesList)

      mem_series = constructMemSeries(memfile, h_index-1, lstmtimestamps)
      memseriesList = [mem_series] * txsArray.shape[0]
      memSeqList.extend(memseriesList)
  return txfeatureList,txOutputList,np.array(blockSeqList),np.array(memSeqList)














cost_sensitive =False








discretization_list = [ 'EPI']
for classes in classes_list:
    for discretization in discretization_list:
    ####Construct Training and Testing dataset
        train_txfeatureList1,train_txOutputList1,strain_blockSeqList1,strain_memSeqList1=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,START_EstimateBlock-training_blocks,START_EstimateBlock+total_EstimateBlock,classes, discretization, smallest_classes)
        #######Scale Features
        trn_X1=np.array(train_txfeatureList1)
        trn_Y1 = np.array(train_txOutputList1)
        #Normalization
        ss_x = MinMaxScaler()
        strain_txfeatureList1 = ss_x.fit_transform(trn_X1)
        # encoder = LabelEncoder()
        # temp_Y = encoder.fit_transform(trn_Y1)





        lastUpdateHeight=START_EstimateBlock+addaccount

        train_txfeatureList,train_txOutputList,strain_blockSeqList,strain_memSeqList=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,lastUpdateHeight-training_blocks,lastUpdateHeight,classes, discretization, smallest_classes)
        #######Scale Features
        trn_X=np.array(train_txfeatureList)
        trn_Y=np.array(train_txOutputList)
        #Normalization

        strain_txfeatureList = ss_x.transform(trn_X)


        
        #encoded_Y = encoder.transform(trn_Y)
        # convert integers to dummy variables (one hot encoding)
        blocksInterval = Classfyining(classes)
        REAL_classLimit=max(blocksInterval)+1
        strain_txOutputList = np_utils.to_categorical(trn_Y,num_classes=REAL_classLimit)
















        fel_model = selfAttentionmempoolModel(layers, lstmunits, lstmtimestamps, BocFeaLens,MemFeaLens, TxFeaLens,REAL_classLimit)
        plot_model(fel_model, to_file=dir_abbv+str(classes)+discretization+'.png', show_shapes=True)


        filepath_fel = result_path + SelectionModule + lossfunction + target+str(lastUpdateHeight)+str(classes)+discretization+'.h5'

        checkpoint = ModelCheckpoint(
          filepath=filepath_fel,
          monitor='acc',
          save_best_only=True,
          verbose=0,
          mode='auto',
          save_weights_only=False,
          period=1)

        fel_hist = fel_model.fit([strain_blockSeqList,strain_memSeqList, strain_txfeatureList], strain_txOutputList, epochs=prediction_epoch,
                                 batch_size=bachsize, verbose=0,callbacks=[checkpoint])
