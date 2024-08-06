'''
Created on 4 Jun. 2019

@author: limengzhang
'''
import json
import sys
import time

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

sys.path.append("../..")
import G_Variables
import os

######加载全局变量
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
train_timeinterval=180
test_timeinterval=60
######模型配置和选择
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

######文件路径和数据文件
TestGroup='16'
START_EstimateBlock=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables,'total_EstimateBlock_S'+TestGroup)
result_path = getattr(G_Variables,'result_path_S'+TestGroup)
blockfile= '../'+getattr(G_Variables,'blockfile_S'+TestGroup)
txfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
totalblockfile='../1000BlockTotal.csv'
SelectionModule='Time'
dir_abbv='MLPValExpSingle'
result_path='.'+result_path
dirs= result_path+SelectionModule+dir_abbv
if not os.path.exists(dirs):
    os.makedirs(dirs)
result_path=result_path+SelectionModule+dir_abbv+'/'



training_blocks=2
total_EstimateBlock=1

######特征选择
TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx,lastBlockIntervalintx,waitedTimeintx]
####实际还需要添加与观测点相关的数据
####Input
#  Waiting time since last block
#  waitedtime since entering
####Output
# 还需要等待的确认时间=confrimationTime-观测点
BocFeatureSelection=[n_txBinx,sizeBinx,bitsBinx,intervalBinx,valid_weightBinx,valid_sizeBinx,avg_feerateBinx,avg_waitingBinx,med_waitingBinx]
PredictSelection=[timeToConfirmintx]

######BlockScaler
blockcsv = pd.read_csv(blockfile, sep=",", header=None)
blocksfortrain = blockcsv[(blockcsv[blockHeightBinx]>=START_EstimateBlock-training_blocks)&(blockcsv[blockHeightBinx]<START_EstimateBlock)]
blockdata = np.array(blocksfortrain)
scalerBlock = StandardScaler()
blockFeatures = blockdata[:, BocFeatureSelection]
sblockFeatures = scalerBlock.fit_transform(blockFeatures)




def MLPModel(layers,txFeatureDim):
  np.random.seed(9)
 

  model2_auxiliary_input = Input(shape=(txFeatureDim,), name='model2_tx_input')
 
  model2_merged_vector = Dropout(dropout_factor)(model2_auxiliary_input)
  model2_layer1_vector = Dense(layers[0],kernel_initializer='uniform', activation='relu')(model2_merged_vector)
  model2_layer1_vector=Dropout(dropout_factor)(model2_layer1_vector)
  model2_layer2_vector = Dense(layers[1], activation='relu')(model2_layer1_vector)
  model2_layer2_vector=Dropout(dropout_factor)(model2_layer2_vector)
  model2_predictions = Dense(layers[2], activation='relu')(model2_layer2_vector)
  model2 = Model(inputs=model2_auxiliary_input, outputs=model2_predictions)
  model2.compile(loss=lossfunction, optimizer=optimizer_model,metrics=[lossfunction])
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






def txDatasetConstruction(txfile,blockfile,totalblockfile,lstmtimestamps,timeinterval,starttime,startEsitimateBlock):
  txcollection = pd.read_csv(txfile, sep=",",header=None)
  txcollection[lastBlockIntervalintx] = txcollection[0].apply(lambda x: 0)
  txcollection[waitedTimeintx]= txcollection[0].apply(lambda x: 0)
  txcollection[timeToConfirmintx] = txcollection[0].apply(lambda x: 0)

  txfeatureList=[]
  txOutputList=[]
  blockSeqList=[]
  obserTime=starttime
  end_time=getHeightTime(totalblockfile,startEsitimateBlock)
  #print('end_time'+str(end_time))
  while(obserTime<end_time):
      #print('obserTime'+str(obserTime))
      lastBlockHeight,lastBlocktime=getLastBlockTime(totalblockfile,obserTime)
      txsSelected=txcollection[(txcollection[validtimeintx]<=obserTime)&(txcollection[confirmedtimeintx]>=obserTime)]
      ####1. intital to zero
      txsSelected=txsSelected.copy()
      txsSelected[lastBlockIntervalintx] =txsSelected[lastBlockIntervalintx].apply(lambda x: obserTime-lastBlocktime)
      #txsSelected[lastBlockIntervalintx]=txsSelected[0]-txsSelected[0]
      txsSelected[waitedTimeintx]=txsSelected[validtimeintx].apply(lambda x:obserTime-x)

      txsSelected[timeToConfirmintx]=txsSelected[confirmedtimeintx].apply(lambda x:x-obserTime)
      ###2. dataselection and process
      txsArray=np.array(txsSelected)
      txfeatureArray=txsArray[:,TxFeatureSelection]
      txOutputArray=txsArray[:,PredictSelection]
      txfeatureList.extend(txfeatureArray.tolist())
      txOutputList.extend(txOutputArray.tolist())
      block_series = constructBlockSeries(blockfile, lastBlockHeight, lstmtimestamps)
      block_seriesList=[block_series]*txfeatureArray.shape[0]
      blockSeqList.extend(block_seriesList)
      obserTime=obserTime+timeinterval
  return txfeatureList,txOutputList,blockSeqList









####Construct Training and Testing dataset
train_starttime=getHeightTime(totalblockfile,START_EstimateBlock-training_blocks)
train_txfeatureList,train_txOutputList,train_blockSeqList=txDatasetConstruction(txfile,blockfile,totalblockfile,lstmtimestamps,train_timeinterval,train_starttime,START_EstimateBlock)
test_starttime=getHeightTime(totalblockfile,START_EstimateBlock)
test_txfeatureList,test_txOutputList,test_blockSeqList=txDatasetConstruction(txfile,blockfile,totalblockfile,lstmtimestamps,test_timeinterval,test_starttime,START_EstimateBlock+total_EstimateBlock)





#######Scale Features
trn_X=np.array(train_txfeatureList)
trn_Y=np.array(train_txOutputList)
tst_X=np.array(test_txfeatureList)
tst_Y=np.array(test_txOutputList)
#Normalization
ss_x,ss_y = StandardScaler(),StandardScaler()
strain_txfeatureList = ss_x.fit_transform(trn_X)
stest_txfeatureList = ss_x.transform(tst_X)
strain_txOutputList = ss_y.fit_transform(trn_Y.reshape([-1,1])).reshape(-1)
stest_txOutputList = ss_y.transform(tst_Y.reshape([-1,1])).reshape(-1)


strain_txfeatureList = strain_txfeatureList
stest_txfeatureList = stest_txfeatureList
strain_txOutputList = strain_txOutputList
stest_txOutputList = stest_txOutputList









#####Split training and validation
trn_txfeatureList,val_txfeatureList, trn_txOutputList, val_txOutputList,trn_blockSeqList,val_blockSeqList = train_test_split(
    strain_txfeatureList,strain_txOutputList,train_blockSeqList, test_size=0.25, random_state=9)

  






class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(NpEncoder, self).default(obj)

begin_time = time.time()
print('Processing ends')
print(begin_time)
#def LSTMWeightedAttentionmempoolModel(layers,lstmunits,lstmtimestamps,blockFeatureDim,txFeatureDim):
fel_model = MLPModel(layers,  len(TxFeatureSelection))
plot_model(fel_model, to_file=dir_abbv+'.png', show_shapes=True)
filepath_fel = result_path + SelectionModule + lossfunction + target+"Model_{epoch:d}.h5"
checkpoint = ModelCheckpoint(
  filepath=filepath_fel,
  monitor='val_loss',
  save_best_only=True,
  verbose=0,
  save_weights_only=False,
  period=1)
begin_time = time.time()
fel_hist = fel_model.fit([trn_txfeatureList], trn_txOutputList, epochs=prediction_epoch,
                         batch_size=bachsize, verbose=1,
                         validation_data=([val_txfeatureList], val_txOutputList)
                         , callbacks=[checkpoint])

end_time = time.time()
run_time = end_time - begin_time
# log.write(str(run_time))
Total_record={}
Total_record[target] = fel_hist.history
Total_record['time']=run_time
Total_record['trn_size']=len(trn_txfeatureList)
Total_record['val_size']=len(val_txfeatureList)
Total_record['tst_size']=len(stest_txfeatureList)


with open( 'S'+TestGroup+target + SelectionModule + lossfunction  + dir_abbv+'.txt', 'w') as file:
    file.write(json.dumps(Total_record, cls=NpEncoder))
    file.write('\n')













