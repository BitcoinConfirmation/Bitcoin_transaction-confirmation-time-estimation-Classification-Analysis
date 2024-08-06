import matplotlib.pyplot as plt
from keras.utils import np_utils

from keras.utils.vis_utils import plot_model
import keras


from keras.layers import Input, Embedding, LSTM,GRU, Dense,Dropout,Bidirectional
from keras.models import Model
from keras.layers.core import Flatten
from keras.models import load_model
from keras_self_attention import SeqSelfAttention,SeqWeightedAttention
from scipy import stats
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

#ChangeFeerateInterval from 0.1 to 0.001
FeerateIntervalLabel='Interval1000'
#########



######
TestGroup=31
TestGroup=str(TestGroup)
discretization_list = [ 'EPI']
classes_list = [8]
smallest_classes = 4
biggest_classes = 5

START_EstimateBlock=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables,'total_EstimateBlock_S'+TestGroup)
result_path = getattr(G_Variables,'result_path_S'+TestGroup)

blockfile_sub=FeerateIntervalLabel+str(classes_list[0])+'FeerateVector'
memfile_sub=FeerateIntervalLabel+str(classes_list[0])+'MemFeerateVector'

blockfile= '../../'+blockfile_sub+getattr(G_Variables,'blockfile_S'+TestGroup)
memfile='../../'+memfile_sub+getattr(G_Variables,'blockfile_S'+TestGroup)
txfile= '../../'+getattr(G_Variables,'txfile_S'+TestGroup)
totalblockfile='../1000BlockTotal.csv'



classLabel=classes_list[0]

memcsv = pd.read_csv(memfile, sep=",", header=None)
memDis=memcsv.values[:,-classLabel:]
blocksv=pd.read_csv(blockfile, sep=",", header=None)
blockDis=blocksv.values[:,-classLabel:]


plt.figure()
for i in range(classLabel):
    plt.plot(memDis[:,i],label='mem_Class'+str(i+1))
    plt.plot(blockDis[:, i], label='block_Class' + str(i + 1))
plt.legend()
plt.show()