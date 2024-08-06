import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
from tensorflow.keras.models import load_model
import math

import random
from scipy import stats
import numpy as np
from tensorflow.keras.utils import plot_model,to_categorical
import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

import sys
sys.path.append("..")
import G_Variables_WWWJ
import os


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
relayintx=G_Variables_WWWJ.relayintx
feeintx=G_Variables_WWWJ.feeintx
feerateintx = G_Variables_WWWJ.feerateintx
blockHeightintx = G_Variables_WWWJ.blockHeightintx
lastBlockIntervalintx=G_Variables_WWWJ.lastBlockIntervalintx## obsertime-latblocktime

###block feature
blockHeightBinx=G_Variables_WWWJ.blockHeightBinx
###mem feature
blockHeightMeminx=G_Variables_WWWJ.blockHeightMeminx


###Block Distribution
BLOCKSIZE=4000000 # the size (weight) of a block
BucketLen=36
FeerateIntervalLabel='ClassifiedFeerateBin'+str(BucketLen)
######


###Traning Parameters
training_blocks=G_Variables_WWWJ.training_blocks
lstmunits=G_Variables_WWWJ.lstmunits
lstmtimestamps=G_Variables_WWWJ.lstmtimestamps
prediction_epoch=G_Variables_WWWJ.prediction_epoch*3
bachsize=G_Variables_WWWJ.bachsize
optimizer_model=G_Variables_WWWJ.optimizer_model
dropout_factor=G_Variables_WWWJ.dropout_factor
evametrics='accuracy'
lossfunction='categorical_crossentropy'




#****Classification
clusterNumber=4
#************


TestGroup=36
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
dir_abbv='Transformer'
result_path='.'+result_path
sleep_time= random.randint(1,40) ##sleep
time.sleep(sleep_time)
dirs= result_path+SelectionModule+dir_abbv
if not os.path.exists(dirs):
    os.makedirs(dirs)
result_path=result_path+SelectionModule+dir_abbv+'/'







TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx,lastBlockIntervalintx]
blockFeatureDim=BucketLen
mempoolDim=BucketLen
txFeatureDim=len(TxFeatureSelection)+1

#***Parameter****
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
dropout_factor=0.2
FC_units=[64,48,36,24,18,12,9]
FC_units.append(clusterNumber)
#***************






class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":self.ff_dim,
            "rate":self.rate,
        }
        base_config = super(TransformerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.
  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths. Defined and formulized in
   "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).
  Args:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
  """

  def __init__(self,
               hidden_size: int,
               min_timescale: float = 1.0,
               max_timescale: float = 1.0e4,
               **kwargs):
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically
    # unstable in float16.
    if "dtype" not in kwargs:
      kwargs["dtype"] = "float32"

    super().__init__(**kwargs)
    self._hidden_size = hidden_size
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def get_config(self):
    config = {
        "hidden_size": self._hidden_size,
        "min_timescale": self._min_timescale,
        "max_timescale": self._max_timescale,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, length=None):
    """Implements call() for the layer.
    Args:
      inputs: An tensor whose second dimension will be used as `length`. If
        `None`, the other `length` argument must be specified.
      length: An optional integer specifying the number of positions. If both
        `inputs` and `length` are spcified, `length` must be equal to the second
        dimension of `inputs`.
    Returns:
      A tensor in shape of `(length, hidden_size)`.
    """
    if inputs is None and length is None:
      raise ValueError("If inputs is None, `length` must be set in "
                       "RelativePositionEmbedding().")
    if inputs is not None:
      input_shape = inputs.shape.as_list()
      if length is not None and length != input_shape[1]:
        raise ValueError(
            "If inputs is not None, `length` must equal to input_shape[1].")
      length = input_shape[1]
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = self._hidden_size // 2
    min_timescale, max_timescale = self._min_timescale, self._max_timescale
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) *
        -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    position_embeddings = tf.concat(
        [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return inputs+position_embeddings

def TransformerModel(num_heads,ff_dim,FC_units,  lstmtimestamps, blockFeatureDim, mempoolDim, txFeatureDim):

    model2_input1 = layers.Input(shape=(lstmtimestamps, blockFeatureDim), name='input1_BlockSeq')
    positional_layer=PositionEmbedding(hidden_size=blockFeatureDim)(model2_input1)
    transformer_Block=TransformerBlock(blockFeatureDim, num_heads, ff_dim)
    model2_transformer_out= transformer_Block(positional_layer) ##Should match the inputshape (bacause of residensial)
    model2_bloSeq_out=layers.GlobalAveragePooling1D()(model2_transformer_out)


    model2_input2 = layers.Input(shape=(lstmtimestamps, mempoolDim), name='input2_memSeq')
    positional_layer_mem=PositionEmbedding(hidden_size=blockFeatureDim)(model2_input2)
    model2_transformer_out2=TransformerBlock(mempoolDim, num_heads, ff_dim)(positional_layer_mem) ##Should match the inputshape (bacause of residensial)
    model2_memSeq_out=layers.GlobalAveragePooling1D()(model2_transformer_out2)

    model2_auxiliary_input =layers.Input(shape=(txFeatureDim,), name='input3_Tx')

    model2_merged_vector = layers.concatenate([model2_bloSeq_out, model2_memSeq_out, model2_auxiliary_input], axis=-1,name='FeaturesConcate')
    model2_merged_vector = layers.Dropout(dropout_factor,name='ConcateFeaDropout')(model2_merged_vector)
    layer1_vector=model2_merged_vector
    for lay_inx in range(len(FC_units)-1):
        layer1_vector = layers.Dense(FC_units[lay_inx], kernel_initializer='uniform', activation='relu',name='Dense_FC_'+str(lay_inx+1))(
            layer1_vector)
        layer1_vector = layers.Dropout(dropout_factor,name='Drop_FC_'+str(lay_inx+1))(layer1_vector)


    model2_predictions = layers.Dense(FC_units[-1], activation='softmax',name='Dense_FC_'+str(len(FC_units)))(layer1_vector)
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

#***Model Training**
transformer_model = TransformerModel(num_heads, ff_dim, FC_units, lstmtimestamps, blockFeatureDim, mempoolDim, txFeatureDim)
plot_model(transformer_model, to_file='Class'+str(clusterNumber)+'Transformer.png', show_shapes=True)
filepath_fel = result_path + SelectionModule +str(lastUpdateHeight)+str(clusterNumber)+'.h5'
checkpoint = ModelCheckpoint(
  filepath=filepath_fel,
  monitor='accuracy',
  save_best_only=True,
  verbose=0,
  mode='auto',
  save_weights_only=False,
  period=1)

fel_hist = transformer_model.fit([strain_blockSeqList,strain_memSeqList, strain_txfeatureList], strain_txOutputList, epochs=prediction_epoch,
                         batch_size=bachsize, verbose=0,callbacks=[checkpoint])



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

Y_hat_original=transformer_model.predict([stest_blockSeqList,stest_memSeqList, stest_txfeatureList])
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





