
# coding: utf-8
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from statistics import mean
from costsensitive import WeightedOneVsRest
import sklearn.metrics as metrics
import copy
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
from itertools import product
import multiprocessing
import pandas as pd

import numpy as np


import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")
import G_Variables_WWWJ
import os
import time
import random
sleep_time= random.randint(1,6)





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


dir_abbv='deepForest'

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



estimators = 100
repeattime = 1
smallest_classes = 4
biggest_classes = 5
cost_sensitiveList = [False]
discretization_list = ['']
discretization=''





def calculate_error_sample(predictions_proba_sample, label):
    predictions_proba_wrong = copy.deepcopy(predictions_proba_sample)
    predictions_proba_wrong[label] = 1 - predictions_proba_wrong[label]
    error_sample = np.dot(predictions_proba_wrong, cost_matrix[label])
    return error_sample


def calculate_error_estimator(predictions_proba, label):
    error_all_sample = []
    for i in range(len(label)):
        error_sample = calculate_error_sample(predictions_proba[i], label[i])
        error_all_sample.append(error_sample)
    error_estimator = mean(error_all_sample)
    return error_estimator


def calculate_error_layer(predictions_proba, label):
    error_all_estimator = []
    for estimator in range(4):
        error_estimator = calculate_error_estimator(predictions_proba[estimator], label)
        error_all_estimator.append(error_estimator)
    error_layer = mean(error_all_estimator)
    return error_layer


def get_prediction(predictions_proba):
    predictions_proba_average = (predictions_proba[0] + predictions_proba[1] + 
                                       predictions_proba[2] + predictions_proba[3]) / 4
    predictions = predictions_proba_average.argmax(1)
    return predictions


# ## no cost sensitive

def deep_forest_non(estimators, X_train, X_test, y_train, y_test, cost_matrix):
    RandomForestEstimator_1 = RandomForestClassifier(n_estimators=estimators, n_jobs=-1, random_state=1)
    RandomForestEstimator_2 = RandomForestClassifier(n_estimators=estimators, n_jobs=-1, random_state=2)
    ExtraTreesEstimator_3 = ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1, random_state=3)
    ExtraTreesEstimator_4 = ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1, random_state=4)
    LayerClassifier = [RandomForestEstimator_1, RandomForestEstimator_2, 
                       ExtraTreesEstimator_3, ExtraTreesEstimator_4]
    DeepForest = []
    for layer in range(3):
        DeepForest.append(copy.deepcopy(LayerClassifier))
        if layer == 0:
            X_retrain = X_train
            X_retest = X_test
        else:
            X_retrain = np.concatenate((X_train, concatenate_predictions_proba(predictions_proba_train)), axis=1)
            X_retest = np.concatenate((X_test, concatenate_predictions_proba(predictions_proba_test)), axis=1)
        predictions_proba_train = [0, 1, 2, 3]
        predictions_proba_test = [0, 1, 2, 3]
        for estimator in range(4):
            DeepForest[layer][estimator].fit(X_retrain, y_train)
            predictions_proba_train[estimator] = DeepForest[layer][estimator].predict_proba(X_retrain)
            predictions_proba_test[estimator] = DeepForest[layer][estimator].predict_proba(X_retest)
        
        error_layer_train = calculate_error_layer(predictions_proba_train, y_train)
        error_layer_test = calculate_error_layer(predictions_proba_test, y_test)
        if (layer > 0) and (error_layer_train_ - error_layer_train < 0.01 * error_layer_train):
            break
        error_layer_train_ = error_layer_train
        predictions_layer_test = get_prediction(predictions_proba_test)
    return layer, predictions_layer_test


# ## cost sensitive


def deep_forest_cs(estimators, X_train, X_test, y_train, y_test, cost_matrix):
    CostSensitiveRandomForestEstimator_1 = WeightedOneVsRest(RandomForestClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=1))
    CostSensitiveRandomForestEstimator_2 = WeightedOneVsRest(RandomForestClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=2))
    CostSensitiveExtraTreesEstimator_3 = WeightedOneVsRest(ExtraTreesClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=3))
    CostSensitiveExtraTreesEstimator_4 = WeightedOneVsRest(ExtraTreesClassifier(
        n_estimators=estimators, n_jobs=-1, random_state=4))
    LayerClassifier = [CostSensitiveRandomForestEstimator_1, CostSensitiveRandomForestEstimator_2, 
                       CostSensitiveExtraTreesEstimator_3, CostSensitiveExtraTreesEstimator_4]
    DeepForest = []
    C_train = np.array([cost_matrix[i] for i in y_train])
    C_test = np.array([cost_matrix[i] for i in y_test])
    for layer in range(3):
        DeepForest.append(copy.deepcopy(LayerClassifier))
        if layer == 0:
            X_retrain = X_train
            X_retest = X_test
        else:
            X_retrain = np.concatenate((X_train, concatenate_predictions_proba(predictions_proba_train)), axis=1)
            X_retest = np.concatenate((X_test, concatenate_predictions_proba(predictions_proba_test)), axis=1)
        predictions_proba_train = [0, 1, 2, 3]
        predictions_proba_test = [0, 1, 2, 3]
        for estimator in range(4):
            DeepForest[layer][estimator].fit(X_retrain, C_train)
            predictions_proba_train[estimator] = DeepForest[layer][estimator].decision_function(X_retrain)
            predictions_proba_test[estimator] = DeepForest[layer][estimator].decision_function(X_retest)
        error_layer_train = calculate_error_layer(predictions_proba_train, y_train)
        error_layer_test = calculate_error_layer(predictions_proba_test, y_test)
        if (layer > 0) and (error_layer_train_ - error_layer_train < 0.01 * error_layer_train):
            break
        error_layer_train_ = error_layer_train
        predictions_layer_test = get_prediction(predictions_proba_test)
    return layer, predictions_layer_test



def calculate_cost(predictions, y_test, cost_matrix):
    cost_sum = 0
    for n in range(len(y_test)):
        cost_sum += cost_matrix[int(y_test[n]), int(predictions[n])]
    cost = cost_sum/len(y_test)
    return cost





def get_metrics(y_test, predictions, cost_matrix):
    accuracy=metrics.accuracy_score(y_test,predictions)
    precision_mac=metrics.precision_score(y_test,predictions,average='macro')
    precision_mic = metrics.precision_score(y_test, predictions, average='micro')
    precision_wht=metrics.precision_score(y_test, predictions, average='weighted')
    recall_mac=metrics.recall_score(y_test,predictions,average='macro')
    recall_mic = metrics.recall_score(y_test, predictions, average='micro')
    recall_wht = metrics.recall_score(y_test, predictions, average='weighted')
    f1_macro=metrics.f1_score(y_test,predictions,average='macro')
    f1_micro = metrics.f1_score(y_test, predictions, average='micro')
    f1_wht = metrics.f1_score(y_test, predictions, average='weighted')
    cost = calculate_cost(predictions, y_test, cost_matrix)
    return accuracy,precision_mac,precision_mic,precision_wht,recall_mac,recall_mic,recall_wht,f1_macro,f1_micro,f1_wht,cost







def deep_forest(estimators, X_train, X_test, y_train, y_test, cost_matrix, cost_sensitive):
    if cost_sensitive:
        layer, predictions = deep_forest_cs(estimators, X_train, X_test, y_train, y_test, cost_matrix)
    else:
        layer, predictions = deep_forest_non(estimators, X_train, X_test, y_train, y_test, cost_matrix)
    accuracy, precision_mac, precision_mic,precision_wht,recall_mac, recall_mic,recall_wht,f1_score_mac, f1_score_mic,f1_score_wht,cost = get_metrics(y_test, predictions, cost_matrix)
    return layer, accuracy, precision_mac, precision_mic,precision_wht,recall_mac, recall_mic,recall_wht,f1_score_mac, f1_score_mic,f1_score_wht,cost


# # NEW

def average(prediction_proba):
    if len(prediction_proba) == 4:
        prediction_proba_average = (prediction_proba[0] + prediction_proba[1] + 
                                    prediction_proba[2] + prediction_proba[3]) / 4
    if len(prediction_proba) == 5:
        prediction_proba_average = (prediction_proba[0] + prediction_proba[1] + 
                                    prediction_proba[2] + prediction_proba[3] + prediction_proba[3]) / 5
    return prediction_proba_average


def realign(prediction_prob, index):
    for i in range(len(prediction_prob)):
        prediction_prob[i] = pd.DataFrame(data=prediction_prob[i], index=index[i])
    df_realignment = prediction_prob[0]
    for i in range(1, len(prediction_prob)):
        df_realignment = df_realignment.append(prediction_prob[i])
    df_realignment = df_realignment.sort_index()
    return df_realignment.values


def cross_validate_estimator(clf, X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    classifers = [0, 1, 2, 3, 4]
    prediction_prob_test_cv = [0, 1, 2, 3, 4]
    prediction_prob_test = [0, 1, 2, 3, 4]
    index = [0, 1, 2, 3, 4]
    i = 0
    for train_index_cv, test_index_cv in skf.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train[train_index_cv], X_train[test_index_cv]
        y_train_cv, y_test_cv = y_train[train_index_cv], y_train[test_index_cv]
        index[i] = test_index_cv
        classifers[i] = copy.deepcopy(clf)
        if cost_sensitive:
            C = np.array([cost_matrix[int(i)] for i in y_train_cv])
            classifers[i].fit(X_train_cv, C)
            prediction_prob_test_cv[i] = classifers[i].decision_function(X_test_cv)
            prediction_prob_test[i] = classifers[i].decision_function(X_test)
        else:
            classifers[i].fit(X_train_cv, y_train_cv)
            prediction_prob_test_cv[i] = classifers[i].predict_proba(X_test_cv)
            prediction_prob_test[i] = classifers[i].predict_proba(X_test)
        i += 1
    prediction_prob_train = realign(prediction_prob_test_cv, index)    
    prediction_prob_test = average(prediction_prob_test)
    return prediction_prob_train, prediction_prob_test



def layer_estimate(estimators, X_retrain, y_train, X_retest, y_test, cost_matrix, cost_sensitive):
    non_forests = [RandomForestClassifier(n_estimators=estimators, n_jobs=-1), 
                   RandomForestClassifier(n_estimators=estimators, n_jobs=-1), 
                   ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1), 
                   ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1)]
    cs_forests = [WeightedOneVsRest(RandomForestClassifier(n_estimators=estimators, n_jobs=-1)), 
                  WeightedOneVsRest(RandomForestClassifier(n_estimators=estimators, n_jobs=-1)), 
                  WeightedOneVsRest(ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1)), 
                  WeightedOneVsRest(ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1))]
    if cost_sensitive:
        forests = cs_forests
    else:
        forests = non_forests
    prediction_prob_train = [0, 1, 2, 3]
    prediction_prob_test = [0, 1, 2, 3]
    i = 0
    for frs in forests:
        prediction_prob_train[i], prediction_prob_test[i] = cross_validate_estimator(
            frs, X_retrain, y_train, X_retest, y_test, cost_matrix, cost_sensitive)
        i += 1
    return prediction_prob_train, prediction_prob_test



def get_layer_metrics(prediction_prob_train, prediction_prob_test, y_train, y_test, cost_matrix):
    prediction_train = average(prediction_prob_train)
    prediction_test = average(prediction_prob_test)
    predictions_train = prediction_train.argmax(1)
    predictions_test = prediction_test.argmax(1)
    accuracy_train, precision_mac_train,precision_mic_train,precision_wht_train, recall_mac_train,recall_mic_train,recall_wht_train, f1_score_mac_train,f1_score_mic_train,f1_score_wht_train, cost_train = get_metrics(
        y_train, predictions_train, cost_matrix)
    accuracy_test, precision_mac_test,precision_mic_test,precision_wht_test, recall_mac_test, recall_mic_test,recall_wht_test,f1_score_mac_test,f1_score_mic_test,f1_score_wht_test, cost_test = get_metrics(
        y_test, predictions_test, cost_matrix)
    return [accuracy_train, precision_mac_train,precision_mic_train,precision_wht_train, recall_mac_train,recall_mic_train,recall_wht_train, f1_score_mac_train,f1_score_mic_train,f1_score_wht_train, cost_train,
            accuracy_test, precision_mac_test,precision_mic_test,precision_wht_test, recall_mac_test, recall_mic_test,recall_wht_test,f1_score_mac_test,f1_score_mic_test,f1_score_wht_test, cost_test]



def concatenate_predictions_proba(predictions_proba):
    predictions_proba_all = np.concatenate((predictions_proba[0], predictions_proba[1], 
                                            predictions_proba[2], predictions_proba[3]), axis=1)
    return predictions_proba_all


def train_test_once(estimators, X_train, X_test, y_train, y_test, cost_matrix, cost_sensitive):
    stopping = 0
    for layer in range(3):
        if layer == 0:
            X_retrain = X_train
            X_retest = X_test

        else:
            X_retrain = np.concatenate((X_train, concatenate_predictions_proba(prediction_prob_train)), axis=1)
            X_retest = np.concatenate((X_test, concatenate_predictions_proba(prediction_prob_test)), axis=1)
        prediction_prob_train, prediction_prob_test = layer_estimate(
            estimators, X_retrain, y_train, X_retest, y_test, cost_matrix, cost_sensitive)
        [accuracy_train, precision_mac_train,precision_mic_train,precision_wht_train, recall_mac_train,recall_mic_train,recall_wht_train, f1_score_mac_train,f1_score_mic_train,f1_score_wht_train, cost_train,
         accuracy_test, precision_mac_test,precision_mic_test,precision_wht_test, recall_mac_test, recall_mic_test,recall_wht_test,f1_score_mac_test,f1_score_mic_test,f1_score_wht_test, cost_test] = get_layer_metrics(
            prediction_prob_train, prediction_prob_test, y_train, y_test, cost_matrix)
        if cost_sensitive:
            if (layer == 0) or (best_cost_train - cost_train > 0.001 * best_cost_train):
                best_layer = copy.deepcopy(layer) + 1

                best_accuracy_train = copy.deepcopy(accuracy_train)
                best_cost_train = copy.deepcopy(cost_train)
                best_accuracy_test = copy.deepcopy(accuracy_test)
                best_precision_mac_test = copy.deepcopy(precision_mac_test)
                best_precision_mic_test = copy.deepcopy(precision_mic_test)
                best_precision_wht_test = copy.deepcopy(precision_wht_test)
                best_recall_mac_test = copy.deepcopy(recall_mac_test)
                best_recall_mic_test = copy.deepcopy(recall_mic_test)
                best_recall_wht_test = copy.deepcopy(recall_wht_test)
                best_f1_score_mac_test = copy.deepcopy(f1_score_mac_test)
                best_f1_score_mic_test = copy.deepcopy(f1_score_mic_test)
                best_f1_score_wht_test = copy.deepcopy(f1_score_wht_test)
                best_cost_test = copy.deepcopy(cost_test)
                stopping = 0
            else:
                stopping += 1
        else:
            if (layer == 0) or (accuracy_train - best_accuracy_train > 0.001 * best_accuracy_train):
                best_layer = copy.deepcopy(layer) + 1

                best_accuracy_train = copy.deepcopy(accuracy_train)
                best_cost_train = copy.deepcopy(cost_train)
                best_accuracy_test = copy.deepcopy(accuracy_test)
                best_precision_mac_test = copy.deepcopy(precision_mac_test)
                best_precision_mic_test = copy.deepcopy(precision_mic_test)
                best_precision_wht_test = copy.deepcopy(precision_wht_test)
                best_recall_mac_test = copy.deepcopy(recall_mac_test)
                best_recall_mic_test = copy.deepcopy(recall_mic_test)
                best_recall_wht_test = copy.deepcopy(recall_wht_test)
                best_f1_score_mac_test = copy.deepcopy(f1_score_mac_test)
                best_f1_score_mic_test = copy.deepcopy(f1_score_mic_test)
                best_f1_score_wht_test = copy.deepcopy(f1_score_wht_test)


                best_cost_test = copy.deepcopy(cost_test)
                stopping = 0
            else:
                stopping += 1
        if stopping == 3:
            break
    return best_layer, best_accuracy_test, best_precision_mac_test,best_precision_mic_test,best_precision_wht_test, best_recall_mac_test,best_recall_mic_test,best_recall_wht_test, best_f1_score_mac_test,best_f1_score_mic_test,best_f1_score_wht_test, best_cost_test



def get_mean_results(estimators, X_train, y_train, X_test, y_test, cost_matrix, cost_sensitive, repeattime):

    layer_list = []
    accuracy_list = []
    precision_mac_list = []
    precision_mic_list = []
    precision_wht_list = []
    recall_mac_list = []
    recall_mic_list = []
    recall_wht_list = []
    f1_score_mac_list = []
    f1_score_mic_list = []
    f1_score_wht_list = []
    cost_list = []

    for i in range(repeattime):  # time = repeat time for experiments
        layer_, accuracy_, precision_mac_,precision_mic_,precision_wht_, recall_mac_, recall_mic_,recall_wht_,f1_score_mac_,f1_score_mic_,f1_score_wht_, cost_ = train_test_once(
            estimators, X_train, X_test, y_train, y_test, cost_matrix, cost_sensitive)

        layer_list.append(layer_)
        accuracy_list.append(accuracy_)
        precision_mac_list.append(precision_mac_)
        precision_mic_list.append(precision_mic_)
        precision_wht_list.append(precision_wht_)

        recall_mac_list.append(recall_mac_)
        recall_mic_list.append(recall_mic_)
        recall_wht_list.append(recall_wht_)

        f1_score_mac_list.append(f1_score_mac_)
        f1_score_mic_list.append(f1_score_mic_)
        f1_score_wht_list.append(f1_score_wht_)

        cost_list.append(cost_)

    layer = mean(layer_list)
    accuracy = mean(accuracy_list)
    precision_mac = mean(precision_mac_list)
    precision_mic = mean(precision_mic_list)
    precision_wht = mean(precision_wht_list)

    recall_mac = mean(recall_mac_list)
    recall_mic = mean(recall_mic_list)
    recall_wht = mean(recall_wht_list)

    f1_score_mac = mean(f1_score_mac_list)
    f1_score_mic = mean(f1_score_mic_list)
    f1_score_wht = mean(f1_score_wht_list)
    cost = mean(cost_list)

    return layer, accuracy, precision_mac,precision_mic,precision_wht, recall_mac,recall_mic,recall_wht, f1_score_mac,f1_score_mic,f1_score_wht, cost




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













def txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,startSearchBlock,endSearchBlock,classes):
  txcollection = pd.read_csv(txfile, sep=",",header=None)
 # txcollection=txcollection[txcollection[waitingtimeinx]>0]

  txcollection = pd.read_csv(txfile, sep=",",header=None)
  blocksInterval=Classfyining(classes)
  newcol=txcollection.shape[1]
  txcollection[newcol]=txcollection[waitingblockintx]
  txcollection[newcol]=  txcollection[newcol].apply(lambda x: blocksInterval[int(x-1)])



  txfeatureList=[]
  txOutputList_class=[]
  txOutputList_value=[]
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
          higherWeights = sum(higherUnconftx[:, weightintx])
          virtualBlockPos = higherWeights / BLOCKSIZE
          txfeatureay = tx[TxFeatureSelection]
          fea_list = txfeatureay.tolist()
          fea_list.append(virtualBlockPos)
          txfeatureList.append(fea_list)



      txOutputClass = txsArray[:, -1]
      ####The last colum is class label
      txOutputList_class.extend(txOutputClass.tolist())
      txOutputTime=txsArray[:, waitingblockintx]
      txOutputList_value.extend(txOutputTime)
      
      
      # block_series = constructBlockSeries(blockfile, h_index-1, lstmtimestamps)
      # blockseriesList = [block_series] * txsArray.shape[0]
      # blockSeqList.extend(blockseriesList)

      # mem_series = constructMemSeries(memfile, h_index-1, lstmtimestamps)
      # memseriesList = [mem_series] * txsArray.shape[0]
      # memSeqList.extend(memseriesList)
  Y_time =  pd.Series(data=np.array(txOutputList_value).reshape(-1),name='price')
  Y_label=pd.Series(data=np.array(txOutputList_class).reshape(-1),name='label')
  txOutputList = pd.concat([Y_time, Y_label], axis=1)
  
  All_time=pd.Series(data=txcollection[waitingblockintx].values,name='price')
  All_label=pd.Series(data=txcollection[newcol].values,name='label')
  Df_label = pd.concat([All_time, All_label], axis=1)
  return txfeatureList,txOutputList,np.array(blockSeqList),np.array(memSeqList),Df_label


def save_to_file(file_name, contents):
    fh = open(file_name, 'a')
    fh.write(contents)
    fh.write('\n')
    fh.close()




def find_centroid(df_label, classes):
    
    centroid = []
    for i in range(classes):
        df_sub = df_label['price'][df_label['label'] == i]
        mean = df_sub.mean()
        centroid.append(mean)
        
    return centroid




def get_cost_matrix(centroid,classes):
    cost_matrix = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            cost_matrix[i, j] = abs(centroid[i]-centroid[j])
    return cost_matrix



CurrentHeight=START_EstimateBlock+addaccount
def train_test_df(txfile, smallest_classes, classes, discretization, cost_sensitiveLabel, estimators, repeattime):
    

    train_txfeatureList1,train_txOutputList1,strain_blockSeqList1,strain_memSeqList1,Df_label=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,CurrentHeight-training_blocks,CurrentHeight+total_EstimateBlock,classes)
    #######Scale Features
    trn_X1=np.array(train_txfeatureList1)
    #Normalization
    ss_x = MinMaxScaler()
    strain_txfeatureList1 = ss_x.fit_transform(trn_X1)

    
    train_txfeatureList,train_txOutputList,strain_blockSeqList,strain_memSeqList,Df_label=txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,CurrentHeight-training_blocks,CurrentHeight,classes)
    #######Scale Features
    trn_X=np.array(train_txfeatureList)
    #Normalization
    X_train = ss_x.transform(trn_X)
    y_train=train_txOutputList['label'].astype(int).values
    

    
    test_txfeatureList, test_txOutputList,stest_blockSeqList,stest_memSeqList,Df_label = txDatasetConstruction(txfile,blockfile,memfile,lstmtimestamps,CurrentHeight, CurrentHeight + total_EstimateBlock,classes)
    tst_X = np.array(test_txfeatureList)
    X_test = ss_x.transform(tst_X)
    y_test=test_txOutputList['label'].astype(int).values
    

    
    centroid = find_centroid(Df_label, classes)
    cost_matrix = get_cost_matrix(centroid,classes)
    

    layer, accuracy, precision_mac, precision_mic,precision_wht,recall_mac,recall_mic,recall_wht, f1_score_mac,f1_score_mic,f1_score_wht, cost = get_mean_results(
        estimators, X_train, y_train, X_test, y_test, cost_matrix, cost_sensitiveLabel, repeattime)
    df_save = pd.DataFrame([[classes, discretization, cost_sensitiveLabel, estimators, repeattime, layer,cost, accuracy, precision_mac, precision_mic,precision_wht,recall_mac,recall_mic,recall_wht, f1_score_mac,f1_score_mic,f1_score_wht ]])
    if cost_sensitiveLabel:
        flag='costsensitive'
    else:
        flag='noncost'
    df_save.to_csv('../WWWJ_OverallPerformance/' + SelectionModule + TestGroup + dir_abbv + str(
        START_EstimateBlock + addaccount) + str(clusterNumber) + 'results.csv',
                   mode='w', encoding='utf-8', index=False, header=False)

    print(classes, discretization, cost_sensitiveLabel, estimators, repeattime, layer,  cost,accuracy, precision_mac, precision_mic,precision_wht,recall_mac,recall_mic,recall_wht, f1_score_mac,f1_score_mic,f1_score_wht)


if __name__ == '__main__':


   
    
    for cost_sensitiveLable in cost_sensitiveList:
    #pool = mp.Pool(processes=1, maxtasksperchild=1)
        train_test_df(txfile,smallest_classes, clusterNumber, discretization, cost_sensitiveLable, estimators, repeattime)

