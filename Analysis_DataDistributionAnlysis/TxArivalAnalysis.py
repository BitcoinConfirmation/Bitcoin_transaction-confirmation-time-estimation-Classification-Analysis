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

import sys

sys.path.append("..")
import G_Variables

import matplotlib.pyplot as plt
import joblib

intx = 1
outtx = 2
vertx = 3
sizeintx = 4
weightintx = 5
receivetimeintx = 6
relayintx = 7
lockintx = 8
feeintx = 9
blockHeightintx = 10
blockindexintx = 11
confirmedtimeintx = 12
waitingtimeinx = 13
feerateintx = 14
enterBlockintx = 15
waitingblockintx = 16
# Because of locktime info
validtimeintx = 17
validblockintx = 18
validwaitingintx = 19
# RelatedTo observation time
lastBlockIntervalintx = 20  ## obsertime-latblocktime(obseveBased)
waitedTimeintx = 21  # obsertime-receivetime
timeToConfirmintx = 22  # confirm

###tx feature
intx = G_Variables.intx
outtx = G_Variables.outtx
vertx = G_Variables.vertx
sizeintx = G_Variables.sizeintx
weightintx = G_Variables.weightintx
receivetimeintx = G_Variables.receivetimeintx
relayintx = G_Variables.relayintx
lockintx = G_Variables.lockintx
feeintx = G_Variables.feeintx
blockHeightintx = G_Variables.blockHeightintx
confirmedtimeintx = G_Variables.confirmedtimeintx
waitingtimeinx = G_Variables.waitingtimeinx
feerateintx = G_Variables.feerateintx
enterBlockintx = G_Variables.enterBlockintx
waitingblockintx = G_Variables.waitingblockintx

SHORT_BLOCK_PERIODS = 12
SHORT_SCALE = 1
MED_BLOCK_PERIODS = 24
MED_SCALE = 2
LONG_BLOCK_PERIODS = 1000
LONG_SCALE = 1
OLDEST_ESTIMATE_HISTORY = 6 * 1008

SHORT_DECAY = .962
MED_DECAY = .9952

# LONG_DECAY=SHORT_DECAY
# LONG_DECAY = .99931
LONG_DECAY = 0.962
# Require greater than 60% of X feerate transactions to be confirmed within Y/2 blocks*/
HALF_SUCCESS_PCT = .6
# Require greater than 85% of X feerate transactions to be confirmed within Y blocks*/
SUCCESS_PCT = .85
# Require greater than 95% of X feerate transactions to be confirmed within 2 * Y blocks*/
DOUBLE_SUCCESS_PCT = .95

# Require an avg of 0.1 tx in the combined feerate bucket per block to have stat significance */
SUFFICIENT_FEETXS = 0.1
# Require an avg of 0.5 tx when using short decay since there are fewer blocks considered*/
SUFFICIENT_TXS_SHORT = 0.5

MIN_BUCKET_TIME = 100
MAX_BUCKET_TIME = 600 * 50
FEE_SPACING = 1.05

training_blocks = G_Variables.training_blocks

TestGroup = 16
TestGroup = str(TestGroup)

MaxTime = 20000


def GenerateFeerateTimePoints(TestGroup, maxFeerate):
    txinblockfile = '../' + getattr(G_Variables, 'txfile_S' + TestGroup)
    txdata = pd.read_csv(txinblockfile, sep=",", header=None)
    txdata[feerateintx] = 4 * txdata[feeintx] / txdata[weightintx]
    txcount = np.array(txdata)
    txcount = txcount.shape[0]
    print(txcount)

    txdata = txdata[(txdata[feerateintx] <= maxFeerate) & (txdata[waitingtimeinx] <= MaxTime)]
    txArray = np.array(txdata)
    Points_x = txArray[:, feerateintx]
    Points_y = txArray[:, waitingtimeinx]

    return Points_x, Points_y


Points_x, Points_y = GenerateFeerateTimePoints(TestGroup, maxFeerate=1000)

fig = plt.figure(figsize=(9, 6))
plt.scatter(Points_x, Points_y, s=2)
plt.xlabel('Feerate')
plt.ylabel('Transaction Time')
plt.title("Confirmation Time Under Feerates (Time is almost fixed when feerate is over 1000)")
plt.savefig("ConfirmationTimeUnderFeerates.png")
plt.show()

# fig = plt.figure(figsize=(9,6))
# sns.distplot(a=Points_y,bins=MaxTime)
# plt.xlabel('Transaction Time Distribution time<'+str(MaxTime))
# plt.ylabel('Pencentage')
# plt.title("Time distrition  density ")
# plt.savefig("TimeDensity"+str(MaxTime)+".png")
# plt.show()

res_freq = stats.relfreq(Points_y, numbins=MaxTime)  # numbins 是统计一次的间隔(步长)是多大
pdf_value = res_freq.frequency
cdf_value = np.cumsum(res_freq.frequency)
x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
plt.figure()
plt.bar(x, pdf_value, width=res_freq.binsize)
plt.title('probability density function when Time under ' + str(MaxTime) + '.png')
plt.savefig('probability density function when Time under ' + str(MaxTime) + '.png')
plt.show()
plt.figure()
plt.plot(x, cdf_value)
plt.title('cumulative distribution function (5000-0.93) when Time under ' + str(MaxTime))
plt.savefig('cumulative distribution function when Time under ' + str(MaxTime) + '.png')
plt.show()
