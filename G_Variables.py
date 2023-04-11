
#############Tx Features

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
timeToConfirmintx=22# confirmtime-obsertime


#############Block Features
blockHeightBinx=1
n_txBinx=2
sizeBinx=3
bitsBinx=4
feeBinx=5
verBinx=6
timeBinx=7
intervalBinx=8
valid_weightBinx=9
valid_sizeBinx=10
avg_feerateBinx=11
avg_waitingBinx=12
med_waitingBinx=13

#############Mem Features
blockHeightMeminx=0
# the remaining part is the tx distribution



test=2
lstmunits=8
lstmunits32=32
lstmtimestamps=3
layers=[64,8,1]


training_blocks=180
prediction_epoch=100
bachsize=1000


# training_blocks=18
# prediction_epoch=1
# bachsize=1000


# Target aim  to estimation transaction fee confirmed within CONFTarget







optimizer_model='adam'
dropout_factor=0.2







#Set31
START_EstimateBlock_S31=621185
total_EstimateBlock_S31=45
result_path_S31 = './NeuarlResult/set31ResultFor90Blocks/NoneEnterBlock/'
blockfile_S31='TimeBlock621500.csv'
txfile_S31='TimetxinBlock621500.csv'


#Set32
START_EstimateBlock_S32=621435
total_EstimateBlock_S32=45
result_path_S32 = './NeuarlResult/set32ResultFor90Blocks/NoneEnterBlock/'
blockfile_S32='TimeBlock621500.csv'
txfile_S32='TimetxinBlock621500.csv'


#Set33
START_EstimateBlock_S33=621685
total_EstimateBlock_S33=45
result_path_S33 = './NeuarlResult/set33ResultFor90Blocks/NoneEnterBlock/'
blockfile_S33='TimeBlock622000.csv'
txfile_S33='TimetxinBlock622000.csv'


#Set34
START_EstimateBlock_S34=621935
total_EstimateBlock_S34=45
result_path_S34 = './NeuarlResult/set34ResultFor90Blocks/NoneEnterBlock/'
blockfile_S34='TimeBlock622000.csv'
txfile_S34='TimetxinBlock622000.csv'

#Set35
START_EstimateBlock_S35=622185
total_EstimateBlock_S35=45
result_path_S35 = './NeuarlResult/set35ResultFor90Blocks/NoneEnterBlock/'
blockfile_S35='TimeBlock622500.csv'
txfile_S35='TimetxinBlock622500.csv'

#Set36
START_EstimateBlock_S36=622435
total_EstimateBlock_S36=45
result_path_S36 = './NeuarlResult/set36ResultFor90Blocks/NoneEnterBlock/'
blockfile_S36='TimeBlock622500.csv'
txfile_S36='TimetxinBlock622500.csv'
