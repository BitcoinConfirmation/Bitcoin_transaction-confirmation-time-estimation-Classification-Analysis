
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




#Set1
START_EstimateBlock=516738
total_EstimateBlock=45
result_path = './NeuarlResult/ResultFor90Blocks/NoneEnterBlock/'
blockfile='TimeBlock2.csv'
txfile='TimetxinBlock2.csv'



#Set2
START_EstimateBlock_S2=621301
total_EstimateBlock_S2=45
result_path_S2 = './NeuarlResult/set2ResultFor90Blocks/NoneEnterBlock/'
blockfile_S2='TimeBlock621500.csv'
txfile_S2='TimetxinBlock621500.csv'


#Set3
START_EstimateBlock_S3=621801
total_EstimateBlock_S3=45
result_path_S3 = './NeuarlResult/set3ResultFor90Blocks/NoneEnterBlock/'

blockfile_S3='TimeBlock622000.csv'
txfile_S3='TimetxinBlock622000.csv'


#Set4
START_EstimateBlock_S4=622301
total_EstimateBlock_S4=45
result_path_S4 = './NeuarlResult/set4ResultFor90Blocks/NoneEnterBlock/'
result_path_S4_test = './NeuarlResult/ImpromentTest/'
blockfile_S4='TimeBlock622500.csv'
txfile_S4='TimetxinBlock622500.csv'


#Set5
START_EstimateBlock_S5=621293
total_EstimateBlock_S5=45
result_path_S5 = './NeuarlResult/set5ResultFor90Blocks/NoneEnterBlock/'
blockfile_S5='TimeBlock621500.csv'
txfile_S5='TimetxinBlock621500.csv'

#Set6
START_EstimateBlock_S6=621320
total_EstimateBlock_S6=45
result_path_S6 = './NeuarlResult/set6ResultFor90Blocks/NoneEnterBlock/'
blockfile_S6='TimeBlock621500.csv'
txfile_S6='TimetxinBlock621500.csv'


#Set7
START_EstimateBlock_S7=622214
total_EstimateBlock_S7=45
result_path_S7 = './NeuarlResult/set7ResultFor90Blocks/NoneEnterBlock/'
blockfile_S7='TimeBlock622500.csv'
txfile_S7='TimetxinBlock622500.csv'


#Set8
START_EstimateBlock_S8=622328
total_EstimateBlock_S8=45
result_path_S8 = './NeuarlResult/set8ResultFor90Blocks/NoneEnterBlock/'
blockfile_S8='TimeBlock622500.csv'
txfile_S8='TimetxinBlock622500.csv'


#Set9
START_EstimateBlock_S9=621801
total_EstimateBlock_S9=45
result_path_S9 = './NeuarlResult/set9ResultFor90Blocks/NoneEnterBlock/'

blockfile_S9='TimeBlock622000.csv'
txfile_S9='TimetxinBlock622000.csv'


#Set10
START_EstimateBlock_S10=621801
total_EstimateBlock_S10=45
result_path_S10 = './NeuarlResult/set10ResultFor90Blocks/NoneEnterBlock/'

blockfile_S10='TimeBlock622000.csv'
txfile_S10='TimetxinBlock622000.csv'


#Set11
START_EstimateBlock_S11=621801
total_EstimateBlock_S11=45
result_path_S11 = './NeuarlResult/set11ResultFor90Blocks/NoneEnterBlock/'

blockfile_S11='TimeBlock622000.csv'
txfile_S11='TimetxinBlock622000.csv'


#Set12
START_EstimateBlock_S12=621801
total_EstimateBlock_S12=45
result_path_S12 = './NeuarlResult/set12ResultFor90Blocks/NoneEnterBlock/'

blockfile_S12='TimeBlock622000.csv'
txfile_S12='TimetxinBlock622000.csv'

#Set13
START_EstimateBlock_S13=621801
total_EstimateBlock_S13=45
result_path_S13 = './NeuarlResult/set13ResultFor90Blocks/NoneEnterBlock/'

blockfile_S13='TimeBlock622000.csv'
txfile_S13='TimetxinBlock622000.csv'



#Set14
START_EstimateBlock_S14=621801
total_EstimateBlock_S14=45
result_path_S14 = './NeuarlResult/set14ResultFor90Blocks/NoneEnterBlock/'

blockfile_S14='TimeBlock622000.csv'
txfile_S14='TimetxinBlock622000.csv'

#Set15
START_EstimateBlock_S15=621756
total_EstimateBlock_S15=45
result_path_S15 = './NeuarlResult/set15ResultFor90Blocks/NoneEnterBlock/'

blockfile_S15='TimeBlock622000.csv'
txfile_S15='TimetxinBlock622000.csv'







#Set16-18 Changing Blockfeatures in Experiments (For S2 S3 and S4)

#Set16
START_EstimateBlock_S16=621301
total_EstimateBlock_S16=45
result_path_S16 = './NeuarlResult/set16ResultFor90Blocks/NoneEnterBlock/'
blockfile_S16='TimeBlock621500.csv'
txfile_S16='TimetxinBlock621500.csv'


#Set17
START_EstimateBlock_S17=621801
total_EstimateBlock_S17=45
result_path_S17 = './NeuarlResult/set17ResultFor90Blocks/NoneEnterBlock/'

blockfile_S17='TimeBlock622000.csv'
txfile_S17='TimetxinBlock622000.csv'


#Set18
START_EstimateBlock_S18=622301
total_EstimateBlock_S18=45
result_path_S18 = './NeuarlResult/set18ResultFor90Blocks/NoneEnterBlock/'
blockfile_S18='TimeBlock622500.csv'
txfile_S18='TimetxinBlock622500.csv'



#Set16-18 Changing Blockfeatures in Experiments (For S6  S7and S8) with distinctive trends
#Set19(S6)
START_EstimateBlock_S19=621320
total_EstimateBlock_S19=45
result_path_S19 = './NeuarlResult/set19ResultFor90Blocks/NoneEnterBlock/'
blockfile_S19='TimeBlock621500.csv'
txfile_S19='TimetxinBlock621500.csv'






#Set20(S7)
START_EstimateBlock_S20=622214
total_EstimateBlock_S20=45
result_path_S20 = './NeuarlResult/set20ResultFor90Blocks/NoneEnterBlock/'
blockfile_S20='TimeBlock622500.csv'
txfile_S20='TimetxinBlock622500.csv'



#Set21(S8)
START_EstimateBlock_S21=622328
total_EstimateBlock_S21=45
result_path_S21 = './NeuarlResult/set21ResultFor90Blocks/NoneEnterBlock/'
blockfile_S21='TimeBlock622500.csv'
txfile_S21='TimetxinBlock622500.csv'

############New Two with new blockFature
#Set22(New From S2 ensure it has a incresing trend in both test and training )
START_EstimateBlock_S22=621185
total_EstimateBlock_S22=45
result_path_S22 = './NeuarlResult/set22ResultFor90Blocks/NoneEnterBlock/'
blockfile_S22='TimeBlock621500.csv'
txfile_S22='TimetxinBlock621500.csv'

#Set23(New From S3 ensure it has a decresaing trend in both testing)
START_EstimateBlock_S23=621853
total_EstimateBlock_S23=45
result_path_S23 = './NeuarlResult/set23ResultFor90Blocks/NoneEnterBlock/'
blockfile_S23='TimeBlock622000.csv'
txfile_S23='TimetxinBlock622000.csv'


#Set24(S5 with new BlockFeature)
TestGroup='24'
START_EstimateBlock_S24=621293
total_EstimateBlock_S24=45
result_path_S24 = './NeuarlResult/set'+TestGroup+'ResultFor90Blocks/NoneEnterBlock/'
blockfile_S24='TimeBlock621500.csv'
txfile_S24='TimetxinBlock621500.csv'





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
