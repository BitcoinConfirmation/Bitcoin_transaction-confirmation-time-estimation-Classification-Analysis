import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import G_Variables_WWWJ




ClassList=[2,4,6,8]
modelList =['xgBoostLSTM0Layer','xgBoostLSTM1Layer','xgBoostLSTM2Layer','xgBoostLSTM3Layer','xgBoostLSTM4Layer','xgBoostLSTM6Layer','xgBoostLSTM7Layer']


colums_list=['acc','prec_macro','prec_micro','prec_wht','recall_macro','recall_micro','recall_wht','f1_macro','f1_micro','f1_wht']




#
#
# def GetMetricResult(testgroup,alg_List):
#     testgroup=str(testgroup)
#     START_EstimateBlock = getattr(G_Variables_WWWJ, 'START_EstimateBlock_S' + testgroup)
#
#     height = START_EstimateBlock
#     df = pd.DataFrame(columns=colums_list)
#     for classlabel in ClassList:
#         for alg in alg_List:
#
#             sourceDate = pd.read_csv('../WWWJ_OverallPerformance/ClassifiedFeerateBin36Feerate' + str(classlabel)+'ClassTime'+testgroup + alg+str(height)+str(classlabel)+'results.csv',
#                                      header=None)
#             temp_data = sourceDate.values
#             if alg.__contains__('final'):
#                 alg=alg.replace('final','')
#                 alg=alg.replace('Attentions', '')
#             df.loc[alg + '_' + str(classlabel)] = temp_data[0, -10:].tolist()
#
#     df.to_csv(str(testgroup) + 'AllAlgorithmsSelected.csv')
#
# for testgroup in [31,32,33,34,35,36]:
#     testSet=str(testgroup)
#     GetMetricResult(testSet,alg_List)

testgroup=31
sourceData=pd.read_csv(str(31)+'AllAlgorithmsSelectedLSTMLayers.csv',index_col = 0)
Sum_temp=sourceData[['acc','prec_macro','recall_macro','f1_macro']]
count=1
for testgroup in [32,33,34,35,36]:
    sourceData2=pd.read_csv(str(testgroup)+'AllAlgorithmsSelectedLSTMLayers.csv',index_col = 0)
    temp2=sourceData2[['acc','prec_macro','recall_macro','f1_macro']]
    Sum_temp=Sum_temp+temp2
    count=count+1
OverallResult=Sum_temp/count
OverallResult.to_csv('OverallResult_AllAlgorithmsSelectedLSTMLayers.csv')

print('hello')
tempvalue=sourceData.values
tempvalue2=sourceData2.values
sumed=tempvalue+tempvalue2
sumed=sumed[:,[0,1,4,7]]


evaluationList=['acc','prec_macro','recall_macro','f1_macro']
modelList =['xgBoostLSTM0Layer','xgBoostLSTM1Layer','xgBoostLSTM2Layer','xgBoostLSTM3Layer','xgBoostLSTM4Layer','xgBoostLSTM5Layer','xgBoostLSTM6Layer','xgBoostLSTM7Layer']

colorList=['#002c53','#ffa510','#0c84c6' ,'#ffbd66','#f74d4d' ,'#2455a4','#41b7ac','#943c39']
modelName_nic =['Layer_0','Layer_1','Layer_2','Layer_3','Layer_4','Layer_5','Layer_6','Layer_7']
evaluation_nic=['accuracy','precision','recall','f1-score']
evaluationList=['acc']
evaluation_nic=['accuracy']

#
# for style in  plt.style.available:
for style in ['seaborn-deep']:

    for eva_inx in range(len(evaluationList)):
        evalabel=evaluationList[eva_inx]

        plt.style.use(style)
        plt.figure(figsize=(4,3))
        name_list = ['k=2', 'k=4', 'k=6', 'k=8']
        classList=[2,4,6,8]
        x = list(range(len(classList)))

        width_1 = 0.08

        for model_inx in range(len(modelList)):

            model_result = []
            for classlabel in [2,4,6,8]:
                print(modelList[model_inx])
                model_result.append(OverallResult.loc[modelList[model_inx] + '_' + str(classlabel), evalabel])

            plt.bar(np.arange(len(model_result)) + model_inx*width_1, model_result, width=width_1, tick_label=name_list,label=modelName_nic[model_inx])
        plt.ylabel(evaluation_nic[eva_inx])
        plt.xlabel('class size')
        plt.legend(loc='lower center',
                   ncol=4, bbox_to_anchor=(0.5, 0.97),
                   borderaxespad=0.
                   , fontsize=8)
        plt.savefig('./LSTM_figures/'+evalabel + 'BarChart_LSTMLayers.pdf')
        plt.show()
