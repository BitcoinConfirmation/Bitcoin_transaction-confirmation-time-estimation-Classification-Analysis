import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import G_Variables


#
#
# ClassList=[8,6,4,2]
# alg_List =['finalMLP','finalAttentionsAdv',
#       'finalAttentionsWht',
#        'finalAttentionsQVK',
#       'finalLSTM','randomForest','rotationForest','xgBoost','lightGBM','deepForestTx','deepForestTx_cost','AdvxgBoost','AdvxgBoostLayer2','AdvxgBoostLayer1','AdvxgBoostLayer0','BaselineClassification']
#
#
#
#
# def GetMetricResult(testgroup,metric):
#     testgroup=str(testgroup)
#     START_EstimateBlock = getattr(G_Variables, 'START_EstimateBlock_S' + testgroup)
#
#     height = START_EstimateBlock
#
#     sourceDate = pd.read_csv('../Conclusion_Step4FinalMethodsSumarization/ForExp_FeerateVectorSet' + testgroup + '.csv',
#                              index_col=0)
#     df_acc=pd.DataFrame(columns=['2','4','6','8'])
#     for alg in alg_List:
#         alg_list=[]
#         for classLabel in ClassList:
#             if alg.__contains__('final'):
#                 keyName_sourceData='Interval1000Feerate'+str(classLabel)+'ClassTime'+alg+str(height)
#             else:
#                 keyName_sourceData = str(classLabel) +  alg + str(height)
#             if alg.__contains__('cost') and classLabel==2:
#                 metric_value=0
#             else:
#                 if alg.__contains__('cost'):
#                     keyName_sourceData = str(classLabel) + 'deepForestTx' + str(height)+'costsensitive'
#                 elif alg.__contains__('deepForestTx'):
#                     keyName_sourceData = str(classLabel) + 'deepForestTx' + str(height) + 'noncost'
#                 elif alg.__contains__('XgBoostAndRF'):
#                     keyName_sourceData = str(classLabel) + 'deepForest_XgBoostAndRF' + str(height) + 'noncost'
#                 elif alg.__contains__('deepForest_XgBoost'):
#                     keyName_sourceData = str(classLabel) + 'deepForest_XgBoost' + str(height) + 'noncost'
#                 elif alg.__contains__('AdvxgBoostLayer2'):
#                     keyName_sourceData = str(classLabel) + 'AdvxgBoostLayer2' + str(height)
#                 elif alg.__contains__('AdvxgBoostLayer1'):
#                     keyName_sourceData = str(classLabel) + 'AdvxgBoostLayer1' + str(height)
#                 elif alg.__contains__('AdvxgBoostLayer0'):
#                     keyName_sourceData = str(classLabel) + 'AdvxgBoostLayer0' + str(height)
#                 elif alg.__contains__('AdvxgBoost'):
#                     keyName_sourceData = str(classLabel) + 'AdvxgBoost' + str(height)
#                 metric_value=sourceDate.loc[keyName_sourceData][metric]
#             alg_list.append(metric_value)
#         df_acc.loc[alg]=alg_list
#     return df_acc
#
#
# name_list = alg_List
# name_list =['MLP','Adv',
#       'Wht',
#        'QVK',
#       'Lstm','RF','RoF','xgBoost','lightGBM','DF','DF_cost','DF_xg','DF_xgRF','AdvxgBoost','AdvxgBoostLayer2','AdvxgBoostLayer1','AdvxgBoostLayer0','Base']
#
# #x = list(range(len(name_list)))
# x = np.arange(len(name_list))
#
# total_width, n = 0.8, 4
# width =0.2
# labelList=['2','4','6','8']
# coloList=['#3B4992','#DE8F44','#01AAD5','#B34745']
#
#
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         plt.text(rect.get_x()+rect.get_width()/2.-0.1, 1.03*height, '%.3f' % float(height))
#
# def drawFigure(metricResult,metric):
#     figure = plt.figure(figsize=(10,4))
#     plt.rcParams['font.family'] = "Times New Roman"
#     for nLabel in range(len(labelList)):
#         #x_pos = [v - nLabel*width for v in x]
#         a=plt.bar(x+nLabel*width, metricResult[labelList[nLabel]], width=width, label='K='+labelList[nLabel],fc=coloList[nLabel],tick_label=name_list)
#         #autolabel(a)
#     plt.xlabel('Dataset 5')
#     plt.ylabel(metric)
#     plt.legend()
#     plt.subplots_adjust(left=0.126,bottom=0.136,right=None,top=0.88,wspace=0.15,hspace=0.15)
#     plt.savefig(str(testgroup)+metric+'.pdf')
#     plt.show()
#
#
#
#
# testgroup=31
#
#
# for testgroup in [31,32,33,34,35,36]:
#     metric='prec_macro'
#     prec_metricResults=GetMetricResult(testgroup,metric)
#
#     metric='recall_macro'
#     recall_metricResults=GetMetricResult(testgroup,metric)
#
#     metric='f1_macro'
#     f1_metricResults=GetMetricResult(testgroup,metric)
#     #drawFigure(f1_metricResults,metric)
#
#     metric='acc'
#     acc_metricResults=GetMetricResult(testgroup,metric)
#     #drawFigure(acc_metricResults,metric)
#
#
#
#     df_Overal_precRecall=pd.DataFrame()
#     df_Overal_f1acc= pd.DataFrame()
#     for classLabel in ClassList:
#         classLabel=str(classLabel)
#         df_Overal_precRecall[classLabel + '_prec'] = prec_metricResults[classLabel]
#         df_Overal_precRecall[classLabel + '_recall'] = recall_metricResults[classLabel]
#         df_Overal_f1acc[classLabel + '_f1-score'] = f1_metricResults[classLabel]
#         df_Overal_f1acc[classLabel + '_acc'] = acc_metricResults[classLabel]
#
#     df_Overal_precRecall.to_csv(str(testgroup)+'AllAlgorithmsprecRecall.csv' )
#     df_Overal_f1acc.to_csv(str(testgroup) + 'AllAlgorithmsf1acc.csv')
#



for evacreteria in ['AllAlgorithms']:
    testgroup=31
    sourceData=pd.read_csv(str(31)+evacreteria+'.csv',index_col = 0)
    Sum_temp=sourceData
    count=1
    for testgroup in [32,33,34,35,36]:
        sourceData2=pd.read_csv(str(testgroup)+evacreteria+'.csv',index_col = 0)
        temp2=sourceData2[['acc']]
        Sum_temp=Sum_temp+temp2
        count=count+1
    OverallResult=Sum_temp/count
    OverallResult.to_csv('ZLMOverallResult_'+evacreteria+'.csv')
