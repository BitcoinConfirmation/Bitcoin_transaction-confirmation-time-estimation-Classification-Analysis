import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import G_Variables_WWWJ




ClassList=[2,4,6,8]

alg_List =['finalMLPSimpleLayers','finalAttentionsAdvSimpleLayers',
     'finalAttentionsWhtSimpleLayers',
       'finalAttentionsQVKSimpleLayers',
     'finalLSTMSimpleLayers','TransformerSimpleLayers','randomForest','xgBoost','deepForest','xgBoostAdv0Layer','xgBoostLSTM0Layer','xgBoostLSTM0LayerSimpleLayers']

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
sourceData=pd.read_csv(str(31)+'AllAlgorithmsSelected.csv',index_col = 0)
Sum_temp=sourceData[['acc','prec_macro','recall_macro','f1_macro']]
count=1
for testgroup in [32,33,34,35,36]:
    sourceData2=pd.read_csv(str(testgroup)+'AllAlgorithmsSelected.csv',index_col = 0)
    temp2=sourceData2[['acc','prec_macro','recall_macro','f1_macro']]
    Sum_temp=Sum_temp+temp2
    count=count+1
OverallResult=Sum_temp/count
OverallResult.to_csv('OverallResult_AllAlgorithmsSelected.csv')

print('hello')
tempvalue=sourceData.values
tempvalue2=sourceData2.values
sumed=tempvalue+tempvalue2
sumed=sumed[:,[0,1,4,7]]


evaluationList=['acc','prec_macro','recall_macro','f1_macro']
modelList= ['MLPSimpleLayers','AdvSimpleLayers','WhtSimpleLayers','QVKSimpleLayers','LSTMSimpleLayers','TransformerSimpleLayers','randomForest','xgBoost','deepForest','xgBoostAdv0Layer','xgBoostLSTM0Layer','xgBoostLSTM0LayerSimpleLayers']



for label in evaluationList:
    plt.figure()
    #for modelname in ['MLP','Adv','Wht','LSTM','Transformer','randomForest','xgBoost','xgBoostAdv0Layer','xgBoostLSTM0Layer']:
    Overall_df = pd.DataFrame(columns=['k=8','k=6','k=4','k=2'])
    for modelname in modelList:

        model_result=[]
        for classlabel in [8,6,4,2]:
            model_result.append(OverallResult.loc[modelname+'_'+str(classlabel),label])
        plt.plot(model_result,label=modelname)
        Overall_df.loc[modelname]=model_result
        Overall_df.to_csv('./AllAlgorithm/'+label+ '_AllAlgorithmsSelected.csv')
    plt.legend()
    plt.savefig('./AllAlgorithm/'+label+'AllAlgorithmsSelected.pdf')




for classlabel in ClassList:

    plt.figure()
    #for modelname in ['MLP','Adv','Wht','LSTM','Transformer','randomForest','xgBoost','xgBoostAdv0Layer','xgBoostLSTM0Layer']:
    Overall_df = pd.DataFrame(columns=['k=8','k=6','k=4','k=2'])
    for modelname in modelList:

        model_result=[]
        for classlabel in [8,6,4,2]:
            model_result.append(OverallResult.loc[modelname+'_'+str(classlabel),label])
        plt.plot(model_result,label=modelname)
        Overall_df.loc[modelname]=model_result
        Overall_df.to_csv('./AllAlgorithm/'+label+ '_AllAlgorithmsSelected.csv')
    plt.legend()
    plt.savefig('./AllAlgorithm/'+label+'AllAlgorithmsSelected.pdf')
