import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import G_Variables_WWWJ




ClassList=[2,4,6,8]


alg_List =['finalMLPSimpleLayers','finalAttentionsAdvSimpleLayers',
      'finalAttentionsWhtSimpleLayers',
       'finalAttentionsQVKSimpleLayers',
     'finalLSTMSimpleLayers','TransformerSimpleLayers','rotationForest','randomForest','xgBoost','lightGBM','deepForest','deepForestCost','xgBoostAdv0LayerSimpleLayers','xgBoostLSTM0Layer','xgBoostLSTM0LayerSimpleLayers']

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
sourceData=pd.read_csv(str(31)+'AllAlgorithmsSelected_WWWJ.csv',index_col = 0)
Sum_temp=sourceData[['acc','prec_macro','recall_macro','f1_macro']]
count=1
for testgroup in [32,33,34,35,36]:
    sourceData2=pd.read_csv(str(testgroup)+'AllAlgorithmsSelected_WWWJ.csv',index_col = 0)
    temp2=sourceData2[['acc','prec_macro','recall_macro','f1_macro']]
    Sum_temp=Sum_temp+temp2
    count=count+1
OverallResult=Sum_temp/count
OverallResult.to_csv('OverallResult_AllAlgorithmsSelected_WWWJ.csv')

print('hello')
tempvalue=sourceData.values
tempvalue2=sourceData2.values
sumed=tempvalue+tempvalue2
sumed=sumed[:,[0,1,4,7]]


evaluationList=['acc','prec_macro','recall_macro','f1_macro']
modelList= ['MLPSimpleLayers','AdvSimpleLayers','WhtSimpleLayers','QVKSimpleLayers','LSTMSimpleLayers','TransformerSimpleLayers','rotationForest','randomForest','xgBoost','lightGBM','deepForest','deepForestCost','xgBoostAdv0LayerSimpleLayers','xgBoostLSTM0Layer','xgBoostLSTM0LayerSimpleLayers']



evas_in_Table=['prec_macro','recall_macro']
evas_in_Table2=['f1_macro','acc']

def GenerateFigure(evas_in_Table):
    cols=[]
    for k in ClassList:
        for eva in evas_in_Table:
            colName=str(k)+eva
            cols.append(colName)
    # plt.figure()
    #for modelname in ['MLP','Adv','Wht','LSTM','Transformer','randomForest','xgBoost','xgBoostAdv0Layer','xgBoostLSTM0Layer']:
    Overall_df = pd.DataFrame(columns=cols)
    for modelname in modelList:

        model_result=[]
        for classlabel in [2,4,6,8]:
            for eva_inx in range(2):
                eva=evas_in_Table[eva_inx]
                model_result.append(OverallResult.loc[modelname+'_'+str(classlabel),eva])
        # plt.plot(model_result,label=modelname)
        Overall_df.loc[modelname]=model_result
    Overall_df.to_csv('./AllAlgorithm/'+evas_in_Table[0]+evas_in_Table[0]+ '_Summary_WWWJ.csv')

GenerateFigure(evas_in_Table)
GenerateFigure(evas_in_Table2)

