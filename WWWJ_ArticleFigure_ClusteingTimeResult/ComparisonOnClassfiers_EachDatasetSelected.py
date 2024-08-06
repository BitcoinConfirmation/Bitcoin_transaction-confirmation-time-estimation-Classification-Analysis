import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import G_Variables_WWWJ




ClassList=[2,4,6,8]
alg_List =['finalMLP','finalMLPSimpleLayers','finalAttentionsAdv','finalAttentionsAdvSimpleLayers',
      'finalAttentionsWht','finalAttentionsWhtSimpleLayers','finalAttentionsQVK',
       'finalAttentionsQVKSimpleLayers',
      'finalLSTM','finalLSTMSimpleLayers','Transformer','TransformerSimpleLayers','randomForest','xgBoost','deepForest','xgBoostAdv0Layer','xgBoostAdv0LayerSimpleLayers','xgBoostAdv1LayerSimpleLayers','xgBoostAdv2LayerSimpleLayers','xgBoostLSTM0Layer','xgBoostLSTM0LayerSimpleLayers','xgBoostLSTM1LayerSimpleLayers','xgBoostLSTM2LayerSimpleLayers']

colums_list=['acc','prec_macro','prec_micro','prec_wht','recall_macro','recall_micro','recall_wht','f1_macro','f1_micro','f1_wht']






def GetMetricResult(testgroup,alg_List):
    testgroup=str(testgroup)
    START_EstimateBlock = getattr(G_Variables_WWWJ, 'START_EstimateBlock_S' + testgroup)

    height = START_EstimateBlock
    df = pd.DataFrame(columns=colums_list)
    for classlabel in ClassList:
        for alg in alg_List:

            sourceDate = pd.read_csv('../WWWJ_OverallPerformance/ClassifiedFeerateBin36Feerate' + str(classlabel)+'ClassTime'+testgroup + alg+str(height)+str(classlabel)+'results.csv',
                                     header=None)
            temp_data = sourceDate.values
            if alg.__contains__('final'):
                alg=alg.replace('final','')
                alg=alg.replace('Attentions', '')
            df.loc[alg + '_' + str(classlabel)] = temp_data[0, -10:].tolist()

    df.to_csv(str(testgroup) + 'AllAlgorithmsSelected.csv')

for testgroup in [31,32,33,34,35,36]:
    testSet=str(testgroup)
    GetMetricResult(testSet,alg_List)
