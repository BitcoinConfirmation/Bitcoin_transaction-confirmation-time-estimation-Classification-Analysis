import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




ClassList=[2,4,6,8]


# colums_list=['acc','prec_macro','prec_micro','prec_wht','recall_macro','recall_micro','recall_wht','f1_macro','f1_micro','f1_wht']




OverallResult=pd.read_csv('OverallResult_LSTMHybrid_ManualAddOld.csv',index_col = 0)


evaluationList=['acc','prec_macro','recall_macro','f1_macro']
modelList =['LSTM','LSTMSimpleLayers','LSTMOld','xgBoostLSTM0Layer','xgBoostLSTM0LayerSimpleLayers']

modelName_nic=['Lstm$^+$','Lstm','Lstm_prev','HybridLstm$^+$','HybridLstm']
evaluation_nic=['accuracy','precision','recall','f1-score']



style='seaborn-deep'
Markers=['p','p','p','*','*','o','o']
LineStleList=['-','-.',':','-','-.','-','-.']
colorList=['#002c53','#002c53','#002c53','#ffa510','#ffa510','#0c84c6','#0c84c6' ,'#943c39','#943c39']

for eva_inx in range(len(evaluationList)):
    evalabel=evaluationList[eva_inx]

    plt.style.use(style)
    plt.figure(figsize=(5,3))

    classList=[2,4,6,8]

    x = [1, 2, 3, 4]
    name_list = ['k=2', 'k=4', 'k=6', 'k=8']




    width_1 = 0.08

    for model_inx in range(len(modelList)):

        model_result = []
        for classlabel in [2,4,6,8]:
            print(modelList[model_inx])
            model_result.append(OverallResult.loc[modelList[model_inx] + '_' + str(classlabel), evalabel])

        plt.plot( x,model_result, label=modelName_nic[model_inx],marker=Markers[model_inx],linestyle=LineStleList[model_inx],color=colorList[model_inx])
    plt.ylabel(evaluation_nic[eva_inx])
    plt.xlabel('class size')
    plt.xticks(x, name_list)
    plt.legend(  loc=1,borderaxespad=0, fontsize=6.8)
    plt.savefig('./LSTM_figures/NewLSTMAndHbrid'+style+evalabel + 'Comparison.pdf')
    plt.show()
