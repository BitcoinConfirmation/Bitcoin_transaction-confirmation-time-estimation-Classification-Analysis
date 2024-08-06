from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
import numpy as np
import seaborn as sn

# 支持中文字体显示, 使用于Mac系统

y_test = [1,1,2,1,3]

y_pred = [1,2,2,1,3]

def GenerateConfusionMatrix(sourcedir,classNum, testgroup,cmpBackground):
    sourcefile=sourcedir+'Confusion_Figs/'+str(testgroup)+'_'+str(classNum)+'Confusion.npy'
    temp = np.load(sourcefile, allow_pickle=True)
    y_test = temp.item()['y_true']
    y_pred = temp.item()['Y_pred']
    classes = [str(x+1) for x in range(classNum)]
    confusion = confusion_matrix(y_true=y_test, y_pred=y_pred,labels=[x for x in range(classNum)])
    indices = range(len(confusion))

    plt.figure()
    sn.heatmap(confusion, annot=True,cmap=cmpBackground,fmt='g',annot_kws={'size':13},cbar=False)

    plt.xticks(indices, classes,fontsize=16 )
    plt.yticks(indices, classes,fontsize=16)
    plt.xlabel('Predicted',fontsize=16)
    plt.ylabel('Expected',fontsize=16)
    abb_file = sourcedir.split("_")
    algName=abb_file[2]
    plt.savefig('./ResultsMatrix/'+algName+str(testgroup)+'_'+str(classNum)+'Confusion.pdf',bbox_inches = 'tight')


sourcedir='../WWWJ_RW08_LSTMXgBoostLayers_0_ForConfusion/'
testgroup=33
classNum=6

for testgroup in [33]:
    for classNum in [2,4,6,8]:
        sourcedir = '../WWWJ_RW08_LSTMXgBoostLayers_0_ForConfusion/'
        GenerateConfusionMatrix(sourcedir,classNum, testgroup,'Reds')
        sourcedir2 = '../WWWJ_RW08_xgBoostTime_ForConfusion/'
        GenerateConfusionMatrix(sourcedir2,classNum, testgroup,'Blues')
