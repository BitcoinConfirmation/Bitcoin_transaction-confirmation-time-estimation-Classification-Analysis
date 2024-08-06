from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score

# 支持中文字体显示, 使用于Mac系统

y_test = [1,1,2,1,3]

y_pred = [1,2,2,1,3]

classes = ['Class'+str(x+1) for x in range(3)]
confusion = confusion_matrix(y_true=y_test, y_pred=y_pred)
indices = range(len(confusion))

print(precision_score(y_test,y_pred,average='macro'))
print(recall_score(y_test,y_pred,average='macro'))



import seaborn as sn

# plt.figure()
sn.heatmap(confusion, annot=True,cmap='Blues')

plt.xticks(indices, classes )
plt.yticks(indices, classes)
plt.xlabel('Prediction')
plt.ylabel('True')
plt.savefig('sn_testConfusion.pdf')

