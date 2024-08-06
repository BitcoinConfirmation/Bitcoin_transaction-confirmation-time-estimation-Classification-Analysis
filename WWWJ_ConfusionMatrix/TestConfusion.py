from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

# 支持中文字体显示, 使用于Mac系统

y_test = [1,1,2,1]

y_pred = [1,2,2,1]

classes = [1,2]
confusion = confusion_matrix(y_true=y_test, y_pred=y_pred)


# 绘制热度图
plt.imshow(confusion, cmap=plt.cm.Reds)
indices = range(len(confusion))
plt.xticks(indices, classes )
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('y_pred')
plt.ylabel('y_true')

# 显示数据
for first_index in range(len(confusion)):
  for second_index in range(len(confusion[first_index])):
    plt.text(first_index, second_index, confusion[first_index][second_index])

# 显示图片
plt.show()
plt.savefig('testConfusion.pdf')