import numpy as np
import pandas as pd

ages = np.array([1,5,10,40,36,12,58,62,77,89,100,18,20,25,30,32]) #年龄数据
# s=pd.cut(ages, [0,5,20,30,50,100], labels=[u"婴儿",u"青年",u"中年",u"壮年",u"老年"])
s=pd.cut(ages, [0,5,20,30,50,100])
#将ages分为了5个区间(0, 5],(5, 20],(20, 30],(30,50],(50,100].
print(s.value_counts().values)