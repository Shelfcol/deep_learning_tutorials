import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt 

X,Y=make_classification(
	n_samples=300,
	n_features=2,
	n_redundant=0,
	n_informative=2,
	random_state=22,#给定一个随机值
	n_clusters_per_class=1,
	scale=100)

# print(X.shape,Y.shape)
# plt.scatter(X[:,0],X[:,1])
# plt.show()
X=preprocessing.scale(X)#归一化之后可以提高回归的准确率
# X=preprocessing.minmax_scale(X,feature_range=(0,1))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
classfier=SVC()
classfier.fit(X_train,Y_train)

score=classfier.score(X_test,Y_test)
print(score)