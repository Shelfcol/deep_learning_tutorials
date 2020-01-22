# -*- coding:utf-8 -*-
import numpy as np 
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris=datasets.load_iris()
keys=iris.keys()
feature_names=iris.feature_names
iris_X=iris.data#每行的数据，一共四列，每一列映射为feature_name对应的值
iris_Y=iris.target#target
# print(feature_names)
# print(iris.target.shape)
# print(iris_X[:2,:])
# print(iris_Y)

'''
#分开并打乱数据
X_train,X_test,Y_train,Y_test=train_test_split(
	iris_X,
	iris_Y,
	test_size=0.2
	)

# print(Y_train)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)

print(knn.score(X_test,Y_test))
'''

#寻找更好的n_neighbors参数，以及利用交叉验证更好地证明模型的准确率
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 

k_range=range(1,31)
k_scores=[]

for k in k_range:
	knn=KNeighborsClassifier(n_neighbors=k)
	# loss=-cross_val_score(knn,iris_X,iris_Y,cv=10,scoring='neg_mean_squared_error')#for regression
	# k_scores.append(loss.mean())

	scores=cross_val_score(knn,iris_X,iris_Y,cv=10,scoring='accuracy')#for calssification
	k_scores.append(scores.mean())

plt.scatter(k_range,k_scores)
plt.xlabel('n_neighbors')
plt.ylabel('loss')
plt.show()
