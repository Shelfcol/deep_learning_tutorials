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

#分开并打乱数据
X_train,X_test,Y_train,Y_test=train_test_split(
	iris_X,
	iris_Y,
	test_size=0.3
	)

# print(Y_train)
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)

print(knn.predict(X_test))
print(Y_test)