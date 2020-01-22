import numpy as np 
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

boston=datasets.load_boston()

X=boston.data
Y=boston.target

# model=sklearn.linear_model.LinearRegression #不行
model=LinearRegression()
model.fit(X,Y)
# print(X.shape)
# print(Y.shape)
# # plt.scatter(Y)
# plt.scatter(Y,model.predict(X))
# plt.show()

print(model.coef_)
print(model.intercept_)
print(model.score(X,Y))#对比真实数据和预测数据的准确度