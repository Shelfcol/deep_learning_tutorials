from sklearn.model_selection import learning_curve#可视化学习过程，降低学习误差
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np 

digits=load_digits()
X=digits.data
Y=digits.target
#learning_curve 可视化模型是否过拟合
train_sizes,train_loss,test_loss=learning_curve(
	SVC(gamma=0.001),X,Y,cv=10,scoring='neg_mean_squared_error',
	train_sizes=[0.1,0.25,0.5,0.75,1])
#train_size用于产生learning_curve的样本数量，比如[0.1,0.25,0.5,0.75,1]
#就是当样本是总样本数量的10%,25%,…100%时产生learning_curve，其实就是对应折
#线图上那几个点的横坐标（见下图），因为样本数量很多，因此都设置比例，当然你也可以
#直接设置样本数量，默认是np.linspace(0.1, 1.0, 5)。

train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='Training')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='Cross-validation')
plt.xlabel('Training example')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
