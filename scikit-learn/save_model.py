from sklearn import svm
from sklearn import datasets
import pickle
from sklearn.externals import joblib

iris=datasets.load_iris()
X=iris.data
Y=iris.target

#method1 pickle
'''
clf=svm.SVC()
clf.fit(X,Y)
#将模型存入pickle文件
with open('clf.pickle','wb') as f:
	pickle.dump(clf,f)
'''
'''
#验证模型
with open('clf.pickle','rb') as f:
	clf2=pickle.load(f)
	print(clf2.predict(X[0:1]))
	print(Y[0:1])#两者一致
'''


#method2 joblib
clf=svm.SVC()
clf.fit(X,Y)
#save
joblib.dump(clf,'clf.pkl')
#restore
clf3=joblib.load('clf.pkl')
print(clf3.predict(X[0:1]))
print(Y[0:1])#两者一致