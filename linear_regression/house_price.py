import csv
import random
import math
import operator
import pandas as pd
import numpy as np



data = pd.read_csv('./USA_Housing.csv')
#print(data)

data = data.drop(['Address'],axis=1)
#print(data)

y = data['Price']
X = data.drop(['Price'],axis=1)

#print(X.shape,y.shape)


def normalize(X):

	features = X.columns

	for feature in features:
		minimum = X[feature].min()
		maximum = X[feature].max()

		for i in range(X.shape[0]):
			X.iloc[i][feature] = (X.iloc[i][feature]-minimum)/(maximum-minimum)

	return X
X = normalize(X)


X_np = np.array(X)
y_np = np.array(y)

W = ((X_np.transpose().dot(X_np))).dot(X_np.transpose()).dot(y_np)
y_pred = X_np.dot(W)

error = (y-y_pred)
print(np.linalg.norm(error,2)) ### 2.4 e^16


w = np.array([0]*5)
w = np.reshape(w, (5,1))
lr = 0.001
epochs = 5

print(y.shape)
print(y)
y = np.array(y)
y = np.reshape(y,(5000,1))
y = y.transpose()
print(y.shape)

X_np = np.array(X)
y_np = np.array(y)

w = w.transpose()

for e in range(epochs):

	#print(((w.dot(X) - y.transpose()).dot(X)), "GRADIENT")
	w_x = w.dot(X.transpose())
	#print(w_x.shape,y.shape)

	#print(y.shape)

	#print(w_x.shape, y.shape)
	w = w - lr*(w_x - y).dot(X)



	#W = ((X_np.transpose().dot(X_np))).dot(X_np.transpose()).dot(y_np)
	
	#print(X_np.shape,w.shape)
	y_pred = X_np.dot(w.transpose())

	#print(y_pred, y)
	error = (y-y_pred)
	print("ERROR",np.linalg.norm(error,2)) ### 2.4 e^16
	
	#print( - )
	#w = w - lr




