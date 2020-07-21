from knn import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
from collections import Counter

def accuracy(y_true, y_pred):
    cnt = 0
    for i in range(len(y_true)):
    	if y_true[i] == y_pred[i]:
    		cnt = cnt + 1

    return cnt/len(y_true)*100

iris = datasets.load_iris()
X, Y = iris.data, iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


k = 5
classifier = KNN(k=k)

classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

Y_test = list(Y_test)

print("custom KNN classification accuracy", accuracy(Y_test, predictions))
