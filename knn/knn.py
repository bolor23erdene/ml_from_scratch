import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
	return np.sqrt(np.sum((x1-x2)**2))



class KNN:

	def __init__(self, k=3):
		self.k = k


	def fit(self, X, Y):
		self.X_train = X
		self.Y_train = Y


	def predict(self, X):
		predictions = [self.predict_f(x) for x in X]

		return predictions

	def predict_f(self, x1):

		# compute distances
		distances = [euclidean_distance(x1, x) for x in self.X_train]

		# take the top k points
		top_k = np.argsort(distances)[:self.k]
		top_k_labels = [self.Y_train[indice] for indice in top_k]

		#print(top_k_labels)

		# vote 
		most_common_vote = Counter(top_k_labels).most_common(1)

		#print(most_common_vote[0][0])

		return most_common_vote[0][0]


