from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from itertools import product

import keras
from keras import optimizers, regularizers # 优化器，正则化项
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
from keras.optimizers import SGD, Adam, Adadelta
import os
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
 
# fit model on dataset
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
	# define model 
	model = Sequential()
	model.add(Dense(196, input_dim=256, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy_enc, epochs=100, verbose=0)
	return model
 
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# weighted sum across ensemble members
	summed = tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
	result = argmax(summed, axis=1)
	return result
 
# evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
	# make prediction
	yhat = ensemble_predictions(members, weights, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat)
 
# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result
 
# grid search weights
def grid_search(members, testX, testy):
	# define weights to consider
	w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	best_score, best_weights = 0.0, None
	# iterate all possible combinations (cartesian product)
	for weights in product(w, repeat=len(members)):
		# skip if all weights are equal
		if len(set(weights)) == 1:
			continue
		# hack, normalize weight vector
		weights = normalize(weights)
		# evaluate weights
		score = evaluate_ensemble(members, weights, testX, testy)
		if score > best_score:
			best_score, best_weights = score, weights
			print('>%s %.5f' % (best_weights, best_score))
	return list(best_weights)
 
# generate 2d classification dataset
# X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2, random_state=2)
# # split into train and test
# n_train = 200
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# print(trainX.shape, testX.shape)

# dissimilarity  homogeneity
# data_train = np.loadtxt(open("D:/vscode/vscodework/Python-Image-feature-extraction/energy.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
# X_train, y_train = data_train[:, :-1], data_train[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.5, random_state = 3)

# data_train = np.loadtxt(open("D:/vscode/vscodework/Python-Image-feature-extraction/dissimilarity.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
# X_train, y_train = data_train[:, :-1], data_train[:, -1]
# trainXdis, testXdis, trainydis, testydis = train_test_split(X_train, y_train, test_size = 0.5, random_state = 3)
# print(trainydis[:10], testydis[:10])
# X_train = np.hstack((trainXdis, X_train))
# X_test = np.hstack((testXdis, X_test))

data_train = np.loadtxt(open("D:/vscode/vscodework/zangwen/lbptrain.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
X_train, y_train = data_train[:, :-1], data_train[:, -1]
data_test = np.loadtxt(open("D:/vscode/vscodework/zangwen/lbptest.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
X_test, y_test = data_test[:, :-1], data_test[:, -1]



# fit all models
n_members = 10
members = [fit_model(X_train, y_train) for _ in range(n_members)]
# evaluate each single model on the test set
testy_enc = to_categorical(y_test)
for i in range(n_members):
	_, test_acc = members[i].evaluate(X_test, testy_enc, verbose=0)
	print('Model %d: %.5f' % (i+1, test_acc))
# evaluate averaging ensemble (equal weights)
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights, X_test, y_test)
print('Equal Weights Score: %.5f' % score)
# grid search weights
weights = grid_search(members, X_test, y_test)
score = evaluate_ensemble(members, weights, X_test, y_test)
print('Grid Search Weights: %s, Score: %.5f' % (weights, score))