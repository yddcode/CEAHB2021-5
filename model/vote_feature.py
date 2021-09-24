from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax

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


# dissimilarity  homogeneity
data_train = np.loadtxt(open("E:/dissimilarity.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
X_train, y_train = data_train[:, :-1], data_train[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state= 3)

# fit model on dataset
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
	# define model
	model = Sequential()
	model.add(Dense(96, input_dim=40, activation='relu'))
	# model.add(Dense(96, activation='relu'))  
	# model.add(Dropout(0.5))  
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy_enc, epochs=200, verbose=0)
	return model
 
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	print(len(yhats))
	yhats = array(yhats)
	# print(yhats[0])
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)     # 按列相加  概率相加
	print(summed[:6])
	# argmax across classes
	result = argmax(summed, axis=1)   # 软投票
	print(result[:6])
	return result
 
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat), precision_score(testy, yhat, average='macro'), recall_score(testy, yhat, average='macro'), f1_score(testy, yhat, average='macro')
 
# generate 2d classification dataset
# X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2, random_state=2)
# # split into train and test
# n_train = 200
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# print(trainX.shape, testX.shape)
# fit all models  
n_members = 5
members = [fit_model(X_train, y_train) for _ in range(n_members)]
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores, ps, recalls, f1s = list(), list(), list(), list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score, p, recall, f1 = evaluate_n_members(members, i, X_test, y_test)
	# evaluate the i'th model standalone
	testy_enc = to_categorical(y_test)
	_, single_score = members[i-1].evaluate(X_test, testy_enc, verbose=0)
	# summarize this step
	print('> %d: single=%.5f,%.5f,%.5f,%.5f; ensemble=%.5f' % (i, single_score, p, recall, f1, ensemble_score))
	ensemble_scores.append(ensemble_score)
	ps.append(p)
	recalls.append(recall)
	f1s.append(f1)
	single_scores.append(single_score)
# summarize average accuracy of a single final model
print('Accuracy %.5f (%.5f)' % (mean(single_scores), std(single_scores)))
print('ps %.5f (%.5f)' % (mean(ps), std(ps)))
print('recalls %.5f (%.5f)' % (mean(recalls), std(recalls)))
print('f1s %.5f (%.5f)' % (mean(f1s), std(f1s)))
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='x')
pyplot.show()