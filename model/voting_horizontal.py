from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from os import makedirs

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


# dissimilarity  homogeneity
data_train = np.loadtxt(open("E:/homogeneity.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
X_train, y_train = data_train[:, :-1], data_train[:, -1]
trainX, testX, trainy, testy = train_test_split(X_train, y_train, test_size = 0.3, random_state= 3)


# generate 2d classification dataset
# X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2, random_state=2)
# # one hot encode output variable
# y = to_categorical(y)
# # split into train and test
# n_train = 200
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)
# define model
model = Sequential()
model.add(Dense(96, input_dim=40, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# create directory for models
# makedirs('models')
# fit model
n_epochs, n_save_after = 100, 80
for i in range(n_epochs):
	# fit model for a single epoch
	trainy_enc = to_categorical(trainy)
	model.fit(trainX, trainy_enc, epochs=1, verbose=0)
	# check if we should save the model
	if i >= n_save_after:
		model.save('votingmodel_' + str(i) + '.h5')

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
 
# load models from file
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
		# define filename for this ensemble
		filename = 'votingmodel_' + str(epoch) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
 
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result
 
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat)
 
# generate 2d classification dataset
# X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2, random_state=2)
# # split into train and test
# n_train = 200
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# print(trainX.shape, testX.shape)
# load models in order
members = load_all_models(80, 100)
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, testX, testy)
	# evaluate the i'th model standalone
	testy_enc = to_categorical(testy)
	_, single_score = members[i-1].evaluate(testX, testy_enc, verbose=0)
	# summarize this step
	print('> %d: single=%.5f, ensemble=%.5f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# summarize average accuracy of a single final model
print('Accuracy %.5f (%.5f)' % (mean(single_scores), std(single_scores)))
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()