from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras import backend
from math import pi
from math import cos
from math import floor
import keras
from matplotlib import pyplot
 
# define custom learning rate schedule
class CosineAnnealingLearningRateSchedule(Callback):
	# constructor
	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
		self.epochs = n_epochs
		self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()
 
	# calculate learning rate for an epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)
 
	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs=None):
		# calculate learning rate
		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
		# set learning rate
		backend.set_value(self.model.optimizer.lr, lr)
		# log value
		self.lrates.append(lr)
 
# generate 2d classification dataset
# X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# # one hot encode output variable
# y = to_categorical(y)
# # split into train and test
# n_train = 100
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# # define model
# model = Sequential()
# model.add(Dense(96, input_dim=40, activation='relu'))
# model.add(Dense(2, activation='softmax'))
# opt = SGD(momentum=0.9)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# # define learning rate callback
# n_epochs = 400
# n_cycles = n_epochs / 50
# ca = CosineAnnealingLearningRateSchedule(n_epochs, n_cycles, 0.01)
# # fit model
# history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=n_epochs, verbose=0, callbacks=[ca])
# # evaluate the model
# _, train_acc = model.evaluate(trainX, trainy, verbose=0)
# _, test_acc = model.evaluate(testX, testy, verbose=0)
# print('Train: %.5f, Test: %.5f' % (train_acc, test_acc))
# # plot learning rate
# pyplot.plot(ca.lrates)
# pyplot.show()
# # learning curves of model accuracy
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()

from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras import backend
from math import pi
from math import cos
from math import floor

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy as np
from numpy import array
from numpy import argmax
 
# snapshot ensemble with custom learning rate schedule
class SnapshotEnsemble(Callback):
	# constructor
	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
		self.epochs = n_epochs
		self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()
 
	# calculate learning rate for epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)
 
	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs={}):
		# calculate learning rate
		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
		# set learning rate
		backend.set_value(self.model.optimizer.lr, lr)
		# log value
		self.lrates.append(lr)
 
	# save models at the end of each cycle
	def on_epoch_end(self, epoch, logs={}):
		# check if we can save model
		epochs_per_cycle = floor(self.epochs / self.cycles)
		if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
			# save model to file
			filename = "snapshot_model_dissimilarity_%d.h5" % int((epoch + 1) / epochs_per_cycle)
			self.model.save(filename)
			print('>saved snapshot %s, epoch %d' % (filename, epoch))
 
# generate 2d classification dataset
# X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# # one hot encode output variable
# y = to_categorical(y)
# # split into train and test
# n_train = 100
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# dissimilarity  homogeneity ASM contrast energy  D:/vscode/vscodework/zangwen/doc7feature_train
data_train = np.loadtxt(open("D:/vscode/vscodework/Python-Image-feature-extraction/dissimilarity.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
X_train, y_train = data_train[:, :-1], data_train[:, -1]
trainX, testX, trainydis, testydis = train_test_split(X_train, y_train, test_size = 0.5, random_state = 3)
print(trainydis[:10], testydis[:10])
# doc7feature_train
data_train = np.loadtxt(open("D:/vscode/vscodework/zangwen/multi_feature_train.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
trainX1, trainy = data_train[:, :-1], data_train[:, -1]
data_test = np.loadtxt(open("D:/vscode/vscodework/zangwen/multi_feature_test.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
testX1, testy = data_test[:, :-1], data_test[:, -1]

# data_train = np.loadtxt(open("D:/vscode/vscodework/zangwen/lbptrain.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
# X_trainlbp, y_train = data_train[:, :-1], data_train[:, -1]
# data_test = np.loadtxt(open("D:/vscode/vscodework/zangwen/lbptest.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
# X_testlbp, y_test = data_test[:, :-1], data_test[:, -1]

# trainX = np.hstack((trainXdis, trainX))
# testX = np.hstack((testXdis, testX))
# trainX = np.hstack((X_trainlbp, trainX))
# testX = np.hstack((X_testlbp, testX))
print(trainy[:10], testy[:10])
print(trainX.shape, testX.shape, trainy.shape)

# define model
model = Sequential()
trainy_enc = to_categorical(trainy)
testy_enc = to_categorical(testy)
print(trainX.shape, testX.shape, trainy.shape)
model.add(Dense(96, input_dim=40, activation='relu'))
model.add(Dense(5, activation='softmax'))
opt = SGD(learning_rate=0.0001, momentum=0.96)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
# create snapshot ensemble callback
n_epochs = 600
n_cycles = n_epochs / 200
batch_size = 128
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.0001)
# fit model
model.fit(trainX, trainy_enc, batch_size=batch_size, validation_data=(testX, testy_enc), epochs=n_epochs, verbose=0, callbacks=[ca])
 
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'snapshot_model_dissimilarity_' + str(i + 1) + '.h5'
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
# X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# # split into train and test
# n_train = 100
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# print(trainX.shape, testX.shape)
# load models in order
members = load_all_models(3)
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, testX, testy)
	# evaluate the i'th model standalone
	# testy_enc = to_categorical(testy)
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
pyplot.plot(x_axis, ensemble_scores, marker='x')
pyplot.show()