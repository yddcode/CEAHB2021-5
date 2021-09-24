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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
import keras
from keras import optimizers, regularizers # 优化器，正则化项
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
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
			# save model to file snapshot1_model_567_
			filename = "snapshot1_model_567_%d.h5" % int((epoch + 1) / epochs_per_cycle)
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
# dissimilarity  homogeneity
# data_train = np.loadtxt(open("E:/homogeneity.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
# X_train, y_train = data_train[:, :-1], data_train[:, -1]
# trainX, testX, trainy, testy = train_test_split(X_train, y_train, test_size = 0.3, random_state = 3)
# print(trainX.shape, testX.shape, trainy.shape)

# define model
# model = Sequential()
# trainy_enc = to_categorical(trainy)
# testy_enc = to_categorical(testy)
# print(trainX.shape, testX.shape, trainy.shape)
# model.add(Dense(96, input_dim=40, activation='relu'))
# model.add(Dense(2, activation='softmax'))
# opt = SGD(momentum=0.9)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


batch_size = 8
num_classes = 5
epochs = 1

# input image dimensions
img_rows, img_cols = 380, 380

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    im = im.resize((380, 380))
    data = np.array(im)
    return data
# D:\vscode\vscodework\zangwen
images = []
for fn in os.listdir('D:/guji_resizedata510'):
    if fn.endswith('.png') or fn.endswith('.jpg'):
        fd = os.path.join('D:/guji_resizedata510',fn)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)

y = np.loadtxt('E:/gujilabel.txt')
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 3)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
trainy_enc = keras.utils.to_categorical(y_train, num_classes)
testy_enc = keras.utils.to_categorical(y_test, num_classes)

# create snapshot ensemble callback
n_epochs = 200
n_cycles = n_epochs / 20
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.0001)

# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',  kernel_initializer='uniform', padding='same', input_shape=(img_rows, img_cols, 1)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',  kernel_initializer='uniform', padding='same'))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
# # fit model
# model.fit(X_train, trainy_enc, validation_data=(X_test, testy_enc), epochs=n_epochs, verbose=2, callbacks=[ca])

batch_size = 8
epochs = 1

def addmodel():
	input0 = keras.layers.Input(shape=(img_cols, img_rows, 1))
	x0 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(input0)
	x0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x0)
	x0 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x0)
	x0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x0)
	x0 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x0)
	x0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x0)
	x0 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x0)
	x0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x0)
	x0 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x0)
	x0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x0)
	x0 = Flatten()(x0)
	x0 = Dense(128, activation='relu')(x0)

	input1 = keras.layers.Input(shape=(img_cols, img_rows, 1))
	x1 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(input1)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)
	x1 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)
	x1 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)
	x1 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)
	x1 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)
	x1 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)
	x1 = Flatten()(x1)
	x1 = Dense(128, activation='relu')(x1)

	input2 = keras.layers.Input(shape=(img_cols, img_rows, 1))
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(input2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
	x2 = Flatten()(x2)
	x2 = Dense(128, activation='relu')(x2)
	# 相当于 added = keras.layers.add([x1, x2])
	added = keras.layers.Add(name='add1')([x0, x1, x2])  
	# added = keras.layers.concatenate([x0, x1, x2], name='add1', axis=-1)
	# Dropout(0.5, name='dense1')

	out = keras.layers.Dense(num_classes, activation='softmax')(added)
	model = keras.models.Model(inputs=[input0, input1, input2], outputs=out)
	return  model
# model.summary()
# model.add(Dense(num_classes, activation='softmax'))
#######
# model = addmodel()
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(lr=0.0001),
#               metrics=['accuracy'])
# model.fit([X_train, X_train, X_train], trainy_enc,
#               batch_size=batch_size,
#               epochs=n_epochs,
#               verbose=1,
#               validation_data=([X_test, X_test, X_test], testy_enc), callbacks=[ca])

# # model.fit(X_train, trainy_enc, validation_data=(X_test, testy_enc), epochs=n_epochs, verbose=2, callbacks=[ca])
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'snapshot1_model_567_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
 
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict([testX, testX, testX]) for model in members]
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
	return accuracy_score(testy, yhat), precision_score(testy, yhat, average='macro'), recall_score(testy, yhat, average='macro'), f1_score(testy, yhat, average='macro')

# precision_score(y_test1, pred, average='macro')
# acc = accuracy_score(y_test1, pred)
# f1 = f1_score(y_test1, pred, average='macro')
# recall = recall_score(y_test1, pred, average='macro')
# classify_report = classification_report(y_test1, pred, digits=4)

# generate 2d classification dataset
# X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# # split into train and test
# n_train = 100
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# print(trainX.shape, testX.shape)
# load models in order
members = load_all_models(10)
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores, ps, rs, fs = list(), list(), list(), list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score, p, r, f = evaluate_n_members(members, i, X_test, y_test)
	# evaluate the i'th model standalone
	# testy_enc = to_categorical(testy)
	_, single_score = members[i-1].evaluate([X_test, X_test, X_test], testy_enc, verbose=0, batch_size=8)
	# summarize this step
	print('> %d: single = %.5f, ensemble = %.5f, %.5f, %.5f, %.5f' % (i, single_score, ensemble_score, p, r, f))
	ensemble_scores.append(ensemble_score)
	ps.append(p)
	rs.append(r)
	fs.append(f)
	single_scores.append(single_score)
# summarize average accuracy of a single final model
print('single Accuracy %.5f (%.5f)' % (mean(single_scores), std(single_scores)))
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='x')
pyplot.show()