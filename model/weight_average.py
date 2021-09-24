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
# from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

batch_size = 8
num_classes = 13
epochs = 1

# input image dimensions
img_rows, img_cols = 128, 128

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data
# D:\vscode\vscodework\zangwen
images = []
for fn in os.listdir('D:/vscode/vscodework/zangwen/train'):
    if fn.endswith('.png'):
        fd = os.path.join('D:/vscode/vscodework/zangwen/train',fn)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)

y = np.loadtxt('D:/vscode/vscodework/zangwen/out.txt')
print (y.shape)

# iris = datasets.load_iris()
# X, y = iris.data[:, 1:3], iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 30)
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
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test_enc = keras.utils.to_categorical(y_test, num_classes)

def fit_model(X_train, y_train):
    model = Sequential()
    model.add(Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(Conv2D(384, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5, name='dense1'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.0001),
                metrics=['accuracy'])
    H = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test_enc))
    return model
 
# fit model on dataset
# def fit_model(trainX, trainy):
# 	trainy_enc = to_categorical(trainy)
# 	# define model
# 	model = Sequential()
# 	model.add(Dense(25, input_dim=2, activation='relu'))
# 	model.add(Dense(3, activation='softmax'))
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# fit model
# 	model.fit(trainX, trainy_enc, epochs=5, verbose=1)
# 	return model
 
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
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
	print(testy[:6], yhat[:6])
	return accuracy_score(testy, yhat)
 
# generate 2d classification dataset
# X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2, random_state=2)
# # split into train and test
# n_train = 200
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
# print(trainX.shape, testX.shape)
# fit all models  
n_members = 6
members = [fit_model(X_train, y_train) for _ in range(n_members)]
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, X_test, y_test)
	# evaluate the i'th model standalone
	testy_enc = to_categorical(y_test)
	_, single_score = members[i-1].evaluate(X_test, testy_enc, verbose=0)
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