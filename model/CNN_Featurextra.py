# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:12:24 2019

@author: HP
"""

# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import time
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, precision_score


time_start = time.time()
# batch_size = 128
# num_classes = 81
# epochs = 60
# img_rows, img_cols = 61, 61


# def read_image(img_name):
#     im = Image.open(img_name).convert('L')
#     data = np.array(im)
#     return data
# images = []
# for fn in os.listdir('Mtrain'):
#     if fn.endswith('.png'):
#         fd = os.path.join('Mtrain',fn)
#         images.append(read_image(fd))
# print('load success!')
# X_train = np.array(images)
# print(X_train.shape)
# y_train = np.loadtxt('Mtrain.txt')
# print(y_train.shape)
# images1 = []
# for fn in os.listdir('Mtest'):
#     if fn.endswith('.png'):
#         fd = os.path.join('Mtest',fn)
#         images1.append(read_image(fd))
# print('load success!')
# X_test = np.array(images1)
# print(X_test.shape)
# y_test = np.loadtxt('Mtest.txt')
# print(y_test.shape)

import os, cv2
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_size = 16
num_classes = 13
epochs = 20

# input image dimensions
img_rows, img_cols = 128, 128

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data
# D:/vscode/vscodework/zangwen/train

images = []
for fn in os.listdir('E:/guji_resizedata510'):
    if fn.endswith('.png') or fn.endswith('.jpg'):
        fd = os.path.join('E:/guji_resizedata510/',fn)
        # img = cv2.imread(fd)
        # print(img.shape)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)

y = np.loadtxt('D:/vscode/vscodework/zangwen/out.txt')
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 3)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

# np.savetxt('zangwen/train.csv', y_train, delimiter=',')
# np.savetxt('zangwen/test.csv', y_test, delimiter=',')
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
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(96, (5, 5), strides=(1, 1), input_shape=(img_rows, img_cols, 1), padding='same', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5, name='dense1'))
model.add(Dense(num_classes, activation='softmax'))

# def model_AlexNet(trainX, trainy):
# define model
# model = Sequential()
# model.add(Conv2D(96, (11, 11), strides=(1, 1), input_shape=(img_rows, img_cols, 1), padding='same', activation='relu',
#                 kernel_initializer='uniform'))
# # 池化层
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
# model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# #使用池化层，步长为2
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# # 第三层卷积，大小为3x3的卷积核使用384个
# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# # 第四层卷积,同第三层
# model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# # 第五层卷积使用的卷积核为256个，其他同上
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# # model.add(Dense(10, activation='softmax'))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
# # sgd = SGD(decay=0.001,momentum=0.9,nesterov=True)  
# # model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
# # fit model
# model.fit(trainX, trainy, verbose=2, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
# return model
model.summary()
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(lr=0.0001),
#               metrics=['accuracy'])
# H = model.fit(X_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(X_test, y_test))

# score = model.evaluate(X_test, y_test, verbose=0)
# pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
# pred = [np.argmax(one_hot)for one_hot in pred]
# precision = precision_score(y_test1, pred, average='macro')
# acc = accuracy_score(y_test1, pred)
# f1 = f1_score(y_test1, pred, average='macro')
# recall = recall_score(y_test1, pred, average='macro')
# classify_report = classification_report(y_test1, pred, digits=4)

# output = sys.stdout
# outputfile = open("zangwen/20cnn2.txt","a")
# sys.stdout = outputfile
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print(' Test classify_report : \n', classify_report)
# # print('Stacked Test confusion_matrix : \n', confusion_matrix)
# print(' Test Accuracy: %.5f  \n' % acc)
# print(' Test f1 score: %.5f  \n' % f1)
# print(' Test recall score: %.5f  \n' % recall)
# print(' Test precision: %.5f  \n' % precision)
# # define the layer for feature extraction
# intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('dense1').output)
# # get engineered features for training and validation
# feature_engineered_train = intermediate_layer.predict(X_train)
# feature_engineered_train = pd.DataFrame(feature_engineered_train)
# feature_engineered_train.to_csv('zangwen/20cnn2feature_train.csv')
# feature_engineered_test = intermediate_layer.predict(X_test)
# feature_engineered_test = pd.DataFrame(feature_engineered_test)
# feature_engineered_test.to_csv('zangwen/20cnn2feature_test.csv')


