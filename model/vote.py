from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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

batch_size = 4
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

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

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
y_test = keras.utils.to_categorical(y_test, num_classes)

def Cnnm():
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
    H = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    return model

def Cnnm1():
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
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
    H = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    return model
cnnm = Cnnm()
cnnm1 = Cnnm1()
eclf = VotingClassifier(
    estimators=[('cnn1', cnnm1), ('cnn', cnnm)],
    voting='hard')

for clf, label in zip([eclf], ['Ensemble']):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print('score', scores)