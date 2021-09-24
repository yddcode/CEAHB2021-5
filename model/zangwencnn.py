'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
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
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_size = 64
num_classes = 13
epochs = 40

# input image dimensions
img_rows, img_cols = 128, 128

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data
# D:\vscode\vscodework\zangwen
images = []
for fn in os.listdir('D:/vscode/vscodework/zangwen/train1'):
    if fn.endswith('.png'):
        fd = os.path.join('D:/vscode/vscodework/zangwen/train1',fn)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)

y = np.loadtxt('D:/vscode/vscodework/zangwen/out.txt')
print (y.shape)

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
#* 2-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(lr=0.0001, ),
#               metrics=['accuracy'])
# sgd = SGD(decay=0.001,momentum=0.9,nesterov=True)  
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
H = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))

acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs,acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.savefig('./trainacc.png', dpi = 400)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./tloss.png', dpi = 400)
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
print('y_test: %s \n, pred: %s \n', y_test, pred)
y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
print('y_test1, pred:  \n', y_test1, pred)
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/2-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)

# 1-1-1 adam
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/1-1-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)

# 3) 2-1-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/2-1-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)
# 3-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/3-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)

# 4) 2-2-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/2-2-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)
# 4-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/4-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)

# 5) 2-2-1-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/2-2-1-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)
# 5-1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict(X_test)
# y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)
# confusion_matrix = confusion_matrix(y_test1, pred)

output = sys.stdout
outputfile = open("./zangwen/5-1-adam_cnn.txt","a")
sys.stdout = outputfile
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Stacked Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print('Stacked Test Accuracy: %.5f  \n' % acc)
print('Stacked Test f1 score: %.5f  \n' % f1)
print('Stacked Test recall score: %.5f  \n' % recall)