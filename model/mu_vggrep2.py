import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, precision_score
from keras.datasets import mnist
from keras import optimizers, regularizers # 优化器，正则化项
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, add
from keras.layers import Conv2D,MaxPool2D, GlobalAvgPool2D, MaxPooling2D,Activation,Conv2D,MaxPooling2D,Flatten,ZeroPadding2D,BatchNormalization,AveragePooling2D,concatenate
from PIL import Image
import numpy as np
from keras.initializers import glorot_uniform
from keras.optimizers import SGD, Adam
from keras.layers import Input,Add,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten 
from keras.layers import BatchNormalization,AveragePooling2D,concatenate  
from keras.layers import ZeroPadding2D,add
from keras.layers import Dropout, Activation
from keras.models import Model,load_model
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model
from keras import regularizers
from keras import layers
from keras import backend as K
from keras.optimizers import SGD, Adam, Adadelta
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications import inception_v3
time_start = time.time()
import sys, cv2, os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

num_classes = 5

# input image dimensions
img_rows, img_cols = 196, 196

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    im = im.resize((196, 196))
    data = np.array(im)
    # print(data.shape)
    return data

# D:\vscode\vscodework\zangwen
images = []
for fn in os.listdir('D:/guji_resizedata510'):
    if fn.endswith('.jpg'):
        fd = os.path.join('D:/guji_resizedata510',fn)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)

y = np.loadtxt('E:/gujilabel.txt')
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state= 3)
X_train, X_test, y_train, y_test = X_train[:600], X_test[:600], y_train[:600], y_test[:600]
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
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
batch_size = 1
epochs = 40

input2 = keras.layers.Input(shape=(img_cols, img_rows, 1))
x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(input2)
x_1 = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(input2)
x2 = keras.layers.Add(name='add1')([x2, x_1])  
x_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add2')([x2, x_1, x_3])  
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x_3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(128, (1, 1), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add3')([x_1, x_3])  
x_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(128, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add4')([x2, x_1, x_3])  
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x_3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(256, (1, 1), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add5')([x_1, x_3])  
x_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add6')([x2, x_1, x_3])  
x_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add7')([x2, x_1, x_3])  
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x_3 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add8')([x_1, x_3])  
x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add9')([x2, x_1, x_3])  
x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add10')([x2, x_1, x_3])  
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x_3 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add11')([x_1, x_3])  
x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add12')([x2, x_1, x_3])  
x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add(name='add13')([x2, x_1, x_3])  
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# x2 = SpatialPyramidPooling([1, 2, 4])(x2)
x2 = Flatten()(x2)
x2 = Dense(4096, activation='relu')(x2)
x2 = Dense(4096, activation='relu')(x2)
x2 = Dropout(0.5)(x2)

out = keras.layers.Dense(num_classes, activation='softmax')(x2)
model = keras.models.Model(inputs=input2, outputs=out)
model.summary()
# model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08), # lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True
              metrics=['accuracy'])
H = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
model.save('./zangwen/m_vggrep40eadam_duibi3.h5')
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# with CustomObjectScope({'SpatialPyramidPooling': SpatialPyramidPooling}):
#     model = load_model('./zangwen/m_repvgg40e.h5')
# model = load_model('./zangwen/m_repvgg40e.h5')

accu = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(len(accu))

plt.plot(epochs,accu, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.savefig('./trainvggrep40eadam.png', dpi = 600)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./tlossvggrep40eadam.png', dpi = 600)
# plt.show()

pred = model.predict(X_test, batch_size=8, verbose=1)
y_test1 = [np.argmax(one_hot)for one_hot in y_test]
pred = [np.argmax(one_hot)for one_hot in pred]
precision = precision_score(y_test1, pred, average='macro')
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)

output = sys.stdout
outputfile = open("zangwen/multispp40.txt","a")
sys.stdout = outputfile
print('vggrep 40e duibi3 adam Test loss:', val_loss)
print('Test accuracy:', val_acc)
print('Train loss:', loss)
print('Train accuracy:', accu)

print('model vggrep 40 epoch net Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print(' Test Accuracy: %.5f  \n' % acc)
print(' Test precision: %.5f  \n' % precision)
print(' Test recall score: %.5f  \n' % recall)
print(' Test f1 score: %.5f  \n' % f1)