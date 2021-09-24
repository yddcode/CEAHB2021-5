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
from keras.optimizers import SGD, Adam, Adadelta, RMSprop, Nadam
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
img_rows, img_cols = 386, 386

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    im = im.resize((386, 386), Image.BILINEAR) # 
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
# X_train, X_test, y_train, y_test = X_train[:600], X_test[:600], y_train[:600], y_test[:600]
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
batch_size = 2
epochs = 40

from keras.engine.topology import Layer
import keras.backend as K

class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        print(self.dim_ordering)
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'channels_first':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'channels_last':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'channels_first':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'channels_last':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs
# input2 = keras.layers.Input(shape=(img_cols, img_rows, 1))
# x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(input2)
# x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# x2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# x2 = Flatten()(x2)
# x2 = Dense(4096, activation='relu')(x2)
# x2 = Dense(4096, activation='relu')(x2)
# x2 = Dropout(0.5)(x2)

# out = keras.layers.Dense(num_classes, activation='softmax')(x2)
# model = keras.models.Model(inputs=input2, outputs=out)
# model.summary()
# model.add(Dense(num_classes, activation='softmax'))

from keras.applications.vgg16 import VGG16
# model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
# for layer in model_vgg.layers:
#     layer.trainable = False
     
# model = Flatten()(model_vgg.output)
# model = Dense(4096, activation='relu', name='fc1')(model)
# model = Dense(4096, activation='relu', name='fc2')(model)
# model = Dropout(0.5)(model)
# model = Dense(num_classes, activation='softmax', name='prediction')(model)
# model = Sequential()

# model.add(Conv2D(64,(3,3), strides = (1,1), input_shape = (386,386,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D((2,2), strides = (2,2)))

# model.add(Conv2D(128,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(128,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D((2,2), strides = (2,2)))

# model.add(Conv2D(256,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(256,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(256,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# # model.add(Conv2D(256,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D((2,2), strides = (2,2)))

# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# # model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D((2,2), strides = (2,2)))

# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(Conv2D(512,(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D((2,2), strides = (2,2)))
# model.add(SpatialPyramidPooling([1, 2, 4]))
# # model.add(Flatten())
# model.add(Dense(4096, activation = 'relu'))
# model.add(Dense(4096, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(5, activation = 'softmax'))
# model = Model(model_vgg.input, model, name='vgg16_pretrain')
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
x_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add()([x2, x_1, x_3])  
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
x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add()([x2, x_1, x_3])  
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
x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
x2 = keras.layers.Add()([x2, x_1, x_3])  
# x2 = MaxPooling2D(pool_size=(2, 2))(x2)

x2 = SpatialPyramidPooling([1, 2, 4])(x2)
# x2 = Flatten()(x2)
x2 = Dense(4096, activation='relu')(x2)
x2 = Dense(4096, activation='relu')(x2)
x2 = Dropout(0.5)(x2)

out = keras.layers.Dense(num_classes, activation='softmax')(x2)
model = keras.models.Model(inputs=input2, outputs=out)

model.summary()

sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.99)
rms = RMSprop(lr=0.0001, rho=0.9)
ada = Adadelta(lr=0.0001, rho=0.95)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, amsgrad=True) # , beta_1=0.9, beta_2=0.999
nadam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.96, epsilon=1e-08, schedule_decay=0.002)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam, # keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
              metrics=['accuracy'])
H = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
# model.save('./zangwen/m_vgglr001adam40espp.h5')
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
plt.savefig('./trainvgg19adam40spp.png', dpi = 600)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./tlossvgg19adam40spp.png', dpi = 600)
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
print('vgg lr adam 20e spp Test loss:', val_loss)
print('Test accuracy:', val_acc)
print('Train loss:', loss)
print('Train accuracy:', accu)

print('model vgg19 lr001adam 20 epoch spp net Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print(' Test Accuracy: %.5f  \n' % acc)
print(' Test precision: %.5f  \n' % precision)
print(' Test recall score: %.5f  \n' % recall)
print(' Test f1 score: %.5f  \n' % f1)