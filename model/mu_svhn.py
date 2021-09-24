import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
import seaborn as sns
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
set_session(tf.Session(config=config))

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
 
image_size = 32
num_labels = 10
 
def display_data(path):
    print ('loading Matlab data...')
    train = sio.loadmat(path + 'train_32x32.mat')
    data=train['X']
    label=train['y']
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.title(label[i][0])
        plt.imshow(data[...,i])
        plt.axis('off')
    plt.show()
 
def load_data(path, one_hot = False):
    
    train = sio.loadmat(path + 'train_32x32.mat')
    test = sio.loadmat(path + 'test_32x32.mat')

    train_extra = sio.loadmat(path + 'extra_32x32.mat')
    
    train_data=train['X']
    train_label=train['y']

    extra_data=train_extra['X']
    extra_label=train_extra['y']

    test_data=test['X']
    test_label=test['y']  
    
    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)

    extra_data = np.swapaxes(extra_data, 0, 3)
    extra_data = np.swapaxes(extra_data, 2, 3)
    extra_data = np.swapaxes(extra_data, 1, 2)

    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    test_data = np.swapaxes(test_data, 1, 2)

    train, extra, test = [], [], []
    for img in train_data[:]:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        train.append(img)
    for img in extra_data[:]:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        extra.append(img)
    for img in test_data[:]:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        test.append(img)
    train_data = np.array(train)
    extra_data = np.array(extra)
    test_data = np.array(test)  

    train_data = np.concatenate((train_data, extra_data))
    train_label = np.concatenate((train_label, extra_label))
    
    test_data = test_data / 255.
    train_data = train_data / 255.
    
    for i in range(train_label.shape[0]):
         if train_label[i][0] == 10:
             train_label[i][0] = 0
                        
    for i in range(test_label.shape[0]):
         if test_label[i][0] == 10:
             test_label[i][0] = 0
 
    if one_hot:
        train_label = (np.arange(num_labels) == train_label[:,]).astype(np.float32)
        test_label = (np.arange(num_labels) == test_label[:,]).astype(np.float32)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    return train_data, train_label, test_data, test_label
 
if __name__ == '__main__':
    path = 'D:/SVHN/'
    X_train, y_train, X_test, y_test = load_data(path, one_hot = True)
    # display_data(path)
    img_cols, img_rows = 224, 224
    num_classes = 10
    batch_size = 4
    epochs = 40

    input2 = keras.layers.Input(shape=(img_cols, img_rows, 3))
    x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(input2)
    x_1 = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(input2)
    x2 = keras.layers.Add(name='add1')([x2, x_1])  
    x_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add2')([x2, x_1, x_3])  
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(128, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add3')([x_1, x_3])  
    x_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(128, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add4')([x2, x_1, x_3])  
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add5')([x_1, x_3])  
    x_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add6')([x2, x_1, x_3])  
    x_3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add7')([x2, x_1, x_3])  
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add8')([x_1, x_3])  
    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add9')([x2, x_1, x_3])  
    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add10')([x2, x_1, x_3])  
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add11')([x_1, x_3])  
    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add12')([x2, x_1, x_3])  
    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
    x_1 = Conv2D(512, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='uniform')(x2)
    x2 = keras.layers.Add(name='add13')([x2, x_1, x_3])  
    # x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x2 = SpatialPyramidPooling([1, 2, 4])(x2)
    # x2 = Flatten()(x2)
    x2 = Dense(4096, activation='relu')(x2)
    x2 = Dense(4096, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    # 相当于 added = keras.layers.add([x1, x2])
    # added = keras.layers.Add(name='add1')([x0, x1, x2])  
    # Dropout(0.5, name='dense1')

    out = keras.layers.Dense(num_classes, activation='softmax')(x2)
    model = keras.models.Model(inputs=input2, outputs=out)
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True),
                metrics=['accuracy'])
    H = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))
    # model.save('./zangwen/m_repvgg40e2.h5')
    from keras.models import load_model
    # from keras.utils import CustomObjectScope
    # with CustomObjectScope({'SpatialPyramidPooling': SpatialPyramidPooling}):
    #     model = load_model('./zangwen/m_repvgg40e.h5')
    # model = load_model('./zangwen/m_repvgg40e.h5')

    # accu = H.history['accuracy']
    # val_acc = H.history['val_accuracy']
    # loss = H.history['loss']
    # val_loss = H.history['val_loss']
    # epochs = range(len(accu))

    # plt.plot(epochs,accu, 'b', label='Training accuracy')
    # plt.plot(epochs, val_acc, 'r', label='validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend(loc='lower right')
    # plt.savefig('./trainrepspp40e2.png', dpi = 600)
    # plt.figure()

    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.savefig('./tlossrepspp40e2.png', dpi = 600)
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
    outputfile = open("/multispp40.txt","a")
    sys.stdout = outputfile
    # print('4 svhn Test loss:', val_loss)
    # print('Test accuracy:', val_acc)
    # print('Train loss:', loss)
    # print('Train accuracy:', accu)

    print('model milti 4 svhn 40 epoch net Test classify_report : \n', classify_report)
    # print('Stacked Test confusion_matrix : \n', confusion_matrix)
    print(' Test Accuracy: %.5f  \n' % acc)
    print(' Test precision: %.5f  \n' % precision)
    print(' Test recall score: %.5f  \n' % recall)
    print(' Test f1 score: %.5f  \n' % f1)
