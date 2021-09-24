from tensorflow.python.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    MaxPooling2D,
    UpSampling2D,
    AveragePooling2D,
    Conv2D,
    BatchNormalization,
    Concatenate,
    Activation,
    Flatten,
    Add,
    Multiply,
    Reshape,
    Lambda,
    Dropout
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf


class SE_HRNet(object):
    def __init__(self, blocks=3, reduction_ratio=4, init_filters=64, expansion=4, training=True):
        self.blocks = blocks
        self.training = training
        self.reduction_ratio = reduction_ratio
        self.init_filters = init_filters
        self.expansion = expansion
        print('Expansion part has not been implemented.')

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Conv2D(filters=self.init_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Conv2D(filters=self.init_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
            norm = BatchNormalization(axis=-1)(x, training=self.training)
            act = Activation("relu")(norm)
            return act

    def bottleneck_block(self, x, filters, strides, scope):
        with tf.name_scope(scope):
            x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation("relu")(x)

            x = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation("relu")(x)

            x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)

            return x

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):

            squeeze = GlobalAveragePooling2D()(input_x)

            excitation = Dense(units=out_dim / ratio)(squeeze)
            excitation = Activation("relu")(excitation)
            excitation = Dense(units=out_dim)(excitation)
            excitation = Activation("sigmoid")(excitation)
            excitation = Reshape([1, 1, out_dim])(excitation)
            scale = Multiply()([input_x, excitation])

            return scale

    def residual_layer(self, out_dim, scope, first_layer_stride=(2, 2), res_block=None):
        if res_block is None:
            res_block = self.blocks

        def f(input_x):
            # split + transform(bottleneck) + transition + merge
            # input_dim = input_x.get_shape().as_list()[-1]
            for i in range(res_block):
                if i == 0:
                    strides = first_layer_stride
                    # filters = input_x.get_shape().as_list()[-1]
                else:
                    strides = (1, 1)
                    # filters = out_dim
                x = self.bottleneck_block(input_x, filters=out_dim, strides=strides, scope='bottleneck_' + str(i))
                # x = self.squeeze_excitation_layer(x, out_dim=x.get_shape().as_list()[-1], ratio=self.reduction_ratio, layer_name='squeeze_layer_' + str(i))
                if i != 0:  # Leave the first block without residual connection due to unequal shape
                    x = Add()([input_x, x])
                x = Activation('relu')(x)
                input_x = x
            return input_x
        return f

    def multi_resolution_concat(self, maps, filter_list, scope='multi_resolution_concat'):
        fuse_layers = []
        print('Input', maps)
        with tf.name_scope(scope):
            for idx, _ in enumerate(maps):
                fuse_list = []
                for j in range(len(maps)):
                    x = maps[j]
                    # Upsamples, high resolution first
                    if j < idx:
                        # Downsamples, high resolution first
                        for k in range(idx - j):
                            x = Conv2D(filter_list[idx], kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
                            x = BatchNormalization(axis=-1)(x, training=self.training)
                            if k == idx - j - 1:
                                x = Activation("relu")(x)
                    elif j == idx:
                        # Original feature map
                        pass
                    elif j > idx:
                        for k in range(j - idx):
                            x = Conv2D(filter_list[idx], kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
                            x = BatchNormalization(axis=-1)(x, training=self.training)
                            x = UpSampling2D(size=(2, 2))(x)
                    else:
                        raise ValueError()
                    fuse_list.append(x)
                    print(idx, j, maps[j])
                    print(filter_list[idx], x)
                if len(fuse_list) > 1:
                    concat = Add()(fuse_list)
                    x = Activation("relu")(concat)
                    fuse_layers.append(x)
                    # print('Assemble', concat)
                else:
                    fuse_layers.append(fuse_list[0])
                    # print('Assemble O', fuse_list[0])
            print('Out', fuse_layers)
        return fuse_layers

    def extract_multi_resolution_feature(self, repetitions=3):

        def f(input_x):
            x = self.first_layer(input_x, scope='first_layer')
            features = []
            filters = self.init_filters
            self.filter_list = [filters]
            # First Layer consumed one stage
            for i in range(repetitions):
                print('\nBuilding ... %d/%d' % (i, repetitions))
                # Get Downsample
                scope = 'stage_%d' % (i + 1)
                if i == 0:
                    down_x = self.residual_layer(filters, scope=scope, first_layer_stride=(2, 2), res_block=self.blocks)(x)
                else:
                    down_x = self.residual_layer(filters, scope=scope, first_layer_stride=(2, 2), res_block=self.blocks)(features[-1])
                features.append(down_x)
                # Get concatenated feature maps
                out_maps = self.multi_resolution_concat(features, self.filter_list)
                features = []
                print('Identity Mapping:')
                # Residual connection with 3x3 kernel, 1x1 stride with same number of filters
                for idx, (fm, num_filter) in enumerate(zip(out_maps, self.filter_list)):
                    x = Lambda(lambda x: x, output_shape=x.get_shape().as_list())(fm)
                    print(idx, x)
                    features.append(x)
                filters *= 2
                self.filter_list.append(filters)
            return features

        return f

    def make_classification_head(self, feature_maps, filter_list):
        previous_fm = None
        for idx, fm in enumerate(feature_maps):
            if previous_fm is None:
                previous_fm = fm
                continue
            # if idx == len(feature_maps):
            #     # The final feature map no need to add
            #     continue
            print(previous_fm.get_shape().as_list(), fm.get_shape().as_list(), filter_list[idx], filter_list)
            x = Conv2D(filter_list[idx], kernel_size=(3, 3), strides=(2, 2), padding='same')(previous_fm)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation("relu")(x)
            previous_fm = Add()([fm, x])
        return previous_fm

    def build(self, input_shape, num_output, repetitions=3):
        input_x = Input(shape=input_shape)

        feature_maps = self.extract_multi_resolution_feature(repetitions=repetitions)(input_x)
        x = self.make_classification_head(feature_maps, self.filter_list)

        x = Conv2D(filters=x.get_shape().as_list()[-1] * 2, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=-1)(x, training=self.training)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        x = Dense(units=num_output,
                  name='final_fully_connected',
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1e-4),
                  activation='softmax')(x)

        return Model(inputs=input_x, outputs=x)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    import tensorflow.keras.backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, precision_score
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import optimizers, regularizers # 优化器，正则化项
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, add
from tensorflow.python.keras.layers import Conv2D,MaxPool2D, GlobalAvgPool2D, MaxPooling2D,Activation,Conv2D,MaxPooling2D,Flatten,ZeroPadding2D,BatchNormalization,AveragePooling2D,concatenate
from PIL import Image
import numpy as np
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.layers import Input,Add,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten 
from tensorflow.python.keras.layers import BatchNormalization,AveragePooling2D,concatenate  
from tensorflow.python.keras.layers import ZeroPadding2D,add
from tensorflow.python.keras.layers import Dropout, Activation
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD, Adam, Adadelta
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.applications import inception_v3
time_start = time.time()
import sys, cv2, os, keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import tensorflow.python.keras

if __name__ == '__main__':

    num_classes = 5
    # input image dimensions
    img_rows, img_cols = 384, 384

    # the data, split between train and test sets
    def read_image(img_name):
        im = Image.open(img_name).convert('L')
        im = im.resize((384, 384))
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
    # X_train, y_train = X_train[:5000], y_train[:5000]
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
    batch_size = 4
    epochs = 40

    # SE block delete
    model = SE_HRNet(blocks=3, reduction_ratio=4, init_filters=32, training=True).build(input_shape=(384, 384, 1), num_output=5, repetitions=4)
    # print(model.summary())
    model.summary()
    # print(get_model_memory_usage(1, model))
    # from tensorflow.python.keras.utils import plot_model
    # plot_model(model, to_file='./zangwen/se_hrnet.png')

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True),
              metrics=['accuracy'])
    H = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))
    model.save('./zangwen/m_HRNet.h5')
    from keras.models import load_model
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
    plt.savefig('./trainHRNet.png', dpi = 600)
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('./tlossHRNet.png', dpi = 600)
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
    print('HRNet Test loss:', val_loss)
    print('Test accuracy:', val_acc)
    print('Train loss:', loss)
    print('Train accuracy:', accu)

    print('model HRNet 40 epoch net Test classify_report : \n', classify_report)
    # print('Stacked Test confusion_matrix : \n', confusion_matrix)
    print(' Test Accuracy: %.5f  \n' % acc)
    print(' Test precision: %.5f  \n' % precision)
    print(' Test recall score: %.5f  \n' % recall)
    print(' Test f1 score: %.5f  \n' % f1)