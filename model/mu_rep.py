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

num_classes = 5

# input image dimensions
img_rows, img_cols = 386, 386

# the data, split between train and test sets
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    im = im.resize((386, 386))
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
# batch_size = 2
# epochs = 40

input2 = keras.layers.Input(shape=(img_cols, img_rows, 1))
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

# out = keras.layers.Dense(num_classes, activation='softmax')(x2)
# model = keras.models.Model(inputs=input2, outputs=out)
# model.summary()

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True),
#               metrics=['accuracy'])
# H = model.fit(X_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(X_test, y_test))
# model.save('./zangwen/m_repvgg40e2.h5')
from keras.models import load_model
from keras.utils import CustomObjectScope
with CustomObjectScope({'SpatialPyramidPooling': SpatialPyramidPooling}):
    model = load_model('./zangwen/m_repvgg40e2.h5')
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
np.savetxt('./zangwen/mu_Y_test.csv', y_test1, delimiter=',')
np.savetxt('./zangwen/mu_Y_test_pred.csv', pred, delimiter=',')
precision = precision_score(y_test1, pred, average='macro')
acc = accuracy_score(y_test1, pred)
f1 = f1_score(y_test1, pred, average='macro')
recall = recall_score(y_test1, pred, average='macro')
classify_report = classification_report(y_test1, pred, digits=4)

output = sys.stdout
outputfile = open("zangwen/multispp40.txt","a")
sys.stdout = outputfile
# print('repspp40 2 Test loss:', val_loss)
# print('Test accuracy:', val_acc)
# print('Train loss:', loss)
# print('Train accuracy:', accu)

print('model milti 2 spp repvgg 40 epoch net Test classify_report : \n', classify_report)
# print('Stacked Test confusion_matrix : \n', confusion_matrix)
print(' Test Accuracy: %.5f  \n' % acc)
print(' Test precision: %.5f  \n' % precision)
print(' Test recall score: %.5f  \n' % recall)
print(' Test f1 score: %.5f  \n' % f1)

# define the layer for feature extraction
# intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('dense1').output)
# # get engineered features for training and validation
# feature_engineered_train = intermediate_layer.predict([X_train,X_train,X_train])
# feature_engineered_train = pd.DataFrame(feature_engineered_train)
# feature_engineered_train.to_csv('zangwen/multii_feature_train.csv')
# feature_engineered_test = intermediate_layer.predict([X_test, X_test, X_test])
# feature_engineered_test = pd.DataFrame(feature_engineered_test)
# feature_engineered_test.to_csv('zangwen/multii_feature_test.csv')


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names) # , rotation=45
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('./zangwen/mconfusionmatrixspprep40e4.png',dpi=600)
    plt.show()
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val, labels, pred):
    # predictions = model.predict(x_val, batch_size=8)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=pred)
    print('conf_mat::', conf_mat)
    # font = {'family': 'Times New Roman',
    #     'size': 12,
    #     }
    # sns.set(font_scale=1.2)
    # plt.rc('font',family='Times New Roman')
    plt.rc('font', family='Times New Roman', size='12')  # 设置字体样式、大小
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')
#=========================================================================================
#最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
#labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
#比如这里我的labels列表
labels=['Dai','Naxi','Shui','Yi','Tibet']

plot_confuse(model, X_test, y_test, labels, pred)


con_mat = confusion_matrix(y_test1, pred)
print('con_mat:: ', con_mat)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)
font = {'family': 'Times New Roman',
        'size': 12,
        }
sns.set(font_scale=1.2)
plt.rc('font',family='Times New Roman')

for i in range(con_mat_norm.shape[0]):
    for j in range(con_mat_norm.shape[1]):
        plt.text(x=j, y=i, s="{:,}".format(con_mat[i, j]), va='center', ha='center', color='white') # s=int(con_mat[i, j])

# === plot ===
plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45) # 
plt.yticks(tick_marks, labels)
# plt.ylim(0, 10)
plt.tight_layout()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('./zangwen/mconfusionmatrixspprep40e44.png',dpi=600)
plt.show()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_Matrix(cm, title=None, cmap=plt.cm.Purples, labels=labels):
    plt.rc('font', family='Times New Roman', size='12')  # 设置字体样式、大小
    # font = {'family': 'Times New Roman',
    #     'size': 12,
    #     }
    # sns.set(font_scale=1.2)
    # plt.rc('font',family='Times New Roman')
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots(figsize = (9, 9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    print(im)
    ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45) # 
    plt.yticks(tick_marks, labels)
    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('./zangwen/mconfusionmatrixspprep40e444.jpg', dpi=600)
    plt.show()

# y_true = np.loadtxt(open("Y_test1.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
# y_pred = np.loadtxt(open("Y_test_pred1.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
c = confusion_matrix(y_test1, pred)
print('c', c)
plot_Matrix(c, labels=labels)


def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input
    print('shape', input_img.shape)
    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(name=layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]

def vis_conv(images, n, name, t, numb):
    """visualize conv output and conv filter.
    Args:
           img: original image.
           n: number of col and row.
           t: vis type.
           name: save name.
    """
    size = 64
    margin = 5

    if t == 'filter':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin, 3))
    if t == 'conv':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin))

    for i in range(n):
        for j in range(n):
            if t == 'filter':
                filter_img = images[i + (j * n)]
            if t == 'conv':
                filter_img = images[..., i + (j * n)]
            filter_img = cv2.resize(filter_img, (size, size))

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            if t == 'filter':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
            if t == 'conv':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img

    # Display the results grid
    plt.imshow(results)
    cv2.imwrite('./zangwen/' + str(numb) + 'cv2images{}_{}.jpg'.format(t, name), results)
    plt.savefig('./zangwen/' + str(numb) + 'images{}_{}.jpg'.format(t, name), dpi=600)
    # plt.show()

# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
for i in range(2,16):
    img1 = X_test[i].reshape(1, img_rows, img_cols, 1)
    img = conv_output(model, 'conv2d_1', img1)
    vis_conv(img, 8, 'conv2d_1', 'conv', i)

    img = conv_output(model, 'conv2d_3', img1)
    vis_conv(img, 8, 'conv2d_3', 'conv', i)

    img = conv_output(model, 'conv2d_5', img1)
    vis_conv(img, 8, 'conv2d_5', 'conv', i)

    img = conv_output(model, 'conv2d_9', img1)
    vis_conv(img, 8, 'conv2d_9', 'conv', i)

    img = conv_output(model, 'conv2d_13', img1)
    vis_conv(img, 8, 'conv2d_13', 'conv', i)

    img = conv_output(model, 'conv2d_15', img1)
    vis_conv(img, 8, 'conv2d_15', 'conv', i)

    img = conv_output(model, 'conv2d_21', img1)
    vis_conv(img, 8, 'conv2d_21', 'conv', i)

    img = conv_output(model, 'conv2d_25', img1)
    vis_conv(img, 8, 'conv2d_25', 'conv', i)


    img = conv_output(model, 'conv2d_2', img1)
    vis_conv(img, 8, 'conv2d_2', 'conv', i)

    img = conv_output(model, 'conv2d_4', img1)
    vis_conv(img, 8, 'conv2d_4', 'conv', i)

    img = conv_output(model, 'conv2d_8', img1)
    vis_conv(img, 8, 'conv2d_8', 'conv', i)

    img = conv_output(model, 'conv2d_12', img1)
    vis_conv(img, 8, 'conv2d_12', 'conv', i)

    img = conv_output(model, 'conv2d_14', img1)
    vis_conv(img, 8, 'conv2d_14', 'conv', i)

    img = conv_output(model, 'conv2d_20', img1)
    vis_conv(img, 8, 'conv2d_20', 'conv', i)

    img = conv_output(model, 'conv2d_24', img1)
    vis_conv(img, 8, 'conv2d_24', 'conv', i)
