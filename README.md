# CEAHB2021-5 database

https://paperswithcode.com/sota/image-classification-on-mnist  CNN on MNIST

https://paperswithcode.com/methods/category/convolutional-neural-networks  CNN list

https://github.com/search?q=mixnet+keras

https://github.com/search?q=shuffle+keras

https://github.com/search?q=darknet+keras

https://github.com/leondgarse/Keras_efficientnet_v2

https://github.com/kaparoo/Efficient-CapsNet

https://github.com/shyhyawJou/ExquisiteNetV2

https://github.com/pranavphoenix/WaveMix

```
# 加载模型权重 环境配置需要相同
from keras.models import load_model
from keras.utils import CustomObjectScope

with CustomObjectScope( {'SpatialPyramidPooling': SpatialPyramidPooling} ):
    model = load_model('m_repvgg40e2.h5', compile=False)
model.summary()
```

替换模型方法：以cspnet.py为例

把cspnet.py和mu_rep.py/MSVGG.py放在同级目录下，把本模型代码注释掉，添加下面两行代码：
```
# 文件名不要和类名或函数名一致，会报错
# cspnet->文件名  csp_densenet->函数名

from cspnet import csp_densenet
model = csp_densenet(input_shape=(386,386,1))
```
E:\压缩包\SVHN 数据集路径，只需加载train_32x32.mat及text_32x32.mat

FashionMNIST   E:\zhudawei_pycharm\vit-pytorch-main\Fashion_MNIST\FashionMNIST\raw

MNIST    E:\zhudawei_pycharm\GhostNet-MNIST-master\data\MNIST\raw  MLe2e CVSI2015

https://github.com/joewellhe/textCaps/tree/master/textCaps

https://github.com/vinojjayasundara/textcaps

https://github.com/hypnopump/SimpleNet-Keras/blob/master/simplenet.py

https://github.com/philipperemy/tensorflow-maxout

https://github.com/paniabhisek/maxout

# https://github.com/DrMahdiRezaei/Deep-CAPTCHA
```
本项目中model中有对应加载及训练代码
环境配置：
keras==2.2.4
tensorflow-gpu==1.14/1.15
```
5折交叉验证：
```
num_classes = 5
img_rows, img_cols = 386, 386

def read_image(img_name):
    im = Image.open(img_name).convert('L')
    im = im.resize((386, 386))
    data = np.array(im)
    # print(data.shape)
    return data

# D:\vscode\vscodework\zangwen
images = []
for fn in os.listdir('E:/guji_resizedata510'):
    if fn.endswith('.jpg'):
        fd = os.path.join('E:/guji_resizedata510',fn)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)
y = np.loadtxt('E:/gujilabel.txt')
print (y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state= 3)
from sklearn.model_selection import KFold

n_folds = 5
kfold = KFold(n_folds, shuffle=True, random_state=1)
scores = []
for train_index, test_index in kfold.split(X):
    X_train, y_train, X_test,  y_test = X[train_index], y[train_index], X[test_index], y[test_index]
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
    epochs = 1

    # 更换基础模型架构
    # from  import GhostNet
    # model = GhostNet((386,386,1),5).build(False)
    
    model = csp_densenet(input_shape=(386,386,1))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True),
                  metrics=['accuracy'])

    H = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))
    # model.save('./zangwen/m_repvgg_5fold.h5')
    acc = model.evaluate(X_test, y_test, verbose=0)
    # print(' %.3f' % (acc * 100.0))
    scores.append(acc)
print("5折交叉验证结果：", scores, np.mean(scores))    
```

中国少数民族古籍文档文种识别 是 中国少数民族古籍数字化平台 的重要组成部分，该平台主要包括文档文种识别、图像处理、版面分析、内容识别及版面重建等。 该平台进展情况->
https://blog.csdn.net/qq_35200351/article/details/116127962

Chinese ancient books script identification with deep convolutional neural networks via multi-branch and spatial pyramid pooling

dataset：Chinese Ethnic Ancient Handwritten Books database, CEAHD2021-5

E.g.:

![image](https://github.com/yddcode/CNAHB2021-5/blob/main/img/20210328093743.png)

The dataset was published at Baidu Drive and Google Drive.

Baidu Drive ![link] (https://pan.baidu.com/s/19mKhhH2klRfxnjXqfKcrog?pwd=CNAH )
Extraction code:CNAH 


Google Drive ![link] (https://drive.google.com/file/d/1zoJi466Dw_80Z8HL1HY4buDuppyOROgm/view?usp=sharing)
or (https://drive.google.com/file/d/11Rcn5ibQCEnTvAeRG8Xj6rv4e-CCf889/view?usp=sharing, https://drive.google.com/file/d/1zoJi466Dw_80Z8HL1HY4buDuppyOROgm/view?usp=sharing)


If you use the dataset, please cite this paper: (Accepted 01.25.2023 Pattern Analysis and Applications)


Chinese ancient books script identification with deep CNNs via multi-branch and SPP.
