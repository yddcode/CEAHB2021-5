from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras

iris = load_iris()
seed = 7
X = iris.data
Y = iris.target

Y_encode = to_categorical(Y)

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(7,activation='tanh',input_shape=(4,)))
    model.add(keras.layers.Dense(3,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer='sgd',metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model, epochs=10, batch_size=2, verbose=2)
kfold = KFold(n_splits=5,shuffle=True,random_state=seed)
result = cross_val_score(model,X,Y_encode,cv=kfold)
print("============ result: ===============")
print("mean:",result.mean())
print("std:",result.std())
