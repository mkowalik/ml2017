import os
import keras
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

import tqdm
from keras.models import Sequential
from keras.layers.noise import GaussianNoise
from keras.layers import Dense, Activation, Dropout, Flatten, Input, UpSampling2D
from keras.optimizers import SGD
from keras.models import Model

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm

import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('tf')

def save_labels(arr, filename):
    pd_array = pd.DataFrame(arr)
    pd_array.index.names = ["Id"]
    pd_array.columns = ["Prediction"]
    pd_array.to_csv(filename)

def load_labels(filename):
    return pd.read_csv(filename, index_col=0).values.ravel()

X_train = np.load("X_train.npy")
y_train = load_labels("y_train.csv")
X_test = np.load("X_test.npy")

X_train_small = np.load("X_train_small.npy")
y_train_small = load_labels("y_train_small.csv")

y_train_one_hot = keras.utils.to_categorical(y_train)
y_train_small_one_hot = keras.utils.to_categorical(y_train_small)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train_small = X_train_small.astype('float32') / 255.0

print (X_train.shape)
print (X_test.shape)
print (X_train_small.shape)

print (y_train_small.max())
print (y_train_small.min())
classes_number = y_train_small.max() - y_train_small.min() + 1
print (classes_number)

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_train_small, y_train_small_one_hot, test_size=0.25)
X_tr, y_tr = X_train, y_train

def pack_color_image(tab):
    tab_red = tab[:,:1024]
    tab_green = tab[:,1024:2048]
    tab_blue = tab[:,2048:3072]

    ret = np.dstack((tab_red, tab_green, tab_blue))

    return ret

X_tr_s_reshaped = pack_color_image(X_tr_s).reshape(-1, 32, 32, 3)
X_te_s_reshaped = pack_color_image(X_te_s).reshape(-1, 32, 32, 3)
X_train_small_reshaped = pack_color_image(X_train_small).reshape(-1, 32, 32, 3)
print (X_tr_s_reshaped.shape)
print (X_te_s_reshaped.shape)

print (y_tr_s.shape)
print (y_te_s.shape)

num_classes = y_te_s.shape[1]
print (num_classes)

X_tr_reshaped = pack_color_image(X_tr).reshape(-1, 32, 32, 3)
print (X_tr_reshaped.shape)

X_test_reshaped = pack_color_image(X_test).reshape(-1, 32, 32, 3)

from keras.layers import Input, Embedding

dense_size = 1024
dropout = 0.2
kernel_size = (3, 3)
lrate = 0.001

input_img = Input(shape=(32, 32, 3))

x = Conv2D(32, kernel_size, padding='same', activation='relu')(input_img)
x = Dropout(dropout)(x)
x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
encoded = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, kernel_size, activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', data_format="channels_last")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

encoder = Model(input_img, encoded)
encoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

autoencoder.summary()

x = Flatten()(encoded)
x = Dense(dense_size, activation='relu', kernel_constraint=maxnorm(3))(x)
x = Dropout(dropout)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(input_img, output_layer)
epochs = 100
decay = lrate / epochs
momentum = 0.9
sgd = SGD(lr=lrate, momentum=momentum, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print (X_tr_reshaped.shape)

autoencoder.fit(X_tr_reshaped,X_tr_reshaped,
                epochs=epochs,
                batch_size=32,
                verbose=2)

model.fit(X_train_small_reshaped, y_train_small_one_hot, epochs=epochs, batch_size=32, verbose=2)

y_pred = model.predict(X_test_reshaped, batch_size=32)
y_pred = y_pred.argmax(axis=1)

save_labels(y_pred, 'y_pred_unsupervised_kowalik.csv')

print ("DONE!")