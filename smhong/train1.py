import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from sklearn.metrics import roc_curve
from tensorflow.keras import layers, models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from sklearn.metrics import roc_curve
import time

import pandas as pd
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tqdm
from scipy import signal
from tensorflow import keras
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# This method loads your data from the .npz file

data = np.load('ecg_child_data.npz')
X_train_child = data['ecg_child_data_array']

print("X_train_child.shape=", X_train_child.shape)
print("X_train_child[0].shape=", X_train_child[0].shape)

data1 = np.load('y_child_train.npz')
y_train_child = data1['y_child_train']

print("y_train_child.shape=", y_train_child.shape)
print("y_train_child[0].shape=", y_train_child[0].shape)
print("y_train_child[0:10]=", y_train_child[0:10])

# y_train_child의 소숫점값을 정수로 변환
y_train_child = y_train_child.astype(int)
print("y_train_child[0:10]=", y_train_child[0:10])

#y_train_child의 최소값과 최대값 출력
print("y_train_child.min()=", y_train_child.min())
print("y_train_child.max()=", y_train_child.max())

BATCH_SIZE = 2

train_dset = tf.data.Dataset.from_tensor_slices((X_train_child, y_train_child)).batch(BATCH_SIZE)

#Model hyperparameters
#하이퍼 파라미터를 설정합니다.
# learning_rate = 0.001
# num_epochs = 10000
# batch_size = 10
# display_step = 1
# input_size = 5000
# output_size = 10


# 모델 생성
def attia_network_age(input_shape, num_leads):
    input_layer = keras.layers.Input(shape=(input_shape, num_leads))

    # Temporal analysis block 1
    conv1 = keras.layers.Conv1D(filters=16, kernel_size=7, strides=1, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation("relu")(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)

    # Temporal analysis block 2
    conv2 = keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation("relu")(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)

    # Temporal analysis block 3
    conv3 = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation("relu")(conv3)
    conv3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)

    # Temporal analysis block 4
    conv4 = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation("relu")(conv4)
    conv4 = keras.layers.MaxPooling1D(pool_size=2)(conv4)

    # Temporal analysis block 5
    conv5 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Activation("relu")(conv5)
    conv5 = keras.layers.MaxPooling1D(pool_size=2)(conv5)

    # Temporal analysis block 6
    conv6 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv5)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Activation("relu")(conv6)
    conv6 = keras.layers.MaxPooling1D(pool_size=2)(conv6)

    # Temporal analysis block 7
    conv7 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv6)
    conv7 = keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.Activation("relu")(conv7)
    conv7 = keras.layers.MaxPooling1D(pool_size=2)(conv7)

    # Temporal analysis block 8
    conv8 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv7)
    conv8 = keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.Activation("relu")(conv8)
    conv8 = keras.layers.MaxPooling1D(pool_size=2)(conv8)

    # Spatial analysis block 1
    spatial_block_1 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='same')(conv8)
    spatial_block_1 = keras.layers.BatchNormalization()(spatial_block_1)
    spatial_block_1 = keras.layers.Activation("relu")(spatial_block_1)
    spatial_block_1 = keras.layers.MaxPooling1D(pool_size=2)(spatial_block_1)
    spatial_block_1 = tf.keras.layers.Flatten()(spatial_block_1)

    # Fully Connected block 1
    fc_block_1 = keras.layers.Dense(units=128)(spatial_block_1)
    fc_block_1 = keras.layers.BatchNormalization()(fc_block_1)
    fc_block_1 = keras.layers.Activation("relu")(fc_block_1)
    fc_block_1 = keras.layers.Dropout(rate=0.2)(fc_block_1)

    # Fully Connected block 1
    fc_block_2 = keras.layers.Dense(units=64)(fc_block_1)
    fc_block_2 = keras.layers.BatchNormalization()(fc_block_2)
    fc_block_2 = keras.layers.Activation("relu")(fc_block_2)
    fc_block_2 = keras.layers.Dropout(rate=0.2)(fc_block_2)

    # output_layer_1 = keras.layers.Dense(units=1,activation='linear')(last_dense)
    output_layer = keras.layers.Dense(units=1, activation='linear')(fc_block_2)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def build_model(input_shape, nb_classes, depth=50, use_residual=True):
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(units=nb_classes, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss=[macro_double_soft_f1], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


samp_freq = 100
time = 10
num_leads = 12
batchsize = 16
epoch = 25

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min',
    min_delta=0.0001, cooldown=2, min_lr=0
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

print("Training model...")
for i in range(len(folds)):
    train_ind = folds[i][0]
    test_ind = folds[i][1]

    # model = attia_network_age(samp_freq,time,num_leads) # velg modell
    model = build_model((samp_freq * time, num_leads), 1)
    model.fit(x=shuffle_batch_generator_age(batch_size=batchsize, gen_x=generate_X_age(ecg_filenames[train_ind]),
                                            gen_y=generate_y_age(age[train_ind]), num_leads=num_leads), epochs=epoch,
              steps_per_epoch=(len(train_ind) / batchsize),
              validation_data=shuffle_batch_generator_age(batch_size=batchsize,
                                                          gen_x=generate_X_age(ecg_filenames[test_ind]),
                                                          gen_y=generate_y_age(age[test_ind]), num_leads=num_leads),
              validation_freq=1, validation_steps=(len(test_ind) / batchsize),
              verbose=1,
              callbacks=[reduce_lr]
              )