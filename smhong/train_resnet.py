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
from tensorflow.keras.utils import to_categorical
from model_cnn import cnn_model
from model_resnet import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import tensorflow as tf
from sklearn.metrics import roc_curve
import time

# This method loads your data from the .npz file
directory_path='/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/dataset/'

data = np.load(directory_path + 'ecg_child_data0.npz')
X_train_child = data['ecg_child_data_array']

data1 = np.load(directory_path + 'ecg_child_data1.npz')
X_vali_child = data1['ecg_child_data_array']

print("X_train_child.shape=", X_train_child.shape)
print("X_train_child[0].shape=", X_train_child[0].shape)
print("X_vali_child.shape=", X_vali_child.shape)
print("X_vali_child[0].shape=", X_vali_child[0].shape)

data2 = np.load(directory_path + 'y_child_train0.npz')
y_train_child = data2['y_child_train']

data3 = np.load(directory_path + 'y_child_train1.npz')
y_vali_child = data3['y_child_train']

print("y_train_child.shape=", y_train_child.shape)
print("y_train_child[0].shape=", y_train_child[0].shape)
print("y_train_child[0:10]=", y_train_child[0:10])

print("y_vali_child.shape=", y_vali_child.shape)
print("y_vali_child[0].shape=", y_vali_child[0].shape)
print("y_vali_child[0:10]=", y_vali_child[0:10])

# y_train_child의 소숫점값을 정수로 변환
y_train_child = y_train_child.astype(int)
print("y_train_child[0:10]=", y_train_child[0:10])

# y_vali_child의 소숫점값을 정수로 변환
y_vali_child = y_vali_child.astype(int)
print("y_vali_child[0:10]=", y_vali_child[0:10])

#y_train_child의 최소값과 최대값 출력
print("y_train_child.min()=", y_train_child.min())
print("y_train_child.max()=", y_train_child.max())

# One-Hot 인코딩
y_train_child_encoded = to_categorical(y_train_child, num_classes=9)
print("y_train_child_encoded[0:10]=", y_train_child_encoded[0:10])

y_vali_child_encoded = to_categorical(y_vali_child, num_classes=9)
print("y_vali_child_encoded[0:10]=", y_vali_child_encoded[0:10])

BATCH_SIZE = 10

train_dset = tf.data.Dataset.from_tensor_slices((X_train_child, y_train_child_encoded)).batch(BATCH_SIZE)
valid_dset = tf.data.Dataset.from_tensor_slices((X_vali_child, y_vali_child_encoded)).batch(BATCH_SIZE)

# 모델 생성
#model = cnn_model()
#model = resnet_model()
model = resnet19_model()

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 아키텍처 요약
model.summary()

# 모델 학습 , train_dset, valid_dset
model.fit(train_dset, epochs=100, validation_data=valid_dset)

print("model training complete!!!")