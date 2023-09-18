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

#train_dset = tf.data.Dataset.from_tensor_slices((X_train_child, y_train_child)).batch(BATCH_SIZE)
#valid_dset = tf.data.Dataset.from_tensor_slices((X_vali_child, y_vali_child)).batch(BATCH_SIZE)
train_dset = tf.data.Dataset.from_tensor_slices((X_train_child, y_train_child_encoded)).batch(BATCH_SIZE)
valid_dset = tf.data.Dataset.from_tensor_slices((X_vali_child, y_vali_child_encoded)).batch(BATCH_SIZE)

#Model hyperparameters
#하이퍼 파라미터를 설정합니다.
# learning_rate = 0.001
# num_epochs = 10000
# batch_size = 10
# display_step = 1
# input_size = 5000
# output_size = 10


# 모델 생성
# CNN 모델을 정의합니다.
# 모델 생성
model = models.Sequential()

# 1D 컨볼루션 레이어
model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(5000, 12)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))

# 전연결 레이어
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(1, activation='softmax'))  # 101개의 클래스에 대한 softmax 출력
model.add(layers.Dense(9, activation='softmax'))  # 101개의 클래스에 대한 softmax 출력

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 아키텍처 요약
model.summary()

# 모델 학습 , train_dset, valid_dset
model.fit(train_dset, epochs=100, validation_data=valid_dset)
#model.fit(train_dset, epochs=10)

print("model training complete!!!")