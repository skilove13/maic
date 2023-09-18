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
# CNN 모델을 정의합니다.
# 모델 생성
model = models.Sequential()

# 1D 컨볼루션 레이어
model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(5000, 12)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))

# 전연결 레이어
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))  # 101개의 클래스에 대한 softmax 출력

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 아키텍처 요약
model.summary()

# 모델 학습
model.fit(train_dset, epochs=2)

print("model training complete!!!")