import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import tensorflow as tf
from sklearn.metrics import roc_curve
from tensorflow.keras.utils import to_categorical
from itertools import combinations
import time
from sklearn.model_selection import train_test_split
from model_resnet import *
from tensorflow.keras.callbacks import EarlyStopping

# 데이터셋 경로
directory_path = '/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/dataset/'

option = 1  # 0: child, 1: adult
# 데이터셋 파일명
# train_datasets = ['ecg_adult_data4.npz', 'ecg_adult_data3.npz', 'ecg_adult_data2.npz','ecg_adult_data6.npz',
#                   'ecg_adult_data1.npz', 'ecg_adult_data0.npz', 'ecg_adult_data5.npz']  # , 'ecg_child_data1.npz']
# y_train_datasets = ['y_adult_train4.npz', 'y_adult_train3.npz', 'y_adult_train2.npz','y_adult_train6.npz',
#                     'y_adult_train1.npz', 'y_adult_train0.npz', 'y_adult_train5.npz']  # ,'y_child_train1.npz']
train_datasets = ['ecg_adult_data0.npz']
y_train_datasets = ['y_adult_train0.npz']

valid_datasets = ['ecg_adult_data6.npz']
y_valid_datasets = ['y_adult_train6.npz']

# train_datasets = ['ecg_adult_data6.npz']
# y_train_datasets = ['y_adult_train6.npz']


# Validation dataset

data2 = np.load(directory_path + 'ecg_adult_data6.npz')
data3 = np.load(directory_path + 'y_adult_train6.npz')

x_val_adult = data2['ecg_adult_data_array']
y_val_adult = data3['y_adult_train']

# y_val_child의 소숫점값을 정수로 변환
y_val_adult = y_val_adult.astype(int)
print("y_val_child[0:10]=", y_val_adult[0:10])

# y_val_child의 최소값과 최대값 출력
print("y_val_adult.min()=", y_val_adult.min())
print("y_val_adult.max()=", y_val_adult.max())

# One-Hot 인코딩 , Max adult age = 122
y_val_adult_encoded = to_categorical(y_val_adult, num_classes=123)
print("y_val_adult_encoded[0:10]=", y_val_adult_encoded[0:10])

# 데이터셋 리스트를 4개씩 묶음
# train_datasets_grouped = [train_datasets[i:i+4]for i in range(0, len(train_datasets), 4)]
# y_train_datasets_grouped = [y_train_datasets[i:i+4]for i in range(0, len(y_train_datasets), 4)]
# 데이터셋 리스트를 모든 경우의 수로 4개씩 묶음
train_datasets_grouped = []
y_train_datasets_grouped = []


# train_datasets_grouped
train_datasets_grouped.append(list(train_datasets[0:]))
y_train_datasets_grouped.append(list(y_train_datasets[0:]))

# train_datasets_grouped.append(list(train_datasets[4:]))
# y_train_datasets_grouped.append(list(y_train_datasets[4:]))


print("train_datasets_grouped =", train_datasets_grouped)
print("y_train_datasets_grouped =", y_train_datasets_grouped)

print("len(train_datasets_grouped) =", len(train_datasets_grouped))
print("len(y_train_datasets_grouped) =", len(y_train_datasets_grouped))

print("====================================")

model_number = 1  # 30
# 각 그룹별로 학습 수행

for i, (train_group, y_train_group) in enumerate(zip(train_datasets_grouped, y_train_datasets_grouped)):
    x_train_adult = []
    y_train_adult = []

    for dataset, y_dataset in zip(train_group, y_train_group):
        print("dataset =", dataset)
        print("y_dataset =", y_dataset)
        data = np.load(directory_path + dataset)
        if 'child' in dataset:
            x_train_adult.append(data['ecg_child_data_array'])
        else:
            x_train_adult.append(data['ecg_adult_data_array'])

        y_data = np.load(directory_path + y_dataset)
        if 'child' in y_dataset:
            y_train_adult.append(y_data['y_child_train'])
        else:
            y_train_adult.append(y_data['y_adult_train'])

    x_train_adult = np.concatenate(x_train_adult, axis=0)
    y_train_adult = np.concatenate(y_train_adult, axis=0)

    # y_train_adult의 소숫점값을 정수로 변환
    y_train_adult = y_train_adult.astype(int)

    # One-Hot 인코딩
    y_train_adult_encoded = to_categorical(y_train_adult, num_classes=123)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # 데이터셋 shuffle 및 batch 처리
    BATCH_SIZE = 128 #256
    #train_dset = tf.data.Dataset.from_tensor_slices((x_train_adult, y_train_adult_encoded)).shuffle(buffer_size=len(x_train_adult)).batch(BATCH_SIZE)
    #valid_dset = tf.data.Dataset.from_tensor_slices((x_val_adult, y_val_adult_encoded)).shuffle(buffer_size=len(x_val_adult)).batch(BATCH_SIZE)
    # train set을 tf.data.Dataset으로 변환
    # 데이터를 train set과 test set으로 나눔
    x_train, x_test, y_train, y_test = train_test_split(x_train_adult, y_train_adult_encoded, test_size=0.05,random_state=42)
    train_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=len(x_train)).batch(BATCH_SIZE)
    # test set을 tf.data.Dataset으로 변환
    valid_dset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    # Early Stopping 콜백 정의

    # early_stopping_callback = EarlyStopping(
    #     monitor='accuracy',
    #     patience=3,
    #     restore_best_weights=True,
    #     min_delta=0.01,
    #     mode='max',
    #     baseline=0.9
    # )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 모델 로드
    print("model_number=", model_number)
    model_path = '/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/model'
    with strategy.scope():
        #model = resnet32_model()
        model = tf.keras.models.load_model(model_path + '/resnet19_model_adult' + str(model_number) + '.h5')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # 모델 컴파일
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    start_time = time.time()
    # 학습
    #history = model.fit(train_dset, epochs=20, validation_data=valid_dset)
    history = model.fit(train_dset, epochs=30, validation_data=valid_dset, callbacks=[early_stopping_callback])
    #
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 모델 번호 증가
    model_number += 1
    # 모델 저장
    model.save(model_path + '/resnet19_model_adult' + str(model_number) + '.h5')
    end_time = time.time()
    print("trained time :", end_time - start_time)
    print("model training number =", model_number - 10)

    del train_dset
    del model
    del x_train_adult
    del y_train_adult
    del history
    del valid_dset
    del x_train
    del x_test
    del y_train
    del y_test
    del y_train_adult_encoded
    del strategy


    print("==================================================")