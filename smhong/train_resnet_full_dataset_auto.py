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

# 데이터셋 경로
directory_path = '/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/dataset/'

# 데이터셋 파일명
train_datasets = ['ecg_adult_data0.npz', 'ecg_adult_data1.npz', 'ecg_adult_data2.npz', 'ecg_adult_data3.npz', 'ecg_adult_data4.npz', 'ecg_adult_data5.npz', 'ecg_adult_data6.npz', 'ecg_child_data0.npz', 'ecg_child_data1.npz']
y_train_datasets = ['y_adult_train0.npz', 'y_adult_train1.npz', 'y_adult_train2.npz', 'y_adult_train3.npz', 'y_adult_train4.npz', 'y_adult_train5.npz' , 'y_adult_train6.npz', 'y_child_train0.npz','y_child_train1.npz']

# 데이터셋 리스트를 모든 경우의 수로 4개씩 묶음
train_datasets_grouped = []
y_train_datasets_grouped = []

#
for i in range(1, 5):
    train_combinations = combinations(train_datasets, i)
    y_train_combinations = combinations(y_train_datasets, i)

    for train_comb in train_combinations:
        if len(train_comb) == 4:
            train_datasets_grouped.append(list(train_comb))

    for y_train_comb in y_train_combinations:
        if len(y_train_comb) == 4:
            y_train_datasets_grouped.append(list(y_train_comb))

print("train_datasets_grouped =", train_datasets_grouped)
print("y_train_datasets_grouped =", y_train_datasets_grouped)

print("len(train_datasets_grouped) =", len(train_datasets_grouped))
print("len(y_train_datasets_grouped) =", len(y_train_datasets_grouped))

print("====================================")

model_number = 4
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

    #multi gpu 사용시 code 추가
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # 데이터셋 shuffle 및 batch 처리
    BATCH_SIZE = 64
    #train_dset = tf.data.Dataset.from_tensor_slices((x_train_adult, y_train_adult_encoded)).shuffle(buffer_size=len(x_train_adult)).batch(BATCH_SIZE)
    #valid_dset = tf.data.Dataset.from_tensor_slices((x_val_adult, y_val_adult_encoded)).shuffle(buffer_size=len(x_val_adult)).batch(BATCH_SIZE)
    x_train , x_test , y_train , y_test = train_test_split(x_train_adult, y_train_adult_encoded, test_size=0.1, random_state=42)
    train_dset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=len(x_train)).batch(BATCH_SIZE)
    valid_dset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=len(x_test)).batch(BATCH_SIZE)

    # 모델 로드
    print("model_number=", model_number)
    model_path = '/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/model'
    #multi gpu 사용시 code로 변경
    with strategy.scope():
        model = tf.keras.models.load_model(model_path + '/resnet19_model_adult' +str(model_number) + '.h5')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    # 학습
    model.fit(train_dset, epochs=100)

    model_number += 1
    # 모델 저장
    model.save(model_path + '/resnet19_model_adult' + str(model_number) + '.h5')
    end_time = time.time()
    print("trained time :", end_time - start_time)
    print("end 1 model training")

    del train_dset
    del model
    del x_train_adult
    del y_train_adult

    print("==================================================")