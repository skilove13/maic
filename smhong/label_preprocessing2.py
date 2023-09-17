
import json
import os
from tqdm import tqdm
# from dataloader import BatchDataloader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import glob
import time

directory_path='E:/smhong/micai_ai_challange/dataset/'

#ECG_child_age_train.csv 파일의 AGE부분만 읽어와서 데이터프레임으로 저장
y_adult_train = pd.read_csv(directory_path + 'ECG_adult_age_train.csv' , usecols=['AGE'] )

# 데이터프레임 확인
print(y_adult_train.head())

# 데이터프레임의 shape 확인
print(y_adult_train.shape)

# AGE 프레임 확인
print(y_adult_train.head())
print(len(y_adult_train))
print(type(y_adult_train))

# list로 변환
y_adult_train = y_adult_train.values.tolist()
print(y_adult_train[0:10])

y_adult_train = [value[0] for value in y_adult_train]
print(y_adult_train[0:10])

import numpy as np

# 'y_adult_train' 리스트를 5000개씩 나누어 저장할 부분 개수 계산
num_parts = len(y_adult_train) // 5000

# 5000개씩 나누어 저장
for i in range(num_parts):
    start_idx = i * 5000
    end_idx = (i + 1) * 5000
    y_adult_train_part = y_adult_train[start_idx:end_idx]

    # NumPy 배열로 변환
    y_adult_train_part_array = np.array(y_adult_train_part)

    # npz 파일로 저장
    np.savez(f'y_adult_train{i}.npz', y_adult_train_part_array=y_adult_train_part_array)

# 남은 데이터 (마지막 부분) 저장
remaining_data = y_adult_train[num_parts * 5000:]
if remaining_data:
    remaining_data_array = np.array(remaining_data)
    np.savez(f'y_adult_train{num_parts}.npz', y_adult_train_part_array=remaining_data_array)

# y_adult_train을 npz 파일로 저장

# np.savez('y_adult_train.npz', y_adult_train=y_adult_train)

# # npz 파일을 읽어와서 데이터프레임으로 저장
# y_child_train = np.load('y_child_train.npz')
# y_child_train = y_child_train['y_adult_train']
# print(y_child_train[0:10])

