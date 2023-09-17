
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
y_child_train = pd.read_csv(directory_path + 'ECG_child_age_train.csv' , usecols=['AGE'] )

# 데이터프레임 확인
print(y_child_train.head())

# 데이터프레임의 shape 확인
print(y_child_train.shape)

# AGE 프레임 확인
print(y_child_train.head())
print(len(y_child_train))
print(type(y_child_train))

# list로 변환
y_child_train = y_child_train.values.tolist()
print(y_child_train[0:10])

y_child_train = [value[0] for value in y_child_train]
print(y_child_train[0:10])


# AGE 프레임을 npz 파일로 저장
np.savez('y_child_train.npz', y_child_train=y_child_train)

# # npz 파일을 읽어와서 데이터프레임으로 저장
# y_child_train = np.load('y_child_train.npz')
# y_child_train = y_child_train['y_child_train']
# print(y_child_train[0:10])

