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

y_adult_train0 = y_adult_train[0:5000]
y_adult_train1 = y_adult_train[5000:10000]
y_adult_train2 = y_adult_train[10000:15000]
y_adult_train3 = y_adult_train[15000:20000]
y_adult_train4 = y_adult_train[20000:25000]
y_adult_train5 = y_adult_train[25000:30000]
y_adult_train6 = y_adult_train[30000:]

np.savez('y_adult_train0.npz', y_adult_train=y_adult_train0)
np.savez('y_adult_train1.npz', y_adult_train=y_adult_train1)
np.savez('y_adult_train2.npz', y_adult_train=y_adult_train2)
np.savez('y_adult_train3.npz', y_adult_train=y_adult_train3)
np.savez('y_adult_train4.npz', y_adult_train=y_adult_train4)
np.savez('y_adult_train5.npz', y_adult_train=y_adult_train5)
np.savez('y_adult_train6.npz', y_adult_train=y_adult_train6)

print("np savez Done!")
