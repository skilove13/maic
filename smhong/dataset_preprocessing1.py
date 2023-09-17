# library
import json
import os
from tqdm import tqdm
# from dataloader import BatchDataloader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import glob

directory_path='E:/smhong/micai_ai_challange/dataset/ECG_child_numpy_train'

#ecg_child_data_path를 glob.glob() 함수를 사용하여 리스트로 저장하는데 이름순으로 순차적으로 정렬
ecg_child_data_path = glob.glob("E:/smhong/micai_ai_challange/dataset/ECG_child_numpy_train/ecg_child_*.npy")

ecg_child_data = []
for i in range(len(ecg_child_data_path)):
    #ecg_child_data[i] = glob.glob("E:/smhong/micai_ai_challange/dataset/ECG_child_numpy_train/" + "ecg_child_" + str(i) + ".npy")
    file_path = "E:/smhong/micai_ai_challange/dataset/ECG_child_numpy_train/" + "ecg_child_" + str(i) + ".npy"
    ecg_child_data.append(file_path)

print(ecg_child_data_path[0:10])
print(ecg_child_data[0:100])
print("len(ecg_child_data) = ",len(ecg_child_data))

ecg_data_example = np.load(ecg_child_data[0])
print("ecg_data_example[0].shape=" , ecg_data_example.shape)
print("ecg_data_example[0]=", ecg_data_example)

# 데이터를 저장할 리스트 초기화
ecg_data_list = []
# 각 파일을 읽어와서 리스트에 추가
for file_path in ecg_child_data[0:]:
    ecg_data = np.load(file_path)
    # 데이터를 12개로 나누고 (5000, 12) 형식으로 저장
    ecg_data_split = np.array_split(ecg_data, 5000)
    ecg_data_list.append(ecg_data_split)

# 리스트를 Numpy 배열로 변환
ecg_child_data_array = np.array(ecg_data_list)

# ecg_data_array의 shape 확인
print("ecg_data_array의 shape:", ecg_child_data_array.shape)

# npz 파일로 저장
np.savez('ecg_child_data.npz', ecg_child_data_array=ecg_child_data_array)

