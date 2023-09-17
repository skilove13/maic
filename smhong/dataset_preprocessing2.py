"""
1. ECG_adult_numpy_train폴더의 파일들을 읽어서 리스트 형태로 저장
2. 리스트 형태의 데이터를 12개로 나누고 하나의 npz파일 60000 -> (5000, 12) 형식으로 저장
3. ecg_adult_data.npz파일로 저장
"""

# library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time

directory_path='E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train'

#ecg_adult_data_path를 glob.glob() 함수를 사용하여 리스트로 저장하는데 이름순으로 순차적으로 정렬
ecg_adult_data_path = glob.glob("E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train/ecg_adult_*.npy")

ecg_adult_data = []
for i in range(len(ecg_adult_data_path)):
    file_path = "E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train/" + "ecg_adult_" + str(i) + ".npy"
    ecg_adult_data.append(file_path)

print(ecg_adult_data_path[0:10])
print(ecg_adult_data[0:10])
print("len(ecg_child_data) = ",len(ecg_adult_data))

ecg_data_example = np.load(ecg_adult_data[0])
print("ecg_data_example[0].shape=" , ecg_data_example.shape)
print("ecg_data_example[0]=", ecg_data_example)

# 데이터를 저장할 리스트 초기화
ecg_data_list = []
i = 0
# 각 파일을 읽어와서 리스트에 추가
for file_path in ecg_adult_data[0:]:
    ecg_data = np.load(file_path)
    # 데이터를 12개로 나누고 (5000, 12) 형식으로 저장
    ecg_data_split = np.array_split(ecg_data, 5000)
    ecg_data_list.append(ecg_data_split)
    print(i)
    i += 1

# 리스트를 Numpy 배열로 변환
ecg_adult_data_array = np.array(ecg_data_list)

# ecg_data_array의 shape 확인
print("ecg_data_array의 shape:", ecg_adult_data_array.shape)

start = time.time()  # 시작 시간 저장
# npz 파일로 저장
np.savez('ecg_adult_data.npz', ecg_adult_data_array=ecg_adult_data_array)

end = time.time()  # 종료 시간 저장
print("npz save time :", end - start)  # 현재시각 - 시작시간 = 실행 시간

