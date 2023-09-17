"""
1. ECG_adult_numpy_train폴더의 파일들을 읽어서 리스트 형태로 저장
2. 리스트 형태의 데이터를 12개로 나누고 하나의 npz파일 60000 -> (5000, 12) 형식으로 저장
3. 5000개씩 나누어서 ecg_adult_data0.npz ~ ecg_adult_data6.npz저장
"""

# library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time

dataset_path = 'E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train'

#ecg_adult_data_path를 glob.glob() 함수를 사용하여 리스트로 저장하는데 이름순으로 순차적으로 정렬
ecg_adult_data_path = glob.glob("E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train/ecg_adult_*.npy")

print(len(ecg_adult_data_path))

ecg_adult_data = []

# .npy 파일들을 리스트로 로드
npy_files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]

# 파일 이름에서 숫자 부분을 추출하여 정렬
npy_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
# 정렬된 파일 리스트 출력
print(npy_files[0:100])
print("len(npy_files) = ",len(npy_files))
print("type of npy_files = ",type(npy_files))

for i in range(len(npy_files)):
    file_path = "E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train/" + str(npy_files[i])
    ecg_adult_data.append(file_path)

# print(ecg_adult_data[0:30])
print("len(ecg_adult_data) = ",len(ecg_adult_data))


#
ecg_data_example = np.load(ecg_adult_data[0])
print("ecg_data_example[0].shape=" , ecg_data_example.shape)
print("ecg_data_example[0]=", ecg_data_example)
#
# # 데이터를 저장할 리스트 초기화
ecg_data_list = []
i = 0
# 각 파일을 읽어와서 리스트에 추가
#for file_path in ecg_adult_data[0:5000]:
#for file_path in ecg_adult_data[5000:10000]:
#for file_path in ecg_adult_data[10000:15000]:
#for file_path in ecg_adult_data[15000:20000]:
#for file_path in ecg_adult_data[20000:25000]:
#for file_path in ecg_adult_data[25000:30000]:
for file_path in ecg_adult_data[30000:]:
    ecg_data = np.load(file_path)
    # 데이터를 12개로 나누고 (5000, 12) 형식으로 저장
    ecg_data_split = np.array_split(ecg_data, 5000)
    ecg_data_list.append(ecg_data_split)
    print(i)
    i += 1

# # 리스트를 Numpy 배열로 변환
ecg_adult_data_array = np.array(ecg_data_list)

# ecg_data_array의 shape 확인
print("ecg_data_array의 shape:", ecg_adult_data_array.shape)

start = time.time()  # 시작 시간 저장
# npz 파일로 저장
#np.savez('ecg_adult_data0.npz', ecg_adult_data_array=ecg_adult_data_array)
np.savez('ecg_adult_data6.npz', ecg_adult_data_array=ecg_adult_data_array)

end = time.time()  # 종료 시간 저장
print("npz save time :", end - start)  # 현재시각 - 시작시간 = 실행 시간
print("ecg_adult_data_array[0].shape=", ecg_adult_data_array[0].shape)

