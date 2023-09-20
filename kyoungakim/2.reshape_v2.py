#!/usr/bin/env python
# coding: utf-8

# In[3]:


#원본 데이터(5000, 12) reshape -> (4096,12) trim -> hdf5 파일만들기  

import os
import numpy as np
import h5py

# 디렉토리 경로
directory_path = '/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/ECG_child_numpy_train'
output_file = '/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v3/m_ECG_child_numpy_train.h5'  # 저장할 hdf5 파일 이름

# 디렉토리 내의 모든 .npy 파일 가져오기
npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

# 파일 이름에서 숫자 부분을 추출하여 정렬
npy_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 정렬된 파일 리스트 출력
print(npy_files)

# 결과를 저장할 리스트 초기화
result_data = []

# .npy 파일 읽기 및 처리
for npy_file in npy_files:
    # .npy 파일 불러오기
    data = np.load(os.path.join(directory_path, npy_file))
    
    # (5000, 12)로 reshape
    data_reshaped = data.reshape(5000, 12)
    
    # (4096, 12)로 trim (앞뒤로 455개 행씩 제거)
    trim_size = (data_reshaped.shape[0] - 4096) // 2
    trimmed_data = data_reshaped[trim_size:trim_size + 4096]
    
    # 결과 리스트에 추가
    result_data.append(trimmed_data)

# 결과를 하나의 NumPy 배열로 변환
final_result = np.array(result_data)
print("final_result의 shape:", final_result.shape)

# final_result는 이제 (5000, 4096, 12) 모양의 NumPy 배열입니다.


##########
# HDF5 파일로 저장
with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('tracings', data=result_data)  # 컬럼의 이름을 tracings라고 명명

# HDF5 파일 생성확인
file_exists = os.path.exists('/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v3/m_ECG_child_numpy_train.h5')
print(file_exists)


# In[5]:


import pandas as pd

# CSV 파일 경로
csv_file_path = "/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v3/ECG_child_age_train.csv"  # 실제 파일 경로로 변경하세요.

# CSV 파일을 pandas DataFrame으로 읽어오기
df = pd.read_csv(csv_file_path)

# 열 이름 변경
df.rename(columns={"AGE": "age"}, inplace=True)  # "AGE"를 "age"로 변경

# 변경된 DataFrame을 새로운 CSV 파일로 저장 (선택사항)
new_csv_file_path = "/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v3/ECG_child_s_age_train.csv"  # 새로운 파일 경로로 변경 가능
df.to_csv(new_csv_file_path, index=False)  # index를 저장하지 않음

# 변경된 DataFrame 확인
print(df)


# In[ ]:




