#!/usr/bin/env python
# coding: utf-8

# In[18]:


#(5000,5000,12) 형태의 npz파일을  -> (5000, 4096,12) npz파일로 변환
#  1) (5000,5000,12) 형태의 npz파일이 있는 디렉토리 안에서 아래 코드 실행하면 자동으로 trim_ecg_adult_data*.npz와 trim_ecg_child_data*.npz 를 생성

import numpy as np
import glob

root = '/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v2/'

# ecg_adult_data*.npz 파일 로드 및 데이터 재구조화

for file_path in glob.glob(root + 'ecg_adult_data*.npz'):
    # npz 파일 로드
    data = np.load(file_path)
    
    # 데이터 모양 확인
    print(data['ecg_adult_data_array'].shape)  # (5000, 5000, 12)
    
    # 데이터 재구조화
    # (5000, 4096, 12)로 trim (앞뒤로 455개 행씩 제거)
    trim_size = (data['ecg_adult_data_array'].shape[1] - 4096) // 2
    trimmed_data = data['ecg_adult_data_array'][:, trim_size:trim_size+4096, :]
    
    # 변경된 모양 확인
    print(trimmed_data.shape)  # (5000, 4096, 12)
    
    # 변경된 모양을 새로운 npz 파일로 저장
    trimmed_file_path = file_path.replace('ecg_adult_data', 'trim_ecg_adult_data')
    np.savez(trimmed_file_path, ecg_adult_data_array=trimmed_data)

    print(f"Trimmed data saved to {trimmed_file_path}")



# In[25]:


root = '/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v2/'

# ecg_child_data*.npz 파일 로드 및 데이터 재구조화
for file_path in glob.glob(root + 'ecg_child_data*.npz'):
    # npz 파일 로드
    data = np.load(file_path)
    
    # 데이터 모양 확인
    print(data['ecg_child_data_array'].shape)  # (5000, 5000, 12)
    
    # 데이터 재구조화
    # (5000, 4096, 12)로 trim (앞뒤로 455개 행씩 제거)
    trim_size = (data['ecg_child_data_array'].shape[1] - 4096) // 2
    trimmed_data = data['ecg_child_data_array'][:, trim_size:trim_size+4096, :]
    
    # 변경된 모양 확인
    print(trimmed_data.shape)  # (5000, 4096, 12)
    
    # 변경된 모양을 새로운 npz 파일로 저장
    trimmed_file_path = file_path.replace('ecg_child_data', 'trim_ecg_child_data')
    np.savez(trimmed_file_path, ecg_adult_data_array=trimmed_data)

    print(f"Trimmed data saved to {trimmed_file_path}")

