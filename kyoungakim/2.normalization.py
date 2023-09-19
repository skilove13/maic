#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np

# 정규화 방법 
# 1) lead (axis=2) 별로 정규화함

# # 1. 데이터 불러오기

directory_path = '/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input/input_v2/'

data = np.load(directory_path + 'ecg_child_data0.npz')                      #이름 바꾸기
# npz 파일에서 배열을 추출합니다. 
data_array = data['ecg_adult_data_array']                                    #이름 확인


# 세 번째 차원(lead)을 정규화합니다.
mean = np.mean(data_array, axis=2)
std = np.std(data_array, axis=2)

# 분모가 0이 되는 경우를 처리하기 위해 epsilon 값을 추가하여 나눗셈을 수행합니다.
epsilon = 1e-8  # 아주 작은 값을 설정합니다.
normalized_data = (data_array - mean[:, :, np.newaxis]) / (std[:, :, np.newaxis] + epsilon)


 #3. 정규화된 데이터 저장 
np.savez('normalized_ecg_child_data0.npz', normalized_data)                #이름 바꾸기

# 첫 번째 배열 가져와 확인하기
root = '/home/work/ecg-age-prediction/ecg-age-prediction-main/'
data = np.load(root + 'normalized_ecg_child_data0.npz')                    #이름 바꾸기
first_array = data[list(data.keys())[0]]
print(data.files)

