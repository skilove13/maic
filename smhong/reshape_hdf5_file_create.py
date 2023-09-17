#2. 데이터를 (5000, 12)로 reshape후 hdf5 파일만들기
import h5py
import os
import numpy as np

dataset_path = 'E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train'
output_file = 'E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train/ECG_adult_numpy_train.h5'  # 저장할 hdf5 파일 이름

# # .npy 파일들을 리스트로 로드
# npy_files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]
# print(npy_files[0:10])
# data = np.concatenate([np.load(os.path.join(dataset_path, f)) for f in npy_files])
#
# # (5000, 12)로 reshape
# data_reshaped = data.reshape(-1, 5000, 12)
#
# # HDF5 파일로 저장
# with h5py.File(output_file, 'w') as hf:
#     hf.create_dataset('tracings', data=data_reshaped)  # 컬럼의 이름을 tracings라고 명명
#
# # HDF5 파일 생성확인
# file_exists = os.path.exists('E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_valid.h5')
# print(file_exists)

# .npy 파일들을 리스트로 로드
npy_files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]

# 파일 이름에서 숫자 부분을 추출하여 정렬
npy_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 정렬된 파일 리스트 출력
print(npy_files[0:10])