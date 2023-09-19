## 데이터 npz를 반으로 나누어서 저장하는 코드입니다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2

# This method loads your data from the .npz file
directory_path = 'E:/smhong/micai_ai_challange/maic_github/maic/smhong/'

data = np.load(directory_path + 'ecg_child_data0.npz')
X_train_child = data['ecg_child_data_array']
print("X_train_child.shape=", X_train_child.shape)


split_index = len(X_train_child) // 2
X_train_split_child1 = X_train_child[:split_index]
X_train_split_child2 = X_train_child[split_index:]

# 나눈 데이터를 각각의 npz 파일로 저장합니다.
np.savez('X_train_split_child1.npz', X_train_split_child1)
np.savez('X_train_split_child2.npz', X_train_split_child2)

print("X_train_split_child1.shape =", X_train_split_child1.shape)
print("X_train_split_child2.shape =", X_train_split_child2.shape)

print("end of program")