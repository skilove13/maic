# library
import json
#import torch
import os
from tqdm import tqdm
# from dataloader import BatchDataloader
#import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

#directory_path='/home/work/ecg-age-prediction/ecg-age-prediction-main/competition/input'
directory_path='E:/smhong/micai_ai_challange/dataset/ECG_child_numpy_train'  # 압축 해

ROOT = "E:/smhong/micai_ai_challange/dataset/"
item = pd.read_csv(ROOT + 'ECG_child_age_train.csv')

print("=====================================")
print("======ECG_child_age_train.csv========")

print(item.head())
print('legth:',len(item))
print('결측치:', item.isna().sum())
duplicates_count = item['FILENAME'].duplicated().sum()
print("중복된 값의 개수:", duplicates_count)

print('summary:', item.describe())

category_counts=item['GENDER'].value_counts()
print(item['GENDER'].value_counts())
category_counts1=item['AGE'].value_counts()
print(item['AGE'].value_counts())

# barplot
plt.bar(category_counts.index, category_counts.values, color='skyblue')
plt.xlabel('GENDER')
plt.ylabel('COUNTS')
#plt title 추가 
plt.title('ECG_child_age_train.csv')
plt.show()

# barplot
plt.bar(category_counts1.index, category_counts1.values, color='skyblue')
plt.xlabel('AGE')
plt.ylabel('COUNTS')
#plt title 추가
plt.title('ECG_child_age_train age')
plt.show()

# # histigram
plt.hist(item['AGE'], bins=5, color='skyblue', edgecolor='black')

print("=====================================")
print("======ECG_adult_age_train.csv========")
item1 = pd.read_csv(ROOT + 'ECG_adult_age_train.csv')

print(item1.head())
print('legth:',len(item1))
print('결측치:', item1.isna().sum())
duplicates_count = item1['FILENAME'].duplicated().sum()
print("중복된 값의 개수:", duplicates_count)

print('summary:', item1.describe())

category_counts1=item1['GENDER'].value_counts()
print(item1['GENDER'].value_counts())

category_counts2=item1['AGE'].value_counts()
print(item1['AGE'].value_counts())

# barplot
plt.bar(category_counts1.index, category_counts1.values, color='skyblue')
plt.xlabel('GENDER')
plt.ylabel('COUNTS')
#plt title 추가 
plt.title('ECG_child_adult_train.csv')
plt.show()

# barplot
plt.bar(category_counts2.index, category_counts2.values, color='skyblue')
plt.xlabel('AGE')
plt.ylabel('COUNTS')
#plt title 추가
plt.title('ECG_child_adult_train age')
plt.show()

# %% Read datasets
# for dirname, _, filenames in os.walk(directory_path):  #디렉토리 변경
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
        
