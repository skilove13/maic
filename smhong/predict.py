import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd

# 데이터셋 경로
directory_path = '/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/dataset/'

# Validation 데이터셋 로드
# data_valid = np.load(directory_path + 'ecg_adult_data_valid.npz')
# x_val_adult = data_valid['ecg_adult_data_valid']

data_valid = np.load(directory_path + 'ecg_child_data_valid.npz')
x_val_child = data_valid['ecg_child_data_valid']

# 모델 로드
model_path = '/users/VC/sungman.hong/PycharmProjects/pythonProject/maic/model'
model = tf.keras.models.load_model(model_path + '/resnet19_model_child2.h5')

# 예측
#predictions = model.predict(x_val_adult)
predictions = model.predict(x_val_child)

# 예측 결과 출력
predicted_ages = np.argmax(predictions, axis=1)
print("Predicted ages:", predicted_ages)

# 예측 결과를 CSV 파일로 저장
df = pd.DataFrame({'Predicted Age': predicted_ages})
#df.to_csv('predicted_adult_ages.csv', index=False)
df.to_csv('predicted_child_ages1.csv', index=False)

print("Done!")