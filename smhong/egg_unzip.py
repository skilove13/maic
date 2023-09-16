import zipfile

egg_file_path = 'E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train.egg'  # 압축 해제할 .egg 파일의 경로

# 압축 해제할 디렉토리 지정
extracted_dir = 'E:/smhong/micai_ai_challange/dataset/ECG_adult_numpy_train'

# .egg 파일 열기
with zipfile.ZipFile(egg_file_path, 'r') as zip_ref:
    # .egg 파일의 모든 내용을 압축 해제할 디렉토리에 추출
    zip_ref.extractall(extracted_dir)

print(f'{egg_file_path}를 {extracted_dir}로 성공적으로 압축 해제했습니다.')
