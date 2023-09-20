# maic







[전략]

#1. 1st idea: machine learning
> Num. of Each develop values: 31

: Overfitting -> undersampling -> acc: 0.5

: Maybe be planning to build xgboost model and then...give up this idea


#2. 2nd idea: CNN
> Draw the a image of all develop values (Seperating the height)

> Draw an each develop values images and concat them


#3. 3rd idea: Time series-LSTM
> Multiple time series prediction

- 시그널 스펙트럼에서 모든 시그널이 동등하게 중요하지는 않을 것 같음
    - 일부 Frequency 는 생략하거나 가중치를 주고 정규화 하자는 아이디어
- 일부 ECG 시계열 그래프가 다른 것과 달랐음
    - 이상치는 제거하고 Clean 데이터로 학습데이터를 구성하는 것이 유리함
- 도메인 지식에 기반한 Clean 데이터 구성이 성능 개선에 필수적이라고 판단











[성능 개선을 위한 아이디어]

SMOTE
Imbalanced 데이터

total_develop_df['label'].value_counts()
>0    3252704
>1      33120
>Name: label, dtype: int64

양성이 1% 정도밖에 안됨





Up-sampling

이상한 데이터 일일이 수작업으로 삭제??












[F1]


랜덤 포레스트

0.1064 (submission 데이터의 N/A 처리 Interpolate)
0.0389 (submission 데이터의 N/A 처리 ffill)
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.3 / submission 데이터의 N/A 처리 Interpolate)
        음성 : 양성 비율이 거의 100 : 1 이라서
        855,559 개를 전부 0으로 라벨링 해도
        0.4976 이 나옴
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.001 / submission 데이터의 N/A 처리 Interpolate)
        전부 0으로 라벨링함

              precision    recall  f1-score   support

         0.0       0.49      0.49      0.49      5591
         1.0       0.50      0.50      0.50      5697

    accuracy                           0.50     11288
   macro avg       0.50      0.50      0.50     11288
weighted avg       0.50      0.50      0.50     11288

-----------------------------------------------------

나이브 베이즈

0.3493 (submission 데이터의 N/A 처리 Interpolate)
0.0627 (submission 데이터의 N/A 처리 ffill)
0.2173 (모델의 학습 데이터량 3,285,824 / Split 0.3 / submission 데이터의 N/A 처리 Interpolate)
0.2194 (모델의 학습 데이터량 3,285,824 / Split 0.001 / submission 데이터의 N/A 처리 Interpolate)

              precision    recall  f1-score   support

         0.0       0.50      0.71      0.59      5591
         1.0       0.52      0.30      0.38      5697

    accuracy                           0.50     11288
   macro avg       0.51      0.51      0.48     11288
weighted avg       0.51      0.50      0.48     11288

-----------------------------------------------------

Light GBM

0.4984 (submission 데이터의 N/A 처리 Interpolate)
0.4984 (submission 데이터의 N/A 처리 ffill)
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.3 / submission 데이터의 N/A 처리 Interpolate)
        전부 0으로 라벨링함
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.001 / submission 데이터의 N/A 처리 Interpolate)
        전부 0으로 라벨링함

              precision    recall  f1-score   support

         0.0       0.50      0.71      0.59      5591
         1.0       0.52      0.30      0.38      5697

    accuracy                           0.50     11288
   macro avg       0.51      0.51      0.48     11288
weighted avg       0.51      0.50      0.48     11288

-----------------------------------------------------

Gradient Boost

0.4125 (submission 데이터의 N/A 처리 Interpolate)
0.4305 (submission 데이터의 N/A 처리 ffill)
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.001 / submission 데이터의 N/A 처리 Interpolate)
        전부 0으로 라벨링함

              precision    recall  f1-score   support

         0.0       0.49      0.60      0.54      5591
         1.0       0.50      0.39      0.44      5697

    accuracy                           0.49     11288
   macro avg       0.49      0.49      0.49     11288
weighted avg       0.49      0.49      0.49     11288

-----------------------------------------------------

XGBOOST

0.3768 (submission 데이터의 N/A 처리 Interpolate)
0.4413 (submission 데이터의 N/A 처리 ffill)
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.001 / submission 데이터의 N/A 처리 Interpolate)
        전부 0으로 라벨링함

              precision    recall  f1-score   support

         0.0       0.49      0.61      0.55      5591
         1.0       0.50      0.39      0.44      5697

    accuracy                           0.50     11288
   macro avg       0.50      0.50      0.49     11288
weighted avg       0.50      0.50      0.49     11288

-----------------------------------------------------

Adaptive Boost

0.4415 (submission 데이터의 N/A 처리 Interpolate)
0.4925 (submission 데이터의 N/A 처리 ffill)
0.4976 (모델의 학습 데이터량 3,285,824 / Split 0.001 / submission 데이터의 N/A 처리 Interpolate)
        전부 0으로 라벨링함

              precision    recall  f1-score   support

         0.0       0.49      0.60      0.54      5591
         1.0       0.50      0.39      0.44      5697

    accuracy                           0.49     11288
   macro avg       0.49      0.50      0.49     11288
weighted avg       0.49      0.49      0.49     11288

-----------------------------------------------------

SGD Classifier

0.1583 (N/A, Interpolate)
0.0173 (N/A, ffill)

              precision    recall  f1-score   support

         0.0       0.50      0.90      0.64      5591
         1.0       0.52      0.11      0.17      5697

    accuracy                           0.50     11288
   macro avg       0.51      0.50      0.41     11288
weighted avg       0.51      0.50      0.41     11288










[데이터 전처리]

- Data-centric AI
    - 초반 데이터 전처리가 중요
    - 데이터 전처리가 안되고, Clean 데이터가 아닌데
    - 모델만 복잡하게 적용한다고 성능이 올라가지 않음
- Signal 데이터에 집중
    - 시계열 데이터를 이미지로 변환하여 CNN 적용 
    - 이미지 Classification 에 기반하여 저산소혈증 여부를 진단
- Demographic 데이터
    - Demographic Data 는 단순 참조용이라는 임상의 의견
    - 환자의 나이, 성별, 몸무게, 키 등의 정보를 활용해서
        - 같은 시그널이라도 다르게 판단
- Demographic 데이터를 다른 데이터와 어떻게 합칠 수 있을까??
    - Demographic 데이터를 시그널 데이터와 연관 지을 때
        - Data_id 와 매핑 가능
    - Demographic 데이터를 CDM 데이터와 연관 지을 때
        - Person_id 와 opdata 와 매핑 가능
- CDM (Measurement Data)
    - Demographic 데이터의 환자 id 와 결합해서 사용 가능













[데이터 설명]

- Demographic Data
    - Signal data & CDM data
    - 임상정보는 나이, 성별, 키, 몸무게, 검사 데이터로 구성되어있으며, 
        - 환자 혹은 수술 파일 간 식별이 가능한 File ID와 함께 csv 파일로 제공됨.
    - Age, sex, height, weight
        - EMR(Electronic Medical Records)에서 수집
- Signal Data
    - 생체신호 데이터는 5가지(SpO2, EtCO2, FiO2, TV, PIP)를 포함하며, 
        1) 산소포화도(SpO2)
            - 국소 관류를 통한 동맥혈의 산소수준 측정
        2) 호기말 이산화탄소 분압(EtCO2, End tidal CO2)
            - 환기 상태가 적절한지 볼 수 있는 지표
            - 기관 내 튜브에 연결하여 호기 시 폐포에서 배출되는 이산화탄소 분압을 측정
        3) 흡입 산소 농도(FiO2, fraction of inspired oxygen)
            - 흡기 시 흡입하는 기체 중 산소가 차지하는 부피 분율
        4) 일회 호흡량(TV, tidal volume)
            - 일회 호흡 당 흡기시 전달되는 공기의 양
        5) 최고 흡기 압력 (PIP, peak inspiratory pressure)
            - 설정된 일회호흡량에 의한 환기시 최대 흡기 기도 압력
    - 각 수술 파일을 1분 단위로 자른 segments로 구성되어있음. 
    - Dictionary 자료형을 Python Pickle File(.pkl or .p)파일 형식으로 제공함.
    - 데이터에 대한 라벨은 
        - 1분 단위의 segement당 1분 후 데이터에서의 저산소혈증 발생 유무를 0/1로 표시해둠. 
        - 생체신호 데이터 파일에 함께 Dictionary 자료형, Python Pickle File(.pkl or .p)파일 형식으로 제공
- CDM Data
    - 환자의 검사데이터는 OMOP CDM 의 Measurement Table 형태로 제공되며, 
        - File ID 와 함께 제공되는 Person_id 를 통해 Vital 데이터를 연계할 수 있음.









주어진 Develop 데이터 중 일부에서 (Develop_637 등) hypoxemia 발생이 아예 관측되지 않는, 
모든 label이 0인 환자가 존재하는 것을 확인하였습니다.
본 대회의 데이터는 hypoxemia 발생 환자들만을 대상으로 한 것으로 알고 있는데, 
그렇다면 수술 과정 중 일부만 제공하는 과정에서 hypoxemia가 발생하지 않은 부분만 포함된 것인지, 
아니면 다른 이유가 있는 것인지 궁금합니다.

Develop_731, 
Develop_905 환자의 경우 
weight가 -1.0, height값은 표기되어 있지 않은데 
이는 측정값 오류로 인한 것인지, 
어떻게 해석하는 것이 옳은지도 궁금합니다.


대회 데이터 정의상 SpO2<90미만인 hypoxemia가 발생한 데이터만 포함되어야하나, 
최소 SpO2값이 90이어서 전체 구간의 라벨이 0인 데이터가 일부 포함되어있습니다. 
이 부분을 미리 공지드리지 못한 점 죄송합니다. 

최소 SpO2값이 90인 데이터들도 임상선생님들께서 확인해주실 때 hypoxemia event가 발생하였다고 여겨져서 코호트에 포함되었지만, 
대회에서의 hypoxemia label=1 조건인 SpO2<90은 충족하지 못하여 해당 데이터들이 포함된 것으로 보입니다

아래 데모에 대한 질문은 
측정값 오류이기 때문에 해석하거나 대체하는 것은 참가자 자율입니다. 감사합니다.














[이미지 전환 및 Merge]

- Input Data 전처리
    - XYZ 3가지 컬럼에서 시계열 이미지를 저장함
        - Normal 와 osa 구분해서 개별 저장
    - ECG 1가지 컬럼에서 시계열 이미지를 저장함
        - Normal 와 osa 구분해서 개별 저장
    - Mel 스펙트로그램 20가지 컬럼에서 시계열 이미지를 저장함
        - Normal 와 osa 구분해서 개별 저장
    - XYZ, ECG, Mel 3가지 이미지를 하나로 Merge 함
        - Normal 와 osa 구분해서 개별 저장
- Test Data 전처리
    - XYZ 3가지 컬럼에서 시계열 이미지를 저장함
    - ECG 1가지 컬럼에서 시계열 이미지를 저장함
    - Mel 스펙트로그램 20가지 컬럼에서 시계열 이미지를 저장함
        - XYZ, ECG, Mel 3가지 이미지를 하나로 Merge 함
- 이미지 용량 줄이기
   - 현업 데이터에 오컴의 면도날처럼 데이터 feature 를 줄이면,
   - 학습 속도도 증대되고,
   - 일부 수준에서는 Accuracy 가 증대됨.
- 이미지 사이즈 증대
    - IMAGE_SIZE = (150,150) 로 설정해보고
        - 사이즈를 크게 해봤으나 성능이 크게 좋아지지는 않았음 
- 데이터 Augmentation
    - [Data augmentation Techniques](https://iq.opengenus.org/data-augmentation/)
    - Image augmentationdataset size 작아서training set 을 다양하게 하기 위한 솔루션
    - 데이터 어그멘테이션 (블러, 그레이스케일, 크롭, 리사이즈 등)
        - Grayscale, EdgeDetection, CenterCrop, Flip, Rotate …
        - 예상했던 것 보다, GrayScale, EdgeDetection 효과 크지 않음. 칼라 정보도 필요함
        - CenterCrop의 경우 학습에 따라 다른 테스트해 본 방법들에 비해 Accuracy가 증가, 감소 폭이 큼.










[Obstacle 과 문제 해결]

- Obstacle 1
    - 이미지(.png)로 변환된 데이터를 
        - 일괄적으로 CNN 모델에 넣으려니까
        - 정렬이 되지 않는 문제가 발생함
        - sort() 함수 사용시, 한 자리, 두 자리 숫자 혼합됨
    - 솔루션
        - import natsort 를 적용해서
            - 데이터 정렬 이슈를 해결함
- Obstacle 2
    - 커널이 다운되는 이슈가 발생함
    - 솔루션
        - 120장의 이미지를 한 번에 생성하는 것이 아니라
            - 30장씩 4회 나눠서 진행함











[데이터 전처리]

- Data Augmentation
    - 크로마토그램, 시계열 그래프를 Flip, Crop 등등 해도 괜찮은가?
    J의견: Data augmentation에서 feature에 영향을 주는 기법은 하지 않아야 함으로 Crop은 사용하지 않는것이 좋다고 생각합니다. 이런 그래프 파형의 이미지는 특히나 그래프 파형이 feature로 잡히기 때문에...






[모델]

- 레이어를 얕게 쌓을 경우 (에폭 50)
    - 전부 1로 예측함
- 레이어를 깊게 쌓을 경우 (에폭 5)
    - 전부 0으로 예측함

- RandomForestClassifier(n_estimators=10~100)
    - 전부 0으로 예측함 > Undersampling 시도: 정확도 50%







[외부 모델]

- ResNet18
    - 단순한 이진분류에서는 가볍고 성능이 가장 적합함
- 모델
    - PyTorch resnext50_32x4d
    - 다양한 모델 구현 후 결국엔 ensamble 적용
    







[파라미터 최적화]

- Optimizer
    - 시간 부족으로 다양한 hyperparameter 시도가 어려웠음
        - Global optimum을 찾기 위한 genetic algorithm 시도 고려
    - SGD위주로, learning rate 단순변경
    - Cosine Annealing Warm Restarts를 이용하여 learning rate 조절 시도
    - Stochastic weight Averaging를 이용하여learning rate 조절 시도
- Loss function
    - 기본적으로는 Cross Entropy loss를 이용
    - 다른 loss function인 Focal loss도 이용했음
- 파라미터 조정
   - 러닝레이트, 배치사이즈, 에폭



