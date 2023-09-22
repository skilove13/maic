import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_model():
    # 모델 생성
    model = models.Sequential()

    # ResNet 블록 정의
    def resnet_block(inputs, filters, kernel_size, strides):
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, inputs])
        x = layers.Activation('relu')(x)
        return x

    # 입력 레이어
    inputs = layers.Input(shape=(5000, 12))

    # 1D 컨볼루션 레이어
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # ResNet 블록 3개
    x = resnet_block(x, filters=64, kernel_size=3, strides=1)
    x = resnet_block(x, filters=64, kernel_size=3, strides=1)
    x = resnet_block(x, filters=64, kernel_size=3, strides=1)

    # 전연결 레이어
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(123, activation='softmax')(x)  # 123개의 클래스에 대한 softmax 출력

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=x)

    return model


def resnet19_model(dropout_rate=0.2):
    # 모델 생성
    model = models.Sequential()

    # ResNet 블록 정의
    def resnet_block(inputs, filters, strides):
        x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        #Dropout 추가
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        shortcut = inputs
        if strides != 1 or filters != inputs.shape[-1]:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # 입력 레이어
    inputs = layers.Input(shape=(5000, 12))

    # 1D 컨볼루션 레이어
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # # ResNet 블록 4개 , child block
    # x = resnet_block(x, filters=64, strides=1)
    # x = resnet_block(x, filters=64, strides=1)
    # x = resnet_block(x, filters=128, strides=2)
    # x = resnet_block(x, filters=128, strides=1)

    # ResNet 블록 4개 , adult block
    x = resnet_block(x, filters=128, strides=1)
    x = resnet_block(x, filters=196, strides=1)
    x = resnet_block(x, filters=256, strides=2)
    x = resnet_block(x, filters=320, strides=1)

    # 전연결 레이어
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(123, activation='softmax')(x)  # 123개의 클래스에 대한 softmax 출력

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=x)

    return model


def resnet32_model():
    # 모델 생성
    model = models.Sequential()

    # ResNet 블록 정의
    def resnet_block(inputs, filters, strides):
        x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        shortcut = inputs
        if strides != 1 or filters != inputs.shape[-1]:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # 입력 레이어
    inputs = layers.Input(shape=(5000, 12))

    # 1D 컨볼루션 레이어
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # ResNet 블록 8개
    x = resnet_block(x, filters=64, strides=1)
    for _ in range(7):
        x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=128, strides=2)
    for _ in range(7):
        x = resnet_block(x, filters=128, strides=1)

    # Dropout 레이어 추가
    x = layers.Dropout(0.3)(x)  # 0.5는 드롭아웃 비율을 나타냅니다.

    # 전연결 레이어
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(123, activation='softmax')(x)  # 123개의 클래스에 대한 softmax 출력

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=x)

    return model


def resnet56_model():
    # 모델 생성
    model = models.Sequential()

    # ResNet 블록 정의
    def resnet_block(inputs, filters, strides):
        x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        shortcut = inputs
        if strides != 1 or filters != inputs.shape[-1]:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # 입력 레이어
    inputs = layers.Input(shape=(5000, 12))

    # 1D 컨볼루션 레이어
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # ResNet 블록 18개
    x = resnet_block(x, filters=64, strides=1)
    for _ in range(5):
        x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=128, strides=2)
    for _ in range(5):
        x = resnet_block(x, filters=128, strides=1)
    x = resnet_block(x, filters=256, strides=2)
    for _ in range(5):
        x = resnet_block(x, filters=256, strides=1)

    # Dropout 레이어 추가
    x = layers.Dropout(0.3)(x)  # 0.5는 드롭아웃 비율을 나타냅니다.

    # 전연결 레이어
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(123, activation='softmax')(x)  # 123개의 클래스에 대한 softmax 출력

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=x)

    return model

def resnet110_model():
    # 모델 생성
    model = models.Sequential()

    # ResNet 블록 정의
    def resnet_block(inputs, filters, strides):
        x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        shortcut = inputs
        if strides != 1 or filters != inputs.shape[-1]:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # 입력 레이어
    inputs = layers.Input(shape=(5000, 12))

    # 1D 컨볼루션 레이어
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # ResNet 블록 36개
    x = resnet_block(x, filters=64, strides=1)
    for _ in range(11):
        x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=128, strides=2)
    for _ in range(11):
        x = resnet_block(x, filters=128, strides=1)
    x = resnet_block(x, filters=256, strides=2)
    for _ in range(11):
        x = resnet_block(x, filters=256, strides=1)

    # Dropout 레이어 추가
    x = layers.Dropout(0.3)(x)  # 0.5는 드롭아웃 비율을 나타냅니다.

    # 전연결 레이어
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(123, activation='softmax')(x)  # 123개의 클래스에 대한 softmax 출력

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=x)

    return model