import tensorflow as tf
from tensorflow.keras import layers, models

def create_swin_transformer_model():
    # 모델 생성
    model = models.Sequential()

    # Swin Transformer 모델 정의
    def swin_transformer_model():
        num_classes = 9  # 클래스 수

        # Swin Transformer 모델 생성
        model = SwinTransformer(
            num_classes=num_classes,
            num_layers=12,
            num_heads=4,
            window_size=7,
            expand_ratio=4,
            dropout_rate=0.2,
            patch_size=4,
            hidden_dim=128,
            mlp_dim=256
        )

        return model

    # 입력 레이어
    inputs = layers.Input(shape=(5000, 12))

    # Swin Transformer 모델
    x = swin_transformer_model()(inputs)

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=x)

    return model




class SwinTransformer(layers.Layer):
    def __init__(self, num_classes, num_layers, num_heads, window_size, expand_ratio, dropout_rate, patch_size, hidden_dim, mlp_dim):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window_size = window_size
        self.expand_ratio = expand_ratio
        self.dropout_rate = dropout_rate
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.patch_embedding = layers.Conv1D(hidden_dim, kernel_size=patch_size, strides=patch_size, padding='valid')
        self.positional_embedding = layers.Embedding(input_dim=5000, output_dim=hidden_dim)

        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerLayer(num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout_rate=dropout_rate)
            )

        self.layer_norm = layers.LayerNormalization(epsilon=1e-5)
        self.pooling = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        x += self.positional_embedding(tf.range(inputs.shape[1]))

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        x = self.layer_norm(x)
        x = self.pooling(x)
        x = self.fc(x)

        return x

class TransformerLayer(layers.Layer):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout_rate):
        super(TransformerLayer, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.multihead_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = models.Sequential([
            layers.Dense(mlp_dim, activation='relu'),
            layers.Dense(hidden_dim)
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs):
        x = self.multihead_attention(inputs, inputs)
        x = self.dropout1(x)
        x = x + inputs
        x = self.layer_norm1(x)

        y = self.mlp(x)
        y = self.dropout2(y)
        y = y + x
        y = self.layer_norm2(y)

        return y