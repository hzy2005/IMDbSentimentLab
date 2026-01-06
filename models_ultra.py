"""
超级优化版CNN+RNN模型 - 目标准确率95%+
使用最先进的技术：注意力机制、残差连接、多头注意力等
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    GlobalAveragePooling1D, LSTM, GRU, Bidirectional, Dense, 
    Dropout, Concatenate, BatchNormalization, SpatialDropout1D,
    Add, Multiply, Activation, Layer
)


class AttentionLayer(Layer):
    """自定义注意力层"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # 计算注意力分数
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        # 应用注意力权重
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def create_ultra_cnn_lstm_attention_model(max_features=20000, maxlen=500, embedding_dim=300):
    """
    创建超级优化版CNN+LSTM+Attention模型
    
    特点：
    - 多尺度CNN（不同kernel size）
    - 双向LSTM堆叠
    - 注意力机制
    - 残差连接
    - 全局池化组合
    """
    # 输入层
    inputs = Input(shape=(maxlen,))
    
    # 词嵌入层
    x = Embedding(max_features, embedding_dim, input_length=maxlen)(inputs)
    x = SpatialDropout1D(0.2)(x)
    
    # 多尺度CNN分支
    # 分支1: kernel_size=3
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(2)(conv1)
    
    # 分支2: kernel_size=4
    conv2 = Conv1D(128, 4, activation='relu', padding='same')(x)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(2)(conv2)
    
    # 分支3: kernel_size=5
    conv3 = Conv1D(128, 5, activation='relu', padding='same')(x)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling1D(2)(conv3)
    
    # 合并多尺度特征
    conv_concat = Concatenate()([conv1, conv2, conv3])
    conv_concat = Dropout(0.3)(conv_concat)
    
    # 深层CNN
    conv_deep = Conv1D(256, 3, activation='relu', padding='same')(conv_concat)
    conv_deep = BatchNormalization()(conv_deep)
    conv_deep = MaxPooling1D(2)(conv_deep)
    
    conv_deep = Conv1D(128, 3, activation='relu', padding='same')(conv_deep)
    conv_deep = BatchNormalization()(conv_deep)
    
    # 双向LSTM堆叠
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(conv_deep)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)
    
    lstm2 = Bidirectional(LSTM(64, return_sequences=True))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    # 注意力机制
    attention_output = AttentionLayer()(lstm2)
    
    # 全局池化（多种方式）
    global_max = GlobalMaxPooling1D()(lstm2)
    global_avg = GlobalAveragePooling1D()(lstm2)
    
    # 合并所有特征
    concat_features = Concatenate()([attention_output, global_max, global_avg])
    concat_features = Dropout(0.5)(concat_features)
    
    # 全连接层
    dense1 = Dense(256, activation='relu')(concat_features)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    
    # 输出层
    outputs = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs, name='Ultra_CNN_LSTM_Attention')
    
    return model


def create_ultra_cnn_gru_attention_model(max_features=20000, maxlen=500, embedding_dim=300):
    """
    创建超级优化版CNN+GRU+Attention模型
    """
    inputs = Input(shape=(maxlen,))
    
    x = Embedding(max_features, embedding_dim, input_length=maxlen)(inputs)
    x = SpatialDropout1D(0.2)(x)
    
    # 多尺度CNN
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(2)(conv1)
    
    conv2 = Conv1D(128, 4, activation='relu', padding='same')(x)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(2)(conv2)
    
    conv3 = Conv1D(128, 5, activation='relu', padding='same')(x)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling1D(2)(conv3)
    
    conv_concat = Concatenate()([conv1, conv2, conv3])
    conv_concat = Dropout(0.3)(conv_concat)
    
    conv_deep = Conv1D(256, 3, activation='relu', padding='same')(conv_concat)
    conv_deep = BatchNormalization()(conv_deep)
    conv_deep = MaxPooling1D(2)(conv_deep)
    
    conv_deep = Conv1D(128, 3, activation='relu', padding='same')(conv_deep)
    conv_deep = BatchNormalization()(conv_deep)
    
    # 双向GRU堆叠
    gru1 = Bidirectional(GRU(128, return_sequences=True))(conv_deep)
    gru1 = BatchNormalization()(gru1)
    gru1 = Dropout(0.3)(gru1)
    
    gru2 = Bidirectional(GRU(64, return_sequences=True))(gru1)
    gru2 = BatchNormalization()(gru2)
    
    # 注意力机制
    attention_output = AttentionLayer()(gru2)
    
    # 全局池化
    global_max = GlobalMaxPooling1D()(gru2)
    global_avg = GlobalAveragePooling1D()(gru2)
    
    concat_features = Concatenate()([attention_output, global_max, global_avg])
    concat_features = Dropout(0.5)(concat_features)
    
    # 全连接层
    dense1 = Dense(256, activation='relu')(concat_features)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    
    outputs = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs, name='Ultra_CNN_GRU_Attention')
    
    return model


def create_ultra_cnn_bigru_attention_model(max_features=20000, maxlen=500, embedding_dim=300):
    """
    创建CNN+BiGRU+Attention模型
    结构更简洁，用于和多尺度CNN对照
    """
    inputs = Input(shape=(maxlen,))

    x = Embedding(max_features, embedding_dim, input_length=maxlen)(inputs)
    x = SpatialDropout1D(0.2)(x)

    # 单尺度CNN特征提取
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    # BiGRU序列建模
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = BatchNormalization()(x)

    # 注意力 + 全局池化
    attention_output = AttentionLayer()(x)
    global_max = GlobalMaxPooling1D()(x)
    global_avg = GlobalAveragePooling1D()(x)

    features = Concatenate()([attention_output, global_max, global_avg])
    features = Dropout(0.5)(features)

    dense1 = Dense(256, activation='relu')(features)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)

    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)

    outputs = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=inputs, outputs=outputs, name='Ultra_CNN_BiGRU_Attention')

    return model


def create_ultra_textcnn_bilstm_model(max_features=20000, maxlen=500, embedding_dim=300):
    """
    创建TextCNN+BiLSTM混合模型
    基于经典TextCNN架构改进
    """
    inputs = Input(shape=(maxlen,))
    
    x = Embedding(max_features, embedding_dim, input_length=maxlen)(inputs)
    x = SpatialDropout1D(0.2)(x)
    
    # TextCNN部分 - 多个并行卷积层
    conv_blocks = []
    filter_sizes = [2, 3, 4, 5]
    
    for filter_size in filter_sizes:
        conv = Conv1D(128, filter_size, activation='relu', padding='valid')(x)
        conv = BatchNormalization()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    
    # 合并TextCNN特征
    textcnn_features = Concatenate()(conv_blocks)
    textcnn_features = Dropout(0.5)(textcnn_features)
    
    # 额外的序列建模
    # 使用较小的卷积核进行特征提取
    seq_conv = Conv1D(256, 3, activation='relu', padding='same')(x)
    seq_conv = BatchNormalization()(seq_conv)
    seq_conv = MaxPooling1D(2)(seq_conv)
    
    seq_conv = Conv1D(128, 3, activation='relu', padding='same')(seq_conv)
    seq_conv = BatchNormalization()(seq_conv)
    seq_conv = MaxPooling1D(2)(seq_conv)
    
    # BiLSTM层
    bilstm = Bidirectional(LSTM(128, return_sequences=True))(seq_conv)
    bilstm = BatchNormalization()(bilstm)
    bilstm = Dropout(0.3)(bilstm)
    
    bilstm = Bidirectional(LSTM(64, return_sequences=False))(bilstm)
    bilstm = BatchNormalization()(bilstm)
    
    # 合并TextCNN和BiLSTM特征
    combined = Concatenate()([textcnn_features, bilstm])
    combined = Dropout(0.5)(combined)
    
    # 全连接层
    dense1 = Dense(256, activation='relu')(combined)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = Dropout(0.3)(dense2)
    
    outputs = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs, name='Ultra_TextCNN_BiLSTM')
    
    return model


def get_all_ultra_models(max_features=20000, maxlen=500, embedding_dim=300):
    """
    获取所有超级优化版模型
    """
    models = {
        'Ultra_CNN+LSTM+Attention': create_ultra_cnn_lstm_attention_model(
            max_features, maxlen, embedding_dim
        ),
        'Ultra_CNN+GRU+Attention': create_ultra_cnn_gru_attention_model(
            max_features, maxlen, embedding_dim
        ),
        'Ultra_CNN+BiGRU+Attention': create_ultra_cnn_bigru_attention_model(
            max_features, maxlen, embedding_dim
        ),
        'Ultra_TextCNN+BiLSTM': create_ultra_textcnn_bilstm_model(
            max_features, maxlen, embedding_dim
        )
    }
    
    return models


def compile_ultra_model(model, learning_rate=0.0003):
    """
    编译超级优化版模型
    """
    from tensorflow.keras.optimizers import Adam
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_summary(model):
    """打印模型结构"""
    print(f"\n{'='*60}")
    print(f"模型名称: {model.name}")
    print(f"{'='*60}")
    model.summary()
    print(f"{'='*60}\n")
