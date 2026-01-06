"""
更新说明:
- CNN 部分使用 BatchNormalization 提升特征稳定性
- 降低正则强度，RNN 禁用 kernel 正则
- Attention 使用 tanh 以增强早期权重差异
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    GlobalAveragePooling1D, LSTM, GRU, Bidirectional, Dense,
    Dropout, Concatenate, SpatialDropout1D, Add, Layer,
    LayerNormalization, Activation, BatchNormalization
)
from tensorflow.keras.regularizers import l2


class AttentionLayer(Layer):
    """时间注意力层（temporal attention）"""

    def __init__(self, return_attention=False, use_tanh=True, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention
        self.use_tanh = use_tanh
        self._last_attention_weights = None

    def build(self, input_shape):
        self.score_dense = Dense(1, use_bias=True)
        super().build(input_shape)

    def call(self, x):
        scores = self.score_dense(x)
        if self.use_tanh:
            scores = tf.tanh(scores)
        weights = tf.nn.softmax(scores, axis=1)
        self._last_attention_weights = weights
        context = tf.reduce_sum(x * weights, axis=1)
        if self.return_attention:
            return context, weights
        return context

    def get_last_attention_weights(self):
        return self._last_attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "return_attention": self.return_attention,
            "use_tanh": self.use_tanh
        })
        return config


def _conv_norm_relu(x, filters, kernel_size, l2_cnn, use_batchnorm=True):
    x = Conv1D(filters, kernel_size, activation=None, padding='same',
               kernel_regularizer=l2(l2_cnn))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    else:
        x = LayerNormalization()(x)
    x = Activation('relu')(x)
    return x


def _residual_block(x, filters, kernel_size, l2_cnn, use_batchnorm=True):
    shortcut = x
    x = _conv_norm_relu(x, filters, kernel_size, l2_cnn, use_batchnorm)
    x = Conv1D(filters, kernel_size, activation=None, padding='same',
               kernel_regularizer=l2(l2_cnn))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    else:
        x = LayerNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x


def create_ultra_cnn_lstm_attention_model(max_features=20000, maxlen=500, embedding_dim=300,
                                          return_attention=False):
    """
    创建超强版CNN+LSTM+Attention模型
    """
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))

    x = Embedding(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv1 = _conv_norm_relu(x, 128, 3, l2_cnn, use_batchnorm)
    conv1 = MaxPooling1D(2)(conv1)

    conv2 = _conv_norm_relu(x, 128, 4, l2_cnn, use_batchnorm)
    conv2 = MaxPooling1D(2)(conv2)

    conv3 = _conv_norm_relu(x, 128, 5, l2_cnn, use_batchnorm)
    conv3 = MaxPooling1D(2)(conv3)

    conv_concat = Concatenate()([conv1, conv2, conv3])
    conv_concat = Dropout(0.25)(conv_concat)

    conv_deep = _conv_norm_relu(conv_concat, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = MaxPooling1D(2)(conv_deep)

    conv_proj = _conv_norm_relu(conv_deep, 128, 3, l2_cnn, use_batchnorm)

    lstm1 = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.25,
             recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(conv_proj)
    lstm1 = LayerNormalization()(lstm1)
    lstm1 = Dropout(0.25)(lstm1)

    lstm2 = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.25,
             recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(lstm1)
    lstm2 = LayerNormalization()(lstm2)

    att_layer = AttentionLayer(return_attention=return_attention, use_tanh=True)
    if return_attention:
        attention_output, att_weights = att_layer(lstm2)
    else:
        attention_output = att_layer(lstm2)
        att_weights = None

    global_max = GlobalMaxPooling1D()(lstm2)
    global_avg = GlobalAveragePooling1D()(lstm2)

    concat_features = Concatenate()([attention_output, global_max, global_avg])
    concat_features = Dropout(0.35)(concat_features)

    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(l2_cnn))(concat_features)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.35)(dense1)

    dense2 = Dense(128, activation='relu', kernel_regularizer=l2(l2_cnn))(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.25)(dense2)

    outputs = Dense(1, activation='sigmoid')(dense2)

    if return_attention:
        model = Model(inputs=inputs, outputs=[outputs, att_weights],
                      name='Ultra_MSResCNN_LSTM_Attention')
    else:
        model = Model(inputs=inputs, outputs=outputs,
                      name='Ultra_MSResCNN_LSTM_Attention')

    return model


def create_ultra_cnn_gru_attention_model(max_features=20000, maxlen=500, embedding_dim=300,
                                         return_attention=False):
    """
    创建超强版CNN+GRU+Attention模型
    """
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))

    x = Embedding(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv1 = _conv_norm_relu(x, 128, 3, l2_cnn, use_batchnorm)
    conv1 = MaxPooling1D(2)(conv1)

    conv2 = _conv_norm_relu(x, 128, 4, l2_cnn, use_batchnorm)
    conv2 = MaxPooling1D(2)(conv2)

    conv3 = _conv_norm_relu(x, 128, 5, l2_cnn, use_batchnorm)
    conv3 = MaxPooling1D(2)(conv3)

    conv_concat = Concatenate()([conv1, conv2, conv3])
    conv_concat = Dropout(0.25)(conv_concat)

    conv_deep = _conv_norm_relu(conv_concat, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = MaxPooling1D(2)(conv_deep)

    conv_proj = _conv_norm_relu(conv_deep, 128, 3, l2_cnn, use_batchnorm)

    gru1 = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.25,
            recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(conv_proj)
    gru1 = LayerNormalization()(gru1)
    gru1 = Dropout(0.25)(gru1)

    gru2 = Bidirectional(
        GRU(64, return_sequences=True, dropout=0.25,
            recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(gru1)
    gru2 = LayerNormalization()(gru2)

    att_layer = AttentionLayer(return_attention=return_attention, use_tanh=True)
    if return_attention:
        attention_output, att_weights = att_layer(gru2)
    else:
        attention_output = att_layer(gru2)
        att_weights = None

    global_max = GlobalMaxPooling1D()(gru2)
    global_avg = GlobalAveragePooling1D()(gru2)

    concat_features = Concatenate()([attention_output, global_max, global_avg])
    concat_features = Dropout(0.35)(concat_features)

    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(l2_cnn))(concat_features)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.35)(dense1)

    dense2 = Dense(128, activation='relu', kernel_regularizer=l2(l2_cnn))(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.25)(dense2)

    outputs = Dense(1, activation='sigmoid')(dense2)

    if return_attention:
        model = Model(inputs=inputs, outputs=[outputs, att_weights],
                      name='Ultra_MSResCNN_BiGRU_Attention')
    else:
        model = Model(inputs=inputs, outputs=outputs,
                      name='Ultra_MSResCNN_BiGRU_Attention')

    return model


def create_ultra_cnn_bigru_attention_model(max_features=20000, maxlen=500, embedding_dim=300,
                                           return_attention=False):
    """
    创建CNN+BiGRU+Attention模型
    结构更简洁，用于和多尺度CNN对照
    """
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))

    x = Embedding(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    x = _conv_norm_relu(x, 128, 3, l2_cnn, use_batchnorm)
    x = _residual_block(x, 128, 3, l2_cnn, use_batchnorm)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = _conv_norm_relu(x, 128, 3, l2_cnn, use_batchnorm)
    x = MaxPooling1D(2)(x)

    gru1 = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.25,
            recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(x)
    gru1 = LayerNormalization()(gru1)
    gru1 = Dropout(0.25)(gru1)

    gru2 = Bidirectional(
        GRU(64, return_sequences=True, dropout=0.25,
            recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(gru1)
    gru2 = LayerNormalization()(gru2)

    att_layer = AttentionLayer(return_attention=return_attention, use_tanh=True)
    if return_attention:
        attention_output, att_weights = att_layer(gru2)
    else:
        attention_output = att_layer(gru2)
        att_weights = None

    global_max = GlobalMaxPooling1D()(gru2)
    global_avg = GlobalAveragePooling1D()(gru2)

    features = Concatenate()([attention_output, global_max, global_avg])
    features = Dropout(0.35)(features)

    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(l2_cnn))(features)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.35)(dense1)

    dense2 = Dense(128, activation='relu', kernel_regularizer=l2(l2_cnn))(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.25)(dense2)

    outputs = Dense(1, activation='sigmoid')(dense2)

    if return_attention:
        model = Model(inputs=inputs, outputs=[outputs, att_weights],
                      name='Ultra_ResCNN_BiGRU_Attention')
    else:
        model = Model(inputs=inputs, outputs=outputs,
                      name='Ultra_ResCNN_BiGRU_Attention')

    return model


def create_ultra_textcnn_bilstm_model(max_features=20000, maxlen=500, embedding_dim=300):
    """
    创建TextCNN+BiLSTM混合模型（简化版）
    """
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))

    x = Embedding(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv_blocks = []
    filter_sizes = [2, 3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(64, filter_size, activation=None, padding='valid',
                      kernel_regularizer=l2(l2_cnn))(x)
        if use_batchnorm:
            conv = BatchNormalization()(conv)
        else:
            conv = LayerNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    textcnn_features = Concatenate()(conv_blocks)
    textcnn_features = Dropout(0.35)(textcnn_features)

    bilstm = Bidirectional(
        LSTM(96, return_sequences=False, dropout=0.25,
             recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(x)
    bilstm = LayerNormalization()(bilstm)

    combined = Concatenate()([textcnn_features, bilstm])
    combined = Dropout(0.35)(combined)

    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(l2_cnn))(combined)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)

    outputs = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=outputs, name='Ultra_TextCNN_BiLSTM_PoolCat')

    return model


def get_all_ultra_models(max_features=20000, maxlen=500, embedding_dim=300):
    """
    获取所有超级优化版模型
    """
    models = {
        'Ultra_MSResCNN+LSTM+Attention': create_ultra_cnn_lstm_attention_model(
            max_features, maxlen, embedding_dim
        ),
        'Ultra_MSResCNN+BiGRU+Attention': create_ultra_cnn_gru_attention_model(
            max_features, maxlen, embedding_dim
        ),
        'Ultra_ResCNN+BiGRU+Attention': create_ultra_cnn_bigru_attention_model(
            max_features, maxlen, embedding_dim
        ),
        'Ultra_TextCNN+BiLSTM_PoolCat': create_ultra_textcnn_bilstm_model(
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


def debug_model_predictions(model, x, n=256):
    """
    打印模型在给定输入上的预测分布：mean/std/min/max
    用于快速判断是否输出恒定 0.5
    """
    preds = model.predict(x[:n], verbose=0)
    preds = preds.reshape(-1)
    print(
        f"pred stats -> mean: {preds.mean():.4f}, "
        f"std: {preds.std():.4f}, min: {preds.min():.4f}, max: {preds.max():.4f}"
    )
