"""
增强说明:
- Masked Attention 支持 padding mask 并对齐 CNN 池化后的时间步
- CNN 改为后置 pooling，减少序列信息丢失；可选 GloVe 词向量
- 稳定性：BatchNorm(CNN) + LayerNorm(RNN/Dense) + 适中 Dropout
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    GlobalAveragePooling1D, LSTM, GRU, Bidirectional, Dense,
    Dropout, Concatenate, SpatialDropout1D, Add, Layer,
    LayerNormalization, Activation, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import imdb


USE_GLOVE = os.environ.get("USE_GLOVE", "0") == "1"
GLOVE_PATH = os.environ.get("GLOVE_PATH", "")
GLOVE_INDEX_FROM = 3


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, end_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * (step / tf.maximum(1.0, warmup_steps))
        progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        pi = tf.constant(np.pi, dtype=progress.dtype)
        cosine_lr = self.end_lr + 0.5 * (self.base_lr - self.end_lr) * (1.0 + tf.cos(pi * progress))

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "end_lr": self.end_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }


def load_glove_embeddings(glove_path, word_index, embedding_dim, max_features, index_from=3):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            if coefs.shape[0] != embedding_dim:
                continue
            embeddings_index[word] = coefs

    embedding_matrix = np.random.normal(scale=0.05, size=(max_features, embedding_dim)).astype("float32")
    embedding_matrix[0] = 0.0

    for word, i in word_index.items():
        idx = i + index_from
        if idx >= max_features:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec

    return embedding_matrix


def _get_embedding_layer(max_features, embedding_dim, trainable=True):
    if USE_GLOVE and GLOVE_PATH and os.path.exists(GLOVE_PATH):
        try:
            word_index = imdb.get_word_index()
            emb_matrix = load_glove_embeddings(
                GLOVE_PATH, word_index, embedding_dim, max_features, index_from=GLOVE_INDEX_FROM
            )
            return Embedding(
                max_features,
                embedding_dim,
                weights=[emb_matrix],
                trainable=False,
                mask_zero=True
            )
        except Exception:
            pass

    return Embedding(
        max_features,
        embedding_dim,
        mask_zero=True,
        embeddings_initializer="glorot_uniform",
        trainable=trainable
    )


class AttentionLayer(Layer):
    """时间注意力层（temporal attention），支持 mask"""

    def __init__(self, return_attention=False, use_tanh=True, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention
        self.use_tanh = use_tanh
        self._last_attention_weights = None
        self.supports_masking = True

    def build(self, input_shape):
        self.score_dense = Dense(1, use_bias=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        scores = self.score_dense(x)
        if self.use_tanh:
            scores = tf.tanh(scores)

        if mask is not None:
            if mask.shape.rank == 2:
                mask = tf.expand_dims(mask, axis=-1)
            mask = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask) * (-1e9)

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
    x = Conv1D(filters, kernel_size, activation=None, padding="same",
               kernel_regularizer=l2(l2_cnn))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    else:
        x = LayerNormalization()(x)
    x = Activation("relu")(x)
    return x


def _residual_block(x, filters, kernel_size, l2_cnn, use_batchnorm=True):
    shortcut = x
    x = _conv_norm_relu(x, filters, kernel_size, l2_cnn, use_batchnorm)
    x = Conv1D(filters, kernel_size, activation=None, padding="same",
               kernel_regularizer=l2(l2_cnn))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    else:
        x = LayerNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x


def _build_mask_from_inputs(inputs, pool_count=0, pool_size=2):
    mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    for _ in range(pool_count):
        mask = MaxPooling1D(pool_size=pool_size, strides=pool_size, padding="valid")(mask)
    mask = tf.squeeze(mask, axis=-1)
    return mask


def create_ultra_cnn_lstm_attention_model(max_features=20000, maxlen=500, embedding_dim=300,
                                          return_attention=False):
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))
    mask_pool_count = 2

    x = _get_embedding_layer(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv1 = _conv_norm_relu(x, 128, 3, l2_cnn, use_batchnorm)
    conv2 = _conv_norm_relu(x, 128, 4, l2_cnn, use_batchnorm)
    conv3 = _conv_norm_relu(x, 128, 5, l2_cnn, use_batchnorm)

    conv_concat = Concatenate()([conv1, conv2, conv3])
    conv_concat = Dropout(0.25)(conv_concat)

    conv_deep = _conv_norm_relu(conv_concat, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = MaxPooling1D(2)(conv_deep)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = MaxPooling1D(2)(conv_deep)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)

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

    mask = _build_mask_from_inputs(inputs, pool_count=mask_pool_count, pool_size=2)

    att_layer = AttentionLayer(return_attention=return_attention, use_tanh=True)
    if return_attention:
        attention_output, att_weights = att_layer(lstm2, mask=mask)
    else:
        attention_output = att_layer(lstm2, mask=mask)
        att_weights = None

    global_max = GlobalMaxPooling1D()(lstm2)
    global_avg = GlobalAveragePooling1D()(lstm2)

    concat_features = Concatenate()([attention_output, global_max, global_avg])
    concat_features = Dropout(0.25)(concat_features)

    dense1 = Dense(256, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(concat_features)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)

    dense2 = Dense(128, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.20)(dense2)

    outputs = Dense(1, activation="sigmoid")(dense2)

    if return_attention:
        model = Model(inputs=inputs, outputs=[outputs, att_weights],
                      name="Ultra_MSResCNN_LSTM_Attention")
    else:
        model = Model(inputs=inputs, outputs=outputs,
                      name="Ultra_MSResCNN_LSTM_Attention")

    return model


def create_ultra_cnn_gru_attention_model(max_features=20000, maxlen=500, embedding_dim=300,
                                         return_attention=False):
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))
    mask_pool_count = 2

    x = _get_embedding_layer(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv1 = _conv_norm_relu(x, 128, 3, l2_cnn, use_batchnorm)
    conv2 = _conv_norm_relu(x, 128, 4, l2_cnn, use_batchnorm)
    conv3 = _conv_norm_relu(x, 128, 5, l2_cnn, use_batchnorm)

    conv_concat = Concatenate()([conv1, conv2, conv3])
    conv_concat = Dropout(0.25)(conv_concat)

    conv_deep = _conv_norm_relu(conv_concat, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = MaxPooling1D(2)(conv_deep)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)
    conv_deep = MaxPooling1D(2)(conv_deep)
    conv_deep = _residual_block(conv_deep, 256, 3, l2_cnn, use_batchnorm)

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

    mask = _build_mask_from_inputs(inputs, pool_count=mask_pool_count, pool_size=2)

    att_layer = AttentionLayer(return_attention=return_attention, use_tanh=True)
    if return_attention:
        attention_output, att_weights = att_layer(gru2, mask=mask)
    else:
        attention_output = att_layer(gru2, mask=mask)
        att_weights = None

    global_max = GlobalMaxPooling1D()(gru2)
    global_avg = GlobalAveragePooling1D()(gru2)

    concat_features = Concatenate()([attention_output, global_max, global_avg])
    concat_features = Dropout(0.25)(concat_features)

    dense1 = Dense(256, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(concat_features)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)

    dense2 = Dense(128, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.20)(dense2)

    outputs = Dense(1, activation="sigmoid")(dense2)

    if return_attention:
        model = Model(inputs=inputs, outputs=[outputs, att_weights],
                      name="Ultra_MSResCNN_BiGRU_Attention")
    else:
        model = Model(inputs=inputs, outputs=outputs,
                      name="Ultra_MSResCNN_BiGRU_Attention")

    return model


def create_ultra_cnn_bigru_attention_model(max_features=20000, maxlen=500, embedding_dim=300,
                                           return_attention=False):
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))
    mask_pool_count = 2

    x = _get_embedding_layer(max_features, embedding_dim)(inputs)
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

    mask = _build_mask_from_inputs(inputs, pool_count=mask_pool_count, pool_size=2)

    att_layer = AttentionLayer(return_attention=return_attention, use_tanh=True)
    if return_attention:
        attention_output, att_weights = att_layer(gru2, mask=mask)
    else:
        attention_output = att_layer(gru2, mask=mask)
        att_weights = None

    global_max = GlobalMaxPooling1D()(gru2)
    global_avg = GlobalAveragePooling1D()(gru2)

    features = Concatenate()([attention_output, global_max, global_avg])
    features = Dropout(0.25)(features)

    dense1 = Dense(256, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(features)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)

    dense2 = Dense(128, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(0.20)(dense2)

    outputs = Dense(1, activation="sigmoid")(dense2)

    if return_attention:
        model = Model(inputs=inputs, outputs=[outputs, att_weights],
                      name="Ultra_ResCNN_BiGRU_Attention")
    else:
        model = Model(inputs=inputs, outputs=outputs,
                      name="Ultra_ResCNN_BiGRU_Attention")

    return model


def create_ultra_textcnn_bilstm_model(max_features=20000, maxlen=500, embedding_dim=300):
    l2_cnn = 1e-5
    l2_rnn = 0.0
    use_batchnorm = True

    inputs = Input(shape=(maxlen,))
    mask = _build_mask_from_inputs(inputs, pool_count=0, pool_size=2)

    x = _get_embedding_layer(max_features, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv_blocks = []
    filter_sizes = [2, 3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(64, filter_size, activation=None, padding="valid",
                      kernel_regularizer=l2(l2_cnn))(x)
        if use_batchnorm:
            conv = BatchNormalization()(conv)
        else:
            conv = LayerNormalization()(conv)
        conv = Activation("relu")(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    textcnn_features = Concatenate()(conv_blocks)
    textcnn_features = Dropout(0.25)(textcnn_features)

    bilstm = Bidirectional(
        LSTM(96, return_sequences=True, dropout=0.25,
             recurrent_dropout=0.0, kernel_regularizer=l2(l2_rnn))
    )(x)
    bilstm = LayerNormalization()(bilstm)

    att_layer = AttentionLayer(return_attention=False, use_tanh=True)
    att_context = att_layer(bilstm, mask=mask)

    global_max = GlobalMaxPooling1D()(bilstm)
    global_avg = GlobalAveragePooling1D()(bilstm)

    combined = Concatenate()([textcnn_features, att_context, global_max, global_avg])
    combined = Dropout(0.25)(combined)

    dense1 = Dense(128, activation="relu", kernel_regularizer=l2(l2_cnn),
                   kernel_initializer="he_normal")(combined)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(0.20)(dense1)

    outputs = Dense(1, activation="sigmoid")(dense1)

    model = Model(inputs=inputs, outputs=outputs, name="Ultra_TextCNN_BiLSTM_PoolCat")

    return model


def get_all_ultra_models(max_features=20000, maxlen=500, embedding_dim=300):
    models = {
        "Ultra_MSResCNN+LSTM+Attention": create_ultra_cnn_lstm_attention_model(
            max_features, maxlen, embedding_dim
        ),
        "Ultra_MSResCNN+BiGRU+Attention": create_ultra_cnn_gru_attention_model(
            max_features, maxlen, embedding_dim
        ),
        "Ultra_ResCNN+BiGRU+Attention": create_ultra_cnn_bigru_attention_model(
            max_features, maxlen, embedding_dim
        ),
        "Ultra_TextCNN+BiLSTM_PoolCat": create_ultra_textcnn_bilstm_model(
            max_features, maxlen, embedding_dim
        )
    }

    return models


def compile_ultra_model(model, learning_rate=0.0003, steps_per_epoch=None, epochs=None):
    from tensorflow.keras.optimizers import Adam

    base_lr = float(learning_rate)
    if steps_per_epoch is not None and epochs is not None:
        total_steps = int(steps_per_epoch * epochs)
    else:
        total_steps = 50000
    warmup_steps = int(0.1 * total_steps)
    warmup_steps = max(200, min(2000, warmup_steps))
    end_lr = base_lr * 0.05
    if os.environ.get("DEBUG_LR") == "1":
        print(f"[LR] total_steps={total_steps} warmup_steps={warmup_steps} "
              f"base_lr={base_lr} end_lr={end_lr}")
    lr_schedule = WarmupCosineSchedule(
        base_lr=base_lr,
        end_lr=end_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    optimizer = None
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=5e-5, clipnorm=1.0
        )
    except Exception:
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


def print_model_summary(model):
    print(f"\n{'='*60}")
    print(f"模型名称: {model.name}")
    print(f"{'='*60}")
    model.summary()
    print(f"{'='*60}\n")


def print_trainability_summary(model):
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    total = trainable + non_trainable
    print(f"Trainable params: {trainable}")
    print(f"Non-trainable params: {non_trainable}")
    print(f"Total params: {total}")


def debug_model_predictions(model, x, n=256):
    """
    训练前/后可调用，用于检查预测是否仍接近 0.5。
    例如：debug_model_predictions(model, x_test)
    """
    preds = model.predict(x[:n], verbose=0)
    preds = preds.reshape(-1)
    print(
        f"pred stats -> mean: {preds.mean():.4f}, "
        f"std: {preds.std():.4f}, min: {preds.min():.4f}, max: {preds.max():.4f}"
    )
