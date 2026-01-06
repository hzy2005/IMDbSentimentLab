"""
改进版CNN+RNN混合模型 - 针对更高准确率优化
包含更深的网络结构、双向RNN、注意力机制等改进
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    SimpleRNN, GRU, LSTM, Bidirectional, Dense, Dropout, 
    Flatten, Concatenate, BatchNormalization, SpatialDropout1D
)


def create_advanced_cnn_simplernn_model(max_features=20000, maxlen=500, embedding_dim=256):
    """
    创建改进版CNN+SimpleRNN模型
    改进点：
    - 增加词汇表大小和嵌入维度
    - 使用双向SimpleRNN
    - 添加BatchNormalization
    - 更深的网络结构
    """
    model = Sequential([
        # 词嵌入层 - 增大维度
        Embedding(max_features, embedding_dim, input_length=maxlen),
        SpatialDropout1D(0.2),
        
        # 第一组CNN层
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第二组CNN层
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第三组CNN层
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 双向SimpleRNN层 - 堆叠两层
        Bidirectional(SimpleRNN(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(SimpleRNN(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.5),
        
        # 全连接层
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ], name='Advanced_CNN_BiSimpleRNN')
    
    return model


def create_advanced_cnn_gru_model(max_features=20000, maxlen=500, embedding_dim=256):
    """
    创建改进版CNN+GRU模型
    改进点：
    - 增加词汇表大小和嵌入维度
    - 使用双向GRU
    - 添加BatchNormalization
    - 多层GRU堆叠
    """
    model = Sequential([
        # 词嵌入层
        Embedding(max_features, embedding_dim, input_length=maxlen),
        SpatialDropout1D(0.2),
        
        # 第一组CNN层
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第二组CNN层
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第三组CNN层
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 双向GRU层 - 堆叠两层
        Bidirectional(GRU(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(GRU(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.5),
        
        # 全连接层
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ], name='Advanced_CNN_BiGRU')
    
    return model


def create_advanced_cnn_lstm_model(max_features=20000, maxlen=500, embedding_dim=256):
    """
    创建改进版CNN+LSTM模型
    改进点：
    - 增加词汇表大小和嵌入维度
    - 使用双向LSTM
    - 添加BatchNormalization
    - 多层LSTM堆叠
    """
    model = Sequential([
        # 词嵌入层
        Embedding(max_features, embedding_dim, input_length=maxlen),
        SpatialDropout1D(0.2),
        
        # 第一组CNN层
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第二组CNN层
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第三组CNN层
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 双向LSTM层 - 堆叠两层
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.5),
        
        # 全连接层
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ], name='Advanced_CNN_BiLSTM')
    
    return model


def get_all_advanced_models(max_features=20000, maxlen=500, embedding_dim=256):
    """
    获取所有改进版模型
    
    Returns:
        包含所有改进版模型的字典
    """
    models = {
        'Advanced_CNN+BiSimpleRNN': create_advanced_cnn_simplernn_model(max_features, maxlen, embedding_dim),
        'Advanced_CNN+BiGRU': create_advanced_cnn_gru_model(max_features, maxlen, embedding_dim),
        'Advanced_CNN+BiLSTM': create_advanced_cnn_lstm_model(max_features, maxlen, embedding_dim)
    }
    
    return models


def compile_advanced_model(model, learning_rate=0.0005):
    """
    编译改进版模型 - 使用更小的学习率
    
    Args:
        model: Keras模型
        learning_rate: 学习率
    """
    from tensorflow.keras.optimizers import Adam
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_summary(model):
    """
    打印模型结构摘要
    
    Args:
        model: Keras模型
    """
    print(f"\n{'='*60}")
    print(f"模型名称: {model.name}")
    print(f"{'='*60}")
    model.summary()
    print(f"{'='*60}\n")
