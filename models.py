"""
CNN+RNN混合模型定义
包含三种不同的RNN变体：SimpleRNN, GRU, LSTM
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    SimpleRNN, GRU, LSTM, Dense, Dropout, Flatten
)


def create_cnn_simplernn_model(max_features=10000, maxlen=500, embedding_dim=128):
    """
    创建CNN+SimpleRNN模型
    
    Args:
        max_features: 词汇表大小
        maxlen: 序列最大长度
        embedding_dim: 词嵌入维度
    
    Returns:
        Keras模型
    """
    model = Sequential([
        # 词嵌入层
        Embedding(max_features, embedding_dim, input_length=maxlen),
        
        # CNN层 - 提取局部特征
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        # SimpleRNN层 - 捕获序列依赖
        SimpleRNN(64, return_sequences=False),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ], name='CNN_SimpleRNN')
    
    return model


def create_cnn_gru_model(max_features=10000, maxlen=500, embedding_dim=128):
    """
    创建CNN+GRU模型
    
    Args:
        max_features: 词汇表大小
        maxlen: 序列最大长度
        embedding_dim: 词嵌入维度
    
    Returns:
        Keras模型
    """
    model = Sequential([
        # 词嵌入层
        Embedding(max_features, embedding_dim, input_length=maxlen),
        
        # CNN层 - 提取局部特征
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        # GRU层 - 捕获序列依赖
        GRU(64, return_sequences=False),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ], name='CNN_GRU')
    
    return model


def create_cnn_lstm_model(max_features=10000, maxlen=500, embedding_dim=128):
    """
    创建CNN+LSTM模型
    
    Args:
        max_features: 词汇表大小
        maxlen: 序列最大长度
        embedding_dim: 词嵌入维度
    
    Returns:
        Keras模型
    """
    model = Sequential([
        # 词嵌入层
        Embedding(max_features, embedding_dim, input_length=maxlen),
        
        # CNN层 - 提取局部特征
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        # LSTM层 - 捕获序列依赖
        LSTM(64, return_sequences=False),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ], name='CNN_LSTM')
    
    return model


def get_all_models(max_features=10000, maxlen=500, embedding_dim=128):
    """
    获取所有模型
    
    Returns:
        包含所有模型的字典
    """
    models = {
        'CNN+SimpleRNN': create_cnn_simplernn_model(max_features, maxlen, embedding_dim),
        'CNN+GRU': create_cnn_gru_model(max_features, maxlen, embedding_dim),
        'CNN+LSTM': create_cnn_lstm_model(max_features, maxlen, embedding_dim)
    }
    
    return models


def compile_model(model, learning_rate=0.001):
    """
    编译模型
    
    Args:
        model: Keras模型
        learning_rate: 学习率
    """
    model.compile(
        optimizer='adam',
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