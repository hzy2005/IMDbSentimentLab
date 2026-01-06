"""
IMDb电影评论数据加载和预处理模块
"""
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical


class IMDbDataLoader:
    """IMDb数据加载器"""
    
    def __init__(self, max_features=10000, maxlen=300):
        """
        初始化数据加载器
        
        Args:
            max_features: 词汇表大小
            maxlen: 序列最大长度
        """
        self.max_features = max_features
        self.maxlen = maxlen
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_data(self):
        """加载IMDb数据集"""
        print(f"正在加载IMDb数据集...")
        print(f"词汇表大小: {self.max_features}")
        print(f"序列最大长度: {self.maxlen}")
        
        # 加载数据
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=self.max_features
        )
        
        print(f"训练集大小: {len(x_train)}")
        print(f"测试集大小: {len(x_test)}")
        
        # 填充序列到相同长度
        
        """
        pad_sequences 默认是padding='pre',truncating='pre'  
        序列长度不足的：前面补大量 0 序列过长的：从前面截断 
        而 IMDb 评论的情感信息常常在开头（甚至标题、前几句）就出现，你这种截断会把开头截掉。

        """
    
        # self.x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        # self.x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        self.x_train = sequence.pad_sequences(
            x_train, maxlen=self.maxlen, padding='post', truncating='post'
        )
        self.x_test = sequence.pad_sequences(
            x_test, maxlen=self.maxlen, padding='post', truncating='post'
        )
        
        # 标签保持为0和1（二分类）
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"训练数据形状: {self.x_train.shape}")
        print(f"测试数据形状: {self.x_test.shape}")
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def get_data(self):
        """获取处理后的数据"""
        if self.x_train is None:
            self.load_data()
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def get_sample_text(self, index=0, dataset='train'):
        """
        获取样本文本（用于展示）
        
        Args:
            index: 样本索引
            dataset: 'train' 或 'test'
        """
        # 获取词汇索引映射
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        # 选择数据集
        if dataset == 'train':
            sequence_data = self.x_train[index] if self.x_train is not None else None
            label = self.y_train[index] if self.y_train is not None else None
        else:
            sequence_data = self.x_test[index] if self.x_test is not None else None
            label = self.y_test[index] if self.y_test is not None else None
        
        if sequence_data is None:
            return None, None
        
        # 解码序列
        # 这大体是对的，但你会把 1(start)、2(oov)也当成词处理，可能出现乱码
        # decoded = ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence_data if i > 0])
        index_from = 3
        decoded = ' '.join([reverse_word_index.get(i - index_from, '?') 
                            for i in sequence_data if i >= index_from])

        return decoded, label