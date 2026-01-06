# IMDb电影评论情感分析 - CNN+RNN混合模型

## 项目简介

本项目使用深度学习方法对IMDb电影评论进行情感分析（二分类：正面/负面）。项目实现了多种CNN+RNN混合模型，并对它们的性能进行了全面对比分析。

**包含三个实验组：**
- **经典组**：标准CNN+RNN模型（准确率~88%）
- **改进组**：优化CNN+RNN模型（双向RNN + BatchNorm）
- **超强组**：更先进架构（注意力机制 + 多尺度CNN + TextCNN，目标95%+）

## 模型架构

经典组包含以下三种模型：

### 1. CNN + SimpleRNN
- **Embedding层**: 将词索引转换为密集向量表示
- **CNN层**: 使用卷积操作提取局部文本特征
- **MaxPooling层**: 降维并保留重要特征
- **SimpleRNN层**: 捕获序列的时序依赖关系
- **全连接层**: 进行分类决策

### 2. CNN + GRU
- **Embedding层**: 词嵌入表示
- **CNN层**: 提取n-gram特征
- **MaxPooling层**: 特征降维
- **GRU层**: 使用门控机制捕获长期依赖
- **全连接层**: 输出分类结果

### 3. CNN + LSTM
- **Embedding层**: 词向量表示
- **CNN层**: 局部特征提取
- **MaxPooling层**: 池化操作
- **LSTM层**: 通过记忆单元处理长序列依赖
- **全连接层**: 最终分类

改进组与超强组模型定义见 `models_advanced.py` 与 `models_ultra.py`。

## 模型清单（按实验组）

**经典组（models.py）**
- CNN + SimpleRNN
- CNN + GRU
- CNN + LSTM

**改进组（models_advanced.py）**
- Advanced CNN + SimpleRNN（双向）
- Advanced CNN + GRU（双向，双层GRU）
- Advanced CNN + LSTM（双向，双层LSTM）

**超强组（models_ultra.py）**
- Ultra CNN + LSTM + Attention（多尺度CNN + 注意力）
- Ultra CNN + GRU + Attention（多尺度CNN + 注意力）
- Ultra CNN + BiGRU + Attention（单尺度CNN + 注意力）
- Ultra TextCNN + BiLSTM

## 项目结构

```
.
├── data_loader.py      # 数据加载和预处理模块
├── models.py           # 经典CNN+RNN模型定义
├── models_advanced.py  # 改进CNN+RNN模型定义
├── models_ultra.py     # 超强CNN+RNN模型定义
├── trainer.py          # 训练和评估模块
├── visualizer.py       # 可视化模块
├── main.py             # 统一主程序入口
├── requirements.txt    # 依赖包列表
├── README.md           # 项目说明文档
├── results/            # 经典组输出目录（运行后自动创建）
├── results_advanced/   # 改进组输出目录（运行后自动创建）
└── results_ultra/      # 超强组输出目录（运行后自动创建）
```

## 环境要求

- Python 3.7+
- TensorFlow 2.10+
- NumPy 1.21+
- Matplotlib 3.5+
- scikit-learn 1.0+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方式1: 运行统一入口（推荐）

```bash
python main.py
```

**统一入口说明：**
- `main.py` 内部包含经典/改进/超强三组模型
- 可在 `main.py` 中修改 `run_groups` 控制要运行的实验组
- 默认输出目录：`results/`、`results_advanced/`、`results_ultra/`

程序将自动完成以下步骤：
1. 加载并预处理IMDb数据集
2. 创建三组不同复杂度的CNN+RNN模型
3. 训练所有模型
4. 评估模型性能
5. 生成可视化结果
6. 保存所有结果到对应的输出目录

### 自定义参数

可以在 `main.py` 的 `configs` 中修改以下超参数（不同组分别配置）：

```python
max_features = 10000
maxlen = 500
embedding_dim = 128
batch_size = 128
epochs = 20
learning_rate = 0.001
```

## 输出结果

运行完成后，各输出目录将包含：

### 1. 模型结构文件
- `*_structure.txt`

### 2. 训练历史可视化
- `*_training_history.png`

每张图包含两个子图：
- 训练/验证准确率变化曲线
- 训练/验证损失变化曲线

### 3. 模型对比图
- `all_models_comparison.png`

包含四个子图：
- 训练准确率对比
- 验证准确率对比
- 训练损失对比
- 验证损失对比

### 4. 结果对比柱状图
- `results_comparison.png`

包含四个子图：
- 测试准确率对比
- 测试损失对比
- 训练时间对比
- 训练轮数对比

### 5. 详细结果文本
- `detailed_results.txt`

### 6. 实验要求对照说明
- `experiment_summary.txt`

## 模型对比分析

经典组三种模型的主要区别：

| 特性 | SimpleRNN | GRU | LSTM |
|------|-----------|-----|------|
| **参数量** | 最少 | 中等 | 最多 |
| **训练速度** | 最快 | 中等 | 最慢 |
| **长期依赖** | 较弱 | 强 | 最强 |
| **梯度问题** | 容易梯度消失 | 有门控机制缓解 | 有记忆单元缓解 |
| **适用场景** | 短序列 | 中长序列 | 长序列 |

## 实验特点

1. **数据预处理**
   - 使用Keras内置的IMDb数据集
   - 序列填充到统一长度
   - 训练集/验证集/测试集划分

2. **模型训练**
   - 使用Adam优化器
   - 二元交叉熵损失函数
   - 早停机制防止过拟合
   - 学习率自适应调整

3. **评估指标**
   - 准确率（Accuracy）
   - 损失值（Loss）
   - 训练时间
   - 训练轮数

4. **可视化**
   - 训练过程曲线
   - 多模型对比图
   - 结果柱状图
   - 支持中文显示

## 技术要点

### CNN部分
- 使用1D卷积提取局部n-gram特征
- 多层卷积逐步提取更高层次的特征
- MaxPooling降维并保留关键信息

### RNN部分
- SimpleRNN: 基础循环神经网络
- GRU: 使用更新门和重置门
- LSTM: 使用输入门、遗忘门和输出门

### 训练策略
- EarlyStopping: 验证损失3轮不降则停止
- ReduceLROnPlateau: 验证损失2轮不降则降低学习率
- Dropout: 防止过拟合

## 预期结果

根据模型特性，预期结果：
- **准确率**: LSTM ≈ GRU > SimpleRNN
- **训练速度**: SimpleRNN > GRU > LSTM
- **稳定性**: LSTM ≥ GRU > SimpleRNN

## 实验组对比

| 特性 | 经典组 | 改进组 | 超强组 |
|------|--------|--------|--------|
| **词汇表大小** | 10,000 | 20,000 | 20,000 |
| **嵌入维度** | 128 | 256 | 300 |
| **RNN类型** | 单向 | 双向 | 双向 |
| **网络深度** | 2层CNN + 1层RNN | 3层CNN + 2层RNN | 多尺度CNN + 2层RNN |
| **特殊架构** | 无 | 无 | 注意力 + TextCNN |
| **BatchNorm** | 无 | 有 | 有 |
| **批次大小** | 128 | 64 | 32 |
| **训练轮数** | 20 | 30 | 50 |
| **学习率** | 0.001 | 0.0005 | 0.0003 |
| **预期准确率** | ~88% | ~88-90% | 95%+ |
| **训练时间** | 短 | 中等 | 长 |
| **模型参数** | ~900K | ~8M | ~15M |
| **适用场景** | 快速验证 | 平衡性能 | 追求极致准确率 |

## 注意事项

1. 首次运行会自动下载IMDb数据集（约80MB）
2. **经典组**训练时间约5-10分钟（CPU），**改进组**需要20-40分钟
3. **强烈建议使用GPU**加速训练，尤其是改进组/超强组
4. 如果内存不足，可以减小`BATCH_SIZE`或`MAXLEN`
5. 改进组/超强组模型参数量更大，需要更多内存

## 扩展建议

1. **模型改进**
   - 尝试双向RNN（Bidirectional）
   - 增加注意力机制（Attention）
   - 使用预训练词向量（Word2Vec, GloVe）

2. **超参数调优**
   - 网格搜索最优参数组合
   - 调整网络层数和神经元数量
   - 尝试不同的优化器和学习率

3. **数据增强**
   - 使用更大的词汇表
   - 尝试不同的序列长度
   - 添加数据增强技术

## 参考资料

- [Keras官方文档](https://keras.io/)
- [TensorFlow官方文档](https://www.tensorflow.org/)
- [IMDb数据集](https://ai.stanford.edu/~amaas/data/sentiment/)

## 作者

深度学习期末大作业

## 许可证

MIT License
