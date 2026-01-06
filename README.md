# IMDb电影评论情感分析 - CNN+RNN混合模型

## 项目简介

本项目使用深度学习方法对IMDb电影评论进行情感分析（二分类：正面/负面）。项目实现了多种CNN+RNN混合模型，并对性能进行对比分析，同时支持注意力可视化与结果报告输出。

**包含三个实验组：**
- **经典组**：标准CNN+RNN模型（准确率~88%）
- **改进组**：优化CNN+RNN模型（双向RNN + BatchNorm）
- **超强组**：更先进架构（多尺度CNN + 残差 + BiRNN + Masked Attention）

## 模型清单（按实验组）

**经典组（models.py）**
- CNN+SimpleRNN
- CNN+GRU
- CNN+LSTM
- CNN_Only
- RNN_Only_LSTM

**改进组（models_advanced.py）**
- Advanced_CNN+BiSimpleRNN
- Advanced_CNN+BiGRU
- Advanced_CNN+BiLSTM

**超强组（models_ultra.py）**
- Ultra_MSResCNN+LSTM+Attention
- Ultra_MSResCNN+BiGRU+Attention
- Ultra_ResCNN+BiGRU+Attention
- Ultra_TextCNN+BiLSTM_PoolCat

## 项目结构

```
.
├── data_loader.py      # 数据加载和预处理模块
├── models.py           # 经典CNN+RNN模型定义
├── models_advanced.py  # 改进CNN+RNN模型定义
├── models_ultra.py     # 超强CNN+RNN模型定义
├── attention_visualizer.py  # 注意力可视化工具
├── trainer.py          # 训练和评估模块
├── visualizer.py       # 训练曲线/对比图可视化模块
├── main.py             # 统一主程序入口
├── requirements.txt    # 依赖包列表
├── README.md           # 项目说明文档
├── results/            # 经典组输出目录
├── results_advanced/   # 改进组输出目录
└── results_ultra/      # 超强组输出目录（含 attention/）
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

### 运行统一入口（推荐）

```bash
python main.py
```

**统一入口说明：**
- `main.py` 内部包含经典/改进/超强三组模型
- 可在 `main.py` 中修改 `run_groups` 控制要运行的实验组
- 默认输出目录：`results/`、`results_advanced/`、`results_ultra/`

## 超强组关键增强

- Masked Attention：忽略padding，注意力更稳更可解释
- 多尺度CNN + 残差块：提升特征表达
- AdamW + warmup+cosine 学习率（若可用）
- 输出包含 attention 可视化（默认保存到 `results_ultra/attention/`）

## 输出结果

运行完成后，各输出目录将包含：

1) `*_structure.txt`：模型结构  
2) `*_training_history.png`：训练/验证曲线  
3) `all_models_comparison.png`：训练/验证对比图  
4) `results_comparison.png`：指标柱状对比  
5) `detailed_results.txt`：详细指标  
6) `experiment_summary.txt`：实验要求对照说明  
7) `results_ultra/attention/*.png`：注意力可视化图（仅Attention模型）

## 注意事项

1. 首次运行会自动下载IMDb数据集（约80MB）
2. 超强组计算量更大，建议使用GPU
3. 如需预训练词向量，可设置环境变量：
   - `USE_GLOVE=1`
   - `GLOVE_PATH=path/to/glove.6B.300d.txt`

## 参考资料

- https://keras.io/
- https://www.tensorflow.org/
- https://ai.stanford.edu/~amaas/data/sentiment/

## 许可证

MIT License
