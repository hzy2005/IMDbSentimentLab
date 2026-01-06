"""
快速测试脚本 - 用于验证代码是否正常工作
使用较小的参数快速运行一次完整流程
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader import IMDbDataLoader
from models import get_all_models, compile_model, print_model_summary
from trainer import ModelTrainer, compare_models
from visualizer import Visualizer


def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def quick_test():
    """快速测试函数"""
    print("="*80)
    print(" "*25 + "快速测试模式")
    print("="*80)
    
    set_seed(42)
    
    # 使用较小的参数进行快速测试
    MAX_FEATURES = 5000   # 减小词汇表
    MAXLEN = 200          # 减小序列长度
    EMBEDDING_DIM = 64    # 减小嵌入维度
    BATCH_SIZE = 256      # 增大批次大小
    EPOCHS = 3            # 只训练3轮
    
    print("\n【测试配置】")
    print(f"词汇表大小: {MAX_FEATURES}")
    print(f"序列长度: {MAXLEN}")
    print(f"嵌入维度: {EMBEDDING_DIM}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    
    # 加载数据
    print("\n【加载数据】")
    data_loader = IMDbDataLoader(max_features=MAX_FEATURES, maxlen=MAXLEN)
    (x_train_full, y_train_full), (x_test, y_test) = data_loader.load_data()
    
    # 使用更小的数据集进行快速测试
    x_train_full = x_train_full[:5000]
    y_train_full = y_train_full[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"训练集: {x_train.shape[0]} 样本")
    print(f"验证集: {x_val.shape[0]} 样本")
    print(f"测试集: {x_test.shape[0]} 样本")
    
    # 创建模型
    print("\n【创建模型】")
    models_dict = get_all_models(
        max_features=MAX_FEATURES,
        maxlen=MAXLEN,
        embedding_dim=EMBEDDING_DIM
    )
    
    for model_name, model in models_dict.items():
        compile_model(model)
        print(f"✓ {model_name} 创建成功")
    
    # 初始化可视化器
    visualizer = Visualizer(save_dir='test_results')
    
    # 训练和评估
    print("\n【训练和评估】")
    trainers = {}
    histories = {}
    results_list = []
    
    for model_name, model in models_dict.items():
        print(f"\n训练 {model_name}...")
        
        trainer = ModelTrainer(model, model_name)
        history = trainer.train(
            x_train, y_train,
            x_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0  # 不显示详细训练过程
        )
        
        trainers[model_name] = trainer
        histories[model_name] = history
        
        # 评估
        results = trainer.evaluate(x_test, y_test)
        results_list.append(results)
        
        print(f"✓ {model_name} 完成 - 测试准确率: {results['test_accuracy']:.4f}")
    
    # 对比结果
    print("\n【模型对比】")
    compare_models(results_list)
    
    # 生成可视化
    print("\n【生成可视化】")
    for model_name, history in histories.items():
        visualizer.plot_training_history(history, model_name)
    
    visualizer.plot_all_models_comparison(histories)
    visualizer.plot_results_comparison(results_list)
    
    print("\n" + "="*80)
    print(" "*25 + "测试完成！")
    print("="*80)
    print("\n结果已保存到 'test_results' 目录")
    print("如果测试成功，可以运行 'python main.py' 进行完整训练")
    print("="*80 + "\n")


if __name__ == '__main__':
    quick_test()