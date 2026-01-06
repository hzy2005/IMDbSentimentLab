"""
IMDb电影评论情感分析 - 统一主程序
同时支持经典/改进/超强CNN+RNN模型对比实验
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb

from data_loader import IMDbDataLoader
from models import get_all_models, compile_model, print_model_summary
from models_advanced import get_all_advanced_models, compile_advanced_model
from models_ultra import get_all_ultra_models, compile_ultra_model
from trainer import ModelTrainer, compare_models
from visualizer import Visualizer
from attention_visualizer import decode_review, extract_attention_weights, visualize_attention


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _safe_name(name):
    return name.replace('+', '_').replace(' ', '_')


def _write_experiment_summary(save_dir, title, model_names):
    summary_file = os.path.join(save_dir, 'experiment_summary.txt')
    structure_files = [f"{_safe_name(name)}_structure.txt" for name in model_names]
    history_files = [f"{_safe_name(name)}_training_history.png" for name in model_names]
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{title} - 实验要求对照说明\n")
        f.write("="*60 + "\n")
        f.write("1) 模型数量与类型\n")
        f.write("   - CNN+RNN 三个及以上模型对比\n")
        f.write("   - 当前模型列表:\n")
        for name in model_names:
            f.write(f"     * {name}\n")
        f.write("\n2) 模型结构输出\n")
        for filename in structure_files:
            f.write(f"   - {save_dir}/{filename}\n")
        f.write("\n3) 损失可视化输出\n")
        for filename in history_files:
            f.write(f"   - {save_dir}/{filename}（含训练/验证损失曲线）\n")
        f.write(f"   - {save_dir}/all_models_comparison.png（含训练/验证损失对比）\n\n")
        f.write("4) 对比分析结果\n")
        f.write(f"   - {save_dir}/detailed_results.txt\n")
        f.write(f"   - {save_dir}/results_comparison.png\n")
    print(f"实验要求对照说明已保存到: {summary_file}")


def _select_indices_by_label(y, label, k):
    indices = np.where(y == label)[0]
    return indices[:k].tolist()


def _run_attention_visualization(trainers, x_test, y_test, save_dir, word_index, top_k=20):
    attention_dir = os.path.join(save_dir, "attention")
    os.makedirs(attention_dir, exist_ok=True)

    pos_indices = _select_indices_by_label(y_test, 1, 3)
    neg_indices = _select_indices_by_label(y_test, 0, 3)
    sample_list = [(i, "pos") for i in pos_indices] + [(i, "neg") for i in neg_indices]

    for model_name, trainer in trainers.items():
        if "Attention" not in model_name:
            continue

        print(f"\n[Attention] 模型: {model_name}")
        model = trainer.model

        for j, (idx, label_text) in enumerate(sample_list):
            x_sample = x_test[idx:idx + 1]
            pred = float(model.predict(x_sample, verbose=0).reshape(-1)[0])

            tokens, _ = decode_review(x_sample[0].tolist(), word_index)
            weights = extract_attention_weights(model, x_sample)
            if weights is None:
                print(f"  Sample {label_text}_{j}: attention weights not found.")
                continue

            filename = f"{_safe_name(model_name)}_{label_text}_{j}.png"
            save_path = os.path.join(attention_dir, filename)
            top_items = visualize_attention(tokens, weights, save_path, top_k=top_k)

            true_label = 1 if label_text == "pos" else 0
            print(f"  Sample {label_text}_{j} | true={true_label} pred={pred:.4f}")
            if top_items:
                top_str = ", ".join([f"{w}({weight:.4f})" for w, weight in top_items])
                print(f"  Top-{len(top_items)}: {top_str}")


def run_experiment(config):
    """执行单个实验组（经典/改进/超强）"""
    title = config['title']
    print("\n" + "="*80)
    print(f"{title} | {config['desc']}")
    print("="*80)

    print("\n【步骤】数据加载与预处理")
    print("-"*80)
    data_loader = IMDbDataLoader(
        max_features=config['max_features'],
        maxlen=config['maxlen']
    )
    (x_train_full, y_train_full), (x_test, y_test) = data_loader.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=0.2,
        random_state=42
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {x_train.shape[0]} 样本")
    print(f"  验证集: {x_val.shape[0]} 样本")
    print(f"  测试集: {x_test.shape[0]} 样本")

    if config.get('show_sample', False):
        print(f"\n样本展示:")
        sample_text, sample_label = data_loader.get_sample_text(0, 'train')
        if sample_text:
            print(f"  标签: {'正面评价' if sample_label == 1 else '负面评价'}")
            print(f"  文本: {sample_text[:200]}...")

    print("\n【步骤】创建模型")
    print("-"*80)
    models_dict = config['get_models'](
        max_features=config['max_features'],
        maxlen=config['maxlen'],
        embedding_dim=config['embedding_dim']
    )

    print("当前模型列表:")
    for name in models_dict.keys():
        print(f"  - {name}")

    for model_name, model in models_dict.items():
        if config.get('learning_rate') is None:
            config['compile_model'](model)
        else:
            config['compile_model'](model, learning_rate=config['learning_rate'])
        print_model_summary(model)

    visualizer = Visualizer(save_dir=config['save_dir'])

    print("\n保存模型结构...")
    for model_name, model in models_dict.items():
        visualizer.save_model_structure(model, model_name)

    print("\n【步骤】训练模型")
    print("-"*80)
    trainers = {}
    histories = {}

    for model_name, model in models_dict.items():
        print(f"\n>>> 训练模型: {model_name}")
        trainer = ModelTrainer(model, model_name)
        history = trainer.train(
            x_train, y_train,
            x_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=1
        )
        trainers[model_name] = trainer
        histories[model_name] = history
        visualizer.plot_training_history(history, model_name)

    print("\n【步骤】评估模型")
    print("-"*80)
    results_list = []
    for model_name, trainer in trainers.items():
        results = trainer.evaluate(x_test, y_test)
        print(
            f"{model_name} | Test Loss: {results['test_loss']:.4f} "
            f"| Test Accuracy: {results['test_accuracy']:.4f}"
        )
        results_list.append(results)

    if config.get('enable_attention_vis', False):
        word_index = imdb.get_word_index()
        _run_attention_visualization(
            trainers,
            x_test,
            y_test,
            config['save_dir'],
            word_index,
            top_k=config.get('attention_top_k', 20)
        )

    print("\n【步骤】模型对比分析")
    print("-"*80)
    compare_models(results_list)
    visualizer.plot_all_models_comparison(histories)
    visualizer.plot_results_comparison(results_list)

    print("\n【步骤】保存结果")
    print("-"*80)
    results_file = os.path.join(config['save_dir'], 'detailed_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{title} - IMDb电影评论情感分析实验结果\n")
        f.write("="*80 + "\n\n")

        f.write("实验配置:\n")
        f.write(f"  词汇表大小: {config['max_features']}\n")
        f.write(f"  序列最大长度: {config['maxlen']}\n")
        f.write(f"  词嵌入维度: {config['embedding_dim']}\n")
        f.write(f"  批次大小: {config['batch_size']}\n")
        f.write(f"  最大训练轮数: {config['epochs']}\n")
        if config.get('learning_rate') is not None:
            f.write(f"  学习率: {config['learning_rate']}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("模型详细结果:\n")
        f.write("="*80 + "\n\n")

        for result in results_list:
            f.write(f"模型: {result['model_name']}\n")
            f.write(f"  测试准确率: {result['test_accuracy']:.4f}\n")
            f.write(f"  测试损失: {result['test_loss']:.4f}\n")
            f.write(f"  最佳训练准确率: {result['best_train_accuracy']:.4f}\n")
            f.write(f"  最佳验证准确率: {result['best_val_accuracy']:.4f}\n")
            f.write(f"  最终训练损失: {result['final_train_loss']:.4f}\n")
            f.write(f"  最终验证损失: {result['final_val_loss']:.4f}\n")
            f.write(f"  训练时间: {result['train_time']:.2f}秒\n")
            f.write(f"  训练轮数: {result['epochs_trained']}\n")
            f.write("\n" + "-"*80 + "\n\n")

        best_acc_model = max(results_list, key=lambda x: x['test_accuracy'])
        best_loss_model = min(results_list, key=lambda x: x['test_loss'])
        fastest_model = min(results_list, key=lambda x: x['train_time'])

        f.write("="*80 + "\n")
        f.write("最佳模型总结:\n")
        f.write("="*80 + "\n\n")
        f.write(f"最佳准确率模型: {best_acc_model['model_name']}\n")
        f.write(f"  准确率: {best_acc_model['test_accuracy']:.4f}\n\n")
        f.write(f"最低损失模型: {best_loss_model['model_name']}\n")
        f.write(f"  损失: {best_loss_model['test_loss']:.4f}\n\n")
        f.write(f"最快训练模型: {fastest_model['model_name']}\n")
        f.write(f"  时间: {fastest_model['train_time']:.2f}秒\n")

    print(f"详细结果已保存到: {results_file}")
    _write_experiment_summary(config['save_dir'], title, list(models_dict.keys()))

    print("\n" + "="*80)
    print(f"{title} 完成")
    print("="*80)
    print(f"结果保存目录: {config['save_dir']}")


def main():
    """主函数"""
    print("="*80)
    print(" "*12 + "IMDb电影评论情感分析 - 统一对比实验")
    print(" "*6 + "包含经典/改进/超强CNN+RNN模型，目标准确率95%+")
    print("="*80)

    set_seed(42)

    # 修改此列表可控制要运行的实验组
    run_groups = ['ultra','advanced']

    configs = [
        {
            'key': 'classic',
            'title': '经典组',
            'desc': 'CNN+SimpleRNN/GRU/LSTM（含CNN-only与RNN-only基线）',
            'get_models': get_all_models,
            'compile_model': compile_model,
            'max_features': 10000,
            'maxlen': 500,
            'embedding_dim': 128,
            'batch_size': 128,
            'epochs': 20,
            'learning_rate': 0.001,
            'save_dir': 'results',
            'show_sample': True
        },
        {
            'key': 'advanced',
            'title': '改进组',
            'desc': '更深CNN + 双向RNN + BatchNorm/SpatialDropout',
            'get_models': get_all_advanced_models,
            'compile_model': compile_advanced_model,
            'max_features': 20000,
            'maxlen': 500,
            'embedding_dim': 256,
            'batch_size': 64,
            'epochs': 30,
            'learning_rate': 0.0005,
            'save_dir': 'results_advanced'
        },
        {
            'key': 'ultra',
            'title': '超强组',
            'desc': '多尺度CNN + 注意力 + TextCNN + BiLSTM（目标95%+）',
            'get_models': get_all_ultra_models,
            'compile_model': compile_ultra_model,
            'max_features': 20000,
            'maxlen': 300,
            'embedding_dim': 300,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.0003,
            'save_dir': 'results_ultra',
            'enable_attention_vis': True,
            'attention_top_k': 20
        }
    ]

    for cfg in configs:
        if cfg['key'] in run_groups:
            run_experiment(cfg)

    print("\n" + "="*80)
    print(" "*25 + "全部实验完成")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
