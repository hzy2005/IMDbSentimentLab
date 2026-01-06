"""
可视化模块 - 用于绘制训练过程和模型对比
"""
import matplotlib.pyplot as plt
import numpy as np
import os


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, save_dir='results'):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存图片的目录
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_training_history(self, history, model_name):
        """
        绘制单个模型的训练历史
        
        Args:
            history: Keras训练历史对象
            model_name: 模型名称
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 绘制准确率
        axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
        axes[0].set_title(f'{model_name} - 准确率变化', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[0].set_ylabel('准确率 (Accuracy)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 绘制损失
        axes[1].plot(history.history['loss'], label='训练损失', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='验证损失', linewidth=2)
        axes[1].set_title(f'{model_name} - 损失变化', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[1].set_ylabel('损失 (Loss)', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"{model_name.replace('+', '_').replace(' ', '_')}_training_history.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存: {filepath}")
        
        plt.close()
    
    def plot_all_models_comparison(self, histories_dict):
        """
        绘制所有模型的对比图
        
        Args:
            histories_dict: 字典，键为模型名称，值为训练历史
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 1. 训练准确率对比
        for idx, (model_name, history) in enumerate(histories_dict.items()):
            axes[0, 0].plot(history.history['accuracy'], 
                          label=model_name, 
                          linewidth=2, 
                          color=colors[idx % len(colors)])
        axes[0, 0].set_title('训练准确率对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[0, 0].set_ylabel('准确率 (Accuracy)', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 验证准确率对比
        for idx, (model_name, history) in enumerate(histories_dict.items()):
            axes[0, 1].plot(history.history['val_accuracy'], 
                          label=model_name, 
                          linewidth=2, 
                          color=colors[idx % len(colors)])
        axes[0, 1].set_title('验证准确率对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[0, 1].set_ylabel('准确率 (Accuracy)', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练损失对比
        for idx, (model_name, history) in enumerate(histories_dict.items()):
            axes[1, 0].plot(history.history['loss'], 
                          label=model_name, 
                          linewidth=2, 
                          color=colors[idx % len(colors)])
        axes[1, 0].set_title('训练损失对比', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[1, 0].set_ylabel('损失 (Loss)', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 验证损失对比
        for idx, (model_name, history) in enumerate(histories_dict.items()):
            axes[1, 1].plot(history.history['val_loss'], 
                          label=model_name, 
                          linewidth=2, 
                          color=colors[idx % len(colors)])
        axes[1, 1].set_title('验证损失对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[1, 1].set_ylabel('损失 (Loss)', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'all_models_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存: {filepath}")
        
        plt.close()
    
    def plot_results_comparison(self, results_list):
        """
        绘制模型结果对比柱状图
        
        Args:
            results_list: 结果列表
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        model_names = [r['model_name'] for r in results_list]
        test_accs = [r['test_accuracy'] for r in results_list]
        test_losses = [r['test_loss'] for r in results_list]
        train_times = [r['train_time'] for r in results_list]
        epochs = [r['epochs_trained'] for r in results_list]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 1. 测试准确率对比
        bars1 = axes[0, 0].bar(model_names, test_accs, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('测试准确率对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('准确率 (Accuracy)', fontsize=12)
        axes[0, 0].set_ylim([min(test_accs) - 0.02, max(test_accs) + 0.02])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. 测试损失对比
        bars2 = axes[0, 1].bar(model_names, test_losses, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('测试损失对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('损失 (Loss)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. 训练时间对比
        bars3 = axes[1, 0].bar(model_names, train_times, color=colors, alpha=0.8, edgecolor='black')
        axes[1, 0].set_title('训练时间对比', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('时间 (秒)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}s',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. 训练轮数对比
        bars4 = axes[1, 1].bar(model_names, epochs, color=colors, alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('训练轮数对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('轮数 (Epochs)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar in bars4:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'results_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"结果对比图已保存: {filepath}")
        
        plt.close()
    
    def save_model_structure(self, model, model_name):
        """
        保存模型结构到文本文件
        
        Args:
            model: Keras模型
            model_name: 模型名称
        """
        filename = f"{model_name.replace('+', '_').replace(' ', '_')}_structure.txt"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 重定向模型摘要输出到文件
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print(f"模型结构已保存: {filepath}")