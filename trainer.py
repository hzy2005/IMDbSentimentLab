"""
模型训练和评估模块
"""
import numpy as np
import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, model_name):
        """
        初始化训练器
        
        Args:
            model: Keras模型
            model_name: 模型名称
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        self.train_time = 0
        
    def train(self, x_train, y_train, x_val, y_val, 
              epochs=20, batch_size=128, verbose=1):
        """
        训练模型
        
        Args:
            x_train: 训练数据
            y_train: 训练标签
            x_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 详细程度
        
        Returns:
            训练历史
        """
        print(f"\n{'='*60}")
        print(f"开始训练模型: {self.model_name}")
        print(f"{'='*60}")
        
        # 定义回调函数
        callbacks = [
            # 早停：验证损失3轮不下降则停止
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            # 学习率衰减：验证损失2轮不下降则降低学习率
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        # 记录训练时间
        self.train_time = time.time() - start_time
        
        print(f"\n训练完成！耗时: {self.train_time:.2f}秒")
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """
        评估模型
        
        Args:
            x_test: 测试数据
            y_test: 测试标签
        
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"评估模型: {self.model_name}")
        print(f"{'='*60}")
        
        # 评估模型
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        
        # 获取训练历史中的最佳结果
        best_train_acc = max(self.history.history['accuracy'])
        best_val_acc = max(self.history.history['val_accuracy'])
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        results = {
            'model_name': self.model_name,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_train_accuracy': best_train_acc,
            'best_val_accuracy': best_val_acc,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'train_time': self.train_time,
            'epochs_trained': len(self.history.history['loss'])
        }
        
        # 打印结果
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"最佳训练准确率: {best_train_acc:.4f}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
        print(f"训练时间: {self.train_time:.2f}秒")
        print(f"训练轮数: {results['epochs_trained']}")
        print(f"{'='*60}\n")
        
        return results
    
    def predict(self, x):
        """
        预测
        
        Args:
            x: 输入数据
        
        Returns:
            预测结果
        """
        return self.model.predict(x)
    
    def get_history(self):
        """获取训练历史"""
        return self.history


def compare_models(results_list):
    """
    比较多个模型的结果
    
    Args:
        results_list: 结果列表
    """
    print(f"\n{'='*80}")
    print(f"{'模型对比分析':^80}")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'模型名称':<20} {'测试准确率':<12} {'测试损失':<12} {'训练时间(秒)':<15} {'训练轮数':<10}")
    print(f"{'-'*80}")
    
    # 打印每个模型的结果
    for result in results_list:
        print(f"{result['model_name']:<20} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['test_loss']:<12.4f} "
              f"{result['train_time']:<15.2f} "
              f"{result['epochs_trained']:<10}")
    
    print(f"{'='*80}")
    
    # 找出最佳模型
    best_acc_model = max(results_list, key=lambda x: x['test_accuracy'])
    best_loss_model = min(results_list, key=lambda x: x['test_loss'])
    fastest_model = min(results_list, key=lambda x: x['train_time'])
    
    print(f"\n最佳准确率模型: {best_acc_model['model_name']} "
          f"(准确率: {best_acc_model['test_accuracy']:.4f})")
    print(f"最低损失模型: {best_loss_model['model_name']} "
          f"(损失: {best_loss_model['test_loss']:.4f})")
    print(f"最快训练模型: {fastest_model['model_name']} "
          f"(时间: {fastest_model['train_time']:.2f}秒)")
    print(f"{'='*80}\n")