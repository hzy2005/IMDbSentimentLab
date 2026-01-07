"""
模型训练和评估模块
"""
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None
        self.train_time = 0
        self.monitor_metric_used = None

    def _has_auc_metric(self) -> bool:
        """
        稳健判断模型是否包含 AUC 指标（兼容 compile_metrics 包装情况）
        """
        # 1) 优先从 model.metrics 对象里查
        try:
            for m in (self.model.metrics or []):
                name = getattr(m, "name", "").lower()
                if "auc" == name or name.endswith("auc"):
                    return True
                if isinstance(m, tf.keras.metrics.AUC):
                    return True
        except Exception:
            pass

        # 2) fallback：metrics_names
        metrics_names = [str(m).lower() for m in (self.model.metrics_names or [])]
        if any("auc" in m for m in metrics_names):
            return True

        return False

    def _get_monitor_metric(self) -> str:
        if "Ultra" in self.model_name:
            return "val_auc"
        if self._has_auc_metric():
            return "val_auc"
        return "val_accuracy"

    def _use_reduce_on_plateau(self) -> bool:
        """
        若 optimizer 使用 LearningRateSchedule（WarmupCosineSchedule 等），
        不应再叠加 ReduceLROnPlateau。
        """
        opt = self.model.optimizer
        lr_attr = getattr(opt, "learning_rate", None)
        base_lr = getattr(opt, "_learning_rate", None)
        if isinstance(lr_attr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return False
        if isinstance(base_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return False
        return True

    def train(self, x_train, y_train, x_val, y_val,
              epochs=20, batch_size=128, verbose=1):
        print(f"\n{'='*60}")
        print(f"开始训练模型: {self.model_name}")
        print(f"{'='*60}")

        monitor_metric = self._get_monitor_metric()
        self.monitor_metric_used = monitor_metric
        print(f"监控指标: {monitor_metric}")

        patience = 8 if "Ultra" in self.model_name else 5
        callbacks = [
            EarlyStopping(
                monitor=monitor_metric,
                mode="max",
                patience=patience,
                min_delta=0.0005,
                restore_best_weights=True,
                verbose=1
            )
        ]

        if self._use_reduce_on_plateau():
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=monitor_metric,
                    mode="max",
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=1
                )
            )
        else:
            print("[Info] 检测到 LearningRateSchedule，跳过 ReduceLROnPlateau")

        start_time = time.time()

        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )

        self.train_time = time.time() - start_time
        print(f"\n训练完成！耗时: {self.train_time:.2f}秒")

        return self.history

    def evaluate(self, x_test, y_test):
        print(f"\n{'='*60}")
        print(f"评估模型: {self.model_name}")
        print(f"{'='*60}")

        eval_dict = self.model.evaluate(x_test, y_test, verbose=0, return_dict=True)
        test_loss = float(eval_dict.get("loss", 0.0))
        test_acc = float(eval_dict.get("accuracy", 0.0))
        test_auc = float(eval_dict["auc"]) if "auc" in eval_dict else None

        # history best
        best_train_acc = max(self.history.history.get("accuracy", [0.0]))
        best_val_acc = max(self.history.history.get("val_accuracy", [0.0]))
        final_train_loss = self.history.history.get("loss", [0.0])[-1]
        final_val_loss = self.history.history.get("val_loss", [0.0])[-1]

        best_train_auc = max(self.history.history.get("auc", [0.0])) if "auc" in self.history.history else None
        best_val_auc = max(self.history.history.get("val_auc", [0.0])) if "val_auc" in self.history.history else None

        results = {
            "model_name": self.model_name,
            "monitor_metric_used": self.monitor_metric_used,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_auc": test_auc,
            "best_train_accuracy": best_train_acc,
            "best_val_accuracy": best_val_acc,
            "best_train_auc": best_train_auc,
            "best_val_auc": best_val_auc,
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "train_time": float(self.train_time),
            "epochs_trained": len(self.history.history.get("loss", []))
        }

        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        if test_auc is not None:
            print(f"测试AUC: {test_auc:.4f}")
        print(f"最佳训练准确率: {best_train_acc:.4f}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
        if best_train_auc is not None:
            print(f"最佳训练AUC: {best_train_auc:.4f}")
        if best_val_auc is not None:
            print(f"最佳验证AUC: {best_val_auc:.4f}")
        print(f"训练时间: {self.train_time:.2f}秒")
        print(f"训练轮数: {results['epochs_trained']}")
        print(f"{'='*60}\n")

        return results

    def predict(self, x):
        return self.model.predict(x)

    def get_history(self):
        return self.history


def compare_models(results_list):
    print(f"\n{'='*80}")
    print(f"{'模型对比分析':^80}")
    print(f"{'='*80}")

    print(f"{'模型名称':<26} {'测试准确率':<12} {'测试AUC':<10} {'测试损失':<12} {'训练时间(秒)':<15} {'轮数':<6}")
    print(f"{'-'*80}")

    for result in results_list:
        auc = result.get("test_auc")
        auc_text = "-" if auc is None else f"{auc:.4f}"
        print(
            f"{result['model_name']:<26} "
            f"{result['test_accuracy']:<12.4f} "
            f"{auc_text:<10} "
            f"{result['test_loss']:<12.4f} "
            f"{result['train_time']:<15.2f} "
            f"{result['epochs_trained']:<6}"
        )

    print(f"{'='*80}")

    best_acc_model = max(results_list, key=lambda x: x["test_accuracy"])
    best_loss_model = min(results_list, key=lambda x: x["test_loss"])
    fastest_model = min(results_list, key=lambda x: x["train_time"])

    print(f"\n最佳准确率模型: {best_acc_model['model_name']} "
          f"(准确率: {best_acc_model['test_accuracy']:.4f})")
    if best_acc_model.get("test_auc") is not None:
        print(f"  参考AUC: {best_acc_model['test_auc']:.4f}")
    print(f"最低损失模型: {best_loss_model['model_name']} "
          f"(损失: {best_loss_model['test_loss']:.4f})")
    print(f"最快训练模型: {fastest_model['model_name']} "
          f"(时间: {fastest_model['train_time']:.2f}秒)")
    print(f"{'='*80}\n")
