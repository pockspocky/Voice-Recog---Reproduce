#!/usr/bin/env python3
"""
集成声学模型类，整合了多种声学建模方法
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime  # 添加datetime导入
from typing import Dict, List, Union, Optional  # 添加typing导入
from config_manager import ConfigManager

# 导入各个声学模型组件
from .feature_extractor import FeatureExtractor
from .gmm_hmm import GMMHMM
from .dnn_model import DNNModel, DNNTrainer, DNNDataset
from .cnn_model import CNNModel, CNNTrainer, CNNDataset
from .rnn_model import RNNModel, RNNTrainer, RNNDataset
from .transformer_model import TransformerModel, TransformerTrainer, TransformerDataset
from .ctc_model import CTCModel

class IntegratedAcousticModel:
    """
    集成声学模型类，整合多种声学建模方法，包括GMM-HMM、DNN、CNN、RNN、Transformer和CTC。
    
    该类提供了一个统一的接口，用于训练和评估不同类型的声学模型，以及对它们进行集成预测。
    """
    
    def __init__(self, speaker_id: str, model_type: str = "all"):
        """
        初始化集成声学模型。
        
        Args:
            speaker_id (str): 说话人ID
            model_type (str): 要初始化的模型类型，可以是'all'、'gmm_hmm'、'dnn'、'cnn'、'rnn'、'transformer'或'ctc'
        """
        self.speaker_id = speaker_id
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(speaker_id)
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(**self.config["feature_config"])
        
        # 初始化模型
        self.models = {}
        self._init_models()
        
    def _init_models(self):
        """初始化选定的模型"""
        if self.model_type in ["all", "gmm_hmm"]:
            gmm_config = self.config_manager.get_model_config(self.speaker_id, "gmm_hmm")
            self.models["gmm_hmm"] = GMMHMM(
                n_states=gmm_config.get('n_states', 5),
                n_mix=gmm_config.get('n_mix', 4),
                cov_type=gmm_config.get('cov_type', 'diag'),
                n_iter=gmm_config.get('n_iter', 20)
            )
            
        if self.model_type in ["all", "dnn"]:
            dnn_config = self.config_manager.get_model_config(self.speaker_id, "dnn")
            self.models["dnn"] = DNNModel(
                input_dim=dnn_config.get('input_dim', 39),
                hidden_dims=dnn_config.get('hidden_dims', [512, 512]),
                output_dim=dnn_config.get('output_dim', 48),
                dropout_prob=dnn_config.get('dropout_prob', 0.2)
            )
            
        if self.model_type in ["all", "cnn"]:
            cnn_config = self.config_manager.get_model_config(self.speaker_id, "cnn")
            self.models["cnn"] = CNNModel(
                input_channels=cnn_config.get('input_channels', 1),
                feature_dim=cnn_config.get('feature_dim', 39),
                num_classes=cnn_config.get('num_classes', 48),
                context_window=cnn_config.get('context_window', 11),
                cnn_channels=cnn_config.get('cnn_channels', [64, 128, 256]),
                kernel_sizes=cnn_config.get('kernel_sizes', [3, 3, 3]),
                fc_dims=cnn_config.get('fc_dims', [1024, 512])
            )
            
        if self.model_type in ["all", "rnn"]:
            rnn_config = self.config_manager.get_model_config(self.speaker_id, "rnn")
            self.models["rnn"] = RNNModel(
                input_dim=rnn_config.get('input_dim', 39),
                hidden_dim=rnn_config.get('hidden_dim', 256),
                num_layers=rnn_config.get('num_layers', 3),
                output_dim=rnn_config.get('output_dim', 48),
                bidirectional=rnn_config.get('bidirectional', True),
                rnn_type=rnn_config.get('rnn_type', 'lstm'),
                dropout=rnn_config.get('dropout', 0.2)
            )
            
        if self.model_type in ["all", "transformer"]:
            transformer_config = self.config_manager.get_model_config(self.speaker_id, "transformer")
            self.models["transformer"] = TransformerModel(
                input_dim=transformer_config.get('input_dim', 39),
                d_model=transformer_config.get('d_model', 512),
                nhead=transformer_config.get('nhead', 8),
                num_encoder_layers=transformer_config.get('num_encoder_layers', 6),
                dim_feedforward=transformer_config.get('dim_feedforward', 2048),
                output_dim=transformer_config.get('output_dim', 48),
                dropout=transformer_config.get('dropout', 0.1),
                max_len=transformer_config.get('max_len', 1000)
            )
            
        if self.model_type in ["all", "ctc"]:
            ctc_config = self.config_manager.get_model_config(self.speaker_id, "ctc")
            self.models["ctc"] = CTCModel(
                input_dim=ctc_config.get('input_dim', 39),
                hidden_dim=ctc_config.get('hidden_dim', 256),
                output_dim=ctc_config.get('output_dim', 49),  # 包括blank标签
                num_layers=ctc_config.get('num_layers', 4),
                bidirectional=ctc_config.get('bidirectional', True),
                speaker_id=self.speaker_id
            )
    
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """提取音频特征"""
        return self.feature_extractor.extract_all_features(audio_data)
    
    def prepare_datasets(self, features: Dict[str, np.ndarray], labels: np.ndarray,
                        test_size: float = 0.15, val_size: float = 0.15) -> Dict[str, torch.utils.data.DataLoader]:
        """准备数据集"""
        # 检查样本数量
        n_samples = len(labels) if hasattr(labels, '__len__') else 1
        
        # 将特征字典转换为数组
        if isinstance(features, dict):
            # 选择一个特征类型，例如mfcc_librosa
            if 'mfcc_librosa' in features:
                feature_array = features['mfcc_librosa']
            elif 'melspectrogram' in features:
                feature_array = features['melspectrogram']
            else:
                # 使用第一个可用的特征
                feature_array = next(iter(features.values()))
        else:
            feature_array = features
        
        # 处理少样本情况
        if n_samples < 3:
            self.logger.warning(f"样本数量太少 ({n_samples}), 无法进行训练/验证/测试拆分。将只使用单个训练集。")
            # 创建单个训练集
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(feature_array),
                torch.LongTensor(labels)
            )
            loaders = {
                "train": torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=True
                )
            }
            return loaders
        
        # 首先分割训练集和测试集
        train_val_features, test_features, train_val_labels, test_labels = train_test_split(
            feature_array, labels, test_size=test_size, random_state=42
        )
        
        # 然后分割训练集和验证集
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_val_features, train_val_labels, test_size=val_size/(1-test_size), random_state=42
        )
        
        # 创建数据加载器
        loaders = {}
        for name, (feat, lab) in [
            ("train", (train_features, train_labels)),
            ("val", (val_features, val_labels)),
            ("test", (test_features, test_labels))
        ]:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(feat),
                torch.LongTensor(lab)
            )
            loaders[name] = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=(name == "train")
            )
        
        return loaders
    
    def train(self, features: Dict[str, np.ndarray], labels: np.ndarray,
              epochs: int = 30, batch_size: int = 32,
              learning_rate: float = 0.001, weight_decay: float = 0.0001,
              patience: int = 5) -> Dict[str, Dict[str, List[float]]]:
        """训练模型"""
        start_time = datetime.now()
        histories = {}
        
        # 准备数据集
        loaders = self.prepare_datasets(features, labels)
        
        # 训练每个模型
        for model_name, model in self.models.items():
            self.logger.info(f"开始训练 {model_name} 模型...")
            
            if isinstance(model, GMMHMM):
                # GMM-HMM模型训练
                history = model.train(features, labels)
            else:
                # 神经网络模型训练
                if model_name == "dnn":
                    trainer = DNNTrainer(model, "cpu")
                elif model_name == "cnn": 
                    trainer = CNNTrainer(model, "cpu")
                elif model_name == "rnn":
                    trainer = RNNTrainer(model, "cpu")
                elif model_name == "transformer":
                    trainer = TransformerTrainer(model, "cpu")
                elif model_name == "ctc":
                    # CTCModel已经内置了trainer
                    history = model.train(features, labels, epochs=epochs)
                    histories[model_name] = history
                    self._save_model(model_name, model)
                    continue
                
                # 检查是否有验证集
                if 'val' in loaders:
                    history = trainer.train(
                        loaders['train'],
                        loaders['val'],
                        epochs=epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        patience=patience
                    )
                else:
                    # 没有验证集时的训练方式
                    self.logger.info("由于样本量不足，将不使用验证集。")
                    history = trainer.train_without_validation(
                        loaders['train'],
                        epochs=epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay
                    )
            
            histories[model_name] = history
            
            # 保存模型
            self._save_model(model_name, model)
        
        # 更新训练历史
        training_time = (datetime.now() - start_time).total_seconds()
        # 检查历史是否有验证准确率
        if any('val_accuracy' in hist for hist in histories.values()):
            best_accuracy = max([max(hist.get('val_accuracy', [0])) for hist in histories.values()])
        else:
            best_accuracy = max([max(hist.get('train_accuracy', [0])) for hist in histories.values()])
            
        self.config_manager.update_training_history(self.speaker_id, best_accuracy, training_time)
        
        return histories
    
    def evaluate(self, features: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """评估模型"""
        metrics = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"评估 {model_name} 模型...")
            
            if isinstance(model, GMMHMM):
                # GMM-HMM模型评估
                metrics[model_name] = self._evaluate_gmm_hmm(model, features, labels)
            else:
                # 神经网络模型评估
                trainer = DNNTrainer(model, self.device)
                test_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.FloatTensor(features),
                        torch.LongTensor(labels)
                    ),
                    batch_size=32
                )
                metrics[model_name] = trainer.evaluate(test_loader)
        
        return metrics
    
    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """模型预测"""
        predictions = {}
        
        for model_name, model in self.models.items():
            if isinstance(model, GMMHMM):
                # GMM-HMM模型预测
                predictions[model_name] = self._predict_gmm_hmm(model, features)
            else:
                # 神经网络模型预测
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features)
                    outputs = model(features_tensor)
                    predictions[model_name] = outputs.numpy()
        
        return predictions
    
    def ensemble_predict(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """集成预测"""
        predictions = self.predict(features)
        
        # 简单投票
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        # 加权平均
        weights = {
            "gmm_hmm": 0.2,
            "dnn": 0.15,
            "cnn": 0.15,
            "rnn": 0.2,
            "transformer": 0.2,
            "ctc": 0.1
        }
        
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                ensemble_pred += weights[model_name] * pred
                total_weight += weights[model_name]
        
        return ensemble_pred / total_weight
    
    def _save_model(self, model_name: str, model: Union[GMMHMM, torch.nn.Module]):
        """保存模型"""
        save_dir = os.path.join("models", self.speaker_id)
        os.makedirs(save_dir, exist_ok=True)
        
        if isinstance(model, GMMHMM):
            # 保存GMM-HMM模型
            model.save_models(os.path.join(save_dir, model_name))
        else:
            # 保存神经网络模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_model.pth"))
    
    def _load_model(self, model_name: str) -> Optional[Union[GMMHMM, torch.nn.Module]]:
        """加载模型"""
        save_dir = os.path.join("models", self.speaker_id)
        
        if not os.path.exists(save_dir):
            self.logger.warning(f"模型目录不存在: {save_dir}")
            return None
        
        if model_name == "gmm_hmm":
            # 加载GMM-HMM模型
            model_path = os.path.join(save_dir, model_name)
            if os.path.exists(model_path):
                model = GMMHMM()
                model.load_models(model_path)
                return model
            else:
                self.logger.warning(f"GMM-HMM模型不存在: {model_path}")
                return None
        else:
            model_class = {
                "dnn": DNNModel,
                "cnn": CNNModel,
                "rnn": RNNModel,
                "transformer": TransformerModel,
                "ctc": CTCModel
            }[model_name]
            
            model = model_class(
                input_dim=self.config_manager.get_model_config(self.speaker_id, model_name).get('input_dim', 39),
                hidden_dim=self.config_manager.get_model_config(self.speaker_id, model_name).get('hidden_dim', 256),
                output_dim=self.config_manager.get_model_config(self.speaker_id, model_name).get('output_dim', 48),
                num_layers=self.config_manager.get_model_config(self.speaker_id, model_name).get('num_layers', 4),
                bidirectional=self.config_manager.get_model_config(self.speaker_id, model_name).get('bidirectional', True),
                dropout=self.config_manager.get_model_config(self.speaker_id, model_name).get('dropout', 0.2),
                rnn_type=self.config_manager.get_model_config(self.speaker_id, model_name).get('rnn_type', 'lstm'),
                speaker_id=self.speaker_id
            )
            model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_model.pth")))
        
        return model
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, figsize=(10, 8)):
        """
        绘制混淆矩阵
        
        Args:
            y_true (numpy.ndarray): 真实标签
            y_pred (numpy.ndarray): 预测标签
            class_names (list, optional): 类别名称
            figsize (tuple): 图形大小
            
        Returns:
            matplotlib.figure.Figure: 混淆矩阵图形
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制混淆矩阵
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 设置轴标签
        if class_names is not None:
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                ylabel='True label',
                xlabel='Predicted label'
            )
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在每个单元格中添加文本
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 设置图形标题
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        
        return fig
    
    def _evaluate_gmm_hmm(self, model, features, labels):
        """
        评估GMM-HMM模型
        
        参数:
            model: GMMHMM模型
            features: 特征字典或数组
            labels: 标签数组
            
        返回:
            评估指标字典
        """
        # 进行预测
        predictions = self._predict_gmm_hmm(model, features)
        
        # 将features转换为数组（如果是字典的话）
        if isinstance(features, dict):
            feature_array = np.vstack([feat for feat_name, feat in features.items()])
        else:
            feature_array = features
            
        # 计算准确率
        accuracy = np.mean(predictions == labels)
        
        # 计算每个类别的得分
        unique_labels = np.unique(labels)
        class_scores = {}
        
        for label in unique_labels:
            label_mask = (labels == label)
            if np.sum(label_mask) > 0:
                class_accuracy = np.mean(predictions[label_mask] == label)
                class_scores[str(label)] = class_accuracy
        
        # 返回评估指标
        return {
            'accuracy': accuracy,
            'class_scores': class_scores
        }
    
    def _predict_gmm_hmm(self, model, features):
        """
        使用GMM-HMM模型进行预测
        
        参数:
            model: GMMHMM模型
            features: 特征字典或数组
            
        返回:
            预测的标签数组
        """
        return model.predict(features) 