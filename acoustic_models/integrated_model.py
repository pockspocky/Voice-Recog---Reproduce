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

# 导入各个声学模型组件
from .feature_extractor import FeatureExtractor
from .gmm_hmm import GMMHMMModel
from .dnn_model import DNNModel, DNNTrainer, DNNDataset
from .cnn_model import CNNModel, CNNTrainer, CNNDataset
from .rnn_model import RNNModel, RNNTrainer, RNNDataset
from .transformer_model import TransformerModel, TransformerTrainer, TransformerDataset
from .ctc_model import CTCRNN, CTCTrainer, CTCDataset

class IntegratedAcousticModel:
    """
    集成声学模型类，整合多种声学建模方法，包括GMM-HMM、DNN、CNN、RNN、Transformer和CTC。
    
    该类提供了一个统一的接口，用于训练和评估不同类型的声学模型，以及对它们进行集成预测。
    """
    
    def __init__(self, config, device='cpu'):
        """
        初始化集成声学模型。
        
        Args:
            config (dict): 配置字典，包含各个模型的参数。
            device (str): 计算设备，可以是'cpu'或'cuda'。
        """
        self.config = config
        self.device = device
        self.models = {}
        self.trainers = {}
        self.logger = self._setup_logger()
        
        # 创建特征提取器
        self.feature_extractor = FeatureExtractor(
            sample_rate=config.get('sample_rate', 16000),
            n_fft=config.get('n_fft', 512),
            hop_length=config.get('hop_length', 160),
            n_mels=config.get('n_mels', 80),
            n_mfcc=config.get('n_mfcc', 39)
        )
        
        # 初始化各个模型组件
        self._initialize_models()
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('IntegratedAcousticModel')
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # 将处理器添加到日志记录器
        logger.addHandler(ch)
        
        return logger
    
    def _initialize_models(self):
        """初始化各个模型组件"""
        # 初始化GMM-HMM模型
        if 'gmm_hmm' in self.config:
            self.logger.info("初始化GMM-HMM模型")
            gmm_config = self.config['gmm_hmm']
            self.models['gmm_hmm'] = GMMHMMModel(
                n_states=gmm_config.get('n_states', 5),
                n_mix=gmm_config.get('n_mix', 4),
                cov_type=gmm_config.get('cov_type', 'diag'),
                n_iter=gmm_config.get('n_iter', 20)
            )
        
        # 初始化DNN模型
        if 'dnn' in self.config:
            self.logger.info("初始化DNN模型")
            dnn_config = self.config['dnn']
            self.models['dnn'] = DNNModel(
                input_dim=dnn_config.get('input_dim', 39),
                hidden_dims=dnn_config.get('hidden_dims', [512, 512]),
                output_dim=dnn_config.get('output_dim', 48),
                dropout_prob=dnn_config.get('dropout_prob', 0.2)
            ).to(self.device)
            
            self.trainers['dnn'] = DNNTrainer(self.models['dnn'], self.device)
        
        # 初始化CNN模型
        if 'cnn' in self.config:
            self.logger.info("初始化CNN模型")
            cnn_config = self.config['cnn']
            self.models['cnn'] = CNNModel(
                input_channels=cnn_config.get('input_channels', 1),
                feature_dim=cnn_config.get('feature_dim', 39),
                num_classes=cnn_config.get('num_classes', 48),
                context_window=cnn_config.get('context_window', 11),
                cnn_channels=cnn_config.get('cnn_channels', [64, 128, 256]),
                kernel_sizes=cnn_config.get('kernel_sizes', [3, 3, 3]),
                fc_dims=cnn_config.get('fc_dims', [1024, 512])
            ).to(self.device)
            
            self.trainers['cnn'] = CNNTrainer(self.models['cnn'], self.device)
        
        # 初始化RNN模型
        if 'rnn' in self.config:
            self.logger.info("初始化RNN模型")
            rnn_config = self.config['rnn']
            self.models['rnn'] = RNNModel(
                input_dim=rnn_config.get('input_dim', 39),
                hidden_dim=rnn_config.get('hidden_dim', 256),
                num_layers=rnn_config.get('num_layers', 3),
                output_dim=rnn_config.get('output_dim', 48),
                bidirectional=rnn_config.get('bidirectional', True),
                rnn_type=rnn_config.get('rnn_type', 'lstm'),
                dropout=rnn_config.get('dropout', 0.2)
            ).to(self.device)
            
            self.trainers['rnn'] = RNNTrainer(self.models['rnn'], self.device)
        
        # 初始化Transformer模型
        if 'transformer' in self.config:
            self.logger.info("初始化Transformer模型")
            tf_config = self.config['transformer']
            self.models['transformer'] = TransformerModel(
                input_dim=tf_config.get('input_dim', 39),
                d_model=tf_config.get('d_model', 512),
                nhead=tf_config.get('nhead', 8),
                num_encoder_layers=tf_config.get('num_encoder_layers', 6),
                dim_feedforward=tf_config.get('dim_feedforward', 2048),
                output_dim=tf_config.get('output_dim', 48),
                dropout=tf_config.get('dropout', 0.1),
                max_len=tf_config.get('max_len', 1000)
            ).to(self.device)
            
            self.trainers['transformer'] = TransformerTrainer(self.models['transformer'], self.device)
        
        # 初始化CTC模型
        if 'ctc' in self.config:
            self.logger.info("初始化CTC模型")
            ctc_config = self.config['ctc']
            self.models['ctc'] = CTCRNN(
                input_dim=ctc_config.get('input_dim', 39),
                hidden_dim=ctc_config.get('hidden_dim', 256),
                output_dim=ctc_config.get('output_dim', 49),  # 包括blank标签
                num_layers=ctc_config.get('num_layers', 4),
                bidirectional=ctc_config.get('bidirectional', True),
                dropout=ctc_config.get('dropout', 0.2),
                rnn_type=ctc_config.get('rnn_type', 'lstm')
            ).to(self.device)
            
            self.trainers['ctc'] = CTCTrainer(self.models['ctc'], self.device)
    
    def _create_trainer(self, model_type):
        """
        根据模型类型创建适当的训练器
        
        Args:
            model_type (str): 模型类型
            
        Returns:
            trainer: 对应的训练器对象
        """
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return None
        
        if model_type == 'dnn':
            return DNNTrainer(self.models[model_type], self.device)
        elif model_type == 'cnn':
            return CNNTrainer(self.models[model_type], self.device)
        elif model_type == 'rnn':
            return RNNTrainer(self.models[model_type], self.device)
        elif model_type == 'transformer':
            return TransformerTrainer(self.models[model_type], self.device)
        elif model_type == 'ctc':
            return CTCTrainer(self.models[model_type], self.device)
        else:
            self.logger.error(f"未知的模型类型: {model_type}")
            return None
    
    def extract_features(self, audio_data, sample_rate=None):
        """
        从音频数据中提取特征
        
        Args:
            audio_data (numpy.ndarray): 音频数据
            sample_rate (int, optional): 采样率，如果不提供，则使用配置中的采样率
            
        Returns:
            dict: 包含不同特征的字典
        """
        if sample_rate is None:
            sample_rate = self.config.get('sample_rate', 16000)
        
        self.logger.info("提取特征...")
        return self.feature_extractor.extract_all_features(
            audio_data, 
            sample_rate,
            compute_deltas=self.config.get('deltas', True),
            apply_cmvn=self.config.get('cmvn', True)
        )
    
    def prepare_datasets(self, features_list, labels_list, model_type, 
                        train_ratio=0.7, val_ratio=0.15, batch_size=32):
        """
        准备数据集并创建数据加载器
        
        Args:
            features_list (list): 特征列表
            labels_list (list): 标签列表
            model_type (str): 模型类型
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            batch_size (int): 批量大小
            
        Returns:
            dict: 包含训练、验证和测试数据加载器的字典
        """
        self.logger.info(f"准备 {model_type} 模型的数据集")
        
        # 将所有特征和标签合并为单个数组
        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
        
        # 划分训练、验证和测试集
        test_ratio = 1.0 - train_ratio - val_ratio
        
        # 先划分训练集和临时集
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_features, all_labels, test_size=(val_ratio + test_ratio), random_state=42
        )
        
        # 然后从临时集中划分验证集和测试集
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
        )
        
        # 创建适当的数据集
        if model_type == 'dnn':
            train_dataset = DNNDataset(X_train, y_train)
            val_dataset = DNNDataset(X_val, y_val)
            test_dataset = DNNDataset(X_test, y_test)
        elif model_type == 'cnn':
            # 对于CNN，我们需要上下文窗口
            context_window = self.config['cnn'].get('context_window', 11)
            train_dataset = CNNDataset(X_train, y_train, context_window=context_window)
            val_dataset = CNNDataset(X_val, y_val, context_window=context_window)
            test_dataset = CNNDataset(X_test, y_test, context_window=context_window)
        elif model_type == 'rnn':
            # 对于RNN，我们将特征序列转换为批次
            # 这里简化处理，将每个样本视为独立序列
            train_dataset = RNNDataset(X_train, y_train)
            val_dataset = RNNDataset(X_val, y_val)
            test_dataset = RNNDataset(X_test, y_test)
        elif model_type == 'transformer':
            # 对于Transformer，我们需要处理序列
            max_len = self.config['transformer'].get('max_len', 1000)
            train_dataset = TransformerDataset(X_train, y_train, max_len=max_len)
            val_dataset = TransformerDataset(X_val, y_val, max_len=max_len)
            test_dataset = TransformerDataset(X_test, y_test, max_len=max_len)
        elif model_type == 'ctc':
            # 对于CTC，我们需要处理序列和长度
            train_dataset = CTCDataset(X_train, y_train)
            val_dataset = CTCDataset(X_val, y_val)
            test_dataset = CTCDataset(X_test, y_test)
        else:
            self.logger.error(f"未知的模型类型: {model_type}")
            return None
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def train_model(self, model_type, loaders, **kwargs):
        """
        训练指定类型的模型
        
        Args:
            model_type (str): 模型类型
            loaders (dict): 包含训练、验证和测试数据加载器的字典
            **kwargs: 其他训练参数
            
        Returns:
            dict: 训练历史记录
        """
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return None
        
        if model_type not in self.trainers:
            self.trainers[model_type] = self._create_trainer(model_type)
        
        trainer = self.trainers[model_type]
        if trainer is None:
            self.logger.error(f"无法为模型 {model_type} 创建训练器")
            return None
        
        self.logger.info(f"开始训练 {model_type} 模型")
        
        # 设置训练参数
        epochs = kwargs.get('epochs', 10)
        learning_rate = kwargs.get('learning_rate', 0.001)
        weight_decay = kwargs.get('weight_decay', 1e-5)
        patience = kwargs.get('patience', 5)
        save_dir = kwargs.get('save_dir', f'models/{model_type}')
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练模型
        history = trainer.train(
            loaders['train'], 
            loaders['val'],
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            model_save_path=os.path.join(save_dir, f'{model_type}_model.pt')
        )
        
        # 绘制训练历史
        trainer.plot_history(history)
        plt.savefig(os.path.join(save_dir, f'{model_type}_training_history.png'))
        
        return history
    
    def evaluate_model(self, model_type, test_loader):
        """
        评估指定类型的模型
        
        Args:
            model_type (str): 模型类型
            test_loader: 测试数据加载器
            
        Returns:
            dict: 评估结果
        """
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return None
        
        if model_type not in self.trainers:
            self.trainers[model_type] = self._create_trainer(model_type)
        
        trainer = self.trainers[model_type]
        if trainer is None:
            self.logger.error(f"无法为模型 {model_type} 创建训练器")
            return None
        
        self.logger.info(f"评估 {model_type} 模型")
        
        # 评估模型
        metrics = trainer.evaluate(test_loader)
        
        # 显示评估结果
        self.logger.info(f"{model_type} 模型评估结果:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value}")
        
        return metrics
    
    def predict(self, model_type, features):
        """
        使用指定类型的模型进行预测
        
        Args:
            model_type (str): 模型类型
            features (numpy.ndarray): 特征数据
            
        Returns:
            numpy.ndarray: 预测结果
        """
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return None
        
        if model_type not in self.trainers:
            self.trainers[model_type] = self._create_trainer(model_type)
        
        trainer = self.trainers[model_type]
        if trainer is None:
            self.logger.error(f"无法为模型 {model_type} 创建训练器")
            return None
        
        self.logger.info(f"使用 {model_type} 模型进行预测")
        
        # 进行预测
        predictions = trainer.predict(features)
        
        return predictions
    
    def train_gmm_hmm(self, features_list, phoneme_ids):
        """
        训练GMM-HMM模型
        
        Args:
            features_list (list): 特征列表，每个音素对应一个特征数组
            phoneme_ids (list): 音素ID列表
            
        Returns:
            GMMHMMModel: 训练后的GMM-HMM模型
        """
        if 'gmm_hmm' not in self.models:
            self.logger.error("GMM-HMM模型未初始化")
            return None
        
        gmm_hmm = self.models['gmm_hmm']
        
        self.logger.info("训练GMM-HMM模型")
        gmm_hmm.train(features_list, phoneme_ids)
        
        return gmm_hmm
    
    def gmm_hmm_decode(self, features, phoneme_ids=None):
        """
        使用GMM-HMM模型解码特征序列
        
        Args:
            features (numpy.ndarray): 特征序列
            phoneme_ids (list, optional): 音素ID列表，如果不提供，则使用模型中的所有音素
            
        Returns:
            tuple: (最可能的音素ID, 对数似然)
        """
        if 'gmm_hmm' not in self.models:
            self.logger.error("GMM-HMM模型未初始化")
            return None, None
        
        gmm_hmm = self.models['gmm_hmm']
        
        self.logger.info("使用GMM-HMM模型解码")
        best_phoneme, log_likelihood = gmm_hmm.decode(features, phoneme_ids)
        
        return best_phoneme, log_likelihood
    
    def save_model(self, model_type, filepath):
        """
        保存指定类型的模型
        
        Args:
            model_type (str): 模型类型
            filepath (str): 保存路径
            
        Returns:
            bool: 保存是否成功
        """
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return False
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if model_type == 'gmm_hmm':
            self.models[model_type].save(filepath)
            return True
        else:
            torch.save(self.models[model_type].state_dict(), filepath)
            return True
    
    def load_model(self, model_type, filepath):
        """
        加载指定类型的模型
        
        Args:
            model_type (str): 模型类型
            filepath (str): 加载路径
            
        Returns:
            bool: 加载是否成功
        """
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return False
        
        if not os.path.exists(filepath):
            self.logger.error(f"模型文件 {filepath} 不存在")
            return False
        
        if model_type == 'gmm_hmm':
            self.models[model_type].load(filepath)
            return True
        else:
            self.models[model_type].load_state_dict(torch.load(filepath, map_location=self.device))
            self.models[model_type].eval()
            return True
    
    def ensemble_predict(self, features, model_types=['gmm_hmm', 'dnn', 'cnn', 'rnn', 'transformer', 'ctc']):
        """
        使用集成方法进行预测
        
        Args:
            features (numpy.ndarray): 特征数据
            model_types (list): 要使用的模型类型列表
            
        Returns:
            dict: 包含预测结果的字典
        """
        results = {}
        
        # 过滤掉未初始化的模型
        available_models = [m for m in model_types if m in self.models]
        
        if not available_models:
            self.logger.error("没有可用的模型进行预测")
            return {'prediction': None}
        
        self.logger.info(f"使用模型进行集成预测: {', '.join(available_models)}")
        
        all_predictions = []
        
        # 对每个模型进行预测
        for model_type in available_models:
            if model_type == 'gmm_hmm':
                # GMM-HMM预测处理单独进行
                continue
            
            # 使用模型进行预测
            pred = self.predict(model_type, features)
            
            if pred is not None:
                all_predictions.append(pred)
                results[model_type] = pred
        
        # 如果有多个神经网络模型预测结果，进行集成
        if len(all_predictions) > 1:
            # 将所有预测转换为概率分布
            prob_predictions = []
            
            for pred in all_predictions:
                if len(pred.shape) == 1:
                    # 对于非序列预测，转换为one-hot
                    one_hot = np.zeros((pred.shape[0], pred.max() + 1))
                    one_hot[np.arange(pred.shape[0]), pred] = 1
                    prob_predictions.append(one_hot)
                else:
                    # 对于序列预测，已经是概率分布
                    prob_predictions.append(pred)
            
            # 计算平均概率
            if prob_predictions:
                avg_prob = np.mean(prob_predictions, axis=0)
                # 取最高概率的类别作为最终预测
                ensemble_pred = np.argmax(avg_prob, axis=-1)
                results['ensemble'] = ensemble_pred
                results['prediction'] = ensemble_pred
            else:
                # 如果没有有效的预测，使用第一个模型的预测
                if all_predictions:
                    results['prediction'] = all_predictions[0]
        else:
            # 如果只有一个模型预测，直接使用该结果
            if all_predictions:
                results['prediction'] = all_predictions[0]
        
        return results
    
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