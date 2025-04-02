"""
Transformer声学模型
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import logging


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码
        
        参数:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # 注册缓冲区（不是模型参数）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, seq_len, d_model)
            
        返回:
            输出张量，形状为(batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDataset(Dataset):
    """Transformer模型的数据集类"""
    
    def __init__(self, features, labels, max_len=None):
        """
        初始化数据集
        
        参数:
            features: 特征列表，每个元素是一个形状为(n_frames, n_features)的NumPy数组
            labels: 标签列表，每个元素是一个形状为(n_frames,)的NumPy数组
            max_len: 最大序列长度，如果为None则使用原始序列长度
        """
        self.features = []
        self.labels = []
        self.lengths = []
        self.max_len = max_len
        
        # 如果没有指定最大长度，找出最长的序列
        if max_len is None:
            self.max_len = max([len(f) for f in features])
        
        # 处理每个样本，进行填充或裁剪
        for i in range(len(features)):
            feature = features[i]
            label = labels[i]
            
            # 记录原始长度
            self.lengths.append(len(feature))
            
            # 裁剪或填充
            if len(feature) > self.max_len:
                # 裁剪
                feature = feature[:self.max_len]
                label = label[:self.max_len]
            elif len(feature) < self.max_len:
                # 填充
                pad_len = self.max_len - len(feature)
                feature_pad = np.zeros((pad_len, feature.shape[1]))
                label_pad = np.full(pad_len, -100)  # -100是忽略标签
                feature = np.vstack([feature, feature_pad])
                label = np.concatenate([label, label_pad])
            
            self.features.append(feature)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor(self.labels[idx])
        length = self.lengths[idx]
        
        # 创建掩码，1表示填充部分
        mask = torch.zeros(self.max_len)
        if length < self.max_len:
            mask[length:] = 1
        
        return feature, label, mask


class TransformerModel(nn.Module):
    """Transformer声学模型"""
    
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, 
                 output_dim, dropout=0.1, max_len=5000):
        """
        初始化Transformer模型
        
        参数:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 多头注意力中的头数
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            output_dim: 输出维度（音素/状态数）
            dropout: Dropout概率
            max_len: 最大序列长度
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.dropout = dropout
        self.max_len = max_len
        
        # 添加日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 输入映射层
        self.input_mapping = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 输出映射层
        self.output_mapping = nn.Linear(d_model, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, src_key_padding_mask=None):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, seq_len, input_dim)
            src_key_padding_mask: 源序列填充掩码，形状为(batch_size, seq_len)，
                                  True表示填充位置
            
        返回:
            输出，形状为(batch_size, seq_len, output_dim)
        """
        # 检查输入维度，确保与模型期望的维度匹配
        if len(x.shape) == 2:  # 如果输入是(batch_size, input_dim)
            x = x.unsqueeze(1)  # 变成(batch_size, 1, input_dim)
        
        # 确保d_model的值匹配
        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            self.logger.warning(f"输入特征维度不匹配: 预期{self.input_dim}，实际{feature_dim}")
            # 如果维度不匹配，可以通过线性层或其他方式进行调整
            x = x.reshape(batch_size, seq_len, -1)
            # 如果特征维度太大，截断；如果太小，填充
            if feature_dim > self.input_dim:
                x = x[:, :, :self.input_dim]
            elif feature_dim < self.input_dim:
                padding = torch.zeros(batch_size, seq_len, self.input_dim - feature_dim, device=x.device)
                x = torch.cat([x, padding], dim=2)
        
        # 输入映射
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, d_model)
        src = self.input_mapping(x)
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码器
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # 输出映射
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, output_dim)
        output = self.output_mapping(output)
        
        return output


class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        参数:
            model: Transformer模型实例
            device: 训练设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, train_loader, val_loader=None, epochs=50, learning_rate=0.0001, 
              weight_decay=1e-5, patience=5, warmup_steps=4000, clip_grad=1.0, save_dir="models"):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            warmup_steps: 预热步数
            clip_grad: 梯度裁剪值
            save_dir: 模型保存目录
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in pbar:
                # 检查batch是返回2个值还是3个值
                if len(batch) == 3:
                    features, labels, mask = batch
                    # 创建填充掩码
                    padding_mask = mask.bool()  # True表示填充位置
                else:
                    features, labels = batch
                    # 不使用掩码，创建一个全False的掩码
                    padding_mask = torch.zeros(features.size(0), features.size(1) if len(features.shape) > 2 else 1, dtype=torch.bool, device=features.device)
                
                features, labels = features.to(self.device), labels.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 检查和调整padding_mask的形状
                if padding_mask.shape[1] != 1 and len(features.shape) > 2 and features.shape[1] != padding_mask.shape[1]:
                    # 调整padding_mask大小与features的序列长度匹配
                    if padding_mask.shape[1] > features.shape[1]:
                        padding_mask = padding_mask[:, :features.shape[1]]
                    else:
                        # 扩展padding_mask
                        new_padding = torch.zeros(padding_mask.shape[0], features.shape[1] - padding_mask.shape[1], 
                                                 dtype=torch.bool, device=padding_mask.device)
                        padding_mask = torch.cat([padding_mask, new_padding], dim=1)
                
                outputs = self.model(features, padding_mask)
                
                # 重塑输出和标签以计算损失
                # outputs形状: (batch_size, seq_len, output_dim)
                # 转换为: (batch_size * seq_len, output_dim)
                batch_size, seq_len, output_dim = outputs.size()
                outputs = outputs.reshape(-1, output_dim)
                labels = labels.reshape(-1)
                
                # 使用交叉熵损失
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                
                self.optimizer.step()
                
                train_loss += loss.item() * batch_size
                
                # 计算准确率（忽略填充标记）
                _, predicted = torch.max(outputs, 1)
                valid_indices = (labels != -100)
                train_total += valid_indices.sum().item()
                train_correct += ((predicted == labels) & valid_indices).sum().item()
                
                # 更新进度条
                pbar.set_postfix({"loss": loss.item(), "acc": train_correct/(train_total+1e-8)})
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / (train_total + 1e-8)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证阶段
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self._save_model(os.path.join(save_dir, "best_model.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停: 验证损失在{patience}个epoch内没有改善")
                        break
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                
                # 每10个epoch保存一次
                if (epoch + 1) % 10 == 0:
                    self._save_model(os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))
        
        # 保存最终模型
        self._save_model(os.path.join(save_dir, "final_model.pth"))
        
        # 绘制训练历史
        self._plot_history()
        
        return self.history
    
    def _validate(self, val_loader):
        """
        验证模型
        
        参数:
            val_loader: 验证数据加载器
            
        返回:
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # 检查batch是返回2个值还是3个值
                if len(batch) == 3:
                    features, labels, mask = batch
                    # 创建填充掩码
                    padding_mask = mask.bool()  # True表示填充位置
                else:
                    features, labels = batch
                    # 不使用掩码，创建一个全False的掩码
                    padding_mask = torch.zeros(features.size(0), features.size(1) if len(features.shape) > 2 else 1, dtype=torch.bool, device=features.device)
                
                features, labels = features.to(self.device), labels.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                # 检查和调整padding_mask的形状
                if padding_mask.shape[1] != 1 and len(features.shape) > 2 and features.shape[1] != padding_mask.shape[1]:
                    # 调整padding_mask大小与features的序列长度匹配
                    if padding_mask.shape[1] > features.shape[1]:
                        padding_mask = padding_mask[:, :features.shape[1]]
                    else:
                        # 扩展padding_mask
                        new_padding = torch.zeros(padding_mask.shape[0], features.shape[1] - padding_mask.shape[1], 
                                                 dtype=torch.bool, device=padding_mask.device)
                        padding_mask = torch.cat([padding_mask, new_padding], dim=1)
                
                # 前向传播
                outputs = self.model(features, padding_mask)
                
                # 重塑输出和标签以计算损失
                batch_size, seq_len, output_dim = outputs.size()
                outputs = outputs.reshape(-1, output_dim)
                labels = labels.reshape(-1)
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * batch_size
                
                # 计算准确率（忽略填充标记）
                _, predicted = torch.max(outputs, 1)
                valid_indices = (labels != -100)
                val_total += valid_indices.sum().item()
                val_correct += ((predicted == labels) & valid_indices).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / (val_total + 1e-8)
        
        return val_loss, val_acc
    
    def predict(self, loader):
        """
        使用模型进行预测
        
        参数:
            loader: 数据加载器
            
        返回:
            predictions: 预测结果，列表，每个元素是一个形状为(seq_len,)的NumPy数组
            probs: 预测概率，列表，每个元素是一个形状为(seq_len, num_classes)的NumPy数组
        """
        self.model.eval()
        predictions = []
        probs = []
        
        with torch.no_grad():
            for features, _, mask in tqdm(loader, desc="Predicting"):
                features, mask = features.to(self.device), mask.to(self.device)
                
                # 前向传播
                outputs = self.model(features, src_key_padding_mask=mask.bool())
                
                # 计算每帧的预测概率和类别
                batch_probs = torch.softmax(outputs, dim=2).cpu().numpy()
                batch_preds = np.argmax(batch_probs, axis=2)
                
                # 获取每个样本的长度（非填充部分）
                lengths = (~mask.bool()).sum(dim=1).cpu().numpy()
                
                # 将批次结果添加到列表中
                for i in range(features.size(0)):
                    # 只保留非填充部分
                    length = lengths[i]
                    predictions.append(batch_preds[i, :length])
                    probs.append(batch_probs[i, :length])
        
        return predictions, probs
    
    def _save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'd_model': self.model.d_model,
            'nhead': self.model.nhead,
            'num_encoder_layers': self.model.num_encoder_layers,
            'dim_feedforward': self.model.dim_feedforward,
            'output_dim': self.model.output_dim,
            'dropout': self.model.dropout,
            'max_len': self.model.max_len
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        if self.device == "cpu":
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        return self.model
    
    def _plot_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        if 'val_acc' in self.history and self.history['val_acc']:
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('transformer_training_history.png')
        plt.close() 