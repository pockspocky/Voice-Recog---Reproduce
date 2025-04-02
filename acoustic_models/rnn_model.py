"""
循环神经网络声学模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class RNNDataset(Dataset):
    """RNN模型的数据集类"""
    
    def __init__(self, features, labels, seq_len=None):
        """
        初始化数据集
        
        参数:
            features: 特征列表，每个元素是一个形状为(n_frames, n_features)的NumPy数组
            labels: 标签列表，每个元素是一个形状为(n_frames,)的NumPy数组
            seq_len: 序列长度，如果为None则使用原始序列长度
        """
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # 如果指定了序列长度，则截断或填充序列
        if self.seq_len is not None:
            if len(feature) > self.seq_len:
                # 截断序列
                feature = feature[:self.seq_len]
                label = label[:self.seq_len]
            elif len(feature) < self.seq_len:
                # 填充序列
                pad_len = self.seq_len - len(feature)
                feature_pad = np.zeros((pad_len, feature.shape[1]))
                label_pad = np.zeros(pad_len)
                feature = np.vstack([feature, feature_pad])
                label = np.concatenate([label, label_pad])
        
        # 转换为PyTorch张量
        feature_tensor = torch.FloatTensor(feature)
        label_tensor = torch.LongTensor(label)
        
        return feature_tensor, label_tensor


class RNNModel(nn.Module):
    """循环神经网络声学模型"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=True, 
                 rnn_type="lstm", dropout=0.2):
        """
        初始化RNN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: RNN层数
            output_dim: 输出维度（音素/状态数）
            bidirectional: 是否使用双向RNN
            rnn_type: RNN类型，"lstm"或"gru"
            dropout: Dropout概率
        """
        super(RNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        
        # 方向数（单向或双向）
        self.num_directions = 2 if bidirectional else 1
        
        # 选择RNN类型
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                bidirectional=bidirectional, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                bidirectional=bidirectional, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}，必须是'lstm'或'gru'")
        
        # 全连接层将RNN输出转换为音素分类
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x, hidden=None):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, seq_len, input_dim)
            hidden: 初始隐藏状态，默认为None
            
        返回:
            输出，形状为(batch_size, seq_len, output_dim)
        """
        # RNN前向传播
        # x形状: (batch_size, seq_len, input_dim)
        # output形状: (batch_size, seq_len, hidden_dim * num_directions)
        if hidden is None:
            output, _ = self.rnn(x)
        else:
            output, _ = self.rnn(x, hidden)
        
        # 应用dropout
        output = self.dropout_layer(output)
        
        # 全连接层
        # 将RNN输出形状从(batch_size, seq_len, hidden_dim * num_directions)
        # 转换为(batch_size, seq_len, output_dim)
        output = self.fc(output)
        
        return output
    
    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态
        
        参数:
            batch_size: 批大小
            device: 设备
            
        返回:
            初始隐藏状态
        """
        if self.rnn_type.lower() == "lstm":
            # LSTM需要隐藏状态和细胞状态
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device)
            return (h0, c0)
        else:
            # GRU只需要隐藏状态
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device)


class RNNTrainer:
    """RNN模型训练器"""
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        参数:
            model: RNN模型实例
            device: 训练设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, train_loader, val_loader=None, epochs=50, learning_rate=0.001, 
              weight_decay=1e-5, patience=5, clip_grad=5.0, save_dir="models"):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
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
            for features, labels in pbar:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # 创建掩码以忽略填充标记
                mask = (labels != -100)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(features)
                
                # 重塑输出和标签以计算损失
                # outputs形状: (batch_size, seq_len, output_dim)
                # 转换为: (batch_size * seq_len, output_dim)
                batch_size, seq_len, output_dim = outputs.size()
                outputs = outputs.reshape(-1, output_dim)
                labels = labels.reshape(-1)
                
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
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
                
                # 学习率调整
                self.scheduler.step(val_loss)
                
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
            for features, labels in tqdm(val_loader, desc="Validating"):
                features, labels = features.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                
                # 重塑输出和标签以计算损失
                batch_size, seq_len, output_dim = outputs.size()
                outputs = outputs.reshape(-1, output_dim)
                labels = labels.reshape(-1)
                
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
            for features, _ in tqdm(loader, desc="Predicting"):
                features = features.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                
                # 计算每帧的预测概率和类别
                batch_probs = torch.softmax(outputs, dim=2).cpu().numpy()
                batch_preds = np.argmax(batch_probs, axis=2)
                
                # 将批次结果添加到列表中
                for i in range(features.size(0)):
                    predictions.append(batch_preds[i])
                    probs.append(batch_probs[i])
        
        return predictions, probs
    
    def _save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'output_dim': self.model.output_dim,
            'bidirectional': self.model.bidirectional,
            'rnn_type': self.model.rnn_type,
            'dropout': self.model.dropout
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
        plt.savefig('rnn_training_history.png')
        plt.close() 