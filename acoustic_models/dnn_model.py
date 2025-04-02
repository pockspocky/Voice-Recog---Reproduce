"""
深度神经网络声学模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class DNNDataset(Dataset):
    """DNN模型的数据集类"""
    
    def __init__(self, features, labels):
        """
        初始化数据集
        
        参数:
            features: 特征列表，每个元素是一个形状为(n_frames, n_features)的NumPy数组
            labels: 标签列表，每个元素是一个形状为(n_frames, n_classes)的NumPy数组
        """
        self.features = []
        self.labels = []
        
        # 将所有特征和标签展平
        for i in range(len(features)):
            self.features.append(features[i])
            self.labels.append(labels[i])
            
        self.features = np.vstack(self.features)
        self.labels = np.vstack(self.labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor(self.labels[idx])
        return feature, label


class DNNModel(nn.Module):
    """深度神经网络声学模型"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.2):
        """
        初始化DNN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（音素/状态数）
            dropout_prob: Dropout概率
        """
        super(DNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        
        # 构建DNN层
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.Dropout(dropout_prob))
        
        # 输出层
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)


class DNNTrainer:
    """DNN模型训练器"""
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        参数:
            model: DNN模型实例
            device: 训练设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, train_loader, val_loader=None, epochs=50, learning_rate=0.001, 
              weight_decay=1e-5, patience=5, save_dir="models"):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
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
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * features.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({"loss": loss.item(), "acc": train_correct/train_total})
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
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
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        return val_loss, val_acc
    
    def predict(self, loader):
        """
        使用模型进行预测
        
        参数:
            loader: 数据加载器
            
        返回:
            predictions: 预测结果
            probs: 预测概率
        """
        self.model.eval()
        predictions = []
        probs = []
        
        with torch.no_grad():
            for features, _ in tqdm(loader, desc="Predicting"):
                features = features.to(self.device)
                
                outputs = self.model(features)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
        
        predictions = np.concatenate(predictions)
        probs = np.concatenate(probs)
        
        return predictions, probs
    
    def _save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dims': self.model.hidden_dims,
            'output_dim': self.model.output_dim,
            'dropout_prob': self.model.dropout_prob
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
        plt.savefig('training_history.png')
        plt.close() 