"""
卷积神经网络声学模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CNNDataset(Dataset):
    """CNN模型的数据集类"""
    
    def __init__(self, features, labels, context_window=11):
        """
        初始化数据集
        
        参数:
            features: 特征列表，每个元素是一个形状为(n_frames, n_features)的NumPy数组
            labels: 标签列表，每个元素是一个形状为(n_frames,)的NumPy数组
            context_window: 上下文窗口大小（必须是奇数）
        """
        assert context_window % 2 == 1, "上下文窗口大小必须是奇数"
        
        self.context_window = context_window
        self.half_window = context_window // 2
        
        self.features = []
        self.labels = []
        
        # 处理每个样本，添加上下文窗口
        for i in range(len(features)):
            feature = features[i]
            label = labels[i]
            
            # 通过在特征的开头和结尾填充零来添加上下文
            padded_feature = np.pad(
                feature, 
                ((self.half_window, self.half_window), (0, 0)),
                mode='constant'
            )
            
            # 为每一帧创建一个包含上下文的特征矩阵
            for j in range(len(label)):
                # 提取当前帧的上下文窗口
                context_feature = padded_feature[j:j+self.context_window]
                self.features.append(context_feature)
                self.labels.append(label[j])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 返回形状为(context_window, feature_dim)的特征
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return feature, label


class CNNModel(nn.Module):
    """卷积神经网络声学模型"""
    
    def __init__(self, input_channels, feature_dim, num_classes, 
                 context_window=11, cnn_channels=[64, 128, 256], 
                 kernel_sizes=[3, 3, 3], fc_dims=[1024, 512]):
        """
        初始化CNN模型
        
        参数:
            input_channels: 输入通道数（通常为1）
            feature_dim: 特征维度
            num_classes: 类别数（音素/状态数）
            context_window: 上下文窗口大小
            cnn_channels: CNN层通道数列表
            kernel_sizes: 卷积核大小列表
            fc_dims: 全连接层维度列表
        """
        super(CNNModel, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.context_window = context_window
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.fc_dims = fc_dims
        
        # CNN层
        cnn_layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2
                )
            )
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.cnn_layers = nn.Sequential(*cnn_layers)
        
        # 计算CNN层输出大小
        # 假设每次MaxPool后大小减半
        out_height = context_window
        out_width = feature_dim
        for _ in range(len(cnn_channels)):
            out_height = out_height // 2
            out_width = out_width // 2
        
        # 为了防止出现大小为0的情况
        out_height = max(1, out_height)
        out_width = max(1, out_width)
        
        self.fc_input_dim = out_height * out_width * cnn_channels[-1]
        
        # 全连接层
        fc_layers = []
        in_dim = self.fc_input_dim
        
        for out_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.BatchNorm1d(out_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2))
            in_dim = out_dim
        
        # 输出层
        fc_layers.append(nn.Linear(in_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, context_window, feature_dim)
               或(batch_size, feature_dim)
            
        返回:
            输出，形状为(batch_size, num_classes)
        """
        # 检查输入维度
        if len(x.shape) == 2:
            # 如果输入是(batch_size, feature_dim)，重塑为(batch_size, context_window, feature_dim)
            batch_size, feature_dim = x.shape
            # 使用单帧作为所有上下文
            x = x.unsqueeze(1).repeat(1, self.context_window, 1)
        
        # 确保特征维度正确
        batch_size, ctx_window, feat_dim = x.shape
        if feat_dim != self.feature_dim:
            # 处理特征维度不匹配
            if feat_dim > self.feature_dim:
                # 如果特征维度过大，截断
                x = x[:, :, :self.feature_dim]
            else:
                # 如果特征维度过小，填充
                padding = torch.zeros(batch_size, ctx_window, self.feature_dim - feat_dim, device=x.device)
                x = torch.cat([x, padding], dim=2)
        
        # 将输入重塑为四维张量:(batch_size, channels, height, width)
        # 其中height=context_window, width=feature_dim
        x = x.unsqueeze(1)  # 添加通道维度
        
        # 通过CNN层
        x = self.cnn_layers(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 通过全连接层
        x = self.fc_layers(x)
        
        return x


class CNNTrainer:
    """CNN模型训练器"""
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        参数:
            model: CNN模型实例
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
                
                # 计算损失 - 保护性地处理标签维度
                if labels.dim() > 1 and labels.size(1) == 1:
                    # 如果是 [batch_size, 1] 则压缩
                    loss = self.criterion(outputs, labels.squeeze(1))
                else:
                    # 已经是 [batch_size] 或其他情况
                    loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * features.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                
                # 处理标签维度
                if labels.dim() > 1:
                    if labels.size(1) == 1:
                        # 如果标签是[batch_size, 1]，压缩成[batch_size]
                        labels_for_acc = labels.squeeze(1)
                    else:
                        labels_for_acc = labels
                else:
                    # 已经是[batch_size]
                    labels_for_acc = labels
                
                train_total += labels.size(0)
                train_correct += (predicted == labels_for_acc).sum().item()
                
                # 更新进度条
                pbar.set_postfix({"loss": loss.item(), "acc": train_correct/(train_total+1e-8)})
            
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
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                
                # 处理标签维度
                if labels.dim() > 1:
                    if labels.size(1) == 1:
                        # 如果标签是[batch_size, 1]，压缩成[batch_size]
                        labels_for_acc = labels.squeeze(1)
                    else:
                        labels_for_acc = labels
                else:
                    # 已经是[batch_size]
                    labels_for_acc = labels
                
                val_total += labels.size(0)
                val_correct += (predicted == labels_for_acc).sum().item()
        
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
                prob = torch.softmax(outputs, dim=1).cpu().numpy()
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                
                probs.append(prob)
                predictions.append(pred)
        
        if len(probs) > 0:
            probs = np.concatenate(probs)
            predictions = np.concatenate(predictions)
        
        return predictions, probs
    
    def _save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_channels': self.model.input_channels,
            'feature_dim': self.model.feature_dim,
            'num_classes': self.model.num_classes,
            'context_window': self.model.context_window,
            'cnn_channels': self.model.cnn_channels,
            'kernel_sizes': self.model.kernel_sizes,
            'fc_dims': self.model.fc_dims
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
        plt.savefig('cnn_training_history.png')
        plt.close() 