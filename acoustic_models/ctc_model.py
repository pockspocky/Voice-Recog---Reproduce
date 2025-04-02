"""
CTC（Connectionist Temporal Classification）声学模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class CTCDataset(Dataset):
    """CTC模型的数据集类"""
    
    def __init__(self, features, targets, feature_lens=None, target_lens=None):
        """
        初始化数据集
        
        参数:
            features: 特征列表，每个元素是一个形状为(time_steps, feature_dim)的NumPy数组
            targets: 目标列表，每个元素是一个形状为(target_len,)的NumPy数组
            feature_lens: 特征长度列表，如果为None则使用实际长度
            target_lens: 目标长度列表，如果为None则使用实际长度
        """
        self.features = features
        self.targets = targets
        
        # 计算特征长度
        if feature_lens is None:
            self.feature_lens = [len(f) for f in features]
        else:
            self.feature_lens = feature_lens
        
        # 计算目标长度
        if target_lens is None:
            # 检查目标是否为标量值数组
            if hasattr(targets, 'ndim') and targets.ndim == 1:
                # 如果targets是一维数组，则每个目标的长度为1
                self.target_lens = [1] * len(targets)
                # 将每个标量目标转换为长度为1的数组
                self.targets = [[t] for t in targets]
            else:
                # 正常情况，每个目标是一个序列
                self.target_lens = [len(t) for t in targets]
        else:
            self.target_lens = target_lens
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        target = torch.LongTensor(self.targets[idx])
        feature_len = self.feature_lens[idx]
        target_len = self.target_lens[idx]
        
        return feature, target, feature_len, target_len


def collate_fn(batch):
    """
    数据批处理函数
    
    参数:
        batch: 批次数据，每个元素是(feature, target, feature_len, target_len)
    
    返回:
        features_padded: 填充后的特征张量，形状为(batch_size, max_time, feature_dim)
        targets_padded: 填充后的目标张量，形状为(batch_size, max_target_len)
        feature_lens: 特征长度列表
        target_lens: 目标长度列表
    """
    # 排序批次（按特征长度降序）
    batch.sort(key=lambda x: x[2], reverse=True)
    
    # 解包批次
    features, targets, feature_lens, target_lens = zip(*batch)
    
    # 获取最大长度
    max_feature_len = max(feature_lens)
    max_target_len = max(target_lens)
    
    # 获取特征维度
    if len(features[0].shape) > 1:
        feature_dim = features[0].size(1)
    else:
        # 如果特征是一维的，将维度设为1
        feature_dim = 1
    
    # 初始化填充张量
    features_padded = torch.zeros(len(features), max_feature_len, feature_dim)
    targets_padded = torch.ones(len(targets), max_target_len, dtype=torch.long) * -1  # -1表示填充
    
    # 填充
    for i, (feature, target) in enumerate(zip(features, targets)):
        # 处理特征的形状
        if len(feature.shape) == 1:
            # 如果是一维特征，添加一个维度
            feature = feature.unsqueeze(1)
        
        features_padded[i, :feature_lens[i]] = feature
        targets_padded[i, :target_lens[i]] = target
    
    return features_padded, targets_padded, feature_lens, target_lens


class CTCRNN(nn.Module):
    """基于RNN的CTC声学模型"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 bidirectional=True, dropout=0.2, rnn_type="lstm"):
        """
        初始化CTC-RNN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（音素/字符数 + 1，加1是为了blank标记）
            num_layers: RNN层数
            bidirectional: 是否使用双向RNN
            dropout: Dropout概率
            rnn_type: RNN类型，"lstm"或"gru"
        """
        super(CTCRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn_type = rnn_type
        
        # 方向数（单向或双向）
        self.num_directions = 2 if bidirectional else 1
        
        # 选择RNN类型
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                bidirectional=bidirectional, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                bidirectional=bidirectional, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}，必须是'lstm'或'gru'")
        
        # 全连接层将RNN输出转换为音素/字符分类
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        
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
    
    def forward(self, x, x_lens):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, seq_len, input_dim)
            x_lens: 输入序列长度，形状为(batch_size,)
            
        返回:
            logits: 输出对数概率，形状为(batch_size, seq_len, output_dim)
            output_lens: 输出序列长度，形状为(batch_size,)
        """
        # 检查和修正输入维度
        if x.size(2) != self.input_dim:
            # 处理输入维度不匹配
            batch_size, seq_len, feature_dim = x.size()
            if feature_dim > self.input_dim:
                # 如果特征维度太大，截断
                x = x[:, :, :self.input_dim]
            else:
                # 如果特征维度太小，填充
                padding = torch.zeros(batch_size, seq_len, self.input_dim - feature_dim, device=x.device)
                x = torch.cat([x, padding], dim=2)
        
        # 打包序列以忽略填充
        packed_x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        
        # RNN前向传播
        packed_output, _ = self.rnn(packed_x)
        
        # 解包序列
        output, output_lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 全连接层
        logits = self.fc(output)
        
        # 对数softmax
        log_probs = nn.functional.log_softmax(logits, dim=2)
        
        return log_probs, output_lens


class CTCTrainer:
    """CTC模型训练器"""
    
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        参数:
            model: CTC模型实例
            device: 训练设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, train_loader, val_loader=None, epochs=30, learning_rate=0.0001, weight_decay=1e-5, 
              patience=5, clip_grad=3.0, save_dir="models/ctc"):
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
        self.optimizer = optim.RMSprop(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            momentum=0.9
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=2, verbose=True
        )
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for features, targets, feature_lens, target_lens in pbar:
                features, targets = features.to(self.device), targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                log_probs, output_lens = self.model(features, feature_lens)
                
                # 计算CTC损失
                loss = self.criterion(
                    log_probs.permute(1, 0, 2),  # 转换为(time, batch, classes)
                    targets,
                    torch.tensor(output_lens, dtype=torch.int32, device=self.device),
                    torch.tensor(target_lens, dtype=torch.int32, device=self.device)
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({"loss": loss.item()})
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
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
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}")
                
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
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, targets, feature_lens, target_lens in tqdm(val_loader, desc="Validating"):
                features, targets = features.to(self.device), targets.to(self.device)
                
                # 前向传播
                log_probs, output_lens = self.model(features, feature_lens)
                
                # 计算CTC损失
                loss = self.criterion(
                    log_probs.permute(1, 0, 2),  # 转换为(time, batch, classes)
                    targets,
                    torch.tensor(output_lens, dtype=torch.int32, device=self.device),
                    torch.tensor(target_lens, dtype=torch.int32, device=self.device)
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        return val_loss
    
    def decode(self, log_probs, output_lens, blank=0, beam_width=5):
        """
        使用贪婪解码或束搜索解码CTC输出
        
        参数:
            log_probs: 对数概率，形状为(batch_size, seq_len, output_dim)
            output_lens: 输出序列长度，形状为(batch_size,)
            blank: 空白标记的索引
            beam_width: 束宽度，如果为1则使用贪婪解码
            
        返回:
            decoded: 解码结果列表
        """
        import ctcdecode
        
        batch_size = log_probs.size(0)
        decoded = []
        
        # 将log_probs转换为CPU NumPy数组
        log_probs_np = log_probs.cpu().numpy()
        
        if beam_width == 1:
            # 贪婪解码
            greedy_result = log_probs.argmax(dim=2).cpu().numpy()
            
            for i in range(batch_size):
                # 获取非填充部分
                length = output_lens[i]
                prediction = greedy_result[i, :length]
                
                # 合并重复标签并移除空白标签
                previous = blank
                result = []
                for p in prediction:
                    if p != previous and p != blank:
                        result.append(p)
                    previous = p
                
                decoded.append(result)
        else:
            # 束搜索解码
            # 创建解码器
            labels = [str(i) for i in range(1, log_probs.size(2))]  # 跳过blank标记
            decoder = ctcdecode.CTCBeamDecoder(
                labels,
                blank_id=blank,
                beam_width=beam_width,
                log_probs_input=True
            )
            
            # 解码
            beam_result, beam_scores, timesteps, out_lens = decoder.decode(log_probs.cpu())
            
            # 处理结果
            for i in range(batch_size):
                result = beam_result[i, 0, :out_lens[i, 0]].tolist()
                decoded.append(result)
        
        return decoded
    
    def predict(self, loader, beam_width=5):
        """
        使用模型进行预测
        
        参数:
            loader: 数据加载器
            beam_width: 束宽度，如果为1则使用贪婪解码
            
        返回:
            predictions: 预测结果列表
            log_probs_list: 对数概率列表
        """
        self.model.eval()
        predictions = []
        log_probs_list = []
        
        with torch.no_grad():
            for features, _, feature_lens, _ in tqdm(loader, desc="Predicting"):
                features = features.to(self.device)
                
                # 前向传播
                log_probs, output_lens = self.model(features, feature_lens)
                
                # 解码
                batch_predictions = self.decode(log_probs, output_lens, beam_width=beam_width)
                
                predictions.extend(batch_predictions)
                log_probs_list.append(log_probs.cpu().numpy())
        
        return predictions, log_probs_list
    
    def _save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'output_dim': self.model.output_dim,
            'num_layers': self.model.num_layers,
            'bidirectional': self.model.bidirectional,
            'dropout': self.model.dropout,
            'rnn_type': self.model.rnn_type
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
        plt.figure(figsize=(6, 4))
        
        plt.plot(self.history['train_loss'], label='Training Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('CTC Training and Validation Loss')
        
        plt.tight_layout()
        plt.savefig('ctc_training_history.png')
        plt.close()


class CTCModel:
    """CTC声学模型封装类，用于与系统其他部分交互"""
    
    def __init__(self, input_dim=40, hidden_dim=512, output_dim=42, 
                 num_layers=2, bidirectional=True, speaker_id=None):
        """
        初始化CTC模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（音素/字符数 + 1，加1是为了blank标记）
            num_layers: RNN层数
            bidirectional: 是否使用双向RNN
            speaker_id: 说话人ID
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.speaker_id = speaker_id
        
        # 创建模型
        self.model = CTCRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        
        # 创建训练器
        self.trainer = CTCTrainer(self.model)
        
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # 训练历史
        self.history = None
        
    def to(self, device):
        """
        将模型移动到指定设备
        
        参数:
            device: 目标设备（"cuda"或"cpu"）
            
        返回:
            self: 返回自身，便于链式调用
        """
        self.device = device
        self.model.to(device)
        self.trainer.device = device
        return self
        
    def train(self, features, labels, epochs=30, batch_size=32, learning_rate=0.001):
        """
        训练模型
        
        参数:
            features: 特征列表，每个元素是一个形状为(time_steps, feature_dim)的NumPy数组
            labels: 标签列表，每个元素是一个形状为(label_len,)的NumPy数组
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        返回:
            训练历史
        """
        # 处理特征字典情况
        if isinstance(features, dict):
            # 选择一个特征类型（如MFCC）用于训练
            if 'mfcc_librosa' in features:
                selected_features = features['mfcc_librosa']
            elif 'melspectrogram' in features:
                selected_features = features['melspectrogram']
            else:
                # 使用第一个可用的特征类型
                selected_features = next(iter(features.values()))
            features = selected_features
        
        # 确保标签是整数类型
        labels = np.array(labels, dtype=np.int64)
        
        # 处理一维数组标签
        if labels.ndim == 1:
            # 每个样本一个标签
            labels = [[l] for l in labels]
        
        # 创建数据集
        dataset = CTCDataset(features, labels)
        
        # 创建数据加载器
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 训练模型
        self.history = self.trainer.train(
            train_loader=loader,
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir=f"models/ctc_{self.speaker_id}" if self.speaker_id else "models/ctc"
        )
        
        return self.history
    
    def predict(self, features, batch_size=32):
        """
        使用模型进行预测
        
        参数:
            features: 特征列表，每个元素是一个形状为(time_steps, feature_dim)的NumPy数组
            batch_size: 批次大小
            
        返回:
            预测结果列表
        """
        # 创建假标签（仅用于数据集构建）
        dummy_labels = [np.zeros(1, dtype=np.int64) for _ in range(len(features))]
        
        # 创建数据集
        dataset = CTCDataset(features, dummy_labels)
        
        # 创建数据加载器
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 进行预测
        predictions = self.trainer.predict(loader)
        
        return predictions
    
    def evaluate(self, features, labels, batch_size=32):
        """
        评估模型
        
        参数:
            features: 特征列表，每个元素是一个形状为(time_steps, feature_dim)的NumPy数组
            labels: 标签列表，每个元素是一个形状为(label_len,)的NumPy数组
            batch_size: 批次大小
            
        返回:
            评估结果（损失和准确率）
        """
        # 确保标签是整数类型
        labels = [l.astype(np.int64) for l in labels]
        
        # 创建数据集
        dataset = CTCDataset(features, labels)
        
        # 创建数据加载器
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 评估模型
        metrics = self.trainer._validate(loader)
        
        return metrics
    
    def save(self, path=None):
        """保存模型"""
        if path is None:
            path = f"models/ctc_{self.speaker_id}/model.pt" if self.speaker_id else "models/ctc/model.pt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.trainer._save_model(path)
        
    def load(self, path):
        """加载模型"""
        self.trainer.load_model(path)
        
    def state_dict(self):
        """返回模型的状态字典，用于保存模型"""
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        """加载模型的状态字典
        
        参数:
            state_dict: 模型状态字典
        """
        self.model.load_state_dict(state_dict)
        return self 