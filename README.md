# 语音识别系统 - 音频捕获与预处理、特征提取与声学建模

这个项目实现了语音识别系统的核心步骤：音频捕获与预处理、特征提取以及声学建模。包括音频录制/加载、信号预处理（预加重、归一化、静音检测、分帧等）、多种特征提取方法，声学模型训练和评估，以及可视化功能。

## 功能特点

- 音频捕获：支持从麦克风录制或从文件加载
- 音频预处理：
  - 预加重：增强高频信号
  - 归一化：统一音量
  - 静音检测和去除：去除无声部分
  - 分帧和加窗：为特征提取做准备
- 特征提取：同时支持多种特征提取方法
  - MFCC特征 (梅尔频率倒谱系数)：包含一阶和二阶差分
  - 滤波器组特征 (Filter Bank)：梅尔频谱特征
  - 频谱特征 (Spectral Features)：包含谱质心、谱带宽、谱平坦度、谱衰减、过零率、RMS能量
  - 色度特征 (Chroma Features)：表示音乐色度相关特征
  - 谱对比度特征 (Spectral Contrast)：表示声音谱峰谷差异
  - 音调网络特征 (Tonnetz)：表示音调关系
- 声学建模：支持多种声学模型架构
  - GMM-HMM：传统声学模型（高斯混合模型-隐马尔可夫模型）
  - DNN：深度神经网络模型
  - CNN：卷积神经网络模型
  - RNN：循环神经网络模型
  - Transformer：基于自注意力机制的模型
  - CTC：连接时序分类模型
  - 集成模型：结合多种模型优势的集成方法
- 特征保存：保存为多种格式（JSON和Pickle）
- 说话人分类：支持通过ID对不同人的声音进行分类和管理
- 可视化：
  - 波形图：显示时域信号
  - 频谱图：显示频域信息
  - 声谱图：时频分析
  - MFCC特征：显示提取的特征
  - 训练历史：显示模型训练过程
  - 混淆矩阵：显示分类性能

## 安装依赖

```bash
pip install -r requirements.txt
```

注意：在一些系统上安装PyAudio可能需要额外步骤：
- Windows: `pip install pyaudio`
- macOS: `brew install portaudio` 然后 `pip install pyaudio`
- Linux: `sudo apt-get install python3-pyaudio` 或 `sudo apt-get install portaudio19-dev` 然后 `pip install pyaudio`

## 使用方法

### 基本用法

```bash
# 从麦克风录制5秒并处理，会提示输入说话人ID
python main.py

# 指定说话人ID进行录音
python main.py --speaker_id user1

# 加载音频文件并处理
python main.py --input example.wav --speaker_id user1

# 录制10秒并处理
python main.py --duration 10 --speaker_id user1

# 处理并保存到指定位置
python main.py --output processed/my_audio.wav --speaker_id user1

# 显示可视化结果
python main.py --visualize --speaker_id user1

# 提取特征并保存
python main.py --input example.wav --extract_features --speaker_id user1

# 完整流程：处理、提取特征并可视化
python main.py --input example.wav --extract_features --visualize --speaker_id user1
```

### 声学模型使用方法

```bash
# 训练DNN声学模型
python main.py --input example.wav --extract_features --model_type dnn --train --speaker_id user1

# 使用CNN模型进行预测
python main.py --input example.wav --extract_features --model_type cnn --predict --speaker_id user1

# 训练并评估RNN模型
python main.py --input example.wav --extract_features --model_type rnn --train --evaluate --speaker_id user1

# 使用Transformer模型
python main.py --input example.wav --extract_features --model_type transformer --train --predict --speaker_id user1

# 使用CTC模型
python main.py --input example.wav --extract_features --model_type ctc --train --predict --speaker_id user1

# 使用GMM-HMM模型
python main.py --input example.wav --extract_features --model_type gmm_hmm --train --predict --speaker_id user1

# 使用集成模型（结合所有模型）
python main.py --input example.wav --extract_features --model_type all --train --predict --speaker_id user1
```

### 命令行参数

- `--input`, `-i`: 输入音频文件路径
- `--output`, `-o`: 输出音频文件路径（默认：'output/processed_audio.wav'）
- `--features_dir`, `-fd`: 特征保存目录（默认：'output/features'）
- `--duration`, `-d`: 录音时长(秒)，仅在未提供输入文件时使用（默认：5）
- `--sample_rate`, `-sr`: 采样率（默认：16000）
- `--visualize`, `-v`: 显示可视化结果
- `--extract_features`, `-ef`: 是否提取特征
- `--speaker_id`, `-sid`: 说话人ID，用于分类不同人的声音
- `--model_type`, `-mt`: 要使用的声学模型类型（默认：'all'，可选：'gmm_hmm', 'dnn', 'cnn', 'rnn', 'transformer', 'ctc', 'all'）
- `--train`, `-t`: 是否训练模型
- `--evaluate`, `-e`: 是否评估模型
- `--predict`, `-p`: 是否进行预测
- `--phoneme_set`, `-ps`: 音素集合文件路径（默认：'phonemes.txt'）

## 说话人分类说明

该系统支持通过说话人ID对不同人的声音进行分类。具体实现方式：

1. 可以通过`--speaker_id`参数指定说话人ID，如果未指定，程序会提示手动输入。
2. 音频文件和特征将按说话人ID进行组织保存：
   - 音频文件保存格式：`[说话人ID]_[文件名].wav`
   - 特征文件保存在对应说话人ID的目录下：`features/[说话人ID]/...`
3. 这种组织方式便于后续按说话人进行模型训练或识别。

## 特征提取说明

系统提取并保存多种音频特征，针对原始音频和预处理后的音频各保存一份：

1. **MFCC特征**：梅尔频率倒谱系数，包含基础MFCC特征、一阶差分和二阶差分，共39维。
2. **滤波器组特征**：梅尔频谱特征，包含原始特征、一阶差分和二阶差分，共120维。
3. **频谱特征**：包含6种不同的频谱特征（谱质心、谱带宽、谱平坦度等）。
4. **色度特征**：与音乐色度相关的特征，12维。
5. **谱对比度特征**：表示声音谱峰谷差异的特征，7维。
6. **音调网络特征**：表示音调关系的特征，6维。

所有特征都保存为两种格式：
- `.pkl`：Python pickle格式，适合Python程序直接加载
- `.json`：JSON格式，适合跨平台和跨语言使用

## 声学模型说明

系统实现了多种声学模型，能够对语音特征进行建模和预测：

### GMM-HMM模型

传统的声学建模方法，结合高斯混合模型和隐马尔可夫模型：
- 为每个音素建立独立的声学模型
- 使用EM算法训练GMM组件
- 通过HMM捕捉时序关系
- 适合数据量较小的场景

### DNN模型

深度神经网络模型：
- 多层前馈神经网络结构
- 使用ReLU激活函数
- 包含Dropout层防止过拟合
- 适合帧级别的声学分类

### CNN模型

卷积神经网络模型：
- 使用卷积层捕捉局部特征模式
- 利用上下文窗口提供时域信息
- 通过池化层减少特征维度
- 适合捕捉声学特征的局部模式

### RNN模型

循环神经网络模型：
- 支持LSTM或GRU单元
- 可选双向结构增强上下文获取
- 原生支持序列数据处理
- 适合捕捉长时依赖关系

### Transformer模型

基于自注意力机制的模型：
- 使用多头自注意力机制
- 包含位置编码捕捉序列位置信息
- 具有并行计算能力
- 适合捕捉全局上下文关系

### CTC模型

连接时序分类模型：
- 基于RNN+CTC结构
- 无需精确的帧级对齐
- 直接输出音素序列
- 适合端到端的语音识别

### 集成模型

集成多种模型的综合方法：
- 结合不同模型的预测结果
- 使用加权投票或概率平均
- 可自定义模型组合和权重
- 通常优于单一模型的性能

## 训练和评估方法

系统提供了统一的训练和评估接口：

### 训练过程

1. **数据准备**：
   - 提取音频特征（通常使用MFCC）
   - 准备对应的标签数据
   - 划分训练集、验证集和测试集

2. **模型配置**：
   - 设置模型超参数
   - 初始化模型结构
   - 配置优化器和损失函数

3. **训练流程**：
   - 进行多轮（epoch）训练
   - 在验证集上监控性能
   - 使用早停（early stopping）防止过拟合
   - 保存最佳模型

### 评估方法

1. **模型预测**：
   - 加载训练好的模型
   - 对测试数据进行预测

2. **性能评估**：
   - 计算准确率等指标
   - 生成混淆矩阵
   - 对于序列模型，计算序列错误率

3. **可视化分析**：
   - 绘制训练历史曲线
   - 显示混淆矩阵
   - 分析错误预测案例

## 项目结构

```
.
├── main.py              # 主程序
├── requirements.txt     # 依赖需求
├── README.md            # 项目说明
├── output/              # 输出目录
│   ├── [说话人ID]_processed_audio.wav  # 处理后的音频
│   └── features/        # 特征目录
│       └── [说话人ID]/  # 按说话人分类
│           ├── raw/     # 原始音频特征
│           └── processed/ # 处理后音频特征
├── models/              # 模型保存目录
│   ├── gmm_hmm/         # GMM-HMM模型
│   ├── dnn/             # DNN模型
│   ├── cnn/             # CNN模型
│   ├── rnn/             # RNN模型
│   ├── transformer/     # Transformer模型
│   └── ctc/             # CTC模型
├── acoustic_models/     # 声学模型模块
│   ├── __init__.py      # 包初始化
│   ├── feature_extractor.py # 特征提取器
│   ├── gmm_hmm.py       # GMM-HMM模型
│   ├── dnn_model.py     # DNN模型
│   ├── cnn_model.py     # CNN模型
│   ├── rnn_model.py     # RNN模型
│   ├── transformer_model.py # Transformer模型
│   ├── ctc_model.py     # CTC模型
│   └── integrated_model.py # 集成模型
└── utils/               # 工具函数
    ├── __init__.py      # 包初始化
    ├── audio_capture.py # 音频捕获模块
    ├── audio_preprocessing.py # 音频预处理模块
    ├── feature_extraction.py # 特征提取模块
    ├── file_io.py       # 文件输入输出模块
    └── visualization.py # 可视化模块
```

## 后续步骤

这个项目实现了语音识别的前三个主要步骤。后续步骤将包括：

1. 语言模型：结合语言规则和统计模型预测词序列
2. 解码器：整合声学模型和语言模型，输出最终文本
3. 端到端语音识别：使用完全端到端的架构简化流程 