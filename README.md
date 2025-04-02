# 语音识别系统 - 音频捕获与预处理、特征提取

这个项目实现了语音识别系统的前两个步骤：音频捕获与预处理，以及特征提取。包括音频录制/加载、信号预处理（预加重、归一化、静音检测、分帧等）、多种特征提取方法，以及可视化功能。

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
- 特征保存：保存为多种格式（JSON和Pickle）
- 说话人分类：支持通过ID对不同人的声音进行分类和管理
- 可视化：
  - 波形图：显示时域信号
  - 频谱图：显示频域信息
  - 声谱图：时频分析
  - MFCC特征：显示提取的特征

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

### 命令行参数

- `--input`, `-i`: 输入音频文件路径
- `--output`, `-o`: 输出音频文件路径（默认：'output/processed_audio.wav'）
- `--features_dir`, `-fd`: 特征保存目录（默认：'output/features'）
- `--duration`, `-d`: 录音时长(秒)，仅在未提供输入文件时使用（默认：5）
- `--sample_rate`, `-sr`: 采样率（默认：16000）
- `--visualize`, `-v`: 显示可视化结果
- `--extract_features`, `-ef`: 是否提取特征
- `--speaker_id`, `-sid`: 说话人ID，用于分类不同人的声音

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
└── utils/               # 工具函数
    ├── __init__.py      # 包初始化
    ├── audio_capture.py # 音频捕获模块
    ├── audio_preprocessing.py # 音频预处理模块
    ├── feature_extraction.py # 特征提取模块
    ├── file_io.py       # 文件输入输出模块
    └── visualization.py # 可视化模块
```

## 后续步骤

这个项目实现了语音识别的前两个步骤。后续步骤将包括：

1. 声学模型：使用深度学习模型识别语音单元
2. 语言模型：结合语言规则和统计模型预测词序列
3. 解码器：整合以上组件，输出最终文本 