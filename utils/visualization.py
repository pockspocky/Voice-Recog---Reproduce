"""
音频可视化模块：绘制波形图和频谱图等
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def plot_waveform(audio_data, sample_rate, title="波形图"):
    """
    绘制音频波形图
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        title: 图表标题
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title(title)
    plt.xlabel("时间 (秒)")
    plt.ylabel("振幅")
    plt.tight_layout()
    plt.show()

def plot_spectrum(audio_data, sample_rate, title="频谱图"):
    """
    绘制音频频谱图
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        title: 图表标题
    """
    # 计算频谱
    X = np.fft.fft(audio_data)
    X_mag = np.absolute(X)
    
    # 频率轴
    freq = np.linspace(0, sample_rate, len(X_mag))
    
    # 只显示前半部分（由于对称性）
    half_len = len(X_mag) // 2
    
    plt.figure(figsize=(12, 4))
    plt.plot(freq[:half_len], X_mag[:half_len])
    plt.title(title)
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio_data, sample_rate, title="声谱图"):
    """
    绘制声谱图
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        title: 图表标题
    """
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mfcc(audio_data, sample_rate, n_mfcc=13, title="MFCC特征"):
    """
    绘制MFCC特征
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_mfcc: MFCC系数的数量
        title: 图表标题
    """
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_all(audio_data, sample_rate, processed_audio=None):
    """
    显示所有可视化结果
    
    参数:
        audio_data: 原始音频数据
        sample_rate: 采样率
        processed_audio: 处理后的音频数据，如果提供则会对比显示
    """
    plt.figure(figsize=(12, 10))
    
    # 原始波形
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title("原始波形")
    
    # 处理后波形（如果提供）
    if processed_audio is not None:
        plt.subplot(4, 1, 2)
        librosa.display.waveshow(processed_audio, sr=sample_rate)
        plt.title("预处理后波形")
        start_idx = 3
    else:
        start_idx = 2
    
    # 频谱图
    plt.subplot(4, 1, start_idx)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("声谱图")
    
    # MFCC
    plt.subplot(4, 1, start_idx+1)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title("MFCC特征")
    
    plt.tight_layout()
    plt.show() 