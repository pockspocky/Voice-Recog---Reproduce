"""
音频预处理模块：实现信号预加重、归一化、静音检测和分帧等处理
"""

import numpy as np
from scipy import signal

def preemphasis(audio_data, coef=0.97):
    """
    预加重 - 增强高频信号
    
    参数:
        audio_data: 音频数据
        coef: 预加重系数，一般取0.95-0.97
        
    返回:
        经过预加重处理的音频数据
    """
    return np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])

def normalize(audio_data):
    """
    归一化音频数据
    
    参数:
        audio_data: 音频数据
        
    返回:
        归一化后的音频数据
    """
    if np.max(np.abs(audio_data)) > 0:
        return audio_data / np.max(np.abs(audio_data))
    return audio_data

def remove_silence(audio_data, energy_threshold=0.05, pad_ms=50):
    """
    静音检测和去除
    
    参数:
        audio_data: 音频数据
        energy_threshold: 能量阈值
        pad_ms: 在语音前后保留的毫秒数
        
    返回:
        去除静音后的音频数据
    """
    # 计算能量
    energy = np.square(audio_data)
    
    # 找出能量高于阈值的帧
    frames_with_voice = energy > energy_threshold
    
    # 如果全部是静音，返回原始数据
    if not np.any(frames_with_voice):
        print("警告: 未检测到语音")
        return audio_data
    
    # 找出第一个和最后一个有声音的索引
    voiced_indices = np.where(frames_with_voice)[0]
    start_idx = max(0, voiced_indices[0] - pad_ms)
    end_idx = min(len(audio_data), voiced_indices[-1] + pad_ms)
    
    # 返回有声音的部分
    return audio_data[start_idx:end_idx]

def frame_signal(audio_data, sample_rate, frame_length_ms=25, frame_step_ms=10):
    """
    将音频信号分帧
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        frame_length_ms: 帧长(毫秒)
        frame_step_ms: 帧移(毫秒)
        
    返回:
        分帧后的数据矩阵，每一行是一帧
    """
    # 计算每帧的采样点数
    frame_length = int(sample_rate * frame_length_ms / 1000)
    # 计算帧移的采样点数
    frame_step = int(sample_rate * frame_step_ms / 1000)
    
    # 计算帧数
    signal_length = len(audio_data)
    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_length - frame_length) / frame_step))
    
    # 填充信号以确保所有帧都能完整分割
    pad_length = (num_frames - 1) * frame_step + frame_length
    padded_signal = np.zeros(pad_length)
    padded_signal[:signal_length] = audio_data
    
    # 创建帧矩阵
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        frames[i] = padded_signal[i * frame_step : i * frame_step + frame_length]
    
    return frames

def apply_window(frames, window_type='hamming'):
    """
    对分帧信号应用窗函数
    
    参数:
        frames: 分帧后的信号
        window_type: 窗函数类型，如'hamming', 'hann'等
        
    返回:
        加窗后的帧
    """
    if window_type == 'hamming':
        window = np.hamming(frames.shape[1])
    elif window_type == 'hann':
        window = np.hanning(frames.shape[1])
    else:
        print(f"不支持的窗类型: {window_type}，使用默认hamming窗")
        window = np.hamming(frames.shape[1])
    
    return frames * window

def preprocess_audio(audio_data, sample_rate):
    """
    音频预处理主函数：组合所有预处理步骤
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
    
    返回:
        处理后的帧数据和归一化后的音频数据
    """
    # 步骤1: 预加重
    emphasized_audio = preemphasis(audio_data)
    
    # 步骤2: 归一化
    normalized_audio = normalize(emphasized_audio)
    
    # 步骤3: 去除静音
    voice_audio = remove_silence(normalized_audio)
    
    # 步骤4: 分帧
    frames = frame_signal(voice_audio, sample_rate)
    
    # 步骤5: 加窗
    windowed_frames = apply_window(frames)
    
    print(f"预处理完成: 共{len(frames)}帧")
    
    return windowed_frames, normalized_audio 