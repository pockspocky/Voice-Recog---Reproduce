"""
音频文件输入输出模块：负责音频文件的保存和加载
"""

import os
import numpy as np
import soundfile as sf
import librosa

def save_audio(audio_data, sample_rate, file_path, format='wav'):
    """
    保存音频数据到文件
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        file_path: 保存路径
        format: 文件格式，默认为wav
        
    返回:
        保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 保存音频
        sf.write(file_path, audio_data, sample_rate, format=format)
        print(f"音频已保存至: {file_path}")
        return True
    except Exception as e:
        print(f"保存音频失败: {e}")
        return False

def load_audio(file_path, sample_rate=None):
    """
    从文件加载音频数据
    
    参数:
        file_path: 音频文件路径
        sample_rate: 目标采样率，如果为None则使用文件原始采样率
        
    返回:
        audio_data: 音频数据
        sample_rate: 采样率
    """
    try:
        audio_data, original_sr = librosa.load(file_path, sr=sample_rate)
        print(f"已加载音频文件: {file_path}")
        print(f"采样率: {original_sr if sample_rate is None else sample_rate} Hz")
        print(f"时长: {len(audio_data)/(original_sr if sample_rate is None else sample_rate):.2f} 秒")
        return audio_data, original_sr if sample_rate is None else sample_rate
    except Exception as e:
        print(f"加载音频失败: {e}")
        return None, None

def list_audio_files(directory, formats=None):
    """
    列出目录中的所有音频文件
    
    参数:
        directory: 目录路径
        formats: 文件格式列表，如['wav', 'mp3']，为None则检查所有常见音频格式
        
    返回:
        音频文件路径列表
    """
    if formats is None:
        formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
    
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(f'.{fmt}') for fmt in formats):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def get_audio_info(file_path):
    """
    获取音频文件的信息
    
    参数:
        file_path: 音频文件路径
        
    返回:
        包含音频信息的字典
    """
    try:
        audio_data, sr = librosa.load(file_path, sr=None)
        duration = len(audio_data) / sr
        
        info = {
            "文件路径": file_path,
            "采样率": sr,
            "时长(秒)": duration,
            "样本数": len(audio_data),
            "声道数": 1 if audio_data.ndim == 1 else audio_data.shape[1],
            "文件大小(MB)": os.path.getsize(file_path) / (1024 * 1024)
        }
        
        return info
    except Exception as e:
        print(f"获取音频信息失败: {e}")
        return None 