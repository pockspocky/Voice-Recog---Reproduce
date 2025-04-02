"""
音频特征提取模块：实现多种音频特征提取方法
"""

import numpy as np
import librosa
import os
import pickle
import json
from scipy.fftpack import dct

def extract_mfcc(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    提取MFCC特征 (梅尔频率倒谱系数)
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_mfcc: MFCC系数数量
        n_fft: FFT窗口大小
        hop_length: 帧移
        
    返回:
        MFCC特征矩阵
    """
    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(
        y=audio_data, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # 添加一阶差分和二阶差分特征 (delta和delta-delta)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # 合并所有特征
    combined_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    # 标准化特征
    mean = np.mean(combined_mfccs, axis=1, keepdims=True)
    std = np.std(combined_mfccs, axis=1, keepdims=True)
    normalized_mfccs = (combined_mfccs - mean) / (std + 1e-10)
    
    print(f"MFCC特征shape: {normalized_mfccs.shape}")
    return normalized_mfccs

def extract_filterbank(audio_data, sample_rate, n_fft=2048, hop_length=512, n_mels=40):
    """
    提取滤波器组特征 (Filter Bank)
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: Mel滤波器数量
        
    返回:
        滤波器组特征矩阵
    """
    # 计算梅尔频谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # 转换为对数刻度
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    # 添加一阶差分和二阶差分特征
    delta = librosa.feature.delta(log_mel_spectrogram)
    delta2 = librosa.feature.delta(log_mel_spectrogram, order=2)
    
    # 合并所有特征
    combined_features = np.vstack([log_mel_spectrogram, delta, delta2])
    
    # 标准化特征
    mean = np.mean(combined_features, axis=1, keepdims=True)
    std = np.std(combined_features, axis=1, keepdims=True)
    normalized_features = (combined_features - mean) / (std + 1e-10)
    
    print(f"滤波器组特征shape: {normalized_features.shape}")
    return normalized_features

def extract_spectral_features(audio_data, sample_rate, n_fft=2048, hop_length=512):
    """
    提取频谱特征 (Spectral Features)
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        
    返回:
        频谱特征字典
    """
    spectral_features = {}
    
    # 1. 谱质心 (Spectral Centroid)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_features['spectral_centroid'] = spectral_centroid
    
    # 2. 谱带宽 (Spectral Bandwidth)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_features['spectral_bandwidth'] = spectral_bandwidth
    
    # 3. 谱平坦度 (Spectral Flatness)
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio_data, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_features['spectral_flatness'] = spectral_flatness
    
    # 4. 谱衰减 (Spectral Rolloff)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    spectral_features['spectral_rolloff'] = spectral_rolloff
    
    # 5. 过零率 (Zero Crossing Rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio_data, frame_length=n_fft, hop_length=hop_length
    )[0]
    spectral_features['zero_crossing_rate'] = zero_crossing_rate
    
    # 6. RMS能量 (Root Mean Square Energy)
    rms = librosa.feature.rms(
        y=audio_data, frame_length=n_fft, hop_length=hop_length
    )[0]
    spectral_features['rms_energy'] = rms
    
    print(f"频谱特征数量: {len(spectral_features)}")
    return spectral_features

def extract_chroma_features(audio_data, sample_rate, n_fft=2048, hop_length=512, n_chroma=12):
    """
    提取色度特征 (Chroma Features)
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_chroma: 色度特征维度
        
    返回:
        色度特征矩阵
    """
    # 提取色度特征
    chroma = librosa.feature.chroma_stft(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_chroma=n_chroma
    )
    
    print(f"色度特征shape: {chroma.shape}")
    return chroma

def extract_spectral_contrast(audio_data, sample_rate, n_fft=2048, hop_length=512, n_bands=6):
    """
    提取谱对比度特征 (Spectral Contrast)
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_bands: 频带数量
        
    返回:
        谱对比度特征矩阵
    """
    # 提取谱对比度特征
    contrast = librosa.feature.spectral_contrast(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_bands=n_bands
    )
    
    print(f"谱对比度特征shape: {contrast.shape}")
    return contrast

def extract_tonnetz(audio_data, sample_rate):
    """
    提取音调网络特征 (Tonnetz)
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        
    返回:
        音调网络特征矩阵
    """
    # 提取音调网络特征
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
    
    print(f"音调网络特征shape: {tonnetz.shape}")
    return tonnetz

def extract_all_features(audio_data, sample_rate, n_fft=2048, hop_length=512):
    """
    提取所有特征并保存为字典
    
    参数:
        audio_data: 音频数据
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        
    返回:
        包含所有特征的字典
    """
    features = {}
    
    # 1. MFCC特征
    features['mfcc'] = extract_mfcc(audio_data, sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    # 2. 滤波器组特征
    features['filterbank'] = extract_filterbank(audio_data, sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    # 3. 频谱特征
    features['spectral'] = extract_spectral_features(audio_data, sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    # 4. 色度特征
    features['chroma'] = extract_chroma_features(audio_data, sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    # 5. 谱对比度特征
    features['contrast'] = extract_spectral_contrast(audio_data, sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    # 6. 音调网络特征
    features['tonnetz'] = extract_tonnetz(audio_data, sample_rate)
    
    return features

def save_features(features, output_dir, file_name):
    """
    保存特征到文件
    
    参数:
        features: 特征字典
        output_dir: 输出目录
        file_name: 文件名前缀
        
    返回:
        保存的文件路径列表
    """
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # 保存每种特征
    for feature_name, feature_data in features.items():
        # 二进制保存（使用pickle）
        pickle_path = os.path.join(output_dir, f"{file_name}_{feature_name}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(feature_data, f)
        saved_files.append(pickle_path)
        
        # 尝试保存为JSON格式（如果特征是可序列化的）
        try:
            if feature_name != 'spectral':  # 频谱特征是字典，单独处理
                # 转换numpy数组为列表
                json_data = feature_data.tolist()
                json_path = os.path.join(output_dir, f"{file_name}_{feature_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f)
                saved_files.append(json_path)
            else:
                # 频谱特征是字典，需要将每个值转换为列表
                json_data = {}
                for key, value in feature_data.items():
                    json_data[key] = value.tolist()
                json_path = os.path.join(output_dir, f"{file_name}_{feature_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f)
                saved_files.append(json_path)
        except:
            print(f"无法将{feature_name}特征保存为JSON格式")
    
    # 保存特征汇总信息
    info = {
        "feature_types": list(features.keys()),
        "file_prefix": file_name,
        "files": saved_files
    }
    
    info_path = os.path.join(output_dir, f"{file_name}_features_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f)
    saved_files.append(info_path)
    
    print(f"所有特征已保存至: {output_dir}")
    return saved_files 