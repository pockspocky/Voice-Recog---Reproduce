"""
声学特征提取器，提供多种特征提取方法
"""

import numpy as np
import librosa
import torch
import torchaudio
from python_speech_features import mfcc, fbank, logfbank
from python_speech_features import delta


class FeatureExtractor:
    """声学特征提取器，支持多种特征提取方法"""
    
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, 
                 n_mels=80, n_mfcc=13, deltas=True, cmvn=True):
        """
        初始化特征提取器
        
        参数:
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            n_mels: Mel滤波器组数量
            n_mfcc: MFCC系数数量
            deltas: 是否计算一阶和二阶差分
            cmvn: 是否进行倒谱均值方差归一化
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.deltas = deltas
        self.cmvn = cmvn
        
        # torchaudio特征提取器
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def extract_mfcc_librosa(self, audio):
        """使用librosa提取MFCC特征"""
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        if self.deltas:
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            mfccs = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
        
        if self.cmvn:
            mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
            
        return mfccs.T  # 返回形状 (time_steps, features)
    
    def extract_mfcc_torchaudio(self, audio):
        """使用torchaudio提取MFCC特征"""
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # 调整维度
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        mfccs = self.mfcc_transform(audio)
        
        # 添加一阶和二阶差分
        if self.deltas:
            delta_mfccs = torchaudio.functional.compute_deltas(mfccs)
            delta2_mfccs = torchaudio.functional.compute_deltas(delta_mfccs)
            mfccs = torch.cat([mfccs, delta_mfccs, delta2_mfccs], dim=1)
        
        # 倒谱均值方差归一化
        if self.cmvn:
            mean = torch.mean(mfccs, dim=2, keepdim=True)
            std = torch.std(mfccs, dim=2, keepdim=True)
            mfccs = (mfccs - mean) / (std + 1e-8)
        
        return mfccs.squeeze(0).transpose(0, 1).numpy()  # 返回形状 (time_steps, features)
    
    def extract_filterbank(self, audio):
        """提取滤波器组特征"""
        # 使用python_speech_features提取滤波器组特征
        filter_banks, energies = fbank(audio, samplerate=self.sample_rate, 
                                      nfilt=self.n_mels, nfft=self.n_fft, 
                                      winlen=self.n_fft/self.sample_rate, 
                                      winstep=self.hop_length/self.sample_rate)
        
        # 计算对数滤波器组特征
        log_filter_banks = np.log(filter_banks + 1e-8)
        
        if self.deltas:
            delta_fbanks = delta(log_filter_banks, 2)
            delta2_fbanks = delta(delta_fbanks, 2)
            log_filter_banks = np.hstack([log_filter_banks, delta_fbanks, delta2_fbanks])
        
        if self.cmvn:
            log_filter_banks = (log_filter_banks - np.mean(log_filter_banks, axis=0)) / np.std(log_filter_banks, axis=0)
        
        return log_filter_banks  # 返回形状 (time_steps, features)
    
    def extract_melspectrogram(self, audio):
        """提取梅尔频谱图特征"""
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # 调整维度
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        mel_specgram = self.mel_spec_transform(audio)
        
        # 转换为对数梅尔频谱图
        log_mel_specgram = torch.log(mel_specgram + 1e-8)
        
        # 转换为NumPy数组并转置
        log_mel_specgram = log_mel_specgram.squeeze(0).transpose(0, 1).numpy()
        
        if self.cmvn:
            log_mel_specgram = (log_mel_specgram - np.mean(log_mel_specgram, axis=0)) / (np.std(log_mel_specgram, axis=0) + 1e-8)
        
        return log_mel_specgram  # 返回形状 (time_steps, features)
    
    def extract_all_features(self, audio):
        """提取所有特征"""
        features = {
            'mfcc_librosa': self.extract_mfcc_librosa(audio),
            'mfcc_torchaudio': self.extract_mfcc_torchaudio(audio),
            'filterbank': self.extract_filterbank(audio),
            'melspectrogram': self.extract_melspectrogram(audio)
        }
        return features 