#!/usr/bin/env python3
"""
语音识别系统 - 音频捕获、预处理与特征提取
"""

import os
import argparse
import time
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, List
from utils.audio_capture import capture_audio
from utils.audio_preprocessing import preprocess_audio
from utils.visualization import visualize_all, plot_waveform, plot_spectrogram
from utils.file_io import save_audio, load_audio, get_audio_info, list_audio_files
from utils.feature_extraction import extract_all_features, save_features
from acoustic_models.integrated_model import IntegratedAcousticModel
from config_manager import ConfigManager
from acoustic_models.feature_extractor import FeatureExtractor
import logging

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('voice_recognition.log'),
            logging.StreamHandler()
        ]
    )

def capture_audio(duration: float = 5.0, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """录制音频"""
    print(f"开始录音，持续{duration}秒...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("录音结束")
    return audio_data, sample_rate

def preprocess_audio(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """预处理音频数据"""
    # 归一化
    audio_data = audio_data / np.max(np.abs(audio_data))
    return audio_data

def save_audio(audio_data: np.ndarray, sample_rate: int, output_path: str):
    """保存音频文件"""
    sf.write(output_path, audio_data, sample_rate)

def plot_audio(audio_data: np.ndarray, sample_rate: int, output_path: str):
    """绘制音频波形"""
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data)
    plt.title('音频波形')
    plt.xlabel('采样点')
    plt.ylabel('振幅')
    plt.savefig(output_path)
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='语音识别系统')
    parser.add_argument('--input', type=str, help='输入音频文件路径')
    parser.add_argument('--input_dir', type=str, help='输入音频文件夹路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--speaker_id', type=str, required=True, help='说话人ID')
    parser.add_argument('--model_type', type=str, default='all', 
                      choices=['all', 'gmm_hmm', 'dnn', 'cnn', 'rnn', 'transformer', 'ctc'],
                      help='要使用的模型类型')
    parser.add_argument('--train', action='store_true', help='是否训练模型')
    parser.add_argument('--evaluate', action='store_true', help='是否评估模型')
    parser.add_argument('--predict', action='store_true', help='是否进行预测')
    parser.add_argument('--duration', type=float, default=5.0, help='录音时长（秒）')
    parser.add_argument('--sample_rate', type=int, default=16000, help='采样率')
    parser.add_argument('--features_dir', '-fd', type=str, default='output/features', 
                        help='特征保存目录')
    parser.add_argument('--visualize', '-v', action='store_true', 
                        help='是否显示可视化结果')
    parser.add_argument('--extract_features', '-ef', action='store_true', 
                        help='是否提取特征')
    parser.add_argument('--phoneme_set', '-ps', type=str, default='phonemes.txt',
                        help='音素集合文件路径')
    parser.add_argument('--formats', type=str, default='wav,mp3,flac', 
                        help='处理的音频格式，用逗号分隔')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 获取音频数据
    audio_files = []
    
    if args.input_dir:
        logger.info(f"从目录加载音频文件: {args.input_dir}")
        formats = args.formats.split(',')
        audio_files = list_audio_files(args.input_dir, formats=formats)
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        if not audio_files:
            logger.warning(f"在目录 {args.input_dir} 中未找到音频文件")
            return
    elif args.input:
        logger.info(f"从文件加载音频: {args.input}")
        audio_files = [args.input]
    else:
        logger.info("开始录音...")
        audio_data, sample_rate = capture_audio(args.duration, args.sample_rate)
        # 保存录音到输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(args.output, f"audio_{timestamp}.wav")
        save_audio(audio_data, sample_rate, audio_path)
        logger.info(f"录音已保存: {audio_path}")
        audio_files = [audio_path]
    
    # 初始化声学模型
    acoustic_model = IntegratedAcousticModel(args.speaker_id, args.model_type)
    
    # 处理所有音频文件
    all_features = []
    
    for audio_file in audio_files:
        logger.info(f"处理音频文件: {audio_file}")
        audio_data, sample_rate = load_audio(audio_file)
        
        if audio_data is None:
            logger.error(f"无法加载音频文件: {audio_file}")
            continue
            
        # 预处理音频
        audio_data = preprocess_audio(audio_data, sample_rate)
        
        # 生成当前音频文件的输出路径
        file_basename = os.path.basename(audio_file)
        filename, _ = os.path.splitext(file_basename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存处理后的音频
        processed_audio_path = os.path.join(args.output, f"{filename}_processed_{timestamp}.wav")
        save_audio(audio_data, sample_rate, processed_audio_path)
        logger.info(f"处理后的音频已保存: {processed_audio_path}")
        
        # 绘制音频波形
        waveform_path = os.path.join(args.output, f"{filename}_waveform_{timestamp}.png")
        plot_audio(audio_data, sample_rate, waveform_path)
        logger.info(f"波形图已保存: {waveform_path}")
        
        # 提取特征
        features = acoustic_model.extract_features(audio_data)
        all_features.append(features)
        
        if args.predict:
            logger.info(f"为文件 {file_basename} 生成预测...")
            predictions = acoustic_model.predict(features)
            ensemble_pred = acoustic_model.ensemble_predict(features)
            
            # 保存预测结果
            results_path = os.path.join(args.output, f"{filename}_predictions_{timestamp}.txt")
            with open(results_path, 'w') as f:
                f.write(f"文件: {audio_file}\n")
                f.write("各模型预测结果:\n")
                for model_name, pred in predictions.items():
                    f.write(f"{model_name}: {pred}\n")
                f.write(f"\n集成预测结果: {ensemble_pred}\n")
            logger.info(f"预测结果已保存: {results_path}")
    
    # 合并所有特征用于训练或评估
    if all_features:
        # 选择一个特征类型（如MFCC）用于训练
        if isinstance(all_features[0], dict):
            # 如果特征是字典，选择一种特征类型
            if 'mfcc_librosa' in all_features[0]:
                selected_feature_type = 'mfcc_librosa'
            elif 'melspectrogram' in all_features[0]:
                selected_feature_type = 'melspectrogram'
            else:
                # 使用第一个可用的特征类型
                selected_feature_type = next(iter(all_features[0].keys()))
            
            # 提取所选特征类型
            selected_features = [f[selected_feature_type] for f in all_features]
        else:
            # 如果特征已经是数组形式
            selected_features = all_features
        
        # 合并特征
        combined_features = np.vstack(selected_features)
        
        # 为特征创建简单标签（全0），因为我们只有一个类别
        dummy_labels = np.zeros(len(combined_features), dtype=np.int64)
        
        if args.train:
            logger.info("开始训练模型...")
            history = acoustic_model.train(
                combined_features,
                dummy_labels,
                epochs=10
            )
            logger.info("模型训练完成")
        
        if args.evaluate:
            logger.info("开始评估模型...")
            metrics = acoustic_model.evaluate(
                combined_features,
                dummy_labels
            )
            logger.info(f"评估结果: {metrics}")

if __name__ == "__main__":
    main() 