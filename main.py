#!/usr/bin/env python3
"""
语音识别系统 - 音频捕获、预处理与特征提取
"""

import os
import argparse
import time
import numpy as np
import torch
from utils.audio_capture import capture_audio
from utils.audio_preprocessing import preprocess_audio
from utils.visualization import visualize_all, plot_waveform, plot_spectrogram
from utils.file_io import save_audio, load_audio, get_audio_info
from utils.feature_extraction import extract_all_features, save_features
from acoustic_models.integrated_model import IntegratedAcousticModel

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='语音识别系统 - 声学建模')
    parser.add_argument('--input', '-i', type=str, help='输入音频文件路径')
    parser.add_argument('--output', '-o', type=str, default='output/processed_audio.wav', 
                        help='输出音频文件路径')
    parser.add_argument('--features_dir', '-fd', type=str, default='output/features', 
                        help='特征保存目录')
    parser.add_argument('--duration', '-d', type=int, default=5, 
                        help='录音时长(秒)，仅在未提供输入文件时使用')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000, 
                        help='采样率')
    parser.add_argument('--visualize', '-v', action='store_true', 
                        help='是否显示可视化结果')
    parser.add_argument('--extract_features', '-ef', action='store_true', 
                        help='是否提取特征')
    parser.add_argument('--speaker_id', '-sid', type=str, 
                        help='说话人ID，用于分类不同人的声音')
    parser.add_argument('--model_type', '-mt', type=str, default='all',
                        choices=['gmm_hmm', 'dnn', 'cnn', 'rnn', 'transformer', 'ctc', 'all'],
                        help='要使用的声学模型类型')
    parser.add_argument('--train', '-t', action='store_true',
                        help='是否训练模型')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='是否评估模型')
    parser.add_argument('--predict', '-p', action='store_true',
                        help='是否进行预测')
    parser.add_argument('--phoneme_set', '-ps', type=str, default='phonemes.txt',
                        help='音素集合文件路径')
    
    args = parser.parse_args()
    
    # 如果未提供说话人ID，提示用户输入
    speaker_id = args.speaker_id
    if not speaker_id:
        speaker_id = input("\n请输入说话人ID (用于分类不同人的声音): ")
        # 如果用户仍未输入，使用默认值
        if not speaker_id:
            speaker_id = "unknown"
    
    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加说话人ID到输出文件名中
    base_output = os.path.basename(args.output)
    base_name, ext = os.path.splitext(base_output)
    output_with_speaker = os.path.join(output_dir, f"{speaker_id}_{base_name}{ext}")
    
    # 第一步：捕获或加载音频
    print("\n=== 步骤1: 音频捕获 ===")
    if args.input:
        # 加载音频文件
        audio_data, sample_rate = load_audio(args.input, args.sample_rate)
        file_name = os.path.splitext(os.path.basename(args.input))[0]
        if audio_data is None:
            print("加载音频失败，程序退出")
            return
    else:
        # 录制新音频
        print(f"\n开始为说话人 '{speaker_id}' 录音...")
        audio_data, sample_rate = capture_audio(None, args.duration, args.sample_rate)
        timestamp = int(time.time())
        file_name = f"{speaker_id}_recorded_{timestamp}"
    
    # 显示音频信息
    print(f"\n说话人ID: {speaker_id}")
    print("音频信息:")
    print(f"采样率: {sample_rate} Hz")
    print(f"音频长度: {len(audio_data)} 采样点")
    print(f"时长: {len(audio_data)/sample_rate:.2f} 秒")
    
    # 第二步：音频预处理
    print("\n=== 步骤2: 音频预处理 ===")
    frames, processed_audio = preprocess_audio(audio_data, sample_rate)
    
    # 保存处理后的音频
    print("\n=== 步骤3: 保存预处理后的音频 ===")
    save_audio(processed_audio, sample_rate, output_with_speaker)
    
    # 第四步：特征提取
    features = None
    if args.extract_features:
        print("\n=== 步骤4: 特征提取 ===")
        
        # 创建带有说话人ID的特征目录
        speaker_features_dir = os.path.join(args.features_dir, speaker_id)
        
        # 提取原始音频特征
        print("\n提取原始音频特征...")
        raw_features = extract_all_features(audio_data, sample_rate)
        
        # 提取预处理后音频特征
        print("\n提取预处理后音频特征...")
        processed_features = extract_all_features(processed_audio, sample_rate)
        
        # 保存特征
        print("\n保存特征...")
        # 确保特征保存目录存在
        os.makedirs(speaker_features_dir, exist_ok=True)
        
        # 保存原始特征
        raw_features_files = save_features(
            raw_features, 
            os.path.join(speaker_features_dir, 'raw'), 
            file_name
        )
        
        # 保存预处理后特征
        processed_features_files = save_features(
            processed_features, 
            os.path.join(speaker_features_dir, 'processed'), 
            file_name
        )
        
        print(f"\n原始特征文件数量: {len(raw_features_files)}")
        print(f"预处理后特征文件数量: {len(processed_features_files)}")
        print(f"特征已保存至: {speaker_features_dir}")
        
        # 将特征保存为变量以供后续使用
        features = processed_features
    
    # 可视化（如果需要）
    if args.visualize:
        print("\n=== 步骤5: 可视化 ===")
        visualize_all(audio_data, sample_rate, processed_audio)
    
    # 第六步：声学建模（如果需要）
    if args.train or args.evaluate or args.predict:
        print("\n=== 步骤6: 声学建模 ===")
        
        # 如果尚未提取特征，则提取特征
        if features is None:
            print("\n提取特征用于声学建模...")
            features = extract_all_features(processed_audio, sample_rate)
        
        # 设置模型配置
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_config = {
            'sample_rate': sample_rate,
            'n_fft': 512,
            'hop_length': 160,
            'n_mels': 80,
            'n_mfcc': 39,  # 13个MFCC系数 + delta + delta-delta
            'deltas': True,
            'cmvn': True,
            
            # GMM-HMM模型配置
            'gmm_hmm': {
                'n_states': 5,
                'n_mix': 4,
                'cov_type': 'diag',
                'n_iter': 20
            },
            
            # DNN模型配置
            'dnn': {
                'input_dim': 39,  # MFCC特征维度
                'hidden_dims': [512, 512, 512],
                'output_dim': 48,  # 音素数量
                'dropout_prob': 0.2
            },
            
            # CNN模型配置
            'cnn': {
                'input_channels': 1,
                'feature_dim': 39,  # MFCC特征维度
                'num_classes': 48,  # 音素数量
                'context_window': 11,
                'cnn_channels': [64, 128, 256],
                'kernel_sizes': [3, 3, 3],
                'fc_dims': [1024, 512]
            },
            
            # RNN模型配置
            'rnn': {
                'input_dim': 39,  # MFCC特征维度
                'hidden_dim': 256,
                'num_layers': 3,
                'output_dim': 48,  # 音素数量
                'bidirectional': True,
                'rnn_type': 'lstm',
                'dropout': 0.2
            },
            
            # Transformer模型配置
            'transformer': {
                'input_dim': 39,  # MFCC特征维度
                'd_model': 512,
                'nhead': 8,
                'num_encoder_layers': 6,
                'dim_feedforward': 2048,
                'output_dim': 48,  # 音素数量
                'dropout': 0.1,
                'max_len': 1000
            },
            
            # CTC模型配置
            'ctc': {
                'input_dim': 39,  # MFCC特征维度
                'hidden_dim': 256,
                'output_dim': 49,  # 音素数量 + blank
                'num_layers': 4,
                'bidirectional': True,
                'dropout': 0.2,
                'rnn_type': 'lstm'
            }
        }
        
        # 选择要使用的模型
        models_to_use = []
        if args.model_type == 'all':
            models_to_use = ['gmm_hmm', 'dnn', 'cnn', 'rnn', 'transformer', 'ctc']
        else:
            models_to_use = [args.model_type]
        
        # 过滤model_config，只保留要使用的模型配置
        filtered_config = {k: model_config[k] for k in model_config if k in models_to_use or k not in ['gmm_hmm', 'dnn', 'cnn', 'rnn', 'transformer', 'ctc']}
        
        # 创建集成声学模型
        print(f"\n创建集成声学模型，使用模型: {', '.join(models_to_use)}")
        acoustic_model = IntegratedAcousticModel(filtered_config, device)
        
        # 训练模型（如果需要）
        if args.train:
            print("\n训练声学模型...")
            
            # 这里，我们需要加载训练数据
            # 实际应用中，应该从数据集加载特征和标签
            # 这里我们只是演示，使用随机生成的标签
            
            # 假设我们有一个音素集合
            phonemes = []
            if os.path.exists(args.phoneme_set):
                with open(args.phoneme_set, 'r') as f:
                    phonemes = [line.strip() for line in f.readlines()]
            else:
                # 使用默认的48个音素（示例）
                phonemes = [f"p{i}" for i in range(48)]
            
            # 为当前特征生成随机标签（仅作为演示）
            np.random.seed(42)
            feature_key = 'mfcc_librosa'  # 使用MFCC特征
            
            if feature_key in features:
                feature_data = features[feature_key]
                n_frames = len(feature_data)
                
                # 生成随机标签，表示音素ID
                labels = np.random.randint(0, len(phonemes), n_frames)
                
                # 将当前特征和标签转换为列表格式
                features_list = [feature_data]
                labels_list = [labels]
                
                # 为每个模型准备数据集并训练
                for model_type in models_to_use:
                    if model_type == 'gmm_hmm':
                        # 训练GMM-HMM模型
                        print(f"\n训练GMM-HMM模型...")
                        # 为每个音素生成一些示例特征
                        phoneme_features = []
                        phoneme_ids = []
                        
                        for i, phoneme in enumerate(phonemes):
                            # 随机选择帧作为该音素的样本
                            mask = labels == i
                            if np.sum(mask) > 0:
                                phoneme_features.append(feature_data[mask])
                                phoneme_ids.append(phoneme)
                        
                        acoustic_model.train_gmm_hmm(phoneme_features, phoneme_ids)
                    else:
                        # 为神经网络模型准备数据集
                        print(f"\n准备{model_type}模型数据集...")
                        loaders = acoustic_model.prepare_datasets(
                            features_list, 
                            labels_list, 
                            model_type, 
                            train_ratio=0.7, 
                            val_ratio=0.15, 
                            batch_size=32
                        )
                        
                        # 训练模型
                        print(f"\n训练{model_type}模型...")
                        # 为了演示，我们只训练一个epoch
                        train_params = {
                            'epochs': 1,
                            'learning_rate': 0.001,
                            'weight_decay': 1e-5,
                            'patience': 3,
                            'save_dir': f"models/{model_type}"
                        }
                        
                        acoustic_model.train_model(model_type, loaders, **train_params)
            else:
                print(f"未找到{feature_key}特征，无法训练模型")
        
        # 评估模型（如果需要）
        if args.evaluate:
            print("\n评估声学模型...")
            # 这里应该加载测试数据集，但我们跳过这一步，因为这只是演示
            
        # 预测（如果需要）
        if args.predict:
            print("\n使用声学模型进行预测...")
            feature_key = 'mfcc_librosa'  # 使用MFCC特征
            
            if feature_key in features:
                feature_data = features[feature_key]
                
                # 使用集成模型进行预测
                print("\n使用集成模型进行预测...")
                prediction = acoustic_model.ensemble_predict(feature_data, models_to_use)
                
                print("\n预测结果:")
                if 'prediction' in prediction:
                    pred = prediction['prediction']
                    if isinstance(pred, np.ndarray):
                        # 对于序列预测，我们只显示前10个预测
                        print(f"预测序列 (前10个): {pred[:10]}...")
                    else:
                        print(f"预测结果: {pred}")
                
                # 如果使用了GMM-HMM模型，显示音素识别结果
                if 'gmm_hmm' in models_to_use:
                    # 假设我们有一个音素集合
                    phonemes = []
                    if os.path.exists(args.phoneme_set):
                        with open(args.phoneme_set, 'r') as f:
                            phonemes = [line.strip() for line in f.readlines()]
                    else:
                        # 使用默认的48个音素（示例）
                        phonemes = [f"p{i}" for i in range(48)]
                    
                    # 使用GMM-HMM进行音素识别
                    best_phoneme_id, log_likelihood = acoustic_model.gmm_hmm_decode(
                        feature_data, phonemes
                    )
                    
                    if best_phoneme_id is not None:
                        print(f"\nGMM-HMM识别结果:")
                        print(f"最可能的音素: {best_phoneme_id}")
                        print(f"对数似然: {log_likelihood:.4f}")
            else:
                print(f"未找到{feature_key}特征，无法进行预测")
    
    print(f"\n处理完成。说话人 '{speaker_id}' 的音频已保存至: {output_with_speaker}")

if __name__ == "__main__":
    main() 