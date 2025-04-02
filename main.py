#!/usr/bin/env python3
"""
语音识别系统 - 音频捕获、预处理与特征提取
"""

import os
import argparse
import time
from utils.audio_capture import capture_audio
from utils.audio_preprocessing import preprocess_audio
from utils.visualization import visualize_all, plot_waveform, plot_spectrogram
from utils.file_io import save_audio, load_audio, get_audio_info
from utils.feature_extraction import extract_all_features, save_features

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音频捕获、预处理与特征提取')
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
    
    # 第四步：特征提取（如果需要）
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
    
    # 可视化（如果需要）
    if args.visualize:
        print("\n=== 步骤5: 可视化 ===")
        visualize_all(audio_data, sample_rate, processed_audio)
    
    print(f"\n处理完成。说话人 '{speaker_id}' 的音频已保存至: {output_with_speaker}")

if __name__ == "__main__":
    main() 