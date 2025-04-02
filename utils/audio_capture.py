"""
音频捕获模块：负责音频文件加载和实时录音
"""

import numpy as np
import librosa

def capture_audio(file_path=None, duration=5, sample_rate=16000):
    """
    捕获音频或加载音频文件
    
    参数:
        file_path: 音频文件路径，如为None则录制新音频
        duration: 录制时长(秒)
        sample_rate: 采样率
    
    返回:
        audio_data: 音频数据
        sample_rate: 采样率
    """
    if file_path:
        # 加载已有音频文件
        audio_data, sample_rate = librosa.load(file_path, sr=sample_rate)
        print(f"已加载音频文件: {file_path}")
        print(f"时长: {len(audio_data)/sample_rate:.2f}秒")
        return audio_data, sample_rate
    else:
        # 这里需要录制新音频
        try:
            import pyaudio
            import wave
            
            temp_file = "temp_recording.wav"
            
            # 录音参数设置
            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            
            # 初始化PyAudio
            p = pyaudio.PyAudio()
            
            # 开始录音
            print("开始录音...")
            stream = p.open(format=format,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk)
            
            frames = []
            for i in range(0, int(sample_rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
                
            # 停止录音
            print("录音结束")
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 保存录音
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # 加载录制的音频
            audio_data, sample_rate = librosa.load(temp_file, sr=sample_rate)
            print(f"录音已保存至: {temp_file}")
            return audio_data, sample_rate
            
        except ImportError:
            print("警告: 未安装PyAudio，无法录音。请使用 'pip install pyaudio' 安装")
            # 返回一个空的音频样本
            return np.zeros(sample_rate * duration), sample_rate 