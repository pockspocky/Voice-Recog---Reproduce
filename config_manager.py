import json
import os
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class ConfigManager:
    def __init__(self, base_dir: str = "models/speaker_configs"):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        
    def get_speaker_dir(self, speaker_id: str) -> str:
        """获取说话人配置目录"""
        return os.path.join(self.base_dir, speaker_id)
    
    def get_config_path(self, speaker_id: str) -> str:
        """获取说话人配置文件路径"""
        return os.path.join(self.get_speaker_dir(speaker_id), "config.json")
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "speaker_id": "",
            "model_configs": {
                "gmm_hmm": {
                    "n_components": 16,
                    "n_iter": 100,
                    "covariance_type": "diag"
                },
                "dnn": {
                    "hidden_layers": [256, 128],
                    "dropout_rate": 0.3,
                    "learning_rate": 0.001
                },
                "cnn": {
                    "conv_layers": [(32, 3), (64, 3)],
                    "pool_size": 2,
                    "learning_rate": 0.001
                },
                "rnn": {
                    "hidden_size": 256,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "learning_rate": 0.001
                },
                "transformer": {
                    "d_model": 256,
                    "nhead": 8,
                    "num_layers": 6,
                    "dropout": 0.1,
                    "learning_rate": 0.0001
                },
                "ctc": {
                    "hidden_size": 256,
                    "num_layers": 3,
                    "dropout": 0.3,
                    "learning_rate": 0.001
                }
            },
            "training_history": {
                "last_training": None,
                "best_accuracy": 0.0,
                "total_training_time": 0.0
            },
            "feature_config": {
                "sample_rate": 16000,
                "n_mfcc": 13,
                "n_mels": 80,
                "hop_length": 160,
                "n_fft": 512,
                "deltas": True,
                "cmvn": True
            }
        }
    
    def create_speaker_config(self, speaker_id: str) -> Dict[str, Any]:
        """创建新的说话人配置"""
        config = self.get_default_config()
        config["speaker_id"] = speaker_id
        
        # 创建目录
        speaker_dir = self.get_speaker_dir(speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # 保存配置
        self.save_config(speaker_id, config)
        return config
    
    def load_config(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """加载说话人配置"""
        config_path = self.get_config_path(speaker_id)
        if not os.path.exists(config_path):
            self.logger.warning(f"配置不存在，创建新配置: {speaker_id}")
            return self.create_speaker_config(speaker_id)
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            return self.create_speaker_config(speaker_id)
    
    def save_config(self, speaker_id: str, config: Dict[str, Any]) -> bool:
        """保存说话人配置"""
        try:
            config_path = self.get_config_path(speaker_id)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False
    
    def update_training_history(self, speaker_id: str, accuracy: float, training_time: float) -> bool:
        """更新训练历史"""
        config = self.load_config(speaker_id)
        if not config:
            return False
            
        config["training_history"].update({
            "last_training": str(datetime.now()),
            "best_accuracy": max(config["training_history"]["best_accuracy"], accuracy),
            "total_training_time": config["training_history"]["total_training_time"] + training_time
        })
        
        return self.save_config(speaker_id, config)
    
    def get_model_config(self, speaker_id: str, model_type: str) -> Optional[Dict[str, Any]]:
        """获取特定模型的配置"""
        config = self.load_config(speaker_id)
        if not config:
            return None
        return config["model_configs"].get(model_type)
    
    def update_model_config(self, speaker_id: str, model_type: str, new_config: Dict[str, Any]) -> bool:
        """更新特定模型的配置"""
        config = self.load_config(speaker_id)
        if not config:
            return False
            
        config["model_configs"][model_type].update(new_config)
        return self.save_config(speaker_id, config) 