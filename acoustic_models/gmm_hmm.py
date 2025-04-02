"""
GMM-HMM声学模型
"""

import os
import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import joblib


class GMMHMM:
    """GMM-HMM声学模型，用于传统的语音识别"""
    
    def __init__(self, n_states=5, n_mix=2, cov_type='diag', n_iter=10):
        """
        初始化GMM-HMM模型
        
        参数:
            n_states: HMM的状态数
            n_mix: 每个状态的高斯混合成分数
            cov_type: 协方差类型 ('diag', 'full', 'tied', 'spherical')
            n_iter: 训练迭代次数
        """
        self.n_states = n_states
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = {}  # 存储不同音素的HMM模型
        self.gmms = {}    # 存储不同音素的GMM模型
        
    def train_gmm(self, features, phoneme_id):
        """
        训练单个音素的GMM模型
        
        参数:
            features: 特征序列，形状为(n_samples, n_features)
            phoneme_id: 音素ID
        """
        gmm = GaussianMixture(
            n_components=self.n_mix,
            covariance_type=self.cov_type,
            max_iter=self.n_iter,
            random_state=42
        )
        gmm.fit(features)
        self.gmms[phoneme_id] = gmm
        return gmm
    
    def train_hmm(self, features_list, phoneme_id):
        """
        训练单个音素的HMM模型
        
        参数:
            features_list: 特征序列列表，每个元素形状为(n_samples, n_features)
            phoneme_id: 音素ID
        """
        # 将所有特征拼接起来，用于计算统计量
        all_features = np.vstack(features_list)
        
        # 初始化HMM模型
        model = hmm.GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_mix,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            init_params='stmcw',  # 初始化状态转移矩阵、均值和协方差
            random_state=42
        )
        
        # 设置初始估计
        model.startprob_ = np.array([0.7] + [0.3 / (self.n_states - 1)] * (self.n_states - 1))
        
        # 设置状态转移矩阵（左右HMM，只允许自跳转和向右转移）
        transmat = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if i == self.n_states - 1:
                transmat[i, i] = 1.0  # 最后一个状态自跳转概率为1
            else:
                transmat[i, i] = 0.6   # 自跳转概率
                transmat[i, i + 1] = 0.4  # 向右转移概率
        model.transmat_ = transmat
        
        # 训练模型
        lengths = [len(f) for f in features_list]
        X = np.vstack(features_list)
        model.fit(X, lengths)
        
        self.models[phoneme_id] = model
        return model
    
    def score(self, features, phoneme_id):
        """
        计算特征序列在特定音素HMM模型下的对数似然
        
        参数:
            features: 特征序列，形状为(n_samples, n_features)
            phoneme_id: 音素ID
            
        返回:
            log_likelihood: 对数似然值
        """
        if phoneme_id in self.models:
            return self.models[phoneme_id].score(features)
        else:
            raise ValueError(f"模型中不存在音素 {phoneme_id}")
    
    def decode(self, features, phoneme_id):
        """
        使用Viterbi算法对特征序列进行解码
        
        参数:
            features: 特征序列，形状为(n_samples, n_features)
            phoneme_id: 音素ID
            
        返回:
            log_likelihood: 对数似然值
            states: 维特比最佳状态序列
        """
        if phoneme_id in self.models:
            log_likelihood, states = self.models[phoneme_id].decode(features)
            return log_likelihood, states
        else:
            raise ValueError(f"模型中不存在音素 {phoneme_id}")
    
    def gmm_score(self, features, phoneme_id):
        """
        计算特征在GMM模型下的对数似然
        
        参数:
            features: 特征序列，形状为(n_samples, n_features)
            phoneme_id: 音素ID
            
        返回:
            log_likelihood: 对数似然值
        """
        if phoneme_id in self.gmms:
            return self.gmms[phoneme_id].score_samples(features).mean()
        else:
            raise ValueError(f"模型中不存在音素GMM {phoneme_id}")
    
    def save_models(self, model_dir):
        """
        保存所有模型
        
        参数:
            model_dir: 模型保存目录
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存GMM模型
        gmm_dir = os.path.join(model_dir, 'gmm')
        os.makedirs(gmm_dir, exist_ok=True)
        for phoneme_id, gmm in self.gmms.items():
            joblib.dump(gmm, os.path.join(gmm_dir, f"{phoneme_id}.pkl"))
        
        # 保存HMM模型
        hmm_dir = os.path.join(model_dir, 'hmm')
        os.makedirs(hmm_dir, exist_ok=True)
        for phoneme_id, model in self.models.items():
            joblib.dump(model, os.path.join(hmm_dir, f"{phoneme_id}.pkl"))
            
        # 保存模型参数
        params = {
            'n_states': self.n_states,
            'n_mix': self.n_mix,
            'cov_type': self.cov_type,
            'n_iter': self.n_iter
        }
        with open(os.path.join(model_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)
    
    def load_models(self, model_dir):
        """
        加载所有模型
        
        参数:
            model_dir: 模型保存目录
        """
        # 加载模型参数
        with open(os.path.join(model_dir, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)
            self.n_states = params['n_states']
            self.n_mix = params['n_mix']
            self.cov_type = params['cov_type']
            self.n_iter = params['n_iter']
        
        # 加载GMM模型
        gmm_dir = os.path.join(model_dir, 'gmm')
        if os.path.exists(gmm_dir):
            for model_file in os.listdir(gmm_dir):
                if model_file.endswith('.pkl'):
                    phoneme_id = model_file[:-4]  # 去掉.pkl后缀
                    self.gmms[phoneme_id] = joblib.load(os.path.join(gmm_dir, model_file))
        
        # 加载HMM模型
        hmm_dir = os.path.join(model_dir, 'hmm')
        if os.path.exists(hmm_dir):
            for model_file in os.listdir(hmm_dir):
                if model_file.endswith('.pkl'):
                    phoneme_id = model_file[:-4]  # 去掉.pkl后缀
                    self.models[phoneme_id] = joblib.load(os.path.join(hmm_dir, model_file)) 