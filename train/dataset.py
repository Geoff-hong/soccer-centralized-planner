# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class SoccerDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        try:
            data = np.load(data_path)
            self.obs = torch.FloatTensor(data['obs'])
            self.acts = torch.FloatTensor(data['acts'])
        except Exception as e:
            raise FileNotFoundError(f"无法加载数据: {e}")

        self.total_frames = len(self.obs)
        # 找出哪些帧有 Kick (Trigger > 0.5)
        # acts: [N, 11, 4], trigger index is 2
        # 只要任意一个球员 trigger=1，这一帧就是 Kick Frame
        self.kick_mask = (self.acts[:, :, 2] > 0.5).any(dim=1) 
        self.n_kicks = self.kick_mask.sum().item()
        
        print(f" > Total: {self.total_frames}, Kicks: {self.n_kicks}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

    # === 新增：计算采样权重 ===
    def get_sample_weights(self):
        # 目标：让 Kick 帧被采样的概率增大
        # 1. 计算类别权重
        n_no_kick = self.total_frames - self.n_kicks
        weight_kick = 1.0 / self.n_kicks  # 稀少样本权重高
        weight_no_kick = 1.0 / n_no_kick  # 常见样本权重低
        
        # 2. 为每个样本分配权重
        weights = np.zeros(self.total_frames)
        weights[self.kick_mask.numpy()] = weight_kick
        weights[~self.kick_mask.numpy()] = weight_no_kick
        
        return torch.DoubleTensor(weights)