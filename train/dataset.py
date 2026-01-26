import torch
from torch.utils.data import Dataset
import numpy as np

class SoccerDataset(Dataset):
    def __init__(self, data_path, augment=False):
        """
        Args:
            data_path: .npz 文件的路径
            augment: 是否进行数据增强 (暂未实现，预留接口)
        """
        # 加载数据
        data = np.load(data_path)
        self.obs = torch.FloatTensor(data['obs']) # [N, 23, 3]
        self.acts = torch.FloatTensor(data['acts']) # [N, 11, 4]
        
        # 预计算一些统计信息，方便训练时调整采样权重
        # 找出哪些帧包含踢球动作 (任意一个球员 trigger > 0.5)
        # acts: [N, 11, 4], trigger at index 2
        self.kick_indices = (self.acts[:, :, 2] > 0.5).any(dim=1).nonzero(as_tuple=True)[0]
        
        print(f"Loaded {len(self.obs)} frames.")
        print(f" > Found {len(self.kick_indices)} frames with KICK events ({len(self.kick_indices)/len(self.obs)*100:.2f}%)")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # 返回 (Observation, Action_Label)
        return self.obs[idx], self.acts[idx]

    # --- 高级功能：加权采样器 ---
    # 如果不用 pos_weight，我们可以用 WeightedRandomSampler
    # 这里我们先手动写一个函数获取样本权重
    def get_sample_weights(self):
        # 目标：给包含 Kick 的帧更高的采样概率
        # 1. 初始化所有权重为 1
        weights = torch.ones(len(self))
        
        # 2. 给 Kick 帧增加权重 (比如 10 倍)
        weights[self.kick_indices] = 10.0
        
        return weights