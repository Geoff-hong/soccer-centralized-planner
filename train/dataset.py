import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob  # [新增] 用于查找文件

class SoccerDataset(Dataset):
    def __init__(self, data_path_or_dir=None, obs_history=1, file_paths=None):
        """
        data_path_or_dir: 可以是单个 .npz 文件路径，也可以是包含多个 .npz 文件的文件夹路径
        """
        self.obs_history = max(1, int(obs_history))
        self.file_paths = []
        
        if file_paths is not None:
            # 显式传入文件列表
            self.file_paths = list(file_paths)
            self.file_paths.sort()
        else:
            # 1. 智能判断：是文件夹还是文件？
            if data_path_or_dir is None:
                raise ValueError("data_path_or_dir 不能为空")
            if os.path.isdir(data_path_or_dir):
                # 如果是文件夹，查找里面所有的 .npz 文件
                # 这里的 **/*.npz 可以根据需要调整，目前假设都在根目录下
                self.file_paths = glob.glob(os.path.join(data_path_or_dir, "*.npz"))
                self.file_paths.sort() # 排序，保证每次加载顺序一致
            elif os.path.isfile(data_path_or_dir):
                # 如果是单个文件
                self.file_paths = [data_path_or_dir]
            else:
                raise ValueError(f"路径不存在或无效: {data_path_or_dir}")
            
        if not self.file_paths:
            raise FileNotFoundError(f"在 {data_path_or_dir} 中未找到任何 .npz 数据文件")

        print(f"Found {len(self.file_paths)} data files. Loading and merging...")

        # 2. 循环加载并合并数据
        obs_list = []
        move_list = []
        pass_flag_list = []
        passer_id_list = []
        pass_dir_list = []
        receiver_id_list = []
        
        for fp in self.file_paths:
            fname = os.path.basename(fp)
            try:
                data = np.load(fp)
                # 简单的完整性检查
                if 'obs' not in data:
                    print(f"  [Skip] {fname} 缺少 obs 键")
                    continue

                curr_obs = data['obs']

                # 新格式
                if 'act_move' in data and 'pass_flag' in data:
                    curr_move = data['act_move']
                    curr_pass_flag = data['pass_flag']
                    curr_passer_id = data.get('passer_id', np.full(len(curr_obs), -1, dtype=np.int64))
                    curr_pass_dir = data.get('pass_dir', np.full(len(curr_obs), -1, dtype=np.int64))
                    curr_receiver_id = data.get('receiver_id', np.full(len(curr_obs), -1, dtype=np.int64))

                    obs_list.append(curr_obs)
                    move_list.append(curr_move)
                    pass_flag_list.append(curr_pass_flag)
                    passer_id_list.append(curr_passer_id)
                    pass_dir_list.append(curr_pass_dir)
                    receiver_id_list.append(curr_receiver_id)
                    print(f"  > Loaded {fname}: {curr_obs.shape[0]} frames (new format)")
                # 兼容旧格式
                elif 'acts' in data:
                    curr_acts = data['acts']
                    obs_list.append(curr_obs)
                    move_list.append(curr_acts[:, :, 0:2])
                    pass_flag_list.append((curr_acts[:, :, 2] > 0.5).any(axis=1).astype(np.float32))
                    passer_id_list.append(np.full(len(curr_obs), -1, dtype=np.int64))
                    pass_dir_list.append(np.full(len(curr_obs), -1, dtype=np.int64))
                    receiver_id_list.append(np.full(len(curr_obs), -1, dtype=np.int64))
                    print(f"  > Loaded {fname}: {curr_obs.shape[0]} frames (legacy acts)")
                else:
                    print(f"  [Skip] {fname} 缺少 act_move/pass_flag 或 acts 键")
                    continue
                
            except Exception as e:
                print(f"  [Error] Failed to load {fname}: {e}")
        
        # 3. 合并所有比赛的数据 (Concatenate)
        if not obs_list:
            raise RuntimeError("没有成功加载任何数据！请检查路径和文件格式。")

        # 使用 numpy 进行合并（效率最高）
        # axis=0 表示在“帧数”维度上叠加
        combined_obs = np.concatenate(obs_list, axis=0)
        combined_move = np.concatenate(move_list, axis=0)
        combined_pass_flag = np.concatenate(pass_flag_list, axis=0)
        combined_passer_id = np.concatenate(passer_id_list, axis=0)
        combined_pass_dir = np.concatenate(pass_dir_list, axis=0)
        combined_receiver_id = np.concatenate(receiver_id_list, axis=0)

        # 转为 Tensor
        self.obs = torch.FloatTensor(combined_obs)
        self.move = torch.FloatTensor(combined_move)
        self.pass_flag = torch.FloatTensor(combined_pass_flag)
        self.passer_id = torch.LongTensor(combined_passer_id)
        self.pass_dir = torch.LongTensor(combined_pass_dir)
        self.receiver_id = torch.LongTensor(combined_receiver_id)

        # 4. 重新计算统计信息 (逻辑与之前一致，但现在是针对所有比赛的总和)
        self.total_frames = len(self.obs)
        self.kick_mask = self.pass_flag > 0.5
        self.n_kicks = self.kick_mask.sum().item()
        
        print(f"Dataset Ready! Total Frames: {self.total_frames}, Total Pass Frames: {self.n_kicks}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if self.obs_history == 1:
            return self.obs[idx], self.move[idx], self.pass_flag[idx], self.passer_id[idx], self.pass_dir[idx], self.receiver_id[idx]

        # 堆叠过去 K 帧（不够则用第一帧补齐）
        start = max(0, idx - (self.obs_history - 1))
        frames = self.obs[start:idx + 1]  # [t, 23, 3]

        if frames.shape[0] < self.obs_history:
            pad = frames[0].unsqueeze(0).repeat(self.obs_history - frames.shape[0], 1, 1)
            frames = torch.cat([pad, frames], dim=0)

        stacked = frames.permute(1, 0, 2).reshape(23, self.obs_history * 3)
        return stacked, self.move[idx], self.pass_flag[idx], self.passer_id[idx], self.pass_dir[idx], self.receiver_id[idx]

    # === 计算采样权重 (逻辑完全不用变，因为它基于合并后的 self.total_frames 计算) ===
    def get_sample_weights(self):
        n_no_kick = self.total_frames - self.n_kicks
        
        # 防止除以零（极少数情况）
        if self.n_kicks == 0: return torch.ones(self.total_frames, dtype=torch.double)
        
        weight_kick = 1.0 / self.n_kicks 
        weight_no_kick = 1.0 / n_no_kick 
        
        weights = np.zeros(self.total_frames)
        weights[self.kick_mask.numpy()] = weight_kick
        weights[~self.kick_mask.numpy()] = weight_no_kick
        
        return torch.DoubleTensor(weights)
