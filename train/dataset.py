import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob  # [新增] 用于查找文件

class SoccerDataset(Dataset):
    def __init__(self, data_path_or_dir=None, obs_history=1, file_paths=None, move_horizon=1):
        """
        data_path_or_dir: 可以是单个 .npz 文件路径，也可以是包含多个 .npz 文件的文件夹路径
        """
        self.obs_history = max(1, int(obs_history))
        self.move_horizon = max(1, int(move_horizon))
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
        dt_list = []
        expected_agents = None
        expected_attackers = None
        
        self.file_lengths = []
        for fp in self.file_paths:
            fname = os.path.basename(fp)
            try:
                data = np.load(fp)
                # 简单的完整性检查
                if 'obs' not in data:
                    print(f"  [Skip] {fname} 缺少 obs 键")
                    continue

                curr_obs = data['obs']

                def _normalize_dt(dt_val, n_frames):
                    # Ensure dt is 1-D with length = n_frames
                    if dt_val is None:
                        return np.full(n_frames, 0.1, dtype=np.float32)
                    dt_arr = np.asarray(dt_val, dtype=np.float32)
                    if dt_arr.ndim == 0:
                        return np.full(n_frames, float(dt_arr), dtype=np.float32)
                    if dt_arr.size == 0:
                        return np.full(n_frames, 0.1, dtype=np.float32)
                    if dt_arr.shape[0] == n_frames:
                        return dt_arr
                    if dt_arr.size == 1:
                        return np.full(n_frames, float(dt_arr.reshape(-1)[0]), dtype=np.float32)
                    # Fallback: trim or pad with last value
                    if dt_arr.shape[0] > n_frames:
                        return dt_arr[:n_frames]
                    pad = np.full(n_frames - dt_arr.shape[0], dt_arr[-1], dtype=np.float32)
                    return np.concatenate([dt_arr, pad], axis=0)

                # 新格式
                if 'act_move' in data and 'pass_flag' in data:
                    curr_move = data['act_move']
                    curr_pass_flag = data['pass_flag']
                    curr_passer_id = data.get('passer_id', np.full(len(curr_obs), -1, dtype=np.int64))
                    curr_pass_dir = data.get('pass_dir', np.full(len(curr_obs), -1, dtype=np.int64))
                    curr_receiver_id = data.get('receiver_id', np.full(len(curr_obs), -1, dtype=np.int64))
                    curr_dt = _normalize_dt(data.get('dt', 0.1), len(curr_obs))
                    # Normalize act_move to m/s if needed (some legacy files store m/frame).
                    move_unit = data.get('move_unit', None)
                    if move_unit is not None:
                        try:
                            if hasattr(move_unit, "item"):
                                move_unit = move_unit.item()
                        except Exception:
                            pass
                        if isinstance(move_unit, bytes):
                            move_unit = move_unit.decode("utf-8", errors="ignore")
                        if isinstance(move_unit, str):
                            unit_norm = move_unit.strip().lower()
                            if unit_norm in {"mframe", "m/frame", "m_per_frame", "per_frame"}:
                                dt_safe = np.clip(curr_dt, 1e-6, None)
                                curr_move = curr_move / dt_safe[:, None, None]

                    # 基本 shape 校验（避免混入不同人数的数据）
                    if curr_obs.ndim != 3 or curr_obs.shape[2] < 3:
                        raise ValueError(f"{fname} obs shape invalid: {curr_obs.shape}")
                    if curr_move.ndim != 3 or curr_move.shape[2] != 2:
                        raise ValueError(f"{fname} act_move shape invalid: {curr_move.shape}")
                    n_agents = curr_obs.shape[1]
                    n_att = curr_move.shape[1]
                    if expected_agents is None:
                        expected_agents = n_agents
                        expected_attackers = n_att
                    else:
                        if n_agents != expected_agents or n_att != expected_attackers:
                            raise ValueError(
                                f"{fname} agent/attacker count mismatch: "
                                f"obs={n_agents}, act_move={n_att} vs expected {expected_agents}/{expected_attackers}"
                            )

                    obs_list.append(curr_obs)
                    move_list.append(curr_move)
                    pass_flag_list.append(curr_pass_flag)
                    passer_id_list.append(curr_passer_id)
                    pass_dir_list.append(curr_pass_dir)
                    receiver_id_list.append(curr_receiver_id)
                    dt_list.append(curr_dt)
                    self.file_lengths.append(curr_obs.shape[0])
                    print(f"  > Loaded {fname}: {curr_obs.shape[0]} frames (new format)")
                # 兼容旧格式
                elif 'acts' in data:
                    curr_acts = data['acts']
                    if curr_acts.ndim != 3 or curr_acts.shape[2] < 2:
                        raise ValueError(f"{fname} acts shape invalid: {curr_acts.shape}")
                    if curr_obs.ndim != 3 or curr_obs.shape[2] < 3:
                        raise ValueError(f"{fname} obs shape invalid: {curr_obs.shape}")
                    n_agents = curr_obs.shape[1]
                    n_att = curr_acts.shape[1]
                    if expected_agents is None:
                        expected_agents = n_agents
                        expected_attackers = n_att
                    else:
                        if n_agents != expected_agents or n_att != expected_attackers:
                            raise ValueError(
                                f"{fname} agent/attacker count mismatch: "
                                f"obs={n_agents}, acts={n_att} vs expected {expected_agents}/{expected_attackers}"
                            )
                    obs_list.append(curr_obs)
                    move_list.append(curr_acts[:, :, 0:2])
                    pass_flag_list.append((curr_acts[:, :, 2] > 0.5).any(axis=1).astype(np.float32))
                    passer_id_list.append(np.full(len(curr_obs), -1, dtype=np.int64))
                    pass_dir_list.append(np.full(len(curr_obs), -1, dtype=np.int64))
                    receiver_id_list.append(np.full(len(curr_obs), -1, dtype=np.int64))
                    curr_dt = _normalize_dt(data.get('dt', 0.1), len(curr_obs))
                    dt_list.append(curr_dt)
                    self.file_lengths.append(curr_obs.shape[0])
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
        combined_dt = np.concatenate(dt_list, axis=0)

        # 转为 Tensor
        self.obs = torch.FloatTensor(combined_obs)
        self.move = torch.FloatTensor(combined_move)
        self.pass_flag = torch.FloatTensor(combined_pass_flag)
        self.passer_id = torch.LongTensor(combined_passer_id)
        self.pass_dir = torch.LongTensor(combined_pass_dir)
        self.receiver_id = torch.LongTensor(combined_receiver_id)
        self.dt = torch.FloatTensor(combined_dt)
        self.n_agents = self.obs.shape[1]
        self.n_attackers = self.move.shape[1]

        # 4. 重新计算统计信息 (逻辑与之前一致，但现在是针对所有比赛的总和)
        self.total_frames = len(self.obs)
        self.kick_mask = self.pass_flag > 0.5
        self.n_kicks = self.kick_mask.sum().item()

        # 5. 记录每场比赛的帧范围，用于避免跨场历史堆叠或多步预测
        self.file_start_idxs = []
        self.file_end_idxs = []
        acc = 0
        for n in self.file_lengths:
            self.file_start_idxs.append(acc)
            acc += n
            self.file_end_idxs.append(acc)  # end is exclusive
        if acc != self.total_frames:
            raise RuntimeError("文件长度累积与总帧数不一致，数据拼接可能有误")

        print(f"Dataset Ready! Total Frames: {self.total_frames}, Total Pass Frames: {self.n_kicks}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # 确定当前帧所在文件的范围，避免跨场历史堆叠或多步预测
        # 线性扫描足够快（文件数<=几十），如有性能需求再优化为二分
        file_start = 0
        file_end = self.total_frames
        for s, e in zip(self.file_start_idxs, self.file_end_idxs):
            if s <= idx < e:
                file_start, file_end = s, e
                break

        # 多步目标
        if self.move_horizon == 1:
            move_seq = self.move[idx].unsqueeze(0)
            move_mask = torch.ones(1, dtype=torch.float32)
            dt_seq = self.dt[idx].unsqueeze(0)
        else:
            max_end = min(idx + self.move_horizon, file_end)
            seq = self.move[idx:max_end]
            seq_dt = self.dt[idx:max_end]
            if seq.shape[0] < self.move_horizon:
                pad = torch.zeros(self.move_horizon - seq.shape[0], seq.shape[1], seq.shape[2])
                move_seq = torch.cat([seq, pad], dim=0)
                move_mask = torch.zeros(self.move_horizon, dtype=torch.float32)
                move_mask[:seq.shape[0]] = 1.0
                dt_pad = torch.zeros(self.move_horizon - seq_dt.shape[0], dtype=seq_dt.dtype)
                dt_seq = torch.cat([seq_dt, dt_pad], dim=0)
                if seq_dt.shape[0] > 0:
                    dt_seq[seq_dt.shape[0]:] = dt_seq[seq_dt.shape[0] - 1]
            else:
                move_seq = seq
                move_mask = torch.ones(self.move_horizon, dtype=torch.float32)
                dt_seq = seq_dt

        if self.obs_history == 1:
            return self.obs[idx], move_seq, move_mask, dt_seq, self.pass_flag[idx], self.passer_id[idx], self.pass_dir[idx], self.receiver_id[idx]

        # 堆叠过去 K 帧（不够则用本场第一帧补齐）
        start = max(file_start, idx - (self.obs_history - 1))
        frames = self.obs[start:idx + 1]  # [t, E, 3]

        if frames.shape[0] < self.obs_history:
            pad = frames[0].unsqueeze(0).repeat(self.obs_history - frames.shape[0], 1, 1)
            frames = torch.cat([pad, frames], dim=0)

        n_agents = frames.shape[1]
        stacked = frames.permute(1, 0, 2).reshape(n_agents, self.obs_history * 3)
        return stacked, move_seq, move_mask, dt_seq, self.pass_flag[idx], self.passer_id[idx], self.pass_dir[idx], self.receiver_id[idx]

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
