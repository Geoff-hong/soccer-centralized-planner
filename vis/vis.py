import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import math
import sys
import os
import glob
from collections import deque

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 导入真正的模型
from train.model import SoccerPolicy 

# ===========================
# 配置参数
# ===========================
class SimConfig:
    # 场地参数 (米)
    WIDTH = 105.0
    HEIGHT = 68.0
    TRAIN_DT = 0.04      # 训练数据的时间步长（默认）
    DT = 0.04            # 仿真步长（建议与训练保持一致）
    
    # 物理参数
    PLAYER_SPEED_SCALE = 1.0 
    BALL_FRICTION = 0.96    
    KICK_POWER = 25.0        
    DRIBBLE_DIST = 2.0       
    KICK_TRIGGER_THR = 0.6   
    MAX_PLAYER_SPEED = 8.0   # m/s，用于限制异常速度

    # 球权判定（与 data_processing 保持一致）
    POSSESSION_DIST_THR = 2.0
    POSSESSION_HYSTERESIS = 5
    
    # 模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    MODEL_PATH = os.path.join(project_root, "checkpoints", "best_model.pth")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化位置 (从 npz 加载)
    DATA_DIR = os.path.join(project_root, "data")
    INIT_NPZ_PATH = "data/metrica_match2_train_attacker_centric.npz"  # None 时自动选择 data/ 下第一个 .npz
    INIT_FRAME_WINDOW = 5  # 取前几帧做平均初始化
    OBS_HISTORY = 5         # 与训练保持一致

    # --- GIF 录制配置 (新增) ---
    SAVE_GIF = True            # 是否保存GIF
    GIF_FILENAME = "soccer_match.gif"
    TOTAL_FRAMES = 200         # 录制总帧数
    FPS = 15                   # GIF的播放帧率

    # --- 日志 ---
    SPEED_LOG_INTERVAL = 10    # 每隔多少帧打印一次进攻方速度统计

class PossessionManager:
    def __init__(self, dist_thr, hysteresis):
        self.dist_thr = dist_thr
        self.hysteresis = hysteresis
        self.current_attacking_team = None
        self.consecutive_frames_opp = 0
        self.initialized = False

    def reset(self, attacking_team="Attack"):
        self.current_attacking_team = attacking_team
        self.consecutive_frames_opp = 0
        self.initialized = True

    def update(self, attackers_pos, defenders_pos, ball_pos):
        dists_att = np.linalg.norm(attackers_pos - ball_pos, axis=1)
        dists_def = np.linalg.norm(defenders_pos - ball_pos, axis=1)
        min_dist_att = float(np.min(dists_att)) if len(dists_att) > 0 else float("inf")
        min_dist_def = float(np.min(dists_def)) if len(dists_def) > 0 else float("inf")

        if not self.initialized:
            self.current_attacking_team = "Attack" if min_dist_att < min_dist_def else "Defense"
            self.initialized = True
            return self.current_attacking_team

        att_has_ball = min_dist_att < self.dist_thr
        def_has_ball = min_dist_def < self.dist_thr

        if self.current_attacking_team == "Attack":
            if def_has_ball and not att_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            if self.consecutive_frames_opp > self.hysteresis:
                self.current_attacking_team = "Defense"
                self.consecutive_frames_opp = 0
        else:
            if att_has_ball and not def_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            if self.consecutive_frames_opp > self.hysteresis:
                self.current_attacking_team = "Attack"
                self.consecutive_frames_opp = 0

        return self.current_attacking_team

class SoccerSimulation:
    def __init__(self):
        self.cfg = SimConfig()
        self.device = torch.device(self.cfg.DEVICE)
        
        # 1. 加载模型
        self.model = self._load_model()

        # 1.5 球权管理器
        self.possession_manager = PossessionManager(
            dist_thr=self.cfg.POSSESSION_DIST_THR,
            hysteresis=self.cfg.POSSESSION_HYSTERESIS
        )
        
        # 2. 初始化比赛状态
        self.reset_game()

    def _load_model(self):
        print(f"Loading model logic...")
        # 使用与训练时相同的配置
        model = SoccerPolicy(
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.3,
            input_dim=3 * self.cfg.OBS_HISTORY
        ).to(self.device)
        try:
            state_dict = torch.load(self.cfg.MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {self.cfg.MODEL_PATH}")
        except FileNotFoundError:
            print("Warning: 未找到模型文件，将使用随机初始化的模型运行（仅用于调试代码逻辑）。")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Warning: 模型结构不匹配，将使用随机初始化的模型运行（仅用于调试代码逻辑）。")
        
        model.eval()
        return model

    def _normalize_positions(self, positions):
        center = np.array([self.cfg.WIDTH / 2.0, self.cfg.HEIGHT / 2.0], dtype=np.float32)
        scale = np.array([self.cfg.WIDTH / 2.0, self.cfg.HEIGHT / 2.0], dtype=np.float32)
        return (positions - center) / scale

    def _denormalize_positions(self, norm_positions):
        center = np.array([self.cfg.WIDTH / 2.0, self.cfg.HEIGHT / 2.0], dtype=np.float32)
        scale = np.array([self.cfg.WIDTH / 2.0, self.cfg.HEIGHT / 2.0], dtype=np.float32)
        return norm_positions * scale + center

    def _load_initial_positions(self):
        npz_path = self.cfg.INIT_NPZ_PATH
        if npz_path is None:
            candidates = sorted(glob.glob(os.path.join(self.cfg.DATA_DIR, "*.npz")))
            if not candidates:
                return None
            npz_path = candidates[0]
        try:
            data = np.load(npz_path)
            obs = data["obs"]
            if "dt" in data:
                dt_val = float(data["dt"])
                self.cfg.TRAIN_DT = dt_val
                self.cfg.DT = dt_val
                print(f"[Init] Loaded dt={dt_val} from {npz_path}")
        except Exception as e:
            print(f"[Init] Failed to load npz {npz_path}: {e}")
            return None

        if obs.ndim != 3 or obs.shape[1] < 23 or obs.shape[2] < 2:
            print(f"[Init] Unexpected obs shape in {npz_path}: {obs.shape}")
            return None

        window = min(self.cfg.INIT_FRAME_WINDOW, obs.shape[0])
        frame = obs[:window].mean(axis=0)

        att_norm = frame[:11, :2]
        def_norm = frame[11:22, :2]
        ball_norm = frame[22, :2]

        attackers_pos = self._denormalize_positions(att_norm)
        defenders_pos = self._denormalize_positions(def_norm)
        ball_pos = self._denormalize_positions(ball_norm.reshape(1, 2))[0]

        attackers_pos = np.clip(attackers_pos, [0, 0], [self.cfg.WIDTH, self.cfg.HEIGHT])
        defenders_pos = np.clip(defenders_pos, [0, 0], [self.cfg.WIDTH, self.cfg.HEIGHT])
        ball_pos = np.clip(ball_pos, [0, 0], [self.cfg.WIDTH, self.cfg.HEIGHT])

        print(f"[Init] Loaded initial positions from {npz_path}")
        return attackers_pos.astype(np.float32), defenders_pos.astype(np.float32), ball_pos.astype(np.float32)

    def _assign_ball_owner(self):
        dists_att = np.linalg.norm(self.attackers_pos - self.ball_pos, axis=1)
        nearest_att = int(np.argmin(dists_att))
        if dists_att[nearest_att] < self.cfg.DRIBBLE_DIST:
            return nearest_att
        return None

    def reset_game(self):
        """初始化全场站位"""
        init_data = self._load_initial_positions()
        if init_data is not None:
            self.attackers_pos, self.defenders_pos, self.ball_pos = init_data
            self.defenders_home = self.defenders_pos.copy()
        else:
            self.attackers_pos = np.array([
                [52, 34], # 0: 开球
                [45, 30], [45, 38], [40, 20], [40, 48],
                [30, 15], [30, 34], [30, 53],
                [20, 25], [20, 43], 
                [5, 34]   # 门将
            ], dtype=np.float32)
            
            self.defenders_home = np.array([
                [53, 34], # 前锋
                [60, 30], [60, 38], [65, 20], [65, 48],
                [75, 15], [75, 34], [75, 53],
                [85, 25], [85, 43],
                [100, 34]
            ], dtype=np.float32)
            self.defenders_pos = self.defenders_home.copy()
            self.ball_pos = np.array([52.0, 34.0], dtype=np.float32)

        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.ball_owner_idx = self._assign_ball_owner()
        self.last_vel_cmds = None
        self.last_triggers = None

        self.possession_manager.reset(attacking_team="Attack")
        # 初始化观测历史
        self.obs_history = deque(maxlen=self.cfg.OBS_HISTORY)
        obs_now = self._build_obs()
        for _ in range(self.cfg.OBS_HISTORY):
            self.obs_history.append(obs_now.copy())
        # print(">>> Game Reset!") # 录制模式下减少print以防刷屏

    def _get_kick_vector(self, class_idx):
        angle_deg = class_idx * 30.0
        angle_rad = math.radians(angle_deg)
        return np.array([math.cos(angle_rad), math.sin(angle_rad)])

    def _build_obs(self):
        att_norm = self._normalize_positions(self.attackers_pos)
        def_norm = self._normalize_positions(self.defenders_pos)
        ball_norm = self._normalize_positions(self.ball_pos.reshape(1, 2))[0]

        att_has = (np.linalg.norm(self.attackers_pos - self.ball_pos, axis=1) < self.cfg.POSSESSION_DIST_THR).astype(np.float32)
        def_has = (np.linalg.norm(self.defenders_pos - self.ball_pos, axis=1) < self.cfg.POSSESSION_DIST_THR).astype(np.float32)

        att_feat = np.concatenate([att_norm, att_has.reshape(11, 1)], axis=1)
        def_feat = np.concatenate([def_norm, def_has.reshape(11, 1)], axis=1)
        ball_feat = np.array([[ball_norm[0], ball_norm[1], 0.0]], dtype=np.float32)
        return np.concatenate([att_feat, def_feat, ball_feat], axis=0)

    def _get_stacked_obs(self, obs_now):
        self.obs_history.append(obs_now)
        if len(self.obs_history) < self.cfg.OBS_HISTORY:
            pad_count = self.cfg.OBS_HISTORY - len(self.obs_history)
            for _ in range(pad_count):
                self.obs_history.appendleft(obs_now.copy())
        obs_stack = np.stack(list(self.obs_history), axis=0)  # [K, 23, 3]
        return obs_stack.transpose(1, 0, 2).reshape(23, self.cfg.OBS_HISTORY * 3)

    def get_model_action(self):
        obs_now = self._build_obs()
        obs_np = self._get_stacked_obs(obs_now)
        obs_tensor = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            p_vel, p_pass, p_passer, p_dir, p_receiver = self.model(obs_tensor)
            
        p_vel = p_vel.cpu().numpy()[0]
        pass_prob = torch.sigmoid(p_pass).cpu().numpy()[0, 0]
        passer_id = int(torch.argmax(p_passer, dim=1).cpu().numpy()[0])
        pass_dir = int(torch.argmax(p_dir, dim=1).cpu().numpy()[0])
        receiver_id = int(torch.argmax(p_receiver, dim=1).cpu().numpy()[0])
        
        return p_vel, pass_prob, passer_id, pass_dir, receiver_id

    def step(self):
        """执行单步仿真"""
        vel_cmds, pass_prob, passer_id, pass_dir, _ = self.get_model_action()
        self.last_triggers = pass_prob

        # 限制最大速度（防止异常发散）: vel_cmds 为 m/s
        speeds = np.linalg.norm(vel_cmds, axis=1)
        if np.any(speeds > self.cfg.MAX_PLAYER_SPEED):
            scale = np.minimum(1.0, self.cfg.MAX_PLAYER_SPEED / (speeds + 1e-6))
            vel_cmds = vel_cmds * scale.reshape(-1, 1)
        self.last_vel_cmds = vel_cmds
        
        # 进攻方移动
        self.attackers_pos += vel_cmds * self.cfg.PLAYER_SPEED_SCALE * self.cfg.DT
        
        # 防守方移动 (Heuristic)
        dists_def = np.linalg.norm(self.defenders_pos - self.ball_pos, axis=1)
        closest_indices = np.argsort(dists_def)[:3]
        def_speed = 4.0  # m/s
        
        for i in range(11):
            target = self.ball_pos if i in closest_indices else self.defenders_home[i]
            direction = target - self.defenders_pos[i]
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                vel = (direction / dist) * def_speed
                self.defenders_pos[i] += vel * self.cfg.DT

        self.attackers_pos = np.clip(self.attackers_pos, [0,0], [self.cfg.WIDTH, self.cfg.HEIGHT])
        self.defenders_pos = np.clip(self.defenders_pos, [0,0], [self.cfg.WIDTH, self.cfg.HEIGHT])

        # 球的逻辑
        if self.ball_owner_idx is not None:
            idx = self.ball_owner_idx
            dist_to_owner = np.linalg.norm(self.ball_pos - self.attackers_pos[idx])
            if dist_to_owner > self.cfg.DRIBBLE_DIST * 1.5:
                self.ball_owner_idx = None
            else:
                if pass_prob > self.cfg.KICK_TRIGGER_THR and passer_id == idx:
                    kick_vec = self._get_kick_vector(pass_dir)
                    self.ball_vel = kick_vec * self.cfg.KICK_POWER
                    self.ball_owner_idx = None
                else:
                    player_disp = vel_cmds[idx] * self.cfg.PLAYER_SPEED_SCALE * self.cfg.DT
                    speed = np.linalg.norm(player_disp)
                    offset = np.array([0.0, 0.0])
                    if speed > 1e-3:
                        offset = (player_disp / speed) * 0.5
                    self.ball_pos = self.attackers_pos[idx] + offset
                    self.ball_vel = vel_cmds[idx] * self.cfg.PLAYER_SPEED_SCALE

        # 如果没有持球人，但模型想传球，且 passer 接近球，允许起脚
        if self.ball_owner_idx is None and pass_prob > self.cfg.KICK_TRIGGER_THR:
            dist_passer = np.linalg.norm(self.attackers_pos[passer_id] - self.ball_pos)
            if dist_passer < self.cfg.DRIBBLE_DIST:
                kick_vec = self._get_kick_vector(pass_dir)
                self.ball_vel = kick_vec * self.cfg.KICK_POWER
                self.ball_owner_idx = None

        if self.ball_owner_idx is None:
            self.ball_pos += self.ball_vel * self.cfg.DT
            self.ball_vel *= self.cfg.BALL_FRICTION

        # 球权判定（避免被近距离防守立即重置）
        possession = self.possession_manager.update(self.attackers_pos, self.defenders_pos, self.ball_pos)
        if possession == "Defense":
            self.reset_game()
            return

        if self.ball_owner_idx is None and possession == "Attack":
            dists_att = np.linalg.norm(self.attackers_pos - self.ball_pos, axis=1)
            nearest_att = int(np.argmin(dists_att))
            if dists_att[nearest_att] < self.cfg.DRIBBLE_DIST:
                self.ball_owner_idx = nearest_att
        
        if not (0 <= self.ball_pos[0] <= self.cfg.WIDTH and 0 <= self.ball_pos[1] <= self.cfg.HEIGHT):
            # print(">>> Out of Bounds.")
            self.reset_game()
            return

    def animate(self):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.set_facecolor('#4B8B3B')
        ax.set_xlim(0, self.cfg.WIDTH)
        ax.set_ylim(0, self.cfg.HEIGHT)
        
        # 绘制场地
        ax.axvline(x=self.cfg.WIDTH/2, color='white', linestyle='--', alpha=0.5)
        ax.add_patch(Rectangle((0, 0), self.cfg.WIDTH, self.cfg.HEIGHT, fill=False, ec='white', lw=2))
        ax.add_patch(Rectangle((self.cfg.WIDTH-16.5, self.cfg.HEIGHT/2-20), 16.5, 40, fill=False, ec='white')) 
        
        att_dots = ax.scatter([], [], c='red', s=80, edgecolors='white', label='AI Attackers')
        def_dots = ax.scatter([], [], c='blue', s=80, edgecolors='white', label='Defenders')
        ball_dot = ax.scatter([], [], c='white', s=50, edgecolors='black', zorder=10, label='Ball')
        quiver = ax.quiver([], [], [], [], color='yellow', scale=20, width=0.003)
        
        ax.legend(loc='upper left')
        title_text = ax.text(0.5, 1.02, "Simulation Initializing...", transform=ax.transAxes, ha='center', fontsize=12)

        def init():
            return att_dots, def_dots, ball_dot, quiver, title_text

        def update(frame):
            self.step()
            
            att_dots.set_offsets(self.attackers_pos)
            def_dots.set_offsets(self.defenders_pos)
            ball_dot.set_offsets(self.ball_pos.reshape(1, -1))
            
            if self.ball_owner_idx is not None and self.last_vel_cmds is not None:
                owner_pos = self.attackers_pos[self.ball_owner_idx]
                owner_vel = self.last_vel_cmds[self.ball_owner_idx]
                quiver.set_offsets(owner_pos.reshape(1, -1))
                quiver.set_UVC(owner_vel[0], owner_vel[1])
            else:
                quiver.set_UVC(0, 0)
            
            # 可以在标题显示当前帧数
            if self.cfg.SAVE_GIF:
                title_text.set_text(f"Recording Frame: {frame}/{self.cfg.TOTAL_FRAMES}")
                # 打印进度 (可选)
                if frame % 10 == 0:
                    print(f"Rendering frame {frame}/{self.cfg.TOTAL_FRAMES}...", end='\r')
            else:
                title_text.set_text("Live Simulation")

            # 进攻方速度统计 (m/s)
            if self.last_vel_cmds is not None and self.cfg.SPEED_LOG_INTERVAL > 0:
                if frame % self.cfg.SPEED_LOG_INTERVAL == 0:
                    speeds_mps = np.linalg.norm(self.last_vel_cmds, axis=1) * self.cfg.PLAYER_SPEED_SCALE
                    avg_speed = float(np.mean(speeds_mps))
                    max_speed = float(np.max(speeds_mps))
                    print(f"[Frame {frame}] Attacker speed avg={avg_speed:.2f} m/s, max={max_speed:.2f} m/s")

            return att_dots, def_dots, ball_dot, quiver, title_text

        # -----------------------------
        # 修改点：GIF 保存逻辑
        # -----------------------------
        if self.cfg.SAVE_GIF:
            print(f"Start Recording {self.cfg.TOTAL_FRAMES} frames...")
            # frames 参数必须是一个具体的数字或迭代器，不能是 None
            ani = animation.FuncAnimation(fig, update, frames=self.cfg.TOTAL_FRAMES, init_func=init, blit=False)
            
            # 使用 PillowWriter 保存 GIF (无需安装 ffmpeg)
            writer = animation.PillowWriter(fps=self.cfg.FPS)
            ani.save(self.cfg.GIF_FILENAME, writer=writer)
            print(f"\nDone! GIF saved to {self.cfg.GIF_FILENAME}")
        else:
            # 原始显示模式
            print("Starting Live Animation...")
            ani = animation.FuncAnimation(fig, update, frames=None, init_func=init, interval=30, blit=False)
            plt.show()

if __name__ == "__main__":
    sim = SoccerSimulation()
    sim.animate()
