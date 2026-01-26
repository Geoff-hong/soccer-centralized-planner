import kloppy
from kloppy import metrica
from kloppy.domain import Dimension, MetricPitchDimensions
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================
KICK_SECTORS = 12       
POSSESSION_DIST_THR = 2.0   
POSSESSION_HYSTERESIS = 5   
LOOKAHEAD_FRAMES = 10   
DT = 0.04               

# ==========================================
# 2. SlotAllocator (保持不变)
# ==========================================
class SlotAllocator:
    def __init__(self, initial_frame_row, all_player_ids, team_prefix='home'):
        self.slots = [None] * 11
        self.team_prefix = team_prefix
        self.all_ids = all_player_ids
        active_ids = []
        for pid in self.all_ids:
            if not pd.isna(initial_frame_row[f'{team_prefix}_{pid}_x']):
                active_ids.append(pid)
        for i in range(min(len(active_ids), 11)):
            self.slots[i] = active_ids[i]

    def update(self, row):
        empty_slot_indices = []
        for i, pid in enumerate(self.slots):
            if pid is None:
                empty_slot_indices.append(i)
                continue
            val = row[f'{self.team_prefix}_{pid}_x']
            if pd.isna(val):
                empty_slot_indices.append(i)

        if not empty_slot_indices:
            return self.slots

        candidates = [pid for pid in self.all_ids if pid not in self.slots]
        new_active_players = []
        for pid in candidates:
            if not pd.isna(row[f'{self.team_prefix}_{pid}_x']):
                new_active_players.append(pid)
        
        for new_pid in new_active_players:
            if empty_slot_indices:
                idx = empty_slot_indices.pop(0)
                self.slots[idx] = new_pid 
                
        return self.slots

# ==========================================
# 3. 辅助函数
# ==========================================
def get_kick_sector(dx, dy, n_sectors=12):
    angle = np.arctan2(dy, dx)
    if angle < 0: angle += 2 * np.pi
    sector_size = 2 * np.pi / n_sectors
    return int(angle / sector_size) % n_sectors

# ==========================================
# [关键修改] 有状态的球权管理器 (包含初始化修正)
# ==========================================
class PossessionManager:
    def __init__(self):
        self.current_attacking_team = None 
        self.consecutive_frames_opp = 0
        self.initialized = False

    def update(self, row, home_ids, away_ids):
        ball_x, ball_y = row['ball_x'], row['ball_y']
        
        # 1. 计算最近距离
        min_dist_home = float('inf')
        for pid in home_ids:
            px, py = row[f'home_{pid}_x'], row[f'home_{pid}_y']
            if pd.notna(px):
                d = (px - ball_x)**2 + (py - ball_y)**2
                if d < min_dist_home: min_dist_home = d
        min_dist_home = np.sqrt(min_dist_home)

        min_dist_away = float('inf')
        for pid in away_ids:
            px, py = row[f'away_{pid}_x'], row[f'away_{pid}_y']
            if pd.notna(px):
                d = (px - ball_x)**2 + (py - ball_y)**2
                if d < min_dist_away: min_dist_away = d
        min_dist_away = np.sqrt(min_dist_away)
        
        # === [修正] 第一帧初始化 ===
        if not self.initialized:
            if min_dist_home < min_dist_away:
                self.current_attacking_team = 'Home'
            else:
                self.current_attacking_team = 'Away'
            self.initialized = True
            return self.current_attacking_team
        # ==========================

        home_has_ball = min_dist_home < POSSESSION_DIST_THR
        away_has_ball = min_dist_away < POSSESSION_DIST_THR
        
        if self.current_attacking_team == 'Home':
            if away_has_ball and not home_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            
            if self.consecutive_frames_opp > POSSESSION_HYSTERESIS:
                self.current_attacking_team = 'Away'
                self.consecutive_frames_opp = 0
                
        else: # Current is Away
            if home_has_ball and not away_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
                
            if self.consecutive_frames_opp > POSSESSION_HYSTERESIS:
                self.current_attacking_team = 'Home'
                self.consecutive_frames_opp = 0
                
        return self.current_attacking_team

# ==========================================
# 4. 主处理流程
# ==========================================
def process_match_data(match_id):
    print(f"\n[1/5] Loading Match {match_id} Tracking Data...")
    
    dataset = metrica.load_open_data(match_id=match_id)
    pitch_dim = MetricPitchDimensions(
        x_dim=Dimension(min=0, max=105),
        y_dim=Dimension(min=0, max=68),
        standardized=False
    )
    dataset = dataset.transform(to_pitch_dimensions=pitch_dim)
    df = dataset.to_df()
    
    home_cols = [c for c in df.columns if c.startswith('home_') and c.endswith('_x')]
    away_cols = [c for c in df.columns if c.startswith('away_') and c.endswith('_x')]
    home_ids_all = [c.split('_')[1] for c in home_cols]
    away_ids_all = [c.split('_')[1] for c in away_cols]
    
    # 读取 Event File
    print("[1.5/5] Loading Inferred Events...")
    event_file = f"data/metrica_match{match_id}_inferred_events.csv"
    if not os.path.exists(event_file):
        raise FileNotFoundError(f"Event file {event_file} not found! Please run track2event.py first.")
    df_events = pd.read_csv(event_file)
    
    # 平滑与速度计算
    print("[2/5] Smoothing & Velocity Calculation...")
    coord_cols = home_cols + [c.replace('_x', '_y') for c in home_cols] + \
                 away_cols + [c.replace('_x', '_y') for c in away_cols] + \
                 ['ball_x', 'ball_y']
                 
    df[coord_cols] = df[coord_cols].interpolate(method='linear', limit=5)
    df_smooth = df.copy()
    df_smooth[coord_cols] = df[coord_cols].rolling(window=5, min_periods=1).mean()
    df_vel = df_smooth[coord_cols].diff().fillna(0.0)

    # 构建 Kick Map
    print("[3/5] Mapping Events from CSV...")
    kick_events = {}
    all_passes = df_events[df_events['type'] == 'PASS']
    for _, row in all_passes.iterrows():
        fid = int(row['start_frame'])
        pid = str(row['from_player'])
        team = row['from_team'] 
        kick_events[fid] = {'pid': pid, 'team': team}

    # =========================================================
    # [新增] 自动检测 Period 1 的进攻方向
    # =========================================================
    print("[3.5/5] Detecting Attack Direction...")
    home_attack_dir_p1 = 1 # 默认为 1 (Left->Right)
    
    period1_data = df_smooth[df_smooth['period_id'] == 1]
    if not period1_data.empty:
        # 取前10帧的平均站位，避免开球瞬间波动
        init_frames = period1_data.iloc[:10]
        home_x_vals = []
        for pid in home_ids_all:
            # 取所有Home球员的x坐标
            vals = init_frames[f'home_{pid}_x'].values
            # 过滤NaN
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                home_x_vals.extend(valid_vals)
        
        if len(home_x_vals) > 0:
            avg_home_x = np.mean(home_x_vals)
            # 如果平均位置小于 52.5 (中场)，说明Home在左边防守，往右攻 (L->R)
            # 如果平均位置大于 52.5，说明Home在右边防守，往左攻 (R->L)
            if avg_home_x < 52.5:
                home_attack_dir_p1 = 1  # Standard: Home Attacks Left->Right
                print(" -> Detected Period 1: Home starts on LEFT (Attacks -> Right)")
            else:
                home_attack_dir_p1 = -1 # Non-Standard: Home Attacks Right->Left
                print(" -> Detected Period 1: Home starts on RIGHT (Attacks -> Left)")
    # =========================================================

    # 生成 Dataset
    print("[4/5] Generating Attacker-Centric Tensors...")
    
    obs_list = []
    act_list = []
    
    home_allocator = SlotAllocator(df_smooth.iloc[0], home_ids_all, 'home')
    away_allocator = SlotAllocator(df_smooth.iloc[0], away_ids_all, 'away')
    possession_manager = PossessionManager()
    
    valid_indices = df_smooth.index[:-LOOKAHEAD_FRAMES]
    
    for i, fid in enumerate(valid_indices):
        if i % 1000 == 0: print(f"   Processing frame {fid}/{valid_indices[-1]}...", end='\r')
        
        row_raw = df_smooth.loc[fid]
        row_vel = df_vel.loc[fid]
        period = row_raw['period_id']
        
        # A. 确定谁在进攻
        attacking_team = possession_manager.update(row_raw, home_ids_all, away_ids_all)
        
        # B. Update Slots
        home_slots = home_allocator.update(row_raw)
        away_slots = away_allocator.update(row_raw)
        
        # C. 动态翻转逻辑 (Flip Logic)
        # 目的：让进攻方永远是从 -1 攻向 1 (Left->Right)
        
        # 1. 确定当前半场 Home 的进攻方向 (1=L->R, -1=R->L)
        if period == 1:
            current_home_dir = home_attack_dir_p1
        else:
            current_home_dir = -home_attack_dir_p1 # 下半场交换场地
            
        # 2. 判断是否需要翻转
        # 如果进攻方目前的真实运动方向是 R->L (-1)，我们就需要 flip
        flip = False
        
        if attacking_team == 'Home':
            # Home 正在进攻。
            # 如果 Home 目前方向是 -1 (R->L)，则需要 flip。
            if current_home_dir == -1:
                flip = True
        else:
            # Away 正在进攻。
            # Away 的方向永远和 Home 相反。
            # Away dir = -current_home_dir
            # 如果 Away dir 是 -1 (R->L)，则需要 flip。
            if -current_home_dir == -1:
                flip = True
        
        # D. 设置 Teammates / Opponents
        if attacking_team == 'Home':
            teammate_slots = home_slots
            opponent_slots = away_slots
            team_prefix = 'home'
            opp_prefix = 'away'
        else:
            teammate_slots = away_slots
            opponent_slots = home_slots
            team_prefix = 'away'
            opp_prefix = 'home'

        # --- Normalization Helper ---
        def norm_pos_x(val):
            # 原始 0..105 -> -1..1
            n = (val - 52.5) / 52.5 
            # 如果 flip=True，说明当前是 R->L 进攻 (正->负)，
            # 乘以 -1 变成 L->R (负->正)
            return -n if flip else n

        def norm_pos_y(val):
            n = (val - 34.0) / 34.0
            return -n if flip else n
            
        def norm_vel(val):
            return -val if flip else val

        # --- E. Build Obs ---
        current_obs = []
        
        # 1. Teammates
        for pid in teammate_slots:
            if pid is None:
                current_obs.append([0, 0, 0])
                continue
            
            px = row_raw[f'{team_prefix}_{pid}_x']
            py = row_raw[f'{team_prefix}_{pid}_y']
            
            if pd.isna(px):
                current_obs.append([0, 0, 0])
            else:
                ball_x, ball_y = row_raw['ball_x'], row_raw['ball_y']
                dist = np.sqrt((px-ball_x)**2 + (py-ball_y)**2)
                has_ball = 1.0 if dist < POSSESSION_DIST_THR else 0.0
                current_obs.append([norm_pos_x(px), norm_pos_y(py), has_ball])

        # 2. Opponents
        for pid in opponent_slots:
            if pid is None:
                current_obs.append([0, 0, 0])
                continue
            px = row_raw[f'{opp_prefix}_{pid}_x']
            py = row_raw[f'{opp_prefix}_{pid}_y']
            if pd.isna(px):
                current_obs.append([0, 0, 0])
            else:
                ball_x, ball_y = row_raw['ball_x'], row_raw['ball_y']
                dist = np.sqrt((px-ball_x)**2 + (py-ball_y)**2)
                has_ball = 1.0 if dist < POSSESSION_DIST_THR else 0.0
                current_obs.append([norm_pos_x(px), norm_pos_y(py), has_ball])
                
        # 3. Ball
        ball_x, ball_y = row_raw['ball_x'], row_raw['ball_y']
        if pd.isna(ball_x):
            current_obs.append([0, 0, 0])
        else:
            current_obs.append([norm_pos_x(ball_x), norm_pos_y(ball_y), 0.0])
            
        obs_list.append(current_obs)
        
        # --- F. Build Acts ---
        current_acts = []
        kick_info = kick_events.get(fid)
        
        for pid in teammate_slots:
            if pid is None:
                current_acts.append([0, 0, 0, 0])
                continue
                
            vx_raw = row_vel[f'{team_prefix}_{pid}_x']
            vy_raw = row_vel[f'{team_prefix}_{pid}_y']
            
            act_vx = norm_vel(vx_raw)
            act_vy = norm_vel(vy_raw)
            
            trigger = 0.0
            kick_class = 0.0
            
            if kick_info and str(kick_info['pid']) == str(pid) and kick_info['team'] == attacking_team:
                trigger = 1.0
                future_row = df_smooth.loc[fid + LOOKAHEAD_FRAMES]
                curr_row = row_raw
                
                if pd.notna(future_row['ball_x']):
                    dx = future_row['ball_x'] - curr_row['ball_x']
                    dy = future_row['ball_y'] - curr_row['ball_y']
                    # 关键：dx, dy 也要 Flip，确保 Action 方向对齐
                    kick_class = float(get_kick_sector(norm_vel(dx), norm_vel(dy), KICK_SECTORS))
            
            current_acts.append([act_vx, act_vy, trigger, kick_class])
            
        act_list.append(current_acts)

    # 保存
    obs_array = np.array(obs_list, dtype=np.float32)
    act_array = np.array(act_list, dtype=np.float32)
    
    print(f"\n[5/5] Done. Match {match_id}")
    print(f"   Obs Shape: {obs_array.shape}")
    os.makedirs("../data", exist_ok=True)
    output_file = f"data/metrica_match{match_id}_train_attacker_centric.npz"
    np.savez_compressed(output_file, obs=obs_array, acts=act_array)
    print(f"   Saved to {output_file}")

if __name__ == "__main__":
    process_match_data(match_id=2)