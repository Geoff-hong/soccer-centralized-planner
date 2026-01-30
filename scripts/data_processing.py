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
POSSESSION_DIST_THR = 3.0   # 与 track2event 对齐
POSSESSION_HYSTERESIS = 5   
LOOKAHEAD_FRAMES = 10   
DT = 0.04               
KICK_EXPAND_FRAMES = 2   # 传球触发扩展到前后几帧
FUTURE_PASS_FRAMES = 25  # 约 1 秒后的球方向/接球人标签 (25 * 0.04s)

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
# 4. PossessionManager (保持不变)
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
        
        # === 初始化 ===
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
# 5. 主处理流程
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
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    
    event_file = os.path.join(data_dir, f"metrica_match{match_id}_inferred_events.csv")
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
        to_pid = str(row['to_player'])
        to_team = row['to_team']
        for delta in range(-KICK_EXPAND_FRAMES, KICK_EXPAND_FRAMES + 1):
            f = fid + delta
            if f < 0:
                continue
            kick_events.setdefault(f, []).append({
                'pid': pid,
                'team': team,
                'to_pid': to_pid,
                'to_team': to_team
            })

    # =========================================================
    # 检测 Period 1 的进攻方向
    # =========================================================
    print("[3.5/5] Detecting Attack Direction...")
    home_attack_dir_p1 = 1 
    
    period1_data = df_smooth[df_smooth['period_id'] == 1]
    if not period1_data.empty:
        init_frames = period1_data.iloc[:10]
        home_x_vals = []
        for pid in home_ids_all:
            vals = init_frames[f'home_{pid}_x'].values
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                home_x_vals.extend(valid_vals)
        
        if len(home_x_vals) > 0:
            avg_home_x = np.mean(home_x_vals)
            if avg_home_x < 52.5:
                home_attack_dir_p1 = 1  
                print(" -> Detected Period 1: Home starts on LEFT (Attacks -> Right)")
            else:
                home_attack_dir_p1 = -1 
                print(" -> Detected Period 1: Home starts on RIGHT (Attacks -> Left)")

    # 生成 Dataset
    print("[4/5] Generating Attacker-Centric Tensors (With Filtering)...")
    
    obs_list = []
    move_list = []
    pass_flag_list = []
    passer_id_list = []
    pass_dir_list = []
    receiver_id_list = []
    
    # 我们需要找到第一个球有效的帧来初始化Allocator
    # 为了简单起见，这里还是用第一帧初始化，后续动态调整
    home_allocator = SlotAllocator(df_smooth.iloc[0], home_ids_all, 'home')
    away_allocator = SlotAllocator(df_smooth.iloc[0], away_ids_all, 'away')
    possession_manager = PossessionManager()
    
    valid_indices = df_smooth.index[:-max(LOOKAHEAD_FRAMES, FUTURE_PASS_FRAMES)]
    
    # 统计被过滤的帧数
    filtered_count = 0
    filtered_ball_nan = 0
    filtered_ball_oob = 0
    total_count = 0

    for i, fid in enumerate(valid_indices):
        if i % 1000 == 0: print(f"   Processing frame {fid}/{valid_indices[-1]}...", end='\r')
        
        row_raw = df_smooth.loc[fid]
        row_vel = df_vel.loc[fid]

        # 预读取事件（用于必要时纠正持球方）
        kick_info_list = kick_events.get(fid, [])
        
        # =========================================================
        # [关键修改] 数据清洗与过滤
        # =========================================================
        ball_x, ball_y = row_raw['ball_x'], row_raw['ball_y']
        
        # 1. 检查 NaN (球没被追踪到)
        if pd.isna(ball_x) or pd.isna(ball_y):
            filtered_count += 1
            filtered_ball_nan += 1
            continue
            
        # 2. 检查球是否在场内 (或者稍微有一点点宽容度，比如角球区)
        # Metrica Pitch: 105 x 68
        # 我们允许 +/- 2米的误差，防止角球或边线球被误删
        if not (-2.0 <= ball_x <= 107.0 and -2.0 <= ball_y <= 70.0):
            filtered_count += 1
            filtered_ball_oob += 1
            continue
            
        total_count += 1
        # =========================================================

        period = row_raw['period_id']
        
        # B. 确定谁在进攻
        attacking_team = possession_manager.update(row_raw, home_ids_all, away_ids_all)
        if kick_info_list:
            event_team = kick_info_list[0].get('team')
            if event_team in ['Home', 'Away'] and event_team != attacking_team:
                attacking_team = event_team
                possession_manager.current_attacking_team = event_team
                possession_manager.consecutive_frames_opp = 0
        
        # C. Update Slots
        home_slots = home_allocator.update(row_raw)
        away_slots = away_allocator.update(row_raw)
        
        # D. 动态翻转逻辑 (Flip Logic)
        if period == 1:
            current_home_dir = home_attack_dir_p1
        else:
            current_home_dir = -home_attack_dir_p1 
            
        flip = False
        if attacking_team == 'Home':
            if current_home_dir == -1: flip = True
        else:
            if -current_home_dir == -1: flip = True
        
        # E. 设置 Teammates / Opponents
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
            n = (val - 52.5) / 52.5 
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
                dist = np.sqrt((px-ball_x)**2 + (py-ball_y)**2)
                has_ball = 1.0 if dist < POSSESSION_DIST_THR else 0.0
                current_obs.append([norm_pos_x(px), norm_pos_y(py), has_ball])
                
        # 3. Ball (现在一定不是NaN了)
        current_obs.append([norm_pos_x(ball_x), norm_pos_y(ball_y), 0.0])
            
        obs_list.append(current_obs)
        
        # --- F. Build Movement Acts (m/s) ---
        current_moves = []
        for pid in teammate_slots:
            if pid is None:
                current_moves.append([0, 0])
                continue
            vx_raw = row_vel[f'{team_prefix}_{pid}_x']
            vy_raw = row_vel[f'{team_prefix}_{pid}_y']
            vx_ms = vx_raw / DT
            vy_ms = vy_raw / DT
            current_moves.append([norm_vel(vx_ms), norm_vel(vy_ms)])
        move_list.append(current_moves)

        # --- G. Build Pass Labels (Global) ---
        pass_flag = 0
        passer_slot = -1
        pass_dir = -1
        receiver_slot = -1

        if kick_info_list:
            for kick_info in kick_info_list:
                pid = kick_info['pid']
                if pid not in teammate_slots:
                    continue
                slot_idx = teammate_slots.index(pid)

                px = row_raw[f'{team_prefix}_{pid}_x']
                py = row_raw[f'{team_prefix}_{pid}_y']
                if pd.isna(px) or pd.isna(py):
                    continue

                pass_flag = 1
                passer_slot = slot_idx

                # 方向：使用未来 1 秒球的位置作为“意图”方向
                future_row = df_smooth.loc[fid + FUTURE_PASS_FRAMES]
                if pd.notna(future_row['ball_x']):
                    dx = future_row['ball_x'] - row_raw['ball_x']
                    dy = future_row['ball_y'] - row_raw['ball_y']
                    pass_dir = int(get_kick_sector(norm_vel(dx), norm_vel(dy), KICK_SECTORS))
                # 接球人：如果传球方一致，映射到当前 slots
                if kick_info['to_team'] == attacking_team:
                    to_pid = kick_info['to_pid']
                    if to_pid in teammate_slots:
                        receiver_slot = teammate_slots.index(to_pid)
                break

        pass_flag_list.append(pass_flag)
        passer_id_list.append(passer_slot)
        pass_dir_list.append(pass_dir)
        receiver_id_list.append(receiver_slot)

    # 保存
    obs_array = np.array(obs_list, dtype=np.float32)
    move_array = np.array(move_list, dtype=np.float32)
    pass_flag_array = np.array(pass_flag_list, dtype=np.float32)
    passer_id_array = np.array(passer_id_list, dtype=np.int64)
    pass_dir_array = np.array(pass_dir_list, dtype=np.int64)
    receiver_id_array = np.array(receiver_id_list, dtype=np.int64)
    
    print(f"\n[5/5] Done. Match {match_id}")
    print(f"   Total Valid Frames: {total_count}")
    print(f"   Filtered Frames: {filtered_count} (NaN or Out-of-Bounds)")
    print(f"     - Ball NaN: {filtered_ball_nan}")
    print(f"     - Ball OOB: {filtered_ball_oob}")
    print(f"   Obs Shape: {obs_array.shape}")
    print(f"   Move Shape: {move_array.shape}")
    print(f"   Pass Frames: {int(pass_flag_array.sum())}")
    
    # 使用之前定义的 data_dir
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, f"metrica_match{match_id}_train_attacker_centric.npz")
    np.savez_compressed(
        output_file,
        obs=obs_array,
        act_move=move_array,
        pass_flag=pass_flag_array,
        passer_id=passer_id_array,
        pass_dir=pass_dir_array,
        receiver_id=receiver_id_array,
        dt=np.array(DT, dtype=np.float32),
    )
    print(f"   Saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Metrica match into attacker-centric dataset.")
    parser.add_argument("--match_id", type=int, default=2, help="Metrica match id (default: 2)")
    args = parser.parse_args()
    process_match_data(match_id=args.match_id)
