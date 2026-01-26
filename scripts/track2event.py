import numpy as np
import pandas as pd
from kloppy import metrica
import warnings

warnings.filterwarnings('ignore')

# ============================
# 参数配置
# ============================
POSSESSION_DIST = 2.0     
MIN_PASS_FRAMES = 2       
MIN_POSSESSION_FRAMES = 2 

def generate_event_data_from_tracking(tracking_df):
    print("--- Starting Event Generation ---")
    
    # 1. 准备数据
    ball_x = tracking_df['ball_x']
    ball_y = tracking_df['ball_y']
    
    home_cols = [c for c in tracking_df.columns if c.startswith('home_') and c.endswith('_x')]
    away_cols = [c for c in tracking_df.columns if c.startswith('away_') and c.endswith('_x')]
    home_ids = [c.split('_')[1] for c in home_cols]
    away_ids = [c.split('_')[1] for c in away_cols]
    
    all_player_ids = [f"Home_{pid}" for pid in home_ids] + [f"Away_{pid}" for pid in away_ids]
    
    # 2. 计算距离矩阵 (Fill NaN with Inf)
    distances = []
    for pid in home_ids:
        d = np.sqrt((tracking_df[f'home_{pid}_x'] - ball_x)**2 + (tracking_df[f'home_{pid}_y'] - ball_y)**2)
        distances.append(d.values)
    for pid in away_ids:
        d = np.sqrt((tracking_df[f'away_{pid}_x'] - ball_x)**2 + (tracking_df[f'away_{pid}_y'] - ball_y)**2)
        distances.append(d.values)
    
    dist_matrix = np.array(distances).T
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)
    
    # 3. 确定控球者
    min_dist_idx = np.argmin(dist_matrix, axis=1)
    min_dists = np.min(dist_matrix, axis=1)
    
    possessor_array = np.full(len(tracking_df), -1, dtype=int)
    has_possession_mask = min_dists < POSSESSION_DIST
    possessor_array[has_possession_mask] = min_dist_idx[has_possession_mask]
    
    # 平滑
    possessor_series = pd.Series(possessor_array)
    possessor_filled = possessor_series.replace(-1, np.nan).ffill(limit=5).fillna(-1).astype(int)
    possessor_list = possessor_filled.tolist()
    
    # 4. 提取片段
    segments = []
    current_seg_pid = -1
    current_seg_start = 0
    
    for i, pid in enumerate(possessor_list):
        if i == 0:
            current_seg_pid = pid
            continue
        if pid != current_seg_pid:
            if current_seg_pid != -1:
                if (i - current_seg_start) >= MIN_POSSESSION_FRAMES:
                    segments.append({
                        'player_idx': current_seg_pid,
                        'start': current_seg_start,
                        'end': i - 1
                    })
            current_seg_pid = pid
            current_seg_start = i

    # 5. 生成事件
    pass_events = []
    for i in range(len(segments) - 1):
        seg_from = segments[i]
        seg_to = segments[i+1]
        
        flight_start = seg_from['end']
        flight_end = seg_to['start']
        duration = flight_end - flight_start
        
        if duration < 1: continue     
        if duration > 200: continue   
        
        p_from_name = all_player_ids[seg_from['player_idx']]
        p_to_name = all_player_ids[seg_to['player_idx']]
        
        team_from, pid_from = p_from_name.split('_')
        team_to, pid_to = p_to_name.split('_')
        
        # 过滤自己传自己 (Dribble)
        if pid_from == pid_to and team_from == team_to:
            continue
            
        event_type = "PASS" if team_from == team_to else "TURNOVER"
        
        pass_events.append({
            'start_frame': flight_start,
            'end_frame': flight_end,
            'type': event_type,
            'from_team': team_from,
            'from_player': pid_from,
            'to_team': team_to,
            'to_player': pid_to,
            'duration_frames': duration
        })

    cols = ['start_frame', 'end_frame', 'type', 'from_team', 'from_player', 'to_team', 'to_player', 'duration_frames']
    return pd.DataFrame(pass_events, columns=cols)

if __name__ == "__main__":
    match_id = 2
    print(f"Loading Match {match_id}...")
    dataset = metrica.load_open_data(match_id=match_id)
    
    from kloppy.domain import Dimension, MetricPitchDimensions
    try:
        dataset = dataset.transform(to_pitch_dimensions=MetricPitchDimensions(
            x_dim=Dimension(min=0, max=105),
            y_dim=Dimension(min=0, max=68),
            standardized=False
        ))
    except TypeError:
        dataset = dataset.transform(to_pitch_dimensions=MetricPitchDimensions(
            x_dim=Dimension(min=0, max=105),
            y_dim=Dimension(min=0, max=68)
        ))
    
    df_tracking = dataset.to_df()
    df_events = generate_event_data_from_tracking(df_tracking)
    
    # === 这里的修改：只保存我们关心的 Home Pass 数据用于训练 ===
    # 你也可以保存所有数据，之后再筛选
    import os
    os.makedirs("../data", exist_ok=True)
    final_output_file = f"data/metrica_match{match_id}_inferred_events.csv"
    
    if not df_events.empty:
        # 统计一下
        n_passes = len(df_events[(df_events['type'] == 'PASS') & (df_events['from_team'] == 'Home')])
        print(f"Found {n_passes} home passes. Saving to {final_output_file}...")
        
        # 保存所有事件，方便后续处理脚本灵活读取
        df_events.to_csv(final_output_file, index=False)
    else:
        print("No events generated!")