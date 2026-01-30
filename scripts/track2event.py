import argparse
import numpy as np
import pandas as pd
from kloppy import metrica
import warnings
import os

warnings.filterwarnings('ignore')

# ============================
# 参数配置
# ============================
POSSESSION_DIST = 3.0
MIN_PASS_FRAMES = 1
MIN_POSSESSION_FRAMES = 1
MAX_FLIGHT_DURATION = 150 # 稍微收紧一点，6秒以上的传球很罕见
MAX_DEAD_BALL_RATIO = 0.8 # 如果飞行过程中超过 80% 的时间球是 NaN，则视为死球事件
GAP_FILL_FRAMES = 10      # 填补短暂丢球/遮挡

def generate_event_data_from_tracking(tracking_df):
    print("--- Starting Event Generation ---")
    
    # [新增] 0. 预处理：轻微插值以修复追踪闪烁 (Flickers)
    # 这能避免 "持球->丢失1帧->持球" 被误判为两次盘带
    # 我们只对用来计算距离的坐标进行插值，不改变原始数据结构
    cols_to_fix = [c for c in tracking_df.columns if c.endswith('_x') or c.endswith('_y')]
    df_calc = tracking_df.copy()
    df_calc[cols_to_fix] = df_calc[cols_to_fix].interpolate(method='linear', limit=3)
    
    # 1. 准备数据
    ball_x = df_calc['ball_x']
    ball_y = df_calc['ball_y']
    
    home_cols = [c for c in df_calc.columns if c.startswith('home_') and c.endswith('_x')]
    away_cols = [c for c in df_calc.columns if c.startswith('away_') and c.endswith('_x')]
    home_ids = [c.split('_')[1] for c in home_cols]
    away_ids = [c.split('_')[1] for c in away_cols]
    
    all_player_ids = [f"Home_{pid}" for pid in home_ids] + [f"Away_{pid}" for pid in away_ids]
    
    # 2. 计算距离矩阵 (Fill NaN with Inf)
    # 如果插值后仍然是 NaN (说明是真的出界了)，这里会变成 Inf，没人持球，这是对的
    distances = []
    for pid in home_ids:
        d = np.sqrt((df_calc[f'home_{pid}_x'] - ball_x)**2 + (df_calc[f'home_{pid}_y'] - ball_y)**2)
        distances.append(d.values)
    for pid in away_ids:
        d = np.sqrt((df_calc[f'away_{pid}_x'] - ball_x)**2 + (df_calc[f'away_{pid}_y'] - ball_y)**2)
        distances.append(d.values)
    
    dist_matrix = np.array(distances).T
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)
    
    # 3. 确定控球者
    min_dist_idx = np.argmin(dist_matrix, axis=1)
    min_dists = np.min(dist_matrix, axis=1)
    
    possessor_array = np.full(len(tracking_df), -1, dtype=int)
    has_possession_mask = min_dists < POSSESSION_DIST
    possessor_array[has_possession_mask] = min_dist_idx[has_possession_mask]
    
    # 平滑 (Fill tiny gaps)
    possessor_series = pd.Series(possessor_array)
    possessor_filled = (
        possessor_series.replace(-1, np.nan)
        .ffill(limit=GAP_FILL_FRAMES)
        .bfill(limit=GAP_FILL_FRAMES)
        .fillna(-1)
        .astype(int)
    )
    possessor_list = possessor_filled.tolist()
    
    # 4. 提取片段 (逻辑保持不变，过滤掉 -1 的片段)
    segments = []
    current_seg_pid = -1
    current_seg_start = 0
    
    for i, pid in enumerate(possessor_list):
        if i == 0:
            current_seg_pid = pid
            continue
        if pid != current_seg_pid:
            # 只有当上一段持球者不是 -1 (有人持球) 时，才记录为一个片段
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
    # 记录原始未插值的 ball_x 用于检查死球
    raw_ball_x = tracking_df['ball_x'] 
    
    skip_short = 0
    skip_long = 0
    skip_dead = 0
    skip_dribble = 0
    for i in range(len(segments) - 1):
        seg_from = segments[i]
        seg_to = segments[i+1]
        
        flight_start = seg_from['end']
        flight_end = seg_to['start']
        duration = flight_end - flight_start
        
        # 基础过滤
        if duration < MIN_PASS_FRAMES:
            skip_short += 1
            continue
        if duration > MAX_FLIGHT_DURATION:
            skip_long += 1
            continue   
        
        # [关键新增] 死球过滤逻辑
        # 检查飞行过程中，球是否大部分时间处于 NaN 状态 (出界/死球)
        # 即使我们上面做了插值计算距离，这里检查一定要用原始数据
        flight_slice = raw_ball_x.iloc[flight_start : flight_end]
        nan_count = flight_slice.isna().sum()
        
        # 如果飞行期间超过 50% 的帧球都不在，说明这是一个 "Kick Out -> Throw In" 的过程
        # 我们不应该把它标记为一次成功的 Pass/Turnover 训练样本
        if len(flight_slice) > 0 and (nan_count / len(flight_slice) > MAX_DEAD_BALL_RATIO):
            skip_dead += 1
            continue

        p_from_name = all_player_ids[seg_from['player_idx']]
        p_to_name = all_player_ids[seg_to['player_idx']]
        
        team_from, pid_from = p_from_name.split('_')
        team_to, pid_to = p_to_name.split('_')
        
        # 过滤自己传自己 (Dribble)
        if pid_from == pid_to and team_from == team_to:
            skip_dribble += 1
            continue
            
        event_type = "PASS" if team_from == team_to else "TURNOVER"
        
        # 使用 tracking_df 的真实索引，避免因索引不从 0 开始导致对齐问题
        start_fid = int(tracking_df.index[flight_start])
        end_fid = int(tracking_df.index[flight_end])

        pass_events.append({
            'start_frame': start_fid,
            'end_frame': end_fid,
            'type': event_type,
            'from_team': team_from,
            'from_player': pid_from,
            'to_team': team_to,
            'to_player': pid_to,
            'duration_frames': duration
        })

    cols = ['start_frame', 'end_frame', 'type', 'from_team', 'from_player', 'to_team', 'to_player', 'duration_frames']
    df_out = pd.DataFrame(pass_events, columns=cols)
    print(f"Segments: {len(segments)} | Events: {len(df_out)} | skip_short={skip_short} skip_long={skip_long} skip_dead={skip_dead} skip_dribble={skip_dribble}")
    return df_out

def process_match(match_id):
    print(f"Loading Match {match_id}...")
    dataset = metrica.load_open_data(match_id=match_id)

    from kloppy.domain import Dimension, MetricPitchDimensions
    # 确保坐标系一致
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

    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    os.makedirs(data_dir, exist_ok=True)
    final_output_file = os.path.join(data_dir, f"metrica_match{match_id}_inferred_events.csv")

    if not df_events.empty:
        n_passes = len(df_events[(df_events['type'] == 'PASS') & (df_events['from_team'] == 'Home')])
        print(f"Found {n_passes} home passes (cleaned). Saving to {final_output_file}...")
        df_events.to_csv(final_output_file, index=False)
    else:
        print("No events generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate inferred pass events from Metrica tracking data.")
    parser.add_argument("--match_id", type=int, default=2, help="Metrica match id (default: 2)")
    args = parser.parse_args()
    process_match(args.match_id)
