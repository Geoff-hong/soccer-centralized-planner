import kloppy
from kloppy import metrica
import pandas as pd
import numpy as np

def check_substitutions(match_id=1):
    print(f"Checking Match {match_id} for substitutions...")
    
    # 1. 加载数据
    dataset = metrica.load_open_data(match_id=match_id)
    df = dataset.to_pandas()
    
    # 2. 提取所有可能的 Home 球员列
    # 格式通常是 home_ID_x
    all_home_cols = [c for c in df.columns if c.startswith('home_') and c.endswith('_x')]
    
    # 3. 逐帧检查每一列是否是 NaN
    # 如果是 NaN，说明该球员不在场上
    # 我们创建一个 boolean dataframe: True=在场, False=不在场
    presence_df = ~df[all_home_cols].isna()
    
    # 4. 统计每一帧在场人数
    players_on_pitch = presence_df.sum(axis=1)
    
    # 打印一些异常帧（人数不是11人的时候）
    abnormal_frames = players_on_pitch[players_on_pitch != 11]
    if not abnormal_frames.empty:
        print(f"Warning: Found {len(abnormal_frames)} frames where player count is not 11.")
        print("Example frames:", abnormal_frames.head().index.tolist())
    else:
        print("Player count is consistently 11. (This is good!)")

    # 5. 检查换人关键点：活跃集合的变化
    # 每一帧的"活跃ID集合"
    # 这是一个比较慢的操作，我们采样检查
    
    current_ids = set()
    
    # 这种方法能找到确切的换人帧
    changes_detected = []
    
    # 获取每一列（每一个球员）在场上的区间
    print("\n--- Player Presence Intervals ---")
    for col in all_home_cols:
        pid = col.split('_')[1]
        # 找到这个球员非 NaN 的索引
        valid_indices = df[col].dropna().index
        if len(valid_indices) > 0:
            start, end = valid_indices[0], valid_indices[-1]
            duration = end - start
            # 如果 duration 比总帧数短很多，且不是从头开始或到尾结束，说明可能是换人
            status = "Full Match"
            if len(valid_indices) < len(df) * 0.9:
                status = "Sub/Partial"
            
            print(f"Player {pid}: Frames {start} to {end} ({status})")
            
            if status == "Sub/Partial":
                changes_detected.append(pid)

    if changes_detected:
        print(f"\n[ALERT] Substitutions detected! Players involved: {changes_detected}")
        print("Strategy: You MUST map these players to fixed slots (0-10).")
    else:
        print("\n[OK] No substitutions detected. Safe to use fixed IDs.")

if __name__ == "__main__":
    check_substitutions(match_id=1) 
    # Metrica Match 2 应该是有换人的，Match 1 可能比较干净