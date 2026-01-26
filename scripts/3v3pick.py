import kloppy
from kloppy import metrica
from kloppy.domain import Dimension, MetricPitchDimensions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mplsoccer import Pitch

def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def identify_closest_player(row, team_prefix, all_ids, ball_x, ball_y):
    """返回当前帧离球最近的一个球员ID"""
    min_dist = float('inf')
    closest_id = None
    
    for pid in all_ids:
        px = row[f'{team_prefix}_{pid}_x']
        py = row[f'{team_prefix}_{pid}_y']
        
        if pd.isna(px) or pd.isna(py):
            continue
            
        dist = get_distance(px, py, ball_x, ball_y)
        if dist < min_dist:
            min_dist = dist
            closest_id = pid
            
    return closest_id

def extract_dynamic_3v3_scenario(df, start_frame, max_duration=200):
    """
    高级提取逻辑：
    1. 动态扫描后续帧，记录参与者。
    2. 若参与者超过3人，提前截断。
    3. 若参与者不足3人，用初始帧最近的替补。
    """
    # 1. 基础设置
    if start_frame >= len(df): return None
    
    # 获取所有可能的 ID (只做一次)
    home_ids_all = [c.split('_')[1] for c in df.columns if c.startswith('home_') and c.endswith('_x')]
    away_ids_all = [c.split('_')[1] for c in df.columns if c.startswith('away_') and c.endswith('_x')]
    
    active_home = [] # 使用列表保持顺序，或者用集合保持唯一性
    active_away = []
    
    valid_duration = 0
    stop_reason = "Max Duration"
    
    # 2. 动态扫描 (Scan Phase)
    # 我们逐帧检查，决定这个片段能有多长
    for i in range(max_duration):
        current_frame = start_frame + i
        if current_frame >= len(df):
            stop_reason = "End of Match"
            break
            
        row = df.iloc[current_frame]
        bx, by = row['ball_x'], row['ball_y']
        
        if pd.isna(bx) or pd.isna(by):
            # 球出界或数据丢失，这里可以选择跳过或中断
            # 为保证连续性，如果球丢了，我们通常中断片段
            stop_reason = "Ball Lost"
            break
            
        # 找到当前帧的主角
        h_id = identify_closest_player(row, 'home', home_ids_all, bx, by)
        a_id = identify_closest_player(row, 'away', away_ids_all, bx, by)
        
        if h_id is None or a_id is None:
            break

        # --- 核心逻辑：检查是否溢出 ---
        
        # 主队检查
        if h_id not in active_home:
            if len(active_home) >= 3:
                stop_reason = "Home > 3 Players"
                break # 第4个人出现了，截断！
            active_home.append(h_id)
            
        # 客队检查
        if a_id not in active_away:
            if len(active_away) >= 3:
                stop_reason = "Away > 3 Players"
                break # 第4个人出现了，截断！
            active_away.append(a_id)
            
        valid_duration += 1

    # 如果片段太短（比如第2帧就换人了），则忽略该片段
    if valid_duration < 50:
        return None

    # 3. 补全阶段 (Fill Phase)
    # 如果扫描完了还是不够3人，回到 start_frame 找离球最近的“路人甲”补位
    start_row = df.iloc[start_frame]
    sbx, sby = start_row['ball_x'], start_row['ball_y']
    
    def fill_team(active_list, all_ids, prefix):
        if len(active_list) < 3:
            # 计算所有人的距离
            candidates = []
            for pid in all_ids:
                if pid in active_list: continue # 已经在名单里的跳过
                px = start_row[f'{prefix}_{pid}_x']
                py = start_row[f'{prefix}_{pid}_y']
                if pd.isna(px): continue
                dist = get_distance(px, py, sbx, sby)
                candidates.append((dist, pid))
            
            # 排序并补充
            candidates.sort(key=lambda x: x[0])
            needed = 3 - len(active_list)
            for _, pid in candidates[:needed]:
                active_list.append(pid)
                
    fill_team(active_home, home_ids_all, 'home')
    fill_team(active_away, away_ids_all, 'away')
    
    # 再次检查（防止本来人就不够，比如红牌）
    if len(active_home) < 3 or len(active_away) < 3:
        return None

    # 4. 构建数据 (Standardization)
    chunk = df.iloc[start_frame : start_frame + valid_duration].copy()
    
    new_data = {
        'frame_idx': chunk.index,
        'chunk_start': start_frame,
        'stop_reason': stop_reason, # 方便调试看为什么停了
        'ball_x': chunk['ball_x'],
        'ball_y': chunk['ball_y']
    }
    
    # 按我们确定的名单提取列
    for i, pid in enumerate(active_home):
        new_data[f'home_{i}_x'] = chunk[f'home_{pid}_x']
        new_data[f'home_{i}_y'] = chunk[f'home_{pid}_y']
        new_data[f'home_{i}_id'] = int(pid)

    for i, pid in enumerate(active_away):
        new_data[f'away_{i}_x'] = chunk[f'away_{pid}_x']
        new_data[f'away_{i}_y'] = chunk[f'away_{pid}_y']
        new_data[f'away_{i}_id'] = int(pid)

    return pd.DataFrame(new_data)

def main():
    # =========================================================================
    # 1. LOADING
    # =========================================================================
    print("[1/4] Loading Metrica Data...")
    dataset = metrica.load_open_data(match_id=2, limit=None)
    
    dataset = dataset.transform(
        to_pitch_dimensions=MetricPitchDimensions(
            standardized=True,
            x_dim=Dimension(min=0, max=105),
            y_dim=Dimension(min=0, max=68),
        )
    )
    full_df = dataset.to_pandas()
    
    # =========================================================================
    # 2. BATCH PROCESSING WITH NEW LOGIC
    # =========================================================================
    print("[2/4] Batch Extracting Dynamic Scenarios...")
    
    scenario_chunks = []
    
    # 循环步长可以设大一点，或者设为 200，保证覆盖
    # 这里我们设为 200，意味着尽可能尝试提取连续的片段
    current_frame = 1000
    max_scan_frames = 5000 # 演示用，只扫前5000帧
    
    while current_frame < min(len(full_df), max_scan_frames):
        # 尝试提取
        chunk_df = extract_dynamic_3v3_scenario(full_df, current_frame, max_duration=200)
        
        if chunk_df is not None:
            scenario_chunks.append(chunk_df)
            real_duration = len(chunk_df)
            reason = chunk_df['stop_reason'].iloc[0]
            print(f"   Frame {current_frame}: Extracted {real_duration} frames. Reason: {reason}")
            
            # 下一次搜索从这个片段结束的地方开始，避免重叠
            current_frame += real_duration + 10 # +10 稍微留点空隙
        else:
            # 提取失败（比如时间太短，或者人不够），往前滑一点继续试
            current_frame += 50

    if not scenario_chunks:
        print("No valid scenarios found.")
        return

    final_viz_df = pd.concat(scenario_chunks, ignore_index=True)
    print(f"   Total frames compiled: {len(final_viz_df)}")

    # =========================================================================
    # 3. VISUALIZATION
    # =========================================================================
    print("[3/4] Generating Logic Verification GIF...")

    pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68, 
                  line_color='black', line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 7))

    scat_home = ax.scatter([], [], c='red', s=120, edgecolors='black', label='Home', zorder=3)
    scat_away = ax.scatter([], [], c='blue', s=120, edgecolors='black', label='Away', zorder=3)
    scat_ball = ax.scatter([], [], c='black', s=60, edgecolors='white', label='Ball', zorder=4)

    # 动态标签
    h_texts = [ax.text(0,0, '', color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5) for _ in range(3)]
    a_texts = [ax.text(0,0, '', color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5) for _ in range(3)]

    info_text = ax.text(52.5, 70, '', ha='center', fontsize=12)
    reason_text = ax.text(52.5, -2, '', ha='center', fontsize=10, color='blue')

    def update(frame_idx):
        row = final_viz_df.iloc[frame_idx]
        
        # Home
        h_xy = []
        for i in range(3):
            x, y = row[f'home_{i}_x'], row[f'home_{i}_y']
            h_xy.append([x, y])
            h_texts[i].set_position((x, y))
            h_texts[i].set_text(str(int(row[f'home_{i}_id'])))

        scat_home.set_offsets(np.array(h_xy))

        # Away
        a_xy = []
        for i in range(3):
            x, y = row[f'away_{i}_x'], row[f'away_{i}_y']
            a_xy.append([x, y])
            a_texts[i].set_position((x, y))
            a_texts[i].set_text(str(int(row[f'away_{i}_id'])))

        scat_away.set_offsets(np.array(a_xy))
        scat_ball.set_offsets(np.c_[row['ball_x'], row['ball_y']])

        info_text.set_text(f"Viz Frame: {frame_idx} | Match Frame: {row['frame_idx']}")
        reason_text.set_text(f"Stop Reason: {row['stop_reason']}")

        return [scat_home, scat_away, scat_ball, info_text, reason_text] + h_texts + a_texts

    anim = FuncAnimation(fig, update, frames=len(final_viz_df), interval=30, blit=True)
    
    import os
    os.makedirs("../output", exist_ok=True)
    output_path = "../output/metrica_dynamic_logic.gif"
    anim.save(output_path, writer='pillow', fps=30)
    print(f"Success! Saved {output_path}")

if __name__ == "__main__":
    main()