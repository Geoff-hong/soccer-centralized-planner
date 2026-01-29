import kloppy
from kloppy import metrica
from kloppy import sportec
# 关键导入：必须使用这两个类来定义球场尺寸
from kloppy.domain import Dimension, MetricPitchDimensions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mplsoccer import Pitch

# ===========================
# 可视化配置
# ===========================
MATCH_ID = 2              # 比赛ID
START_FRAME = 0           # 起始帧（设置为 None 则从头开始）
NUM_FRAMES = 200         # 帧数（设置为 None 则画全部）
FPS = 25                  # 帧率

def main():
    # =========================================================================
    # 1. LOADING DATA
    # =========================================================================
    print(f"[1/4] Loading Metrica tracking data (Match {MATCH_ID})...")
    
    dataset = metrica.load_open_data(match_id=MATCH_ID
    , limit=200)

    # dataset = sportec.load_open_tracking_data(
    #     match_id="J03WMX",
    #     limit=500  # optional: for efficiency, we only load the first 500 frames
    # )

    # =========================================================================
    # 2. TRANSFORMING COORDINATES (FIXED)
    # =========================================================================
    print("[2/4] Transforming coordinates to Metric System (105x68)...")
    
    # ERROR FIX: specifically use PitchDimensions object, not a list
   

    dataset = dataset.transform(
        to_pitch_dimensions=MetricPitchDimensions(
        standardized=True,
        x_dim=Dimension(min=0, max=105),
        y_dim=Dimension(min=0, max=68),
        )
    )

    # =========================================================================
    # 3. EXPORTING DATA
    # =========================================================================
    print("[3/4] Exporting to Pandas DataFrame...")
    df = dataset.to_pandas()
    
    # 帧数切片
    total_frames = len(df)
    start_frame = START_FRAME if START_FRAME is not None else 0
    
    if NUM_FRAMES is not None:
        end_frame = min(start_frame + NUM_FRAMES, total_frames)
    else:
        end_frame = total_frames
    
    df = df.iloc[start_frame:end_frame].reset_index(drop=True)
    
    print(f"   Total frames available: {total_frames}")
    print(f"   Visualizing frames: {start_frame} to {end_frame-1} ({len(df)} frames)")

    # =========================================================================
    # 4. VISUALIZATION
    # =========================================================================
    print("[4/4] Generating visualization video...")
    
    # Print sample coordinates for reference
    print("\n📊 Sample Coordinates (First Frame):")
    first_row = df.iloc[0]
    ball_x_first = first_row['ball_x']
    ball_y_first = first_row['ball_y']
    if not pd.isna(ball_x_first) and not pd.isna(ball_y_first):
        print(f"  Ball: x={ball_x_first:.2f}m, y={ball_y_first:.2f}m")
    else:
        print(f"  Ball: (No data in first frame)")
    
    # Print first available home player
    home_x_cols = [c for c in df.columns if 'home_' in c and c.endswith('_x')]
    if home_x_cols:
        col_name = home_x_cols[0]
        player_id = col_name.replace('home_', '').replace('_x', '')
        x_val = first_row[col_name]
        y_val = first_row[col_name.replace('_x', '_y')]
        if not pd.isna(x_val) and not pd.isna(y_val):
            print(f"  Home Player {player_id}: x={x_val:.2f}m, y={y_val:.2f}m")
    
    # Print coordinate ranges
    print(f"\n📏 Coordinate Ranges:")
    ball_x_min, ball_x_max = df['ball_x'].min(), df['ball_x'].max()
    ball_y_min, ball_y_max = df['ball_y'].min(), df['ball_y'].max()
    if not pd.isna(ball_x_min):
        print(f"  Ball X: [{ball_x_min:.2f}, {ball_x_max:.2f}] meters")
        print(f"  Ball Y: [{ball_y_min:.2f}, {ball_y_max:.2f}] meters")
    print()

    # Define columns
    home_cols_x = [c for c in df.columns if "home_" in c and c.endswith("_x")]
    home_cols_y = [c for c in df.columns if "home_" in c and c.endswith('_y')]
    away_cols_x = [c for c in df.columns if "away_" in c and c.endswith('_x')]
    away_cols_y = [c for c in df.columns if "away_" in c and c.endswith('_y')]
    ball_col_x = "ball_x"
    ball_col_y = "ball_y"

    # Setup the pitch using mplsoccer
    pitch = Pitch(
        pitch_type='custom', 
        pitch_length=105, 
        pitch_width=68,
        line_color='black',
        line_zorder=2,
        axis=True,  # 显示坐标轴
        label=True  # 显示坐标标签
    )

    fig, ax = pitch.draw(figsize=(12, 8))
    
    # 添加坐标轴和网格
    ax.set_xlabel('X coordinate (meters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y coordinate (meters)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # 网格线放在最底层
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(0, 106, 10))
    ax.set_yticks(np.arange(0, 69, 10))
    
    # 添加小刻度
    ax.set_xticks(np.arange(0, 106, 5), minor=True)
    ax.set_yticks(np.arange(0, 69, 5), minor=True)
    ax.grid(which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

    scat_home = ax.scatter([], [], c='red', s=80, edgecolors='white', label='Home', zorder=3)
    scat_away = ax.scatter([], [], c='blue', s=80, edgecolors='white', label='Away', zorder=3)
    scat_ball = ax.scatter([], [], c='black', s=50, edgecolors='white', label='Ball', zorder=4)
    
    # 添加文字显示坐标信息
    txt_info = ax.text(52.5, 73, '', ha='center', fontsize=11, fontweight='bold')
    txt_ball_coords = ax.text(52.5, -3, '', ha='center', fontsize=10, 
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(loc='upper right')

    def update(frame_idx):
        if frame_idx % 100 == 0:
            print(f"   Rendering frame {frame_idx}/{len(df)}...", end='\r')
            
        row = df.iloc[frame_idx]

        # Home Team
        h_x = pd.to_numeric(row[home_cols_x], errors='coerce').values
        h_y = pd.to_numeric(row[home_cols_y], errors='coerce').values
        mask_h = ~np.isnan(h_x)
        scat_home.set_offsets(np.c_[h_x[mask_h], h_y[mask_h]])

        # Away Team
        a_x = pd.to_numeric(row[away_cols_x], errors='coerce').values
        a_y = pd.to_numeric(row[away_cols_y], errors='coerce').values
        mask_a = ~np.isnan(a_x)
        scat_away.set_offsets(np.c_[a_x[mask_a], a_y[mask_a]])

        # Ball
        b_x = pd.to_numeric(row[ball_col_x], errors='coerce')
        b_y = pd.to_numeric(row[ball_col_y], errors='coerce')
        if not np.isnan(b_x) and not np.isnan(b_y):
            scat_ball.set_offsets(np.c_[b_x, b_y])
            # 显示球的坐标
            txt_ball_coords.set_text(f"Ball: ({b_x:.2f}, {b_y:.2f}) m")
        else:
            scat_ball.set_offsets(np.empty((0, 2)))
            txt_ball_coords.set_text("Ball: (N/A)")

        # 显示绝对帧号
        absolute_frame = start_frame + frame_idx
        timestamp = absolute_frame * 0.04
        txt_info.set_text(f"Match {MATCH_ID} | Frame: {absolute_frame} | Time: {timestamp:.2f}s")
        
        return scat_home, scat_away, scat_ball, txt_info, txt_ball_coords

    anim = FuncAnimation(fig, update, frames=len(df), interval=1000/FPS, blit=True)

    import os
    os.makedirs("../output", exist_ok=True)
    
    # 根据配置生成文件名
    if NUM_FRAMES is not None:
        output_filename = f"output/metrica_match{MATCH_ID}_frames_{start_frame}_{end_frame-1}.gif"
    else:
        output_filename = f"output/metrica_match{MATCH_ID}_full.gif"
    
    print(f"\nSaving animation to {output_filename}...")
    anim.save(output_filename, writer='pillow', fps=FPS)
    print(f"Success! Animation saved as {output_filename}")
    print("\n📍 Coordinate System Info:")
    print("  X-axis: 0 to 105 meters (pitch length)")
    print("  Y-axis: 0 to 68 meters (pitch width)")
    print("  Origin (0,0) is at bottom-left corner")

if __name__ == "__main__":
    main()