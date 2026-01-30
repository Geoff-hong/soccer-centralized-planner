import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arrow
import os

# ===========================
# 1. 严格配置：0 到 1000 帧
# ===========================
FILE_NAME = "data/metrica_match2_train_attacker_centric.npz" # 确保文件名对上
START_FRAME = 0     # <--- 强制从第 0 帧开始
NUM_FRAMES = 2000    # <--- 强制画 1000 帧
FPS = 10            
VEL_SCALE = 5.0      
KICK_SECTORS = 12    
PRINT_PASS_LIST = True
PRINT_PASS_RANGE_ONLY = True  # 仅打印可视化范围内的传球事件

def check_stats(obs, acts=None, pass_flag=None):
    print(f"\n{'='*40}")
    print(f" DATA SANITY CHECK REPORT")
    print(f"{'='*40}")
    
    if np.isnan(obs).any():
        print(f"\n[CRITICAL ERROR] Found NaNs in data!")
        return False
    if acts is not None:
        if not np.issubdtype(acts.dtype, np.number):
            print(f"\n[CRITICAL ERROR] Acts dtype is not numeric: {acts.dtype}")
            return False
        if np.isnan(acts).any():
            print(f"\n[CRITICAL ERROR] Found NaNs in actions!")
            return False
        
    print(f"[1] Value Ranges (Normalized)")
    print(f"    Coord X Range: {obs[:,:,0].min():.3f} to {obs[:,:,0].max():.3f}")
    print(f"    Coord Y Range: {obs[:,:,1].min():.3f} to {obs[:,:,1].max():.3f}")
    
    if pass_flag is not None:
        triggers = np.sum(pass_flag)
    else:
        triggers = 0
    print(f"\n[2] Action Stats (Whole Dataset)")
    print(f"    Total Kicks: {int(triggers)}")
    
    return True


def _group_consecutive(indices):
    if len(indices) == 0:
        return []
    groups = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = i
            prev = i
    groups.append((start, prev))
    return groups


def print_pass_events(pass_flag, passer_id=None, receiver_id=None, pass_dir=None, start_f=None, end_f=None):
    if pass_flag is None:
        print("\n[Pass Events] pass_flag missing, skip printing list.")
        return
    idx_all = np.where(pass_flag > 0.5)[0]
    groups_all = _group_consecutive(idx_all.tolist())
    print(f"\n[Pass Events] Total frames: {len(idx_all)} | Unique events (grouped): {len(groups_all)}")

    if start_f is not None and end_f is not None:
        idx = idx_all[(idx_all >= start_f) & (idx_all < end_f)]
        groups = _group_consecutive(idx.tolist())
        print(f"[Pass Events] In range [{start_f}, {end_f}) frames: {len(idx)} | Unique events: {len(groups)}")
    else:
        idx = idx_all
        groups = groups_all

    for start, end in groups:
        i = start
        passer = int(passer_id[i]) if passer_id is not None else -1
        receiver = int(receiver_id[i]) if receiver_id is not None else -1
        pdir = int(pass_dir[i]) if pass_dir is not None else -1
        print(f"  frame {start}-{end}: passer={passer}, receiver={receiver}, dir={pdir}")

def visualize_sequence(
    obs,
    acts,
    start_f,
    length,
    pass_flag=None,
    passer_id=None,
    pass_dir=None,
    dt=0.04,
    output_file="output/match_frames.gif",
    fps=FPS,
    vel_scale=VEL_SCALE,
    match_label="Match",
):
    print(f"\n{'='*40}")
    print(f" GENERATING VISUALIZATION: Frames {start_f} to {start_f+length}")
    print(f"{'='*40}")
    
    # 防止越界
    end_f = min(start_f + length, len(obs))
    real_length = end_f - start_f
    
    obs_seq = obs[start_f:end_f]
    acts_seq = acts[start_f:end_f] if acts is not None else None
    pass_flag_seq = pass_flag[start_f:end_f] if pass_flag is not None else None
    passer_id_seq = passer_id[start_f:end_f] if passer_id is not None else None
    pass_dir_seq = pass_dir[start_f:end_f] if pass_dir is not None else None
    
    # 建立正方形画布 (匹配 [-1, 1] 的数据)
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    def update(frame_idx):
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{real_length}...", end='\r')
            
        ax.clear()
        # 1. 设置正方形的可视化范围 [-1.1, 1.1]
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        # 显示当前的绝对帧号
        current_abs_frame = start_f + frame_idx
        ax.set_title(f"{match_label} | Frame: {current_abs_frame} | (Normalized Space)")
        
        # 2. 画正方形球场边框
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color='gray', linewidth=2))
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5) 
        
        current_obs = obs_seq[frame_idx] 
        current_acts = acts_seq[frame_idx] if acts_seq is not None else None
        current_pass_flag = pass_flag_seq[frame_idx] if pass_flag_seq is not None else None
        current_passer = passer_id_seq[frame_idx] if passer_id_seq is not None else None
        current_pass_dir = pass_dir_seq[frame_idx] if pass_dir_seq is not None else None
        
        # --- 画球 ---
        ball_x, ball_y, _ = current_obs[-1]
        ax.add_patch(Circle((ball_x, ball_y), 0.02, color='orange', zorder=10))
        
        # --- 画球员 ---
        for i in range(22):
            px, py, has_ball = current_obs[i]
            
            # 跳过 Padding (坐标为0且没控球通常是无效/离场)
            if px == 0 and py == 0 and has_ball < 0.1: continue
            
            is_home = i < 11
            color = 'blue' if is_home else 'red'
            
            # 控球环 (Green Ring)
            if has_ball > 0.5:
                ax.add_patch(Circle((px, py), 0.04, fill=False, color='green', linewidth=2))
                color = 'cyan' if is_home else 'magenta'
            
            ax.add_patch(Circle((px, py), 0.025, color=color, zorder=5))
            if is_home:
                # 标注 Home 球员的 ID (0-10)
                ax.text(px, py+0.04, str(i), fontsize=8, color='blue', ha='center')
            
            # --- 画 Action (Home Only) ---
            if is_home and current_acts is not None:
                vx, vy = current_acts[i][0], current_acts[i][1]
                # 速度向量
                # acts 是 m/s，因此可视化用位移 = v * dt
                disp_x, disp_y = vx * dt, vy * dt
                if abs(disp_x) > 0.001 or abs(disp_y) > 0.001:
                    ax.arrow(px, py, disp_x*vel_scale, disp_y*vel_scale, 
                             head_width=0.015, head_length=0.015, fc='k', ec='k', alpha=0.3)

                # 传球事件（新格式）
                if current_pass_flag is not None and current_pass_flag > 0.5 and current_passer == i and current_pass_dir is not None and current_pass_dir >= 0:
                    real_angle = (current_pass_dir / KICK_SECTORS) * 2 * np.pi
                    k_dx = np.cos(real_angle)
                    k_dy = np.sin(real_angle)
                    ax.arrow(px, py, k_dx*0.15, k_dy*0.15, 
                             head_width=0.04, head_length=0.04, fc='red', ec='red', width=0.01, zorder=20)
                    ax.text(px, py-0.08, f"P:{int(current_pass_dir)}", color='red', fontsize=9, fontweight='bold', ha='center')

    ani = animation.FuncAnimation(fig, update, frames=real_length, interval=1000/fps)
    
    print(f"Saving 1000 frames to {output_file} ... (This may take a minute)")
    os.makedirs("output", exist_ok=True)
    ani.save(output_file, writer='pillow', fps=fps)
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and sanity-check processed datasets.")
    parser.add_argument("--match_id", type=int, default=None, help="Metrica match id (1/2). Overrides --file.")
    parser.add_argument("--file", type=str, default=None, help="Path to .npz file. If set, overrides --match_id.")
    parser.add_argument("--start_frame", type=int, default=START_FRAME)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--vel_scale", type=float, default=VEL_SCALE)
    parser.add_argument("--output", type=str, default=None, help="Output GIF path.")
    args = parser.parse_args()

    if args.file:
        file_name = args.file
        match_label = os.path.basename(args.file)
    elif args.match_id is not None:
        file_name = f"data/metrica_match{args.match_id}_train_attacker_centric.npz"
        match_label = f"Match {args.match_id}"
    else:
        file_name = FILE_NAME
        match_label = "Match"

    if args.output:
        output_file = args.output
    else:
        start = args.start_frame
        end = args.start_frame + args.num_frames
        output_file = os.path.join("output", f"{match_label.replace(' ', '_').lower()}_frames_{start}_{end}.gif")

    if not os.path.exists(file_name):
        print(f"File {file_name} not found!")
    else:
        data = np.load(file_name)
        obs = data['obs']
        if 'act_move' in data:
            acts = data['act_move']
        elif 'acts' in data:
            acts = data['acts']
        else:
            acts = None
        pass_flag = data['pass_flag'] if 'pass_flag' in data else None
        passer_id = data['passer_id'] if 'passer_id' in data else None
        pass_dir = data['pass_dir'] if 'pass_dir' in data else None
        receiver_id = data['receiver_id'] if 'receiver_id' in data else None
        dt = float(data['dt']) if 'dt' in data else (0.1 if 'skillcorner' in file_name else 0.04)

        if check_stats(obs, acts, pass_flag):
            visualize_sequence(
                obs,
                acts,
                args.start_frame,
                args.num_frames,
                pass_flag,
                passer_id,
                pass_dir,
                dt=dt,
                output_file=output_file,
                fps=args.fps,
                vel_scale=args.vel_scale,
                match_label=match_label,
            )
            if PRINT_PASS_LIST:
                if PRINT_PASS_RANGE_ONLY:
                    print_pass_events(pass_flag, passer_id, receiver_id, pass_dir, args.start_frame, args.start_frame + args.num_frames)
                else:
                    print_pass_events(pass_flag, passer_id, receiver_id, pass_dir)
