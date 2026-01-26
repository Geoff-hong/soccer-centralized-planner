import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arrow
import os

# ===========================
# 1. 严格配置：0 到 1000 帧
# ===========================
FILE_NAME = "data/metrica_match1_train_attacker_centric.npz" # 确保文件名对上
START_FRAME = 85000     # <--- 强制从第 0 帧开始
NUM_FRAMES = 3000    # <--- 强制画 1000 帧
FPS = 25             
VEL_SCALE = 5.0      
KICK_SECTORS = 12    

def check_stats(obs, acts):
    print(f"\n{'='*40}")
    print(f" DATA SANITY CHECK REPORT")
    print(f"{'='*40}")
    
    if np.isnan(obs).any() or np.isnan(acts).any():
        print(f"\n[CRITICAL ERROR] Found NaNs in data!")
        return False
        
    print(f"[1] Value Ranges (Normalized)")
    print(f"    Coord X Range: {obs[:,:,0].min():.3f} to {obs[:,:,0].max():.3f}")
    print(f"    Coord Y Range: {obs[:,:,1].min():.3f} to {obs[:,:,1].max():.3f}")
    
    triggers = np.sum(acts[:, :, 2])
    print(f"\n[2] Action Stats (Whole Dataset)")
    print(f"    Total Kicks: {int(triggers)}")
    
    return True

def visualize_sequence(obs, acts, start_f, length, output_file="output/match1_frames_0_1000.gif"):
    print(f"\n{'='*40}")
    print(f" GENERATING VISUALIZATION: Frames {start_f} to {start_f+length}")
    print(f"{'='*40}")
    
    # 防止越界
    end_f = min(start_f + length, len(obs))
    real_length = end_f - start_f
    
    obs_seq = obs[start_f:end_f]
    acts_seq = acts[start_f:end_f]
    
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
        ax.set_title(f"Match 2 | Frame: {current_abs_frame} | (Normalized Space)")
        
        # 2. 画正方形球场边框
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color='gray', linewidth=2))
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5) 
        
        current_obs = obs_seq[frame_idx] 
        current_acts = acts_seq[frame_idx] 
        
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
            if is_home:
                vx, vy, trigger, kick_cls = current_acts[i]
                
                # 速度向量
                if abs(vx) > 0.001 or abs(vy) > 0.001:
                    ax.arrow(px, py, vx*VEL_SCALE, vy*VEL_SCALE, 
                             head_width=0.015, head_length=0.015, fc='k', ec='k', alpha=0.3)
                
                # 踢球事件
                if trigger > 0.5:
                    real_angle = (kick_cls / KICK_SECTORS) * 2 * np.pi
                    k_dx = np.cos(real_angle)
                    k_dy = np.sin(real_angle)
                    
                    ax.arrow(px, py, k_dx*0.15, k_dy*0.15, 
                             head_width=0.04, head_length=0.04, fc='red', ec='red', width=0.01, zorder=20)
                    ax.text(px, py-0.08, f"K:{int(kick_cls)}", color='red', fontsize=9, fontweight='bold', ha='center')

    ani = animation.FuncAnimation(fig, update, frames=real_length, interval=1000/FPS)
    
    print(f"Saving 1000 frames to {output_file} ... (This may take a minute)")
    os.makedirs("output", exist_ok=True)
    ani.save(output_file, writer='pillow', fps=FPS)
    print("\nDone!")

if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        print(f"File {FILE_NAME} not found!")
    else:
        data = np.load(FILE_NAME)
        obs = data['obs']
        acts = data['acts']
        
        if check_stats(obs, acts):
            # 直接调用，不再进行条件判断跳转
            visualize_sequence(obs, acts, START_FRAME, NUM_FRAMES)