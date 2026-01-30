import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def draw_frame(ax, obs, acts, pass_flag, passer_id, receiver_id, pass_dir, frame_idx, dt, vel_scale):
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"Frame {frame_idx} (normalized)")

    # Pitch border
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color="black", linewidth=2))

    current_obs = obs[frame_idx]
    current_acts = acts[frame_idx] if acts is not None else None
    current_pass_flag = pass_flag[frame_idx] if pass_flag is not None else 0
    current_passer = int(passer_id[frame_idx]) if passer_id is not None else -1
    current_receiver = int(receiver_id[frame_idx]) if receiver_id is not None else -1
    current_pass_dir = int(pass_dir[frame_idx]) if pass_dir is not None else -1

    # Players (0-10 teammates, 11-21 opponents)
    for i in range(22):
        px, py, has_ball = current_obs[i]
        if px == 0 and py == 0 and has_ball == 0:
            continue
        is_home = i < 11
        color = "blue" if is_home else "red"
        if has_ball > 0.5:
            ax.add_patch(Circle((px, py), 0.04, fill=False, color="green", linewidth=2))
            color = "cyan" if is_home else "magenta"
        ax.add_patch(Circle((px, py), 0.025, color=color, zorder=5))
        if is_home:
            ax.text(px, py + 0.04, str(i), fontsize=8, color="blue", ha="center")

        # Velocity vectors for teammates only
        if is_home and current_acts is not None:
            vx, vy = current_acts[i][0], current_acts[i][1]
            disp_x, disp_y = vx * dt, vy * dt
            if abs(disp_x) > 0.001 or abs(disp_y) > 0.001:
                ax.arrow(
                    px,
                    py,
                    disp_x * vel_scale,
                    disp_y * vel_scale,
                    head_width=0.015,
                    head_length=0.015,
                    fc="k",
                    ec="k",
                    alpha=0.3,
                )

        # Pass arrow
        if (
            is_home
            and current_pass_flag > 0.5
            and current_passer == i
            and current_pass_dir >= 0
        ):
            angle = (current_pass_dir / 12.0) * 2 * np.pi
            k_dx = np.cos(angle)
            k_dy = np.sin(angle)
            ax.arrow(
                px,
                py,
                k_dx * 0.15,
                k_dy * 0.15,
                head_width=0.04,
                head_length=0.04,
                fc="red",
                ec="red",
                width=0.01,
                zorder=20,
            )
            ax.text(
                px,
                py - 0.08,
                f"dir:{current_pass_dir} recv:{current_receiver}",
                color="red",
                fontsize=8,
                fontweight="bold",
                ha="center",
            )

    # Ball (index 22)
    if len(current_obs) > 22:
        bx, by, _ = current_obs[22]
        if not (bx == 0 and by == 0):
            ax.add_patch(Circle((bx, by), 0.02, color="orange", zorder=10))


def main():
    parser = argparse.ArgumentParser(description="Interactive frame-by-frame viewer for SkillCorner npz.")
    parser.add_argument("--file", type=str, required=True, help="Path to npz file.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--vel_scale", type=float, default=5.0)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(args.file)

    data = np.load(args.file)
    obs = data["obs"]
    acts = data["act_move"] if "act_move" in data else None
    pass_flag = data["pass_flag"] if "pass_flag" in data else None
    passer_id = data["passer_id"] if "passer_id" in data else None
    receiver_id = data["receiver_id"] if "receiver_id" in data else None
    pass_dir = data["pass_dir"] if "pass_dir" in data else None
    dt = float(data["dt"]) if "dt" in data else 0.1

    total = len(obs)
    idx = max(0, min(args.start, total - 1))
    playing = False

    fig, ax = plt.subplots(figsize=(7, 7))

    def refresh():
        draw_frame(ax, obs, acts, pass_flag, passer_id, receiver_id, pass_dir, idx, dt, args.vel_scale)
        fig.canvas.draw_idle()
        if pass_flag is not None and pass_flag[idx] > 0.5:
            p = int(passer_id[idx]) if passer_id is not None else -1
            r = int(receiver_id[idx]) if receiver_id is not None else -1
            d = int(pass_dir[idx]) if pass_dir is not None else -1
            print(f"[Pass] frame {idx}: passer={p} receiver={r} dir={d}")

    def on_key(event):
        nonlocal idx, playing
        if event.key == "right":
            idx = min(total - 1, idx + args.step)
            refresh()
        elif event.key == "left":
            idx = max(0, idx - args.step)
            refresh()
        elif event.key == " ":
            playing = not playing
        elif event.key in ("q", "escape"):
            plt.close(fig)

    def on_timer():
        nonlocal idx
        if playing:
            idx = min(total - 1, idx + args.step)
            refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(on_timer)
    timer.start()

    refresh()
    print("Controls: Left/Right to step, Space to play/pause, Q/Esc to quit.")
    plt.show()


if __name__ == "__main__":
    main()
