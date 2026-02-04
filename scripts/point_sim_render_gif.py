#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from point_sim_env import PointSim3v3  # noqa: E402
from point_sim_policy import DefenderPolicy, MixedTacticPolicy  # noqa: E402


def _load_episode(npz_path: str):
    data = np.load(npz_path)
    obs = data["obs"]
    dt = data.get("dt", 0.1)
    if isinstance(dt, np.ndarray):
        if dt.ndim == 0:
            dt = float(dt)
        else:
            dt = float(dt.reshape(-1)[0])
    pass_flag = data.get("pass_flag", None)
    passer_id = data.get("passer_id", None)
    receiver_id = data.get("receiver_id", None)
    return obs, float(dt), pass_flag, passer_id, receiver_id


def _generate_episode(seed: int, max_steps: int):
    env = PointSim3v3(dt=0.1, seed=seed)
    policy = MixedTacticPolicy(dt=0.1)
    def_policy = DefenderPolicy()

    env.reset(seed=seed)
    policy.reset(np.random.default_rng(seed + 1))
    def_policy.reset(np.random.default_rng(seed + 2))

    obs_list = []
    for _ in range(max_steps):
        obs_list.append(env.get_obs())
        v_cmd_att, pass_event = policy.act(env)
        v_cmd_def = def_policy.act(env)
        done, _ = env.step(v_cmd_att, v_cmd_def, pass_event)
        if done:
            obs_list.append(env.get_obs())
            break

    return np.asarray(obs_list, dtype=np.float32), env.dt, None, None, None


def _load_multi_episodes(
    data_dir: str,
    num: int,
    shuffle: bool,
    seed: int,
    gap_frames: int,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No npz files found in {data_dir}")
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
    files = files[: max(1, int(num))]

    obs_all = []
    pass_all = []
    passer_all = []
    receiver_all = []
    episode_ids = []

    dt_ref = None
    for ep_idx, fname in enumerate(files):
        obs, dt, pass_flag, passer_id, receiver_id = _load_episode(os.path.join(data_dir, fname))
        if dt_ref is None:
            dt_ref = dt
        elif abs(dt - dt_ref) > 1e-6:
            print(f"[Warn] dt mismatch in {fname}: {dt} vs {dt_ref}")

        obs_all.append(obs)
        if pass_flag is None:
            pass_all.append(np.zeros(obs.shape[0], dtype=np.float32))
            passer_all.append(np.zeros(obs.shape[0], dtype=np.int64))
            receiver_all.append(np.zeros(obs.shape[0], dtype=np.int64))
        else:
            pass_all.append(np.asarray(pass_flag))
            passer_all.append(np.asarray(passer_id))
            receiver_all.append(np.asarray(receiver_id))
        episode_ids.append(np.full(obs.shape[0], ep_idx, dtype=np.int64))

        if gap_frames > 0 and ep_idx < len(files) - 1:
            gap = np.zeros((gap_frames, obs.shape[1], obs.shape[2]), dtype=np.float32)
            obs_all.append(gap)
            pass_all.append(np.zeros(gap_frames, dtype=np.float32))
            passer_all.append(np.zeros(gap_frames, dtype=np.int64))
            receiver_all.append(np.zeros(gap_frames, dtype=np.int64))
            episode_ids.append(np.full(gap_frames, ep_idx, dtype=np.int64))

    obs_cat = np.concatenate(obs_all, axis=0)
    pass_cat = np.concatenate(pass_all, axis=0) if pass_all else None
    passer_cat = np.concatenate(passer_all, axis=0) if passer_all else None
    receiver_cat = np.concatenate(receiver_all, axis=0) if receiver_all else None
    episode_cat = np.concatenate(episode_ids, axis=0)
    return obs_cat, dt_ref or 0.1, pass_cat, passer_cat, receiver_cat, episode_cat


def _render(
    obs: np.ndarray,
    dt: float,
    out_path: str,
    every: int = 1,
    dpi: int = 120,
    pass_flag: Optional[np.ndarray] = None,
    passer_id: Optional[np.ndarray] = None,
    receiver_id: Optional[np.ndarray] = None,
    pass_hold: float = 0.5,
    episode_ids: Optional[np.ndarray] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

    L = 105.0
    W = 68.0
    half_L = L / 2.0
    half_W = W / 2.0
    goal_y = 3.66

    fps = max(int(round(1.0 / dt)), 1)

    obs = obs[::every]
    if pass_flag is not None:
        pass_flag = pass_flag[::every]
    if passer_id is not None:
        passer_id = passer_id[::every]
    if receiver_id is not None:
        receiver_id = receiver_id[::every]
    if episode_ids is not None:
        episode_ids = episode_ids[::every]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
    ax.set_xlim(-half_L, half_L)
    ax.set_ylim(-half_W, half_W)
    ax.set_aspect("equal")
    ax.axis("off")

    # pitch
    pitch = Rectangle((-half_L, -half_W), L, W, linewidth=2, edgecolor="black", facecolor="#2e7d32")
    ax.add_patch(pitch)
    ax.plot([0, 0], [-half_W, half_W], color="white", linewidth=1.5)
    center_circle = plt.Circle((0, 0), 9.15, color="white", fill=False, linewidth=1.5)
    ax.add_patch(center_circle)
    ax.plot([half_L, half_L], [-goal_y, goal_y], color="white", linewidth=3)
    ax.plot([-half_L, -half_L], [-goal_y, goal_y], color="white", linewidth=3)

    att_dots = ax.scatter([], [], s=120, c="#ef5350", edgecolors="white", linewidths=1.5, zorder=3)
    def_dots = ax.scatter([], [], s=120, c="#42a5f5", edgecolors="white", linewidths=1.5, zorder=3)
    ball_dot = ax.scatter([], [], s=60, c="#fdd835", edgecolors="black", linewidths=1.0, zorder=4)

    has_ball_ring = Circle((0, 0), radius=1.2, fill=False, edgecolor="yellow", linewidth=2.0, zorder=5)
    ax.add_patch(has_ball_ring)

    pass_arrow = FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="-|>",
        mutation_scale=16,
        color="#ffb300",
        linewidth=2.0,
        alpha=0.9,
        zorder=6,
    )
    pass_arrow.set_visible(False)
    ax.add_patch(pass_arrow)

    owner_text = ax.text(
        0.01,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        zorder=7,
    )
    pass_text = ax.text(
        0.01,
        0.93,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        zorder=7,
    )
    episode_text = ax.text(
        0.99,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        zorder=7,
    )

    pass_state = {"timer": 0, "src": None, "dst": None}
    pass_display_frames = max(1, int(round(pass_hold / max(dt, 1e-6))))

    def _denorm(xy):
        return np.stack([xy[:, 0] * half_L, xy[:, 1] * half_W], axis=1)

    def init():
        empty = np.zeros((0, 2), dtype=np.float32)
        att_dots.set_offsets(empty)
        def_dots.set_offsets(empty)
        ball_dot.set_offsets(empty)
        has_ball_ring.set_visible(False)
        pass_arrow.set_visible(False)
        owner_text.set_text("")
        pass_text.set_text("")
        episode_text.set_text("")
        return att_dots, def_dots, ball_dot, has_ball_ring, pass_arrow, owner_text, pass_text, episode_text

    def update(frame_idx):
        frame = obs[frame_idx]
        att = _denorm(frame[0:3, 0:2])
        deff = _denorm(frame[3:6, 0:2])
        ball = frame[6, 0:2] * np.array([half_L, half_W], dtype=np.float32)
        att_dots.set_offsets(att)
        def_dots.set_offsets(deff)
        ball_dot.set_offsets([ball])

        has_ball = frame[:6, 2]
        if has_ball.max() > 0.5:
            idx = int(np.argmax(has_ball))
            if idx < 3:
                ring_pos = att[idx]
                owner_text.set_text(f"Owner: A{idx}")
            else:
                ring_pos = deff[idx - 3]
                owner_text.set_text(f"Owner: D{idx - 3}")
            has_ball_ring.center = (ring_pos[0], ring_pos[1])
            has_ball_ring.set_visible(True)
        else:
            has_ball_ring.set_visible(False)
            owner_text.set_text("Owner: None")

        if pass_flag is not None and passer_id is not None and receiver_id is not None:
            if pass_flag[frame_idx] > 0.5:
                src = int(passer_id[frame_idx])
                dst = int(receiver_id[frame_idx])
                if 0 <= src < 3 and 0 <= dst < 3:
                    pass_state["src"] = att[src]
                    pass_state["dst"] = att[dst]
                    pass_state["timer"] = pass_display_frames
                    pass_text.set_text(f"Pass attempt: A{src} -> A{dst}")
                else:
                    pass_state["timer"] = 0
            if pass_state["timer"] > 0 and pass_state["src"] is not None and pass_state["dst"] is not None:
                pass_arrow.set_positions(pass_state["src"], pass_state["dst"])
                pass_arrow.set_visible(True)
                pass_state["timer"] -= 1
            else:
                pass_arrow.set_visible(False)
                if pass_state["timer"] <= 0:
                    pass_text.set_text("")

        if episode_ids is not None:
            episode_text.set_text(f"EP {int(episode_ids[frame_idx]):03d}")

        return att_dots, def_dots, ball_dot, has_ball_ring, pass_arrow, owner_text, pass_text, episode_text

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(obs), interval=1000.0 / fps, blit=True
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_path.endswith(".gif"):
        try:
            anim.save(out_path, writer="pillow", fps=fps)
        except Exception as e:
            raise RuntimeError("Failed to save GIF. Install pillow: pip install pillow") from e
    elif out_path.endswith(".mp4"):
        try:
            anim.save(out_path, writer="ffmpeg", fps=fps)
        except Exception as e:
            raise RuntimeError("Failed to save MP4. Ensure ffmpeg is installed") from e
    else:
        raise ValueError("out must end with .gif or .mp4")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render 3v3 point-sim episode to GIF/MP4")
    parser.add_argument("--npz", type=str, default="", help="Path to an existing episode npz")
    parser.add_argument("--seed", type=int, default=0, help="Seed for generation when --npz not provided")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--out", type=str, default=os.path.join("output", "point_sim_3v3.gif"))
    parser.add_argument("--every", type=int, default=1, help="Render every N frames")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--pass_hold", type=float, default=0.5, help="Seconds to keep pass arrow visible")
    parser.add_argument("--data_dir", type=str, default="", help="Render multiple episodes from dir")
    parser.add_argument("--num", type=int, default=10, help="Number of episodes to render when using --data_dir")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle episodes when using --data_dir")
    parser.add_argument("--gap_frames", type=int, default=3, help="Gap frames between episodes")
    args = parser.parse_args()

    if args.data_dir:
        obs, dt, pass_flag, passer_id, receiver_id, episode_ids = _load_multi_episodes(
            args.data_dir, args.num, args.shuffle, args.seed, args.gap_frames
        )
    elif args.npz:
        obs, dt, pass_flag, passer_id, receiver_id = _load_episode(args.npz)
        episode_ids = None
    else:
        obs, dt, pass_flag, passer_id, receiver_id = _generate_episode(args.seed, args.max_steps)
        episode_ids = None

    _render(
        obs,
        dt,
        args.out,
        every=args.every,
        dpi=args.dpi,
        pass_flag=pass_flag,
        passer_id=passer_id,
        receiver_id=receiver_id,
        pass_hold=args.pass_hold,
        episode_ids=episode_ids,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
