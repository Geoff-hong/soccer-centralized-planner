import argparse
import math
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from train.model import SoccerPolicy
from scripts.skillcorner_local_utils import resolve_opendata_root, build_tracking_wide_df


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _infer_move_horizon(state_dict):
    for key in ("head_move.2.weight", "head_move.2.bias"):
        if key in state_dict:
            out_dim = int(state_dict[key].shape[0])
            if out_dim % 2 == 0 and out_dim > 0:
                return max(1, out_dim // 2)
            return 1
    return 1


class SlotAllocator:
    def __init__(self, initial_frame_row, all_player_ids, team_prefix="home"):
        self.slots = [None] * 11
        self.team_prefix = team_prefix
        self.all_ids = all_player_ids
        active_ids = []
        for pid in self.all_ids:
            val = initial_frame_row.get(f"{team_prefix}_{pid}_x", np.nan)
            if not pd.isna(val):
                active_ids.append(pid)
        for i in range(min(len(active_ids), 11)):
            self.slots[i] = active_ids[i]

    def update(self, row):
        empty_slots = []
        for i, pid in enumerate(self.slots):
            if pid is None:
                empty_slots.append(i)
                continue
            val = row.get(f"{self.team_prefix}_{pid}_x", np.nan)
            if pd.isna(val):
                empty_slots.append(i)
        if not empty_slots:
            return self.slots
        candidates = [pid for pid in self.all_ids if pid not in self.slots]
        new_active = []
        for pid in candidates:
            val = row.get(f"{self.team_prefix}_{pid}_x", np.nan)
            if not pd.isna(val):
                new_active.append(pid)
        for pid in new_active:
            if empty_slots:
                idx = empty_slots.pop(0)
                self.slots[idx] = pid
        return self.slots


class PossessionManager:
    def __init__(self, dist_thr=2.0, hysteresis=5):
        self.dist_thr = dist_thr
        self.hysteresis = hysteresis
        self.current_attacking_team = None
        self.consecutive_frames_opp = 0
        self.initialized = False

    def reset(self):
        self.current_attacking_team = None
        self.consecutive_frames_opp = 0
        self.initialized = False

    def update(self, row, home_ids, away_ids):
        ball_x = row.get("ball_x", np.nan)
        ball_y = row.get("ball_y", np.nan)
        if pd.isna(ball_x) or pd.isna(ball_y):
            return self.current_attacking_team

        min_dist_home = float("inf")
        for pid in home_ids:
            px = row.get(f"home_{pid}_x", np.nan)
            py = row.get(f"home_{pid}_y", np.nan)
            if pd.isna(px) or pd.isna(py):
                continue
            d = (px - ball_x) ** 2 + (py - ball_y) ** 2
            if d < min_dist_home:
                min_dist_home = d
        min_dist_home = math.sqrt(min_dist_home) if min_dist_home < float("inf") else float("inf")

        min_dist_away = float("inf")
        for pid in away_ids:
            px = row.get(f"away_{pid}_x", np.nan)
            py = row.get(f"away_{pid}_y", np.nan)
            if pd.isna(px) or pd.isna(py):
                continue
            d = (px - ball_x) ** 2 + (py - ball_y) ** 2
            if d < min_dist_away:
                min_dist_away = d
        min_dist_away = math.sqrt(min_dist_away) if min_dist_away < float("inf") else float("inf")

        if not self.initialized:
            self.current_attacking_team = "Home" if min_dist_home < min_dist_away else "Away"
            self.initialized = True
            return self.current_attacking_team

        home_has_ball = min_dist_home < self.dist_thr
        away_has_ball = min_dist_away < self.dist_thr

        if self.current_attacking_team == "Home":
            if away_has_ball and not home_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            if self.consecutive_frames_opp > self.hysteresis:
                self.current_attacking_team = "Away"
                self.consecutive_frames_opp = 0
        else:
            if home_has_ball and not away_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            if self.consecutive_frames_opp > self.hysteresis:
                self.current_attacking_team = "Home"
                self.consecutive_frames_opp = 0
        return self.current_attacking_team


class TrackingDrivenVis:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = self._load_model(args.checkpoint)

        opendata_root = resolve_opendata_root(args.opendata_root)
        self.tracking_df, pitch = build_tracking_wide_df(
            match_id=args.match_id,
            opendata_root=opendata_root,
            limit=args.limit,
            sample_rate=args.sample_rate,
            transform_to_metric=True,
        )
        self.pitch_length = float(pitch["pitch_length"])
        self.pitch_width = float(pitch["pitch_width"])
        self.half_length = self.pitch_length / 2.0
        self.half_width = self.pitch_width / 2.0

        self.frame_index = list(self.tracking_df.index)
        if not self.frame_index:
            raise RuntimeError("No frames loaded from tracking data.")

        home_cols = [c for c in self.tracking_df.columns if c.startswith("home_") and c.endswith("_x")]
        away_cols = [c for c in self.tracking_df.columns if c.startswith("away_") and c.endswith("_x")]
        self.home_ids = [c.split("_")[1] for c in home_cols]
        self.away_ids = [c.split("_")[1] for c in away_cols]

        first_row = self.tracking_df.iloc[0]
        self.home_allocator = SlotAllocator(first_row, self.home_ids, "home")
        self.away_allocator = SlotAllocator(first_row, self.away_ids, "away")
        self.possession_manager = PossessionManager(
            dist_thr=args.possession_dist_thr,
            hysteresis=args.possession_hysteresis,
        )

        self.home_attack_dir_p1 = self._detect_home_attack_dir()

        self.obs_history = deque(maxlen=args.obs_history)
        self.last_vel_cmds = None
        self.last_attacking_team = None
        self.attackers_pos = None

        self._initialize_state(args.start_frame)

    def _load_model(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        state_dict = state.get("model", state)
        state_dict = _strip_module_prefix(state_dict)
        move_horizon = _infer_move_horizon(state_dict)
        model = SoccerPolicy(
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.3,
            input_dim=3 * self.args.obs_history,
            move_horizon=move_horizon,
        ).to(self.device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[Warn] load_state_dict mismatch. Missing={missing} Unexpected={unexpected}")
        model.eval()
        return model

    def _detect_home_attack_dir(self):
        period1 = self.tracking_df[self.tracking_df["period_id"] == 1]
        if period1.empty:
            return 1
        init_frames = period1.iloc[:10]
        home_x_vals = []
        for pid in self.home_ids:
            vals = init_frames[f"home_{pid}_x"].values
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                home_x_vals.extend(vals)
        if not home_x_vals:
            return 1
        avg_home_x = float(np.mean(home_x_vals))
        return 1 if avg_home_x < self.half_length else -1

    def _normalize_pos(self, pos, flip):
        x = (pos[0] - self.half_length) / self.half_length
        y = (pos[1] - self.half_width) / self.half_width
        if flip:
            return np.array([-x, -y], dtype=np.float32)
        return np.array([x, y], dtype=np.float32)

    def _normalize_vel(self, vel, flip):
        if flip:
            return np.array([-vel[0], -vel[1]], dtype=np.float32)
        return np.array([vel[0], vel[1]], dtype=np.float32)

    def _get_team_positions(self, row, prefix, slots, fallback=None):
        positions = []
        for i, pid in enumerate(slots):
            if pid is None:
                positions.append(fallback[i] if fallback is not None else np.array([np.nan, np.nan]))
                continue
            px = row.get(f"{prefix}_{pid}_x", np.nan)
            py = row.get(f"{prefix}_{pid}_y", np.nan)
            if pd.isna(px) or pd.isna(py):
                if fallback is not None:
                    positions.append(fallback[i])
                else:
                    positions.append(np.array([np.nan, np.nan]))
            else:
                positions.append(np.array([px, py], dtype=np.float32))
        return np.stack(positions, axis=0)

    def _initialize_state(self, start_frame):
        start_frame = max(0, min(start_frame, len(self.frame_index) - 1))
        row = self.tracking_df.iloc[start_frame]

        home_slots = self.home_allocator.update(row)
        away_slots = self.away_allocator.update(row)

        attacking_team = self.possession_manager.update(row, self.home_ids, self.away_ids)
        if attacking_team is None:
            attacking_team = "Home"

        if attacking_team == "Home":
            team_prefix = "home"
            teammate_slots = home_slots
        else:
            team_prefix = "away"
            teammate_slots = away_slots

        self.attackers_pos = self._get_team_positions(row, team_prefix, teammate_slots)
        self.last_attacking_team = attacking_team

        obs_now = self._build_obs(row, attacking_team, home_slots, away_slots)
        for _ in range(self.args.obs_history):
            self.obs_history.append(obs_now.copy())

    def _build_obs(self, row, attacking_team, home_slots, away_slots):
        period = row.get("period_id", 1)
        current_home_dir = self.home_attack_dir_p1 if period == 1 else -self.home_attack_dir_p1
        if attacking_team == "Home":
            flip = current_home_dir == -1
            teammate_slots = home_slots
            opponent_slots = away_slots
            team_prefix = "home"
            opp_prefix = "away"
        else:
            flip = (-current_home_dir) == -1
            teammate_slots = away_slots
            opponent_slots = home_slots
            team_prefix = "away"
            opp_prefix = "home"

        ball_x = row.get("ball_x", np.nan)
        ball_y = row.get("ball_y", np.nan)
        ball_pos = np.array([ball_x, ball_y], dtype=np.float32)

        current_obs = []
        # Teammates (model-controlled positions)
        for i, pid in enumerate(teammate_slots):
            if pid is None or self.attackers_pos is None:
                current_obs.append([0, 0, 0])
                continue
            px, py = self.attackers_pos[i]
            if np.isnan(px) or np.isnan(py) or np.isnan(ball_pos).any():
                current_obs.append([0, 0, 0])
                continue
            dist = np.linalg.norm(self.attackers_pos[i] - ball_pos)
            has_ball = 1.0 if dist < self.args.possession_dist_thr else 0.0
            norm = self._normalize_pos(self.attackers_pos[i], flip)
            current_obs.append([norm[0], norm[1], has_ball])

        # Opponents (tracking positions)
        for pid in opponent_slots:
            if pid is None:
                current_obs.append([0, 0, 0])
                continue
            px = row.get(f"{opp_prefix}_{pid}_x", np.nan)
            py = row.get(f"{opp_prefix}_{pid}_y", np.nan)
            if pd.isna(px) or pd.isna(py) or np.isnan(ball_pos).any():
                current_obs.append([0, 0, 0])
                continue
            dist = math.hypot(px - ball_pos[0], py - ball_pos[1])
            has_ball = 1.0 if dist < self.args.possession_dist_thr else 0.0
            norm = self._normalize_pos(np.array([px, py], dtype=np.float32), flip)
            current_obs.append([norm[0], norm[1], has_ball])

        # Ball
        if np.isnan(ball_pos).any():
            current_obs.append([0, 0, 0])
        else:
            norm_ball = self._normalize_pos(ball_pos, flip)
            current_obs.append([norm_ball[0], norm_ball[1], 0.0])

        return np.array(current_obs, dtype=np.float32)

    def _get_stacked_obs(self, obs_now):
        self.obs_history.append(obs_now)
        if len(self.obs_history) < self.args.obs_history:
            pad = self.args.obs_history - len(self.obs_history)
            for _ in range(pad):
                self.obs_history.appendleft(obs_now.copy())
        obs_stack = np.stack(list(self.obs_history), axis=0)
        return obs_stack.transpose(1, 0, 2).reshape(23, self.args.obs_history * 3)

    def step(self, frame_idx):
        row = self.tracking_df.iloc[frame_idx]
        home_slots = self.home_allocator.update(row)
        away_slots = self.away_allocator.update(row)

        attacking_team = self.possession_manager.update(row, self.home_ids, self.away_ids)
        if attacking_team is None:
            attacking_team = self.last_attacking_team or "Home"

        period = row.get("period_id", 1)
        current_home_dir = self.home_attack_dir_p1 if period == 1 else -self.home_attack_dir_p1
        if attacking_team == "Home":
            flip = current_home_dir == -1
            teammate_slots = home_slots
            opponent_slots = away_slots
            opp_prefix = "away"
        else:
            flip = (-current_home_dir) == -1
            teammate_slots = away_slots
            opponent_slots = home_slots
            opp_prefix = "home"

        # Reset attackers to tracking positions when possession switches
        if attacking_team != self.last_attacking_team or self.attackers_pos is None:
            team_prefix = "home" if attacking_team == "Home" else "away"
            self.attackers_pos = self._get_team_positions(row, team_prefix, teammate_slots, fallback=self.attackers_pos)
            self.last_attacking_team = attacking_team

        obs_now = self._build_obs(row, attacking_team, home_slots, away_slots)
        obs_np = self._get_stacked_obs(obs_now)
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            model_out = self.model(obs_tensor)
            if isinstance(model_out, (list, tuple)):
                p_vel = model_out[0]
            else:
                p_vel = model_out
        if p_vel.ndim == 4:
            p_vel = p_vel[:, 0]
        vel_cmds = p_vel.cpu().numpy()[0]

        if self.args.use_move_residual and self.args.obs_history >= 2 and len(self.obs_history) >= 2:
            last = self.obs_history[-1][:11, 0:2]
            prev = self.obs_history[-2][:11, 0:2]
            scale = np.array([self.args.move_pos_scale_x, self.args.move_pos_scale_y], dtype=np.float32)
            base_vel = (last * scale - prev * scale) / max(self.args.dt, 1e-6)
            vel_cmds = vel_cmds + base_vel

        # Unflip velocities back to real coordinates
        vel_real = np.array([self._normalize_vel(v, flip=False) for v in vel_cmds], dtype=np.float32)
        if flip:
            vel_real *= -1.0

        # Clamp max speed
        speeds = np.linalg.norm(vel_real, axis=1)
        if np.any(speeds > self.args.max_player_speed):
            scale = np.minimum(1.0, self.args.max_player_speed / (speeds + 1e-6))
            vel_real = vel_real * scale.reshape(-1, 1)

        self.attackers_pos = self.attackers_pos + vel_real * self.args.dt
        self.attackers_pos = np.clip(self.attackers_pos, [0, 0], [self.pitch_length, self.pitch_width])
        self.last_vel_cmds = vel_real

        # Defenders + ball from tracking (hardcode)
        defenders_pos = self._get_team_positions(row, opp_prefix, opponent_slots)
        ball_pos = np.array([row.get("ball_x", np.nan), row.get("ball_y", np.nan)], dtype=np.float32)

        return defenders_pos, ball_pos

    def animate(self):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.set_facecolor("#4B8B3B")
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.set_aspect("equal")

        # Pitch
        ax.axvline(x=self.pitch_length / 2, color="white", linestyle="--", alpha=0.5)
        ax.add_patch(Rectangle((0, 0), self.pitch_length, self.pitch_width, fill=False, ec="white", lw=2))
        ax.add_patch(
            Rectangle((self.pitch_length - 16.5, self.pitch_width / 2 - 20), 16.5, 40, fill=False, ec="white")
        )

        att_dots = ax.scatter([], [], c="red", s=80, edgecolors="white", label="Model Attackers")
        def_dots = ax.scatter([], [], c="blue", s=80, edgecolors="white", label="Tracking Defenders")
        ball_dot = ax.scatter([], [], c="white", s=50, edgecolors="black", zorder=10, label="Ball")
        # Initialize quiver with attacker positions to avoid size mismatch
        init_vel = np.zeros((self.attackers_pos.shape[0], 2), dtype=np.float32)
        quiver = ax.quiver(
            self.attackers_pos[:, 0],
            self.attackers_pos[:, 1],
            init_vel[:, 0],
            init_vel[:, 1],
            color="yellow",
            scale=20,
            width=0.003,
        )
        ax.legend(loc="upper left")

        title_text = ax.text(0.5, 1.02, "Initializing...", transform=ax.transAxes, ha="center", fontsize=12)

        start = self.args.start_frame
        end = min(start + self.args.num_frames, len(self.frame_index))

        def init():
            return att_dots, def_dots, ball_dot, quiver, title_text

        def update(frame_offset):
            frame_idx = start + frame_offset
            defenders_pos, ball_pos = self.step(frame_idx)

            att_dots.set_offsets(self.attackers_pos)
            def_dots.set_offsets(defenders_pos)
            if not np.isnan(ball_pos).any():
                ball_dot.set_offsets(ball_pos.reshape(1, -1))
            else:
                ball_dot.set_offsets(np.array([[np.nan, np.nan]]))

            quiver.set_offsets(self.attackers_pos)
            if self.last_vel_cmds is not None and self.last_vel_cmds.shape[0] == self.attackers_pos.shape[0]:
                quiver.set_UVC(self.last_vel_cmds[:, 0], self.last_vel_cmds[:, 1])
            else:
                zeros = np.zeros(self.attackers_pos.shape[0], dtype=np.float32)
                quiver.set_UVC(zeros, zeros)

            title_text.set_text(f"Frame {frame_idx} / {self.frame_index[end-1]}")

            if self.args.speed_log_interval > 0 and frame_offset % self.args.speed_log_interval == 0:
                speeds = np.linalg.norm(self.last_vel_cmds, axis=1) if self.last_vel_cmds is not None else np.array([0])
                print(
                    f"[Frame {frame_idx}] Attacker speed avg={float(np.mean(speeds)):.2f} m/s, max={float(np.max(speeds)):.2f} m/s"
                )

            return att_dots, def_dots, ball_dot, quiver, title_text

        frames = max(1, end - start)
        if self.args.save_gif:
            print(f"Start Recording {frames} frames...")
            ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
            writer = animation.PillowWriter(fps=self.args.fps)
            ani.save(self.args.gif_path, writer=writer)
            print(f"Done! GIF saved to {self.args.gif_path}")
        else:
            print("Starting Live Animation...")
            ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, interval=1000 / self.args.fps, blit=False)
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Tracking-driven visualization for SkillCorner matches.")
    parser.add_argument("--match_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opendata_root", type=str, default="")
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--obs_history", type=int, default=5)
    parser.add_argument("--use_move_residual", type=int, default=1, help="1 to add base velocity (match training), 0 to use raw head output")
    parser.add_argument("--move_pos_scale_x", type=float, default=52.5)
    parser.add_argument("--move_pos_scale_y", type=float, default=34.0)
    parser.add_argument("--possession_dist_thr", type=float, default=2.0)
    parser.add_argument("--possession_hysteresis", type=int, default=5)
    parser.add_argument("--max_player_speed", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_path", type=str, default="output/skillcorner_tracking_vis.gif")
    parser.add_argument("--speed_log_interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sample_rate and args.dt:
        # Keep dt consistent with sample_rate
        args.dt = float(args.sample_rate)
        if args.fps == 0:
            args.fps = int(round(1.0 / args.dt))
    os.makedirs(os.path.dirname(args.gif_path), exist_ok=True)
    vis = TrackingDrivenVis(args)
    vis.animate()
