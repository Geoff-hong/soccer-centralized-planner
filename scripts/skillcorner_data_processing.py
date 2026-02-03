import argparse
import os
import warnings

import numpy as np
import pandas as pd

from skillcorner_local_utils import build_tracking_wide_df, resolve_opendata_root

warnings.filterwarnings("ignore")

# ==========================================
# Configuration
# ==========================================
KICK_SECTORS = 12
POSSESSION_DIST_THR = 3.0
POSSESSION_HYSTERESIS = 5
KICK_EXPAND_FRAMES = 2
MAX_SPEED_MS = 12.0
DROP_SPEED_MS = 25.0


class SlotAllocator:
    def __init__(self, initial_frame_row, all_player_ids, team_prefix="home"):
        self.slots = [None] * 11
        self.team_prefix = team_prefix
        self.all_ids = all_player_ids
        active_ids = []
        for pid in self.all_ids:
            if not pd.isna(initial_frame_row.get(f"{team_prefix}_{pid}_x", np.nan)):
                active_ids.append(pid)
        for i in range(min(len(active_ids), 11)):
            self.slots[i] = active_ids[i]

    def update(self, row):
        empty_slot_indices = []
        for i, pid in enumerate(self.slots):
            if pid is None:
                empty_slot_indices.append(i)
                continue
            val = row.get(f"{self.team_prefix}_{pid}_x", np.nan)
            if pd.isna(val):
                empty_slot_indices.append(i)

        if not empty_slot_indices:
            return self.slots

        candidates = [pid for pid in self.all_ids if pid not in self.slots]
        new_active_players = []
        for pid in candidates:
            if not pd.isna(row.get(f"{self.team_prefix}_{pid}_x", np.nan)):
                new_active_players.append(pid)

        for new_pid in new_active_players:
            if empty_slot_indices:
                idx = empty_slot_indices.pop(0)
                self.slots[idx] = new_pid
        return self.slots


def get_kick_sector(dx, dy, n_sectors=12):
    angle = np.arctan2(dy, dx)
    if angle < 0:
        angle += 2 * np.pi
    sector_size = 2 * np.pi / n_sectors
    return int(angle / sector_size) % n_sectors


class PossessionManager:
    def __init__(self):
        self.current_attacking_team = None
        self.consecutive_frames_opp = 0
        self.initialized = False

    def update(self, row, home_ids, away_ids):
        ball_x, ball_y = row["ball_x"], row["ball_y"]

        min_dist_home = float("inf")
        for pid in home_ids:
            px, py = row.get(f"home_{pid}_x", np.nan), row.get(f"home_{pid}_y", np.nan)
            if pd.notna(px):
                d = (px - ball_x) ** 2 + (py - ball_y) ** 2
                if d < min_dist_home:
                    min_dist_home = d
        min_dist_home = np.sqrt(min_dist_home)

        min_dist_away = float("inf")
        for pid in away_ids:
            px, py = row.get(f"away_{pid}_x", np.nan), row.get(f"away_{pid}_y", np.nan)
            if pd.notna(px):
                d = (px - ball_x) ** 2 + (py - ball_y) ** 2
                if d < min_dist_away:
                    min_dist_away = d
        min_dist_away = np.sqrt(min_dist_away)

        if not self.initialized:
            self.current_attacking_team = "Home" if min_dist_home < min_dist_away else "Away"
            self.initialized = True
            return self.current_attacking_team

        home_has_ball = min_dist_home < POSSESSION_DIST_THR
        away_has_ball = min_dist_away < POSSESSION_DIST_THR

        if self.current_attacking_team == "Home":
            if away_has_ball and not home_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            if self.consecutive_frames_opp > POSSESSION_HYSTERESIS:
                self.current_attacking_team = "Away"
                self.consecutive_frames_opp = 0
        else:
            if home_has_ball and not away_has_ball:
                self.consecutive_frames_opp += 1
            else:
                self.consecutive_frames_opp = 0
            if self.consecutive_frames_opp > POSSESSION_HYSTERESIS:
                self.current_attacking_team = "Home"
                self.consecutive_frames_opp = 0

        return self.current_attacking_team


def _extract_player_ids(cols, prefix):
    ids = []
    for c in cols:
        if not c.startswith(prefix + "_") or not c.endswith("_x"):
            continue
        ids.append(c[len(prefix) + 1 : -2])
    return ids


def process_match_data(match_id, opendata_root, sample_rate, limit, transform_metric, event_file, output_file):
    print(f"\n[1/5] Loading SkillCorner Match {match_id} Tracking Data...")
    df, pitch = build_tracking_wide_df(
        match_id=match_id,
        opendata_root=opendata_root,
        limit=limit,
        sample_rate=sample_rate,
        transform_to_metric=transform_metric,
    )
    df = df.sort_index()

    pitch_length = pitch.get("pitch_length", 105.0)
    pitch_width = pitch.get("pitch_width", 68.0)
    half_length = pitch_length / 2.0
    half_width = pitch_width / 2.0

    home_cols = [c for c in df.columns if c.startswith("home_") and c.endswith("_x")]
    away_cols = [c for c in df.columns if c.startswith("away_") and c.endswith("_x")]
    home_ids_all = _extract_player_ids(home_cols, "home")
    away_ids_all = _extract_player_ids(away_cols, "away")

    print("[1.5/5] Loading Inferred Events...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    if not event_file:
        event_file = os.path.join(data_dir, f"skillcorner_match{match_id}_inferred_events.csv")
    if not os.path.exists(event_file):
        raise FileNotFoundError(f"Event file {event_file} not found! Please run skillcorner_track2event.py first.")
    df_events = pd.read_csv(event_file)

    print("[2/5] Smoothing & Velocity Calculation...")
    coord_cols = home_cols + [c.replace("_x", "_y") for c in home_cols] + \
                 away_cols + [c.replace("_x", "_y") for c in away_cols] + \
                 ["ball_x", "ball_y"]

    df[coord_cols] = df[coord_cols].interpolate(method="linear", limit=5)
    df_smooth = df.copy()
    df_smooth[coord_cols] = df[coord_cols].rolling(window=5, min_periods=1).mean()
    df_vel = df_smooth[coord_cols].diff().fillna(0.0)

    print("[3/5] Mapping Events from CSV...")
    kick_events = {}
    all_passes = df_events[df_events["type"] == "PASS"]
    for _, row in all_passes.iterrows():
        fid = int(row["start_frame"])
        pid_raw = row["from_player"]
        if pd.isna(pid_raw):
            continue
        pid = str(int(float(pid_raw)))
        team = row["from_team"]
        to_pid = ""
        if "to_player" in row and not pd.isna(row["to_player"]):
            to_pid = str(int(float(row["to_player"])))
        to_team = row["to_team"] if "to_team" in row else team
        for delta in range(-KICK_EXPAND_FRAMES, KICK_EXPAND_FRAMES + 1):
            f = fid + delta
            if f < 0:
                continue
            kick_events.setdefault(f, []).append(
                {"pid": pid, "team": team, "to_pid": to_pid, "to_team": to_team}
            )

    print("[3.5/5] Detecting Attack Direction...")
    home_attack_dir_p1 = 1
    period1_data = df_smooth[df_smooth["period_id"] == 1] if "period_id" in df_smooth.columns else df_smooth
    if not period1_data.empty:
        init_frames = period1_data.iloc[:10]
        home_x_vals = []
        for pid in home_ids_all:
            vals = init_frames[f"home_{pid}_x"].values
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                home_x_vals.extend(valid_vals)
        if len(home_x_vals) > 0:
            avg_home_x = np.mean(home_x_vals)
            home_attack_dir_p1 = 1 if avg_home_x < half_length else -1

    lookahead_frames = max(5, int(round(1.0 / sample_rate)))
    future_pass_frames = lookahead_frames

    print("[4/5] Generating Attacker-Centric Tensors (With Filtering)...")
    obs_list = []
    move_list = []
    pass_flag_list = []
    passer_id_list = []
    pass_dir_list = []
    receiver_id_list = []

    home_allocator = SlotAllocator(df_smooth.iloc[0], home_ids_all, "home")
    away_allocator = SlotAllocator(df_smooth.iloc[0], away_ids_all, "away")
    possession_manager = PossessionManager()

    valid_indices = df_smooth.index
    filtered_count = 0
    dropped_speed_frames = 0
    clipped_vectors = 0
    total_count = 0

    for i, fid in enumerate(valid_indices):
        if i % 1000 == 0:
            print(f"   Processing frame {fid}/{valid_indices[-1]}...", end="\r")

        row_raw = df_smooth.loc[fid]
        row_vel = df_vel.loc[fid]

        ball_x, ball_y = row_raw["ball_x"], row_raw["ball_y"]
        if pd.isna(ball_x) or pd.isna(ball_y):
            filtered_count += 1
            continue
        if not (-2.0 <= ball_x <= pitch_length + 2.0 and -2.0 <= ball_y <= pitch_width + 2.0):
            filtered_count += 1
            continue

        period = row_raw["period_id"] if "period_id" in row_raw else 1
        attacking_team = possession_manager.update(row_raw, home_ids_all, away_ids_all)

        home_slots = home_allocator.update(row_raw)
        away_slots = away_allocator.update(row_raw)

        current_home_dir = home_attack_dir_p1 if period == 1 else -home_attack_dir_p1
        flip = False
        if attacking_team == "Home":
            if current_home_dir == -1:
                flip = True
        else:
            if -current_home_dir == -1:
                flip = True

        if attacking_team == "Home":
            teammate_slots = home_slots
            opponent_slots = away_slots
            team_prefix = "home"
            opp_prefix = "away"
        else:
            teammate_slots = away_slots
            opponent_slots = home_slots
            team_prefix = "away"
            opp_prefix = "home"

        def norm_pos_x(val):
            n = (val - half_length) / half_length
            return -n if flip else n

        def norm_pos_y(val):
            n = (val - half_width) / half_width
            return -n if flip else n

        def norm_vel(val):
            return -val if flip else val

        # Pre-check speed for teammate slots (drop extreme spikes)
        raw_vels = []
        for pid in teammate_slots:
            if pid is None:
                raw_vels.append((0.0, 0.0))
                continue
            vx_raw = row_vel.get(f"{team_prefix}_{pid}_x", 0.0)
            vy_raw = row_vel.get(f"{team_prefix}_{pid}_y", 0.0)
            raw_vels.append((vx_raw, vy_raw))

        dt = sample_rate
        max_speed = 0.0
        for vx_raw, vy_raw in raw_vels:
            sp = (vx_raw ** 2 + vy_raw ** 2) ** 0.5 / dt
            if sp > max_speed:
                max_speed = sp
        if max_speed > DROP_SPEED_MS:
            dropped_speed_frames += 1
            filtered_count += 1
            continue

        total_count += 1

        current_obs = []
        for pid in teammate_slots:
            if pid is None:
                current_obs.append([0, 0, 0])
                continue
            px = row_raw.get(f"{team_prefix}_{pid}_x", np.nan)
            py = row_raw.get(f"{team_prefix}_{pid}_y", np.nan)
            if pd.isna(px):
                current_obs.append([0, 0, 0])
            else:
                dist = np.sqrt((px - ball_x) ** 2 + (py - ball_y) ** 2)
                has_ball = 1.0 if dist < POSSESSION_DIST_THR else 0.0
                current_obs.append([norm_pos_x(px), norm_pos_y(py), has_ball])

        for pid in opponent_slots:
            if pid is None:
                current_obs.append([0, 0, 0])
                continue
            px = row_raw.get(f"{opp_prefix}_{pid}_x", np.nan)
            py = row_raw.get(f"{opp_prefix}_{pid}_y", np.nan)
            if pd.isna(px):
                current_obs.append([0, 0, 0])
            else:
                dist = np.sqrt((px - ball_x) ** 2 + (py - ball_y) ** 2)
                has_ball = 1.0 if dist < POSSESSION_DIST_THR else 0.0
                current_obs.append([norm_pos_x(px), norm_pos_y(py), has_ball])

        current_obs.append([norm_pos_x(ball_x), norm_pos_y(ball_y), 0.0])
        obs_list.append(current_obs)

        current_moves = []
        for vx_raw, vy_raw in raw_vels:
            if vx_raw == 0.0 and vy_raw == 0.0:
                current_moves.append([0, 0])
                continue
            sp = (vx_raw ** 2 + vy_raw ** 2) ** 0.5 / dt
            if sp > MAX_SPEED_MS:
                scale = MAX_SPEED_MS / max(sp, 1e-6)
                vx_raw *= scale
                vy_raw *= scale
                clipped_vectors += 1
            vx_ms = vx_raw / dt
            vy_ms = vy_raw / dt
            current_moves.append([norm_vel(vx_ms), norm_vel(vy_ms)])
        move_list.append(current_moves)

        pass_flag = 0
        passer_slot = -1
        pass_dir = -1
        receiver_slot = -1

        kick_info_list = kick_events.get(fid, [])
        if kick_info_list:
            for kick_info in kick_info_list:
                if kick_info["team"] != attacking_team:
                    continue
                pid = kick_info["pid"]
                if pid not in teammate_slots:
                    continue
                slot_idx = teammate_slots.index(pid)

                px = row_raw.get(f"{team_prefix}_{pid}_x", np.nan)
                py = row_raw.get(f"{team_prefix}_{pid}_y", np.nan)
                if pd.isna(px) or pd.isna(py):
                    continue
                pass_flag = 1
                passer_slot = slot_idx

                future_pos = i + future_pass_frames
                if future_pos < len(df_smooth):
                    future_row = df_smooth.iloc[future_pos]
                else:
                    future_row = None
                if future_row is not None and pd.notna(future_row["ball_x"]):
                    dx = future_row["ball_x"] - row_raw["ball_x"]
                    dy = future_row["ball_y"] - row_raw["ball_y"]
                    pass_dir = int(get_kick_sector(norm_vel(dx), norm_vel(dy), KICK_SECTORS))

                if kick_info["to_team"] == attacking_team:
                    to_pid = kick_info["to_pid"]
                    if to_pid in teammate_slots:
                        receiver_slot = teammate_slots.index(to_pid)
                break

        pass_flag_list.append(pass_flag)
        passer_id_list.append(passer_slot)
        pass_dir_list.append(pass_dir)
        receiver_id_list.append(receiver_slot)

    obs_array = np.array(obs_list, dtype=np.float32)
    move_array = np.array(move_list, dtype=np.float32)
    pass_flag_array = np.array(pass_flag_list, dtype=np.float32)
    passer_id_array = np.array(passer_id_list, dtype=np.int64)
    pass_dir_array = np.array(pass_dir_list, dtype=np.int64)
    receiver_id_array = np.array(receiver_id_list, dtype=np.int64)

    print(f"\n[5/5] Done. Match {match_id}")
    print(f"   Total Valid Frames: {total_count}")
    print(f"   Filtered Frames: {filtered_count} (NaN or Out-of-Bounds)")
    print(f"   Dropped Frames (speed > {DROP_SPEED_MS} m/s): {dropped_speed_frames}")
    print(f"   Clipped Vectors (speed > {MAX_SPEED_MS} m/s): {clipped_vectors}")
    print(f"   Obs Shape: {obs_array.shape}")
    print(f"   Move Shape: {move_array.shape}")
    print(f"   Pass Frames: {int(pass_flag_array.sum())}")

    os.makedirs(data_dir, exist_ok=True)
    if not output_file:
        output_file = os.path.join(data_dir, f"skillcorner_match{match_id}_train_attacker_centric.npz")
    np.savez_compressed(
        output_file,
        obs=obs_array,
        act_move=move_array,
        pass_flag=pass_flag_array,
        passer_id=passer_id_array,
        pass_dir=pass_dir_array,
        receiver_id=receiver_id_array,
        dt=np.array(sample_rate, dtype=np.float32),
        move_unit="mps",
    )
    print(f"   Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkillCorner data processing to attacker-centric format.")
    parser.add_argument("--match_id", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_rate", type=float, default=1 / 10)
    parser.add_argument("--opendata_root", type=str, default="")
    parser.add_argument("--no_transform", action="store_true")
    parser.add_argument("--event_file", type=str, default="", help="Override inferred events CSV path.")
    parser.add_argument("--output", type=str, default="", help="Override output npz path.")
    args = parser.parse_args()

    opendata_root = resolve_opendata_root(args.opendata_root)
    process_match_data(
        match_id=args.match_id,
        opendata_root=opendata_root,
        sample_rate=args.sample_rate,
        limit=args.limit,
        transform_metric=(not args.no_transform),
        event_file=args.event_file,
        output_file=args.output,
    )
