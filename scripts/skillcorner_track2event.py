import argparse
import os
import warnings

import numpy as np
import pandas as pd

from skillcorner_local_utils import (
    build_tracking_wide_df,
    get_match_paths,
    load_match_json,
    resolve_opendata_root,
)

warnings.filterwarnings("ignore")

# ============================
# Parameters for tracking-based inference
# ============================
POSSESSION_DIST = 3.0
MIN_POSSESSION_FRAMES = 1
MAX_FLIGHT_DURATION = 150
MAX_DEAD_BALL_RATIO = 0.8
GAP_FILL_FRAMES = 10


def _extract_player_ids(cols, prefix):
    ids = []
    for c in cols:
        if not c.startswith(prefix + "_") or not c.endswith("_x"):
            continue
        ids.append(c[len(prefix) + 1 : -2])
    return ids


def generate_event_data_from_tracking(tracking_df):
    print("--- Starting Event Generation (Tracking Inference) ---")

    cols_to_fix = [c for c in tracking_df.columns if c.endswith("_x") or c.endswith("_y")]
    df_calc = tracking_df.copy()
    df_calc[cols_to_fix] = df_calc[cols_to_fix].interpolate(method="linear", limit=3)

    ball_x = df_calc["ball_x"]
    ball_y = df_calc["ball_y"]

    home_cols = [c for c in df_calc.columns if c.startswith("home_") and c.endswith("_x")]
    away_cols = [c for c in df_calc.columns if c.startswith("away_") and c.endswith("_x")]
    home_ids = _extract_player_ids(home_cols, "home")
    away_ids = _extract_player_ids(away_cols, "away")

    all_player_ids = [f"Home_{pid}" for pid in home_ids] + [f"Away_{pid}" for pid in away_ids]

    distances = []
    for pid in home_ids:
        d = np.sqrt((df_calc[f"home_{pid}_x"] - ball_x) ** 2 + (df_calc[f"home_{pid}_y"] - ball_y) ** 2)
        distances.append(d.values)
    for pid in away_ids:
        d = np.sqrt((df_calc[f"away_{pid}_x"] - ball_x) ** 2 + (df_calc[f"away_{pid}_y"] - ball_y) ** 2)
        distances.append(d.values)

    dist_matrix = np.array(distances).T
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)

    min_dist_idx = np.argmin(dist_matrix, axis=1)
    min_dists = np.min(dist_matrix, axis=1)

    possessor_array = np.full(len(tracking_df), -1, dtype=int)
    has_possession_mask = min_dists < POSSESSION_DIST
    possessor_array[has_possession_mask] = min_dist_idx[has_possession_mask]

    possessor_series = pd.Series(possessor_array)
    possessor_filled = possessor_series.replace(-1, np.nan).ffill(limit=GAP_FILL_FRAMES).fillna(-1).astype(int)
    possessor_list = possessor_filled.tolist()

    segments = []
    current_seg_pid = -1
    current_seg_start = 0

    for i, pid in enumerate(possessor_list):
        if i == 0:
            current_seg_pid = pid
            continue
        if pid != current_seg_pid:
            if current_seg_pid != -1:
                if (i - current_seg_start) >= MIN_POSSESSION_FRAMES:
                    segments.append({"player_idx": current_seg_pid, "start": current_seg_start, "end": i - 1})
            current_seg_pid = pid
            current_seg_start = i

    pass_events = []
    raw_ball_x = tracking_df["ball_x"]

    for i in range(len(segments) - 1):
        seg_from = segments[i]
        seg_to = segments[i + 1]

        flight_start = seg_from["end"]
        flight_end = seg_to["start"]
        duration = flight_end - flight_start

        if duration < 1 or duration > MAX_FLIGHT_DURATION:
            continue

        flight_slice = raw_ball_x.iloc[flight_start:flight_end]
        nan_count = flight_slice.isna().sum()
        if len(flight_slice) > 0 and (nan_count / len(flight_slice) > MAX_DEAD_BALL_RATIO):
            continue

        p_from_name = all_player_ids[seg_from["player_idx"]]
        p_to_name = all_player_ids[seg_to["player_idx"]]

        team_from, pid_from = p_from_name.split("_", 1)
        team_to, pid_to = p_to_name.split("_", 1)

        if pid_from == pid_to and team_from == team_to:
            continue

        event_type = "PASS" if team_from == team_to else "TURNOVER"
        pass_events.append(
            {
                "start_frame": int(tracking_df.index[flight_start]),
                "end_frame": int(tracking_df.index[flight_end]),
                "type": event_type,
                "from_team": team_from,
                "from_player": pid_from,
                "to_team": team_to,
                "to_player": pid_to,
                "duration_frames": duration,
            }
        )

    cols = ["start_frame", "end_frame", "type", "from_team", "from_player", "to_team", "to_player", "duration_frames"]
    return pd.DataFrame(pass_events, columns=cols)


def generate_event_data_from_dynamic(dynamic_csv, match_meta, only_successful=False):
    print("--- Starting Event Generation (Dynamic Events) ---")
    df = pd.read_csv(dynamic_csv, low_memory=False)
    ps = df[df["event_type"] == "player_possession"].copy()
    ps = ps[ps["end_type"].astype(str).str.lower() == "pass"]
    if only_successful:
        ps = ps[ps["pass_outcome"].astype(str).str.lower() == "successful"]

    home_team_id = match_meta.get("home_team", {}).get("id")
    away_team_id = match_meta.get("away_team", {}).get("id")
    team_map = {home_team_id: "Home", away_team_id: "Away"}
    player_team = {}
    for p in match_meta.get("players", []):
        pid = p.get("id")
        team_id = p.get("team_id")
        if pid is not None and team_id is not None:
            player_team[int(pid)] = "Home" if team_id == home_team_id else "Away"

    events = []
    for _, row in ps.iterrows():
        frame_end = int(row["frame_end"]) if not pd.isna(row["frame_end"]) else None
        if frame_end is None:
            continue
        from_player = int(row["player_id"]) if not pd.isna(row["player_id"]) else None
        if from_player is None:
            continue
        from_team = team_map.get(row["team_id"], player_team.get(from_player, "Home"))

        to_player = row.get("player_targeted_id", None)
        to_player_str = ""
        to_team = from_team
        if pd.notna(to_player):
            to_player = int(to_player)
            to_player_str = str(to_player)
            to_team = player_team.get(to_player, from_team)

        events.append(
            {
                "start_frame": frame_end,
                "end_frame": frame_end,
                "type": "PASS",
                "from_team": from_team,
                "from_player": str(from_player),
                "to_team": to_team,
                "to_player": to_player_str,
                "pass_outcome": row.get("pass_outcome", ""),
            }
        )

    cols = ["start_frame", "end_frame", "type", "from_team", "from_player", "to_team", "to_player", "pass_outcome"]
    return pd.DataFrame(events, columns=cols)


def main():
    parser = argparse.ArgumentParser(description="Generate pass events from SkillCorner open data.")
    parser.add_argument("--match_id", type=str, default="")
    parser.add_argument("--source", type=str, choices=["dynamic", "tracking"], default="dynamic")
    parser.add_argument("--only_successful", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_rate", type=float, default=1 / 10)
    parser.add_argument("--opendata_root", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    opendata_root = resolve_opendata_root(args.opendata_root)
    if not args.match_id:
        raise ValueError("Please provide --match_id")

    match_meta = load_match_json(args.match_id, opendata_root)
    paths = get_match_paths(args.match_id, opendata_root)

    if args.source == "dynamic":
        df_events = generate_event_data_from_dynamic(
            dynamic_csv=paths["dynamic_events_csv"],
            match_meta=match_meta,
            only_successful=args.only_successful,
        )
    else:
        tracking_df, _ = build_tracking_wide_df(
            match_id=args.match_id,
            opendata_root=opendata_root,
            limit=args.limit,
            sample_rate=args.sample_rate,
            transform_to_metric=True,
        )
        df_events = generate_event_data_from_tracking(tracking_df)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(data_dir, f"skillcorner_match{args.match_id}_inferred_events.csv")

    if not df_events.empty:
        n_passes = len(df_events[df_events["type"] == "PASS"])
        print(f"Found {n_passes} pass events. Saving to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_events.to_csv(output_file, index=False)
    else:
        print("No pass events generated!")


if __name__ == "__main__":
    main()
