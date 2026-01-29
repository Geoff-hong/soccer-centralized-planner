import argparse
import os

import numpy as np
import pandas as pd

from skillcorner_local_utils import build_tracking_wide_df, get_match_paths, load_match_json, resolve_opendata_root
from skillcorner_track2event import generate_event_data_from_dynamic, generate_event_data_from_tracking


def _normalize_events(df):
    df = df.copy()
    df = df[df["type"] == "PASS"]
    df["start_frame"] = df["start_frame"].astype(int)
    df["from_team"] = df["from_team"].astype(str)
    df["from_player"] = df["from_player"].astype(str)
    df["to_team"] = df["to_team"].astype(str)
    df["to_player"] = df["to_player"].astype(str)
    return df


def match_with_tolerance(dynamic_df, tracking_df, tol):
    used_tracking = set()
    matched = []

    for i, row in dynamic_df.iterrows():
        candidates = tracking_df[
            (tracking_df["from_team"] == row["from_team"]) &
            (tracking_df["from_player"] == row["from_player"]) &
            (np.abs(tracking_df["start_frame"] - row["start_frame"]) <= tol)
        ]
        if candidates.empty:
            continue
        # Prefer same receiver if possible
        same_receiver = candidates[candidates["to_player"] == row["to_player"]]
        if not same_receiver.empty:
            cand = same_receiver.iloc[0]
        else:
            cand = candidates.iloc[0]
        if cand.name in used_tracking:
            continue
        used_tracking.add(cand.name)
        matched.append((i, cand.name))
    return matched, used_tracking


def main():
    parser = argparse.ArgumentParser(description="Compare SkillCorner dynamic events vs tracking-inferred passes.")
    parser.add_argument("--match_id", type=str, required=True)
    parser.add_argument("--opendata_root", type=str, default="")
    parser.add_argument("--sample_rate", type=float, default=1 / 10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only_successful", action="store_true")
    parser.add_argument("--tol", type=int, default=2, help="Frame tolerance for matching")
    args = parser.parse_args()

    opendata_root = resolve_opendata_root(args.opendata_root)
    paths = get_match_paths(args.match_id, opendata_root)
    match_meta = load_match_json(args.match_id, opendata_root)

    print("[1/3] Generating dynamic pass events...")
    dyn = generate_event_data_from_dynamic(
        dynamic_csv=paths["dynamic_events_csv"],
        match_meta=match_meta,
        only_successful=args.only_successful,
    )
    dyn = _normalize_events(dyn)

    print("[2/3] Generating tracking-inferred pass events...")
    tracking_df, _ = build_tracking_wide_df(
        match_id=args.match_id,
        opendata_root=opendata_root,
        limit=args.limit,
        sample_rate=args.sample_rate,
        transform_to_metric=True,
    )
    trk = generate_event_data_from_tracking(tracking_df)
    trk = _normalize_events(trk)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    dyn_path = os.path.join(data_dir, f"skillcorner_match{args.match_id}_dynamic_passes.csv")
    trk_path = os.path.join(data_dir, f"skillcorner_match{args.match_id}_tracking_passes.csv")
    dyn.to_csv(dyn_path, index=False)
    trk.to_csv(trk_path, index=False)

    print("[3/3] Comparing events...")
    exact = dyn.merge(
        trk,
        on=["start_frame", "from_team", "from_player", "to_player"],
        how="inner",
        suffixes=("_dyn", "_trk"),
    )
    exact_count = len(exact)

    matched, used_tracking = match_with_tolerance(dyn, trk, args.tol)
    tol_count = len(matched)

    dyn_matched_idx = set([m[0] for m in matched])
    trk_matched_idx = used_tracking
    dyn_unmatched = dyn.drop(index=list(dyn_matched_idx))
    trk_unmatched = trk.drop(index=list(trk_matched_idx))

    dyn_unmatched_path = os.path.join(data_dir, f"skillcorner_match{args.match_id}_dynamic_unmatched.csv")
    trk_unmatched_path = os.path.join(data_dir, f"skillcorner_match{args.match_id}_tracking_unmatched.csv")
    dyn_unmatched.to_csv(dyn_unmatched_path, index=False)
    trk_unmatched.to_csv(trk_unmatched_path, index=False)

    print("========================================")
    print(f"Dynamic passes:  {len(dyn)}")
    print(f"Tracking passes: {len(trk)}")
    print(f"Exact matches (same frame+from+to): {exact_count}")
    print(f"Matched within ±{args.tol} frames (same from, pref same to): {tol_count}")
    print(f"Dynamic unmatched:  {len(dyn_unmatched)} -> {dyn_unmatched_path}")
    print(f"Tracking unmatched: {len(trk_unmatched)} -> {trk_unmatched_path}")
    print("Saved:")
    print(f"  {dyn_path}")
    print(f"  {trk_path}")


if __name__ == "__main__":
    main()
