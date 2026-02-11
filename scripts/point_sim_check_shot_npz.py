#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np


PASS_SHOT = 3


def _list_npz_files(data_dir: str) -> List[str]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    files.sort()
    return [os.path.join(data_dir, f) for f in files]


def _as_scalar_dt(dt_val, T):
    if isinstance(dt_val, np.ndarray):
        if dt_val.ndim == 0:
            return float(dt_val), np.full(T, float(dt_val), dtype=np.float32)
        if dt_val.shape[0] == T:
            return float(dt_val[0]), dt_val.astype(np.float32)
        if dt_val.size == 1:
            return float(dt_val.reshape(-1)[0]), np.full(T, float(dt_val.reshape(-1)[0]), dtype=np.float32)
    return float(dt_val), np.full(T, float(dt_val), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Check shot labeling in point-sim npz dataset")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num", type=int, default=0, help="Number of episodes to check (0=all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--require_goal", action="store_true", help="Fail if shot exists but outcome != goal")
    parser.add_argument("--print_bad", type=int, default=10, help="Print up to N bad files")
    args = parser.parse_args()

    files = _list_npz_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data_dir}")
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(files)
    if args.num and args.num > 0:
        files = files[: args.num]

    total_eps = 0
    total_frames = 0
    pass_frames = 0
    shot_frames = 0
    shot_eps = 0
    bad_files = []
    bad_reasons = {}
    outcome_counts = {}

    for fp in files:
        total_eps += 1
        data = np.load(fp)
        try:
            if "obs" not in data or "act_move" not in data:
                bad_reasons["missing_obs_or_move"] = bad_reasons.get("missing_obs_or_move", 0) + 1
                bad_files.append((fp, "missing_obs_or_move"))
                continue
            obs = data["obs"]
            move = data["act_move"]
            pass_flag = data.get("pass_flag", None)
            passer_id = data.get("passer_id", None)
            receiver_id = data.get("receiver_id", None)
            pass_type = data.get("pass_type", None)
            dt_val = data.get("dt", 0.1)
            outcome = data.get("outcome", None)

            if pass_flag is None or passer_id is None or receiver_id is None:
                bad_reasons["missing_pass_labels"] = bad_reasons.get("missing_pass_labels", 0) + 1
                bad_files.append((fp, "missing_pass_labels"))
                continue

            T = obs.shape[0]
            total_frames += T
            dt_scalar, dt_arr = _as_scalar_dt(dt_val, T)

            # Basic shape checks
            if obs.ndim != 3 or obs.shape[2] < 3:
                bad_reasons["bad_obs_shape"] = bad_reasons.get("bad_obs_shape", 0) + 1
                bad_files.append((fp, f"bad_obs_shape={obs.shape}"))
                continue
            if move.ndim != 3 or move.shape[2] != 2:
                bad_reasons["bad_move_shape"] = bad_reasons.get("bad_move_shape", 0) + 1
                bad_files.append((fp, f"bad_move_shape={move.shape}"))
                continue

            n_att = move.shape[1]
            goal_id = n_att

            pass_flag = pass_flag.astype(np.float32)
            receiver_id = receiver_id.astype(np.int64)
            pass_type = pass_type.astype(np.int64) if pass_type is not None else None

            pass_frames += float(np.sum(pass_flag > 0.5))
            shot_mask = receiver_id == goal_id
            if np.any(shot_mask):
                shot_eps += 1
                shot_frames += int(np.sum(shot_mask))

            # Consistency checks
            if pass_type is not None:
                bad_shot_type = np.logical_and(shot_mask, pass_type != PASS_SHOT)
                if np.any(bad_shot_type):
                    bad_reasons["shot_type_mismatch"] = bad_reasons.get("shot_type_mismatch", 0) + 1
                    bad_files.append((fp, "shot_type_mismatch"))

            bad_shot_flag = np.logical_and(shot_mask, pass_flag <= 0.5)
            if np.any(bad_shot_flag):
                bad_reasons["shot_flag_zero"] = bad_reasons.get("shot_flag_zero", 0) + 1
                bad_files.append((fp, "shot_flag_zero"))

            if outcome is not None:
                outcome_str = outcome if isinstance(outcome, str) else str(outcome)
                outcome_counts[outcome_str] = outcome_counts.get(outcome_str, 0) + 1
                if args.require_goal and np.any(shot_mask) and outcome_str != "goal":
                    bad_reasons["shot_not_goal"] = bad_reasons.get("shot_not_goal", 0) + 1
                    bad_files.append((fp, f"shot_not_goal({outcome_str})"))
        finally:
            data.close()

    print("=== Shot NPZ Check Summary ===")
    print(f"Episodes checked: {total_eps}")
    print(f"Total frames: {total_frames}")
    if total_frames > 0:
        print(f"Pass-positive frame ratio: {pass_frames / total_frames:.4f}")
        print(f"Shot-positive frame ratio: {shot_frames / total_frames:.4f}")
    print(f"Episodes with shots: {shot_eps}")
    if outcome_counts:
        print("Outcome counts:")
        for k, v in sorted(outcome_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    if bad_reasons:
        print("Bad reasons:")
        for k, v in sorted(bad_reasons.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        if args.print_bad > 0:
            print("Bad files (sample):")
            for fp, reason in bad_files[: args.print_bad]:
                print(f"  {os.path.basename(fp)} -> {reason}")
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
