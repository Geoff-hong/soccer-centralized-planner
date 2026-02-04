#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)


def _normalize_dt(dt_val, n_frames):
    if dt_val is None:
        return np.full(n_frames, 0.1, dtype=np.float32)
    dt_arr = np.asarray(dt_val, dtype=np.float32)
    if dt_arr.ndim == 0:
        return np.full(n_frames, float(dt_arr), dtype=np.float32)
    if dt_arr.size == 0:
        return np.full(n_frames, 0.1, dtype=np.float32)
    if dt_arr.shape[0] == n_frames:
        return dt_arr
    if dt_arr.size == 1:
        return np.full(n_frames, float(dt_arr.reshape(-1)[0]), dtype=np.float32)
    if dt_arr.shape[0] > n_frames:
        return dt_arr[:n_frames]
    pad = np.full(n_frames - dt_arr.shape[0], dt_arr[-1], dtype=np.float32)
    return np.concatenate([dt_arr, pad], axis=0)


def list_npz(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.npz"))
        files.sort()
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise FileNotFoundError(f"Invalid path: {path}")


def main():
    parser = argparse.ArgumentParser(description="Verify 3v3 point sim consistency")
    parser.add_argument("--data_dir", default=os.path.join("data", "point_sim_3v3", "npz"))
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = list_npz(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No npz files in {args.data_dir}")

    rng = np.random.default_rng(args.seed)
    sample_files = rng.choice(files, size=min(args.num, len(files)), replace=False)

    max_err = 0.0
    bad_pass = 0
    outcomes = {"goal": 0, "out": 0, "timeout": 0}

    for fp in sample_files:
        data = np.load(fp)
        obs = data["obs"]
        act_move = data["act_move"]
        T = obs.shape[0]
        dt = _normalize_dt(data.get("dt", 0.1), T)

        pos = obs[:, :3, :2] * np.array([52.5, 34.0], dtype=np.float32)
        if T > 1:
            delta = pos[1:] - pos[:-1]
            diff = delta - act_move[:-1] * dt[:-1, None, None]
            err = float(np.max(np.abs(diff)))
            max_err = max(max_err, err)

        if "passer_id" in data:
            pid = data["passer_id"]
            rid = data.get("receiver_id", np.zeros_like(pid))
            if (pid < 0).any() or (pid > 2).any() or (rid < 0).any() or (rid > 2).any():
                bad_pass += 1

        outcome = None
        if "outcome" in data:
            outcome = str(data["outcome"])  # numpy may store as array
            if "goal" in outcome:
                outcomes["goal"] += 1
            elif "out" in outcome:
                outcomes["out"] += 1
            else:
                outcomes["timeout"] += 1
        else:
            ball = obs[-1, 6, :2] * np.array([52.5, 34.0], dtype=np.float32)
            if ball[0] > 52.5 and abs(ball[1]) < 3.66:
                outcomes["goal"] += 1
            elif abs(ball[0]) > 52.5 or abs(ball[1]) > 34.0:
                outcomes["out"] += 1
            else:
                outcomes["timeout"] += 1

    print(f"Checked {len(sample_files)} episodes")
    print(f"Max position consistency error: {max_err:.6f}")
    print(f"Pass label out-of-range files: {bad_pass}")
    print(f"Outcomes: goal={outcomes['goal']} out={outcomes['out']} timeout={outcomes['timeout']}")


if __name__ == "__main__":
    main()
