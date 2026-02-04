#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np


def list_npz(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.npz"))
        files.sort()
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise FileNotFoundError(f"Invalid path: {path}")


def infer_owner(obs_frame):
    # obs_frame: (7,3), has_ball in dim 2 for players 0..5
    has_ball = obs_frame[:6, 2]
    if has_ball.max() > 0.5:
        return int(np.argmax(has_ball))  # 0..5
    return -1


def analyze_episode(data, window, min_free):
    obs = data["obs"]
    T = obs.shape[0]
    pass_flag = data.get("pass_flag", np.zeros(T, dtype=np.float32))
    passer = data.get("passer_id", np.zeros(T, dtype=np.int64))
    receiver = data.get("receiver_id", np.zeros(T, dtype=np.int64))

    owners = np.array([infer_owner(obs[t]) for t in range(T)], dtype=np.int64)

    attempts = 0
    complete_any = 0
    complete_first = 0
    free_frames_sum = 0
    free_frames_count = 0

    for t in range(T):
        if pass_flag[t] <= 0.5:
            continue
        attempts += 1
        recv = int(receiver[t])
        if recv < 0 or recv > 2:
            continue

        # count consecutive free frames right after pass
        free_count = 0
        for k in range(t + 1, min(T, t + 1 + window)):
            if owners[k] == -1:
                free_count += 1
            else:
                break
        if free_count >= min_free:
            free_frames_sum += free_count
            free_frames_count += 1

        # check first possession after pass
        first_owner = None
        for k in range(t + 1, min(T, t + 1 + window)):
            if owners[k] != -1:
                first_owner = owners[k]
                break
        if first_owner is not None and first_owner == recv:
            complete_first += 1

        # check any time receiver gets ball within window
        got = False
        for k in range(t + 1, min(T, t + 1 + window)):
            if owners[k] == recv:
                got = True
                break
        if got:
            complete_any += 1

    avg_free = (free_frames_sum / free_frames_count) if free_frames_count > 0 else 0.0
    return {
        "attempts": attempts,
        "complete_any": complete_any,
        "complete_first": complete_first,
        "avg_free_frames": avg_free,
        "T": T,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze effective passes in 3v3 point-sim npz")
    parser.add_argument("--data_dir", default=os.path.join("data", "point_sim_3v3", "npz"))
    parser.add_argument("--window", type=int, default=20, help="Frames to check after pass")
    parser.add_argument("--min_free", type=int, default=1, help="Min free-ball frames after pass to count")
    parser.add_argument("--print_top", type=int, default=10)
    args = parser.parse_args()

    files = list_npz(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No npz files in {args.data_dir}")

    totals = {
        "episodes": 0,
        "frames": 0,
        "attempts": 0,
        "complete_any": 0,
        "complete_first": 0,
        "avg_free_frames": 0.0,
    }

    per_ep = []

    for fp in files:
        data = np.load(fp)
        stats = analyze_episode(data, args.window, args.min_free)
        totals["episodes"] += 1
        totals["frames"] += stats["T"]
        totals["attempts"] += stats["attempts"]
        totals["complete_any"] += stats["complete_any"]
        totals["complete_first"] += stats["complete_first"]
        per_ep.append((fp, stats))

    avg_free = 0.0
    free_counts = [s["avg_free_frames"] for _, s in per_ep if s["avg_free_frames"] > 0]
    if free_counts:
        avg_free = float(np.mean(free_counts))

    print(f"Episodes: {totals['episodes']}")
    print(f"Total frames: {totals['frames']}")
    print(f"Pass attempts: {totals['attempts']}")
    if totals["attempts"] > 0:
        print(f"Pass success (first possession=receiver): {totals['complete_first']} ({totals['complete_first']/totals['attempts']:.2%})")
        print(f"Pass success (receiver gets within window): {totals['complete_any']} ({totals['complete_any']/totals['attempts']:.2%})")
    print(f"Avg free-ball frames after pass (>=min_free): {avg_free:.2f}")

    per_ep.sort(key=lambda x: x[1]["attempts"], reverse=True)
    if args.print_top > 0:
        print("Top episodes by pass attempts:")
        for fp, s in per_ep[: args.print_top]:
            print(
                f"  {os.path.basename(fp)}: attempts={s['attempts']} first_ok={s['complete_first']} any_ok={s['complete_any']} frames={s['T']}"
            )


if __name__ == "__main__":
    main()
