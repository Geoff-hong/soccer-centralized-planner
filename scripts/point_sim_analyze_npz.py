#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np

TEMPLATE_NAMES = {
    0: "TRIANGLE",
    1: "ONE_TWO",
    2: "THROUGH",
    3: "CUTBACK",
}


def list_npz(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.npz"))
        files.sort()
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise FileNotFoundError(f"Invalid path: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze 3v3 point-sim npz data")
    parser.add_argument("--data_dir", default=os.path.join("data", "point_sim_3v3", "npz"))
    parser.add_argument("--print_top", type=int, default=5, help="Print top-N episodes by pass count")
    args = parser.parse_args()

    files = list_npz(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No npz files in {args.data_dir}")

    total_frames = 0
    total_passes = 0
    pass_counts = []
    pass_type_hist = np.zeros(3, dtype=np.int64)

    has_tactics = True
    tactic_counts = np.zeros(4, dtype=np.int64)
    tactic_switches = []

    for fp in files:
        data = np.load(fp)
        obs = data["obs"]
        T = obs.shape[0]
        total_frames += T

        pass_flag = data.get("pass_flag", np.zeros(T, dtype=np.float32))
        pcount = int(np.sum(pass_flag > 0.5))
        total_passes += pcount
        pass_counts.append((fp, pcount, T))

        if "pass_type" in data:
            pt = data["pass_type"]
            for i in range(3):
                pass_type_hist[i] += int(np.sum((pass_flag > 0.5) & (pt == i)))

        if "template_id" in data:
            tid = data["template_id"]
            for i in range(4):
                tactic_counts[i] += int(np.sum(tid == i))
            if "template_switch" in data:
                tactic_switches.append(int(np.sum(data["template_switch"])))
            else:
                diffs = np.sum(tid[1:] != tid[:-1]) if len(tid) > 1 else 0
                tactic_switches.append(int(diffs))
        else:
            has_tactics = False

    n_episodes = len(files)
    avg_len = total_frames / max(n_episodes, 1)
    avg_pass = total_passes / max(n_episodes, 1)
    pass_rate = total_passes / max(total_frames, 1)

    print(f"Episodes: {n_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Avg frames per episode: {avg_len:.2f}")
    print(f"Total passes: {total_passes}")
    print(f"Avg passes per episode: {avg_pass:.3f}")
    print(f"Pass rate per frame: {pass_rate:.5f}")

    if pass_type_hist.sum() > 0:
        print(
            "Pass type counts: "
            f"short={pass_type_hist[0]} through={pass_type_hist[1]} cutback={pass_type_hist[2]}"
        )

    if has_tactics:
        total_tactic = tactic_counts.sum()
        print("Tactic usage:")
        for i in range(4):
            ratio = tactic_counts[i] / max(total_tactic, 1)
            print(f"  {TEMPLATE_NAMES[i]}: {tactic_counts[i]} ({ratio:.2%})")
        if tactic_switches:
            avg_switch = np.mean(tactic_switches)
            print(f"Avg tactic switches per episode: {avg_switch:.2f}")
    else:
        print("No template_id in data. Re-generate with --save_tactics to analyze tactic usage.")

    pass_counts.sort(key=lambda x: x[1], reverse=True)
    if args.print_top > 0:
        print("Top episodes by pass count:")
        for fp, pcount, T in pass_counts[: args.print_top]:
            print(f"  {os.path.basename(fp)}: passes={pcount}, frames={T}")


if __name__ == "__main__":
    main()
