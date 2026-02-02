#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


PITCH_L = 105.0
PITCH_W = 68.0


def to_metric_xy(xy):
    if np.nanmax(np.abs(xy)) <= 2.0:
        x = (xy[..., 0] * 0.5 + 0.5) * PITCH_L
        y = (xy[..., 1] * 0.5 + 0.5) * PITCH_W
        return np.stack([x, y], axis=-1)
    return xy


def iter_npz_files(path: Path):
    if path.is_dir():
        for p in sorted(path.glob("*.npz")):
            yield p
    else:
        yield path


def compute_rank_std(xy):
    x = xy[:, :, 0]
    ranks = np.argsort(np.argsort(x, axis=1), axis=1)
    return np.std(ranks, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Slot stability stats (standalone)")
    ap.add_argument("--data", required=True, help="NPZ file or directory")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of files")
    ap.add_argument("--max_frames", type=int, default=0, help="Limit frames per file")
    args = ap.parse_args()

    data_path = Path(args.data)
    files = list(iter_npz_files(data_path))
    if args.max_files and len(files) > args.max_files:
        files = files[: args.max_files]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for npz_path in files:
        data = np.load(npz_path)
        obs = data["obs"]
        if args.max_frames and obs.shape[0] > args.max_frames:
            obs = obs[: args.max_frames]
        team_xy = obs[:, :11, :2]
        team_xy = to_metric_xy(team_xy)

        mean_xy = np.nanmean(team_xy, axis=0)
        std_xy = np.nanstd(team_xy, axis=0)
        rank_std = compute_rank_std(team_xy)

        for i in range(11):
            rows.append(
                {
                    "file": npz_path.name,
                    "slot": i,
                    "mean_x": float(mean_xy[i, 0]),
                    "mean_y": float(mean_xy[i, 1]),
                    "std_x": float(std_xy[i, 0]),
                    "std_y": float(std_xy[i, 1]),
                    "rank_std": float(rank_std[i]),
                }
            )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "slot",
                "mean_x",
                "mean_y",
                "std_x",
                "std_y",
                "rank_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
