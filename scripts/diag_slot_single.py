#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _iter_npz_paths(data_path: Path):
    if data_path.is_dir():
        for p in sorted(data_path.glob("*.npz")):
            yield p
    else:
        yield data_path


def _obs_to_metric_xy(obs_xy: np.ndarray) -> np.ndarray:
    # obs_xy: (T, 11, 2) in normalized coords [-1,1] or already metric.
    # If looks normalized, convert to metric 105x68.
    if np.nanmax(np.abs(obs_xy)) <= 2.0:
        out = obs_xy.copy()
        out[..., 0] = out[..., 0] * 52.5 + 52.5
        out[..., 1] = out[..., 1] * 34.0 + 34.0
        return out
    return obs_xy


def analyze_file(path: Path, max_frames: int | None = None):
    data = np.load(path, allow_pickle=True)
    if "obs" not in data:
        raise ValueError(f"{path} missing obs")
    obs = data["obs"]
    if obs.ndim != 3 or obs.shape[1] < 11:
        raise ValueError(f"{path} obs shape unexpected: {obs.shape}")
    obs_xy = obs[:, :11, :2].astype(np.float32)
    if max_frames is not None and obs_xy.shape[0] > max_frames:
        obs_xy = obs_xy[:max_frames]
    obs_xy = _obs_to_metric_xy(obs_xy)

    T = obs_xy.shape[0]
    if T < 2:
        return None

    # For each slot i at time t, check if its closest position at t+1 is itself.
    swap_counts = np.zeros(11, dtype=np.int64)
    total_counts = np.zeros(11, dtype=np.int64)
    self_dist_sum = np.zeros(11, dtype=np.float64)
    nn_dist_sum = np.zeros(11, dtype=np.float64)

    for t in range(T - 1):
        p0 = obs_xy[t]   # (11,2)
        p1 = obs_xy[t + 1]
        # Dist matrix (11x11)
        dmat = np.linalg.norm(p0[:, None, :] - p1[None, :, :], axis=-1)
        nn_idx = np.argmin(dmat, axis=1)
        nn_dist = dmat[np.arange(11), nn_idx]
        self_dist = dmat[np.arange(11), np.arange(11)]

        swap_counts += (nn_idx != np.arange(11)).astype(np.int64)
        total_counts += 1
        self_dist_sum += self_dist
        nn_dist_sum += nn_dist

    swap_rate = swap_counts / np.maximum(total_counts, 1)
    self_dist_mean = self_dist_sum / np.maximum(total_counts, 1)
    nn_dist_mean = nn_dist_sum / np.maximum(total_counts, 1)

    out = {
        "file": str(path),
        "frames": int(T),
        "swap_rate_mean": float(np.mean(swap_rate)),
        "swap_rate_max": float(np.max(swap_rate)),
        "self_dist_mean": float(np.mean(self_dist_mean)),
        "nn_dist_mean": float(np.mean(nn_dist_mean)),
        "per_slot": [
            {
                "slot": int(i),
                "swap_rate": float(swap_rate[i]),
                "self_dist_mean": float(self_dist_mean[i]),
                "nn_dist_mean": float(nn_dist_mean[i]),
            }
            for i in range(11)
        ],
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose slot stability: nearest-neighbor swaps across frames."
    )
    parser.add_argument("--data", required=True, help="npz file or directory")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--out_json", default="output/slot_swap_diag.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    results = []
    for i, p in enumerate(_iter_npz_paths(data_path)):
        if args.max_files is not None and i >= args.max_files:
            break
        res = analyze_file(p, max_frames=args.max_frames)
        if res is not None:
            results.append(res)
            print(
                f"{p.name}: frames={res['frames']} swap_mean={res['swap_rate_mean']:.4f} "
                f"swap_max={res['swap_rate_max']:.4f} self_d={res['self_dist_mean']:.3f} "
                f"nn_d={res['nn_dist_mean']:.3f}"
            )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"Wrote {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
