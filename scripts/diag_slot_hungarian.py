#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _list_npz(path: Path):
    if path.is_file() and path.suffix == ".npz":
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.npz"))
    raise FileNotFoundError(f"Path not found: {path}")


def _to_metric_coords(xy, pitch_length=105.0, pitch_width=68.0):
    # If data already looks metric (values beyond [-2,2]), return as-is.
    if np.nanmax(np.abs(xy)) > 2.0:
        return xy
    # Assume normalized to [-1,1] with center 0.
    out = xy.copy()
    out[..., 0] = (out[..., 0] * (pitch_length / 2.0)) + (pitch_length / 2.0)
    out[..., 1] = (out[..., 1] * (pitch_width / 2.0)) + (pitch_width / 2.0)
    return out


def _hungarian_match(pred_xy, tgt_xy):
    # pred_xy, tgt_xy: (11,2)
    d2 = np.sum((pred_xy[:, None, :] - tgt_xy[None, :, :]) ** 2, axis=-1)
    if _HAS_SCIPY:
        r, c = linear_sum_assignment(d2)
        return d2, r, c
    # Fallback greedy matching (no identity stats)
    d2 = d2.copy()
    used_t = set()
    rows = []
    cols = []
    for i in range(d2.shape[0]):
        j = np.argmin(d2[i])
        while j in used_t:
            d2[i, j] = np.inf
            j = np.argmin(d2[i])
        used_t.add(j)
        rows.append(i)
        cols.append(j)
    return d2, np.array(rows), np.array(cols)


def evaluate_file(npz_path: Path, max_frames: int):
    data = np.load(npz_path, allow_pickle=False)
    obs = data["obs"]  # (T, 23, 3)
    act_move = data["act_move"]  # (T, 11, 2)
    dt = data.get("dt", None)
    if dt is None:
        dt = 0.1
    dt = float(np.array(dt).reshape(-1)[0])

    T = obs.shape[0]
    if max_frames > 0:
        T = min(T, max_frames)

    # attackers in obs: 0..10, positions in first two dims
    pos = obs[:T, :11, :2]
    pos = _to_metric_coords(pos)
    vel = act_move[:T]

    # Use t to predict t+1; so use frames 0..T-2
    n = max(T - 1, 0)
    if n == 0:
        return None

    fixed_mse_coord = []
    fixed_mse_player = []
    hung_mse_player = []
    id_rates = []
    swap_counts = {}
    for t in range(n):
        pred = pos[t] + vel[t] * dt
        tgt = pos[t + 1]
        err = pred - tgt
        fixed_mse_coord.append(float(np.mean(err ** 2)))
        fixed_mse_player.append(float(np.mean(np.sum(err ** 2, axis=-1))))

        d2, r, c = _hungarian_match(pred, tgt)
        hung_mse_player.append(float(d2[r, c].mean()))
        if _HAS_SCIPY:
            id_rates.append(float(np.mean(r == c)))
            for ri, ci in zip(r.tolist(), c.tolist()):
                if ri != ci:
                    swap_counts[(ri, ci)] = swap_counts.get((ri, ci), 0) + 1

    fixed_mse_coord = np.array(fixed_mse_coord)
    fixed_mse_player = np.array(fixed_mse_player)
    hung_mse_player = np.array(hung_mse_player)
    ratio = hung_mse_player.mean() / max(fixed_mse_player.mean(), 1e-9)
    id_rate = float(np.mean(id_rates)) if id_rates else float("nan")

    top_swaps = []
    if swap_counts:
        top_swaps = sorted(swap_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "file": npz_path.name,
        "frames": n,
        "fixed_mse_coord": float(fixed_mse_coord.mean()),
        "fixed_mse_player": float(fixed_mse_player.mean()),
        "hungarian_mse_player": float(hung_mse_player.mean()),
        "ratio": float(ratio),
        "id_rate": id_rate,
        "top_swaps": top_swaps,
    }


def main():
    ap = argparse.ArgumentParser(description="Diagnose slot/permutation confusion via Hungarian matching.")
    ap.add_argument("--data", required=True, help="NPZ file or directory.")
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all).")
    ap.add_argument("--max_frames", type=int, default=50000, help="Limit frames per file (0 = all).")
    args = ap.parse_args()

    files = _list_npz(Path(args.data))
    if args.max_files > 0:
        files = files[: args.max_files]

    results = []
    for f in files:
        out = evaluate_file(f, args.max_frames)
        if out is None:
            continue
        results.append(out)
        print(
            f"{out['file']}: frames={out['frames']} "
            f"fixed_coord_mse={out['fixed_mse_coord']:.6f} "
            f"fixed_player_mse={out['fixed_mse_player']:.6f} "
            f"hungarian_player_mse={out['hungarian_mse_player']:.6f} "
            f"ratio={out['ratio']:.3f} "
            f"id_rate={out['id_rate']:.3f}"
        )
        if out["top_swaps"]:
            top_str = ", ".join([f"{a}->{b}:{cnt}" for (a, b), cnt in out["top_swaps"]])
            print(f"  top_swaps: {top_str}")

    if results:
        fixed_coord = np.mean([r["fixed_mse_coord"] for r in results])
        fixed_player = np.mean([r["fixed_mse_player"] for r in results])
        hung = np.mean([r["hungarian_mse_player"] for r in results])
        ratio = hung / max(fixed_player, 1e-9)
        id_rate = np.mean([r["id_rate"] for r in results if not np.isnan(r["id_rate"])]) if _HAS_SCIPY else float("nan")
        print(
            f"\nAGG: fixed_coord_mse={fixed_coord:.6f} "
            f"fixed_player_mse={fixed_player:.6f} "
            f"hungarian_player_mse={hung:.6f} "
            f"ratio={ratio:.3f} id_rate={id_rate:.3f}"
        )
        print(">> Note: ratio compares per-player MSE (sum over x,y).")
    else:
        print("No valid files.")


if __name__ == "__main__":
    main()
