#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


PITCH_LEN = 105.0
PITCH_WID = 68.0


def _to_metric_xy(norm_xy):
    # norm: x in [-1,1] with center 0 => x = n*52.5 + 52.5
    # norm: y in [-1,1] with center 0 => y = n*34 + 34
    x = norm_xy[..., 0] * (PITCH_LEN / 2.0) + (PITCH_LEN / 2.0)
    y = norm_xy[..., 1] * (PITCH_WID / 2.0) + (PITCH_WID / 2.0)
    return np.stack([x, y], axis=-1)


def _cosine(a, b, eps=1e-8):
    # a,b: (...,2)
    dot = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = np.maximum(na * nb, eps)
    return dot / denom


def _hungarian_mse(act, tgt):
    # act,tgt: (11,2) -> per-frame assignment cost, return mean per-player 2D sq dist
    if not HAS_SCIPY:
        return None, None, None
    cost = np.sum((act[:, None, :] - tgt[None, :, :]) ** 2, axis=-1)
    r, c = linear_sum_assignment(cost)
    return cost[r, c].mean(), r, c


def eval_file(path, speed_thr=0.5, max_frames=None):
    data = np.load(path)
    obs = data["obs"]
    act = data["act_move"]
    dt = float(np.array(data["dt"]).reshape(-1)[0])

    # attacker slots are first 11 rows in obs
    pos_norm = obs[:, :11, :2]
    pos = _to_metric_xy(pos_norm)

    if max_frames is not None:
        pos = pos[:max_frames]
        act = act[:max_frames]

    # finite-diff velocity from positions
    v_fd = (pos[1:] - pos[:-1]) / dt
    act_t = act[:-1]

    diff = act_t - v_fd
    mse_coord = float(np.mean(diff ** 2))
    mse_player = float(np.mean(np.sum(diff ** 2, axis=-1)))

    # speed metrics
    speed_tgt = np.linalg.norm(v_fd, axis=-1)
    speed_act = np.linalg.norm(act_t, axis=-1)
    speed_err = float(np.mean(np.abs(speed_act - speed_tgt)))

    mask = speed_tgt > speed_thr
    if np.any(mask):
        cos = float(np.mean(_cosine(act_t[mask], v_fd[mask])))
    else:
        cos = float("nan")

    # Hungarian diagnostics
    hungarian_mse = None
    id_rate = None
    top_swaps = []
    if HAS_SCIPY:
        id_hits = 0
        total = 0
        swap_counts = {}
        costs = []
        for t in range(act_t.shape[0]):
            cost_t, r, c = _hungarian_mse(act_t[t], v_fd[t])
            if cost_t is None:
                continue
            costs.append(cost_t)
            for rr, cc in zip(r, c):
                total += 1
                if rr == cc:
                    id_hits += 1
                else:
                    key = f"{rr}->{cc}"
                    swap_counts[key] = swap_counts.get(key, 0) + 1
        if total > 0:
            id_rate = id_hits / total
        if costs:
            hungarian_mse = float(np.mean(costs))
        if swap_counts:
            top_swaps = sorted(swap_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "frames": int(pos.shape[0]),
        "dt": dt,
        "mse_coord": mse_coord,
        "mse_player": mse_player,
        "cosine_speed_gt_thr": cos,
        "speed_abs_err": speed_err,
        "hungarian_mse_player": hungarian_mse,
        "hungarian_id_rate": id_rate,
        "top_swaps": top_swaps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz dir or file")
    ap.add_argument("--speed_thr", type=float, default=0.5)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    data_path = Path(args.data)
    if data_path.is_file():
        files = [data_path]
    else:
        files = sorted(p for p in data_path.glob("*.npz"))

    results = {}
    agg = {
        "mse_coord": [],
        "mse_player": [],
        "cos": [],
        "speed_err": [],
        "hung_mse": [],
        "id_rate": [],
    }

    for p in files:
        res = eval_file(p, speed_thr=args.speed_thr, max_frames=args.max_frames)
        results[p.name] = res
        print(
            f"{p.name}: frames={res['frames']} "
            f"mse_coord={res['mse_coord']:.6f} mse_player={res['mse_player']:.6f} "
            f"cos={res['cosine_speed_gt_thr']:.4f} |v|err={res['speed_abs_err']:.4f} "
            f"hung_mse={res['hungarian_mse_player']} id_rate={res['hungarian_id_rate']}"
        )
        agg["mse_coord"].append(res["mse_coord"])
        agg["mse_player"].append(res["mse_player"])
        if not np.isnan(res["cosine_speed_gt_thr"]):
            agg["cos"].append(res["cosine_speed_gt_thr"])
        agg["speed_err"].append(res["speed_abs_err"])
        if res["hungarian_mse_player"] is not None:
            agg["hung_mse"].append(res["hungarian_mse_player"])
        if res["hungarian_id_rate"] is not None:
            agg["id_rate"].append(res["hungarian_id_rate"])

    if agg["mse_coord"]:
        print("\nAGG:")
        print(
            f"mse_coord={np.mean(agg['mse_coord']):.6f} "
            f"mse_player={np.mean(agg['mse_player']):.6f} "
            f"cos={np.mean(agg['cos']) if agg['cos'] else float('nan'):.4f} "
            f"|v|err={np.mean(agg['speed_err']):.4f} "
            f"hung_mse={np.mean(agg['hung_mse']) if agg['hung_mse'] else float('nan'):.6f} "
            f"id_rate={np.mean(agg['id_rate']) if agg['id_rate'] else float('nan'):.4f}"
        )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} entries -> {args.out_json}")


if __name__ == "__main__":
    main()
