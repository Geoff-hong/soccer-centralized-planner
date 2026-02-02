#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "train"))
from model import SoccerPolicy  # noqa: E402


def list_npz(path: str) -> List[str]:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.npz"))
        files.sort()
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise FileNotFoundError(f"Invalid data path: {path}")


def load_npz(fp: str) -> Tuple[np.ndarray, np.ndarray, float]:
    data = np.load(fp)
    obs = data["obs"]
    if "act_move" in data:
        move = data["act_move"]
    elif "acts" in data:
        move = data["acts"][:, :, 0:2]
    else:
        raise KeyError(f"{fp} missing act_move/acts")
    dt = float(np.array(data["dt"]).reshape(-1)[0]) if "dt" in data else 0.1
    return obs, move, dt


def stack_obs(obs: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return obs.astype(np.float32)
    t, n, f = obs.shape
    stacked = np.zeros((t, n, f * k), dtype=np.float32)
    for i in range(t):
        start = max(0, i - (k - 1))
        frames = obs[start : i + 1]
        if frames.shape[0] < k:
            pad = np.repeat(frames[0:1], k - frames.shape[0], axis=0)
            frames = np.concatenate([pad, frames], axis=0)
        stacked[i] = frames.transpose(1, 0, 2).reshape(n, f * k)
    return stacked


def _infer_move_horizon(state_dict: Dict[str, torch.Tensor], default_horizon: int = 1) -> int:
    weight = state_dict.get("head_move.2.weight")
    bias = state_dict.get("head_move.2.bias")
    if weight is None and bias is None:
        return default_horizon
    out_dim = weight.shape[0] if weight is not None else bias.shape[0]
    if out_dim % 2 != 0:
        return default_horizon
    horizon = out_dim // 2
    return max(int(horizon), 1)


def compute_metrics(pred: np.ndarray, tgt: np.ndarray, speed_thr: float, dir_speed_thr: float) -> Dict[str, float]:
    diff = pred - tgt
    mse = float(np.mean(diff ** 2))

    sp_t = np.linalg.norm(tgt, axis=2)
    sp_p = np.linalg.norm(pred, axis=2)
    mag_err = float(np.mean(np.abs(sp_p - sp_t)))

    dot = np.sum(pred * tgt, axis=2)
    denom = (sp_p * sp_t) + 1e-8
    cos = dot / denom
    valid = sp_t > dir_speed_thr
    cos_mean = float(np.mean(cos[valid])) if np.any(valid) else 0.0

    segs = {}
    masks = {
        "low(<2)": sp_t < 2.0,
        "mid(2-4)": (sp_t >= 2.0) & (sp_t <= speed_thr),
        "high(>4)": sp_t > speed_thr,
    }
    for name, m in masks.items():
        if np.any(m):
            segs[f"{name}_mse"] = float(np.mean(diff[m] ** 2))
            segs[f"{name}_mag_err"] = float(np.mean(np.abs(sp_p[m] - sp_t[m])))
            segs[f"{name}_cos"] = float(np.mean(cos[m]))
        else:
            segs[f"{name}_mse"] = float("nan")
            segs[f"{name}_mag_err"] = float("nan")
            segs[f"{name}_cos"] = float("nan")

    out = {"mse": mse, "cosine": cos_mean, "mag_err": mag_err}
    out.update(segs)
    return out


def baseline_mse(tgt: np.ndarray) -> Dict[str, float]:
    zero_mse = float(np.mean(tgt ** 2))
    mean_vec = tgt.mean(axis=(0, 1), keepdims=True)
    mean_mse = float(np.mean((tgt - mean_vec) ** 2))
    if tgt.shape[0] > 1:
        last_mse = float(np.mean((tgt[1:] - tgt[:-1]) ** 2))
    else:
        last_mse = float("nan")
    return {
        "baseline_zero_mse": zero_mse,
        "baseline_mean_mse": mean_mse,
        "baseline_last_mse": last_mse,
    }


def eval_train_split(
    files: List[str],
    checkpoint: str,
    obs_history: int,
    batch_size: int,
    device: str,
    speed_thr: float,
    dir_speed_thr: float,
    max_frames: int,
    d_model: int,
    nhead: int,
    num_layers: int,
) -> None:
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model", state)
    move_horizon = _infer_move_horizon(state_dict, default_horizon=1)
    model = SoccerPolicy(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1,
        input_dim=3 * obs_history,
        move_horizon=move_horizon,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict mismatch. Missing={missing} Unexpected={unexpected}")
    model.eval()

    agg = {
        "count": 0,
        "mse": 0.0,
        "cosine": 0.0,
        "mag_err": 0.0,
        "baseline_zero_mse": 0.0,
        "baseline_mean_mse": 0.0,
        "baseline_last_mse": 0.0,
    }

    for fp in files:
        obs, move, _ = load_npz(fp)
        if max_frames > 0:
            obs = obs[:max_frames]
            move = move[:max_frames]
        stacked = stack_obs(obs, obs_history)
        t = stacked.shape[0]

        preds = []
        with torch.no_grad():
            for i in range(0, t, batch_size):
                x = torch.from_numpy(stacked[i : i + batch_size]).to(device)
                pred_vel, _, _, _ = model(x)
                if pred_vel.ndim == 4:
                    pred_vel = pred_vel[:, 0]
                preds.append(pred_vel.cpu().numpy())
        pred = np.concatenate(preds, axis=0)

        metrics = compute_metrics(pred, move, speed_thr, dir_speed_thr)
        base = baseline_mse(move)

        print(f"\n=== {os.path.basename(fp)} (train split) ===")
        print(f"Frames: {t}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Cosine: {metrics['cosine']:.4f}")
        print(f"|v| err: {metrics['mag_err']:.4f}")
        print(
            "Baseline MSE (zero/mean/last): "
            f"{base['baseline_zero_mse']:.6f} / {base['baseline_mean_mse']:.6f} / {base['baseline_last_mse']:.6f}"
        )
        print("Speed segments (thr=4.0 m/s):")
        print(
            f"  low(<2)  mse={metrics['low(<2)_mse']:.6f} | cos={metrics['low(<2)_cos']:.4f} | |v|err={metrics['low(<2)_mag_err']:.4f}"
        )
        print(
            f"  mid(2-4) mse={metrics['mid(2-4)_mse']:.6f} | cos={metrics['mid(2-4)_cos']:.4f} | |v|err={metrics['mid(2-4)_mag_err']:.4f}"
        )
        print(
            f"  high(>4) mse={metrics['high(>4)_mse']:.6f} | cos={metrics['high(>4)_cos']:.4f} | |v|err={metrics['high(>4)_mag_err']:.4f}"
        )

        agg["count"] += 1
        for k in ["mse", "cosine", "mag_err"]:
            agg[k] += metrics[k]
        for k in ["baseline_zero_mse", "baseline_mean_mse", "baseline_last_mse"]:
            agg[k] += base[k]

    if agg["count"] > 0:
        c = agg["count"]
        print("\n=== AGGREGATE (train split) ===")
        print(f"MSE: {agg['mse'] / c:.6f}")
        print(f"Cosine: {agg['cosine'] / c:.4f}")
        print(f"|v| err: {agg['mag_err'] / c:.4f}")
        print(
            "Baseline MSE (zero/mean/last): "
            f"{agg['baseline_zero_mse'] / c:.6f} / {agg['baseline_mean_mse'] / c:.6f} / {agg['baseline_last_mse'] / c:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="npz file or directory")
    parser.add_argument("--checkpoint", required=True, help="model checkpoint (.pth)")
    parser.add_argument("--obs_history", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_by_file", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speed_thr", type=float, default=4.0)
    parser.add_argument("--dir_speed_thr", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()

    files = list_npz(args.data)
    if args.split_by_file and len(files) > 1:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        files = [files[i] for i in idx]
        val_count = max(1, int(len(files) * args.val_ratio))
        train_files = files[val_count:]
    else:
        train_files = files

    if not train_files:
        print("[Warn] No train files after split; falling back to all files.")
        train_files = files

    eval_train_split(
        train_files,
        args.checkpoint,
        args.obs_history,
        args.batch_size,
        args.device,
        args.speed_thr,
        args.dir_speed_thr,
        args.max_frames,
        args.d_model,
        args.nhead,
        args.num_layers,
    )


if __name__ == "__main__":
    main()
