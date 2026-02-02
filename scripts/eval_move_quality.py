#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Dict, Tuple, List

import numpy as np
import torch

# Local imports
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "train"))
from model import SoccerPolicy  # noqa: E402


def _infer_move_horizon(state: Dict[str, torch.Tensor]) -> int:
    for key in ("head_move.2.weight", "head_move.2.bias"):
        if key in state:
            out_dim = int(state[key].shape[0])
            if out_dim % 2 == 0 and out_dim > 0:
                return max(1, out_dim // 2)
            return 1
    return 1


def _list_npz_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.npz"))
        files.sort()
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise FileNotFoundError(f"Invalid data path: {path}")


def _load_npz(fp: str) -> Tuple[np.ndarray, np.ndarray, float]:
    data = np.load(fp)
    obs = data["obs"]
    if "act_move" in data:
        move = data["act_move"]
    elif "acts" in data:
        move = data["acts"][:, :, 0:2]
    else:
        raise KeyError(f"{fp} missing act_move/acts")
    if "dt" in data:
        dt = float(np.array(data["dt"]).reshape(-1)[0])
    else:
        dt = 0.1
    return obs, move, dt


def _stack_obs(obs: np.ndarray, k: int) -> np.ndarray:
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


def _to_metric_pos(obs: np.ndarray) -> np.ndarray:
    # obs: [T, 23, 3] in normalized coords
    pos = obs[:, :11, 0:2].copy()
    pos[:, :, 0] = pos[:, :, 0] * 52.5 + 52.5
    pos[:, :, 1] = pos[:, :, 1] * 34.0 + 34.0
    return pos


def _compute_metrics(pred: np.ndarray, tgt: np.ndarray, speed_thr: float, dir_speed_thr: float) -> Dict[str, float]:
    # pred/tgt: [T, 11, 2]
    diff = pred - tgt
    mse = float(np.mean(diff ** 2))

    sp_t = np.linalg.norm(tgt, axis=2)
    sp_p = np.linalg.norm(pred, axis=2)
    mag_err = float(np.mean(np.abs(sp_p - sp_t)))

    dot = np.sum(pred * tgt, axis=2)
    denom = (sp_p * sp_t) + 1e-8
    cos = dot / denom
    # Direction should be judged by target speed only (avoid "predict tiny v" hack)
    valid = (sp_t > dir_speed_thr)
    cos_mean = float(np.mean(cos[valid])) if np.any(valid) else 0.0

    # Speed segments
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

    out = {
        "mse": mse,
        "cosine": cos_mean,
        "mag_err": mag_err,
    }
    out.update(segs)
    return out


def _baseline_mse(tgt: np.ndarray) -> Dict[str, float]:
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


def _multistep_ade_fde(pred: np.ndarray, pos: np.ndarray, dt: float, horizon: int) -> Tuple[float, float]:
    t = pred.shape[0]
    if t <= horizon:
        return float("nan"), float("nan")
    # cumulative sum of velocity
    zeros = np.zeros((1, pred.shape[1], 2), dtype=np.float32)
    cs = np.concatenate([zeros, np.cumsum(pred, axis=0)], axis=0) * dt
    total_ade = 0.0
    total_fde = 0.0
    count_steps = 0
    count_fde = 0
    for i in range(t - horizon):
        base = pos[i]
        for s in range(1, horizon + 1):
            disp = cs[i + s] - cs[i]
            pred_pos = base + disp
            gt_pos = pos[i + s]
            err = np.linalg.norm(pred_pos - gt_pos, axis=1)
            total_ade += float(err.sum())
            count_steps += err.shape[0]
            if s == horizon:
                total_fde += float(err.sum())
                count_fde += err.shape[0]
    ade = total_ade / max(count_steps, 1)
    fde = total_fde / max(count_fde, 1)
    return ade, fde


def eval_files(
    files: List[str],
    checkpoint: str,
    obs_history: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    device: str,
    horizon: int,
    speed_thr: float,
    dir_speed_thr: float,
    max_frames: int,
):
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model", state)
    move_horizon = _infer_move_horizon(state_dict)
    model = SoccerPolicy(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        input_dim=3 * obs_history,
        move_horizon=move_horizon,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict mismatch. Missing={missing} Unexpected={unexpected}")
    model.eval()

    # aggregate
    agg = {
        "count": 0,
        "mse": 0.0,
        "cosine": 0.0,
        "mag_err": 0.0,
        "ade": 0.0,
        "fde": 0.0,
        "baseline_zero_mse": 0.0,
        "baseline_mean_mse": 0.0,
        "baseline_last_mse": 0.0,
    }

    for fp in files:
        obs, move, dt = _load_npz(fp)
        if max_frames > 0:
            obs = obs[:max_frames]
            move = move[:max_frames]

        stacked = _stack_obs(obs, obs_history)
        t = stacked.shape[0]
        preds = []
        with torch.no_grad():
            for i in range(0, t, batch_size):
                batch = torch.from_numpy(stacked[i:i + batch_size]).float().to(device)
                pred_vel, _, _, _ = model(batch)
                preds.append(pred_vel.cpu().numpy())
        pred_vel = np.concatenate(preds, axis=0)
        # If multi-step, use step-0 for one-step metrics (consistent with act_move)
        if pred_vel.ndim == 4:
            pred_vel_step0 = pred_vel[:, 0]
        else:
            pred_vel_step0 = pred_vel

        metrics = _compute_metrics(pred_vel_step0, move, speed_thr, dir_speed_thr)
        base = _baseline_mse(move)
        pos = _to_metric_pos(obs)
        ade, fde = _multistep_ade_fde(pred_vel_step0, pos, dt, horizon)

        n = move.shape[0] * move.shape[1]
        agg["count"] += n
        for k in ["mse", "cosine", "mag_err"]:
            agg[k] += metrics[k] * n
        for k in ["baseline_zero_mse", "baseline_mean_mse", "baseline_last_mse"]:
            agg[k] += base[k] * n
        agg["ade"] += ade * n
        agg["fde"] += fde * n

        print(f"\n=== {os.path.basename(fp)} ===")
        print(f"Frames: {move.shape[0]} | dt={dt}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Cosine (tgt speed>{dir_speed_thr}): {metrics['cosine']:.4f}")
        print(f"|v| err: {metrics['mag_err']:.4f}")
        print(f"Baseline MSE (zero/mean/last): {base['baseline_zero_mse']:.6f} / {base['baseline_mean_mse']:.6f} / {base['baseline_last_mse']:.6f}")
        print(f"Multi-step ADE/FDE (K={horizon}): {ade:.4f} / {fde:.4f}")
        print(f"Speed segments (thr={speed_thr} m/s):")
        print(f"  low(<2)  mse={metrics['low(<2)_mse']:.6f} | cos={metrics['low(<2)_cos']:.4f} | |v|err={metrics['low(<2)_mag_err']:.4f}")
        print(f"  mid(2-4) mse={metrics['mid(2-4)_mse']:.6f} | cos={metrics['mid(2-4)_cos']:.4f} | |v|err={metrics['mid(2-4)_mag_err']:.4f}")
        print(f"  high(>4) mse={metrics['high(>4)_mse']:.6f} | cos={metrics['high(>4)_cos']:.4f} | |v|err={metrics['high(>4)_mag_err']:.4f}")

    if agg["count"] > 0:
        for k in ["mse", "cosine", "mag_err", "baseline_zero_mse", "baseline_mean_mse", "baseline_last_mse", "ade", "fde"]:
            agg[k] /= agg["count"]
    print("\n=== AGGREGATE ===")
    print(f"MSE: {agg['mse']:.6f}")
    print(f"Cosine: {agg['cosine']:.4f}")
    print(f"|v| err: {agg['mag_err']:.4f}")
    print(f"Baseline MSE (zero/mean/last): {agg['baseline_zero_mse']:.6f} / {agg['baseline_mean_mse']:.6f} / {agg['baseline_last_mse']:.6f}")
    print(f"Multi-step ADE/FDE (K={horizon}): {agg['ade']:.4f} / {agg['fde']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate movement quality for SoccerPolicy")
    parser.add_argument("--data", type=str, default=os.path.join(PROJECT_ROOT, "data", "skillcorner_processed", "npz"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth"))
    parser.add_argument("--obs_history", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--speed_thr", type=float, default=4.0)
    parser.add_argument("--dir_speed_thr", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames per file for quick test")
    args = parser.parse_args()

    files = _list_npz_files(args.data)
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data}")

    eval_files(
        files=files,
        checkpoint=args.checkpoint,
        obs_history=args.obs_history,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        device=args.device,
        horizon=args.horizon,
        speed_thr=args.speed_thr,
        dir_speed_thr=args.dir_speed_thr,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
