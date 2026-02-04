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


def _normalize_dt(dt_val, n_frames: int) -> np.ndarray:
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


def load_npz(fp: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(fp)
    obs = data["obs"]
    if "act_move" in data:
        move = data["act_move"]
    elif "acts" in data:
        move = data["acts"][:, :, 0:2]
    else:
        raise KeyError(f"{fp} missing act_move/acts")
    dt = _normalize_dt(data.get("dt", 0.1), len(obs))

    move_unit = data.get("move_unit", None)
    if move_unit is not None:
        try:
            if hasattr(move_unit, "item"):
                move_unit = move_unit.item()
        except Exception:
            pass
        if isinstance(move_unit, bytes):
            move_unit = move_unit.decode("utf-8", errors="ignore")
        if isinstance(move_unit, str):
            unit_norm = move_unit.strip().lower()
            if unit_norm in {"mframe", "m/frame", "m_per_frame", "per_frame"}:
                dt_safe = np.clip(dt, 1e-6, None)
                move = move / dt_safe[:, None, None]

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
    use_move_residual: bool,
    move_pos_scale_x: float,
    move_pos_scale_y: float,
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
        obs, move, dt = load_npz(fp)
        if max_frames > 0:
            obs = obs[:max_frames]
            move = move[:max_frames]
            dt = dt[:max_frames]
        stacked = stack_obs(obs, obs_history)
        t = stacked.shape[0]
        base_vel = None
        if use_move_residual and obs_history >= 2:
            scale = np.array([move_pos_scale_x, move_pos_scale_y], dtype=np.float32)
            pos_last = stacked[:, :11, (obs_history - 1) * 3 : (obs_history - 1) * 3 + 2]
            pos_prev = stacked[:, :11, (obs_history - 2) * 3 : (obs_history - 2) * 3 + 2]
            dt_safe = np.clip(dt, 1e-6, None).reshape(-1, 1, 1)
            base_vel = (pos_last * scale - pos_prev * scale) / dt_safe

        preds = []
        with torch.no_grad():
            for i in range(0, t, batch_size):
                x = torch.from_numpy(stacked[i : i + batch_size]).to(device)
                pred_vel, _, _, _ = model(x)
                if pred_vel.ndim == 4:
                    pred_vel = pred_vel[:, 0]
                preds.append(pred_vel.cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        if use_move_residual and base_vel is not None:
            pred = pred + base_vel

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
    parser.add_argument("--use_move_residual", type=int, default=1, help="1 to add base velocity (match training), 0 to use raw head output")
    parser.add_argument("--move_pos_scale_x", type=float, default=52.5)
    parser.add_argument("--move_pos_scale_y", type=float, default=34.0)
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
        bool(args.use_move_residual),
        args.move_pos_scale_x,
        args.move_pos_scale_y,
    )


if __name__ == "__main__":
    main()
