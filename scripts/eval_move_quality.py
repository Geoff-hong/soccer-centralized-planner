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


def _maybe_scalar(val):
    if val is None:
        return None
    try:
        if hasattr(val, "item"):
            return float(val.item())
    except Exception:
        pass
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        return float(val.reshape(-1)[0])
    try:
        return float(val)
    except Exception:
        return None


def _infer_field_scales(data_fields, move_pos_scale_x, move_pos_scale_y, field_L, field_W):
    if move_pos_scale_x is not None and move_pos_scale_y is not None:
        return float(move_pos_scale_x), float(move_pos_scale_y)

    if field_L is None:
        for key in ("field_L", "field_length", "pitch_length", "L"):
            if data_fields is not None and key in data_fields:
                field_L = _maybe_scalar(data_fields.get(key))
                break
    if field_W is None:
        for key in ("field_W", "field_width", "pitch_width", "W"):
            if data_fields is not None and key in data_fields:
                field_W = _maybe_scalar(data_fields.get(key))
                break

    if field_L is not None and field_W is not None:
        return float(field_L) / 2.0, float(field_W) / 2.0

    if move_pos_scale_x is not None:
        return float(move_pos_scale_x), float(move_pos_scale_y or move_pos_scale_x)

    return 52.5, 34.0


def _norm_to_metric(pos_norm: np.ndarray, scale_x: float, scale_y: float, origin: str) -> np.ndarray:
    pos = pos_norm.copy()
    pos[..., 0] = pos[..., 0] * scale_x
    pos[..., 1] = pos[..., 1] * scale_y
    if origin == "corner":
        pos[..., 0] += scale_x
        pos[..., 1] += scale_y
    return pos


def _load_npz(fp: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(fp)
    obs = data["obs"]
    if "act_move" in data:
        move = data["act_move"]
    elif "acts" in data:
        move = data["acts"][:, :, 0:2]
    else:
        raise KeyError(f"{fp} missing act_move/acts")

    dt = _normalize_dt(data.get("dt", 0.1), len(obs))

    # Normalize act_move to m/s if needed (legacy m/frame files).
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

    meta = {
        "pass_flag": data.get("pass_flag", None),
        "passer_id": data.get("passer_id", None),
        "receiver_id": data.get("receiver_id", None),
        "field_L": _maybe_scalar(data.get("field_L", None)),
        "field_W": _maybe_scalar(data.get("field_W", None)),
        "field_length": _maybe_scalar(data.get("field_length", None)),
        "field_width": _maybe_scalar(data.get("field_width", None)),
        "pitch_length": _maybe_scalar(data.get("pitch_length", None)),
        "pitch_width": _maybe_scalar(data.get("pitch_width", None)),
        "L": _maybe_scalar(data.get("L", None)),
        "W": _maybe_scalar(data.get("W", None)),
    }
    data.close()
    return obs, move, dt, meta


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


def _to_metric_pos(obs: np.ndarray, scale_x: float, scale_y: float, origin: str, n_att: int) -> np.ndarray:
    # obs: [T, N, 3] in normalized coords
    pos = obs[:, :n_att, 0:2].copy()
    pos = _norm_to_metric(pos, scale_x, scale_y, origin)
    return pos


def _compute_metrics(pred: np.ndarray, tgt: np.ndarray, speed_thr: float, dir_speed_thr: float) -> Dict[str, float]:
    # pred/tgt: [T, A, 2]
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


def _multistep_ade_fde(
    pred: np.ndarray, pos: np.ndarray, dt: np.ndarray, horizon: int
) -> Tuple[float, float]:
    t = pred.shape[0]
    if t <= horizon:
        return float("nan"), float("nan")
    dt_arr = np.asarray(dt, dtype=np.float32)
    if dt_arr.ndim == 0:
        dt_arr = np.full(t, float(dt_arr), dtype=np.float32)
    if dt_arr.shape[0] != t:
        dt_arr = _normalize_dt(dt_arr, t)
    dt_arr = dt_arr.reshape(-1, 1, 1)

    zeros = np.zeros((1, pred.shape[1], 2), dtype=np.float32)
    cs = np.concatenate([zeros, np.cumsum(pred * dt_arr, axis=0)], axis=0)
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
    use_move_residual: bool,
    move_pos_scale_x: float,
    move_pos_scale_y: float,
    coord_origin: str,
    field_L: float,
    field_W: float,
    eval_pass: bool,
    pass_thr: float,
    topk: int,
):
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model", state)
    move_horizon = _infer_move_horizon(state_dict)
    model = None

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

    pass_tp = pass_fp = pass_fn = 0
    passer_top1 = passer_topk = passer_total = 0
    receiver_top1 = receiver_topk = receiver_total = 0

    for fp in files:
        obs, move, dt, meta = _load_npz(fp)
        if max_frames > 0:
            obs = obs[:max_frames]
            move = move[:max_frames]
            dt = dt[:max_frames]

        n_agents = obs.shape[1]
        n_att = move.shape[1]
        if model is None:
            model = SoccerPolicy(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                input_dim=3 * obs_history,
                move_horizon=move_horizon,
                num_agents=n_agents,
                num_attackers=n_att,
            ).to(device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Warn] load_state_dict mismatch. Missing={missing} Unexpected={unexpected}")
            model.eval()

        scale_x, scale_y = _infer_field_scales(
            meta, move_pos_scale_x, move_pos_scale_y, field_L, field_W
        )

        stacked = _stack_obs(obs, obs_history)
        t = stacked.shape[0]
        base_vel = None
        if use_move_residual and obs_history >= 2:
            pos_last = stacked[:, :n_att, (obs_history - 1) * 3 : (obs_history - 1) * 3 + 2]
            pos_prev = stacked[:, :n_att, (obs_history - 2) * 3 : (obs_history - 2) * 3 + 2]
            pos_last_m = _norm_to_metric(pos_last, scale_x, scale_y, coord_origin)
            pos_prev_m = _norm_to_metric(pos_prev, scale_x, scale_y, coord_origin)
            dt_safe = np.clip(dt, 1e-6, None).reshape(-1, 1, 1)
            base_vel = (pos_last_m - pos_prev_m) / dt_safe
        preds = []
        pass_logits = []
        passer_logits = []
        receiver_logits = []
        with torch.no_grad():
            for i in range(0, t, batch_size):
                batch = torch.from_numpy(stacked[i:i + batch_size]).float().to(device)
                pred_vel, pred_pass, pred_passer, pred_receiver = model(batch)
                preds.append(pred_vel.cpu().numpy())
                pass_logits.append(pred_pass.cpu().numpy())
                passer_logits.append(pred_passer.cpu().numpy())
                receiver_logits.append(pred_receiver.cpu().numpy())
        pred_vel = np.concatenate(preds, axis=0)
        pred_pass = np.concatenate(pass_logits, axis=0).reshape(-1)
        pred_passer = np.concatenate(passer_logits, axis=0)
        pred_receiver = np.concatenate(receiver_logits, axis=0)
        # If multi-step, use step-0 for one-step metrics (consistent with act_move)
        if pred_vel.ndim == 4:
            pred_vel_step0 = pred_vel[:, 0]
        else:
            pred_vel_step0 = pred_vel
        if use_move_residual and base_vel is not None:
            pred_vel_step0 = pred_vel_step0 + base_vel

        metrics = _compute_metrics(pred_vel_step0, move, speed_thr, dir_speed_thr)
        base = _baseline_mse(move)
        pos = _to_metric_pos(obs, scale_x, scale_y, coord_origin, n_att)
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
        print(f"Frames: {move.shape[0]} | dt={float(np.array(dt).reshape(-1)[0]):.6f}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Cosine (tgt speed>{dir_speed_thr}): {metrics['cosine']:.4f}")
        print(f"|v| err: {metrics['mag_err']:.4f}")
        print(f"Baseline MSE (zero/mean/last): {base['baseline_zero_mse']:.6f} / {base['baseline_mean_mse']:.6f} / {base['baseline_last_mse']:.6f}")
        print(f"Multi-step ADE/FDE (K={horizon}): {ade:.4f} / {fde:.4f}")
        print(f"Speed segments (thr={speed_thr} m/s):")
        print(f"  low(<2)  mse={metrics['low(<2)_mse']:.6f} | cos={metrics['low(<2)_cos']:.4f} | |v|err={metrics['low(<2)_mag_err']:.4f}")
        print(f"  mid(2-4) mse={metrics['mid(2-4)_mse']:.6f} | cos={metrics['mid(2-4)_cos']:.4f} | |v|err={metrics['mid(2-4)_mag_err']:.4f}")
        print(f"  high(>4) mse={metrics['high(>4)_mse']:.6f} | cos={metrics['high(>4)_cos']:.4f} | |v|err={metrics['high(>4)_mag_err']:.4f}")

        if eval_pass and meta["pass_flag"] is not None:
            pass_flag = np.asarray(meta["pass_flag"]).reshape(-1)[:t]
            passer_id = np.asarray(meta["passer_id"]) if meta["passer_id"] is not None else None
            receiver_id = np.asarray(meta["receiver_id"]) if meta["receiver_id"] is not None else None

            pass_true = pass_flag > 0.5
            pass_pred = (1.0 / (1.0 + np.exp(-pred_pass))) > pass_thr
            pass_tp += int(np.logical_and(pass_pred, pass_true).sum())
            pass_fp += int(np.logical_and(pass_pred, ~pass_true).sum())
            pass_fn += int(np.logical_and(~pass_pred, pass_true).sum())

            if passer_id is not None:
                passer_id = passer_id.reshape(-1)[:t]
                mask = np.logical_and(pass_true, passer_id >= 0)
                if np.any(mask):
                    top1 = pred_passer.argmax(axis=1)
                    passer_top1 += int((top1[mask] == passer_id[mask]).sum())
                    topk_idx = np.argpartition(-pred_passer, min(topk, pred_passer.shape[1]) - 1, axis=1)[:, :topk]
                    ok = (topk_idx[mask] == passer_id[mask, None]).any(axis=1)
                    passer_topk += int(ok.sum())
                    passer_total += int(mask.sum())

            if receiver_id is not None:
                receiver_id = receiver_id.reshape(-1)[:t]
                mask = np.logical_and(pass_true, receiver_id >= 0)
                if np.any(mask):
                    top1 = pred_receiver.argmax(axis=1)
                    receiver_top1 += int((top1[mask] == receiver_id[mask]).sum())
                    topk_idx = np.argpartition(-pred_receiver, min(topk, pred_receiver.shape[1]) - 1, axis=1)[:, :topk]
                    ok = (topk_idx[mask] == receiver_id[mask, None]).any(axis=1)
                    receiver_topk += int(ok.sum())
                    receiver_total += int(mask.sum())

    if agg["count"] > 0:
        for k in ["mse", "cosine", "mag_err", "baseline_zero_mse", "baseline_mean_mse", "baseline_last_mse", "ade", "fde"]:
            agg[k] /= agg["count"]
    print("\n=== AGGREGATE ===")
    print(f"MSE: {agg['mse']:.6f}")
    print(f"Cosine: {agg['cosine']:.4f}")
    print(f"|v| err: {agg['mag_err']:.4f}")
    print(f"Baseline MSE (zero/mean/last): {agg['baseline_zero_mse']:.6f} / {agg['baseline_mean_mse']:.6f} / {agg['baseline_last_mse']:.6f}")
    print(f"Multi-step ADE/FDE (K={horizon}): {agg['ade']:.4f} / {agg['fde']:.4f}")
    if eval_pass and (pass_tp + pass_fp + pass_fn) > 0:
        precision = pass_tp / max(pass_tp + pass_fp, 1)
        recall = pass_tp / max(pass_tp + pass_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        print(f"Pass P/R/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        if passer_total > 0:
            print(f"Passer top1/top{topk}: {passer_top1 / passer_total:.4f} / {passer_topk / passer_total:.4f}")
        if receiver_total > 0:
            print(f"Receiver top1/top{topk}: {receiver_top1 / receiver_total:.4f} / {receiver_topk / receiver_total:.4f}")


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
    parser.add_argument("--use_move_residual", type=int, default=1, help="1 to add base velocity (match training), 0 to use raw head output")
    parser.add_argument("--move_pos_scale_x", type=float, default=None, help="Half-field scale X (meters). Overrides field_L.")
    parser.add_argument("--move_pos_scale_y", type=float, default=None, help="Half-field scale Y (meters). Overrides field_W.")
    parser.add_argument("--coord_origin", choices=["center", "corner"], default="center")
    parser.add_argument("--field_L", type=float, default=None, help="Full field length (meters). If provided, scale_x=field_L/2.")
    parser.add_argument("--field_W", type=float, default=None, help="Full field width (meters). If provided, scale_y=field_W/2.")
    parser.add_argument("--eval_pass", action="store_true", help="Evaluate pass heads if pass labels exist")
    parser.add_argument("--pass_thr", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=2)
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
        use_move_residual=bool(args.use_move_residual),
        move_pos_scale_x=args.move_pos_scale_x,
        move_pos_scale_y=args.move_pos_scale_y,
        coord_origin=args.coord_origin,
        field_L=args.field_L,
        field_W=args.field_W,
        eval_pass=args.eval_pass,
        pass_thr=args.pass_thr,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
