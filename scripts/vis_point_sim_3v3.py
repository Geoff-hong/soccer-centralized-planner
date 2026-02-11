#!/usr/bin/env python3
import argparse
import os
import sys
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "train"))

from model import SoccerPolicy  # noqa: E402
from point_sim_env import PointSim3v3  # noqa: E402
from point_sim_policy import DefenderPolicy, MixedTacticPolicy, PASS_SHORT, PASS_THROUGH  # noqa: E402

PASS_SHOT = 3
from point_sim_render_gif import _render  # noqa: E402


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


def _infer_field_scales(data, field_L, field_W):
    if field_L is None:
        for key in ("field_L", "field_length", "pitch_length", "L"):
            if key in data:
                field_L = _maybe_scalar(data.get(key))
                break
    if field_W is None:
        for key in ("field_W", "field_width", "pitch_width", "W"):
            if key in data:
                field_W = _maybe_scalar(data.get(key))
                break
    if field_L is None:
        field_L = 105.0
    if field_W is None:
        field_W = 68.0
    return float(field_L), float(field_W)


def _norm_to_metric(pos_norm: np.ndarray, scale_x: float, scale_y: float, origin: str) -> np.ndarray:
    pos = pos_norm.copy()
    pos[..., 0] = pos[..., 0] * scale_x
    pos[..., 1] = pos[..., 1] * scale_y
    if origin == "corner":
        pos[..., 0] += scale_x
        pos[..., 1] += scale_y
    return pos


def _metric_to_norm(pos_m: np.ndarray, scale_x: float, scale_y: float, origin: str) -> np.ndarray:
    pos = pos_m.copy()
    if origin == "corner":
        pos[..., 0] -= scale_x
        pos[..., 1] -= scale_y
    pos[..., 0] = pos[..., 0] / scale_x
    pos[..., 1] = pos[..., 1] / scale_y
    return pos


def _stack_history(history, obs_history):
    frames = list(history)
    if len(frames) < obs_history:
        pad = [frames[0]] * (obs_history - len(frames))
        frames = pad + frames
    stacked = np.stack(frames, axis=0)  # [H, N, 3]
    return stacked.transpose(1, 0, 2).reshape(stacked.shape[1], obs_history * 3)


def _infer_move_horizon(state_dict, default_horizon=1):
    for key in ("head_move.2.weight", "head_move.2.bias"):
        if key in state_dict:
            out_dim = int(state_dict[key].shape[0])
            if out_dim % 2 == 0 and out_dim > 0:
                return max(1, out_dim // 2)
            return 1
    return default_horizon


def _load_model(checkpoint, obs_history, d_model, nhead, num_layers, dropout, num_agents, num_attackers, device):
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model", state)
    move_horizon = _infer_move_horizon(state_dict, default_horizon=1)
    model = SoccerPolicy(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        input_dim=3 * obs_history,
        move_horizon=move_horizon,
        num_agents=num_agents,
        num_attackers=num_attackers,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict mismatch. Missing={missing} Unexpected={unexpected}")
    model.eval()
    return model


def _tune_pass_helper(pass_helper, lead_time_cap, lead_max_offset, lead_long_dist_ref, lead_y_scale):
    if lead_time_cap is not None:
        pass_helper.lead_time_cap = float(lead_time_cap)
    if lead_max_offset is not None:
        pass_helper.lead_max_offset = float(lead_max_offset)
    if lead_long_dist_ref is not None:
        pass_helper.lead_long_dist_ref = float(lead_long_dist_ref)
    if lead_y_scale is not None:
        pass_helper.lead_y_scale = float(lead_y_scale)


def _list_npz_files(data_dir: str) -> List[str]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    files.sort()
    return [os.path.join(data_dir, f) for f in files]


def _compute_has_ball(att_pos_norm, def_pos_norm, ball_pos_norm, scale_x, scale_y, origin, trigger_dist_thr):
    att_m = _norm_to_metric(att_pos_norm, scale_x, scale_y, origin)
    def_m = _norm_to_metric(def_pos_norm, scale_x, scale_y, origin) if def_pos_norm.size > 0 else def_pos_norm
    ball_m = _norm_to_metric(ball_pos_norm, scale_x, scale_y, origin)
    att_dist = np.linalg.norm(att_m - ball_m, axis=1)
    def_dist = np.linalg.norm(def_m - ball_m, axis=1) if def_m.size > 0 else np.zeros(0, dtype=np.float32)
    att_has = (att_dist < trigger_dist_thr).astype(np.float32)
    def_has = (def_dist < trigger_dist_thr).astype(np.float32)
    return att_has, def_has


def _rollout_move_only(
    obs: np.ndarray,
    move: Optional[np.ndarray],
    dt: np.ndarray,
    model,
    obs_history: int,
    use_move_residual: bool,
    coord_origin: str,
    field_L: float,
    field_W: float,
    max_steps: int,
    trigger_dist_thr: float,
    device: str,
):
    n_agents = obs.shape[1]
    n_att = move.shape[1] if move is not None else (n_agents - 1) // 2
    scale_x = field_L / 2.0
    scale_y = field_W / 2.0
    steps = int(max_steps) if int(max_steps) > 0 else obs.shape[0]

    att_pos_norm = obs[0, :n_att, 0:2].copy()
    history = deque(maxlen=obs_history)
    out_frames = []

    for t in range(steps):
        def_pos_norm = obs[t, n_att : n_agents - 1, 0:2]
        ball_pos_norm = obs[t, n_agents - 1, 0:2]
        att_has, def_has = _compute_has_ball(
            att_pos_norm, def_pos_norm, ball_pos_norm, scale_x, scale_y, coord_origin, trigger_dist_thr
        )
        frame = np.zeros((n_agents, 3), dtype=np.float32)
        frame[:n_att, 0:2] = att_pos_norm
        frame[n_att : n_agents - 1, 0:2] = def_pos_norm
        frame[n_agents - 1, 0:2] = ball_pos_norm
        frame[:n_att, 2] = att_has
        frame[n_att : n_agents - 1, 2] = def_has
        frame[n_agents - 1, 2] = 0.0

        history.append(frame)
        stacked = _stack_history(history, obs_history)
        x = torch.from_numpy(stacked).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_vel, _, _, _ = model(x)
        if pred_vel.ndim == 4:
            pred_step = pred_vel[:, 0, :, :].squeeze(0).cpu().numpy()
        else:
            pred_step = pred_vel.squeeze(0).cpu().numpy()

        if use_move_residual and len(history) >= 2:
            last = history[-1][:n_att, 0:2]
            prev = history[-2][:n_att, 0:2]
            last_m = _norm_to_metric(last, scale_x, scale_y, coord_origin)
            prev_m = _norm_to_metric(prev, scale_x, scale_y, coord_origin)
            base_vel = (last_m - prev_m) / max(float(dt[t]), 1e-6)
            pred_step = pred_step + base_vel

        # Update attacker positions
        att_m = _norm_to_metric(att_pos_norm, scale_x, scale_y, coord_origin)
        att_m = att_m + pred_step * float(dt[t])
        att_pos_norm = _metric_to_norm(att_m, scale_x, scale_y, coord_origin)
        att_pos_norm = np.clip(att_pos_norm, -1.0, 1.0)

        out_frames.append(frame)

    return np.asarray(out_frames, dtype=np.float32)


def _init_env_from_obs(env: PointSim3v3, obs: np.ndarray, obs_next: Optional[np.ndarray], dt: float):
    n_agents = obs.shape[0]
    n_att = (n_agents - 1) // 2
    scale_x = env.half_L
    scale_y = env.half_W

    pos_a = obs[:n_att, 0:2] * np.array([scale_x, scale_y], dtype=np.float32)
    pos_d = obs[n_att : n_agents - 1, 0:2] * np.array([scale_x, scale_y], dtype=np.float32)
    ball_pos = obs[n_agents - 1, 0:2] * np.array([scale_x, scale_y], dtype=np.float32)

    vel_a = np.zeros_like(pos_a)
    vel_d = np.zeros_like(pos_d)
    ball_vel = np.zeros(2, dtype=np.float32)
    if obs_next is not None:
        pos_a1 = obs_next[:n_att, 0:2] * np.array([scale_x, scale_y], dtype=np.float32)
        pos_d1 = obs_next[n_att : n_agents - 1, 0:2] * np.array([scale_x, scale_y], dtype=np.float32)
        ball_pos1 = obs_next[n_agents - 1, 0:2] * np.array([scale_x, scale_y], dtype=np.float32)
        vel_a = (pos_a1 - pos_a) / max(dt, 1e-6)
        vel_d = (pos_d1 - pos_d) / max(dt, 1e-6)
        ball_vel = (ball_pos1 - ball_pos) / max(dt, 1e-6)

    has_ball = obs[: n_agents - 1, 2]
    owner_team = -1
    owner_idx = -1
    if has_ball.max() > 0.5:
        idx = int(np.argmax(has_ball))
        if idx < n_att:
            owner_team = 0
            owner_idx = idx
        else:
            owner_team = 1
            owner_idx = idx - n_att

    env.pos_a = pos_a.copy()
    env.pos_d = pos_d.copy()
    env.vel_a = vel_a.copy()
    env.vel_d = vel_d.copy()
    env.ball_pos = ball_pos.copy()
    env.ball_vel = ball_vel.copy()
    if owner_team >= 0:
        env.ball_mode = "possessed"
        env.owner_team = owner_team
        env.owner_idx = owner_idx
    else:
        env.ball_mode = "free"
        env.owner_team = -1
        env.owner_idx = -1
    env.pass_lock_remaining = 0
    env.pass_lock_elapsed = 0
    env.pass_intent_team = -1
    env.pass_intent_idx = -1
    env.pass_sender_team = -1
    env.pass_sender_idx = -1
    env.pass_flight_steps = 0
    env.owner_protect_steps = 0
    env.def_possess_steps = 0


def _rollout_full(
    obs: np.ndarray,
    dt: np.ndarray,
    model,
    obs_history: int,
    use_move_residual: bool,
    max_steps: int,
    pass_thr: float,
    dist_long: float,
    dx_long: float,
    dy_long: float,
    lead_vel_blend: float,
    assist_receive: bool,
    assist_alpha: float,
    lead_vel_gain_long: float,
    short_tau_min: float,
    short_tau_max: float,
    through_tau_min: float,
    through_tau_max: float,
    lead_time_cap: float,
    lead_max_offset: float,
    lead_long_dist_ref: float,
    lead_y_scale: float,
    recv_control_mult: float,
    recv_control_sec: float,
    device: str,
    seed: int,
):
    n_agents = obs.shape[1]
    n_att = (n_agents - 1) // 2
    env = PointSim3v3(
        dt=float(dt[0]),
        seed=seed,
        recv_control_mult=recv_control_mult,
        recv_control_sec=recv_control_sec,
    )
    _init_env_from_obs(env, obs[0], obs[1] if obs.shape[0] > 1 else None, float(dt[0]))
    def_policy = DefenderPolicy()
    def_policy.reset(np.random.default_rng(seed + 13))
    pass_helper = MixedTacticPolicy(dt=float(dt[0]))
    pass_helper.reset(np.random.default_rng(seed + 7))
    _tune_pass_helper(pass_helper, lead_time_cap, lead_max_offset, lead_long_dist_ref, lead_y_scale)

    history = deque(maxlen=obs_history)
    out_frames = []
    pass_flag = []
    passer_id = []
    receiver_id = []

    steps = int(max_steps) if int(max_steps) > 0 else obs.shape[0]
    assist_timer = 0
    assist_recv = -1
    assist_target = None
    # track termination (debug only in previous version)
    for t in range(steps):
        frame = env.get_obs()
        history.append(frame)
        stacked = _stack_history(history, obs_history)
        x = torch.from_numpy(stacked).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_vel, pred_pass, pred_passer, pred_receiver = model(x)
        if pred_vel.ndim == 4:
            pred_step = pred_vel[:, 0, :, :].squeeze(0).cpu().numpy()
        else:
            pred_step = pred_vel.squeeze(0).cpu().numpy()

        if use_move_residual and len(history) >= 2:
            last = history[-1][:n_att, 0:2]
            prev = history[-2][:n_att, 0:2]
            last_m = last * np.array([env.half_L, env.half_W], dtype=np.float32)
            prev_m = prev * np.array([env.half_L, env.half_W], dtype=np.float32)
            base_vel = (last_m - prev_m) / max(float(env.dt), 1e-6)
            pred_step = pred_step + base_vel

        v_cmd_att = pred_step.copy()
        v_cmd_def = def_policy.act(env)

        pass_event = None
        pass_flag.append(0.0)
        passer_id.append(-1)
        receiver_id.append(-1)

        pass_prob = float(torch.sigmoid(pred_pass).squeeze().cpu().item())
        if (
            pass_prob > pass_thr
            and env.ball_mode == "possessed"
            and env.owner_team == 0
            and 0 <= env.owner_idx < n_att
        ):
            recv_logits = pred_receiver.squeeze(0).cpu().numpy()
            order = np.argsort(-recv_logits)
            recv = int(order[0])
            goal_id = n_att
            if recv == env.owner_idx and recv != goal_id and len(order) > 1:
                recv = int(order[1])
            if recv == goal_id:
                passer_pos = env.pos_a[env.owner_idx]
                goal = env._goal_center(0)
                shot_speed = pass_helper._shot_speed(float(np.linalg.norm(goal - passer_pos)))
                pass_event = {
                    "kind": "shot",
                    "passer_id": env.owner_idx,
                    "receiver_id": goal_id,
                    "pass_type": PASS_SHOT,
                    "target_pos": goal,
                    "speed": shot_speed,
                }
                pass_flag[-1] = 1.0
                passer_id[-1] = env.owner_idx
                receiver_id[-1] = goal_id
            elif recv != env.owner_idx:
                passer_pos = env.pos_a[env.owner_idx]
                recv_pos = env.pos_a[recv]
                recv_vel = env.vel_a[recv]
                recv_vel_for_lead = (1.0 - lead_vel_blend) * recv_vel + lead_vel_blend * v_cmd_att[recv]
                delta = recv_pos - passer_pos
                dist = float(np.linalg.norm(delta))
                dx = float(delta[0])
                dy = float(delta[1])
                if dist > dist_long:
                    recv_vel_for_lead = recv_vel_for_lead * lead_vel_gain_long
                if dist > dist_long or dx > dx_long or abs(dy) > dy_long:
                    pass_type = PASS_THROUGH
                    tau_range = (through_tau_min, through_tau_max)
                else:
                    pass_type = PASS_SHORT
                    tau_range = (short_tau_min, short_tau_max)
                res = pass_helper._make_pass(
                    pass_type,
                    passer_pos,
                    recv_pos,
                    recv_vel_for_lead,
                    recv_pos,
                    tau_range,
                    pressured=False,
                    forward_required=False,
                )
                if res is not None:
                    target, speed, tau = res
                    pass_event = {
                        "kind": "pass",
                        "passer_id": env.owner_idx,
                        "receiver_id": recv,
                        "pass_type": pass_type,
                        "target_pos": target,
                        "speed": speed,
                    }
                    pass_flag[-1] = 1.0
                    passer_id[-1] = env.owner_idx
                    receiver_id[-1] = recv
                    if assist_receive:
                        assist_timer = max(1, int(round(tau / max(env.dt, 1e-6))))
                        assist_recv = recv
                        assist_target = target.copy()

        if assist_receive and assist_timer > 0 and assist_recv >= 0 and assist_target is not None:
            recv_pos = env.pos_a[assist_recv]
            delta = assist_target - recv_pos
            v_to = delta / max(env.dt, 1e-6)
            v_norm = np.linalg.norm(v_to)
            if v_norm > env.v_max:
                v_to = v_to / v_norm * env.v_max
            v_cmd_att[assist_recv] = (1.0 - assist_alpha) * v_cmd_att[assist_recv] + assist_alpha * v_to
            assist_timer -= 1

        done, outcome = env.step(v_cmd_att, v_cmd_def, pass_event)
        out_frames.append(frame)
        if done:
            break

    return (
        np.asarray(out_frames, dtype=np.float32),
        np.asarray(pass_flag, dtype=np.float32),
        np.asarray(passer_id, dtype=np.int64),
        np.asarray(receiver_id, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser(description="3v3 point-sim visualization for trained model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", default="", help="Episode npz to visualize")
    parser.add_argument("--data_dir", default="", help="Directory of npz episodes to sample")
    parser.add_argument("--num", type=int, default=10, help="Number of episodes to render when using --data_dir")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle episodes when using --data_dir")
    parser.add_argument("--gap_frames", type=int, default=3, help="Gap frames between episodes")
    parser.add_argument("--mode", choices=["move_only", "full"], default="move_only")
    parser.add_argument("--obs_history", type=int, default=5)
    parser.add_argument("--out", required=True, help="Output .gif or .mp4")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--coord_origin", choices=["center", "corner"], default="center")
    parser.add_argument("--field_L", type=float, default=None)
    parser.add_argument("--field_W", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_move_residual", action="store_true")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trigger_dist_thr", type=float, default=3.0)
    parser.add_argument("--pass_thr", type=float, default=0.5)
    parser.add_argument("--dist_long", type=float, default=24.0)
    parser.add_argument("--dx_long", type=float, default=6.0)
    parser.add_argument("--dy_long", type=float, default=12.0)
    parser.add_argument("--lead_vel_blend", type=float, default=0.5, help="Blend factor for receiver velocity in lead calc")
    parser.add_argument("--lead_vel_gain_long", type=float, default=0.8, help="Extra gain on receiver vel for long passes")
    parser.add_argument("--short_tau_min", type=float, default=0.5)
    parser.add_argument("--short_tau_max", type=float, default=0.7)
    parser.add_argument("--through_tau_min", type=float, default=0.6)
    parser.add_argument("--through_tau_max", type=float, default=0.8)
    parser.add_argument("--lead_time_cap", type=float, default=0.68)
    parser.add_argument("--lead_max_offset", type=float, default=4.5)
    parser.add_argument("--lead_long_dist_ref", type=float, default=35.0)
    parser.add_argument("--lead_y_scale", type=float, default=0.97)
    parser.add_argument("--assist_receive", action="store_true", help="Enable receiver assist after pass")
    parser.add_argument("--assist_alpha", type=float, default=0.7)
    parser.add_argument("--recv_control_mult", type=float, default=1.5)
    parser.add_argument("--recv_control_sec", type=float, default=0.6, help="Intended receiver control window (sec)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.npz and not args.data_dir:
        raise ValueError("Provide --npz or --data_dir")

    def _load_episode(path):
        data = np.load(path)
        obs = data["obs"]
        move = data.get("act_move", None)
        dt_val = data.get("dt", 0.1)
        if isinstance(dt_val, np.ndarray):
            if dt_val.ndim == 0:
                dt_val = float(dt_val)
                dt = np.full(obs.shape[0], dt_val, dtype=np.float32)
            else:
                dt = dt_val.astype(np.float32)
        else:
            dt = np.full(obs.shape[0], float(dt_val), dtype=np.float32)
        field_L, field_W = _infer_field_scales(data, args.field_L, args.field_W)
        data.close()
        return obs, move, dt, field_L, field_W

    model = None
    if args.data_dir:
        files = _list_npz_files(args.data_dir)
        if not files:
            raise FileNotFoundError(f"No npz files found in {args.data_dir}")
        if args.shuffle:
            rng = np.random.default_rng(args.seed)
            rng.shuffle(files)
        files = files[: max(1, int(args.num))]

        obs_all = []
        pass_all = []
        passer_all = []
        receiver_all = []
        episode_ids = []

        dt_ref = None
        for ep_idx, fp in enumerate(files):
            obs, move, dt, field_L, field_W = _load_episode(fp)
            n_agents = obs.shape[1]
            n_att = move.shape[1] if move is not None else (n_agents - 1) // 2
            if model is None:
                model = _load_model(
                    args.checkpoint,
                    args.obs_history,
                    args.d_model,
                    args.nhead,
                    args.num_layers,
                    args.dropout,
                    n_agents,
                    n_att,
                    args.device,
                )
            if dt_ref is None:
                dt_ref = float(dt[0])
            elif abs(float(dt[0]) - dt_ref) > 1e-6:
                print(f"[Warn] dt mismatch in {os.path.basename(fp)}: {float(dt[0])} vs {dt_ref}")

            if args.mode == "move_only":
                obs_pred = _rollout_move_only(
                    obs=obs,
                    move=move,
                    dt=dt,
                    model=model,
                    obs_history=args.obs_history,
                    use_move_residual=args.use_move_residual,
                    coord_origin=args.coord_origin,
                    field_L=field_L,
                    field_W=field_W,
                    max_steps=args.max_steps,
                    trigger_dist_thr=args.trigger_dist_thr,
                    device=args.device,
                )
                pass_flag = None
                passer_id = None
                receiver_id = None
            else:
                obs_pred, pass_flag, passer_id, receiver_id = _rollout_full(
                    obs=obs,
                    dt=dt,
                    model=model,
                    obs_history=args.obs_history,
                    use_move_residual=args.use_move_residual,
                    max_steps=args.max_steps,
                    pass_thr=args.pass_thr,
                    dist_long=args.dist_long,
                    dx_long=args.dx_long,
                    dy_long=args.dy_long,
                    lead_vel_blend=args.lead_vel_blend,
                    assist_receive=args.assist_receive,
                    assist_alpha=args.assist_alpha,
                    lead_vel_gain_long=args.lead_vel_gain_long,
                    short_tau_min=args.short_tau_min,
                    short_tau_max=args.short_tau_max,
                    through_tau_min=args.through_tau_min,
                    through_tau_max=args.through_tau_max,
                    lead_time_cap=args.lead_time_cap,
                    lead_max_offset=args.lead_max_offset,
                    lead_long_dist_ref=args.lead_long_dist_ref,
                    lead_y_scale=args.lead_y_scale,
                    recv_control_mult=args.recv_control_mult,
                    recv_control_sec=args.recv_control_sec,
                    device=args.device,
                    seed=args.seed + ep_idx,
                )

            obs_all.append(obs_pred)
            if pass_flag is None:
                pass_all.append(np.zeros(obs_pred.shape[0], dtype=np.float32))
                passer_all.append(np.zeros(obs_pred.shape[0], dtype=np.int64))
                receiver_all.append(np.zeros(obs_pred.shape[0], dtype=np.int64))
            else:
                pass_all.append(pass_flag)
                passer_all.append(passer_id)
                receiver_all.append(receiver_id)
            episode_ids.append(np.full(obs_pred.shape[0], ep_idx, dtype=np.int64))

            if args.gap_frames > 0 and ep_idx < len(files) - 1:
                gap = np.zeros((args.gap_frames, obs_pred.shape[1], obs_pred.shape[2]), dtype=np.float32)
                obs_all.append(gap)
                pass_all.append(np.zeros(args.gap_frames, dtype=np.float32))
                passer_all.append(np.zeros(args.gap_frames, dtype=np.int64))
                receiver_all.append(np.zeros(args.gap_frames, dtype=np.int64))
                episode_ids.append(np.full(args.gap_frames, ep_idx, dtype=np.int64))

        obs_cat = np.concatenate(obs_all, axis=0)
        pass_cat = np.concatenate(pass_all, axis=0) if pass_all else None
        passer_cat = np.concatenate(passer_all, axis=0) if passer_all else None
        receiver_cat = np.concatenate(receiver_all, axis=0) if receiver_all else None
        episode_cat = np.concatenate(episode_ids, axis=0)

        _render(
            obs_cat,
            dt_ref or 0.1,
            args.out,
            every=1,
            dpi=120,
            pass_flag=pass_cat,
            passer_id=passer_cat,
            receiver_id=receiver_cat,
            episode_ids=episode_cat,
        )
        print(f"Saved to {args.out}")
        return

    # single episode mode
    obs, move, dt, field_L, field_W = _load_episode(args.npz)
    n_agents = obs.shape[1]
    n_att = move.shape[1] if move is not None else (n_agents - 1) // 2

    if model is None:
        model = _load_model(
            args.checkpoint,
            args.obs_history,
            args.d_model,
            args.nhead,
            args.num_layers,
            args.dropout,
            n_agents,
            n_att,
            args.device,
        )

    if args.mode == "move_only":
        obs_pred = _rollout_move_only(
            obs=obs,
            move=move,
            dt=dt,
            model=model,
            obs_history=args.obs_history,
            use_move_residual=args.use_move_residual,
            coord_origin=args.coord_origin,
            field_L=field_L,
            field_W=field_W,
            max_steps=args.max_steps,
            trigger_dist_thr=args.trigger_dist_thr,
            device=args.device,
        )
        _render(obs_pred, float(dt[0]), args.out, every=1, dpi=120)
    else:
        obs_pred, pass_flag, passer_id, receiver_id = _rollout_full(
            obs=obs,
            dt=dt,
            model=model,
            obs_history=args.obs_history,
            use_move_residual=args.use_move_residual,
            max_steps=args.max_steps,
            pass_thr=args.pass_thr,
            dist_long=args.dist_long,
            dx_long=args.dx_long,
            dy_long=args.dy_long,
            lead_vel_blend=args.lead_vel_blend,
            assist_receive=args.assist_receive,
            assist_alpha=args.assist_alpha,
            lead_vel_gain_long=args.lead_vel_gain_long,
            short_tau_min=args.short_tau_min,
            short_tau_max=args.short_tau_max,
            through_tau_min=args.through_tau_min,
            through_tau_max=args.through_tau_max,
            lead_time_cap=args.lead_time_cap,
            lead_max_offset=args.lead_max_offset,
            lead_long_dist_ref=args.lead_long_dist_ref,
            lead_y_scale=args.lead_y_scale,
            recv_control_mult=args.recv_control_mult,
            recv_control_sec=args.recv_control_sec,
            device=args.device,
            seed=args.seed,
        )
        _render(
            obs_pred,
            float(dt[0]),
            args.out,
            every=1,
            dpi=120,
            pass_flag=pass_flag,
            passer_id=passer_id,
            receiver_id=receiver_id,
        )

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
