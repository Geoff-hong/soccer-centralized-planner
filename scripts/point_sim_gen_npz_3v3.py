#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

from point_sim_env import PointSim3v3  # noqa: E402
from point_sim_policy import DefenderPolicy, MixedTacticPolicy  # noqa: E402


def run_episode(env, policy, def_policy, max_steps, debug_tactics=False):
    obs_list = []
    pos_list = []
    pass_flag_list = []
    passer_list = []
    receiver_list = []
    pass_type_list = []
    template_id_list = []
    template_switch_list = []
    owner_team_list = []
    owner_idx_list = []
    prev_template = None

    outcome = "timeout"
    for _ in range(max_steps):
        obs = env.get_obs()
        obs_list.append(obs)
        pos_list.append(env.pos_a.copy())
        if debug_tactics:
            owner_team_list.append(int(env.owner_team))
            owner_idx_list.append(int(env.owner_idx))

        v_cmd_att, pass_event = policy.act(env)
        if debug_tactics:
            tid = int(getattr(policy, "template_id", -1))
            template_id_list.append(tid)
            if prev_template is None:
                template_switch_list.append(0)
            else:
                template_switch_list.append(int(tid != prev_template))
            prev_template = tid
        v_cmd_def = def_policy.act(env)

        if pass_event is not None and pass_event.get("kind") == "pass":
            pass_flag_list.append(1)
            passer_list.append(int(pass_event.get("passer_id", 0)))
            receiver_list.append(int(pass_event.get("receiver_id", 0)))
            pass_type_list.append(int(pass_event.get("pass_type", 0)))
        else:
            pass_flag_list.append(0)
            passer_list.append(0)
            receiver_list.append(0)
            pass_type_list.append(0)

        done, outcome_now = env.step(v_cmd_att, v_cmd_def, pass_event)
        if done:
            outcome = outcome_now or "timeout"
            break

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    pos_arr = np.asarray(pos_list, dtype=np.float32)
    T = obs_arr.shape[0]
    dt = env.dt

    act_move = np.zeros((T, 3, 2), dtype=np.float32)
    if T > 1:
        act_move[:-1] = (pos_arr[1:] - pos_arr[:-1]) / dt
        act_move[-1] = act_move[-2]

    pass_flag = np.asarray(pass_flag_list, dtype=np.float32)
    passer_id = np.asarray(passer_list, dtype=np.int64)
    receiver_id = np.asarray(receiver_list, dtype=np.int64)
    pass_type = np.asarray(pass_type_list, dtype=np.int64)

    out = {
        "obs": obs_arr,
        "act_move": act_move,
        "dt": np.float32(dt),
        "pass_flag": pass_flag,
        "passer_id": passer_id,
        "receiver_id": receiver_id,
        "pass_type": pass_type,
        "pass_dir": pass_type,
        "episode_seed": np.int64(env.episode_seed),
        "outcome": outcome,
    }
    if debug_tactics:
        out["template_id"] = np.asarray(template_id_list, dtype=np.int64)
        out["template_switch"] = np.asarray(template_switch_list, dtype=np.int64)
        out["owner_team"] = np.asarray(owner_team_list, dtype=np.int64)
        out["owner_idx"] = np.asarray(owner_idx_list, dtype=np.int64)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate 3v3 point-sim npz episodes")
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data", "point_sim_3v3", "npz"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p_switch", type=float, default=0.02)
    parser.add_argument("--eps_action", type=float, default=0.05)
    parser.add_argument("--move_horizon", type=int, default=1)
    parser.add_argument("--save_tactics", action="store_true", help="Save template_id/switch info into npz")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)

    env = PointSim3v3(dt=0.1, seed=args.seed)
    policy = MixedTacticPolicy(p_switch=args.p_switch, eps_action=args.eps_action, dt=0.1, v_max=env.v_max)
    def_policy = DefenderPolicy(v_max=env.v_max)

    for ep in range(args.episodes):
        episode_seed = int(master_rng.integers(0, 2**32 - 1))
        # spawn per-episode RNGs for reproducibility
        ss = np.random.SeedSequence(episode_seed)
        child_seeds = ss.spawn(2)
        seed_env = int(child_seeds[0].generate_state(1)[0])
        seed_pol = int(child_seeds[1].generate_state(1)[0])
        env.reset(seed=seed_env)
        policy.reset(np.random.default_rng(seed_pol))
        def_policy.reset(np.random.default_rng(seed_pol + 1))

        data = run_episode(env, policy, def_policy, args.max_steps, debug_tactics=args.save_tactics)
        out_path = os.path.join(args.out_dir, f"ep_{ep:05d}.npz")
        np.savez_compressed(out_path, **data)

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"Generated {ep+1}/{args.episodes} episodes -> {out_path}")


if __name__ == "__main__":
    main()
