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

PASS_SHOT = 3


def run_episode(
    env,
    policy,
    def_policy,
    max_steps,
    debug_tactics=False,
    max_def_possess_steps=5,
    pass_success_window_steps=12,
    min_passes=0,
    min_len=0,
    require_shot_goal=False,
):
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
    pending_pass = None
    pass_attempts = 0
    pass_successes = 0
    shot_attempts = 0
    def_possess_run = 0
    def_possess_run_max = 0
    goal_id = int(getattr(env, "pos_a", np.zeros((3, 2))).shape[0])

    outcome = "timeout"
    invalid_reason = None
    for step in range(max_steps):
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

        if pass_event is not None:
            kind = pass_event.get("kind")
            if kind == "pass":
                pass_flag_list.append(1)
                passer_list.append(int(pass_event.get("passer_id", 0)))
                receiver_list.append(int(pass_event.get("receiver_id", 0)))
                pass_type_list.append(int(pass_event.get("pass_type", 0)))
                pass_attempts += 1
                if pending_pass is not None:
                    invalid_reason = "pass_overlap"
                    break
                pending_pass = {
                    "receiver": int(pass_event.get("receiver_id", -1)),
                    "deadline": step + pass_success_window_steps,
                }
            elif kind == "shot":
                pass_flag_list.append(1)
                passer_list.append(int(pass_event.get("passer_id", 0)))
                receiver_list.append(goal_id)
                pass_type_list.append(PASS_SHOT)
                shot_attempts += 1
            else:
                pass_flag_list.append(0)
                passer_list.append(0)
                receiver_list.append(0)
                pass_type_list.append(0)
        else:
            pass_flag_list.append(0)
            passer_list.append(0)
            receiver_list.append(0)
            pass_type_list.append(0)

        done, outcome_now = env.step(v_cmd_att, v_cmd_def, pass_event)

        # Track defender possession run
        if env.ball_mode == "possessed" and env.owner_team == 1:
            def_possess_run += 1
            def_possess_run_max = max(def_possess_run_max, def_possess_run)
            if def_possess_run >= max_def_possess_steps:
                invalid_reason = "def_possess"
                break
        else:
            def_possess_run = 0

        # Resolve pending pass
        if pending_pass is not None:
            if env.ball_mode == "possessed" and env.owner_team == 0 and env.owner_idx == pending_pass["receiver"]:
                pass_successes += 1
                pending_pass = None
            elif env.ball_mode == "possessed" and env.owner_team == 1:
                invalid_reason = "pass_failed_def"
                break
            elif done and (outcome_now in ("out", "def_possess")):
                invalid_reason = "pass_failed_out"
                break
            elif step >= pending_pass["deadline"]:
                invalid_reason = "pass_failed_timeout"
                break
            elif done and outcome_now is not None:
                invalid_reason = "pass_failed_done"
                break

        if done:
            outcome = outcome_now or "timeout"
            break

    if pending_pass is not None and invalid_reason is None:
        invalid_reason = "pass_failed_timeout"

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    pos_arr = np.asarray(pos_list, dtype=np.float32)
    T = obs_arr.shape[0]
    dt = env.dt

    # Post filters
    if invalid_reason is None:
        if min_len > 0 and T < min_len:
            invalid_reason = "too_short"
        if min_passes > 0 and pass_successes < min_passes:
            invalid_reason = "min_passes"
        if outcome not in ("goal", "timeout"):
            invalid_reason = "outcome_bad"
        if require_shot_goal and shot_attempts > 0 and outcome != "goal":
            invalid_reason = "shot_miss"

    act_move = np.zeros((T, 3, 2), dtype=np.float32)
    if T > 1:
        act_move[:-1] = (pos_arr[1:] - pos_arr[:-1]) / dt
        act_move[-1] = act_move[-2]

    pass_flag = np.asarray(pass_flag_list, dtype=np.float32)
    passer_id = np.asarray(passer_list, dtype=np.int64)
    receiver_id = np.asarray(receiver_list, dtype=np.int64)
    pass_type = np.asarray(pass_type_list, dtype=np.int64)

    info = {
        "invalid_reason": invalid_reason,
        "frames": T,
        "pass_attempts": pass_attempts,
        "pass_successes": pass_successes,
        "shot_attempts": shot_attempts,
        "def_possess_run_max": def_possess_run_max,
        "outcome": outcome,
    }

    if invalid_reason is not None:
        return None, info

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
    return out, info


def main():
    parser = argparse.ArgumentParser(description="Generate 3v3 point-sim npz episodes")
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data", "point_sim_3v3", "npz"))
    parser.add_argument("--episodes", type=int, default=1000, help="Target accepted episodes if --target_frames=0")
    parser.add_argument("--target_frames", type=int, default=0, help="Target accepted frames (overrides --episodes)")
    parser.add_argument("--max_attempts", type=int, default=0, help="Max attempts before stop")
    parser.add_argument("--min_passes", type=int, default=0, help="Min successful passes to keep an episode")
    parser.add_argument("--min_len", type=int, default=0, help="Min frames to keep an episode")
    parser.add_argument("--max_def_possess_s", type=float, default=0.5)
    parser.add_argument("--pass_success_window_s", type=float, default=1.2)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p_switch", type=float, default=0.02)
    parser.add_argument("--eps_action", type=float, default=0.05)
    parser.add_argument("--move_horizon", type=int, default=1)
    parser.add_argument("--save_tactics", action="store_true", help="Save template_id/switch info into npz")
    parser.add_argument("--require_shot_goal", action="store_true", help="Drop episodes with shots but no goal")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)

    env = PointSim3v3(dt=0.1, seed=args.seed)
    policy = MixedTacticPolicy(p_switch=args.p_switch, eps_action=args.eps_action, dt=0.1, v_max=env.v_max)
    def_policy = DefenderPolicy(v_max=env.v_max)

    target_frames = int(args.target_frames)
    target_episodes = int(args.episodes) if target_frames <= 0 else None
    if args.max_attempts > 0:
        max_attempts = int(args.max_attempts)
    else:
        if target_frames > 0:
            approx_eps = max(1, target_frames // max(1, args.max_steps))
            max_attempts = approx_eps * 3
        else:
            max_attempts = int(args.episodes) * 3

    max_def_possess_steps = max(1, int(np.ceil(args.max_def_possess_s / 0.1)))
    pass_success_window_steps = max(1, int(np.ceil(args.pass_success_window_s / 0.1)))

    accepted_frames = 0
    accepted_episodes = 0
    attempts = 0
    discard_counts = {}
    pass_pos_frames = 0
    template_counts = {}

    while attempts < max_attempts:
        if target_frames > 0 and accepted_frames >= target_frames:
            break
        if target_episodes is not None and accepted_episodes >= target_episodes:
            break

        ep = attempts
        episode_seed = int(master_rng.integers(0, 2**32 - 1))
        # spawn per-episode RNGs for reproducibility
        ss = np.random.SeedSequence(episode_seed)
        child_seeds = ss.spawn(2)
        seed_env = int(child_seeds[0].generate_state(1)[0])
        seed_pol = int(child_seeds[1].generate_state(1)[0])
        env.reset(seed=seed_env)
        policy.reset(np.random.default_rng(seed_pol))
        def_policy.reset(np.random.default_rng(seed_pol + 1))

        data, info = run_episode(
            env,
            policy,
            def_policy,
            args.max_steps,
            debug_tactics=args.save_tactics,
            max_def_possess_steps=max_def_possess_steps,
            pass_success_window_steps=pass_success_window_steps,
            min_passes=args.min_passes,
            min_len=args.min_len,
            require_shot_goal=args.require_shot_goal,
        )
        attempts += 1

        if data is None:
            reason = info["invalid_reason"] or "invalid"
            discard_counts[reason] = discard_counts.get(reason, 0) + 1
            continue

        out_path = os.path.join(args.out_dir, f"ep_{accepted_episodes:05d}.npz")
        np.savez_compressed(out_path, **data)
        accepted_episodes += 1
        accepted_frames += info["frames"]
        pass_pos_frames += float(np.sum(data["pass_flag"]))

        if args.save_tactics and "template_id" in data:
            tids, counts = np.unique(data["template_id"], return_counts=True)
            for tid, cnt in zip(tids, counts):
                template_counts[int(tid)] = template_counts.get(int(tid), 0) + int(cnt)

        if (accepted_episodes % 50 == 0) or (accepted_episodes == 1):
            print(
                f"Accepted {accepted_episodes} eps, frames={accepted_frames}, attempts={attempts} -> {out_path}"
            )

    print("=== Generation Summary ===")
    print(f"Attempts: {attempts}")
    print(f"Accepted Episodes: {accepted_episodes}")
    print(f"Accepted Frames: {accepted_frames}")
    if accepted_frames > 0:
        print(f"Pass-positive frame ratio: {pass_pos_frames / accepted_frames:.4f}")
    if discard_counts:
        print("Discard reasons:")
        for k, v in sorted(discard_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    if template_counts:
        total = sum(template_counts.values())
        print("Template distribution:")
        for k in sorted(template_counts.keys()):
            print(f"  {k}: {template_counts[k]} ({template_counts[k] / max(1, total):.2%})")


if __name__ == "__main__":
    main()
