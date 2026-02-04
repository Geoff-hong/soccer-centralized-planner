#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from point_sim_env import PointSim3v3  # noqa: E402
from point_sim_policy import DefenderPolicy, MixedTacticPolicy  # noqa: E402
from point_sim_gen_npz_3v3 import run_episode  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Smoke test for 3v3 point sim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    env = PointSim3v3(dt=0.1, seed=args.seed)
    policy = MixedTacticPolicy(dt=0.1)
    def_policy = DefenderPolicy()

    env.reset(seed=args.seed)
    policy.reset(np.random.default_rng(args.seed + 1))
    def_policy.reset(np.random.default_rng(args.seed + 2))

    data = run_episode(env, policy, def_policy, args.max_steps)
    obs = data["obs"]
    act_move = data["act_move"]

    print(f"obs shape: {obs.shape}")
    print(f"act_move shape: {act_move.shape}")

    has_ball = obs[:, :6, 2]
    max_holders = has_ball.sum(axis=1).max()
    print(f"max has_ball count per frame (players): {max_holders}")

    xy = obs[:, :, 0:2]
    within = (xy >= -1.0) & (xy <= 1.0)
    print(f"xy within [-1,1]: {within.all()}")

    speed = np.linalg.norm(act_move, axis=2)
    print(
        "act_move speed stats: "
        f"min={speed.min():.3f}, mean={speed.mean():.3f}, max={speed.max():.3f}, "
        f">8m/s={(speed > 8.0).mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
