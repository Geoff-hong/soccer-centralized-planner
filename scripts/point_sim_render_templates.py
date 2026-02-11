#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from point_sim_render_gif import _load_episode, _render  # noqa: E402

TEMPLATE_MAP = {
    "TRIANGLE": 0,
    "ONE_TWO": 1,
    "THROUGH": 2,
    "CUTBACK": 3,
}


def _collect_files(data_dir: str) -> List[str]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    files.sort()
    return [os.path.join(data_dir, f) for f in files]


def _select_by_template(files: List[str], template_id: int, num: int, rng: np.random.Generator) -> List[str]:
    if not files:
        return []
    files = list(files)
    rng.shuffle(files)
    selected = []
    for path in files:
        data = np.load(path)
        if "template_id" not in data:
            raise RuntimeError(
                "template_id not found in npz. Re-generate with --save_tactics in point_sim_gen_npz_3v3.py."
            )
        tid = np.asarray(data["template_id"])
        if tid.size > 0 and np.any(tid == template_id):
            selected.append(path)
        if len(selected) >= num:
            break
    return selected


def _concat_episodes(paths: List[str], gap_frames: int):
    obs_all = []
    pass_all = []
    passer_all = []
    receiver_all = []
    template_all = []
    episode_ids = []
    dt_ref = None

    for ep_idx, path in enumerate(paths):
        obs, dt, pass_flag, passer_id, receiver_id, template_id = _load_episode(path)
        if dt_ref is None:
            dt_ref = dt
        elif abs(dt - dt_ref) > 1e-6:
            print(f"[Warn] dt mismatch in {os.path.basename(path)}: {dt} vs {dt_ref}")

        obs_all.append(obs)
        pass_all.append(np.asarray(pass_flag) if pass_flag is not None else np.zeros(obs.shape[0], dtype=np.float32))
        passer_all.append(np.asarray(passer_id) if passer_id is not None else np.zeros(obs.shape[0], dtype=np.int64))
        receiver_all.append(
            np.asarray(receiver_id) if receiver_id is not None else np.zeros(obs.shape[0], dtype=np.int64)
        )
        template_all.append(
            np.asarray(template_id) if template_id is not None else np.full(obs.shape[0], -1, dtype=np.int64)
        )
        episode_ids.append(np.full(obs.shape[0], ep_idx, dtype=np.int64))

        if gap_frames > 0 and ep_idx < len(paths) - 1:
            gap = np.zeros((gap_frames, obs.shape[1], obs.shape[2]), dtype=np.float32)
            obs_all.append(gap)
            pass_all.append(np.zeros(gap_frames, dtype=np.float32))
            passer_all.append(np.zeros(gap_frames, dtype=np.int64))
            receiver_all.append(np.zeros(gap_frames, dtype=np.int64))
            template_all.append(np.full(gap_frames, -1, dtype=np.int64))
            episode_ids.append(np.full(gap_frames, ep_idx, dtype=np.int64))

    obs_cat = np.concatenate(obs_all, axis=0)
    pass_cat = np.concatenate(pass_all, axis=0)
    passer_cat = np.concatenate(passer_all, axis=0)
    receiver_cat = np.concatenate(receiver_all, axis=0)
    template_cat = np.concatenate(template_all, axis=0)
    episode_cat = np.concatenate(episode_ids, axis=0)
    return obs_cat, dt_ref or 0.1, pass_cat, passer_cat, receiver_cat, template_cat, episode_cat


def main():
    parser = argparse.ArgumentParser(description="Render episodes containing specific templates into one GIF/MP4.")
    parser.add_argument("--data_dir", required=True, help="Directory with npz episodes (must include template_id).")
    parser.add_argument("--out", default=os.path.join("output", "point_sim_templates.gif"))
    parser.add_argument("--num_each", type=int, default=5, help="Episodes per template.")
    parser.add_argument("--gap_frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--pass_hold", type=float, default=0.5)
    parser.add_argument(
        "--templates",
        type=str,
        default="ONE_TWO,CUTBACK",
        help="Comma-separated template names. Options: TRIANGLE,ONE_TWO,THROUGH,CUTBACK",
    )
    args = parser.parse_args()

    templates = [t.strip().upper() for t in args.templates.split(",") if t.strip()]
    for t in templates:
        if t not in TEMPLATE_MAP:
            raise ValueError(f"Unknown template: {t}")

    files = _collect_files(args.data_dir)
    rng = np.random.default_rng(args.seed)

    selected_paths = []
    for t in templates:
        tid = TEMPLATE_MAP[t]
        picks = _select_by_template(files, tid, args.num_each, rng)
        if len(picks) < args.num_each:
            print(f"[Warn] Only found {len(picks)}/{args.num_each} episodes for {t}")
        selected_paths.extend(picks)

    if not selected_paths:
        raise RuntimeError("No episodes matched the requested templates.")

    obs, dt, pass_flag, passer_id, receiver_id, template_id, episode_ids = _concat_episodes(
        selected_paths, args.gap_frames
    )

    _render(
        obs,
        dt,
        args.out,
        every=args.every,
        dpi=args.dpi,
        pass_flag=pass_flag,
        passer_id=passer_id,
        receiver_id=receiver_id,
        pass_hold=args.pass_hold,
        template_id=template_id,
        episode_ids=episode_ids,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
