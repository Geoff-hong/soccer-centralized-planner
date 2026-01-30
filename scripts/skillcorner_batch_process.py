import argparse
import os
import subprocess
import sys

from skillcorner_local_utils import resolve_opendata_root


def discover_match_ids(opendata_root, max_matches=None):
    matches_dir = os.path.join(opendata_root, "data", "matches")
    if not os.path.isdir(matches_dir):
        raise FileNotFoundError(f"matches dir not found: {matches_dir}")
    ids = []
    for name in os.listdir(matches_dir):
        if name.isdigit():
            ids.append(name)
    ids.sort()
    if max_matches is not None:
        ids = ids[:max_matches]
    return ids


def run(cmd, cwd):
    cmd_str = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd_str)}")
    subprocess.run(cmd_str, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch process SkillCorner open data (dynamic events).")
    parser.add_argument("--opendata_root", type=str, default="")
    parser.add_argument("--sample_rate", type=float, default=1 / 10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_matches", type=int, default=10)
    parser.add_argument("--match_ids", type=str, nargs="*", default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=2000)
    args = parser.parse_args()

    opendata_root = resolve_opendata_root(args.opendata_root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "data", "skillcorner_processed")
    events_dir = os.path.join(data_root, "events")
    npz_dir = os.path.join(data_root, "npz")
    gifs_dir = os.path.join(data_root, "gifs")
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(gifs_dir, exist_ok=True)

    if args.match_ids:
        match_ids = args.match_ids
    else:
        match_ids = discover_match_ids(opendata_root, args.max_matches)

    py = sys.executable

    for mid in match_ids:
        print(f"\n=== SkillCorner match {mid} ===")
        dynamic_events = os.path.join(events_dir, f"skillcorner_match{mid}_dynamic_events.csv")
        npz_path = os.path.join(npz_dir, f"skillcorner_match{mid}_dynamic_train_attacker_centric.npz")
        gif_path = os.path.join(gifs_dir, f"skillcorner_match{mid}_frames_{args.start_frame}_{args.start_frame + args.num_frames}.gif")

        run(
            [
                py,
                "scripts/skillcorner_track2event.py",
                "--match_id",
                str(mid),
                "--source",
                "dynamic",
                "--output",
                dynamic_events,
            ],
            cwd=project_root,
        )

        run(
            [
                py,
                "scripts/skillcorner_data_processing.py",
                "--match_id",
                str(mid),
                "--sample_rate",
                str(args.sample_rate),
                "--event_file",
                dynamic_events,
                "--output",
                npz_path,
                "--opendata_root",
                opendata_root,
            ]
            + (["--limit", str(args.limit)] if args.limit is not None else []),
            cwd=project_root,
        )

        run(
            [
                py,
                "scripts/check_data.py",
                "--file",
                npz_path,
                "--start_frame",
                str(args.start_frame),
                "--num_frames",
                str(args.num_frames),
                "--output",
                gif_path,
            ],
            cwd=project_root,
        )

    print("\nAll matches processed.")
    print(f"Events: {events_dir}")
    print(f"NPZ:    {npz_dir}")
    print(f"GIFs:   {gifs_dir}")


if __name__ == "__main__":
    main()
