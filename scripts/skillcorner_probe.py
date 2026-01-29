import argparse
import json
import os
from datetime import datetime

import pandas as pd

from skillcorner_local_utils import (
    build_tracking_wide_df,
    load_matches_index,
    resolve_opendata_root,
)


def _extract_player_ids(cols, prefix):
    ids = []
    for c in cols:
        if not c.startswith(prefix + "_") or not c.endswith("_x"):
            continue
        ids.append(c[len(prefix) + 1 : -2])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Probe local SkillCorner open data format.")
    parser.add_argument("--match_id", type=str, default="")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--sample_rate", type=float, default=1 / 10)
    parser.add_argument("--opendata_root", type=str, default="")
    parser.add_argument("--no_transform", action="store_true", help="Do not transform to metric pitch (0-105,0-68).")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    opendata_root = resolve_opendata_root(args.opendata_root)
    matches_index = load_matches_index(opendata_root)
    if not args.match_id:
        print("Available match_ids:")
        for m in matches_index:
            print(f"  - {m['id']}: {m['home_team']['short_name']} vs {m['away_team']['short_name']} ({m['date_time']})")
        return

    df, pitch = build_tracking_wide_df(
        match_id=args.match_id,
        opendata_root=opendata_root,
        limit=args.limit,
        sample_rate=args.sample_rate,
        transform_to_metric=(not args.no_transform),
    )

    print("Loaded SkillCorner dataset")
    print(f"Shape: {df.shape}")
    print("Head:")
    print(df.head())

    home_x_cols = [c for c in df.columns if c.startswith("home_") and c.endswith("_x")]
    away_x_cols = [c for c in df.columns if c.startswith("away_") and c.endswith("_x")]
    home_ids = _extract_player_ids(home_x_cols, "home")
    away_ids = _extract_player_ids(away_x_cols, "away")

    summary = {
        "match_id": args.match_id,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "home_player_ids": home_ids,
        "away_player_ids": away_ids,
        "ball_x_range": [float(df["ball_x"].min()) if "ball_x" in df else None,
                         float(df["ball_x"].max()) if "ball_x" in df else None],
        "ball_y_range": [float(df["ball_y"].min()) if "ball_y" in df else None,
                         float(df["ball_y"].max()) if "ball_y" in df else None],
        "sample_rate": args.sample_rate,
        "limit": args.limit,
        "transformed_to_metric": not args.no_transform,
        "pitch_length": pitch.get("pitch_length"),
        "pitch_width": pitch.get("pitch_width"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
