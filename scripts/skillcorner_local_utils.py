import json
from pathlib import Path

import pandas as pd


def resolve_opendata_root(opendata_root=None):
    if opendata_root:
        return Path(opendata_root).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    # default: ../opendata relative to project root
    candidate = script_dir.parent.parent / "opendata"
    if candidate.exists():
        return candidate.resolve()
    alt = Path.cwd() / "opendata"
    if alt.exists():
        return alt.resolve()
    return candidate.resolve()


def load_matches_index(opendata_root):
    matches_path = Path(opendata_root) / "data" / "matches.json"
    if not matches_path.exists():
        raise FileNotFoundError(f"matches.json not found at {matches_path}")
    with matches_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_match_paths(match_id, opendata_root):
    match_id = str(match_id)
    match_dir = Path(opendata_root) / "data" / "matches" / match_id
    return {
        "match_dir": match_dir,
        "match_json": match_dir / f"{match_id}_match.json",
        "tracking_jsonl": match_dir / f"{match_id}_tracking_extrapolated.jsonl",
        "dynamic_events_csv": match_dir / f"{match_id}_dynamic_events.csv",
        "phases_csv": match_dir / f"{match_id}_phases_of_play.csv",
    }


def is_git_lfs_pointer(path):
    try:
        with Path(path).open("rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def ensure_real_file(path, hint_name="file"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{hint_name} not found: {path}")
    if is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{hint_name} is a Git LFS pointer. Run `git lfs pull` inside the opendata repo to download the real file: {path}"
        )
    return path


def load_match_json(match_id, opendata_root):
    paths = get_match_paths(match_id, opendata_root)
    match_path = ensure_real_file(paths["match_json"], "match json")
    with match_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tracking_long_df(tracking_path, limit=None):
    if limit == 0:
        limit = None
    tracking_path = ensure_real_file(tracking_path, "tracking jsonl")
    raw_data = pd.read_json(tracking_path, lines=True, nrows=limit)

    raw_df = pd.json_normalize(
        raw_data.to_dict("records"),
        "player_data",
        ["frame", "timestamp", "period", "possession", "ball_data"],
    )
    if "player_id" not in raw_df.columns or raw_df.empty:
        if limit is None:
            raise ValueError("No player_data found in tracking file.")
        raise ValueError(
            f"No player_data found in the first {limit} lines. "
            "Increase --limit or set --limit 0 to read more frames."
        )

    raw_df["possession_player_id"] = raw_df["possession"].apply(
        lambda x: x.get("player_id") if isinstance(x, dict) else None
    )
    raw_df["possession_group"] = raw_df["possession"].apply(
        lambda x: x.get("group") if isinstance(x, dict) else None
    )

    ball_df = pd.json_normalize(raw_df["ball_data"])
    raw_df["ball_x"] = ball_df.get("x")
    raw_df["ball_y"] = ball_df.get("y")
    raw_df["ball_z"] = ball_df.get("z")
    raw_df["ball_is_detected"] = ball_df.get("is_detected")

    raw_df = raw_df.drop(columns=["possession", "ball_data"])
    return raw_df


def build_tracking_wide_df(match_id, opendata_root, limit=None, sample_rate=0.1, transform_to_metric=True):
    match_meta = load_match_json(match_id, opendata_root)
    pitch_length = float(match_meta.get("pitch_length", 105.0))
    pitch_width = float(match_meta.get("pitch_width", 68.0))
    half_length = pitch_length / 2.0
    half_width = pitch_width / 2.0

    home_team_id = match_meta.get("home_team", {}).get("id")
    away_team_id = match_meta.get("away_team", {}).get("id")

    players = match_meta.get("players", [])
    id_to_team = {}
    trackable_to_player = {}
    trackable_to_team = {}
    for p in players:
        pid = p.get("id")
        team_id = p.get("team_id")
        trackable = p.get("trackable_object")
        if pid is not None and team_id is not None:
            id_to_team[int(pid)] = "Home" if team_id == home_team_id else "Away"
        if trackable is not None and team_id is not None:
            trackable_to_player[int(trackable)] = int(pid) if pid is not None else int(trackable)
            trackable_to_team[int(trackable)] = "Home" if team_id == home_team_id else "Away"

    paths = get_match_paths(match_id, opendata_root)
    long_df = load_tracking_long_df(paths["tracking_jsonl"], limit=limit)

    long_df["player_id"] = long_df["player_id"].astype(int)
    long_df["player_id_mapped"] = long_df["player_id"].map(trackable_to_player)
    long_df["player_id_mapped"] = long_df["player_id_mapped"].fillna(long_df["player_id"]).astype(int)
    long_df["team"] = long_df["player_id_mapped"].map(id_to_team)

    before = len(long_df)
    long_df = long_df[long_df["team"].isin(["Home", "Away"])]
    dropped = before - len(long_df)
    if dropped > 0:
        print(f"[Warning] Dropped {dropped} player rows with unknown team mapping.")

    long_df["col"] = long_df["team"].str.lower() + "_" + long_df["player_id_mapped"].astype(str)

    base = (
        long_df.groupby("frame")[["timestamp", "period", "ball_x", "ball_y", "ball_z"]]
        .first()
        .rename(columns={"period": "period_id"})
    )

    x_wide = long_df.pivot(index="frame", columns="col", values="x")
    y_wide = long_df.pivot(index="frame", columns="col", values="y")
    x_wide.columns = [f"{c}_x" for c in x_wide.columns]
    y_wide.columns = [f"{c}_y" for c in y_wide.columns]

    df = base.join([x_wide, y_wide], how="left").sort_index()

    if transform_to_metric:
        pos_x_cols = [c for c in df.columns if c.endswith("_x") and (c.startswith("home_") or c.startswith("away_"))]
        pos_y_cols = [c for c in df.columns if c.endswith("_y") and (c.startswith("home_") or c.startswith("away_"))]
        df[pos_x_cols] = df[pos_x_cols] + half_length
        df[pos_y_cols] = df[pos_y_cols] + half_width
        df["ball_x"] = df["ball_x"] + half_length
        df["ball_y"] = df["ball_y"] + half_width

    if sample_rate is not None:
        base_dt = 0.1  # SkillCorner open data at 10 Hz
        step = max(1, int(round(sample_rate / base_dt)))
        df = df.iloc[::step]

    return df, {"pitch_length": pitch_length, "pitch_width": pitch_width}
