import numpy as np
import pandas as pd
import pickle

from . import DATA_DIR, ARTIFACTS_DIR
from .batter_recency import RECENCY_FEATURES, precompute, load_cache

OUTCOMES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play",
    "hit_by_pitch",
]

OUTCOME_MAP = {
    "ball": "ball",
    "blocked_ball": "ball",
    "intent_ball": "ball",
    "pitchout": "ball",
    "automatic_ball": "ball",
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "foul_tip": "swinging_strike",
    "foul": "foul",
    "foul_bunt": "foul",
    "foul_pitchout": "foul",
    "hit_into_play": "in_play",
    "hit_into_play_no_out": "in_play",
    "hit_into_play_score": "in_play",
    "hit_by_pitch": "hit_by_pitch",
}

NUMERIC_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "plate_x",
    "plate_z",
    "pfx_x",
    "pfx_z",
    "release_extension",
    "balls",
    "strikes",
]

GAME_STATE_FEATURES = [
    "runner_1b",
    "runner_2b",
    "runner_3b",
    "outs",
]


def prepare_data(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    min_batter_pa=300,
    min_pitcher_pa=300,
    preprocessing_file=ARTIFACTS_DIR / "preprocessing.pkl",
):
    df = pd.read_pickle(cache_file)

    # One row per pitch, not per plate appearance
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["outcome"] = df["description"].map(OUTCOME_MAP)
    df = df.dropna(subset=["outcome", "batter", "pitcher"])

    # Chronological separation
    train = df[
        (df["game_date"] >= "2024-01-01")
        & (df["game_date"] < "2025-01-01")
    ].copy()
    validation = df[
        (df["game_date"] >= "2025-01-01")
        & (df["game_date"] < "2025-07-01")
    ].copy()
    test = df[
        (df["game_date"] >= "2025-07-01")
        & (df["game_date"] < "2026-01-01")
    ].copy()

    # Count completed plate appearances, using training only
    terminal = train[train["events"].notna()]
    batter_counts = terminal["batter"].value_counts()
    pitcher_counts = terminal["pitcher"].value_counts()

    batters = batter_counts[
        batter_counts >= min_batter_pa
    ].index

    pitchers = pitcher_counts[
        pitcher_counts >= min_pitcher_pa
    ].index

    def filter_players(frame):
        return frame[
            frame["batter"].isin(batters)
            & frame["pitcher"].isin(pitchers)
        ].copy()

    # Add game state features before filtering (uses raw columns)
    def add_game_state(frame):
        frame["runner_1b"] = frame["on_1b"].notna().astype("float32")
        frame["runner_2b"] = frame["on_2b"].notna().astype("float32")
        frame["runner_3b"] = frame["on_3b"].notna().astype("float32")
        frame["outs"] = pd.to_numeric(frame["outs_when_up"], errors="coerce").fillna(0).astype("float32")
        return frame

    train = add_game_state(train)
    validation = add_game_state(validation)
    test = add_game_state(test)

    train = filter_players(train).dropna(
        subset=NUMERIC_FEATURES + ["pitch_type"]
    )

    # Map only players with usable training pitches to embedding indices.
    batter_ids = sorted(train["batter"].unique())
    pitcher_ids = sorted(train["pitcher"].unique())
    batter_map = {pid: i for i, pid in enumerate(batter_ids)}
    pitcher_map = {pid: i for i, pid in enumerate(pitcher_ids)}

    def prepare_evaluation(frame):
        # No embedding is available for players unseen in usable training data.
        return frame[
            frame["batter"].isin(batter_map)
            & frame["pitcher"].isin(pitcher_map)
        ].dropna(subset=NUMERIC_FEATURES + ["pitch_type"])

    validation = prepare_evaluation(validation)
    test = prepare_evaluation(test)

    for name, frame in (("training", train), ("validation", validation), ("test", test)):
        if frame.empty:
            raise ValueError(
                f"No usable {name} rows. Check cached seasons, "
                "player thresholds, and required features."
            )

    # Normalize numerical features using training statistics
    mean = train[NUMERIC_FEATURES].mean()
    std = train[NUMERIC_FEATURES].std().replace(0, 1).fillna(1)

    # Learn only observed training pitch types, even for categorical input.
    train_types = pd.get_dummies(
        train["pitch_type"].astype(str), prefix="pitch", dtype=float
    )
    type_columns = train_types.columns.tolist()

    # Batter recency features (precomputed and cached)
    recency_cache = ARTIFACTS_DIR / "batter_recency_cache.pkl"
    try:
        recency_df, baselines = load_cache(recency_cache)
    except FileNotFoundError:
        recency_df, baselines = precompute(
            cache_file=cache_file,
            recency_cache=recency_cache,
            train_end="2025-01-01",
        )

    # Recency normalization from training data only
    train_recency = recency_df.loc[train.index][RECENCY_FEATURES]
    recency_mean = train_recency.mean().to_numpy("float32")
    recency_std = train_recency.std().to_numpy("float32")
    recency_std = np.where(recency_std == 0, 1.0, recency_std)

    # Game state normalization from training data
    game_state_mean = train[GAME_STATE_FEATURES].mean().to_numpy("float32")
    game_state_std = train[GAME_STATE_FEATURES].std().to_numpy("float32")
    game_state_std = np.where(game_state_std == 0, 1.0, game_state_std)

    preprocessing = {
        "batter_map": batter_map,
        "pitcher_map": pitcher_map,
        "numeric_features": NUMERIC_FEATURES,
        "mean": mean,
        "std": std,
        "type_columns": type_columns,
        "outcomes": OUTCOMES,
        "recency_features": RECENCY_FEATURES,
        "recency_mean": recency_mean,
        "recency_std": recency_std,
        "recency_baselines": baselines,
        "game_state_features": GAME_STATE_FEATURES,
        "game_state_mean": game_state_mean,
        "game_state_std": game_state_std,
    }

    with open(preprocessing_file, "wb") as f:
        pickle.dump(preprocessing, f)

    def make_dataset(frame):
        numeric = (
            (frame[NUMERIC_FEATURES] - mean) / std
        ).astype("float32")

        pitch_types = pd.get_dummies(
            frame["pitch_type"].astype(str), prefix="pitch", dtype=float
        ).reindex(columns=type_columns, fill_value=0)

        rec_raw = recency_df.loc[frame.index][RECENCY_FEATURES].to_numpy("float32")
        rec_normed = (rec_raw - recency_mean) / recency_std

        gs_raw = frame[GAME_STATE_FEATURES].to_numpy("float32")
        gs_normed = (gs_raw - game_state_mean) / game_state_std

        x = np.concatenate(
            [numeric.to_numpy(), pitch_types.to_numpy(), rec_normed, gs_normed],
            axis=1
        ).astype("float32")

        return {
            "features": x,
            "batters": frame["batter"].map(batter_map).to_numpy(dtype="int64"),
            "pitchers": frame["pitcher"].map(pitcher_map).to_numpy(dtype="int64"),
            "labels": frame["outcome"].map(
                {name: i for i, name in enumerate(OUTCOMES)}
            ).to_numpy(dtype="int64"),
            "index": frame.index.to_numpy(),
        }

    print("Training pitches:", len(train))
    print("Validation pitches:", len(validation))
    print("Testing pitches:", len(test))
    print("Batters:", len(batter_map))
    print("Pitchers:", len(pitcher_map))

    return (
        make_dataset(train),
        make_dataset(validation),
        make_dataset(test),
        len(batter_map),
        len(pitcher_map),
    )
