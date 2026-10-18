"""Data preparation for the pitch-selection model."""

import numpy as np
import pandas as pd
import pickle

from . import DATA_DIR, ARTIFACTS_DIR

# Pitch types with enough training volume to model; rest grouped into OTHER.
PITCH_TYPES = [
    "FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV", "KN",
]
OTHER_LABEL = "OTHER"
ALL_PITCH_LABELS = PITCH_TYPES + [OTHER_LABEL]
NUM_CLASSES = len(ALL_PITCH_LABELS)
PITCH_TO_IDX = {pt: i for i, pt in enumerate(ALL_PITCH_LABELS)}


def prepare_data(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    preprocessing_file=ARTIFACTS_DIR / "preprocessing.pkl",
    min_pitcher_pitches=200,
    min_batter_pitches=200,
):
    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["pitch_type", "pitcher", "batter", "balls", "strikes"])

    # Map rare pitch types to OTHER
    df["pitch_label"] = df["pitch_type"].where(
        df["pitch_type"].isin(PITCH_TYPES), OTHER_LABEL
    )

    # Chronological split (same boundaries as pitch_outcome_prediction)
    train = df[(df["game_date"] >= "2024-01-01") & (df["game_date"] < "2025-01-01")].copy()
    val = df[(df["game_date"] >= "2025-01-01") & (df["game_date"] < "2025-07-01")].copy()
    test = df[(df["game_date"] >= "2025-07-01") & (df["game_date"] < "2026-01-01")].copy()

    # Player thresholds based on training data
    pitcher_counts = train["pitcher"].value_counts()
    batter_counts = train["batter"].value_counts()
    valid_pitchers = pitcher_counts[pitcher_counts >= min_pitcher_pitches].index
    valid_batters = batter_counts[batter_counts >= min_batter_pitches].index

    def filter_and_prepare(frame):
        f = frame[
            frame["pitcher"].isin(valid_pitchers) & frame["batter"].isin(valid_batters)
        ].copy()
        return f

    train = filter_and_prepare(train)
    val = filter_and_prepare(val)
    test = filter_and_prepare(test)

    # Build embedding maps from training data
    pitcher_ids = sorted(train["pitcher"].unique())
    batter_ids = sorted(train["batter"].unique())
    pitcher_map = {pid: i for i, pid in enumerate(pitcher_ids)}
    batter_map = {bid: i for i, bid in enumerate(batter_ids)}

    # Filter val/test to known players
    val = val[val["pitcher"].isin(pitcher_map) & val["batter"].isin(batter_map)]
    test = test[test["pitcher"].isin(pitcher_map) & test["batter"].isin(batter_map)]

    # Pitcher repertoire mask: which pitch types each pitcher actually throws
    repertoire = np.zeros((len(pitcher_map), NUM_CLASSES), dtype="float32")
    for pid, idx in pitcher_map.items():
        pitcher_pitches = train[train["pitcher"] == pid]["pitch_label"].value_counts()
        for pt, count in pitcher_pitches.items():
            if count >= 5:  # minimum threshold to consider part of repertoire
                repertoire[idx, PITCH_TO_IDX[pt]] = 1.0
    # Ensure every pitcher has at least one pitch type
    for i in range(len(pitcher_map)):
        if repertoire[i].sum() == 0:
            repertoire[i, PITCH_TO_IDX["FF"]] = 1.0

    # Compute previous pitch for each row
    def add_features(frame):
        frame = frame.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()

        # Convert types
        frame["balls"] = pd.to_numeric(frame["balls"], errors="coerce").fillna(0).astype(int)
        frame["strikes"] = pd.to_numeric(frame["strikes"], errors="coerce").fillna(0).astype(int)
        frame["at_bat_number"] = pd.to_numeric(frame["at_bat_number"], errors="coerce").fillna(1).astype(int)

        # Previous pitch within the same game for the same pitcher
        frame["prev_pitch"] = frame.groupby(["pitcher", "game_pk"])["pitch_label"].shift(1).fillna("NONE")

        # Cumulative pitch count in game for the pitcher (fatigue proxy)
        frame["game_pitch_count"] = frame.groupby(["pitcher", "game_pk"]).cumcount() + 1

        # Handedness encoding
        frame["batter_hand"] = (frame["stand"] == "R").astype(int)
        frame["pitcher_hand"] = (frame["p_throws"] == "R").astype(int)

        # Runners on base
        frame["runner_1b"] = frame["on_1b"].notna().astype(int)
        frame["runner_2b"] = frame["on_2b"].notna().astype(int)
        frame["runner_3b"] = frame["on_3b"].notna().astype(int)

        # Outs
        frame["outs"] = pd.to_numeric(frame["outs_when_up"], errors="coerce").fillna(0).astype(int)

        # Score differential (clamped to [-5, 5] to limit outlier influence)
        bat = pd.to_numeric(frame["bat_score"], errors="coerce").fillna(0)
        fld = pd.to_numeric(frame["fld_score"], errors="coerce").fillna(0)
        frame["score_diff"] = (fld - bat).clip(-5, 5)  # pitcher's perspective: positive = ahead

        return frame

    train = add_features(train)
    val = add_features(val)
    test = add_features(test)

    # Previous-pitch encoding: map to index (NONE=0, then pitch types)
    PREV_LABELS = ["NONE"] + ALL_PITCH_LABELS
    prev_to_idx = {p: i for i, p in enumerate(PREV_LABELS)}

    # Pitcher-specific historical pitch frequencies (baseline)
    pitcher_freq = np.zeros((len(pitcher_map), NUM_CLASSES), dtype="float32")
    for pid, idx in pitcher_map.items():
        counts = train[train["pitcher"] == pid]["pitch_label"].value_counts()
        for pt, c in counts.items():
            pitcher_freq[idx, PITCH_TO_IDX[pt]] = c
    # Normalize to probabilities
    row_sums = pitcher_freq.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pitcher_freq = pitcher_freq / row_sums

    preprocessing = {
        "pitcher_map": pitcher_map,
        "batter_map": batter_map,
        "pitch_types": ALL_PITCH_LABELS,
        "pitch_to_idx": PITCH_TO_IDX,
        "prev_labels": PREV_LABELS,
        "prev_to_idx": prev_to_idx,
        "repertoire": repertoire,
        "pitcher_freq": pitcher_freq,
    }
    with open(preprocessing_file, "wb") as f:
        pickle.dump(preprocessing, f)

    def make_dataset(frame):
        return {
            "pitchers": frame["pitcher"].map(pitcher_map).to_numpy("int64"),
            "batters": frame["batter"].map(batter_map).to_numpy("int64"),
            "balls": frame["balls"].to_numpy("int64"),
            "strikes": frame["strikes"].to_numpy("int64"),
            "prev_pitch": frame["prev_pitch"].map(prev_to_idx).fillna(0).to_numpy("int64"),
            "game_pitch_count": frame["game_pitch_count"].clip(1, 120).to_numpy("float32") / 120.0,
            "batter_hand": frame["batter_hand"].to_numpy("float32"),
            "pitcher_hand": frame["pitcher_hand"].to_numpy("float32"),
            "runner_1b": frame["runner_1b"].to_numpy("float32"),
            "runner_2b": frame["runner_2b"].to_numpy("float32"),
            "runner_3b": frame["runner_3b"].to_numpy("float32"),
            "outs": frame["outs"].to_numpy("float32"),
            "score_diff": frame["score_diff"].to_numpy("float32") / 5.0,  # normalize to ~[-1, 1]
            "labels": frame["pitch_label"].map(PITCH_TO_IDX).to_numpy("int64"),
        }

    print(f"Pitch selection data: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    print(f"Pitchers: {len(pitcher_map)}, Batters: {len(batter_map)}")
    print(f"Pitch classes: {NUM_CLASSES}")

    return (
        make_dataset(train),
        make_dataset(val),
        make_dataset(test),
        len(pitcher_map),
        len(batter_map),
        preprocessing,
    )
