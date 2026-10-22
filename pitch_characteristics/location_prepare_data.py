"""Prepare at-bat sequences for the LSTM pitch location model."""

import numpy as np
import pandas as pd
import pickle

from . import DATA_DIR, ARTIFACTS_DIR

PITCH_TYPES = [
    "FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV", "KN", "OTHER",
]
PT_TO_IDX = {pt: i for i, pt in enumerate(PITCH_TYPES)}
NUM_PT = len(PITCH_TYPES)

# Simplified outcomes for the sequence (what happened to each previous pitch)
OUTCOMES = ["ball", "strike", "foul", "in_play", "hbp"]
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}
NUM_OUTCOMES = len(OUTCOMES)

OUTCOME_MAP = {
    "ball": "ball", "blocked_ball": "ball", "intent_ball": "ball",
    "pitchout": "ball", "automatic_ball": "ball",
    "called_strike": "strike", "automatic_strike": "strike",
    "swinging_strike": "strike", "swinging_strike_blocked": "strike",
    "foul_tip": "strike",
    "foul": "foul", "foul_bunt": "foul", "foul_pitchout": "foul",
    "hit_into_play": "in_play", "hit_into_play_no_out": "in_play",
    "hit_into_play_score": "in_play",
    "hit_by_pitch": "hbp",
    "missed_bunt": "strike", "bunt_foul_tip": "foul",
}

MAX_SEQ_LEN = 15  # max previous pitches to look back


def prepare_data(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    preprocessing_file=ARTIFACTS_DIR / "location_preprocessing.pkl",
    min_pitcher_pitches=200,
    min_batter_pitches=200,
):
    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    for c in ["plate_x", "plate_z", "pitch_number", "at_bat_number", "balls", "strikes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["plate_x", "plate_z", "pitch_type", "pitcher", "batter",
                            "description", "pitch_number", "at_bat_number"])

    df["pt_clean"] = df["pitch_type"].where(df["pitch_type"].isin(PITCH_TYPES[:-1]), "OTHER")
    df["outcome_clean"] = df["description"].map(OUTCOME_MAP).fillna("ball")
    df["balls"] = df["balls"].fillna(0).astype(int)
    df["strikes"] = df["strikes"].fillna(0).astype(int)
    df["batter_hand"] = (df["stand"] == "R").astype(int)
    df["pitcher_hand"] = (df["p_throws"] == "R").astype(int)

    # Sort within each at-bat
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # Chronological split
    train = df[(df["game_date"] >= "2024-01-01") & (df["game_date"] < "2025-01-01")].copy()
    val = df[(df["game_date"] >= "2025-01-01") & (df["game_date"] < "2025-07-01")].copy()
    test = df[(df["game_date"] >= "2025-07-01") & (df["game_date"] < "2026-01-01")].copy()

    # Player filtering
    pitcher_counts = train["pitcher"].value_counts()
    batter_counts = train["batter"].value_counts()
    valid_pitchers = pitcher_counts[pitcher_counts >= min_pitcher_pitches].index
    valid_batters = batter_counts[batter_counts >= min_batter_pitches].index

    def filter_players(frame):
        return frame[frame["pitcher"].isin(valid_pitchers) & frame["batter"].isin(valid_batters)].copy()

    train = filter_players(train)
    val = filter_players(val)
    test = filter_players(test)

    # Embedding maps from training
    pitcher_ids = sorted(train["pitcher"].unique())
    batter_ids = sorted(train["batter"].unique())
    pitcher_map = {pid: i for i, pid in enumerate(pitcher_ids)}
    batter_map = {bid: i for i, bid in enumerate(batter_ids)}

    val = val[val["pitcher"].isin(pitcher_map) & val["batter"].isin(batter_map)]
    test = test[test["pitcher"].isin(pitcher_map) & test["batter"].isin(batter_map)]

    # Normalize plate_x, plate_z targets using training stats
    loc_mean = train[["plate_x", "plate_z"]].mean().to_numpy("float32")
    loc_std = train[["plate_x", "plate_z"]].std().to_numpy("float32")

    # Batter zone heat map: BA in each of Statcast zones 1-9, from training data only
    # Shrink toward league average for small samples
    ZONE_SHRINKAGE = 30
    train_zone = train.copy()
    train_zone["zone"] = pd.to_numeric(train_zone.get("zone", pd.Series(dtype=float)), errors="coerce")
    train_zone["is_hit"] = train_zone["events"].isin(["single", "double", "triple", "home_run"])
    train_zone["is_ab"] = (
        train_zone["events"].notna()
        & ~train_zone["events"].isin(["walk", "hit_by_pitch", "catcher_interf",
                                       "sac_fly", "sac_bunt", "sac_fly_double_play",
                                       "sac_bunt_double_play"])
    )

    # League BA per zone
    league_zone_ba = np.zeros(9, dtype="float32")
    for z in range(1, 10):
        zdata = train_zone[train_zone["zone"] == z]
        ab = zdata["is_ab"].sum()
        hits = zdata["is_hit"].sum()
        league_zone_ba[z - 1] = hits / max(ab, 1)

    # Per-batter zone BA with shrinkage
    batter_zone_ba = {}
    for bid in batter_ids:
        bdata = train_zone[train_zone["batter"] == bid]
        ba = np.zeros(9, dtype="float32")
        for z in range(1, 10):
            zdata = bdata[bdata["zone"] == z]
            ab = zdata["is_ab"].sum()
            hits = zdata["is_hit"].sum()
            if ab > 0:
                raw_ba = hits / ab
                w = ab / (ab + ZONE_SHRINKAGE)
                ba[z - 1] = w * raw_ba + (1 - w) * league_zone_ba[z - 1]
            else:
                ba[z - 1] = league_zone_ba[z - 1]
        batter_zone_ba[bid] = ba

    preprocessing = {
        "pitcher_map": pitcher_map,
        "batter_map": batter_map,
        "pitch_types": PITCH_TYPES,
        "pt_to_idx": PT_TO_IDX,
        "outcomes": OUTCOMES,
        "outcome_to_idx": OUTCOME_TO_IDX,
        "loc_mean": loc_mean,
        "loc_std": loc_std,
        "max_seq_len": MAX_SEQ_LEN,
        "batter_zone_ba": batter_zone_ba,
        "league_zone_ba": league_zone_ba,
    }
    with open(preprocessing_file, "wb") as f:
        pickle.dump(preprocessing, f)

    def build_sequences(frame):
        """Build (sequence_history, target) pairs for each pitch."""
        # Group by at-bat
        groups = frame.groupby(["game_pk", "at_bat_number"])

        all_seq_pt = []       # (N, MAX_SEQ_LEN) pitch types in history
        all_seq_loc = []      # (N, MAX_SEQ_LEN, 2) locations in history
        all_seq_out = []      # (N, MAX_SEQ_LEN) outcomes in history
        all_seq_len = []      # (N,) actual history length
        all_pitcher = []
        all_batter = []
        all_cur_pt = []
        all_balls = []
        all_strikes = []
        all_batter_hand = []
        all_pitcher_hand = []
        all_batter_zone = []  # (N, 9) zone BA heat map
        all_target = []       # (N, 2) plate_x, plate_z

        for _, ab in groups:
            ab = ab.sort_values("pitch_number")
            pitches = ab.to_dict("records")

            for i, pitch in enumerate(pitches):
                pid = pitcher_map.get(pitch["pitcher"])
                bid = batter_map.get(pitch["batter"])
                if pid is None or bid is None:
                    continue

                # History: previous pitches in this at-bat
                history_len = min(i, MAX_SEQ_LEN)
                seq_pt = np.zeros(MAX_SEQ_LEN, dtype="int64")
                seq_loc = np.zeros((MAX_SEQ_LEN, 2), dtype="float32")
                seq_out = np.zeros(MAX_SEQ_LEN, dtype="int64")

                start = max(0, i - MAX_SEQ_LEN)
                for j, prev in enumerate(pitches[start:i]):
                    seq_pt[j] = PT_TO_IDX.get(prev["pt_clean"], PT_TO_IDX["OTHER"])
                    seq_loc[j, 0] = prev["plate_x"]
                    seq_loc[j, 1] = prev["plate_z"]
                    seq_out[j] = OUTCOME_TO_IDX.get(prev["outcome_clean"], 0)

                all_seq_pt.append(seq_pt)
                all_seq_loc.append(seq_loc)
                all_seq_out.append(seq_out)
                all_seq_len.append(history_len)
                all_pitcher.append(pid)
                all_batter.append(bid)
                all_cur_pt.append(PT_TO_IDX.get(pitch["pt_clean"], PT_TO_IDX["OTHER"]))
                all_balls.append(pitch["balls"])
                all_strikes.append(pitch["strikes"])
                all_batter_hand.append(pitch["batter_hand"])
                all_pitcher_hand.append(pitch["pitcher_hand"])
                all_batter_zone.append(
                    batter_zone_ba.get(pitch["batter"], league_zone_ba)
                )
                all_target.append([pitch["plate_x"], pitch["plate_z"]])

        return {
            "seq_pt": np.array(all_seq_pt),
            "seq_loc": np.array(all_seq_loc),
            "seq_out": np.array(all_seq_out),
            "seq_len": np.array(all_seq_len, dtype="int64"),
            "pitcher": np.array(all_pitcher, dtype="int64"),
            "batter": np.array(all_batter, dtype="int64"),
            "cur_pt": np.array(all_cur_pt, dtype="int64"),
            "balls": np.array(all_balls, dtype="float32"),
            "strikes": np.array(all_strikes, dtype="float32"),
            "batter_hand": np.array(all_batter_hand, dtype="float32"),
            "pitcher_hand": np.array(all_pitcher_hand, dtype="float32"),
            "batter_zone_ba": np.array(all_batter_zone, dtype="float32"),
            "target": np.array(all_target, dtype="float32"),
        }

    print("Building at-bat sequences...")
    print(f"  Training pitches: {len(train):,}")
    train_data = build_sequences(train)
    print(f"  Training sequences: {len(train_data['target']):,}")

    print(f"  Validation pitches: {len(val):,}")
    val_data = build_sequences(val)
    print(f"  Validation sequences: {len(val_data['target']):,}")

    print(f"  Test pitches: {len(test):,}")
    test_data = build_sequences(test)
    print(f"  Test sequences: {len(test_data['target']):,}")

    print(f"  Pitchers: {len(pitcher_map)}, Batters: {len(batter_map)}")

    return train_data, val_data, test_data, len(pitcher_map), len(batter_map), preprocessing
