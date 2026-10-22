"""Prepare at-bat sequences for the joint pitch generation model."""

import numpy as np
import pandas as pd
import pickle

from . import DATA_DIR, ARTIFACTS_DIR

PITCH_TYPES = [
    "FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV", "KN", "OTHER",
]
PT_TO_IDX = {pt: i for i, pt in enumerate(PITCH_TYPES)}
NUM_PT = len(PITCH_TYPES)

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

MAX_SEQ_LEN = 15
ZONE_SHRINKAGE = 30


def prepare_data(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    preprocessing_file=ARTIFACTS_DIR / "preprocessing.pkl",
    min_pitcher_pitches=200,
    min_batter_pitches=200,
):
    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    for c in ["plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed",
              "pitch_number", "at_bat_number",
              "balls", "strikes", "bat_score", "fld_score", "outs_when_up"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed",
                            "pitch_type", "pitcher", "batter",
                            "description", "pitch_number", "at_bat_number"])

    df["pt_clean"] = df["pitch_type"].where(df["pitch_type"].isin(PITCH_TYPES[:-1]), "OTHER")
    df["outcome_clean"] = df["description"].map(OUTCOME_MAP).fillna("ball")
    df["balls"] = df["balls"].fillna(0).astype(int)
    df["strikes"] = df["strikes"].fillna(0).astype(int)
    df["batter_hand"] = (df["stand"] == "R").astype(int)
    df["pitcher_hand"] = (df["p_throws"] == "R").astype(int)
    df["runner_1b"] = df["on_1b"].notna().astype(int)
    df["runner_2b"] = df["on_2b"].notna().astype(int)
    df["runner_3b"] = df["on_3b"].notna().astype(int)
    df["outs"] = df["outs_when_up"].fillna(0).astype(int)
    bat = df["bat_score"].fillna(0)
    fld = df["fld_score"].fillna(0)
    df["score_diff"] = (fld - bat).clip(-5, 5) / 5.0  # pitcher perspective, normalized

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

    pitcher_ids = sorted(train["pitcher"].unique())
    batter_ids = sorted(train["batter"].unique())
    pitcher_map = {pid: i for i, pid in enumerate(pitcher_ids)}
    batter_map = {bid: i for i, bid in enumerate(batter_ids)}

    val = val[val["pitcher"].isin(pitcher_map) & val["batter"].isin(batter_map)]
    test = test[test["pitcher"].isin(pitcher_map) & test["batter"].isin(batter_map)]

    # Pitcher repertoire mask
    repertoire = np.zeros((len(pitcher_map), NUM_PT), dtype="float32")
    for pid, idx in pitcher_map.items():
        counts = train[train["pitcher"] == pid]["pt_clean"].value_counts()
        for pt, c in counts.items():
            if c >= 5:
                repertoire[idx, PT_TO_IDX[pt]] = 1.0
    for i in range(len(pitcher_map)):
        if repertoire[i].sum() == 0:
            repertoire[i, PT_TO_IDX["FF"]] = 1.0

    # Batter zone heat map (BA in zones 1-9, from training only)
    train_zone = train.copy()
    train_zone["zone"] = pd.to_numeric(train_zone.get("zone", pd.Series(dtype=float)), errors="coerce")
    train_zone["is_hit"] = train_zone["events"].isin(["single", "double", "triple", "home_run"])
    train_zone["is_ab"] = (
        train_zone["events"].notna()
        & ~train_zone["events"].isin(["walk", "hit_by_pitch", "catcher_interf",
                                       "sac_fly", "sac_bunt", "sac_fly_double_play",
                                       "sac_bunt_double_play"])
    )

    league_zone_ba = np.zeros(9, dtype="float32")
    for z in range(1, 10):
        zd = train_zone[train_zone["zone"] == z]
        ab = zd["is_ab"].sum()
        hits = zd["is_hit"].sum()
        league_zone_ba[z - 1] = hits / max(ab, 1)

    batter_zone_ba = {}
    for bid in batter_ids:
        bd = train_zone[train_zone["batter"] == bid]
        ba = np.zeros(9, dtype="float32")
        for z in range(1, 10):
            zd = bd[bd["zone"] == z]
            ab = zd["is_ab"].sum()
            hits = zd["is_hit"].sum()
            if ab > 0:
                w = ab / (ab + ZONE_SHRINKAGE)
                ba[z - 1] = w * (hits / ab) + (1 - w) * league_zone_ba[z - 1]
            else:
                ba[z - 1] = league_zone_ba[z - 1]
        batter_zone_ba[bid] = ba

    # Pitcher baseline frequencies (for evaluation comparison)
    pitcher_freq = np.zeros((len(pitcher_map), NUM_PT), dtype="float32")
    for pid, idx in pitcher_map.items():
        counts = train[train["pitcher"] == pid]["pt_clean"].value_counts()
        for pt, c in counts.items():
            pitcher_freq[idx, PT_TO_IDX[pt]] = c
    row_sums = pitcher_freq.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pitcher_freq = pitcher_freq / row_sums

    preprocessing = {
        "pitcher_map": pitcher_map,
        "batter_map": batter_map,
        "pitch_types": PITCH_TYPES,
        "pt_to_idx": PT_TO_IDX,
        "outcomes": OUTCOMES,
        "outcome_to_idx": OUTCOME_TO_IDX,
        "max_seq_len": MAX_SEQ_LEN,
        "batter_zone_ba": batter_zone_ba,
        "league_zone_ba": league_zone_ba,
        "repertoire": repertoire,
        "pitcher_freq": pitcher_freq,
    }
    with open(preprocessing_file, "wb") as f:
        pickle.dump(preprocessing, f)

    def build_sequences(frame):
        groups = frame.groupby(["game_pk", "at_bat_number"])

        all_seq_pt = []
        all_seq_loc = []
        all_seq_out = []
        all_seq_len = []
        all_pitcher = []
        all_batter = []
        all_cur_pt = []
        all_balls = []
        all_strikes = []
        all_batter_hand = []
        all_pitcher_hand = []
        all_batter_zone = []
        all_pitch_num = []
        all_runner_1b = []
        all_runner_2b = []
        all_runner_3b = []
        all_outs = []
        all_score_diff = []
        all_game_pitch_count = []
        all_target_chars = []  # (N, 5) plate_x, plate_z, pfx_x, pfx_z, velocity
        all_target_pt = []

        # Precompute game-level pitch counts per pitcher
        pitcher_game_counts = frame.groupby(["pitcher", "game_pk"]).cumcount() + 1

        for (game_pk, ab_num), ab in groups:
            ab = ab.sort_values("pitch_number")
            pitches = ab.to_dict("records")
            ab_indices = ab.index.tolist()

            for i, pitch in enumerate(pitches):
                pid = pitcher_map.get(pitch["pitcher"])
                bid = batter_map.get(pitch["batter"])
                if pid is None or bid is None:
                    continue

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
                all_balls.append(float(pitch["balls"]))
                all_strikes.append(float(pitch["strikes"]))
                all_batter_hand.append(float(pitch["batter_hand"]))
                all_pitcher_hand.append(float(pitch["pitcher_hand"]))
                all_batter_zone.append(batter_zone_ba.get(pitch["batter"], league_zone_ba))
                all_pitch_num.append(min(i + 1, MAX_SEQ_LEN) / MAX_SEQ_LEN)  # normalized
                all_runner_1b.append(float(pitch["runner_1b"]))
                all_runner_2b.append(float(pitch["runner_2b"]))
                all_runner_3b.append(float(pitch["runner_3b"]))
                all_outs.append(float(pitch["outs"]))
                all_score_diff.append(float(pitch["score_diff"]))
                gpc = int(pitcher_game_counts.get(ab_indices[i], 40))
                all_game_pitch_count.append(min(gpc, 120) / 120.0)
                all_target_chars.append([
                    pitch["plate_x"], pitch["plate_z"],
                    pitch["pfx_x"], pitch["pfx_z"],
                    pitch["release_speed"],
                ])
                all_target_pt.append(PT_TO_IDX.get(pitch["pt_clean"], PT_TO_IDX["OTHER"]))

        return {
            "seq_pt": np.array(all_seq_pt),
            "seq_loc": np.array(all_seq_loc, dtype="float32"),
            "seq_out": np.array(all_seq_out),
            "seq_len": np.array(all_seq_len, dtype="int64"),
            "pitcher": np.array(all_pitcher, dtype="int64"),
            "batter": np.array(all_batter, dtype="int64"),
            "balls": np.array(all_balls, dtype="float32"),
            "strikes": np.array(all_strikes, dtype="float32"),
            "batter_hand": np.array(all_batter_hand, dtype="float32"),
            "pitcher_hand": np.array(all_pitcher_hand, dtype="float32"),
            "batter_zone_ba": np.array(all_batter_zone, dtype="float32"),
            "pitch_num": np.array(all_pitch_num, dtype="float32"),
            "runner_1b": np.array(all_runner_1b, dtype="float32"),
            "runner_2b": np.array(all_runner_2b, dtype="float32"),
            "runner_3b": np.array(all_runner_3b, dtype="float32"),
            "outs": np.array(all_outs, dtype="float32"),
            "score_diff": np.array(all_score_diff, dtype="float32"),
            "game_pitch_count": np.array(all_game_pitch_count, dtype="float32"),
            "target_chars": np.array(all_target_chars, dtype="float32"),
            "target_pt": np.array(all_target_pt, dtype="int64"),
        }

    print("Building at-bat sequences...")
    print(f"  Training pitches: {len(train):,}")
    train_data = build_sequences(train)
    print(f"  Training sequences: {len(train_data['target_pt']):,}")

    print(f"  Validation pitches: {len(val):,}")
    val_data = build_sequences(val)
    print(f"  Validation sequences: {len(val_data['target_pt']):,}")

    print(f"  Test pitches: {len(test):,}")
    test_data = build_sequences(test)
    print(f"  Test sequences: {len(test_data['target_pt']):,}")

    print(f"  Pitchers: {len(pitcher_map)}, Batters: {len(batter_map)}")

    return train_data, val_data, test_data, len(pitcher_map), len(batter_map), preprocessing
