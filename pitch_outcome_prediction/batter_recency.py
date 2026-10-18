"""Batter recency features from rolling windows of completed plate appearances."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from . import DATA_DIR, ARTIFACTS_DIR

RECENCY_FEATURES = [
    "batter_swinging_strike_rate",
    "batter_chase_rate",
    "batter_contact_rate",
    "batter_hard_hit_rate",
    "batter_avg_exit_velo",
    "batter_barrel_rate",
]

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "foul", "foul_bunt", "foul_pitchout",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt", "bunt_foul_tip",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}

CONTACT_DESCRIPTIONS = SWING_DESCRIPTIONS - WHIFF_DESCRIPTIONS

BATTED_BALL_DESCRIPTIONS = {
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

WINDOW = 50

_AGG_COLS = [
    "n_pitches", "n_whiffs", "n_outside", "n_chases",
    "n_swings", "n_contacts", "n_has_ev", "sum_ev",
    "n_hard", "n_has_lsa", "n_barrels",
]


def _annotate(df):
    """Add pitch-level flags for recency calculations."""
    desc = df["description"]
    df["_swing"] = desc.isin(SWING_DESCRIPTIONS)
    df["_whiff"] = desc.isin(WHIFF_DESCRIPTIONS)
    df["_contact"] = desc.isin(CONTACT_DESCRIPTIONS)
    df["_batted"] = desc.isin(BATTED_BALL_DESCRIPTIONS)

    zone = pd.to_numeric(df["zone"], errors="coerce")
    df["_outside"] = zone.isin([11, 12, 13, 14])
    df["_chase"] = df["_swing"] & df["_outside"]

    ev = pd.to_numeric(df["launch_speed"], errors="coerce")
    df["_has_ev"] = df["_batted"] & ev.notna()
    df["_ev"] = ev.where(df["_has_ev"], 0.0)
    df["_hard"] = df["_has_ev"] & (ev >= 95.0)

    lsa = pd.to_numeric(df["launch_speed_angle"], errors="coerce")
    df["_has_lsa"] = df["_batted"] & lsa.notna()
    df["_barrel"] = (lsa == 6) & df["_has_lsa"]


def _pa_aggregates(df):
    """Aggregate pitch-level flags to completed-PA level."""
    event_keys = (
        df.loc[df["events"].notna(), ["batter", "game_pk", "at_bat_number"]]
        .drop_duplicates()
    )
    completed = df.merge(event_keys, on=["batter", "game_pk", "at_bat_number"])

    agg = completed.groupby(
        ["batter", "game_pk", "at_bat_number"], sort=False
    ).agg(
        game_date=("game_date", "first"),
        n_pitches=("_swing", "size"),
        n_whiffs=("_whiff", "sum"),
        n_outside=("_outside", "sum"),
        n_chases=("_chase", "sum"),
        n_swings=("_swing", "sum"),
        n_contacts=("_contact", "sum"),
        n_has_ev=("_has_ev", "sum"),
        sum_ev=("_ev", "sum"),
        n_hard=("_hard", "sum"),
        n_has_lsa=("_has_lsa", "sum"),
        n_barrels=("_barrel", "sum"),
    ).reset_index()

    agg["game_pk"] = pd.to_numeric(agg["game_pk"])
    agg["at_bat_number"] = pd.to_numeric(agg["at_bat_number"])
    agg = agg.sort_values(
        ["batter", "game_date", "game_pk", "at_bat_number"]
    ).reset_index(drop=True)
    return agg


def compute_baselines(pa_agg):
    """League-average rates from PA aggregates."""
    s = pa_agg[_AGG_COLS].sum()
    return {
        "batter_swinging_strike_rate": s["n_whiffs"] / max(s["n_pitches"], 1),
        "batter_chase_rate": s["n_chases"] / max(s["n_outside"], 1),
        "batter_contact_rate": s["n_contacts"] / max(s["n_swings"], 1),
        "batter_hard_hit_rate": s["n_hard"] / max(s["n_has_ev"], 1),
        "batter_avg_exit_velo": s["sum_ev"] / max(s["n_has_ev"], 1),
        "batter_barrel_rate": s["n_barrels"] / max(s["n_has_lsa"], 1),
    }


def _rolling_features(pa_agg, baselines, window=WINDOW):
    """Compute per-PA rolling recency from PA aggregates."""
    g = pa_agg.groupby("batter")

    def _srs(col):
        return g[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )

    n_prior = g["n_pitches"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=0).count()
    )

    rs = {c: _srs(c) for c in _AGG_COLS}

    def safe_div(num, den):
        return np.where(den > 0, num / den, np.nan)

    raw = {
        "batter_swinging_strike_rate": safe_div(rs["n_whiffs"], rs["n_pitches"]),
        "batter_chase_rate": safe_div(rs["n_chases"], rs["n_outside"]),
        "batter_contact_rate": safe_div(rs["n_contacts"], rs["n_swings"]),
        "batter_hard_hit_rate": safe_div(rs["n_hard"], rs["n_has_ev"]),
        "batter_avg_exit_velo": safe_div(rs["sum_ev"], rs["n_has_ev"]),
        "batter_barrel_rate": safe_div(rs["n_barrels"], rs["n_has_lsa"]),
    }

    weight = (n_prior / window).clip(0, 1).to_numpy()
    result = pa_agg[["batter", "game_pk", "at_bat_number"]].copy()

    for feat in RECENCY_FEATURES:
        obs = np.where(np.isnan(raw[feat]), baselines[feat], raw[feat])
        result[feat] = weight * obs + (1 - weight) * baselines[feat]

    return result


def precompute(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    recency_cache=ARTIFACTS_DIR / "batter_recency_cache.pkl",
    train_end="2025-01-01",
):
    """Precompute recency features for every regular-season pitch.

    Returns (recency_df, baselines) where recency_df is a DataFrame indexed
    by the raw Statcast DataFrame index with one column per recency feature.
    """
    print("Computing batter recency features...")
    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    _annotate(df)

    pa_agg = _pa_aggregates(df)

    # Baselines from training PAs only
    train_pas = pa_agg[pa_agg["game_date"] < train_end]
    baselines = compute_baselines(train_pas)
    print("League baselines (training):")
    for k, v in baselines.items():
        print(f"  {k}: {v:.4f}")

    pa_recency = _rolling_features(pa_agg, baselines)

    # Map PA-level recency back to pitch level
    pa_lookup = pa_recency.set_index(
        ["batter", "game_pk", "at_bat_number"]
    )[RECENCY_FEATURES]

    keys = pd.MultiIndex.from_arrays(
        [
            df["batter"].values,
            pd.to_numeric(df["game_pk"]).values,
            pd.to_numeric(df["at_bat_number"]).values,
        ],
        names=["batter", "game_pk", "at_bat_number"],
    )

    pitch_recency = pa_lookup.reindex(keys)
    pitch_recency.index = df.index

    for feat in RECENCY_FEATURES:
        pitch_recency[feat] = pitch_recency[feat].fillna(baselines[feat])

    pitch_recency = pitch_recency.astype("float32")

    with open(recency_cache, "wb") as f:
        pickle.dump({"recency": pitch_recency, "baselines": baselines}, f)
    print(f"Cached {len(pitch_recency)} rows to {recency_cache}")
    return pitch_recency, baselines


def load_cache(recency_cache=ARTIFACTS_DIR / "batter_recency_cache.pkl"):
    """Load previously cached recency features."""
    with open(recency_cache, "rb") as f:
        data = pickle.load(f)
    return data["recency"], data["baselines"]


def get_recency_for_batter(
    batter_id, before_date, baselines, raw_df=None, window=WINDOW
):
    """Compute recency features for a single batter at inference time.

    Uses completed PAs strictly before ``before_date``.
    Returns a dict of {feature_name: value}.
    """
    if raw_df is None:
        raw_df = pd.read_pickle(DATA_DIR / "statcast_raw.pkl")

    bp = raw_df[
        (raw_df["batter"] == batter_id) & (raw_df["game_type"] == "R")
    ].copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    bp = bp[bp["game_date"] < pd.to_datetime(before_date)]

    if bp.empty:
        return {f: float(baselines[f]) for f in RECENCY_FEATURES}

    _annotate(bp)

    event_keys = (
        bp.loc[bp["events"].notna(), ["game_pk", "at_bat_number"]]
        .drop_duplicates()
    )
    if event_keys.empty:
        return {f: float(baselines[f]) for f in RECENCY_FEATURES}

    completed = bp.merge(event_keys, on=["game_pk", "at_bat_number"])
    completed["game_pk"] = pd.to_numeric(completed["game_pk"])
    completed["at_bat_number"] = pd.to_numeric(completed["at_bat_number"])

    pa = completed.groupby(["game_pk", "at_bat_number"]).agg(
        game_date=("game_date", "first"),
        n_pitches=("_swing", "size"),
        n_whiffs=("_whiff", "sum"),
        n_outside=("_outside", "sum"),
        n_chases=("_chase", "sum"),
        n_swings=("_swing", "sum"),
        n_contacts=("_contact", "sum"),
        n_has_ev=("_has_ev", "sum"),
        sum_ev=("_ev", "sum"),
        n_hard=("_hard", "sum"),
        n_has_lsa=("_has_lsa", "sum"),
        n_barrels=("_barrel", "sum"),
    ).reset_index()

    pa = pa.sort_values(["game_date", "game_pk", "at_bat_number"])
    recent = pa.tail(window)
    n_pas = len(recent)
    s = recent[_AGG_COLS].sum()

    def sd(num, den):
        return num / den if den > 0 else np.nan

    raw = {
        "batter_swinging_strike_rate": sd(s["n_whiffs"], s["n_pitches"]),
        "batter_chase_rate": sd(s["n_chases"], s["n_outside"]),
        "batter_contact_rate": sd(s["n_contacts"], s["n_swings"]),
        "batter_hard_hit_rate": sd(s["n_hard"], s["n_has_ev"]),
        "batter_avg_exit_velo": sd(s["sum_ev"], s["n_has_ev"]),
        "batter_barrel_rate": sd(s["n_barrels"], s["n_has_lsa"]),
    }

    w = min(n_pas / window, 1.0)
    result = {}
    for feat in RECENCY_FEATURES:
        obs = raw[feat] if not np.isnan(raw[feat]) else baselines[feat]
        result[feat] = float(w * obs + (1 - w) * baselines[feat])
    return result
