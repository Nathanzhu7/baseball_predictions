"""Backtest the simulation engine against actual validation at-bat outcomes.

Simulates at-bats for real matchups from the validation period and compares
the aggregate outcome distributions against what actually happened.

Usage:
    python -m simulation.backtest
"""

import pickle
from collections import Counter

import numpy as np
import pandas as pd

from simulation.engine import simulate_at_bat

# Outcome categorization for actual events
def _categorize_event(event):
    if event in ("strikeout", "strikeout_double_play"):
        return "strikeout"
    if event == "walk":
        return "walk"
    if event == "hit_by_pitch":
        return "hbp"
    if event == "single":
        return "single"
    if event == "double":
        return "double"
    if event == "triple":
        return "triple"
    if event == "home_run":
        return "home_run"
    if event in ("field_out", "force_out", "fielders_choice_out", "other_out"):
        return "field_out"
    if event in ("grounded_into_double_play", "double_play", "triple_play"):
        return "double_play"
    if event in ("sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play"):
        return "sacrifice"
    if event == "fielders_choice":
        return "fielders_choice"
    if event == "field_error":
        return "error"
    return "other"


OUTCOME_ORDER = [
    "strikeout", "field_out", "single", "walk", "double",
    "home_run", "double_play", "hbp", "sacrifice", "triple",
    "fielders_choice", "error",
]


def main(n_sim=5000, seed=42):
    np.random.seed(seed)

    print("Loading validation data...")
    df = pd.read_pickle("data/statcast_raw.pkl")
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    val = df[(df["game_date"] >= "2025-01-01") & (df["game_date"] < "2025-07-01")].copy()

    # Completed PAs
    pa = val[val["events"].notna()].copy()
    pa["outcome"] = pa["events"].map(_categorize_event)

    # Filter to players in all our models
    with open("pitch_selection/artifacts/preprocessing.pkl", "rb") as f:
        ps_prep = pickle.load(f)
    with open("pitch_outcome_prediction/artifacts/preprocessing.pkl", "rb") as f:
        po_prep = pickle.load(f)

    pa = pa[
        pa["pitcher"].isin(ps_prep["pitcher_map"])
        & pa["batter"].isin(ps_prep["batter_map"])
        & pa["pitcher"].isin(po_prep["pitcher_map"])
        & pa["batter"].isin(po_prep["batter_map"])
    ]
    pa = pa[pa["outcome"] != "other"]

    # Convert columns
    pa["outs"] = pd.to_numeric(pa["outs_when_up"], errors="coerce").fillna(0).astype(int)
    pa["runner_1b"] = pa["on_1b"].notna().astype(int)
    pa["runner_2b"] = pa["on_2b"].notna().astype(int)
    pa["runner_3b"] = pa["on_3b"].notna().astype(int)
    bat_score = pd.to_numeric(pa["bat_score"], errors="coerce").fillna(0)
    fld_score = pd.to_numeric(pa["fld_score"], errors="coerce").fillna(0)
    pa["score_diff"] = (fld_score - bat_score).clip(-5, 5).astype(int)

    print(f"Eligible PAs: {len(pa):,}")

    # Pre-warm recency cache for all unique batters (avoids reloading 1GB file per batter)
    unique_batters = pa["batter"].unique()
    print(f"Pre-warming recency cache for {len(unique_batters)} batters...")
    from pitch_outcome_prediction.batter_recency import get_recency_for_batter
    from pitch_outcome_prediction.predict import _recency_cache, prep
    raw_df = pd.read_pickle("data/statcast_raw.pkl")
    for i, bid in enumerate(unique_batters):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(unique_batters)}...")
        cache_key = (int(bid), "2025-04-15")
        if cache_key not in _recency_cache:
            _recency_cache[cache_key] = get_recency_for_batter(
                int(bid), before_date="2025-04-15",
                baselines=prep["recency_baselines"], raw_df=raw_df,
            )
    del raw_df
    print("Cache ready.")

    print(f"Simulating {n_sim} at-bats (sampled from validation matchups)...")

    # Sample real PAs to simulate
    sample = pa.sample(n=n_sim, random_state=seed)

    # Actual outcomes from the sample
    actual_counts = Counter(sample["outcome"])

    # Simulate
    sim_results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_sim}...")

        ab = simulate_at_bat(
            pitcher_id=row["pitcher"],
            batter_id=row["batter"],
            pitcher_hand="R" if row["p_throws"] == "R" else "L",
            batter_hand="R" if row["stand"] == "R" else "L",
            game_pitch_count=40,  # approximate
            outs=row["outs"],
            runner_1b=row["runner_1b"],
            runner_2b=row["runner_2b"],
            runner_3b=row["runner_3b"],
            score_diff=row["score_diff"],
            prediction_date="2025-04-15",
        )
        sim_results.append(ab["result"])

    sim_counts = Counter(sim_results)

    # Print comparison
    print(f"\n{'=' * 62}")
    print(f"Backtest Results ({n_sim:,} at-bats)")
    print(f"{'=' * 62}")
    print(f"\n{'Outcome':<18s} {'Actual':>8s} {'Actual%':>8s}  {'Sim':>8s} {'Sim%':>8s}  {'Diff':>7s}")
    print("-" * 62)

    for outcome in OUTCOME_ORDER:
        a = actual_counts.get(outcome, 0)
        s = sim_counts.get(outcome, 0)
        a_pct = a / n_sim * 100
        s_pct = s / n_sim * 100
        diff = s_pct - a_pct
        print(f"  {outcome:<16s} {a:>8,} {a_pct:>7.1f}%  {s:>8,} {s_pct:>7.1f}%  {diff:>+6.1f}%")

    # Summary stats
    actual_k_rate = actual_counts.get("strikeout", 0) / n_sim
    sim_k_rate = sim_counts.get("strikeout", 0) / n_sim
    actual_bb_rate = actual_counts.get("walk", 0) / n_sim
    sim_bb_rate = sim_counts.get("walk", 0) / n_sim
    actual_hr_rate = actual_counts.get("home_run", 0) / n_sim
    sim_hr_rate = sim_counts.get("home_run", 0) / n_sim
    actual_hit_rate = sum(actual_counts.get(h, 0) for h in ["single", "double", "triple", "home_run"]) / n_sim
    sim_hit_rate = sum(sim_counts.get(h, 0) for h in ["single", "double", "triple", "home_run"]) / n_sim

    print(f"\n{'Summary':<18s} {'Actual':>8s} {'':>8s}  {'Sim':>8s}")
    print("-" * 45)
    print(f"  {'K rate':<16s} {actual_k_rate:>8.1%}           {sim_k_rate:>8.1%}")
    print(f"  {'BB rate':<16s} {actual_bb_rate:>8.1%}           {sim_bb_rate:>8.1%}")
    print(f"  {'HR rate':<16s} {actual_hr_rate:>8.1%}           {sim_hr_rate:>8.1%}")
    print(f"  {'Hit rate':<16s} {actual_hit_rate:>8.1%}           {sim_hit_rate:>8.1%}")

    # Pitches per AB
    # We don't have actual pitch counts per PA easily, but we can compute sim
    print(f"\n  Sim avg pitches/AB: {np.mean([len(r) for r in []]):.1f}" if False else "")


if __name__ == "__main__":
    main()
