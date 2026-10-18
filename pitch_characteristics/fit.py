"""Fit multivariate Gaussian distributions for pitch characteristics.

For each (pitcher, pitch_type) pair, fit a regularized multivariate Gaussian
to the historical training data. Use shrinkage toward pitch-type-level
league distributions when a pitcher has limited observations.

Count adjustments shift the mean based on the ball-strike count (e.g., pitchers
groove pitches on 3-0 and expand the zone on 0-2).

Usage:
    python -m pitch_characteristics.fit
"""

import pickle

import numpy as np
import pandas as pd

from . import DATA_DIR, ARTIFACTS_DIR

CHARACTERISTICS = [
    "release_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "plate_x",
    "plate_z",
    "pfx_x",
    "pfx_z",
    "release_extension",
]

PITCH_TYPES = [
    "FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV", "KN",
]

SHRINKAGE_N = 50
COV_REG = 1e-4


def _fit_gaussian(data):
    """Fit mean and covariance from an (N, D) array."""
    mean = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov += np.eye(cov.shape[0]) * COV_REG
    return mean, cov


def fit(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    output_file=ARTIFACTS_DIR / "pitch_characteristics.pkl",
):
    """Fit pitch-characteristic distributions and save to disk."""
    print("Fitting pitch characteristic distributions...")
    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    train = df[
        (df["game_date"] >= "2024-01-01") & (df["game_date"] < "2025-01-01")
    ].copy()

    for col in CHARACTERISTICS:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    train["balls"] = pd.to_numeric(train["balls"], errors="coerce").fillna(0).astype(int)
    train["strikes"] = pd.to_numeric(train["strikes"], errors="coerce").fillna(0).astype(int)
    train = train.dropna(subset=CHARACTERISTICS + ["pitch_type", "pitcher"])
    train["pitch_type_clean"] = train["pitch_type"].where(
        train["pitch_type"].isin(PITCH_TYPES), "OTHER"
    )

    # Step 1: League-level distributions per pitch type
    league = {}
    for pt in PITCH_TYPES + ["OTHER"]:
        pt_data = train[train["pitch_type_clean"] == pt][CHARACTERISTICS].to_numpy()
        if len(pt_data) >= 10:
            league[pt] = _fit_gaussian(pt_data)
        else:
            all_data = train[CHARACTERISTICS].to_numpy()
            league[pt] = _fit_gaussian(all_data)

    # Step 2: Count adjustments (additive shift relative to overall mean)
    # Computed at the league level per (pitch_type, balls, strikes)
    overall_mean = train[CHARACTERISTICS].to_numpy().mean(axis=0)
    count_adjustments = {}  # (pitch_type, balls, strikes) -> offset array

    for pt in PITCH_TYPES + ["OTHER"]:
        pt_data = train[train["pitch_type_clean"] == pt]
        pt_mean = league[pt][0]  # mean for this pitch type across all counts

        for b in range(4):
            for s in range(3):
                sub = pt_data[
                    (pt_data["balls"] == b) & (pt_data["strikes"] == s)
                ][CHARACTERISTICS].to_numpy()
                n = len(sub)
                if n >= 20:
                    raw_offset = sub.mean(axis=0) - pt_mean
                    # Light shrinkage toward zero
                    w = n / (n + 100)
                    count_adjustments[(pt, b, s)] = w * raw_offset
                else:
                    count_adjustments[(pt, b, s)] = np.zeros(len(CHARACTERISTICS))

    # Step 3: Pitcher-specific distributions with shrinkage
    pitcher_models = {}
    for pitcher_id in train["pitcher"].unique():
        pitcher_data = train[train["pitcher"] == pitcher_id]
        pitcher_models[pitcher_id] = {}

        for pt in PITCH_TYPES + ["OTHER"]:
            pt_data = pitcher_data[pitcher_data["pitch_type_clean"] == pt][CHARACTERISTICS].to_numpy()
            n = len(pt_data)

            if n == 0:
                pitcher_models[pitcher_id][pt] = league[pt]
                continue

            league_mean, league_cov = league[pt]

            if n < 3:
                pitcher_models[pitcher_id][pt] = (league_mean.copy(), league_cov.copy())
                continue

            pitcher_mean, pitcher_cov = _fit_gaussian(pt_data)
            w = n / (n + SHRINKAGE_N)
            shrunk_mean = w * pitcher_mean + (1 - w) * league_mean
            shrunk_cov = w * pitcher_cov + (1 - w) * league_cov
            pitcher_models[pitcher_id][pt] = (shrunk_mean, shrunk_cov)

    # Physical plausibility bounds
    bounds = {
        "release_speed": (50.0, 110.0),
        "release_spin_rate": (500.0, 4000.0),
        "release_pos_x": (-5.0, 5.0),
        "release_pos_z": (3.0, 8.0),
        "plate_x": (-3.0, 3.0),
        "plate_z": (-1.0, 6.0),
        "pfx_x": (-3.0, 3.0),
        "pfx_z": (-3.0, 3.0),
        "release_extension": (3.0, 9.0),
    }

    result = {
        "characteristics": CHARACTERISTICS,
        "pitch_types": PITCH_TYPES,
        "league": league,
        "pitcher_models": pitcher_models,
        "count_adjustments": count_adjustments,
        "bounds": bounds,
    }

    with open(output_file, "wb") as f:
        pickle.dump(result, f)

    print(f"Fitted distributions for {len(pitcher_models)} pitchers")
    print(f"Saved to {output_file}")

    # Print count adjustment summary for fastballs
    print(f"\nCount adjustments for FF (velocity and plate_x, plate_z shifts):")
    vi = CHARACTERISTICS.index("release_speed")
    pxi = CHARACTERISTICS.index("plate_x")
    pzi = CHARACTERISTICS.index("plate_z")
    for b in range(4):
        for s in range(3):
            adj = count_adjustments.get(("FF", b, s), np.zeros(len(CHARACTERISTICS)))
            print(f"  {b}-{s}: velo {adj[vi]:+.2f} mph, "
                  f"plate_x {adj[pxi]:+.3f} ft, plate_z {adj[pzi]:+.3f} ft")

    # Print league summaries
    print(f"\n{'Pitch Type':<12s} {'Count':>8s} {'Velocity':>10s} {'Spin':>10s}")
    print("-" * 44)
    for pt in PITCH_TYPES:
        pt_data = train[train["pitch_type_clean"] == pt]
        n = len(pt_data)
        if n > 0:
            v = league[pt][0][0]
            s = league[pt][0][1]
            print(f"{pt:<12s} {n:>8,} {v:>10.1f} {s:>10.0f}")

    return result


def evaluate(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    model_file=ARTIFACTS_DIR / "pitch_characteristics.pkl",
):
    """Evaluate fitted distributions on validation data."""
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    val = df[
        (df["game_date"] >= "2025-01-01") & (df["game_date"] < "2025-07-01")
    ].copy()

    for col in CHARACTERISTICS:
        val[col] = pd.to_numeric(val[col], errors="coerce")
    val["balls"] = pd.to_numeric(val["balls"], errors="coerce").fillna(0).astype(int)
    val["strikes"] = pd.to_numeric(val["strikes"], errors="coerce").fillna(0).astype(int)
    val = val.dropna(subset=CHARACTERISTICS + ["pitch_type", "pitcher"])
    val["pitch_type_clean"] = val["pitch_type"].where(
        val["pitch_type"].isin(PITCH_TYPES), "OTHER"
    )
    val = val[val["pitcher"].isin(model["pitcher_models"])]

    pitcher_models = model["pitcher_models"]
    league = model["league"]
    count_adj = model["count_adjustments"]

    our_errors = {c: [] for c in CHARACTERISTICS}
    baseline_errors = {c: [] for c in CHARACTERISTICS}

    for _, row in val.iterrows():
        pid = row["pitcher"]
        pt = row["pitch_type_clean"]
        b, s = int(row["balls"]), int(row["strikes"])
        actual = np.array([row[c] for c in CHARACTERISTICS], dtype="float64")

        # Our model: pitcher-specific + count adjustment
        if pt in pitcher_models.get(pid, {}):
            our_mean = pitcher_models[pid][pt][0].copy()
        elif pt in league:
            our_mean = league[pt][0].copy()
        else:
            continue
        our_mean = our_mean + count_adj.get((pt, b, s), np.zeros(len(CHARACTERISTICS)))

        # Baseline: league-level, no count
        if pt in league:
            base_mean = league[pt][0]
        else:
            continue

        for i, c in enumerate(CHARACTERISTICS):
            our_errors[c].append(abs(actual[i] - our_mean[i]))
            baseline_errors[c].append(abs(actual[i] - base_mean[i]))

    print(f"\nPitch Characteristics Evaluation ({len(val):,} validation pitches)")
    print(f"\n{'Metric':<28s} {'Baseline':>12s}  {'Our Model':>12s}")
    print("-" * 56)

    units = {
        "release_speed": "mph", "release_spin_rate": "rpm",
        "release_pos_x": "ft", "release_pos_z": "ft",
        "plate_x": "ft", "plate_z": "ft",
        "pfx_x": "in", "pfx_z": "in",
        "release_extension": "ft",
    }

    for c in CHARACTERISTICS:
        our_mae = np.mean(our_errors[c])
        base_mae = np.mean(baseline_errors[c])
        u = units.get(c, "")
        label = f"{c} MAE ({u})"
        print(f"  {label:<26s} {base_mae:12.3f}  {our_mae:12.3f}")

    return model


if __name__ == "__main__":
    fit()
    evaluate()
