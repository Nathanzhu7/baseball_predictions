"""Batted ball model: generate exit velocity + launch angle, resolve in-play outcome.

Uses an additive model:
    batted_ball = league(pitch_type) + pitcher_effect(pitcher, pitch_type) + batter_effect(batter) + noise

Then maps (exit_velo, launch_angle) -> outcome category using a simple classifier.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import PACKAGE_DIR, PROJECT_ROOT

ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

BATTED_FEATURES = ["launch_speed", "launch_angle"]

IN_PLAY_DESCRIPTIONS = {
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

OUTCOME_CATEGORIES = [
    "field_out", "single", "double", "triple", "home_run",
    "double_play", "sacrifice", "fielders_choice", "error",
]

SHRINKAGE_N_PITCHER = 30
SHRINKAGE_N_BATTER_OVERALL = 30
SHRINKAGE_N_BATTER_PT = 15  # batter+pitch_type shrinks toward batter overall


def _categorize_event(event):
    if event in ("field_out", "force_out", "fielders_choice_out", "other_out"):
        return "field_out"
    if event == "single":
        return "single"
    if event == "double":
        return "double"
    if event == "triple":
        return "triple"
    if event == "home_run":
        return "home_run"
    if event in ("grounded_into_double_play", "double_play", "triple_play"):
        return "double_play"
    if event in ("sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play"):
        return "sacrifice"
    if event == "fielders_choice":
        return "fielders_choice"
    if event == "field_error":
        return "error"
    return "field_out"


def fit(
    cache_file=DATA_DIR / "statcast_raw.pkl",
    output_file=ARTIFACTS_DIR / "batted_ball.pkl",
):
    """Fit the batted ball model from training data."""
    print("Fitting batted ball model...")
    df = pd.read_pickle(cache_file)
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    train = df[
        (df["game_date"] >= "2024-01-01") & (df["game_date"] < "2025-01-01")
    ].copy()

    ip = train[train["description"].isin(IN_PLAY_DESCRIPTIONS)].copy()
    for c in BATTED_FEATURES:
        ip[c] = pd.to_numeric(ip[c], errors="coerce")
    ip = ip.dropna(subset=BATTED_FEATURES + ["events", "pitcher", "batter", "pitch_type"])
    ip["outcome_cat"] = ip["events"].map(_categorize_event)

    print(f"  In-play training samples: {len(ip):,}")

    # --- Part 1: Additive model for (launch_speed, launch_angle) ---

    # Map rare pitch types
    common_pts = ["FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV", "KN"]
    ip["pt_clean"] = ip["pitch_type"].where(ip["pitch_type"].isin(common_pts), "OTHER")

    # League means by pitch type
    league_means = {}
    league_covs = {}
    for pt in common_pts + ["OTHER"]:
        sub = ip[ip["pt_clean"] == pt][BATTED_FEATURES].to_numpy()
        if len(sub) >= 10:
            league_means[pt] = sub.mean(axis=0)
            league_covs[pt] = np.cov(sub, rowvar=False) + np.eye(2) * 1e-4
        else:
            all_data = ip[BATTED_FEATURES].to_numpy()
            league_means[pt] = all_data.mean(axis=0)
            league_covs[pt] = np.cov(all_data, rowvar=False) + np.eye(2) * 1e-4

    # Pitcher effects by pitch type (deviation from league mean, shrunk)
    pitcher_effects = {}
    for pitcher_id in ip["pitcher"].unique():
        pitcher_effects[pitcher_id] = {}
        pitcher_data = ip[ip["pitcher"] == pitcher_id]
        for pt in common_pts + ["OTHER"]:
            sub = pitcher_data[pitcher_data["pt_clean"] == pt][BATTED_FEATURES].to_numpy()
            n = len(sub)
            if n == 0:
                pitcher_effects[pitcher_id][pt] = np.zeros(2)
                continue
            raw_effect = sub.mean(axis=0) - league_means[pt]
            w = n / (n + SHRINKAGE_N_PITCHER)
            pitcher_effects[pitcher_id][pt] = w * raw_effect

    # Batter effects: two levels
    #   1) batter overall effect (shrunk toward 0)
    #   2) batter + pitch_type effect (shrunk toward batter overall)
    grand_mean = ip[BATTED_FEATURES].to_numpy().mean(axis=0)

    batter_effects_overall = {}
    batter_effects_by_pt = {}

    for batter_id in ip["batter"].unique():
        batter_data = ip[ip["batter"] == batter_id]

        # Overall batter effect
        sub_all = batter_data[BATTED_FEATURES].to_numpy()
        n_all = len(sub_all)
        if n_all == 0:
            batter_effects_overall[batter_id] = np.zeros(2)
        else:
            raw_overall = sub_all.mean(axis=0) - grand_mean
            w_overall = n_all / (n_all + SHRINKAGE_N_BATTER_OVERALL)
            batter_effects_overall[batter_id] = w_overall * raw_overall

        # Per pitch-type batter effect (shrunk toward overall)
        batter_effects_by_pt[batter_id] = {}
        for pt in common_pts + ["OTHER"]:
            sub_pt = batter_data[batter_data["pt_clean"] == pt][BATTED_FEATURES].to_numpy()
            n_pt = len(sub_pt)
            if n_pt == 0:
                batter_effects_by_pt[batter_id][pt] = batter_effects_overall[batter_id]
                continue
            # Raw effect relative to league pitch-type mean
            raw_pt = sub_pt.mean(axis=0) - league_means[pt]
            w_pt = n_pt / (n_pt + SHRINKAGE_N_BATTER_PT)
            # Shrink toward the batter's overall effect
            batter_effects_by_pt[batter_id][pt] = (
                w_pt * raw_pt + (1 - w_pt) * batter_effects_overall[batter_id]
            )

    # --- Part 2: Outcome classifier from (launch_speed, launch_angle) ---

    outcome_to_idx = {o: i for i, o in enumerate(OUTCOME_CATEGORIES)}
    X = ip[BATTED_FEATURES].to_numpy()
    y = ip["outcome_cat"].map(outcome_to_idx).to_numpy()

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(X, y)
    train_acc = (clf.predict(X) == y).mean()
    print(f"  Outcome classifier train accuracy: {train_acc:.3f}")

    # --- Save ---
    result = {
        "league_means": league_means,
        "league_covs": league_covs,
        "pitcher_effects": pitcher_effects,
        "batter_effects_overall": batter_effects_overall,
        "batter_effects_by_pt": batter_effects_by_pt,
        "grand_mean": grand_mean,
        "classifier": clf,
        "outcome_categories": OUTCOME_CATEGORIES,
        "outcome_to_idx": outcome_to_idx,
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved to {output_file}")

    # --- Evaluate on validation ---
    _evaluate(df, result)

    return result


def _evaluate(df, model):
    """Evaluate on validation data."""
    val = df[
        (df["game_date"] >= "2025-01-01") & (df["game_date"] < "2025-07-01")
    ].copy()

    ip = val[val["description"].isin(IN_PLAY_DESCRIPTIONS)].copy()
    for c in BATTED_FEATURES:
        ip[c] = pd.to_numeric(ip[c], errors="coerce")
    ip = ip.dropna(subset=BATTED_FEATURES + ["events", "pitcher", "batter", "pitch_type"])
    ip["outcome_cat"] = ip["events"].map(_categorize_event)

    # Filter to known pitchers/batters
    ip = ip[
        ip["pitcher"].isin(model["pitcher_effects"])
        & ip["batter"].isin(model["batter_effects_by_pt"])
    ]

    print(f"\n  Validation in-play samples: {len(ip):,}")

    # Evaluate batted ball generation (additive model MAE)
    common_pts = list(model["league_means"].keys())
    ip["pt_clean"] = ip["pitch_type"].where(ip["pitch_type"].isin(common_pts), "OTHER")

    pred_ev = []
    pred_la = []
    actual_ev = ip["launch_speed"].to_numpy()
    actual_la = ip["launch_angle"].to_numpy()

    for _, row in ip.iterrows():
        pt = row["pt_clean"]
        pid = row["pitcher"]
        bid = row["batter"]
        batter_eff = model["batter_effects_by_pt"].get(bid, {}).get(
            pt, model["batter_effects_overall"].get(bid, np.zeros(2))
        )
        pred = (
            model["league_means"].get(pt, model["grand_mean"])
            + model["pitcher_effects"].get(pid, {}).get(pt, np.zeros(2))
            + batter_eff
        )
        pred_ev.append(pred[0])
        pred_la.append(pred[1])

    pred_ev = np.array(pred_ev)
    pred_la = np.array(pred_la)

    # Baseline: league mean by pitch type only
    base_ev = []
    base_la = []
    for _, row in ip.iterrows():
        pt = row["pt_clean"]
        m = model["league_means"].get(pt, model["grand_mean"])
        base_ev.append(m[0])
        base_la.append(m[1])
    base_ev = np.array(base_ev)
    base_la = np.array(base_la)

    print(f"\n  {'Metric':<28s} {'Baseline':>12s}  {'Our Model':>12s}")
    print(f"  {'-' * 56}")
    print(f"  {'Exit Velo MAE (mph)':<28s} {np.abs(actual_ev - base_ev).mean():12.2f}  {np.abs(actual_ev - pred_ev).mean():12.2f}")
    print(f"  {'Launch Angle MAE (deg)':<28s} {np.abs(actual_la - base_la).mean():12.2f}  {np.abs(actual_la - pred_la).mean():12.2f}")

    # Outcome classifier accuracy on val
    X_val = ip[BATTED_FEATURES].to_numpy()
    outcome_to_idx = model["outcome_to_idx"]
    y_val = ip["outcome_cat"].map(outcome_to_idx).to_numpy()
    val_acc = (model["classifier"].predict(X_val) == y_val).mean()
    print(f"  {'Outcome classifier acc':<28s} {'---':>12s}  {val_acc:12.3f}")


def load(model_file=ARTIFACTS_DIR / "batted_ball.pkl"):
    """Load fitted batted ball model."""
    with open(model_file, "rb") as f:
        return pickle.load(f)


_model = None
_BOUNDS = {
    "launch_speed": (10.0, 120.0),
    "launch_angle": (-90.0, 90.0),
}
_MAX_ATTEMPTS = 20


def _get_model():
    global _model
    if _model is None:
        _model = load()
    return _model


def resolve_in_play(pitcher_id, batter_id, pitch_type):
    """Given a ball in play, sample batted ball characteristics and resolve the outcome.

    Returns
    -------
    dict with:
        outcome : str  (e.g. "single", "field_out", "home_run")
        launch_speed : float
        launch_angle : float
    """
    m = _get_model()

    common_pts = list(m["league_means"].keys())
    pt_clean = pitch_type if pitch_type in common_pts else "OTHER"

    # Additive mean: league + pitcher effect + batter effect (all by pitch type)
    batter_eff = m["batter_effects_by_pt"].get(batter_id, {}).get(
        pt_clean, m["batter_effects_overall"].get(batter_id, np.zeros(2))
    )
    mean = (
        m["league_means"].get(pt_clean, m["grand_mean"])
        + m["pitcher_effects"].get(pitcher_id, {}).get(pt_clean, np.zeros(2))
        + batter_eff
    )

    # Use league covariance for the pitch type
    cov = m["league_covs"].get(pt_clean, m["league_covs"]["FF"])

    # Sample with rejection for plausibility
    for _ in range(_MAX_ATTEMPTS):
        sample = np.random.multivariate_normal(mean, cov)
        if (_BOUNDS["launch_speed"][0] <= sample[0] <= _BOUNDS["launch_speed"][1]
                and _BOUNDS["launch_angle"][0] <= sample[1] <= _BOUNDS["launch_angle"][1]):
            break
    else:
        sample[0] = np.clip(sample[0], *_BOUNDS["launch_speed"])
        sample[1] = np.clip(sample[1], *_BOUNDS["launch_angle"])

    # Classify outcome
    probs = m["classifier"].predict_proba(sample.reshape(1, -1))[0]
    idx = np.random.choice(len(probs), p=probs)
    outcome = m["outcome_categories"][idx]

    return {
        "outcome": outcome,
        "launch_speed": float(sample[0]),
        "launch_angle": float(sample[1]),
    }


if __name__ == "__main__":
    fit()
