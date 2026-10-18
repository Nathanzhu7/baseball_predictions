"""Generate realistic pitch characteristics by sampling from fitted distributions."""

import pickle

import numpy as np

from . import ARTIFACTS_DIR

with open(ARTIFACTS_DIR / "pitch_characteristics.pkl", "rb") as f:
    _model = pickle.load(f)

_MAX_ATTEMPTS = 20


def generate_pitch_characteristics(pitcher_id, batter_id, pitch_type, game_context):
    """Generate a realistic pitch-characteristics dictionary.

    Parameters
    ----------
    pitcher_id : int  MLB pitcher ID
    batter_id  : int  MLB batter ID (unused currently, reserved for future)
    pitch_type : str  e.g. "FF", "SL", "CH"
    game_context : dict with keys:
        balls : int
        strikes : int

    Returns
    -------
    dict with keys matching pitch_outcome_prediction input.
    """
    pt_clean = pitch_type if pitch_type in _model["pitch_types"] else "OTHER"

    # Get pitcher-specific distribution (or league fallback)
    pitcher_models = _model["pitcher_models"]
    if pitcher_id in pitcher_models and pt_clean in pitcher_models[pitcher_id]:
        mean, cov = pitcher_models[pitcher_id][pt_clean]
    elif pt_clean in _model["league"]:
        mean, cov = _model["league"][pt_clean]
    else:
        mean, cov = _model["league"]["FF"]

    # Apply count adjustment
    balls = game_context.get("balls", 0)
    strikes = game_context.get("strikes", 0)
    count_adj = _model["count_adjustments"].get(
        (pt_clean, balls, strikes), np.zeros(len(mean))
    )
    adjusted_mean = mean + count_adj

    bounds = _model["bounds"]
    characteristics = _model["characteristics"]

    # Sample with rejection for physical plausibility
    for _ in range(_MAX_ATTEMPTS):
        sample = np.random.multivariate_normal(adjusted_mean, cov)
        plausible = True
        for i, c in enumerate(characteristics):
            lo, hi = bounds[c]
            if sample[i] < lo or sample[i] > hi:
                plausible = False
                break
        if plausible:
            break
    else:
        for i, c in enumerate(characteristics):
            lo, hi = bounds[c]
            sample[i] = np.clip(sample[i], lo, hi)

    result = {"pitch_type": pitch_type}
    for i, c in enumerate(characteristics):
        result[c] = float(sample[i])

    result["balls"] = balls
    result["strikes"] = strikes

    return result
