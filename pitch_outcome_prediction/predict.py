import pickle

import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS_DIR
from .batter_recency import RECENCY_FEATURES, get_recency_for_batter
from .model import PitchOutcomeModel

with open(ARTIFACTS_DIR / "preprocessing.pkl", "rb") as f:
    prep = pickle.load(f)

model = PitchOutcomeModel(
    num_pitchers=len(prep["pitcher_map"]),
    num_batters=len(prep["batter_map"]),
    num_features=(
        len(prep["numeric_features"])
        + len(prep["type_columns"])
        + len(prep["recency_features"])
        + len(prep["game_state_features"])
    ),
)
model.load_state_dict(
    torch.load(ARTIFACTS_DIR / "pitch_model.pt", map_location="cpu", weights_only=True)
)
model.eval()

_recency_cache = {}


def predict_pitch(pitcher_id, batter_id, pitch, prediction_date=None, game_state=None):
    """Predict pitch outcome probabilities.

    Parameters
    ----------
    pitcher_id : int   MLB pitcher ID
    batter_id  : int   MLB batter ID
    pitch      : dict  Pitch characteristics
    prediction_date : str or None
        Date for recency lookup (e.g. "2025-07-15"). If None, uses all
        available historical data.
    game_state : dict or None
        Optional dict with runner_1b, runner_2b, runner_3b (0/1), outs (0/1/2).
        Defaults to bases empty, 0 outs.
    """
    if game_state is None:
        game_state = {}
    p_id = prep["pitcher_map"][pitcher_id]
    b_id = prep["batter_map"][batter_id]

    # Pitch numeric features
    numeric = pd.Series({
        col: pitch[col] for col in prep["numeric_features"]
    })
    numeric = (
        (numeric - prep["mean"]) / prep["std"]
    ).to_numpy(dtype="float32")

    # Pitch-type one-hot features
    pitch_types = [
        1.0 if col == f"pitch_{pitch['pitch_type']}" else 0.0
        for col in prep["type_columns"]
    ]

    # Batter recency features (cached per batter + date)
    cache_key = (batter_id, prediction_date)
    if cache_key in _recency_cache:
        recency = _recency_cache[cache_key]
    else:
        recency = get_recency_for_batter(
            batter_id,
            before_date=prediction_date or "2099-01-01",
            baselines=prep["recency_baselines"],
        )
        _recency_cache[cache_key] = recency
    recency_arr = np.array(
        [recency[f] for f in RECENCY_FEATURES], dtype="float32"
    )
    recency_norm = (recency_arr - prep["recency_mean"]) / prep["recency_std"]

    # Game state features
    gs_raw = np.array([
        float(game_state.get("runner_1b", 0)),
        float(game_state.get("runner_2b", 0)),
        float(game_state.get("runner_3b", 0)),
        float(game_state.get("outs", 0)),
    ], dtype="float32")
    gs_norm = (gs_raw - prep["game_state_mean"]) / prep["game_state_std"]

    features = torch.tensor(
        [list(numeric) + pitch_types + recency_norm.tolist() + gs_norm.tolist()],
        dtype=torch.float32,
    )
    pitcher = torch.tensor([p_id], dtype=torch.long)
    batter = torch.tensor([b_id], dtype=torch.long)

    with torch.no_grad():
        logits = model(pitcher, batter, features)
        probabilities = torch.softmax(logits, dim=1)[0]

    return dict(zip(prep["outcomes"], probabilities.tolist()))


if __name__ == "__main__":
    pitch = {
        "pitch_type": "FF",
        "release_speed": 95.0,
        "release_spin_rate": 2400,
        "release_pos_x": -2.0,
        "release_pos_z": 6.0,
        "plate_x": 0.2,
        "plate_z": 2.5,
        "pfx_x": -0.5,
        "pfx_z": 1.5,
        "release_extension": 6.2,
        "balls": 1,
        "strikes": 1,
    }

    result = predict_pitch(
        pitcher_id=642547,
        batter_id=605141,
        pitch=pitch,
        prediction_date="2025-07-15",
    )

    for outcome, probability in result.items():
        print(f"{outcome:20s}: {probability:.2%}")
