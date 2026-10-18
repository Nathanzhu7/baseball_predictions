"""Inference for pitch selection: sample a pitch type given context."""

import pickle

import numpy as np
import torch

from . import ARTIFACTS_DIR
from .model import PitchSelectionModel

with open(ARTIFACTS_DIR / "preprocessing.pkl", "rb") as f:
    _prep = pickle.load(f)

_model = PitchSelectionModel(
    num_pitchers=len(_prep["pitcher_map"]),
    num_batters=len(_prep["batter_map"]),
    num_prev_labels=len(_prep["prev_labels"]),
)
_model.load_state_dict(
    torch.load(ARTIFACTS_DIR / "pitch_selection_model.pt", map_location="cpu", weights_only=True)
)
_model.eval()


def sample_pitch_type(pitcher_id, batter_id, game_context):
    """Sample a pitch type from the predicted distribution.

    Parameters
    ----------
    pitcher_id : int  MLB pitcher ID
    batter_id  : int  MLB batter ID
    game_context : dict with keys:
        balls : int
        strikes : int
        prev_pitch_type : str or None
        game_pitch_count : int
        batter_hand : str  ("R" or "L")
        pitcher_hand : str  ("R" or "L")
        runner_1b : bool or int  (default 0)
        runner_2b : bool or int  (default 0)
        runner_3b : bool or int  (default 0)
        outs : int  (0, 1, or 2; default 0)
        score_diff : int  (from pitcher's perspective, positive = ahead; default 0)

    Returns
    -------
    str : sampled pitch type (e.g. "FF", "SL", "CH")
    """
    p_idx = _prep["pitcher_map"][pitcher_id]
    b_idx = _prep["batter_map"][batter_id]

    prev_raw = game_context.get("prev_pitch_type")
    if prev_raw is None:
        prev_label = "NONE"
    elif prev_raw in _prep["pitch_to_idx"]:
        prev_label = prev_raw
    else:
        prev_label = "OTHER"
    prev_idx = _prep["prev_to_idx"][prev_label]

    gpc = min(game_context.get("game_pitch_count", 1), 120) / 120.0
    bh = 1.0 if game_context.get("batter_hand", "R") == "R" else 0.0
    ph = 1.0 if game_context.get("pitcher_hand", "R") == "R" else 0.0
    r1 = float(game_context.get("runner_1b", 0))
    r2 = float(game_context.get("runner_2b", 0))
    r3 = float(game_context.get("runner_3b", 0))
    outs = float(game_context.get("outs", 0))
    sd = max(-5, min(5, game_context.get("score_diff", 0))) / 5.0

    with torch.no_grad():
        logits = _model(
            torch.tensor([p_idx], dtype=torch.long),
            torch.tensor([b_idx], dtype=torch.long),
            torch.tensor([game_context.get("balls", 0)], dtype=torch.long),
            torch.tensor([game_context.get("strikes", 0)], dtype=torch.long),
            torch.tensor([prev_idx], dtype=torch.long),
            torch.tensor([gpc], dtype=torch.float32),
            torch.tensor([bh], dtype=torch.float32),
            torch.tensor([ph], dtype=torch.float32),
            torch.tensor([r1], dtype=torch.float32),
            torch.tensor([r2], dtype=torch.float32),
            torch.tensor([r3], dtype=torch.float32),
            torch.tensor([outs], dtype=torch.float32),
            torch.tensor([sd], dtype=torch.float32),
        )
        # Apply repertoire mask
        mask = torch.tensor(_prep["repertoire"][[p_idx]], dtype=torch.float32)
        logits = logits + (1 - mask) * (-1e9)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    idx = np.random.choice(len(probs), p=probs)
    return _prep["pitch_types"][idx]
