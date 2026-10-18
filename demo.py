"""End-to-end demo: generate a pitch and predict its outcome.

Usage:
    python demo.py
"""

import numpy as np
from pitch_selection.predict import sample_pitch_type
from pitch_characteristics.predict import generate_pitch_characteristics
from pitch_outcome_prediction.predict import predict_pitch


def generate_and_predict(pitcher_id, batter_id, game_context, prediction_date=None):
    """Generate a realistic pitch and predict its outcome.

    Returns (pitch_type, pitch_dict, outcome_probs).
    """
    pitch_type = sample_pitch_type(pitcher_id, batter_id, game_context)

    pitch = generate_pitch_characteristics(
        pitcher_id, batter_id, pitch_type, game_context
    )

    outcome = predict_pitch(
        pitcher_id, batter_id, pitch,
        prediction_date=prediction_date,
    )

    return pitch_type, pitch, outcome


def main():
    # Freddy Peralta (642547) vs Freddie Freeman (518692)
    pitcher_id = 642547
    batter_id = 518692

    game_context = {
        "balls": 1,
        "strikes": 1,
        "prev_pitch_type": None,
        "game_pitch_count": 45,
        "batter_hand": "L",
        "pitcher_hand": "R",
    }

    np.random.seed(42)

    # Single pitch demo
    print("=" * 60)
    print("Single pitch: Peralta vs Freeman, 1-1 count")
    print("=" * 60)

    pitch_type, pitch, outcome = generate_and_predict(
        pitcher_id, batter_id, game_context, prediction_date="2025-07-15"
    )

    print(f"\nPitch type: {pitch_type}")
    print(f"\nCharacteristics:")
    for key in ["release_speed", "release_spin_rate", "plate_x", "plate_z",
                "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension"]:
        print(f"  {key:25s}: {pitch[key]:8.2f}")
    print(f"\nOutcome probabilities:")
    for outcome_name, prob in outcome.items():
        print(f"  {outcome_name:20s}: {prob:.2%}")

    # Generate 100 pitches and summarize
    print("\n" + "=" * 60)
    print("100 pitches: Peralta vs Freeman, 1-1 count")
    print("=" * 60)

    pitch_types_sampled = []
    velocities = []
    spins = []
    plate_xs = []
    plate_zs = []

    for _ in range(100):
        pt, pitch, _ = generate_and_predict(
            pitcher_id, batter_id, game_context, prediction_date="2025-07-15"
        )
        pitch_types_sampled.append(pt)
        velocities.append(pitch["release_speed"])
        spins.append(pitch["release_spin_rate"])
        plate_xs.append(pitch["plate_x"])
        plate_zs.append(pitch["plate_z"])

    # Pitch type frequencies
    print(f"\nPitch type distribution (n=100):")
    from collections import Counter
    counts = Counter(pitch_types_sampled)
    for pt, n in counts.most_common():
        print(f"  {pt:6s}: {n:3d}")

    # Summary statistics
    print(f"\nCharacteristic summary:")
    print(f"  {'':20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*52}")
    for name, vals in [
        ("Velocity (mph)", velocities),
        ("Spin (rpm)", spins),
        ("Plate X (ft)", plate_xs),
        ("Plate Z (ft)", plate_zs),
    ]:
        arr = np.array(vals)
        print(f"  {name:20s} {arr.mean():8.1f} {arr.std():8.1f} {arr.min():8.1f} {arr.max():8.1f}")


if __name__ == "__main__":
    main()
