# Baseball

Pitch-outcome prediction and simulation using Statcast data.

## Project structure

```
baseball/
  data/                              # Shared raw data
    get_data.py                      # Statcast download utility
    statcast_raw.pkl                 # Raw pitch-level Statcast data
  pitch_outcome_prediction/          # Predict pitch outcomes (6 classes)
    model.py                         # PitchOutcomeModel (embeddings + MLP)
    batter_recency.py                # Rolling 50-PA batter recency features
    prepare_data.py                  # Data loading, splits, preprocessing
    train.py                         # Training with early stopping
    predict.py                       # Single-pitch inference
    evaluation.py                    # Loss, accuracy, calibration metrics
    artifacts/                       # Saved weights, preprocessing, plots
  pitch_selection/                   # Predict pitch type distribution
    model.py                         # PitchSelectionModel (embeddings + MLP)
    prepare_data.py                  # Data prep with game context features
    train.py                         # Training with baseline comparison
    predict.py                       # Sample pitch type from distribution
    evaluation.py                    # Loss, accuracy, baseline metrics
    artifacts/                       # Saved weights, preprocessing
  pitch_characteristics/             # Generate realistic pitch characteristics
    fit.py                           # Fit regularized multivariate Gaussians
    predict.py                       # Sample from fitted distributions
    artifacts/                       # Fitted distributions
  simulation/                        # (future)
  no_embeddings/                     # Experiment: embeddings ablation
  demo.py                            # End-to-end pitch generation demo
  strikeout.py                       # Standalone strikeout analysis
```

## Commands

Run all commands from the project root.

**Train pitch outcome model:**
```
.venv/bin/python -m pitch_outcome_prediction.train
```

**Train pitch selection model:**
```
.venv/bin/python -m pitch_selection.train
```

**Fit pitch characteristics distributions:**
```
.venv/bin/python -m pitch_characteristics.fit
```

**Run end-to-end demo** (generate pitch and predict outcome):
```
.venv/bin/python demo.py
```

**Run inference modules individually:**
```
.venv/bin/python -m pitch_outcome_prediction.predict
.venv/bin/python -m pitch_selection.predict
```

## Modules

### Pitch outcome prediction

Predicts one of six outcomes: ball, called strike, swinging strike, foul, in play, hit by pitch.

Inputs: 16-dim pitcher/batter embeddings + 11 pitch characteristics + pitch type one-hot + 6 batter recency features.
Architecture: linear → ReLU → linear(64, 6).

### Pitch selection

Predicts a probability distribution over 12 pitch types (11 common types + OTHER).

Inputs: 16-dim pitcher/batter embeddings + count + previous pitch + game pitch count (fatigue) + handedness.
Architecture: linear → ReLU → linear(64, 12). Pitcher repertoire mask applied at inference.

At inference, randomly samples from the predicted distribution.

### Pitch characteristics

Generates realistic pitch characteristics (velocity, spin, movement, location) from regularized multivariate Gaussians fitted per (pitcher, pitch_type) with shrinkage toward league-level distributions.

Features rejection sampling for physical plausibility bounds.

## Data

Training uses chronological splits (2024 train / 2025-H1 validation / 2025-H2 test) and early stopping on validation loss. Test data is reserved for final evaluation only.
