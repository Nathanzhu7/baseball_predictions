# Baseball

Pitch-outcome prediction and simulation using Statcast data.

## Why this architecture?

The goal is to simulate a full at-bat from just a pitcher, batter, and game context — and get a realistic outcome. The naive approach would be one big model that takes in (pitcher, batter, count) and directly outputs "single" or "strikeout" or whatever. But that's a bad idea because it conflates a bunch of very different decisions into one prediction.

Instead, we split it into a sequential pipeline of four models, each responsible for one step:

1. **Pitch selection** — given the game state (count, runners, outs, score, handedness, previous pitch), what pitch type does the pitcher throw? A neural net predicts a probability distribution over pitch types and we randomly sample from it. This captures pitcher tendencies — Peralta throws more fastballs on 3-0 and more offspeed on 0-2.

2. **Pitch characteristics** — given the pitch type, what does the pitch actually look like physically? We fit multivariate Gaussians per (pitcher, pitch type) with shrinkage toward league averages, so each pitcher has their own velocity/spin/movement profile. Count adjustments shift location toward the zone on hitter's counts and away on pitcher's counts. Each sample is different, so you get realistic variation.

3. **Pitch outcome** — given the full pitch characteristics (velocity, spin, location, movement, pitch type) plus the batter's recent tendencies and the game state (runners, outs), what happens? Ball, called strike, swinging strike, foul, in play, or HBP. This is the core neural net — it takes in the actual physics of the pitch and predicts what happens to it.

4. **In-play resolution** — if the ball is put in play, we need to know if it's a single, home run, ground out, etc. We deliberately did NOT expand the pitch outcome model to predict these directly because the class imbalance would be brutal — triples are 0.1% of all pitches, and the model would just learn to never predict them. Instead, we generate (exit velocity, launch angle) from an additive model conditioned on pitcher and batter, then classify the batted ball outcome from those two numbers. This way a dominant pitcher naturally produces lower exit velocities (more outs), and a good hitter produces higher exit velocities (more hits), without needing direct pitcher-batter matchup data that's way too sparse to estimate reliably.

The key insight is that each model operates on a different level of abstraction. Pitch selection is a strategic decision. Pitch characteristics are physical properties. Pitch outcome is biomechanics (does the batter make contact?). In-play resolution is batted ball physics. Splitting them up means each model has a well-defined job, enough training data for its specific task, and we avoid overfitting on things we can't actually estimate.

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
  simulation/                        # Full at-bat and inning simulation
    engine.py                        # At-bat and inning simulation with runner tracking
    batted_ball.py                   # Exit velo/launch angle generation + outcome resolution
    artifacts/                       # Fitted batted ball model
```

## Commands

Run all commands from the project root.

**Simulate innings:**
```
.venv/bin/python -m simulation.engine
```

**Train/fit individual models:**
```
.venv/bin/python -m pitch_outcome_prediction.train
.venv/bin/python -m pitch_selection.train
.venv/bin/python -m pitch_characteristics.fit
.venv/bin/python -m simulation.batted_ball
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then download Statcast data:
```
cd data && python get_data.py
```

## Data

Training uses chronological splits (2024 train / 2025-H1 validation / 2025-H2 test) and early stopping on validation loss. Test data is reserved for final evaluation only.

## Known limitations

The simulation models pitch-level interactions well but does not currently account for:

- **Errors, defensive quality, fielding range** — all batted balls are resolved purely from exit velocity and launch angle, ignoring who's fielding
- **Baserunning decisions** — no steal attempts, hit-and-runs, or extra-base decisions beyond simplified advancement rules
- **Pitcher fatigue and bullpen management** — game pitch count is an input but there's no automatic pitcher change or realistic fatigue curve
- **Park factors** — Coors Field plays very differently from Oracle Park; the model treats all parks the same
- **Weather** — wind, humidity, and temperature affect ball carry and pitcher grip
- **Umpire effects** — we use a rule-based strike zone rather than modeling individual umpire tendencies
- **In-game substitutions** — no pinch hitters, defensive replacements, or double switches
- **Clutch/momentum effects** — if these exist beyond what's captured by count, runners, and score differential, the model doesn't model them

These factors mostly average out over many simulations but could matter for specific game predictions. Addressing them is future work.
