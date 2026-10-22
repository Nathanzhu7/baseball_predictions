"""Baseball simulation engine.

Simulates at-bats and full half-innings by chaining:
    pitch_selection -> pitch_characteristics -> pitch_outcome_prediction -> batted_ball

Usage:
    python -m simulation.engine
"""

import numpy as np

from pitch_selection.predict import sample_pitch_type
from pitch_characteristics.predict import generate_pitch_characteristics
from pitch_outcome_prediction.predict import predict_pitch
from simulation.batted_ball import resolve_in_play

MAX_PITCHES = 30


# ---------------------------------------------------------------------------
# At-bat simulation
# ---------------------------------------------------------------------------

def simulate_at_bat(pitcher_id, batter_id, pitcher_hand, batter_hand,
                    game_pitch_count=0, outs=0, runner_1b=0, runner_2b=0,
                    runner_3b=0, score_diff=0, prediction_date=None):
    """Simulate a complete plate appearance.

    Returns dict with result, pitches list, and final count.
    """
    balls = 0
    strikes = 0
    prev_pitch_type = None
    pitches = []

    game_state = {
        "runner_1b": runner_1b,
        "runner_2b": runner_2b,
        "runner_3b": runner_3b,
        "outs": outs,
    }

    for pitch_num in range(1, MAX_PITCHES + 1):
        gpc = game_pitch_count + pitch_num

        game_context = {
            "balls": balls,
            "strikes": strikes,
            "prev_pitch_type": prev_pitch_type,
            "game_pitch_count": gpc,
            "batter_hand": batter_hand,
            "pitcher_hand": pitcher_hand,
            "runner_1b": runner_1b,
            "runner_2b": runner_2b,
            "runner_3b": runner_3b,
            "outs": outs,
            "score_diff": score_diff,
        }

        pitch_type = sample_pitch_type(pitcher_id, batter_id, game_context)
        pitch = generate_pitch_characteristics(
            pitcher_id, batter_id, pitch_type, game_context,
        )
        outcome_probs = predict_pitch(
            pitcher_id, batter_id, pitch,
            prediction_date=prediction_date,
            game_state=game_state,
        )
        outcomes = list(outcome_probs.keys())
        probs = np.array(list(outcome_probs.values()))
        probs = probs / probs.sum()
        outcome = np.random.choice(outcomes, p=probs)

        pitch_record = {
            "number": pitch_num, "type": pitch_type,
            "velocity": pitch["release_speed"],
            "plate_x": pitch["plate_x"], "plate_z": pitch["plate_z"],
            "outcome": outcome, "balls": balls, "strikes": strikes,
        }

        if outcome == "hit_by_pitch":
            pitch_record["result"] = "hbp"
            pitches.append(pitch_record)
            return {"result": "hbp", "pitches": pitches,
                    "balls": balls, "strikes": strikes}

        if outcome == "in_play":
            bb = resolve_in_play(pitcher_id, batter_id, pitch_type)
            pitch_record["in_play_outcome"] = bb["outcome"]
            pitch_record["launch_speed"] = bb["launch_speed"]
            pitch_record["launch_angle"] = bb["launch_angle"]
            pitch_record["result"] = bb["outcome"]
            pitches.append(pitch_record)
            return {"result": bb["outcome"], "pitches": pitches,
                    "balls": balls, "strikes": strikes}

        pitches.append(pitch_record)
        prev_pitch_type = pitch_type

        if outcome == "ball":
            balls += 1
            if balls >= 4:
                return {"result": "walk", "pitches": pitches,
                        "balls": balls, "strikes": strikes}
        elif outcome in ("called_strike", "swinging_strike"):
            strikes += 1
            if strikes >= 3:
                return {"result": "strikeout", "pitches": pitches,
                        "balls": balls, "strikes": strikes}
        elif outcome == "foul":
            if strikes < 2:
                strikes += 1

    return {"result": "field_out", "pitches": pitches,
            "balls": balls, "strikes": strikes}


# ---------------------------------------------------------------------------
# Runner advancement
# ---------------------------------------------------------------------------

def advance_runners(bases, result, outs_before):
    """Advance runners and return (new_bases, runs_scored, outs_added).

    bases: [1B, 2B, 3B] where each is 1 (occupied) or 0 (empty).
    result: at-bat result string.

    Returns (new_bases, runs_scored, outs_added).
    Uses simplified advancement rules.
    """
    r1, r2, r3 = bases
    runs = 0
    outs_added = 0

    if result == "home_run":
        runs = r1 + r2 + r3 + 1  # everyone scores
        return [0, 0, 0], runs, 0

    if result == "triple":
        runs = r1 + r2 + r3
        return [0, 0, 1], runs, 0

    if result == "double":
        runs = r2 + r3          # runners on 2nd and 3rd score
        runs += r1              # runner on 1st scores (assumes aggressive baserunning)
        return [0, 1, 0], runs, 0

    if result == "single":
        runs = r3               # runner on 3rd scores
        new_3b = r2             # runner on 2nd to 3rd
        new_2b = r1             # runner on 1st to 2nd
        new_1b = 1              # batter to 1st
        return [new_1b, new_2b, new_3b], runs, 0

    if result in ("walk", "hbp"):
        # Force: runners advance only if forced
        if r1:
            if r2:
                if r3:
                    runs = 1    # bases loaded, runner on 3rd scores
                new_3b = r2 if r1 else r3
                new_3b = 1 if (r1 and r2) else r3
            else:
                new_3b = r3
            new_2b = 1 if r1 else r2
        else:
            new_2b = r2
            new_3b = r3
        new_1b = 1
        # Correct force logic
        new_1b = 1
        new_2b = r1 if r1 else r2
        if r1 and r2:
            new_3b = 1
            if r3:
                runs = 1
        elif r1:
            new_2b = 1
            new_3b = r3
        else:
            new_2b = r2
            new_3b = r3
        return [new_1b, new_2b, new_3b], runs, 0

    if result == "strikeout":
        return [r1, r2, r3], 0, 1

    if result == "double_play":
        outs_added = min(2, 3 - outs_before)  # can't exceed 3 total outs
        # Simplified: lead runner and batter are out, others hold
        if r1:
            new_1b = 0  # runner on 1st out
        else:
            new_1b = 0
        new_2b = r2 if outs_before + outs_added < 3 else 0
        new_3b = r3
        # If runner on 3rd and < 2 outs added, they might score
        if r3 and outs_added < 2:
            runs = 1
            new_3b = 0
        return [new_1b, new_2b, new_3b], runs, outs_added

    if result == "sacrifice":
        outs_added = 1
        # Runners advance one base, batter is out
        runs = r3
        new_3b = r2
        new_2b = r1
        new_1b = 0
        return [new_1b, new_2b, new_3b], runs, outs_added

    if result == "fielders_choice":
        outs_added = 1
        # Lead runner is out, batter reaches first
        if r3:
            new_3b = 0  # out at home
        elif r2:
            new_3b = 0
        elif r1:
            pass
        new_1b = 1
        new_2b = r1 if not r2 else r2
        new_3b = r2 if r3 == 0 else 0
        # Simplified: batter on 1st, lead runner out
        return [1, 0, 0], 0, outs_added

    if result == "error":
        # Treat like a single for runner advancement
        runs = r3
        new_3b = r2
        new_2b = r1
        new_1b = 1
        return [new_1b, new_2b, new_3b], runs, 0

    # field_out (default)
    outs_added = 1
    # Runner on 3rd scores on fly out with < 2 outs (sac fly-like)
    if r3 and outs_before < 2:
        runs = 1
        new_3b = 0
    else:
        new_3b = r3
    new_2b = r2
    new_1b = r1
    return [new_1b, new_2b, new_3b], runs, outs_added


# ---------------------------------------------------------------------------
# Inning simulation
# ---------------------------------------------------------------------------

def simulate_inning(pitcher_id, pitcher_hand, lineup, game_pitch_count=0,
                    score_diff=0, prediction_date=None):
    """Simulate a half-inning.

    Parameters
    ----------
    pitcher_id : int         MLB pitcher ID
    pitcher_hand : str       "R" or "L"
    lineup : list of dict    Batting order, each with:
                             {"batter_id": int, "batter_hand": str}
                             Wraps around if more than 9 batters come up.
    game_pitch_count : int   Pitcher's pitch count entering the inning
    score_diff : int         From pitcher's perspective (positive = ahead)
    prediction_date : str    Date for recency lookup

    Returns
    -------
    dict with:
        runs : int                 Runs scored in the inning
        outs : int                 Should always be 3
        at_bats : list[dict]       Each at-bat result with batter info
        total_pitches : int        Pitches thrown in the inning
        lineup_position : int      Next batter index (for continuing lineup)
    """
    outs = 0
    runs = 0
    bases = [0, 0, 0]  # [1B, 2B, 3B]
    at_bats = []
    total_pitches = 0
    batter_idx = 0

    while outs < 3:
        batter_info = lineup[batter_idx % len(lineup)]
        batter_id = batter_info["batter_id"]
        batter_hand = batter_info["batter_hand"]

        ab = simulate_at_bat(
            pitcher_id, batter_id,
            pitcher_hand=pitcher_hand,
            batter_hand=batter_hand,
            game_pitch_count=game_pitch_count + total_pitches,
            outs=outs,
            runner_1b=bases[0], runner_2b=bases[1], runner_3b=bases[2],
            score_diff=score_diff - runs,  # update as runs score
            prediction_date=prediction_date,
        )

        result = ab["result"]
        total_pitches += len(ab["pitches"])

        new_bases, runs_scored, outs_added = advance_runners(bases, result, outs)
        runs += runs_scored
        outs += outs_added
        bases = new_bases if outs < 3 else bases

        ab_record = {
            "batter_id": batter_id,
            "batter_name": batter_info.get("name", str(batter_id)),
            "batter_hand": batter_hand,
            "result": result,
            "pitches": len(ab["pitches"]),
            "runs_scored": runs_scored,
            "outs_added": outs_added,
            "bases_after": list(bases) if outs < 3 else [0, 0, 0],
        }
        at_bats.append(ab_record)
        batter_idx += 1

    return {
        "runs": runs,
        "outs": 3,
        "at_bats": at_bats,
        "total_pitches": total_pitches,
        "lineup_position": batter_idx % len(lineup),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bases_str(bases):
    labels = []
    if bases[0]: labels.append("1B")
    if bases[1]: labels.append("2B")
    if bases[2]: labels.append("3B")
    return ", ".join(labels) if labels else "empty"


def print_at_bat(ab):
    """Pretty-print a simulated at-bat."""
    for p in ab["pitches"]:
        count = f"{p['balls']}-{p['strikes']}"
        line = (f"  {p['number']:2d}. [{count}] {p['type']:3s} "
                f"{p['velocity']:5.1f} mph  "
                f"({p['plate_x']:+.2f}, {p['plate_z']:.2f})  "
                f"-> {p['outcome']}")
        if "in_play_outcome" in p:
            line += (f" -> {p['in_play_outcome']}"
                     f" ({p['launch_speed']:.0f} mph, {p['launch_angle']:.0f}deg)")
        print(line)
    print(f"  Result: {ab['result'].upper()}")


def print_inning(inning):
    """Pretty-print a simulated inning."""
    bases = [0, 0, 0]
    outs = 0
    for i, ab in enumerate(inning["at_bats"]):
        outs_after = outs + ab["outs_added"]
        name = ab.get("batter_name", str(ab["batter_id"]))
        print(f"  {i+1}. {name:<18s} ({ab['batter_hand']}HB): "
              f"{ab['result'].upper():16s} "
              f"({ab['pitches']} pitches)  "
              f"{ab['runs_scored']}R  "
              f"{outs_after} out  "
              f"bases: {_bases_str(ab['bases_after'])}")
        outs = outs_after
    print(f"  --- {inning['runs']} run(s), {inning['total_pitches']} pitches ---")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    pitcher_id = 642547   # Freddy Peralta (RHP)

    lineup = [
        {"batter_id": 605141, "batter_hand": "R", "name": "Mookie Betts"},
        {"batter_id": 660271, "batter_hand": "L", "name": "Shohei Ohtani"},
        {"batter_id": 518692, "batter_hand": "L", "name": "Freddie Freeman"},
        {"batter_id": 592450, "batter_hand": "R", "name": "Aaron Judge"},
        {"batter_id": 605141, "batter_hand": "R", "name": "Mookie Betts"},
        {"batter_id": 660271, "batter_hand": "L", "name": "Shohei Ohtani"},
        {"batter_id": 518692, "batter_hand": "L", "name": "Freddie Freeman"},
        {"batter_id": 592450, "batter_hand": "R", "name": "Aaron Judge"},
        {"batter_id": 592273, "batter_hand": "R", "name": "Brandon Drury"},
    ]

    print("=" * 65)
    print("Simulating 5 innings: Peralta vs Dodgers lineup")
    print("=" * 65)

    total_runs = 0
    total_pitches = 0
    lineup_pos = 0
    all_results = []

    for inn in range(1, 6):
        print(f"\n--- Inning {inn} ---")
        result = simulate_inning(
            pitcher_id=pitcher_id,
            pitcher_hand="R",
            lineup=lineup[lineup_pos:] + lineup[:lineup_pos],  # rotate
            game_pitch_count=total_pitches,
            score_diff=0 - total_runs,  # pitcher's team tied at 0
            prediction_date="2025-07-15",
        )
        print_inning(result)
        total_runs += result["runs"]
        total_pitches += result["total_pitches"]
        lineup_pos = (lineup_pos + result["lineup_position"]) % len(lineup)

        for ab in result["at_bats"]:
            all_results.append(ab["result"])

    print(f"\n{'=' * 65}")
    print(f"Totals: {total_runs} runs, {total_pitches} pitches over 5 innings")
    print(f"\nAt-bat outcomes:")
    for outcome, count in Counter(all_results).most_common():
        print(f"  {outcome:16s}: {count}")
