"""Joint pitch generation model: predicts pitch type + characteristics together.

Given the at-bat history and game context, outputs:
  1. Probability distribution over pitch types
  2. Predicted (plate_x, plate_z, pfx_x, pfx_z, velocity) for each pitch type

At inference, sample a pitch type from the distribution, then use that
type's predicted characteristics (plus execution noise).
"""

import torch
import torch.nn as nn

from .prepare_data import NUM_PT, NUM_OUTCOMES


class PitchGenerationModel(nn.Module):

    def __init__(self, num_pitchers, num_batters, hidden_size=48, dropout=0.2):
        super().__init__()

        # --- Sequence encoders (previous pitches) ---
        self.seq_pitch_type_emb = nn.Embedding(NUM_PT, 8)
        self.seq_outcome_emb = nn.Embedding(NUM_OUTCOMES, 4)
        # Per-timestep: 8 pitch_type + 2 location + 4 outcome = 14
        self.lstm = nn.LSTM(input_size=14, hidden_size=hidden_size, batch_first=True)

        # --- Static context ---
        self.pitcher_emb = nn.Embedding(num_pitchers, 16)
        self.batter_emb = nn.Embedding(num_batters, 16)

        # Static features: 16 pitcher + 16 batter + 9 zone_ba + 2 count
        # + 2 handedness + 1 pitch_num + 3 runners + 1 outs + 1 score_diff
        # + 1 game_pitch_count = 52
        static_size = 52

        # --- Shared trunk ---
        trunk_input = hidden_size + static_size
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Pitch type head ---
        self.type_head = nn.Linear(96, NUM_PT)

        # --- Characteristics head: per pitch type predict 5 values ---
        # (plate_x, plate_z, pfx_x, pfx_z, release_speed)
        self.char_head = nn.Linear(96, NUM_PT * 5)

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, seq_pitch_types, seq_locations, seq_outcomes, seq_lengths,
                pitcher_ids, batter_ids,
                balls, strikes, batter_hand, pitcher_hand,
                batter_zone_ba, pitch_num,
                runner_1b, runner_2b, runner_3b, outs, score_diff,
                game_pitch_count):
        """
        Returns
        -------
        type_logits    : (batch, NUM_PT)
        characteristics: (batch, NUM_PT, 5)  plate_x, plate_z, pfx_x, pfx_z, velocity per type
        """
        # Encode sequence
        pt_emb = self.seq_pitch_type_emb(seq_pitch_types)
        out_emb = self.seq_outcome_emb(seq_outcomes)
        seq_input = torch.cat([pt_emb, seq_locations, out_emb], dim=2)

        packed = nn.utils.rnn.pack_padded_sequence(
            seq_input, seq_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        lstm_out = h_n.squeeze(0)

        zero_mask = (seq_lengths == 0).unsqueeze(1).expand_as(lstm_out)
        lstm_out = lstm_out.masked_fill(zero_mask, 0.0)

        # Static context
        p_emb = self.pitcher_emb(pitcher_ids)
        b_emb = self.batter_emb(batter_ids)
        count = torch.stack([balls, strikes], dim=1)
        hand = torch.stack([batter_hand, pitcher_hand], dim=1)
        game_state = torch.stack([
            pitch_num, runner_1b, runner_2b, runner_3b, outs, score_diff,
            game_pitch_count,
        ], dim=1)

        static = torch.cat([p_emb, b_emb, batter_zone_ba, count, hand, game_state], dim=1)

        combined = torch.cat([lstm_out, static], dim=1)
        combined = self.dropout(combined)
        trunk_out = self.trunk(combined)

        type_logits = self.type_head(trunk_out)
        char_flat = self.char_head(trunk_out)
        characteristics = char_flat.view(-1, NUM_PT, 5)

        return type_logits, characteristics
