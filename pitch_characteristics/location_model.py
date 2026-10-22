import torch
import torch.nn as nn


class PitchLocationLSTM(nn.Module):
    """LSTM that predicts plate_x, plate_z given at-bat history and context.

    Each timestep represents a previous pitch in the at-bat. The LSTM processes
    the sequence, then its hidden state is combined with static context
    (pitcher, batter, current pitch type, count, handedness, batter zone heat map)
    to predict the location of the next pitch.
    """

    def __init__(self, num_pitchers, num_batters, num_pitch_types, num_outcomes,
                 hidden_size=32, dropout=0.2):
        super().__init__()

        # Embeddings for sequence elements (previous pitches)
        self.seq_pitch_type_emb = nn.Embedding(num_pitch_types, 8)
        self.seq_outcome_emb = nn.Embedding(num_outcomes, 4)
        # Per-timestep input: 8 pitch_type + 2 location + 4 outcome = 14
        self.lstm = nn.LSTM(input_size=14, hidden_size=hidden_size,
                            batch_first=True, dropout=0.0)  # single layer, no internal dropout

        # Static context embeddings
        self.pitcher_emb = nn.Embedding(num_pitchers, 16)
        self.batter_emb = nn.Embedding(num_batters, 16)
        self.current_pitch_type_emb = nn.Embedding(num_pitch_types, 8)

        # hidden_size + 16 pitcher + 16 batter + 8 current_pt + 2 count
        # + 2 handedness + 9 batter_zone_ba = hidden + 53
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 53, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # plate_x, plate_z
        )

        self.hidden_size = hidden_size

    def forward(self, seq_pitch_types, seq_locations, seq_outcomes, seq_lengths,
                pitcher_ids, batter_ids, current_pitch_type,
                balls, strikes, batter_hand, pitcher_hand, batter_zone_ba):
        """
        Parameters
        ----------
        seq_pitch_types : (batch, max_seq_len) long
        seq_locations   : (batch, max_seq_len, 2) float
        seq_outcomes    : (batch, max_seq_len) long
        seq_lengths     : (batch,) long
        pitcher_ids     : (batch,) long
        batter_ids      : (batch,) long
        current_pitch_type : (batch,) long
        balls, strikes  : (batch,) float
        batter_hand, pitcher_hand : (batch,) float
        batter_zone_ba  : (batch, 9) float  BA in each of the 9 strike zone regions

        Returns
        -------
        (batch, 2) float  predicted plate_x, plate_z
        """
        # Encode sequence
        pt_emb = self.seq_pitch_type_emb(seq_pitch_types)   # (B, L, 8)
        out_emb = self.seq_outcome_emb(seq_outcomes)         # (B, L, 4)
        seq_input = torch.cat([pt_emb, seq_locations, out_emb], dim=2)  # (B, L, 14)

        # Pack for variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_input, seq_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        lstm_out = h_n.squeeze(0)  # (B, hidden_size)

        # For sequences of length 0 (first pitch of at-bat), use zeros
        zero_mask = (seq_lengths == 0).unsqueeze(1).expand_as(lstm_out)
        lstm_out = lstm_out.masked_fill(zero_mask, 0.0)

        # Static context
        p_emb = self.pitcher_emb(pitcher_ids)
        b_emb = self.batter_emb(batter_ids)
        cpt_emb = self.current_pitch_type_emb(current_pitch_type)
        count = torch.stack([balls, strikes], dim=1)
        hand = torch.stack([batter_hand, pitcher_hand], dim=1)

        combined = torch.cat([
            lstm_out, p_emb, b_emb, cpt_emb, count, hand, batter_zone_ba,
        ], dim=1)
        combined = self.dropout(combined)
        return self.head(combined)
