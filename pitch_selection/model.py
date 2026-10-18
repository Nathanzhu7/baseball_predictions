import torch
import torch.nn as nn

from .prepare_data import NUM_CLASSES


class PitchSelectionModel(nn.Module):
    def __init__(self, num_pitchers, num_batters, num_prev_labels):
        super().__init__()
        self.pitcher_embedding = nn.Embedding(num_pitchers, 16)
        self.batter_embedding = nn.Embedding(num_batters, 16)
        self.prev_pitch_embedding = nn.Embedding(num_prev_labels, 8)

        # 16 pitcher + 16 batter + 8 prev_pitch + 2 count + 1 fatigue
        # + 2 handedness + 3 runners + 1 outs + 1 score_diff = 50
        self.network = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, pitchers, batters, balls, strikes, prev_pitch,
                game_pitch_count, batter_hand, pitcher_hand,
                runner_1b, runner_2b, runner_3b, outs, score_diff):
        p = self.pitcher_embedding(pitchers)
        b = self.batter_embedding(batters)
        pp = self.prev_pitch_embedding(prev_pitch)

        count = torch.stack([balls.float(), strikes.float()], dim=1)
        context = torch.stack([
            game_pitch_count, batter_hand, pitcher_hand,
            runner_1b, runner_2b, runner_3b, outs, score_diff,
        ], dim=1)

        x = torch.cat([p, b, pp, count, context], dim=1)
        return self.network(x)
