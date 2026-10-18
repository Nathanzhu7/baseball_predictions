
import torch
import torch.nn as nn


class PitchOutcomeModel(nn.Module):
    def __init__(self, num_pitchers, num_batters, num_features):
        super().__init__()

        self.pitcher_embedding = nn.Embedding(num_pitchers, 16)
        self.batter_embedding = nn.Embedding(num_batters, 16)

        self.network = nn.Sequential(
            nn.Linear(32 + num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, pitcher_ids, batter_ids, features):
        pitcher = self.pitcher_embedding(pitcher_ids)
        batter = self.batter_embedding(batter_ids)

        x = torch.cat([pitcher, batter, features], dim=1)

        return self.network(x)
