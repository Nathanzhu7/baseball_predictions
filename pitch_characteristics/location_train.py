"""Train the LSTM pitch location model.

Usage:
    python -m pitch_characteristics.location_train
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import ARTIFACTS_DIR
from .location_prepare_data import prepare_data, NUM_PT, NUM_OUTCOMES
from .location_model import PitchLocationLSTM

MODEL_PATH = ARTIFACTS_DIR / "location_model.pt"


def make_loader(data, shuffle=False, batch_size=512):
    ds = TensorDataset(
        torch.tensor(data["seq_pt"], dtype=torch.long),
        torch.tensor(data["seq_loc"], dtype=torch.float32),
        torch.tensor(data["seq_out"], dtype=torch.long),
        torch.tensor(data["seq_len"], dtype=torch.long),
        torch.tensor(data["pitcher"], dtype=torch.long),
        torch.tensor(data["batter"], dtype=torch.long),
        torch.tensor(data["cur_pt"], dtype=torch.long),
        torch.tensor(data["balls"], dtype=torch.float32),
        torch.tensor(data["strikes"], dtype=torch.float32),
        torch.tensor(data["batter_hand"], dtype=torch.float32),
        torch.tensor(data["pitcher_hand"], dtype=torch.float32),
        torch.tensor(data["batter_zone_ba"], dtype=torch.float32),
        torch.tensor(data["target"], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _unpack(batch):
    return batch[:12], batch[12]


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_x_err = 0.0
    total_z_err = 0.0
    n = 0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            inputs, target = _unpack(batch)
            pred = model(*inputs)
            total_loss += criterion(pred, target).item()
            total_x_err += (pred[:, 0] - target[:, 0]).abs().sum().item()
            total_z_err += (pred[:, 1] - target[:, 1]).abs().sum().item()
            n += len(target)
    return total_loss / n, total_x_err / n, total_z_err / n


def main(max_epochs=20, patience=3, final_eval=False):
    torch.manual_seed(42)
    train, val, test, num_pitchers, num_batters, prep = prepare_data()

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val)

    model = PitchLocationLSTM(
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        num_pitch_types=NUM_PT,
        num_outcomes=NUM_OUTCOMES,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            inputs, target = _unpack(batch)
            optimizer.zero_grad()
            pred = model(*inputs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            n += len(target)

        train_loss = total_loss / n
        val_loss, val_x_mae, val_z_mae = evaluate(model, val_loader)
        print(
            f"Epoch {epoch + 1}: "
            f"Train MSE = {train_loss:.4f}, "
            f"Val MSE = {val_loss:.4f}, "
            f"Val plate_x MAE = {val_x_mae:.3f}, "
            f"Val plate_z MAE = {val_z_mae:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    val_loss, val_x_mae, val_z_mae = evaluate(model, val_loader)

    # Baseline: predict training mean for all pitches
    import numpy as np
    train_mean_x = np.mean(train["target"][:, 0])
    train_mean_z = np.mean(train["target"][:, 1])
    base_x_mae = np.abs(val["target"][:, 0] - train_mean_x).mean()
    base_z_mae = np.abs(val["target"][:, 1] - train_mean_z).mean()

    print(f"\nBest epoch = {best_epoch}")
    print()
    print(f"{'Metric':<28s} {'Baseline':>12s}  {'LSTM':>12s}")
    print("-" * 56)
    print(f"{'plate_x MAE (ft)':<28s} {base_x_mae:12.3f}  {val_x_mae:12.3f}")
    print(f"{'plate_z MAE (ft)':<28s} {base_z_mae:12.3f}  {val_z_mae:12.3f}")

    if final_eval:
        test_loader = make_loader(test)
        _, test_x_mae, test_z_mae = evaluate(model, test_loader)
        print(f"\n{'Test plate_x MAE':<28s} {'---':>12s}  {test_x_mae:12.3f}")
        print(f"{'Test plate_z MAE':<28s} {'---':>12s}  {test_z_mae:12.3f}")

    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()
