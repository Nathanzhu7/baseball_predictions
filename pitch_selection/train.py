"""Train the pitch-selection model.

Usage:
    python -m pitch_selection.train
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import ARTIFACTS_DIR
from .prepare_data import prepare_data
from .model import PitchSelectionModel
from .evaluation import (
    compute_loss_and_accuracy,
    compute_baseline_accuracy,
    compute_baseline_loss,
)

MODEL_PATH = ARTIFACTS_DIR / "pitch_selection_model.pt"

_TENSOR_KEYS = [
    ("pitchers", torch.long), ("batters", torch.long),
    ("balls", torch.long), ("strikes", torch.long),
    ("prev_pitch", torch.long),
    ("game_pitch_count", torch.float32),
    ("batter_hand", torch.float32), ("pitcher_hand", torch.float32),
    ("runner_1b", torch.float32), ("runner_2b", torch.float32),
    ("runner_3b", torch.float32),
    ("outs", torch.float32), ("score_diff", torch.float32),
    ("labels", torch.long),
]


def make_loader(data, shuffle=False):
    tensors = [torch.tensor(data[k], dtype=dt) for k, dt in _TENSOR_KEYS]
    return DataLoader(TensorDataset(*tensors), batch_size=512, shuffle=shuffle)


def _unpack(batch):
    """Unpack batch into (model_inputs..., labels)."""
    return batch[:-1], batch[-1]


def main(max_epochs=15, patience=3, final_eval=False):
    torch.manual_seed(42)
    train, val, test, num_pitchers, num_batters, prep = prepare_data()
    pitcher_freq = prep["pitcher_freq"]

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val)

    model = PitchSelectionModel(
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        num_prev_labels=len(prep["prev_labels"]),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, labels = _unpack(batch)
            optimizer.zero_grad()
            logits = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = compute_loss_and_accuracy(model, val_loader)
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Acc = {val_acc:.4f}"
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
    val_loss, val_acc = compute_loss_and_accuracy(model, val_loader)

    base_acc = compute_baseline_accuracy(val, pitcher_freq)
    base_loss = compute_baseline_loss(val, pitcher_freq)

    print(f"\nBest epoch = {best_epoch}")
    print()
    print(f"{'Metric':<28s} {'Baseline':>12s}  {'Our Model':>12s}")
    print("-" * 56)
    print(f"{'Validation Loss':<28s} {base_loss:12.4f}  {val_loss:12.4f}")
    print(f"{'Top-1 Accuracy':<28s} {base_acc:12.4f}  {val_acc:12.4f}")

    if final_eval:
        test_loader = make_loader(test)
        test_loss, test_acc = compute_loss_and_accuracy(model, test_loader)
        test_base_acc = compute_baseline_accuracy(test, pitcher_freq)
        test_base_loss = compute_baseline_loss(test, pitcher_freq)
        print(f"\n{'Test Loss':<28s} {test_base_loss:12.4f}  {test_loss:12.4f}")
        print(f"{'Test Accuracy':<28s} {test_base_acc:12.4f}  {test_acc:12.4f}")

    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()
