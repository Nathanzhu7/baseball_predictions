"""Train the joint pitch generation model.

Usage:
    python -m pitch_generation.train
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import ARTIFACTS_DIR
from .prepare_data import prepare_data, NUM_PT
from .model import PitchGenerationModel

MODEL_PATH = ARTIFACTS_DIR / "pitch_generation_model.pt"

_KEYS = [
    ("seq_pt", torch.long), ("seq_loc", torch.float32), ("seq_out", torch.long),
    ("seq_len", torch.long),
    ("pitcher", torch.long), ("batter", torch.long),
    ("balls", torch.float32), ("strikes", torch.float32),
    ("batter_hand", torch.float32), ("pitcher_hand", torch.float32),
    ("batter_zone_ba", torch.float32), ("pitch_num", torch.float32),
    ("runner_1b", torch.float32), ("runner_2b", torch.float32),
    ("runner_3b", torch.float32), ("outs", torch.float32),
    ("score_diff", torch.float32), ("game_pitch_count", torch.float32),
    # targets last
    ("target_chars", torch.float32), ("target_pt", torch.long),
]


def make_loader(data, shuffle=False, batch_size=512):
    tensors = [torch.tensor(data[k], dtype=dt) for k, dt in _KEYS]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _unpack(batch):
    """Split batch into model inputs, target characteristics, target pitch type."""
    return batch[:18], batch[18], batch[19]


def evaluate(model, loader, repertoire):
    model.eval()
    total_type_loss = 0.0
    total_char_loss = 0.0
    correct = 0
    n = 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            inputs, target_chars, target_pt = _unpack(batch)
            type_logits, characteristics = model(*inputs)

            total_type_loss += ce(type_logits, target_pt).item()

            pitcher_ids = inputs[4]
            mask = torch.tensor(repertoire[pitcher_ids.numpy()], dtype=torch.float32)
            masked_logits = type_logits + (1 - mask) * (-1e9)
            correct += (masked_logits.argmax(dim=1) == target_pt).sum().item()

            pred_chars = characteristics[torch.arange(len(target_pt)), target_pt]
            total_char_loss += mse(pred_chars, target_chars).item()
            n += len(target_pt)

    return total_type_loss / n, total_char_loss / n, correct / n


def main(max_epochs=40, patience=5, final_eval=False):
    torch.manual_seed(42)
    train, val, test, num_pitchers, num_batters, prep = prepare_data()
    repertoire = prep["repertoire"]
    pitcher_freq = prep["pitcher_freq"]

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val)

    model = PitchGenerationModel(
        num_pitchers=num_pitchers,
        num_batters=num_batters,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Balance the two losses: CE ~1.5, MSE ~1.0, so roughly equal already
    loc_weight = 1.0

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            inputs, target_chars, target_pt = _unpack(batch)
            optimizer.zero_grad()

            type_logits, characteristics = model(*inputs)

            # Type loss
            loss_type = ce_loss(type_logits, target_pt)

            # Characteristics loss: for the actual pitch type thrown
            pred_chars = characteristics[torch.arange(len(target_pt)), target_pt]
            loss_chars = mse_loss(pred_chars, target_chars)

            loss = loss_type + loc_weight * loss_chars
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target_pt)
            n += len(target_pt)

        train_loss = total_loss / n
        val_type_loss, val_char_loss, val_acc = evaluate(model, val_loader, repertoire)
        val_total = val_type_loss + loc_weight * val_char_loss

        print(
            f"Epoch {epoch + 1}: "
            f"Train = {train_loss:.4f}, "
            f"Val Type CE = {val_type_loss:.4f}, "
            f"Val Char MSE = {val_char_loss:.4f}, "
            f"Val Acc = {val_acc:.4f}"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    val_type_loss, val_char_loss, val_acc = evaluate(model, val_loader, repertoire)

    # Baseline: pitcher frequency for type
    base_acc = (pitcher_freq[val["pitcher"]].argmax(axis=1) == val["target_pt"]).mean()
    base_type_loss = -np.log(
        np.clip(pitcher_freq[val["pitcher"]][np.arange(len(val["target_pt"])), val["target_pt"]], 1e-8, 1.0)
    ).mean()

    print(f"\nBest epoch = {best_epoch}")
    print()
    print(f"{'Metric':<28s} {'Baseline':>12s}  {'Joint Model':>12s}")
    print("-" * 56)
    print(f"{'Type CE Loss':<28s} {base_type_loss:12.4f}  {val_type_loss:12.4f}")
    print(f"{'Type Top-1 Accuracy':<28s} {base_acc:12.4f}  {val_acc:12.4f}")
    print(f"{'Characteristics MSE':<28s} {'---':>12s}  {val_char_loss:12.4f}")
    print(f"\nSaved {MODEL_PATH}")

    # Check pitch type embeddings
    _check_embeddings(model)


def _check_embeddings(model):
    """Print similarity between pitch type embeddings."""
    emb = model.seq_pitch_type_emb.weight.detach()
    from .prepare_data import PITCH_TYPES

    # Compute cosine similarity
    norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = emb / norms
    sim = (normed @ normed.T).numpy()

    print(f"\nPitch type embedding similarities (cosine):")
    # Show a few interesting pairs
    pairs = [
        ("FF", "SI", "both fastballs"),
        ("SL", "ST", "slider family"),
        ("SL", "CU", "both breaking"),
        ("FF", "CH", "fastball vs changeup"),
        ("FF", "CU", "fastball vs curveball"),
        ("CH", "FS", "both offspeed"),
    ]
    for pt1, pt2, note in pairs:
        i1 = PITCH_TYPES.index(pt1)
        i2 = PITCH_TYPES.index(pt2)
        print(f"  {pt1}-{pt2}: {sim[i1, i2]:+.3f}  ({note})")


if __name__ == "__main__":
    main()
