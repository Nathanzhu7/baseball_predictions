import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import ARTIFACTS_DIR
from .prepare_data import prepare_data, OUTCOMES
from .evaluation import compute_loss_and_accuracy, per_class_calibration, plot_calibration
from .model import PitchOutcomeModel

MODEL_PATH = ARTIFACTS_DIR / "pitch_model.pt"
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.png"


def make_loader(data, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(data["pitchers"], dtype=torch.long),
        torch.tensor(data["batters"], dtype=torch.long),
        torch.tensor(data["features"], dtype=torch.float32),
        torch.tensor(data["labels"], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=256, shuffle=shuffle)


def main(max_epochs=10, patience=3, model_path=MODEL_PATH, final_eval=False):
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    torch.manual_seed(42)
    train, validation, test, num_batters, num_pitchers = prepare_data()
    train_loader = make_loader(train, shuffle=True)
    validation_loader = make_loader(validation)

    model = PitchOutcomeModel(
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        num_features=train["features"].shape[1],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for pitchers, batters, features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(pitchers, batters, features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = compute_loss_and_accuracy(model, validation_loader)
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Acc = {val_acc:.4f}"
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

    if best_epoch is None:
        raise RuntimeError("No finite validation loss; no model was selected.")

    # Reload best checkpoint for final evaluation and calibration.
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    val_loss, val_acc = compute_loss_and_accuracy(model, validation_loader)
    print(
        f"Best epoch = {best_epoch}, "
        f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"
    )

    if final_eval:
        test_loss, test_acc = compute_loss_and_accuracy(model, make_loader(test))
        print(f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")

    calibration = per_class_calibration(model, validation_loader, OUTCOMES)
    plot_calibration(calibration, save_path=CALIBRATION_PATH)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
