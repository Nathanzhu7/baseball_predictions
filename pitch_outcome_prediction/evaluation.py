import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def compute_loss_and_accuracy(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for pitchers, batters, features, labels in loader:
            logits = model(pitchers, batters, features)
            total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def per_class_calibration(model, loader, class_names, n_bins=10):
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for pitchers, batters, features, labels in loader:
            logits = model(pitchers, batters, features)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    calibration = {}
    for cls_idx, cls_name in enumerate(class_names):
        probs_cls = all_probs[:, cls_idx]
        true_cls = (all_labels == cls_idx).astype(float)

        bin_means = []
        bin_true_freqs = []
        bin_counts = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if hi == bin_edges[-1]:
                mask = (probs_cls >= lo) & (probs_cls <= hi)
            else:
                mask = (probs_cls >= lo) & (probs_cls < hi)
            count = mask.sum()
            bin_counts.append(int(count))
            if count > 0:
                bin_means.append(probs_cls[mask].mean())
                bin_true_freqs.append(true_cls[mask].mean())
            else:
                bin_means.append(np.nan)
                bin_true_freqs.append(np.nan)

        calibration[cls_name] = {
            "bin_edges": bin_edges,
            "mean_predicted": np.array(bin_means),
            "true_frequency": np.array(bin_true_freqs),
            "bin_counts": np.array(bin_counts),
        }

    return calibration


def plot_calibration(calibration, save_path=None):
    n_classes = len(calibration)
    cols = min(n_classes, 3)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes)

    for idx, (cls_name, data) in enumerate(calibration.items()):
        ax = axes[idx // cols, idx % cols]
        mean_pred = data["mean_predicted"]
        true_freq = data["true_frequency"]
        counts = data["bin_counts"]

        valid = ~np.isnan(mean_pred)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(mean_pred[valid], true_freq[valid], "o-", markersize=5, label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(cls_name)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        bin_edges = data["bin_edges"]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.bar(bin_centers, counts, width=0.08, alpha=0.2, color="gray", label="Samples")
        ax2.set_ylabel("Count", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)

    for idx in range(n_classes, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Per-class calibration", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Calibration plot saved to {save_path}")
    plt.close(fig)
