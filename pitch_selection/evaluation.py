"""Evaluation utilities for the pitch-selection model."""

import numpy as np
import torch
import torch.nn as nn


def compute_loss_and_accuracy(model, loader):
    """Compute cross-entropy loss and top-1 accuracy on a data loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[:-1], batch[-1]
            logits = model(*inputs)
            total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def compute_baseline_accuracy(dataset, pitcher_freq):
    """Accuracy of always predicting the pitcher's most frequent pitch type."""
    pitchers = dataset["pitchers"]
    labels = dataset["labels"]
    baseline_preds = pitcher_freq[pitchers].argmax(axis=1)
    return (baseline_preds == labels).mean()


def compute_baseline_loss(dataset, pitcher_freq):
    """Cross-entropy loss of the pitcher frequency baseline."""
    pitchers = dataset["pitchers"]
    labels = dataset["labels"]
    probs = pitcher_freq[pitchers]
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    log_probs = np.log(probs)
    losses = -log_probs[np.arange(len(labels)), labels]
    return losses.mean()
