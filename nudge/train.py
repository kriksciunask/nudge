"""Training helpers shared by every model, written like the lecturer's notebooks
(chapter5_mlp_student.ipynb c13-c21, chapter6_builders_guide_student.ipynb c62-c71).

Differences from the course notebooks, and why (see docs/course-map.md section 4):
- One output + sigmoid + binary cross-entropy (Week 3 slides 16-18) instead of 2 outputs + CrossEntropyLoss,
  so every model gives one number: the chance the shopper comes back and buys.
- We track validation loss and PR-AUC instead of accuracy: only 8.6% of visits are positive, so
  "always no" is already 91.4% accurate.
"""

import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nudge.contract import TOKEN_TO_ID, VOCAB

SEED = 42
BLOCK_SIZE = 64          # keep the last 64 words of a visit (99.99% of visits are shorter)
VOCAB_SIZE = len(VOCAB)  # 63 including <pad> = 0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def try_gpu() -> torch.device:
    """Same idea as builder's guide c62: use the Apple GPU (mps) if there is one, else the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def visits_to_ids(token_lists) -> np.ndarray:
    """Each visit -> a row of BLOCK_SIZE word ids. Long visits keep their last words; short ones are
    padded with 0 (<pad>) at the end. The first word ("new"/"returning") is always kept."""
    ids = np.zeros((len(token_lists), BLOCK_SIZE), dtype=np.int64)
    for row, words in enumerate(token_lists):
        if len(words) > BLOCK_SIZE:
            words = [words[0]] + list(words[-(BLOCK_SIZE - 1):])
        ids[row, : len(words)] = [TOKEN_TO_ID[w] for w in words]
    return ids


def word_counts(ids: np.ndarray) -> np.ndarray:
    """Bag of words: how often each word (except <pad>) appears in each visit. Order is thrown away."""
    counts = np.zeros((len(ids), VOCAB_SIZE), dtype=np.float32)
    rows = np.repeat(np.arange(len(ids)), ids.shape[1])
    np.add.at(counts, (rows, ids.ravel()), 1.0)
    return counts[:, 1:]


def make_loader(features, targets, batch_size=1024, shuffle=False):
    """As in chapter5_mlp_student.ipynb c13, with targets as float for binary cross-entropy."""
    feature_tensor = torch.as_tensor(features)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model, loader, device=None):
    """Chance to come back and buy for every row in the loader."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for features, _ in loader:
        out.append(torch.sigmoid(model(features.to(device))).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def evaluate(model, loader, loss_function):
    """Average loss and PR-AUC over a loader (chapter5 c19 returns loss and accuracy; we use PR-AUC)."""
    device = next(model.parameters()).device
    model.eval()
    total_loss, total_rows, probs, labels = 0.0, 0, [], []
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        total_loss += loss_function(logits, targets).item() * len(targets)
        total_rows += len(targets)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(targets.cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return total_loss / total_rows, average_precision_score(labels, probs)


def train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=5, print_every=1):
    """The course training loop (chapter5 c21): forward pass, loss, backward pass, optimizer step.
    Keeps the weights of the epoch with the lowest validation loss (chapter5 c22, workshop c16)."""
    device = next(model.parameters()).device
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        running, rows = 0.0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(targets)
            rows += len(targets)
        val_loss, val_pr_auc = evaluate(model, val_loader, loss_function)
        history["train_loss"].append(running / rows)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1:2d}/{epochs} | train loss {running / rows:.4f} | "
                  f"val loss {val_loss:.4f} | val PR-AUC {val_pr_auc:.3f}")
    model.load_state_dict(best_state)
    return history


def plot_history(history, title=""):
    """Two-panel learning curves, as in chapter5 c25 / builder's guide c71."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy loss", title=f"{title} loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_pr_auc"], color="tab:orange")
    axes[1].set(xlabel="Epoch", ylabel="PR-AUC", title=f"{title} validation PR-AUC")
    plt.tight_layout()
    plt.show()
