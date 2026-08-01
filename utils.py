"""
utils.py
--------
Shared utilities:
  • EarlyStopping      – patience-based training guard
  • MetricsTracker     – accumulates per-epoch metrics
  • save_checkpoint    – saves model + optimiser state
  • load_checkpoint    – restores from checkpoint
  • plot_training      – accuracy & loss curves
  • count_parameters   – model size summary
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless (no display needed)
import matplotlib.pyplot as plt

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training if validation loss does not improve for `patience` epochs.

    Args
    ────
    patience  : epochs to wait before stopping
    min_delta : minimum improvement considered significant
    verbose   : print a message every time the counter increments
    """

    def __init__(self, patience: int = config.EARLY_STOP_PATIENCE,
                 min_delta: float = 1e-4, verbose: bool = True):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Accumulates loss and correct predictions over an epoch, then computes
    epoch-level averages.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._loss_sum   = 0.0
        self._correct    = 0
        self._total      = 0
        self._num_batches = 0

    def update(self, loss: float, correct: int, total: int):
        self._loss_sum    += loss
        self._correct     += correct
        self._total       += total
        self._num_batches += 1

    @property
    def avg_loss(self) -> float:
        return self._loss_sum / max(self._num_batches, 1)

    @property
    def accuracy(self) -> float:
        return self._correct / max(self._total, 1) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, filepath: str):
    """Saves training state (model, optimiser, epoch, metrics)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"  [Checkpoint] Saved → {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                    optimizer=None, device=config.DEVICE):
    """
    Loads model (and optionally optimiser) weights from checkpoint.

    Returns the checkpoint dict so callers can read epoch / metrics.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  [Checkpoint] Loaded ← {filepath}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history: dict, model_name: str, save_dir: str = config.RESULTS_DIR):
    """
    Plots train/val loss and accuracy curves and saves them as a PNG.

    Args
    ────
    history    : {"train_loss": [...], "val_loss": [...],
                  "train_acc": [...],  "val_acc":  [...]}
    model_name : used in filename and plot title
    save_dir   : directory to write the PNG
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val loss",   markersize=4)
    axes[0].set_title(f"{model_name} – Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy ───────────────────────────────────────────────────────────
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val acc",   markersize=4)
    axes[1].set_title(f"{model_name} – Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_training.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Plot] Saved → {fname}")


def plot_comparison(results: dict, save_dir: str = config.RESULTS_DIR):
    """
    Bar-chart comparing test accuracy across all four models.

    Args
    ────
    results : {"ModelName": {"accuracy": float, ...}, ...}
    """
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison – PlantVillage Test Set")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    fname = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Plot] Comparison chart saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
#  Model info
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> dict:
    """Returns trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def format_time(seconds: float) -> str:
    """Convert seconds to mm:ss string."""
    m = int(seconds // 60)
    s = int(seconds  % 60)
    return f"{m:02d}:{s:02d}"


def print_model_summary(model_name: str, model: torch.nn.Module):
    params = count_parameters(model)
    print(f"\n{'─'*55}")
    print(f"  Model      : {model_name}")
    print(f"  Trainable  : {params['trainable']:,} parameters")
    print(f"  Total      : {params['total']:,} parameters")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """
    Writes training logs to both stdout and a text file.
    """

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, "w")

    def log(self, msg: str):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
