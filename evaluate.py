"""
evaluate.py
-----------
Evaluates all trained models on the test set and generates:
  • Per-model metrics: Accuracy, Precision, Recall, F1-score
  • Confusion matrix PNG per model
  • Comparison table printed to console + saved as JSON
  • Side-by-side bar chart (results/model_comparison.png)

Usage
─────
    python evaluate.py                      # evaluates all 4 models
    python evaluate.py --model hybrid       # evaluate only proposed model
    python evaluate.py --model cnn --data /path/to/PlantVillage
"""

import os
import sys

# ── Always resolve imports relative to this file's directory ─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import config
from data_loader import get_dataloaders
from model import CNNBaseline, ViTBaseline, CNNViT, HybridCNNViT
from utils import load_checkpoint, plot_comparison


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory (same as train.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(key: str) -> nn.Module:
    if key == "cnn":
        return CNNBaseline()
    elif key == "vit":
        return ViTBaseline()
    elif key == "cnn_vit":
        return CNNViT()
    elif key == "hybrid":
        return HybridCNNViT()
    else:
        raise ValueError(f"Unknown model key: {key}")


# ─────────────────────────────────────────────────────────────────────────────
#  Inference pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model: nn.Module, loader, device: torch.device):
    """
    Runs inference over *loader* and collects all predictions + ground truths.

    Returns
    ───────
    y_true : np.ndarray of shape (N,)
    y_pred : np.ndarray of shape (N,)
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
#  Compute metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns a dict with accuracy, macro precision/recall/F1, and
    weighted precision/recall/F1.
    """
    return {
        "accuracy":           accuracy_score(y_true, y_pred) * 100,
        "precision_macro":    precision_score(y_true, y_pred, average="macro",
                                              zero_division=0) * 100,
        "recall_macro":       recall_score(y_true, y_pred, average="macro",
                                           zero_division=0) * 100,
        "f1_macro":           f1_score(y_true, y_pred, average="macro",
                                       zero_division=0) * 100,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted",
                                              zero_division=0) * 100,
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted",
                                           zero_division=0) * 100,
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted",
                                       zero_division=0) * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    class_names: list,
    model_name:  str,
    save_dir:    str = config.RESULTS_DIR,
):
    """
    Saves a confusion matrix heatmap as a PNG.
    For 38 classes the figure is made large enough to be readable.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalise rows so each cell shows recall for that class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        cm_norm,
        annot=False,          # too many cells for text annotations at 38×38
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor="lightgray",
    )
    ax.set_title(f"Confusion Matrix – {model_name}\n(row-normalised recall)",
                 fontsize=14)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()

    fname = os.path.join(
        save_dir,
        f"{model_name.replace(' ', '_')}_confusion_matrix.png"
    )
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Plot] Confusion matrix saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
#  Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    """Prints a formatted table comparing all evaluated models."""
    hdr = (
        f"{'Model':<35} {'Accuracy':>10} {'Precision':>11} "
        f"{'Recall':>9} {'F1':>9}"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print("  MODEL COMPARISON TABLE (macro-averaged metrics)")
    print(sep)
    print(hdr)
    print(sep)
    for name, m in results.items():
        print(
            f"  {name:<33} "
            f"{m['accuracy']:>9.2f}% "
            f"{m['precision_macro']:>10.2f}% "
            f"{m['recall_macro']:>8.2f}% "
            f"{m['f1_macro']:>8.2f}%"
        )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate one model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(key: str, test_loader, class_names: list,
                   device: torch.device) -> dict:
    """
    Loads the best checkpoint for `key`, runs evaluation, plots confusion
    matrix, and returns a metrics dict.
    """
    name      = config.MODEL_NAMES[key]
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{key}_best.pt")

    if not os.path.isfile(ckpt_path):
        print(f"  [Warning] Checkpoint not found for '{key}': {ckpt_path}")
        print(f"  Run: python train.py --model {key}")
        return None

    print(f"\n{'─'*55}")
    print(f"  Evaluating: {name}")

    model = build_model(key).to(device)
    load_checkpoint(ckpt_path, model, device=device)

    y_true, y_pred = get_predictions(model, test_loader, device)
    metrics        = compute_metrics(y_true, y_pred)

    print(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Precision : {metrics['precision_macro']:.2f}% (macro)")
    print(f"  Recall    : {metrics['recall_macro']:.2f}% (macro)")
    print(f"  F1-Score  : {metrics['f1_macro']:.2f}% (macro)")

    # Per-class report
    report_path = os.path.join(config.RESULTS_DIR, f"{key}_classification_report.txt")
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)
    with open(report_path, "w") as f:
        f.write(f"Model: {name}\n\n")
        f.write(report)
    print(f"  [Report] Saved → {report_path}")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, name)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate plant disease models")
    parser.add_argument(
        "--model", type=str, default="all",
        help="Model key: cnn | vit | cnn_vit | hybrid | all (default: all)"
    )
    parser.add_argument(
        "--data", type=str, default=config.DATA_DIR,
        help="Path to PlantVillage dataset root"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = config.DEVICE
    print(f"\n[System] Using device: {device}")

    # DataLoaders (we only need the test loader here)
    _, _, test_loader, class_names = get_dataloaders(args.data)

    keys = (
        list(config.MODEL_NAMES.keys())
        if args.model == "all"
        else [args.model]
    )

    results = {}
    for key in keys:
        metrics = evaluate_model(key, test_loader, class_names, device)
        if metrics is not None:
            results[config.MODEL_NAMES[key]] = metrics

    if not results:
        print("\n[Warning] No models could be evaluated. Train models first.")
        return

    # ── Comparison table ──────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save results JSON ─────────────────────────────────────────────────
    results_path = os.path.join(config.RESULTS_DIR, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Results] Saved → {results_path}")

    # ── Comparison plot ───────────────────────────────────────────────────
    plot_comparison(results)

    print("\n[Done] Evaluation complete.\n")


if __name__ == "__main__":
    main()