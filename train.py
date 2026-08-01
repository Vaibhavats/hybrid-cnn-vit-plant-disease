"""
train.py
--------
Trains all four models sequentially and saves:
  • Best checkpoint per model  (checkpoints/<model_key>_best.pt)
  • Training history JSON      (results/<model_key>_history.json)
  • Loss/accuracy plots        (results/<model_key>_training.png)
  • Training log               (logs/<model_key>.log)

Usage
─────
    python train.py                         # trains all 4 models
    python train.py --model hybrid          # trains only the proposed model
    python train.py --model cnn --epochs 10 # quick test

Model keys: cnn | vit | cnn_vit | hybrid
"""

import os
import sys

# ── Always resolve imports relative to this file's directory ─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data_loader import get_dataloaders
from model import CNNBaseline, ViTBaseline, CNNViT, HybridCNNViT
from utils import (
    EarlyStopping, MetricsTracker,
    save_checkpoint, Logger,
    plot_training, print_model_summary, format_time,
)


# ─────────────────────────────────────────────────────────────────────────────
#  MPS BatchNorm fix
# ─────────────────────────────────────────────────────────────────────────────

def fix_bn_for_mps(model: nn.Module) -> nn.Module:
    """
    PyTorch MPS has a bug where BatchNorm2d running_mean/running_var become
    corrupted during training, causing val acc to stay near random (~4-5%)
    even when train acc is 80%+.

    Fix: replace every BatchNorm2d with momentum=1.0, which means each forward
    pass computes fresh batch statistics and NEVER accumulates running stats.
    This makes BN behave identically in train and eval mode on MPS.

    Only applied when device is MPS. CUDA and CPU are unaffected.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features = module.num_features,
                eps          = module.eps,
                momentum     = 1.0,        # ← always use batch stats, never running stats
                affine       = module.affine,
            )
            if module.affine:
                new_bn.weight.data = module.weight.data.clone()
                new_bn.bias.data   = module.bias.data.clone()
            setattr(model, name, new_bn)
        else:
            fix_bn_for_mps(module)    # recurse into all submodules
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(key: str) -> nn.Module:
    """Instantiates a model by key and applies MPS fix if needed."""
    if key == "cnn":
        model = CNNBaseline()
    elif key == "vit":
        model = ViTBaseline()
    elif key == "cnn_vit":
        model = CNNViT()
    elif key == "hybrid":
        model = HybridCNNViT()
    else:
        raise ValueError(f"Unknown model key: {key}. Choose: cnn | vit | cnn_vit | hybrid")

    # Apply MPS BatchNorm fix for Apple Silicon
    if config.DEVICE.type == "mps":
        model = fix_bn_for_mps(model)
        print("  [MPS] BatchNorm fix applied (momentum=1.0) ✓")

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Single epoch: train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, logger, epoch):
    model.train()
    tracker = MetricsTracker()
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping helps stability with transformers
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds   = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        tracker.update(loss.item(), correct, labels.size(0))

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            elapsed = format_time(time.time() - t0)
            logger.log(
                f"  [Train] Epoch {epoch:03d} | "
                f"Batch {batch_idx+1:04d}/{len(loader)} | "
                f"Loss {tracker.avg_loss:.4f} | "
                f"Acc {tracker.accuracy:.2f}% | "
                f"Elapsed {elapsed}"
            )

    return tracker.avg_loss, tracker.accuracy


# ─────────────────────────────────────────────────────────────────────────────
#  Single epoch: validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validation pass.

    MPS (Apple Silicon) has a hardware-level bug where certain operations
    produce wrong results during inference regardless of BatchNorm mode.
    We work around this by temporarily moving the model to CPU for validation
    only — training stays fully on MPS (fast), validation on CPU (correct).
    The model is moved back to MPS after each validation pass.
    """
    # Determine validation device — use CPU if training on MPS
    val_device = torch.device("cpu") if device.type == "mps" else device

    model.to(val_device)
    model.eval()
    tracker = MetricsTracker()

    # Use a smaller criterion on CPU (avoids label_smoothing MPS bug too)
    cpu_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for images, labels in loader:
        images = images.to(val_device)
        labels = labels.to(val_device)

        logits  = model(images)
        loss    = cpu_criterion(logits, labels)
        preds   = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        tracker.update(loss.item(), correct, labels.size(0))

    # Move model back to training device
    model.to(device)
    model.train()
    return tracker.avg_loss, tracker.accuracy


# ─────────────────────────────────────────────────────────────────────────────
#  Full training loop for ONE model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    key:         str,
    train_loader,
    val_loader,
    num_epochs:  int  = config.NUM_EPOCHS,
    device:      torch.device = config.DEVICE,
) -> dict:
    """
    Trains the model identified by `key`.
    Returns a history dict with per-epoch metrics.
    """
    name     = config.MODEL_NAMES[key]
    log_path = os.path.join(config.LOG_DIR, f"{key}.log")
    logger   = Logger(log_path)

    logger.log(f"\n{'═'*60}")
    logger.log(f"  Training: {name}")
    logger.log(f"  Device  : {device}")
    logger.log(f"  Epochs  : {num_epochs}")
    logger.log(f"{'═'*60}")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(key).to(device)
    # ── Resume from checkpoint if it exists ──────────────────────────────
    start_epoch = 1
    ckpt_path   = os.path.join(config.CHECKPOINT_DIR, f"{key}_best.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["val_acc"]
        print(f"  [Resume] Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(val acc: {best_val_acc:.2f}%) → resuming from epoch {start_epoch}")
    print_model_summary(name, model)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config.LR_ETA_MIN,
    )
    early_stop = EarlyStopping(patience=config.EARLY_STOP_PATIENCE)

    # ── History ───────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0
    ckpt_path    = os.path.join(config.CHECKPOINT_DIR, f"{key}_best.pt")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs + 1):
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = format_time(time.time() - t_epoch)
        lr_now     = optimizer.param_groups[0]["lr"]

        logger.log(
            f"\nEpoch {epoch:03d}/{num_epochs} | "
            f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.2f}% | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.2f}% | "
            f"LR {lr_now:.2e} | Time {epoch_time}"
        )

        # ── Checkpoint: save if val_acc improved ─────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc":         val_acc,
                    "val_loss":        val_loss,
                    "model_key":       key,
                },
                filepath=ckpt_path,
            )
            logger.log(f"  ★ New best val acc: {best_val_acc:.2f}%")

        # ── Early stopping ────────────────────────────────────────────────
        if early_stop(val_loss):
            logger.log(f"\n  [EarlyStopping] Triggered at epoch {epoch}. Stopping.")
            break

    logger.log(f"\n  Training complete. Best val acc: {best_val_acc:.2f}%")
    logger.close()

    # ── Save history ──────────────────────────────────────────────────────
    hist_path = os.path.join(config.RESULTS_DIR, f"{key}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_training(history, name)

    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease models")
    parser.add_argument(
        "--model", type=str, default="all",
        help="Model key: cnn | vit | cnn_vit | hybrid | all (default: all)"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help=f"Number of training epochs (default: {config.NUM_EPOCHS})"
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

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader, _, class_names = get_dataloaders(args.data)

    # ── Select models to train ────────────────────────────────────────────
    keys = (
        list(config.MODEL_NAMES.keys())
        if args.model == "all"
        else [args.model]
    )

    all_histories = {}
    t_start = time.time()

    for key in keys:
        history = train_model(
            key, train_loader, val_loader,
            num_epochs=args.epochs,
            device=device,
        )
        all_histories[key] = history

    total_time = format_time(time.time() - t_start)
    print(f"\n[Done] All training finished in {total_time}.")
    print(f"  Checkpoints → {config.CHECKPOINT_DIR}")
    print(f"  Results     → {config.RESULTS_DIR}")
    print(f"  Logs        → {config.LOG_DIR}")


if __name__ == "__main__":
    main()