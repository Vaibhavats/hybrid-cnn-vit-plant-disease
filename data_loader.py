"""
data_loader.py
--------------
Handles PlantVillage dataset loading, train/val/test splitting,
preprocessing, and augmentation.

Expected folder layout
----------------------
data/PlantVillage/
    Apple___Apple_scab/           ← one folder per class
        image1.jpg
        image2.jpg
        ...
    Apple___Black_rot/
        ...
    ...                           ← 38 class folders total

You can download PlantVillage from:
  https://www.kaggle.com/datasets/emmarex/plantdisease
  or: https://github.com/spMohanty/PlantVillage-Dataset
"""

import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training:
      - Random resized crop (mimics scale variation in field photos)
      - Horizontal & vertical flips
      - Random rotation ±30°
      - Colour jitter (brightness, contrast, saturation)
      - Normalise with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for validation & test (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(data_dir: str = config.DATA_DIR):
    """
    Load PlantVillage from *data_dir* and return (train, val, test) Subsets.

    The full dataset is first loaded once with *no* transform to obtain the
    class-to-index mapping and sample indices; then three Subset wrappers are
    created with the appropriate transforms applied through a thin wrapper.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset directory not found: {data_dir}\n"
            f"Please download PlantVillage and place it at that path.\n"
            f"See data_loader.py docstring for download links."
        )

    # ── 1. Load full dataset (no transform yet) ───────────────────────────
    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes  = len(full_dataset.classes)
    total        = len(full_dataset)

    print(f"[Dataset] Found {total} images across {num_classes} classes.")
    assert num_classes == config.NUM_CLASSES, (
        f"Expected {config.NUM_CLASSES} classes but found {num_classes}. "
        "Check DATA_DIR in config.py."
    )

    # ── 2. Save class-to-index mapping ───────────────────────────────────
    mapping_path = os.path.join(config.RESULTS_DIR, "class_to_idx.json")
    with open(mapping_path, "w") as f:
        json.dump(full_dataset.class_to_idx, f, indent=2)
    print(f"[Dataset] Class mapping saved → {mapping_path}")

    # ── 3. Stratified splits ──────────────────────────────────────────────
    indices = list(range(total))
    labels  = [full_dataset.targets[i] for i in indices]

    # train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    temp_labels = [labels[i] for i in temp_idx]

    # val vs test  (equal halves of the remaining portion)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=temp_labels,
        random_state=config.RANDOM_SEED,
    )

    print(
        f"[Dataset] Split → train: {len(train_idx)} | "
        f"val: {len(val_idx)} | test: {len(test_idx)}"
    )

    # ── 4. Wrap with appropriate transforms ───────────────────────────────
    train_set = TransformSubset(full_dataset, train_idx, get_train_transforms())
    val_set   = TransformSubset(full_dataset, val_idx,   get_eval_transforms())
    test_set  = TransformSubset(full_dataset, test_idx,  get_eval_transforms())

    return train_set, val_set, test_set, full_dataset.classes


class TransformSubset(torch.utils.data.Dataset):
    """
    A Dataset wrapper that applies a *transform* to a Subset of *dataset*.

    Critically: we load the raw PIL image directly from the file path stored
    in dataset.samples, BYPASSING ImageFolder.__getitem__ entirely.
    This avoids the bug where ImageFolder with transform=None returns
    inconsistently-sized PIL images that break normalization on val/test.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform
        # ImageFolder.samples is a list of (filepath, class_idx) tuples
        self.samples   = dataset.samples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filepath, label = self.samples[self.indices[idx]]

        # Load fresh PIL image — always RGB, always consistent
        from PIL import Image
        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir: str = config.DATA_DIR):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    """
    train_set, val_set, test_set, class_names = build_datasets(data_dir)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
        drop_last=True,      # avoids batch-norm issues with tiny last batches
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )

    print(
        f"[DataLoader] Batches → train: {len(train_loader)} | "
        f"val: {len(val_loader)} | test: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")    # (B, 3, 224, 224)
    print(f"Label shape : {labels.shape}")  # (B,)
    print(f"Classes[0]  : {classes[0]}")