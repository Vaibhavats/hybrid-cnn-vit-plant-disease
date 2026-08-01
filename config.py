"""
config.py
---------
Central configuration for the Hybrid CNN-ViT Plant Disease Detection project.
All hyperparameters, paths, and training settings live here.
"""

import os
import torch

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "PlantVillage")   # root of dataset
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# Keep Matplotlib's font/config cache in a writable project location. This is
# important in sandboxed environments where the user's home is read-only.
MATPLOTLIB_CACHE_DIR = os.path.join(RESULTS_DIR, ".matplotlib-cache")
os.environ.setdefault("MPLCONFIGDIR", MATPLOTLIB_CACHE_DIR)
os.makedirs(MATPLOTLIB_CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
NUM_CLASSES     = 15          # default archive variant; loader infers this at runtime
IMAGE_SIZE      = 224         # resize every image to 224×224
TRAIN_SPLIT     = 0.70        # 70 % training
VAL_SPLIT       = 0.15        # 15 % validation
TEST_SPLIT      = 0.15        # 15 % test
RANDOM_SEED     = 42

# ImageNet statistics (used because CNN backbone is pretrained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 4           # portable default; avoids shared-memory worker failures

# Learning-rate scheduler (CosineAnnealingLR)
LR_T_MAX        = NUM_EPOCHS  # period for cosine annealing
LR_ETA_MIN      = 1e-6        # minimum LR

# Early stopping
EARLY_STOP_PATIENCE = 7       # stop if val-loss doesn't improve for N epochs

# ─────────────────────────────────────────────
#  Model – CNN Backbone
# ─────────────────────────────────────────────
CNN_BACKBONE    = "resnet50"  # "resnet50" | "efficientnet_b0"
PRETRAINED      = True
FREEZE_BATCH_NORM = True       # stable transfer learning, especially on Apple MPS
CNN_OUT_DIM     = 2048        # ResNet50 final feature channels (1280 for EfficientNet-B0)
FREEZE_BACKBONE = False

# ─────────────────────────────────────────────
#  Model – Vision Transformer block
# ─────────────────────────────────────────────
PATCH_SIZE      = 1           # treat each spatial location as a token (7×7 grid → 49 tokens)
VIT_EMBED_DIM   = 512         # token embedding dimension after projection
VIT_NUM_HEADS   = 8           # multi-head attention heads
VIT_DEPTH       = 2           # number of transformer encoder layers (lightweight)
VIT_MLP_RATIO   = 4.0         # MLP hidden-dim multiplier
VIT_DROPOUT     = 0.1

# ─────────────────────────────────────────────
#  Model – CBAM Attention
# ─────────────────────────────────────────────
CBAM_REDUCTION  = 16          # channel reduction ratio in CBAM

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
# ─────────────────────────────────────────────
#  Logging / Checkpointing
# ─────────────────────────────────────────────
LOG_INTERVAL    = 50          # print training stats every N batches
SAVE_BEST_ONLY  = True        # only save checkpoint when val-acc improves

# ─────────────────────────────────────────────
#  Model names (used for checkpoint files)
# ─────────────────────────────────────────────
MODEL_NAMES = {
    "cnn":        "ResNet50 (CNN only)",
    "vit":        "ViT (Transformer only)",
    "cnn_vit":    "CNN + ViT (no attention)",
    "hybrid":     "Hybrid CNN-ViT + Dual Attention (proposed)",
}
