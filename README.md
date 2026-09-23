# Hybrid CNN–ViT with Dual Attention for Plant Disease Detection

A research-grade PyTorch project implementing and benchmarking four models for
automated plant disease classification on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
The class count is inferred automatically; the included archive variant has
15 classes and 20,638 leaf images.

---
---

## Internship Certificate

This project was developed as part of my **Image Processing & Computer Vision Internship**
at the **Department of Computer Science and Engineering, Graphic Era Deemed to be University, Dehradun**.

**Duration:** 1 June 2026 – 10 July 2026

📜 **[View Internship Certificate](./certificate.pdf)**

---
## Architecture Overview

```
INPUT (224×224 RGB)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  ResNet50 Backbone (pretrained, ImageNet)                │
│  Output: (B, 2048, 7, 7)                                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼   ◄── CBAM (proposed model only)
             ┌────────────────┐
             │ Channel Attention  (MLP on avg+max pool)    │
             │ Spatial Attention  (conv gate on pooled C)  │
             └────────┬───────┘
                      │
                      ▼
         Flatten spatial grid → 49 tokens
         Linear projection → embed_dim = 512
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Vision Transformer Encoder  (depth=2, heads=8)         │
│  [CLS] + 49 patch tokens + positional embedding         │
│  N × (LayerNorm → MSA → LayerNorm → MLP)                │
└─────────────────────┬───────────────────────────────────┘
                      │  CLS token
                      ▼
             Dropout → FC(512→256) → GELU → FC(256→NUM_CLASSES)
                      │
                      ▼
                   LOGITS (NUM_CLASSES)
```

### Four Models Compared

| # | Model Key | Description |
|---|-----------|-------------|
| 1 | `cnn`     | ResNet50 (CNN only) – Baseline 1 |
| 2 | `vit`     | Lightweight ViT from scratch – Baseline 2 |
| 3 | `cnn_vit` | CNN + ViT (no attention) – Baseline 3 |
| 4 | `hybrid`  | **CNN + CBAM + ViT – Proposed** |

---

## Project Structure

```
project/
├── config.py            ← all hyperparameters & paths
├── data_loader.py       ← dataset, augmentation, splits
├── train.py             ← training loop (all 4 models)
├── evaluate.py          ← test-set metrics & plots
├── utils.py             ← early stopping, logging, plotting
├── requirements.txt
├── README.md
│
├── model/
│   ├── __init__.py
│   ├── attention.py     ← CBAM (channel + spatial)
│   ├── cnn.py           ← CNNBaseline (ResNet50 / EfficientNet-B0)
│   ├── vit.py           ← ViTBaseline (from scratch)
│   └── hybrid_model.py  ← CNNViT (baseline 3) + HybridCNNViT (proposed)
│
├── data/
│   └── PlantVillage/    ← dataset goes here (see §Dataset)
│
├── checkpoints/         ← best model weights per run
├── logs/                ← per-model training logs
└── results/             ← plots, JSON reports, confusion matrices
```

---

## Setup

### 1. Clone / download this project

```bash
git clone <repo-url>
cd project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Replace the torch line in requirements.txt with the CUDA build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Dataset

### Download PlantVillage

Option A – Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/
```

Option B – Manual download:
1. Visit https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download and unzip into `data/PlantVillage/`

### Expected layout

```
data/PlantVillage/
  Apple___Apple_scab/
    Apple_scab_001.jpg
    ...
  Apple___Black_rot/
    ...
  Tomato___Tomato_mosaic_virus/
    ...
  (one folder per class; 15 or 38 depending on the dataset variant)
```

---

## How to Run

### Quick sanity check (dataset only)

```bash
python data_loader.py
```

### Train all 4 models

```bash
python train.py
```

### Train a single model

```bash
python train.py --model cnn        # CNN baseline
python train.py --model vit        # ViT baseline
python train.py --model cnn_vit    # CNN + ViT (no attention)
python train.py --model hybrid     # Proposed model
```

### Train with custom epochs

```bash
python train.py --model hybrid --epochs 50
```

### Evaluate all trained models on the test set

```bash
python evaluate.py
```

### Evaluate a single model

```bash
python evaluate.py --model hybrid
```

### Custom dataset path

```bash
python train.py --data /path/to/PlantVillage
python evaluate.py --data /path/to/PlantVillage
```

---

## Output Files

After training and evaluation:

```
checkpoints/
  cnn_best.pt
  vit_best.pt
  cnn_vit_best.pt
  hybrid_best.pt

results/
  class_to_idx.json
  cnn_history.json
  hybrid_history.json
  ...
  ResNet50_(CNN_only)_training.png
  Hybrid_CNN-ViT+Dual_Attention_(proposed)_training.png
  Hybrid_CNN-ViT+Dual_Attention_(proposed)_confusion_matrix.png
  model_comparison.png
  comparison_results.json
  hybrid_classification_report.txt
  ...

logs/
  cnn.log
  hybrid.log
  ...
```

---

## Key Hyperparameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 15 | Default only; training infers the dataset class count |
| `IMAGE_SIZE` | 224 | Input resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 30 | Max training epochs |
| `LEARNING_RATE` | 1e-4 | Adam initial LR |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation |
| `EARLY_STOP_PATIENCE` | 7 | Patience in epochs |
| `CNN_BACKBONE` | `resnet50` | `resnet50` or `efficientnet_b0` |
| `FREEZE_BATCH_NORM` | `True` | Freeze backbone statistics for stable MPS validation |
| `VIT_EMBED_DIM` | 512 | Transformer token dim |
| `VIT_DEPTH` | 2 | Number of transformer blocks |
| `VIT_NUM_HEADS` | 8 | Attention heads |
| `CBAM_REDUCTION` | 16 | Channel reduction in CBAM |

---

## Memory Tips for Laptops

- Reduce `BATCH_SIZE` to 16 if you run out of RAM.
- Reduce `VIT_DEPTH` to 1 for an even lighter transformer.
- Train one model at a time: `python train.py --model hybrid`.
- Keep `NUM_WORKERS = 0` for maximum portability; increase it only if multiprocessing works on your system.

---

## Citing

```bibtex
@misc{hybrid_cnn_vit_plant,
  title  = {Hybrid CNN-ViT with Dual Attention for Plant Disease Detection},
  year   = {2025},
  note   = {PyTorch research project}
}
```
