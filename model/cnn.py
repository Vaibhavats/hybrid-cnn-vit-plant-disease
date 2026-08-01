"""
model/cnn.py
------------
Baseline 1: Pure CNN classifier.

Uses a pretrained ResNet50 (or EfficientNet-B0) backbone with a replaced
classification head for NUM_CLASSES output.  No transformer, no attention.
"""

import torch
import torch.nn as nn
from torchvision import models

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
#  CNN-only Baseline
# ─────────────────────────────────────────────────────────────────────────────

class CNNBaseline(nn.Module):
    """
    Transfer-learning baseline: pretrained backbone + custom FC head.

    Args
    ────
    num_classes : number of output classes (default from config)
    backbone    : "resnet50" | "efficientnet_b0"
    pretrained  : load ImageNet weights (default True)
    """

    def __init__(
        self,
        num_classes: int  = config.NUM_CLASSES,
        backbone:    str  = config.CNN_BACKBONE,
        pretrained:  bool = config.PRETRAINED,
    ):
        super().__init__()
        self.backbone_name = backbone

        # ── Load pretrained backbone ──────────────────────────────────────
        if backbone == "resnet50":
            weights  = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net      = models.resnet50(weights=weights)
            in_feats = net.fc.in_features          # 2048
            net.fc   = nn.Identity()               # strip original classifier
            self.backbone   = net
            self.feature_dim = in_feats

        elif backbone == "efficientnet_b0":
            weights  = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net      = models.efficientnet_b0(weights=weights)
            in_feats = net.classifier[1].in_features  # 1280
            net.classifier = nn.Identity()
            self.backbone   = net
            self.feature_dim = in_feats

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224)
        returns logits : (B, num_classes)
        """
        feat   = self.backbone(x)   # (B, feature_dim)
        logits = self.head(feat)    # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CNNBaseline(num_classes=38, backbone="resnet50", pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"CNNBaseline output : {out.shape}")   # (2, 38)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params   : {total:,}")