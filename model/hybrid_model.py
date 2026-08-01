"""
model/hybrid_model.py
---------------------
Implements two models:

① CNN + ViT (no attention)  ── Baseline 3
   ResNet50 backbone → feature tokens → Transformer encoder → GAP → FC

② Hybrid CNN-ViT + Dual Attention  ── PROPOSED MODEL
   ResNet50 backbone → CBAM → feature tokens → Transformer encoder → GAP → FC

Both models share the same general structure; the only difference is whether
CBAM attention gates are applied to the CNN feature map before tokenisation.

Architecture detail
───────────────────
ResNet50 (pretrained, layer4 output: B×2048×7×7)
  │
  ├─ [CBAM: channel + spatial attention]  ← proposed only
  │
  ▼
Flatten spatial grid → sequence of tokens (49 tokens, each projected to VIT_EMBED_DIM)
  │
  ├─ Prepend [CLS] token
  ├─ Add learnable positional embedding
  │
  ▼
N × Transformer Encoder blocks
  │
  ▼
Extract [CLS] token → Dropout → FC → logits (38 classes)
"""

import torch
import torch.nn as nn
from torchvision import models

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.attention import CBAM
from model.vit import TransformerBlock     # reuse the block from vit.py


# ─────────────────────────────────────────────────────────────────────────────
#  CNN Feature Extractor (shared between both models)
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_backbone(backbone: str = config.CNN_BACKBONE,
                       pretrained: bool = config.PRETRAINED):
    """
    Returns the CNN backbone (everything except the final FC/pooling) and its
    output channel count.

    ResNet50 → layer4 output: (B, 2048, H/32, W/32)
               for 224×224 input → (B, 2048, 7, 7)
    """
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net     = models.resnet50(weights=weights)
        # Keep everything up to (and including) layer4; drop avgpool + fc
        feature_extractor = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        out_channels = 2048

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net     = models.efficientnet_b0(weights=weights)
        feature_extractor = net.features   # outputs (B, 1280, 7, 7)
        out_channels = 1280

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return feature_extractor, out_channels


# ─────────────────────────────────────────────────────────────────────────────
#  CNN + ViT without Attention  (Baseline 3)
# ─────────────────────────────────────────────────────────────────────────────

class CNNViT(nn.Module):
    """
    Baseline 3: CNN feature extractor → patch tokenisation → Transformer.
    No attention module; this isolates the contribution of CBAM.

    Args
    ────
    num_classes : output classes
    backbone    : CNN backbone name
    pretrained  : use ImageNet weights
    embed_dim   : transformer token dimension
    vit_depth   : number of transformer blocks
    num_heads   : attention heads
    mlp_ratio   : transformer MLP hidden ratio
    dropout     : dropout rate
    """

    def __init__(
        self,
        num_classes = config.NUM_CLASSES,
        backbone    = config.CNN_BACKBONE,
        pretrained  = config.PRETRAINED,
        embed_dim   = config.VIT_EMBED_DIM,
        vit_depth   = config.VIT_DEPTH,
        num_heads   = config.VIT_NUM_HEADS,
        mlp_ratio   = config.VIT_MLP_RATIO,
        dropout     = config.VIT_DROPOUT,
    ):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────
        self.cnn, cnn_channels = build_cnn_backbone(backbone, pretrained)

        # ── Token projection: 1×1 conv to embed_dim ──────────────────────
        self.token_proj = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)

        # ── CLS token & positional embedding ─────────────────────────────
        # num_patches = 7×7 = 49 for 224×224 input with ResNet50
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS; the spatial size is determined at runtime (flexible)
        # We use a fixed 49+1 = 50 here (224/32)^2 = 49
        self.num_patches = 49
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(vit_depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def _tokenise(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        CNN feature map (B, C, H, W) → sequence of tokens (B, H*W, embed_dim).
        """
        B = feat_map.shape[0]
        x = self.token_proj(feat_map)        # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)    # (B, H*W, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN
        feat = self.cnn(x)                              # (B, C, 7, 7)

        # Tokenise
        tokens = self._tokenise(feat)                   # (B, 49, embed_dim)

        # Prepend CLS & add positional embedding
        cls    = self.cls_token.expand(B, -1, -1)       # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 50, embed_dim)
        tokens = tokens + self.pos_embed

        # Transformer
        tokens = self.transformer(tokens)                # (B, 50, embed_dim)
        tokens = self.norm(tokens)

        # CLS token → head
        out = tokens[:, 0]                              # (B, embed_dim)
        return self.head(out)                           # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
#  Hybrid CNN-ViT + Dual Attention  (Proposed)
# ─────────────────────────────────────────────────────────────────────────────

class HybridCNNViT(nn.Module):
    """
    Proposed model: CNN → CBAM (channel + spatial attention) → ViT encoder.

    The CBAM module is inserted between the CNN backbone and the transformer,
    allowing the model to selectively emphasise disease-relevant channels and
    spatial regions before the global sequence modelling in the transformer.

    Args
    ────
    Same as CNNViT, plus:
    cbam_reduction : channel reduction ratio in CBAM (default 16)
    """

    def __init__(
        self,
        num_classes    = config.NUM_CLASSES,
        backbone       = config.CNN_BACKBONE,
        pretrained     = config.PRETRAINED,
        embed_dim      = config.VIT_EMBED_DIM,
        vit_depth      = config.VIT_DEPTH,
        num_heads      = config.VIT_NUM_HEADS,
        mlp_ratio      = config.VIT_MLP_RATIO,
        dropout        = config.VIT_DROPOUT,
        cbam_reduction = config.CBAM_REDUCTION,
    ):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────
        self.cnn, cnn_channels = build_cnn_backbone(backbone, pretrained)

        # ── CBAM Dual Attention ───────────────────────────────────────────
        self.cbam = CBAM(
            in_channels=cnn_channels,
            reduction=cbam_reduction,
            spatial_k=7,
        )

        # ── Token projection ──────────────────────────────────────────────
        self.token_proj = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)

        # ── CLS token & positional embedding ─────────────────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = 49
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(vit_depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def _tokenise(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(feat_map)
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN
        feat = self.cnn(x)                              # (B, 2048, 7, 7)

        # ── CBAM (dual attention) ─────────────────────────────────────────
        feat = self.cbam(feat)                          # (B, 2048, 7, 7)

        # Tokenise
        tokens = self._tokenise(feat)                   # (B, 49, embed_dim)

        # Prepend CLS & positional embedding
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 50, embed_dim)
        tokens = tokens + self.pos_embed

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        out = tokens[:, 0]                              # CLS token
        return self.head(out)                           # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)

    m1 = CNNViT(num_classes=38, pretrained=False)
    o1 = m1(dummy)
    print(f"CNNViT      output : {o1.shape}")
    print(f"CNNViT      params : {sum(p.numel() for p in m1.parameters() if p.requires_grad):,}")

    m2 = HybridCNNViT(num_classes=38, pretrained=False)
    o2 = m2(dummy)
    print(f"HybridCNNViT output : {o2.shape}")
    print(f"HybridCNNViT params : {sum(p.numel() for p in m2.parameters() if p.requires_grad):,}")