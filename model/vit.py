"""
model/vit.py
------------
Baseline 2: Lightweight Vision Transformer (ViT) built from scratch.

Architecture
────────────
Input (B, 3, 224, 224)
  → Patch Embedding (non-overlapping 16×16 patches → 196 tokens)
  → Prepend [CLS] token
  → Add positional embeddings
  → N × Transformer Encoder blocks (MSA + MLP)
  → Extract [CLS] token
  → MLP head → logits

This is intentionally lightweight (depth=6, dim=384) to run on a laptop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
#  Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits the image into non-overlapping patches and linearly projects each.

    (B, 3, H, W) → (B, num_patches, embed_dim)
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # A single conv with stride=patch_size efficiently extracts patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                      # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)                      # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)                 # (B, num_patches, embed_dim)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv     = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        # Compute Q, K, V in one shot
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Encoder Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block: LayerNorm → MSA → residual → LayerNorm → MLP → residual
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone ViT
# ─────────────────────────────────────────────────────────────────────────────

class ViTBaseline(nn.Module):
    """
    Lightweight ViT for plant disease classification.

    Args
    ────
    img_size    : input image resolution (default 224)
    patch_size  : patch size (default 16 → 14×14 = 196 tokens)
    in_channels : image channels (default 3)
    num_classes : output classes (default from config)
    embed_dim   : token embedding dimension
    depth       : number of transformer encoder blocks
    num_heads   : attention heads
    mlp_ratio   : MLP hidden dim ratio
    dropout     : dropout probability
    """

    def __init__(
        self,
        img_size    = config.IMAGE_SIZE,
        patch_size  = 16,
        in_channels = 3,
        num_classes = config.NUM_CLASSES,
        embed_dim   = 384,
        depth       = 6,
        num_heads   = 6,
        mlp_ratio   = 4.0,
        dropout     = 0.1,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # ── Patch + Positional Embeddings ─────────────────────────────────
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)

        # ── Transformer Encoder ───────────────────────────────────────────
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification Head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224)
        returns logits : (B, num_classes)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)                           # (B, N, embed_dim)

        # Prepend CLS token & add positional embedding
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, embed_dim)
        x   = torch.cat([cls, x], dim=1)                  # (B, N+1, embed_dim)
        x   = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        x = self.blocks(x)                                # (B, N+1, embed_dim)
        x = self.norm(x)

        # CLS token → classifier
        cls_out = x[:, 0]                                 # (B, embed_dim)
        return self.head(cls_out)                         # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ViTBaseline(num_classes=38, depth=4, embed_dim=256)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"ViTBaseline output : {out.shape}")     # (2, 38)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params   : {total:,}")