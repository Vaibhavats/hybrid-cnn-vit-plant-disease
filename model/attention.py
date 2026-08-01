"""
model/attention.py
------------------
CBAM – Convolutional Block Attention Module
Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)

CBAM applies two sequential attention gates:
  1. Channel Attention  – "what" to focus on (feature recalibration)
  2. Spatial Attention  – "where" to focus on (spatial localisation)

Both gates produce multiplicative masks in [0, 1], so the module is a
drop-in refinement on top of any conv feature map F ∈ R^(B, C, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Channel Attention
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel attention via shared MLP on both avg-pool and max-pool descriptors.

    Architecture
    ────────────
    F ─► AvgPool ─┐
                  ├─► shared MLP ─► sum ─► sigmoid ─► Mc (B, C, 1, 1)
    F ─► MaxPool ─┘

    The shared MLP has one hidden layer: C → C//reduction → C.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        # Shared weights implemented as two 1×1 convolutions
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Global Average Pool & Max Pool across H×W ─────────────────────
        avg_pool = F.adaptive_avg_pool2d(x, 1)          # (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)          # (B, C, 1, 1)

        # ── Shared MLP ────────────────────────────────────────────────────
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))

        # ── Attention mask ────────────────────────────────────────────────
        scale = torch.sigmoid(avg_out + max_out)        # (B, C, 1, 1)
        return x * scale                                # broadcast over H, W


# ─────────────────────────────────────────────────────────────────────────────
#  Spatial Attention
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Spatial attention via channel-wise pooling followed by a conv gate.

    Architecture
    ────────────
    F ─► [AvgPool_C ; MaxPool_C] ─► 7×7 Conv ─► sigmoid ─► Ms (B, 1, H, W)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        # 2 input channels (avg + max), 1 output channel
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Channel-wise descriptors ───────────────────────────────────────
        avg_pool = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_pool = x.max(dim=1, keepdim=True).values    # (B, 1, H, W)
        pooled   = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)

        # ── Attention mask ────────────────────────────────────────────────
        scale = torch.sigmoid(self.conv(pooled))        # (B, 1, H, W)
        return x * scale                                # broadcast over C


# ─────────────────────────────────────────────────────────────────────────────
#  CBAM (Dual Attention)
# ─────────────────────────────────────────────────────────────────────────────

class CBAM(nn.Module):
    """
    Full CBAM module: Channel Attention → Spatial Attention (sequential).

    Args
    ────
    in_channels : number of input feature channels (C)
    reduction   : channel reduction ratio for MLP (default 16)
    spatial_k   : spatial conv kernel size, 3 or 7 (default 7)

    Usage
    ─────
    >>> cbam = CBAM(in_channels=2048, reduction=16)
    >>> out  = cbam(feature_map)   # same shape as input
    """

    def __init__(self, in_channels: int, reduction: int = 16, spatial_k: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)   # channel gate
        x = self.spatial_att(x)   # spatial gate
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy = torch.randn(2, 2048, 7, 7)
    cbam  = CBAM(in_channels=2048, reduction=16)
    out   = cbam(dummy)
    print(f"CBAM  input : {dummy.shape}")
    print(f"CBAM output : {out.shape}")   # same as input
    assert out.shape == dummy.shape
    print("CBAM test passed ✓")
