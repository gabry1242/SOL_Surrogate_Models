#!/usr/bin/env python3
"""
train_fm_3views.py

Train a conditional Flow Matching model on ONE of the three per-view 4D tensors
produced by build_3view_tensors.py.

=============================================================================
FLOW MATCHING OVERVIEW
=============================================================================

Instead of directly regressing Y from X (as the deterministic UNet did), flow
matching learns a *velocity field* v_θ(y_t, t, x) that transports samples from
a simple prior distribution (standard Gaussian noise) to the data distribution.

The core idea:
  1. Define a linear interpolation path between noise y_0 ~ N(0,I) and data y_1:
       y_t = (1 - t) * y_0  +  t * y_1       for t ∈ [0, 1]
  2. The ground-truth velocity along this path is simply:
       v*(y_t, t) = y_1 - y_0
  3. Train a neural network v_θ to predict this velocity:
       L = E_{t, y_0, y_1} || v_θ(y_t, t, x) - (y_1 - y_0) ||²

At inference time, we start from noise y_0 and integrate the learned velocity
field using an ODE solver (e.g., Euler steps) from t=0 to t=1 to generate
samples from the conditional distribution p(y|x).

=============================================================================
KEY DIFFERENCES FROM THE DETERMINISTIC UNET
=============================================================================

1. TIME CONDITIONING: The UNet now takes an additional scalar input t ∈ [0,1]
   that indicates the position along the interpolation path. This is embedded
   via a sinusoidal positional encoding → MLP → injected into every decoder
   block via adaptive scaling (FiLM-style: scale + shift).

2. NOISY Y AS INPUT: Instead of only seeing X, the network also receives the
   noisy field y_t as additional input channels. The UNet input becomes
   [X_norm; y_t] concatenated along the channel dimension.

3. ACTIVATION REINJECTION (CONDITIONING ENCODER): The conditioning input X
   is first processed by a separate lightweight encoder that produces
   multi-scale feature maps. These feature maps are injected into the
   velocity UNet's decoder via concatenation at matching spatial resolutions.
   This ensures the model has strong access to the conditioning signal at
   every scale, not just at the input level.

4. VELOCITY OUTPUT: The network outputs v_θ with the same spatial shape as Y,
   predicting the velocity field rather than the field itself.

5. PROBABILISTIC: At inference, different noise draws y_0 produce different
   samples → the model captures p(y|x), not just E[y|x].

=============================================================================
PRESERVED FROM THE DETERMINISTIC PIPELINE
=============================================================================

- Channel-specific target transforms (log10 for positive, asinh for signed)
- Masked normalization statistics (mean/std computed only over active cells)
- X normalization (per-channel z-score with mask)
- Per-view independent training
- Channel-balanced loss (mean over per-channel masked errors)
- Mask enforcement (predictions zeroed outside active domain)
- All checkpoint metadata for seamless inference

=============================================================================
USAGE
=============================================================================

# Train on view0
python train_fm_3views.py \\
    --view 0 \\
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \\
    --train_split train \\
    --test_prefix  scripts/tensor/3views_4d/test/global3v \\
    --test_split   test \\
    --run_dir      scripts/runs/fm_3views/view0 \\
    --y_channels   all \\
    --epochs       100 \\
    --batch_size   32 \\
    --base         64 \\
    --lr           5e-4

# Train on view1 / view2: change --view and --run_dir accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Channel definitions & masked statistics
# ═══════════════════════════════════════════════════════════════════════════
# These are IDENTICAL to the deterministic pipeline. They define which Y
# channels get log10 vs asinh transforms, and how to compute masked stats.

POS_CHANNELS    = set(range(0, 12))   # te, ti, na(10) → log10(max(y,0)+eps)
SIGNED_CHANNELS = set(range(12, 22))  # ua(10)         → asinh(y / s_c)


def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support; cannot compute masked statistics.")
    return den


def masked_channel_stats_x(
    x_mem: np.ndarray, mask_ch: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel masked mean/std for X.  Mask channel forced to mean=0, std=1."""
    if x_mem.ndim != 4:
        raise ValueError("X must be 4D (N,C,H,W).")
    _, C, _, _ = x_mem.shape
    m   = x_mem[:, mask_ch:mask_ch + 1, :, :].astype(np.float64)
    den = _masked_den(m)

    mean = np.zeros(C, dtype=np.float64)
    var  = np.zeros(C, dtype=np.float64)
    for c in range(C):
        xc      = x_mem[:, c:c + 1, :, :].astype(np.float64)
        mean[c] = float((xc * m).sum() / den)
    for c in range(C):
        xc     = x_mem[:, c:c + 1, :, :].astype(np.float64)
        var[c] = float((((xc - mean[c]) ** 2) * m).sum() / den)

    std            = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch]  = 0.0
    std[mask_ch]   = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def masked_mean_std_transformed_y(
    y_mem: np.ndarray,
    mask_mem: np.ndarray,
    y_indices: Sequence[int],
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel (mean, std) in TRANSFORMED space + per-channel velocity scales s_c.

    Positive channels  → t = log10(max(y,0) + eps)
    Signed channels    → t = asinh(y / s_c),  s_c = masked physical std

    Returns y_mean, y_std, s_c — all (C_sel,) float32.
    """
    if y_mem.ndim != 4 or mask_mem.ndim != 4 or mask_mem.shape[1] != 1:
        raise ValueError("y_mem must be 4D (N,C,H,W) and mask_mem 4D (N,1,H,W).")
    m   = mask_mem.astype(np.float64)
    den = _masked_den(m)

    C_sel  = len(y_indices)
    y_mean = np.zeros(C_sel, dtype=np.float64)
    y_var  = np.zeros(C_sel, dtype=np.float64)
    s_c    = np.ones(C_sel,  dtype=np.float64)

    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc  = y_mem[:, c:c + 1, :, :].astype(np.float64)
            mu  = float((yc * m).sum() / den)
            var = float((((yc - mu) ** 2) * m).sum() / den)
            s_c[j] = float(np.sqrt(max(var, 1e-12)))

    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_mean[j] = float((t * m).sum() / den)

    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_var[j] = float((((t - y_mean[j]) ** 2) * m).sum() / den)

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Dataset
# ═══════════════════════════════════════════════════════════════════════════
# Same as the deterministic pipeline: loads X and Y, applies transforms and
# normalization.  The flow matching noise sampling happens in the training
# loop, NOT here — the dataset returns clean normalized data.

class ViewTensorDataset(Dataset):
    """
    Returns (x_norm, y_norm, mask) where:
      x_norm : (C_in, H, W)   — normalized conditioning input
      y_norm : (C_sel, H, W)  — transformed + normalized target (y_1 in FM)
      mask   : (1, H, W)      — binary mask of active cells
    """

    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        y_indices: Sequence[int],
        x_mean: Optional[np.ndarray],
        x_std:  Optional[np.ndarray],
        y_mean: np.ndarray,
        y_std:  np.ndarray,
        s_c:    np.ndarray,
        eps:    float = 1e-3,
    ):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")

        if self.X.ndim != 4 or self.Y.ndim != 4:
            raise ValueError("X and Y must be 4D (N,C,H,W).")
        if self.X.shape[0] != self.Y.shape[0] or self.X.shape[2:] != self.Y.shape[2:]:
            raise ValueError(f"X/Y shape mismatch: {self.X.shape} vs {self.Y.shape}")

        self.y_indices = list(map(int, y_indices))
        self.x_mean    = x_mean
        self.x_std     = x_std
        self.y_mean    = np.asarray(y_mean, dtype=np.float32)
        self.y_std     = np.asarray(y_std,  dtype=np.float32)
        self.s_c       = np.asarray(s_c,    dtype=np.float32)
        self.eps       = float(eps)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _transform_y(self, y_sel: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(y_sel)
        for j, c in enumerate(self.y_indices):
            if c in POS_CHANNELS:
                out[j] = torch.log10(torch.clamp(y_sel[j], min=0.0) + self.eps)
            else:
                out[j] = torch.asinh(y_sel[j] / float(self.s_c[j]))
        return out

    def __getitem__(self, idx: int):
        x      = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y_full = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))
        m      = x[0:1]   # (1, H, W) mask channel

        # X normalization
        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x  = (x - xm) / xs

        # Select + transform + normalize Y
        y_sel = y_full[self.y_indices, :, :]
        y_t   = self._transform_y(y_sel)
        ym    = torch.from_numpy(self.y_mean).view(-1, 1, 1)
        ys    = torch.from_numpy(self.y_std).view(-1, 1, 1)
        y_n   = (y_t - ym) / ys

        return x, y_n, m


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Sinusoidal time embedding
# ═══════════════════════════════════════════════════════════════════════════
# The flow matching time t ∈ [0,1] needs to be encoded as a high-dimensional
# vector so the network can distinguish different timesteps.  We use the
# standard sinusoidal encoding from transformers / diffusion models:
#   emb[2i]   = sin(t * 10000^(-2i/d))
#   emb[2i+1] = cos(t * 10000^(-2i/d))
# followed by a 2-layer MLP to project to the desired dimension.

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t → vector of dimension `dim`.
    t can be any shape (...); output is (..., dim).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        # Precompute frequency bands (not trainable)
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freq", freq)

        # Small MLP to project from raw sinusoidal features to final embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        t = t.view(-1).float()
        # (B, half)
        angles = t[:, None] * self.freq[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)   # (B, dim)
        return self.mlp(emb)   # (B, dim)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Conditioning Encoder (Activation Reinjection)
# ═══════════════════════════════════════════════════════════════════════════
# This is the "activation reinjection" mechanism.  A separate lightweight
# encoder processes ONLY the conditioning input X (without the noisy y_t).
# It produces feature maps at multiple spatial resolutions that match the
# decoder stages of the velocity UNet.
#
# These feature maps are concatenated into the velocity UNet's decoder at
# each scale, giving the model strong conditioning signal at every
# resolution — not just at the input.  This is crucial because:
#   - The 8 scalar parameters broadcast over the mask are spatially uniform
#   - Without reinjection, deep layers might "forget" the conditioning
#   - Multi-scale injection lets coarse features capture global structure
#     while fine features preserve local detail

class ConditioningEncoder(nn.Module):
    """
    Lightweight encoder that extracts multi-scale features from the
    conditioning input X.  Produces feature maps at 3 scales matching
    the velocity UNet decoder (1x, 1/2x, 1/4x resolution).
    """
    def __init__(self, c_in: int, base: int = 32):
        super().__init__()
        # Scale 1: full resolution → base channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(c_in, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        # Scale 2: half resolution → base*2 channels
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        # Scale 3: quarter resolution → base*4 channels
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (f1, f2, f3) at full, half, quarter resolution."""
        f1 = self.enc1(x)                  # (B, base,   H,   W)
        f2 = self.enc2(self.pool(f1))      # (B, base*2, H/2, W/2)
        f3 = self.enc3(self.pool(f2))      # (B, base*4, H/4, W/4)
        return f1, f2, f3


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Time-conditioned ConvBlock with FiLM
# ═══════════════════════════════════════════════════════════════════════════
# Each ConvBlock in the decoder receives a time embedding and applies
# Feature-wise Linear Modulation (FiLM):
#   output = scale * conv_output + shift
# where (scale, shift) are predicted from the time embedding.
# This lets the network modulate its behavior based on where along the
# flow path (t=0 noise → t=1 data) the current input sits.

class ConvBlock(nn.Module):
    """Two conv layers with ReLU.  No time conditioning (used in encoder)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,  c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMConvBlock(nn.Module):
    """
    Two conv layers + FiLM modulation from a time embedding.
    After the second conv + ReLU, we apply:
      out = scale * out + shift
    where scale, shift are predicted from the time embedding vector.
    """
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in,  c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.act   = nn.ReLU(inplace=True)
        # Project time embedding → (scale, shift) for c_out channels
        self.film  = nn.Linear(t_dim, c_out * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        # FiLM modulation
        film_params = self.film(t_emb)                     # (B, c_out*2)
        scale, shift = film_params.chunk(2, dim=-1)        # each (B, c_out)
        scale = scale.unsqueeze(-1).unsqueeze(-1)          # (B, c_out, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + scale) + shift    # (1 + scale) so initial scale ≈ 1


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Velocity UNet with conditioning injection
# ═══════════════════════════════════════════════════════════════════════════
# This is the core network v_θ(y_t, t, x).  Key design choices:
#
# INPUT: Concatenation of [X_norm, y_t] along channel dim → c_in + c_out channels
#
# ENCODER: Standard UNet encoder (3 levels + bottleneck), no time conditioning
# needed here — the encoder just extracts features from the noisy input.
#
# DECODER: Each decoder level receives:
#   - Upsampled features from the level below
#   - Skip connection from the encoder (same as standard UNet)
#   - Conditioning features from the ConditioningEncoder (activation reinjection)
#   - FiLM modulation from the time embedding
#
# The conditioning features are concatenated with the skip connections,
# effectively doubling the skip channel count.  This means each decoder
# ConvBlock input has: upsampled_channels + encoder_skip + cond_skip channels.

class VelocityUNet(nn.Module):
    """
    Time-conditioned UNet that predicts the velocity field v_θ.

    Input:  (B, c_in + c_out, H, W)  — concat of [X_norm, y_t]
    Output: (B, c_out, H, W)         — predicted velocity
    """

    def __init__(self, c_in: int, c_out: int, base: int = 32, t_dim: int = 128):
        super().__init__()

        self.c_in  = c_in
        self.c_out = c_out

        # --- Time embedding ---
        self.time_embed = SinusoidalTimeEmbedding(t_dim)

        # --- Conditioning encoder (processes X only) ---
        self.cond_encoder = ConditioningEncoder(c_in, base=base)

        # --- Velocity UNet encoder (processes [X; y_t]) ---
        self.enc1       = ConvBlock(c_in + c_out, base)
        self.enc2       = ConvBlock(base,         base * 2)
        self.enc3       = ConvBlock(base * 2,     base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4,     base * 8)

        # --- Velocity UNet decoder with conditioning injection + FiLM ---
        # At each level, decoder input = upsampled + enc_skip + cond_skip
        # Level 3: up(base*8→base*4) + enc3(base*4) + cond3(base*4) = base*12
        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = FiLMConvBlock(base * 4 + base * 4 + base * 4, base * 4, t_dim)

        # Level 2: up(base*4→base*2) + enc2(base*2) + cond2(base*2) = base*6
        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = FiLMConvBlock(base * 2 + base * 2 + base * 2, base * 2, t_dim)

        # Level 1: up(base*2→base) + enc1(base) + cond1(base) = base*3
        self.up1  = nn.Conv2d(base * 2, base, 1)
        self.dec1 = FiLMConvBlock(base + base + base, base, t_dim)

        # Final output projection
        self.out = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _match_size(x: torch.Tensor, target_hw) -> torch.Tensor:
        """Center-crop or interpolate x to match target spatial dimensions."""
        th, tw = target_hw
        h, w = x.shape[-2], x.shape[-1]
        if h == th and w == tw:
            return x
        if h >= th and w >= tw:
            dh, dw = h - th, w - tw
            return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(
        self,
        x_cond: torch.Tensor,    # (B, c_in, H, W)  — normalized conditioning
        y_t:    torch.Tensor,     # (B, c_out, H, W) — noisy field at time t
        t:      torch.Tensor,     # (B,) or (B,1)    — time in [0, 1]
    ) -> torch.Tensor:
        # --- Time embedding ---
        t_emb = self.time_embed(t)   # (B, t_dim)

        # --- Conditioning encoder (activation reinjection source) ---
        c1, c2, c3 = self.cond_encoder(x_cond)   # multi-scale cond features

        # --- Velocity UNet encoder ---
        inp = torch.cat([x_cond, y_t], dim=1)    # (B, c_in + c_out, H, W)
        e1 = self.enc1(inp)                       # (B, base, H, W)
        e2 = self.enc2(self.pool(e1))             # (B, base*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))             # (B, base*4, H/4, W/4)
        b  = self.bottleneck(self.pool(e3))       # (B, base*8, H/8, W/8)

        # --- Decoder with skip connections + conditioning injection + FiLM ---
        # Level 3
        u3 = self.up3(F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False))
        c3_matched = self._match_size(c3, u3.shape[-2:])
        e3_matched = self._match_size(e3, u3.shape[-2:])
        d3 = self.dec3(torch.cat([u3, e3_matched, c3_matched], dim=1), t_emb)

        # Level 2
        u2 = self.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        c2_matched = self._match_size(c2, u2.shape[-2:])
        e2_matched = self._match_size(e2, u2.shape[-2:])
        d2 = self.dec2(torch.cat([u2, e2_matched, c2_matched], dim=1), t_emb)

        # Level 1
        u1 = self.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        c1_matched = self._match_size(c1, u1.shape[-2:])
        e1_matched = self._match_size(e1, u1.shape[-2:])
        d1 = self.dec1(torch.cat([u1, e1_matched, c1_matched], dim=1), t_emb)

        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Loss and metrics
# ═══════════════════════════════════════════════════════════════════════════
# The flow matching loss is the masked MSE between the predicted velocity
# v_θ and the ground-truth velocity (y_1 - y_0), averaged per channel.
#
# For evaluation, we run full ODE integration (Euler) to generate samples,
# then compute MAE/RMSE in the normalized transformed space.

def masked_mse_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Per-channel masked MSE, averaged over spatial dims and batch."""
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return num / den


def flow_matching_loss(
    v_pred: torch.Tensor,      # predicted velocity
    v_target: torch.Tensor,    # ground-truth velocity = y_1 - y_0
    mask: torch.Tensor,
) -> torch.Tensor:
    """Channel-balanced masked MSE loss for flow matching."""
    return masked_mse_per_channel(v_pred, v_target, mask).mean()


def masked_mae_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).abs().mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return num / den


def masked_rmse_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return torch.sqrt(num / den)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: ODE integration (Euler method)
# ═══════════════════════════════════════════════════════════════════════════
# At inference, we generate samples by integrating the learned velocity
# field from t=0 (noise) to t=1 (data):
#
#   y_{t+dt} = y_t + v_θ(y_t, t, x) * dt
#
# We use simple Euler integration with a configurable number of steps.
# More steps = more accurate but slower.  Typically 50-100 steps suffice.

@torch.no_grad()
def euler_sample(
    model: VelocityUNet,
    x_cond: torch.Tensor,        # (B, c_in, H, W)
    mask: torch.Tensor,          # (B, 1, H, W)
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate samples by Euler integration of the learned velocity field.
    Returns y_1 in normalized transformed space: (B, c_out, H, W).
    """
    B = x_cond.shape[0]
    c_out = model.c_out
    H, W = x_cond.shape[-2], x_cond.shape[-1]

    # Start from standard Gaussian noise, masked to active region
    y = torch.randn(B, c_out, H, W, device=device) * mask
    dt = 1.0 / n_steps

    model.eval()
    for step in range(n_steps):
        t_val = step / n_steps
        t = torch.full((B,), t_val, device=device)
        v = model(x_cond, y, t)
        y = y + v * dt
        y = y * mask   # enforce mask at every step

    return y


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Evaluation (generates samples + computes metrics)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: VelocityUNet,
    loader: DataLoader,
    device: torch.device,
    n_steps: int = 50,
) -> Dict:
    """
    For each batch: generate a sample via Euler integration, compare to
    ground truth in normalized transformed space.
    """
    model.eval()
    mae_acc = rmse_acc = None
    n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = euler_sample(model, x, m, n_steps=n_steps, device=device)
        mae_c  = masked_mae_per_channel(pred, y, m).cpu()
        rmse_c = masked_rmse_per_channel(pred, y, m).cpu()
        mae_acc  = mae_c  if mae_acc  is None else mae_acc  + mae_c
        rmse_acc = rmse_c if rmse_acc is None else rmse_acc + rmse_c
        n += 1
    if n == 0:
        return {"mae_avg": float("nan"), "rmse_avg": float("nan"),
                "mae_per_channel": [], "rmse_per_channel": []}
    mae_acc  /= n
    rmse_acc /= n
    return {
        "mae_avg":          float(mae_acc.mean()),
        "rmse_avg":         float(rmse_acc.mean()),
        "mae_per_channel":  mae_acc.numpy().tolist(),
        "rmse_per_channel": rmse_acc.numpy().tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: Helpers
# ═══════════════════════════════════════════════════════════════════════════

def parse_channels(s: str, c_out_total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(c_out_total))
    ch = [int(p) for p in s.split(",") if p.strip()]
    if any(c < 0 or c >= c_out_total for c in ch):
        raise ValueError(f"y_channels out of bounds: {ch}, total={c_out_total}")
    return ch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: Main training loop
# ═══════════════════════════════════════════════════════════════════════════
# The training loop implements the flow matching objective:
#
# For each batch:
#   1. Sample t ~ Uniform(0, 1)
#   2. Sample noise y_0 ~ N(0, I) masked to active region
#   3. Compute interpolated state: y_t = (1-t)*y_0 + t*y_1
#   4. Compute ground-truth velocity: v* = y_1 - y_0
#   5. Predict velocity: v_θ = model(x, y_t, t)
#   6. Loss = masked MSE(v_θ, v*)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Flow Matching model on a single 3-view tensor (view 0, 1, or 2)."
    )

    ap.add_argument("--view", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_prefix", required=True)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--run_dir", required=True)

    ap.add_argument("--y_channels", default="all")
    ap.add_argument("--epochs",       type=int,   default=100)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base",         type=int,   default=64,
                    help="UNet base channel width.")
    ap.add_argument("--t_dim",        type=int,   default=128,
                    help="Time embedding dimension.")
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps",          type=float, default=1e-5,
                    help="Epsilon for log10 transform on positive channels.")
    ap.add_argument("--eval_steps",   type=int,   default=50,
                    help="Number of Euler steps for evaluation sampling.")
    ap.add_argument("--eval_every",   type=int,   default=5,
                    help="Run full evaluation every N epochs (expensive).")
    args = ap.parse_args()

    # ---- resolve file paths ----
    view_tag     = f"view{args.view}"
    train_x_path = Path(f"{args.tensor_prefix}_{view_tag}_X_img_{args.train_split}.npy")
    train_y_path = Path(f"{args.tensor_prefix}_{view_tag}_Y_img_{args.train_split}.npy")
    test_x_path  = Path(f"{args.test_prefix}_{view_tag}_X_img_{args.test_split}.npy")
    test_y_path  = Path(f"{args.test_prefix}_{view_tag}_Y_img_{args.test_split}.npy")

    for p in (train_x_path, train_y_path, test_x_path, test_y_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # ---- memmaps for stats ----
    X_train = np.load(train_x_path, mmap_mode="r")
    Y_train = np.load(train_y_path, mmap_mode="r")

    c_in        = int(X_train.shape[1])
    c_out_total = int(Y_train.shape[1])

    y_indices = parse_channels(args.y_channels, c_out_total)
    c_out     = len(y_indices)

    print(f"\n=== Training Flow Matching on {view_tag} ===")
    print(f"  Train X: {train_x_path}  {X_train.shape}")
    print(f"  Train Y: {train_y_path}  {Y_train.shape}")
    print(f"  c_in={c_in}  c_out={c_out}  y_indices={y_indices}")

    # ---- normalization statistics (from train split only) ----
    print("Computing X normalization stats ...")
    x_mean, x_std = masked_channel_stats_x(X_train, mask_ch=0)

    mask_train = X_train[:, 0:1, :, :].astype(np.float32)
    print("Computing Y normalization stats in transformed space ...")
    y_mean, y_std, s_c = masked_mean_std_transformed_y(
        Y_train, mask_train, y_indices=y_indices, eps=args.eps
    )

    # ---- datasets ----
    ds_kwargs = dict(
        y_indices=y_indices, x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std, s_c=s_c, eps=args.eps,
    )
    ds_train = ViewTensorDataset(train_x_path, train_y_path, **ds_kwargs)
    ds_test  = ViewTensorDataset(test_x_path,  test_y_path,  **ds_kwargs)

    device = torch.device(args.device)
    pin    = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)

    # ---- model + optimizer ----
    model = VelocityUNet(
        c_in=c_in, c_out=c_out, base=args.base, t_dim=args.t_dim
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # ---- save config ----
    config = dict(
        arch="flow_matching_unet",
        view=args.view, view_tag=view_tag,
        tensor_prefix=args.tensor_prefix, train_split=args.train_split,
        test_prefix=args.test_prefix,     test_split=args.test_split,
        run_dir=str(run_dir),
        y_channels=args.y_channels, y_indices=y_indices,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        base=args.base, t_dim=args.t_dim,
        seed=args.seed, device=args.device, eps=args.eps,
        eval_steps=args.eval_steps, eval_every=args.eval_every,
        n_params=n_params,
    )
    save_json(run_dir / "config.json", config)

    # ---- training loop ----
    best_rmse    = float("inf")
    metrics_hist = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_steps = 0.0, 0

        for x, y1, m in dl_train:
            x, y1, m = x.to(device), y1.to(device), m.to(device)
            B = x.shape[0]

            # ---- Flow matching sampling ----
            # 1. Sample random time t ~ Uniform(0, 1)
            t = torch.rand(B, device=device)

            # 2. Sample noise y_0 ~ N(0, I), masked
            y0 = torch.randn_like(y1) * m

            # 3. Interpolate: y_t = (1-t)*y_0 + t*y_1
            t_expand = t.view(B, 1, 1, 1)
            yt = (1.0 - t_expand) * y0 + t_expand * y1
            yt = yt * m   # enforce mask

            # 4. Ground-truth velocity
            v_target = (y1 - y0) * m

            # 5. Predict velocity
            v_pred = model(x, yt, t) * m

            # 6. Loss
            loss = flow_matching_loss(v_pred, v_target, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_steps  += 1

        train_loss = loss_sum / max(n_steps, 1)

        # ---- Evaluation (expensive, so done periodically) ----
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs) or (epoch == 1)

        if do_eval:
            test_eval = evaluate(model, dl_test, device, n_steps=args.eval_steps)
            metrics_hist["test"].append({"epoch": epoch, **test_eval})
            cur_rmse = float(test_eval["rmse_avg"])
        else:
            test_eval = None
            cur_rmse = float("inf")

        metrics_hist["train"].append({"epoch": epoch, "loss": train_loss})

        # ---- Checkpoint ----
        ckpt = {
            "arch":            "flow_matching_unet",
            "view":            args.view,
            "view_tag":        view_tag,
            "base":            int(args.base),
            "t_dim":           int(args.t_dim),
            "c_in":            int(c_in),
            "c_out":           int(c_out),
            "c_out_total":     int(c_out_total),
            "y_indices":       [int(c) for c in y_indices],
            "pos_channels":    sorted(POS_CHANNELS),
            "signed_channels": sorted(SIGNED_CHANNELS),
            "eps":             float(args.eps),
            "x_mean":          x_mean.tolist(),
            "x_std":           x_std.tolist(),
            "y_mean":          y_mean.tolist(),
            "y_std":           y_std.tolist(),
            "s_c":             s_c.tolist(),
            "model_state":     model.state_dict(),
            "opt_state":       opt.state_dict(),
            "epoch":           int(epoch),
            "eval_steps":      int(args.eval_steps),
        }

        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        save_json(run_dir / "metrics.json", metrics_hist)

        if do_eval:
            print(
                f"[{view_tag}] Epoch {epoch:03d}/{args.epochs} | "
                f"loss={train_loss:.4g} | "
                f"test_mae={test_eval['mae_avg']:.4g} | "
                f"test_rmse={test_eval['rmse_avg']:.4g}"
            )
        else:
            print(
                f"[{view_tag}] Epoch {epoch:03d}/{args.epochs} | "
                f"loss={train_loss:.4g}"
            )

    print(f"\nDone. Best test rmse_avg={best_rmse:.6g}  ->  {run_dir}/checkpoint_best.pt")


if __name__ == "__main__":
    main()
