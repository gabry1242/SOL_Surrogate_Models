#!/usr/bin/env python3
"""
train_unet_3inputs_fast.py

Optimized UNet training for 3-view tensors produced by tensor_3_images.py.

Performance fixes vs. train_unet_3inputs.py:
  1) MaskBoundaryReinjectionConv REMOVED.
     Replaced with plain Conv2d + a single (cheap) mask-multiply after each
     ConvBlock.  The expensive per-layer dilation + valid_count convolutions
     (~28-30 extra full-resolution ops per forward pass) are gone entirely.
  2) num_workers > 0.
     DataLoaders now use num_workers=4 (configurable via --num_workers) so
     data loading / preprocessing runs in background workers and the GPU is
     never stalled waiting for the CPU.
  3) _transform_y_view vectorised.
     Channel-wise log10/asinh transforms are now applied with boolean index
     masks in a single torch operation instead of a Python for-loop over
     22 channels × 3 views.
  4) Train-set evaluation every --eval_train_every epochs (default 5).
     Full pass over the training set was running every epoch; now it is
     skipped most epochs, roughly halving the epoch wall-time.
  5) checkpoint_last saved every --save_every epochs (default 5).
     Previously saved every epoch, adding non-trivial disk I/O.
  6) Normalization stats cached to norm_stats.npz in run_dir and reused on
     subsequent runs (skip recomputation when the file already exists).

Architecture: single UNetSmall, 3 views concatenated along the channel axis.
  Input  (B, 3*C_in,  H, W)
  Output (B, 3*C_out, H, W)

Channel ordering in Y (per view):
  0:    te
  1:    ti
  2-11: na (10 species)
 12-21: ua (10 species)

Usage:
  python scripts/models/UNET_3inputs/train_unet_3inputs_fast.py \
    --train_dir scripts/tensor/3images/train \
    --test_dir  scripts/tensor/3images/test \
    --run_dir   scripts/runs/unet3_fast \
    --y_channels all --epochs 50 --batch_size 32 --base 64 \
    --num_workers 4 --eval_train_every 5 --save_every 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Channel definitions  (adjust if your Y ordering differs)
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))   # te, ti, na(10)  — log10 transform
SIGNED_CHANNELS = set(range(12, 22))  # ua(10)           — asinh transform


# ---------------------------------------------------------------------------
# UNet building blocks  (plain Conv2d, no MBR overhead)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """
    Two 3×3 Conv2d + ReLU layers.
    After the block the caller applies `out = out * mask` to enforce gap zeros.
    This is a single cheap element-wise multiply vs. the old MBR approach which
    ran 3 additional full convolutions per layer.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,  c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# UNet3Views
# ---------------------------------------------------------------------------
class UNet3Views(nn.Module):
    """
    UNet that accepts 3 views concatenated along the channel axis.

    Input :  (B, 3*C_in,  H, W)
    Output:  (B, 3*C_out, H, W)

    The union mask (B, 1, H, W) is applied after every ConvBlock to keep
    gap pixels exactly zero throughout the network.  This is computationally
    near-free (one elementwise multiply) and sufficient to prevent gap bleed.
    """

    def __init__(self, c_in: int, c_out: int, base: int = 32):
        super().__init__()
        ci = 3 * c_in
        co = 3 * c_out

        self.enc1      = ConvBlock(ci,       base)
        self.enc2      = ConvBlock(base,     base * 2)
        self.enc3      = ConvBlock(base * 2, base * 4)
        self.pool      = nn.MaxPool2d(2)
        self.bottleneck= ConvBlock(base * 4, base * 8)

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1  = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, co, 1)

        self.c_in_per_view  = c_in
        self.c_out_per_view = c_out
        self.base           = base

    @staticmethod
    def _center_crop(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        th, tw = target_hw
        h, w = x.shape[-2], x.shape[-1]
        if h == th and w == tw:
            return x
        top  = (h - th) // 2
        left = (w - tw) // 2
        return x[..., top: top + th, left: left + tw]

    @staticmethod
    def _pool_mask(mask: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(mask, 2)

    def forward(self, x: torch.Tensor, mask3: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, 3*C_in, H, W)
        mask3 : (B, 3,      H, W)  — per-view binary masks

        Returns (B, 3*C_out, H, W).
        """
        # Union mask: valid if ANY view marks the pixel valid.
        mask  = mask3.max(dim=1, keepdim=True).values   # (B,1,H,W)

        # Downsampled masks for each encoder level
        m2 = self._pool_mask(mask)
        m3 = self._pool_mask(m2)
        m4 = self._pool_mask(m3)

        # ---- Encoder (apply mask after each block to zero out gaps) ----
        e1 = self.enc1(x)               * mask   # (B, base,   H,   W)
        e2 = self.enc2(self.pool(e1))   * m2     # (B, base*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   * m3     # (B, base*4, H/4, W/4)

        # ---- Bottleneck ----
        b  = self.bottleneck(self.pool(e3)) * m4  # (B, base*8, H/8, W/8)

        # ---- Decoder ----
        u3 = self.up3(F.interpolate(b,  size=e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, self._center_crop(e3, u3.shape[-2:])], dim=1)) * m3

        u2 = self.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, self._center_crop(e2, u2.shape[-2:])], dim=1)) * m2

        u1 = self.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, self._center_crop(e1, u1.shape[-2:])], dim=1)) * mask

        out = self.out_conv(d1) * mask   # (B, 3*C_out, H, W)
        return out


# ---------------------------------------------------------------------------
# Masked stats helpers (NumPy)
# ---------------------------------------------------------------------------
def _masked_den(mask_mem: np.ndarray) -> float:
    den = float(np.asarray(mask_mem, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support; cannot compute masked statistics.")
    return den


def masked_channel_stats_x(
    x_mem: np.ndarray,   # (N, 3, C_in, H, W)
    mask_ch: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mean (C_in,), std (C_in,) float32 computed over masked pixels
    across all N*3 views.  Mask channel forced to mean=0, std=1.
    """
    if x_mem.ndim != 5:
        raise ValueError(f"X must be 5D (N,3,C,H,W), got {x_mem.ndim}D.")
    N, V, C, H, W = x_mem.shape
    xf = x_mem.reshape(N * V, C, H, W).astype(np.float64)
    m  = xf[:, mask_ch:mask_ch + 1, :, :]
    den = _masked_den(m)

    mean = np.zeros(C, dtype=np.float64)
    var  = np.zeros(C, dtype=np.float64)
    for c in range(C):
        xc = xf[:, c:c + 1, :, :]
        mean[c] = float((xc * m).sum() / den)
    for c in range(C):
        xc = xf[:, c:c + 1, :, :]
        var[c] = float((((xc - mean[c]) ** 2) * m).sum() / den)

    std = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch] = 0.0
    std[mask_ch]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def masked_mean_std_transformed_y(
    y_mem:     np.ndarray,        # (N, 3, C_out_total, H, W)
    mask_mem:  np.ndarray,        # (N, 3, 1, H, W)  or  (N, 3, H, W)
    y_indices: Sequence[int],
    eps:       float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns y_mean (C_sel,), y_std (C_sel,), s_c (C_sel,)  float32.
    """
    if y_mem.ndim != 5:
        raise ValueError(f"Y must be 5D (N,3,C,H,W), got {y_mem.ndim}D.")
    N, V, C, H, W = y_mem.shape

    if mask_mem.ndim == 4:                       # (N, 3, H, W) → add channel dim
        mask_mem = mask_mem[:, :, None, :, :]

    yf  = y_mem.reshape(N * V, C, H, W).astype(np.float64)
    mf  = mask_mem.reshape(N * V, 1, H, W).astype(np.float64)
    den = _masked_den(mf)

    C_sel  = len(y_indices)
    y_mean = np.zeros(C_sel, dtype=np.float64)
    y_var  = np.zeros(C_sel, dtype=np.float64)
    s_c    = np.ones( C_sel, dtype=np.float64)

    # 1) velocity scales (signed channels only)
    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc  = yf[:, c:c + 1, :, :]
            mu  = float((yc * mf).sum() / den)
            var = float((((yc - mu) ** 2) * mf).sum() / den)
            s_c[j] = float(np.sqrt(max(var, 1e-12)))

    # 2) mean in transformed space
    for j, c in enumerate(y_indices):
        yc = yf[:, c:c + 1, :, :]
        t  = (np.log10(np.maximum(yc, 0.0) + eps) if c in POS_CHANNELS
              else np.arcsinh(yc / s_c[j]))
        y_mean[j] = float((t * mf).sum() / den)

    # 3) variance in transformed space
    for j, c in enumerate(y_indices):
        yc = yf[:, c:c + 1, :, :]
        t  = (np.log10(np.maximum(yc, 0.0) + eps) if c in POS_CHANNELS
              else np.arcsinh(yc / s_c[j]))
        y_var[j] = float((((t - y_mean[j]) ** 2) * mf).sum() / den)

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset — vectorised Y transform (no Python for-loop over channels)
# ---------------------------------------------------------------------------
class NpyTensor3Dataset(Dataset):
    """
    Loads 5D tensors (N, 3, C, H, W) produced by tensor_3_images.py.

    __getitem__ returns:
      x_cat : (3*C_in,  H, W)  float32  — views concatenated, X-normalised
      y_cat : (3*C_sel, H, W)  float32  — views concatenated, transformed+normalised
      mask3 : (3,       H, W)  float32  — per-view binary masks
    """

    def __init__(
        self,
        x_path:    Path,
        y_path:    Path,
        y_indices: Sequence[int],
        x_mean:    Optional[np.ndarray],
        x_std:     Optional[np.ndarray],
        y_mean:    np.ndarray,
        y_std:     np.ndarray,
        s_c:       np.ndarray,
        eps:       float = 1e-3,
    ):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")

        if self.X.ndim != 5:
            raise ValueError(f"X must be 5D (N,3,C,H,W), got {self.X.shape}.")
        if self.Y.ndim != 5:
            raise ValueError(f"Y must be 5D (N,3,C,H,W), got {self.Y.shape}.")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have same N.")
        if self.X.shape[1] != 3 or self.Y.shape[1] != 3:
            raise ValueError("Axis 1 (views) must equal 3.")

        self.y_indices = list(map(int, y_indices))
        self.eps = float(eps)

        # X normalisation tensors  (C_in,)
        if x_mean is not None and x_std is not None:
            self.xm = torch.from_numpy(x_mean).view(1, -1, 1, 1)  # (1,C_in,1,1)
            self.xs = torch.from_numpy(x_std ).view(1, -1, 1, 1)
        else:
            self.xm = self.xs = None

        # Y normalisation tensors  (C_sel,)
        self.ym = torch.from_numpy(np.asarray(y_mean, np.float32)).view(-1, 1, 1)
        self.ys = torch.from_numpy(np.asarray(y_std,  np.float32)).view(-1, 1, 1)
        self.sc = torch.from_numpy(np.asarray(s_c,    np.float32)).view(-1, 1, 1)

        # ---- pre-build vectorised channel masks (stored as bool tensors) ----
        C_sel = len(y_indices)
        pos_mask    = torch.zeros(C_sel, dtype=torch.bool)
        signed_mask = torch.zeros(C_sel, dtype=torch.bool)
        for j, c in enumerate(y_indices):
            if c in POS_CHANNELS:
                pos_mask[j]    = True
            elif c in SIGNED_CHANNELS:
                signed_mask[j] = True
            else:
                raise ValueError(f"Channel {c} not in POS_CHANNELS or SIGNED_CHANNELS.")
        # reshape for broadcast over (C_sel, H, W)
        self.pos_mask    = pos_mask.view(-1, 1, 1)    # (C_sel,1,1)
        self.signed_mask = signed_mask.view(-1, 1, 1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _transform_y_view(self, y_sel: torch.Tensor) -> torch.Tensor:
        """
        y_sel : (C_sel, H, W) physical units → transformed space.

        Fully vectorised: log10 branch and asinh branch each processed in one
        tensor operation, selected by pre-built boolean index masks.
        No Python for-loop over channels.
        """
        out = torch.empty_like(y_sel)

        # positive channels: log10(max(y,0) + eps)
        if self.pos_mask.any():
            pos_vals = torch.log10(
                y_sel[self.pos_mask.squeeze()].clamp(min=0.0) + self.eps
            )
            out[self.pos_mask.squeeze()] = pos_vals

        # signed channels: asinh(y / s_c)
        if self.signed_mask.any():
            # s_c shape (C_sel,1,1) — select the signed entries
            sc_signed = self.sc[self.signed_mask]  # (n_signed,1,1)
            signed_vals = torch.asinh(y_sel[self.signed_mask.squeeze()] / sc_signed)
            out[self.signed_mask.squeeze()] = signed_vals

        return out

    def __getitem__(self, idx: int):
        # load sample — numpy copy needed to detach from mmap
        x5 = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))  # (3,C_in,H,W)
        y5 = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))  # (3,C_out,H,W)

        # masks: channel 0 of each view
        mask3 = x5[:, 0, :, :]  # (3,H,W)

        # X normalisation  (broadcast over V=3)
        if self.xm is not None:
            x5 = (x5 - self.xm) / self.xs   # (3,C_in,H,W)

        # concatenate views along channel axis
        V, C_in, H, W = x5.shape
        x_cat = x5.reshape(V * C_in, H, W)  # (3*C_in,H,W)

        # Y: select channels, transform, normalise, concatenate
        # Process all 3 views in one batch using the vectorised transform
        y_sel = y5[:, self.y_indices, :, :]   # (3,C_sel,H,W)
        y_views = []
        for v in range(3):
            y_t = self._transform_y_view(y_sel[v])        # (C_sel,H,W)
            y_n = (y_t - self.ym) / self.ys               # normalised
            y_views.append(y_n)
        y_cat = torch.cat(y_views, dim=0)  # (3*C_sel,H,W)

        return x_cat, y_cat, mask3


# ---------------------------------------------------------------------------
# Loss / metrics  (channel-balanced)
# ---------------------------------------------------------------------------
def masked_mae_per_channel(
    pred:   torch.Tensor,   # (B,C,H,W)
    target: torch.Tensor,   # (B,C,H,W)
    mask:   torch.Tensor,   # (B,1,H,W)
    eps:    float = 1e-8,
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    diff = (pred - target).abs() * mask
    den  = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return diff.sum(dim=(0, 2, 3)) / den   # (C,)


def masked_rmse_per_channel(
    pred:   torch.Tensor,
    target: torch.Tensor,
    mask:   torch.Tensor,
    eps:    float = 1e-8,
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    diff2 = (pred - target).pow(2) * mask
    den   = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return torch.sqrt(diff2.sum(dim=(0, 2, 3)) / den)   # (C,)


def channel_balanced_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    return masked_mae_per_channel(pred, target, mask).mean()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model:  UNet3Views,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    mae_acc  = None
    rmse_acc = None
    n_batches = 0

    for x_cat, y_cat, mask3 in loader:
        x_cat  = x_cat.to(device, non_blocking=True)
        y_cat  = y_cat.to(device, non_blocking=True)
        mask3  = mask3.to(device, non_blocking=True)

        union_mask = mask3.max(dim=1, keepdim=True).values

        pred = model(x_cat, mask3)
        pred = pred * union_mask

        mae_c  = masked_mae_per_channel( pred, y_cat, union_mask).detach().cpu()
        rmse_c = masked_rmse_per_channel(pred, y_cat, union_mask).detach().cpu()

        if mae_acc is None:
            mae_acc  = mae_c.clone()
            rmse_acc = rmse_c.clone()
        else:
            mae_acc  += mae_c
            rmse_acc += rmse_c
        n_batches += 1

    if n_batches == 0:
        return {"mae_avg": float("nan"), "rmse_avg": float("nan"),
                "mae_per_channel": [], "rmse_per_channel": []}

    mae_acc  /= n_batches
    rmse_acc /= n_batches
    return {
        "mae_avg":          float(mae_acc.mean()),
        "rmse_avg":         float(rmse_acc.mean()),
        "mae_per_channel":  mae_acc.numpy().astype(float).tolist(),
        "rmse_per_channel": rmse_acc.numpy().astype(float).tolist(),
    }


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------
def build_checkpoint(
    cfg, model, opt, epoch, c_in, c_out_sel, c_out_total,
    y_indices, x_mean, x_std, y_mean, y_std, s_c,
    train_eval, test_eval, train_loss,
) -> dict:
    return {
        "arch":            "unet3views",
        "base":            int(cfg.base),
        "c_in_per_view":   int(c_in),
        "c_out_per_view":  int(c_out_sel),
        "c_out_total":     int(c_out_total),
        "n_views":         3,
        "y_indices":       [int(c) for c in y_indices],
        "pos_channels":    sorted(POS_CHANNELS),
        "signed_channels": sorted(SIGNED_CHANNELS),
        "eps":             float(cfg.eps),
        "x_mean":          x_mean.tolist(),
        "x_std":           x_std.tolist(),
        "y_mean":          y_mean.tolist(),
        "y_std":           y_std.tolist(),
        "s_c":             s_c.tolist(),
        "model_state":     model.state_dict(),
        "opt_state":       opt.state_dict(),
        "epoch":           int(epoch),
        "metrics_last":    {"train": train_eval, "test": test_eval,
                            "train_loss": train_loss},
    }


# ---------------------------------------------------------------------------
# Config / utilities
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    train_dir:        str
    test_dir:         str
    run_dir:          str
    y_channels:       str
    epochs:           int
    batch_size:       int
    lr:               float
    weight_decay:     float
    base:             int
    seed:             int
    device:           str
    eps:              float
    num_workers:      int
    eval_train_every: int
    save_every:       int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_channels(s: str, c_out_total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(c_out_total))
    ch = [int(p.strip()) for p in s.split(",") if p.strip()]
    if any(c < 0 or c >= c_out_total for c in ch):
        raise ValueError(f"y_channels out of bounds. Got {ch}, total={c_out_total}")
    return ch


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir",        required=True)
    ap.add_argument("--test_dir",         required=True)
    ap.add_argument("--run_dir",          required=True)
    ap.add_argument("--y_channels",       default="all")
    ap.add_argument("--epochs",           type=int,   default=50)
    ap.add_argument("--batch_size",       type=int,   default=32)
    ap.add_argument("--lr",               type=float, default=1e-3)
    ap.add_argument("--weight_decay",     type=float, default=0.0)
    ap.add_argument("--base",             type=int,   default=64)
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps",              type=float, default=1e-5)
    ap.add_argument("--prefix",           default="global3")
    # ---- new performance knobs ----
    ap.add_argument("--num_workers",      type=int,   default=4,
                    help="DataLoader worker processes (0 = main thread only, slow)")
    ap.add_argument("--eval_train_every", type=int,   default=5,
                    help="Run full train-set evaluation every N epochs (0 = never)")
    ap.add_argument("--save_every",       type=int,   default=5,
                    help="Save checkpoint_last every N epochs")
    args = ap.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir, test_dir=args.test_dir, run_dir=args.run_dir,
        y_channels=args.y_channels, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, base=args.base,
        seed=args.seed, device=args.device, eps=float(args.eps),
        num_workers=args.num_workers,
        eval_train_every=args.eval_train_every,
        save_every=args.save_every,
    )
    set_seed(cfg.seed)

    train_dir = Path(cfg.train_dir)
    test_dir  = Path(cfg.test_dir)
    run_dir   = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pfx     = args.prefix
    train_x = train_dir / f"{pfx}_X_img_train.npy"
    train_y = train_dir / f"{pfx}_Y_img_train.npy"
    test_x  = test_dir  / f"{pfx}_X_img_test.npy"
    test_y  = test_dir  / f"{pfx}_Y_img_test.npy"

    for p in [train_x, train_y, test_x, test_y]:
        if not p.exists():
            raise FileNotFoundError(f"Required tensor not found: {p}")

    # ---- shapes ----
    X_train = np.load(train_x, mmap_mode="r")
    Y_train = np.load(train_y, mmap_mode="r")

    if X_train.ndim != 5 or Y_train.ndim != 5:
        raise ValueError("Expected 5D tensors (N,3,C,H,W).")
    if X_train.shape[1] != 3 or Y_train.shape[1] != 3:
        raise ValueError("Axis-1 (views) must equal 3.")

    c_in        = int(X_train.shape[2])
    c_out_total = int(Y_train.shape[2])
    y_indices   = parse_channels(cfg.y_channels, c_out_total)
    c_out_sel   = len(y_indices)

    print(f"Tensor shapes  X={X_train.shape}  Y={Y_train.shape}")
    print(f"c_in={c_in}  c_out_total={c_out_total}  selected={c_out_sel}")
    print(f"UNet  in_channels={3*c_in}  out_channels={3*c_out_sel}  base={cfg.base}")

    # ---- normalization stats — cached to disk so rerun skips recomputation ----
    norm_path = run_dir / "norm_stats.npz"
    if norm_path.exists():
        print(f"Loading cached norm stats from {norm_path}")
        nz       = np.load(norm_path)
        x_mean   = nz["x_mean"].astype(np.float32)
        x_std    = nz["x_std" ].astype(np.float32)
        y_mean   = nz["y_mean"].astype(np.float32)
        y_std    = nz["y_std" ].astype(np.float32)
        s_c      = nz["s_c"   ].astype(np.float32)
    else:
        print("Computing X normalisation stats …")
        x_mean, x_std = masked_channel_stats_x(X_train, mask_ch=0)

        mask_train = X_train[:, :, 0:1, :, :].astype(np.float32)
        print("Computing Y normalisation stats …")
        y_mean, y_std, s_c = masked_mean_std_transformed_y(
            Y_train, mask_train, y_indices=y_indices, eps=cfg.eps
        )
        np.savez_compressed(norm_path,
                            x_mean=x_mean, x_std=x_std,
                            y_mean=y_mean, y_std=y_std, s_c=s_c)
        print(f"Saved norm stats → {norm_path}")

    # ---- datasets / dataloaders ----
    ds_train = NpyTensor3Dataset(train_x, train_y, y_indices,
                                 x_mean, x_std, y_mean, y_std, s_c, cfg.eps)
    ds_test  = NpyTensor3Dataset(test_x,  test_y,  y_indices,
                                 x_mean, x_std, y_mean, y_std, s_c, cfg.eps)

    device  = torch.device(cfg.device)
    pin     = device.type == "cuda"
    # num_workers > 0 enables background prefetch so the GPU never idles on I/O
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=pin,
                          persistent_workers=(cfg.num_workers > 0))
    dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=pin,
                          persistent_workers=(cfg.num_workers > 0))

    # ---- model ----
    model    = UNet3Views(c_in=c_in, c_out=c_out_sel, base=cfg.base).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_json(run_dir / "config.json", asdict(cfg))

    best_rmse    = float("inf")
    metrics_hist = {"train": [], "test": []}

    # ---- training loop ----
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_steps  = 0

        for x_cat, y_cat, mask3 in dl_train:
            # non_blocking=True overlaps H→D transfer with GPU work
            x_cat  = x_cat.to(device, non_blocking=True)
            y_cat  = y_cat.to(device, non_blocking=True)
            mask3  = mask3.to(device, non_blocking=True)

            union_mask = mask3.max(dim=1, keepdim=True).values

            pred = model(x_cat, mask3)
            pred = pred * union_mask

            loss = channel_balanced_loss(pred, y_cat, union_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())
            n_steps  += 1

        train_loss = loss_sum / max(n_steps, 1)

        # ---- evaluation ----
        test_eval  = evaluate(model, dl_test, device)

        # train eval only every eval_train_every epochs (expensive)
        do_train_eval = (cfg.eval_train_every > 0 and epoch % cfg.eval_train_every == 0)
        train_eval    = evaluate(model, dl_train, device) if do_train_eval else {}

        metrics_hist["train"].append({"epoch": epoch, "loss": train_loss, **train_eval})
        metrics_hist["test" ].append({"epoch": epoch, **test_eval})

        cur_rmse = float(test_eval["rmse_avg"])

        # ---- checkpoint best (always) ----
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            ckpt = build_checkpoint(
                cfg, model, opt, epoch, c_in, c_out_sel, c_out_total,
                y_indices, x_mean, x_std, y_mean, y_std, s_c,
                train_eval, test_eval, train_loss,
            )
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        # ---- checkpoint last (every save_every epochs) ----
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            ckpt = build_checkpoint(
                cfg, model, opt, epoch, c_in, c_out_sel, c_out_total,
                y_indices, x_mean, x_std, y_mean, y_std, s_c,
                train_eval, test_eval, train_loss,
            )
            torch.save(ckpt, run_dir / "checkpoint_last.pt")

        save_json(run_dir / "metrics.json", metrics_hist)

        train_info = (f"  train_mae={train_eval['mae_avg']:.6g}"
                      if "mae_avg" in train_eval else "")
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6g} | "
            f"test_mae={float(test_eval['mae_avg']):.6g} | "
            f"test_rmse={float(test_eval['rmse_avg']):.6g}"
            f"{train_info}"
        )

    print(f"Done. Best test rmse_avg={best_rmse:.6g}")


if __name__ == "__main__":
    main()



# Example command:
# python scripts/models/UNET_3inputs/train_unet_3inputs.py \
#   --train_dir scripts/tensor/3images/train \
#   --test_dir  scripts/tensor/3images/test \
#   --run_dir   scripts/runs/unet3_width64 \
#   --y_channels all --epochs 50 --batch_size 16 --base 64 
#   --num_workers 4 --eval_train_every 5 --save_every 5
