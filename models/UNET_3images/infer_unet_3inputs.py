#!/usr/bin/env python3
"""
infer_unet_3inputs.py

Inference counterpart to train_unet_3inputs.py.

Loads a checkpoint produced by train_unet_3inputs.py and runs inference on
the 3-view test tensors produced by tensor_3_images.py.

Input tensors (5D):
  global3_X_img_test.npy   (N, 3, C_in,       H, W)
  global3_Y_img_test.npy   (N, 3, C_out_total, H, W)  — ground truth for metrics

Outputs written to --out_dir:
  pred_Y_img_test.npy     (N, 3, C_sel, H, W)  float32, physical units
  test_metrics.json        per-channel MAE/RMSE (physical units) + averages

Usage:
  python scripts/models/UNET_3inputs/infer_unet_3inputs.py \
    --checkpoint scripts/runs/unet3_width64/checkpoint_best.pt \
    --test_dir   scripts/tensor/3images/test \
    --out_dir    scripts/runs/unet3_width64/infer_test \
    --batch_size 16 --prefix global3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Channel definitions  (must match train_unet_3inputs.py)
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


# ---------------------------------------------------------------------------
# Mask-boundary dilation helper
# ---------------------------------------------------------------------------
def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    kernel = torch.ones(1, 1, k, k, dtype=mask.dtype, device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=radius)
    return (dilated > 0).float()


# ---------------------------------------------------------------------------
# MaskBoundaryReinjectionConv  (identical to train file)
# ---------------------------------------------------------------------------
class MaskBoundaryReinjectionConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, bias=bias)
        self.reinject_radius = kernel_size // 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        radius = self.reinject_radius
        out = self.conv(x)

        if radius <= 0:
            return out * mask

        gap        = 1.0 - mask
        gap_dilated    = _dilate_mask(gap, radius)
        boundary_ring  = mask * gap_dilated

        with torch.no_grad():
            ks  = self.conv.kernel_size[0]
            pad = self.conv.padding[0]
            valid_count = F.conv2d(
                mask,
                torch.ones(1, 1, ks, ks, device=mask.device, dtype=mask.dtype),
                padding=pad,
            )
            scale = float(ks * ks) / valid_count.clamp(min=1.0)

        out_rescaled = out * scale

        valid_interior = mask * (1.0 - boundary_ring)
        out_final = (
            out          * valid_interior
            + out_rescaled * boundary_ring
        )
        return out_final


# ---------------------------------------------------------------------------
# ConvBlockMBR
# ---------------------------------------------------------------------------
class ConvBlockMBR(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = MaskBoundaryReinjectionConv(c_in,  c_out, 3, 1)
        self.conv2 = MaskBoundaryReinjectionConv(c_out, c_out, 3, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, mask), inplace=True)
        x = F.relu(self.conv2(x, mask), inplace=True)
        return x


# ---------------------------------------------------------------------------
# UNet3Views  (identical to train file)
# ---------------------------------------------------------------------------
class UNet3Views(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32):
        super().__init__()
        ci = 3 * c_in
        co = 3 * c_out

        self.enc1 = ConvBlockMBR(ci,       base)
        self.enc2 = ConvBlockMBR(base,     base * 2)
        self.enc3 = ConvBlockMBR(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlockMBR(base * 4, base * 8)

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlockMBR(base * 8, base * 4)

        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlockMBR(base * 4, base * 2)

        self.up1  = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlockMBR(base * 2, base)

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
        mask = mask3.max(dim=1, keepdim=True).values

        e1 = self.enc1(x, mask)
        m2 = self._pool_mask(mask)
        e2 = self.enc2(self.pool(e1), m2)
        m3 = self._pool_mask(m2)
        e3 = self.enc3(self.pool(e2), m3)

        m4 = self._pool_mask(m3)
        b  = self.bottleneck(self.pool(e3), m4)

        u3 = F.interpolate(b,  size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        e3c = self._center_crop(e3, u3.shape[-2:])
        d3 = self.dec3(torch.cat([u3, e3c], dim=1), m3)

        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        e2c = self._center_crop(e2, u2.shape[-2:])
        d2 = self.dec2(torch.cat([u2, e2c], dim=1), m2)

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        e1c = self._center_crop(e1, u1.shape[-2:])
        d1 = self.dec1(torch.cat([u1, e1c], dim=1), mask)

        out = self.out_conv(d1)
        out = out * mask
        return out


# ---------------------------------------------------------------------------
# Dataset: X only (inference)
# ---------------------------------------------------------------------------
class NpyX3Dataset(Dataset):
    """
    Loads X (N, 3, C_in, H, W), normalizes, concatenates views → (3*C_in, H, W).
    Also returns mask3 (3, H, W).
    """

    def __init__(
        self,
        x_path: Path,
        x_mean: Optional[np.ndarray],
        x_std:  Optional[np.ndarray],
    ):
        self.X      = np.load(x_path, mmap_mode="r")
        self.x_mean = x_mean
        self.x_std  = x_std

        if self.X.ndim != 5:
            raise ValueError(f"X must be 5D (N,3,C,H,W), got {self.X.shape}.")
        if self.X.shape[1] != 3:
            raise ValueError("Axis-1 (views) must equal 3.")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x5 = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))  # (3, C_in, H, W)
        mask3 = x5[:, 0, :, :]  # (3, H, W)

        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(1, -1, 1, 1)
            xs = torch.from_numpy(self.x_std ).view(1, -1, 1, 1)
            x5 = (x5 - xm) / xs

        V, C_in, H, W = x5.shape
        x_cat = x5.reshape(V * C_in, H, W)  # (3*C_in, H, W)
        return x_cat, mask3


# ---------------------------------------------------------------------------
# Inverse transforms
# ---------------------------------------------------------------------------
def inverse_transform_y(
    y_pred_t:       np.ndarray,      # (N, 3*C_sel, H, W)  transformed, NOT normalized
    y_indices:      Sequence[int],
    pos_channels:   Sequence[int],
    signed_channels: Sequence[int],
    eps:            float,
    s_c:            np.ndarray,       # (C_sel,)
) -> np.ndarray:
    """
    Inverse-transforms a (N, 3*C_sel, H, W) tensor.
    The channel axis is ordered as [view0_ch0, view0_ch1, ..., view1_ch0, ...].
    """
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)
    C_sel      = len(y_indices)

    out = np.empty_like(y_pred_t, dtype=np.float32)
    for v in range(3):
        for j, orig_c in enumerate(y_indices):
            ch = v * C_sel + j
            if int(orig_c) in pos_set:
                out[:, ch] = (np.power(10.0, y_pred_t[:, ch]) - eps).astype(np.float32)
            elif int(orig_c) in signed_set:
                out[:, ch] = (s_c[j] * np.sinh(y_pred_t[:, ch])).astype(np.float32)
            else:
                raise ValueError(f"Channel {orig_c} not categorized.")
    return out


# ---------------------------------------------------------------------------
# Masked metrics  (numpy, physical space)
# ---------------------------------------------------------------------------
def masked_mae_per_channel_np(
    pred:   np.ndarray,  # (N, C, H, W)
    y_true: np.ndarray,  # (N, C, H, W)
    mask:   np.ndarray,  # (N, 1, H, W)
    eps:    float = 1e-8,
) -> np.ndarray:
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred - y_true) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(
    pred:   np.ndarray,
    y_true: np.ndarray,
    mask:   np.ndarray,
    eps:    float = 1e-8,
) -> np.ndarray:
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred - y_true) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint_best.pt from train_unet_3inputs.py")
    ap.add_argument("--test_dir",   required=True,
                    help="Folder with global3_X_img_test.npy / global3_Y_img_test.npy")
    ap.add_argument("--out_dir",    required=True,
                    help="Folder to write predictions and metrics")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefix",     default="global3",
                    help="Tensor file prefix (default: global3)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    test_dir  = Path(args.test_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pfx   = args.prefix
    X_path = test_dir / f"{pfx}_X_img_test.npy"
    Y_path = test_dir / f"{pfx}_Y_img_test.npy"
    for p in [X_path, Y_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required tensor not found: {p}")

    # ---- load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    arch = str(ckpt.get("arch", "unet3views"))
    if arch != "unet3views":
        raise RuntimeError(f"Unsupported arch in checkpoint: {arch}")

    c_in        = int(ckpt["c_in_per_view"])
    c_out       = int(ckpt["c_out_per_view"])
    base        = int(ckpt.get("base", 32))
    y_indices   = [int(x) for x in ckpt["y_indices"]]
    pos_channels    = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))
    eps         = float(ckpt.get("eps", 1e-3))

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32) if ckpt.get("x_mean") is not None else None
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32) if ckpt.get("x_std")  is not None else None
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)

    C_sel = len(y_indices)

    # ---- model ----
    device = torch.device(args.device)
    model = UNet3Views(c_in=c_in, c_out=c_out, base=base).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- dataset / loader ----
    ds = NpyX3Dataset(X_path, x_mean=x_mean, x_std=x_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=(device.type == "cuda"))

    X_raw = np.load(X_path, mmap_mode="r")
    N, _, _, H, W = X_raw.shape

    # predictions in normalized transformed space: (N, 3*C_sel, H, W)
    preds_t_norm = np.zeros((N, 3 * C_sel, H, W), dtype=np.float32)
    masks_union  = np.zeros((N, 1,         H, W), dtype=np.float32)

    idx0 = 0
    for x_cat, mask3 in dl:
        b      = int(x_cat.shape[0])
        x_cat  = x_cat.to(device)
        mask3  = mask3.to(device)
        union  = mask3.max(dim=1, keepdim=True).values  # (B,1,H,W)

        with torch.no_grad():
            pred_n = model(x_cat, mask3)
            pred_n = pred_n * union

        preds_t_norm[idx0:idx0 + b] = pred_n.cpu().numpy().astype(np.float32)
        masks_union [idx0:idx0 + b] = union.cpu().numpy().astype(np.float32)
        idx0 += b

    # ---- denormalize → transformed space ----
    # y_mean/y_std are (C_sel,); broadcast over 3 views and spatial dims
    ym = np.tile(y_mean, 3).reshape(1, 3 * C_sel, 1, 1)  # (1, 3*C_sel, 1, 1)
    ys = np.tile(y_std,  3).reshape(1, 3 * C_sel, 1, 1)
    preds_t = preds_t_norm * ys + ym

    # ---- inverse transform → physical units ----
    preds_phys = inverse_transform_y(
        preds_t,
        y_indices=y_indices,
        pos_channels=pos_channels,
        signed_channels=signed_channels,
        eps=eps,
        s_c=s_c,
    )

    # apply mask
    preds_phys *= masks_union  # (N, 3*C_sel, H, W)

    # ---- reshape to (N, 3, C_sel, H, W) and save ----
    preds_5d = preds_phys.reshape(N, 3, C_sel, H, W)
    np.save(out_dir / "pred_Y_img_test.npy", preds_5d)

    # ---- metrics against ground truth ----
    Y_full = np.load(Y_path, mmap_mode="r")  # (N, 3, C_out_total, H, W)
    # select channels and flatten views for metric computation
    Y_sel  = Y_full[:, :, y_indices, :, :]   # (N, 3, C_sel, H, W)
    Y_flat = Y_sel.reshape(N, 3 * C_sel, H, W).astype(np.float32)

    mae_c  = masked_mae_per_channel_np( preds_phys, Y_flat, masks_union)
    rmse_c = masked_rmse_per_channel_np(preds_phys, Y_flat, masks_union)

    metrics = {
        "checkpoint":        str(ckpt_path),
        "arch":              arch,
        "base":              base,
        "c_in_per_view":     c_in,
        "c_out_per_view":    c_out,
        "n_views":           3,
        "y_indices":         y_indices,
        "mae_avg":           float(np.mean(mae_c)),
        "rmse_avg":          float(np.mean(rmse_c)),
        "mae_per_channel":   mae_c.tolist(),
        "rmse_per_channel":  rmse_c.tolist(),
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions : {out_dir / 'pred_Y_img_test.npy'}  shape={preds_5d.shape}")
    print(f"Test MAE avg : {metrics['mae_avg']:.6g}")
    print(f"Test RMSE avg: {metrics['rmse_avg']:.6g}")


if __name__ == "__main__":
    main()

# Example command:
# python scripts/models/UNET_3inputs/infer_unet_3inputs.py \
#   --checkpoint scripts/runs/unet3_width64/checkpoint_best.pt \
#   --test_dir   scripts/tensor/3images/test \
#   --out_dir    scripts/runs/unet3_width64/infer_test \
#   --batch_size 16 --prefix global3
