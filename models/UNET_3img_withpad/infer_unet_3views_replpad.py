#!/usr/bin/env python3
"""
infer_unet_3views.py

Run inference with a UNet checkpoint produced by train_unet_3views.py.

- Loads the checkpoint (contains all normalisation stats + view tag).
- Normalises X, predicts in transformed space, denormalises + inverse-transforms
  to physical units.
- Saves predictions and per-channel masked MAE/RMSE.

Usage examples
--------------
# Infer with view0 checkpoint
python scripts/models/UNET_3views/infer_unet_3views.py \
    --checkpoint scripts/runs/unet_3views/view0/checkpoint_best.pt \
    --test_prefix  scripts/tensor/3views_4d/test/global3v \
    --test_split   test \
    --out_dir      scripts/runs/unet_3views/view0/infer_test \
    --batch_size   32

# Infer with view1 checkpoint
python scripts/models/UNET_3views/infer_unet_3views.py \
    --checkpoint scripts/runs/unet_3views/view1/checkpoint_best.pt \
    --test_prefix  scripts/tensor/3views_4d/test/global3v \
    --test_split   test \
    --out_dir      scripts/runs/unet_3views/view1/infer_test

Output files (under --out_dir)
-------------------------------
  pred_Y_img_test.npy   (N, C_sel, H, W) float32 in physical units
  test_metrics.json     per-channel MAE/RMSE + averages

Notes
-----
- The view index is read from the checkpoint; you do NOT need to pass --view.
- Pass --checkpoint_last to use checkpoint_last.pt instead of checkpoint_best.pt.

Modification vs infer_unet_3views.py
--------------------------------------
ConvBlock uses replicate padding (padding_mode="replicate") to match
train_unet_3views_replpad.py.  The arch tag in the checkpoint is
"unet_small_replpad".
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
# Channel category sets (must match train_unet_3views.py)
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


# ---------------------------------------------------------------------------
# Dataset (X-only, with normalisation)
# ---------------------------------------------------------------------------

class ViewXDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        x_mean: Optional[np.ndarray],
        x_std:  Optional[np.ndarray],
    ):
        self.X      = np.load(x_path, mmap_mode="r")   # (N, C, H, W)
        self.x_mean = x_mean
        self.x_std  = x_std

        if self.X.ndim != 4:
            raise ValueError("Expected X to be 4D: (N,C,H,W).")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        m = x[0:1]  # mask channel

        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x  = (x - xm) / xs

        return x, m


# ---------------------------------------------------------------------------
# Model — UNetSmall (identical to train_unet_3views.py)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two-layer conv block with replicate padding (matches train_unet_3views_replpad.py)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in,  c_out, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class UNetSmall(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32):
        super().__init__()
        self.enc1       = ConvBlock(c_in,    base)
        self.enc2       = ConvBlock(base,    base * 2)
        self.enc3       = ConvBlock(base * 2, base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1  = nn.Conv2d(base * 2, base,     1)
        self.dec1 = ConvBlock(base * 2, base)
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _crop(x: torch.Tensor, hw) -> torch.Tensor:
        th, tw = hw
        dh, dw = x.shape[-2] - th, x.shape[-1] - tw
        return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        u3 = self.up3(F.interpolate(b,  size=e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, self._crop(e3, u3.shape[-2:])], 1))

        u2 = self.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, self._crop(e2, u2.shape[-2:])], 1))

        u1 = self.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, self._crop(e1, u1.shape[-2:])], 1))

        return self.out(d1)


# ---------------------------------------------------------------------------
# Inverse transform  (transformed space → physical units)
# ---------------------------------------------------------------------------

def inverse_transform_y(
    y_t:             np.ndarray,          # (N, C_sel, H, W) in transformed space
    y_indices:       Sequence[int],
    pos_channels:    Sequence[int],
    signed_channels: Sequence[int],
    eps:             float,
    s_c:             np.ndarray,          # (C_sel,)
) -> np.ndarray:
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)
    out        = np.empty_like(y_t, dtype=np.float32)

    for j, orig_c in enumerate(y_indices):
        if int(orig_c) in pos_set:
            out[:, j] = (10.0 ** y_t[:, j] - eps).astype(np.float32)
        elif int(orig_c) in signed_set:
            out[:, j] = (float(s_c[j]) * np.sinh(y_t[:, j])).astype(np.float32)
        else:
            raise ValueError(f"Channel {orig_c} not categorised as POS or SIGNED.")

    return out


# ---------------------------------------------------------------------------
# Masked metrics (NumPy, physical space)
# ---------------------------------------------------------------------------

def masked_mae_per_channel_np(
    pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred - y_true) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(
    pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred - y_true) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer with a view-specific UNet checkpoint from train_unet_3views.py."
    )
    ap.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint_best.pt (or checkpoint_last.pt) from train_unet_3views.py."
    )
    ap.add_argument(
        "--test_prefix", required=True,
        help=(
            "Path prefix for test tensors (same convention as train). "
            "E.g. scripts/tensor/3views_4d/test/global3v"
        ),
    )
    ap.add_argument("--test_split",  default="test",
                    help="Split tag for the test tensors (default: test).")
    ap.add_argument("--out_dir",     required=True,
                    help="Directory to write predictions and metrics.")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    view_tag         = str(ckpt["view_tag"])          # e.g. "view0"
    c_in             = int(ckpt["c_in"])
    c_out            = int(ckpt["c_out"])
    base             = int(ckpt.get("base", 32))
    y_indices        = [int(c) for c in ckpt["y_indices"]]
    pos_channels     = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels  = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))
    eps              = float(ckpt.get("eps", 1e-3))

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32) if ckpt.get("x_mean") is not None else None
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32) if ckpt.get("x_std")  is not None else None
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)

    # ---- resolve tensor paths from checkpoint's view tag ----
    test_x_path = Path(f"{args.test_prefix}_{view_tag}_X_img_{args.test_split}.npy")
    test_y_path = Path(f"{args.test_prefix}_{view_tag}_Y_img_{args.test_split}.npy")

    for p in (test_x_path, test_y_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Inference: {view_tag} ===")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Test X     : {test_x_path}")
    print(f"  c_in={c_in}  c_out={c_out}  y_indices={y_indices}")

    # ---- model ----
    device = torch.device(args.device)
    model  = UNetSmall(c_in=c_in, c_out=c_out, base=base).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- data loader ----
    ds = ViewXDataset(test_x_path, x_mean=x_mean, x_std=x_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=(device.type == "cuda"))

    N, _, H, W = np.load(test_x_path, mmap_mode="r").shape
    preds_t_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
    masks        = np.zeros((N, 1,    H, W), dtype=np.float32)

    idx0 = 0
    for x, m in dl:
        b  = int(x.shape[0])
        x  = x.to(device)
        m  = m.to(device)
        with torch.no_grad():
            pred = model(x) * m     # enforce gaps = 0 in normalised space
        preds_t_norm[idx0:idx0 + b] = pred.cpu().numpy()
        masks[idx0:idx0 + b]        = m.cpu().numpy()
        idx0 += b

    # ---- denormalise → transformed space ----
    ym     = y_mean.reshape(1, -1, 1, 1)
    ys     = y_std.reshape( 1, -1, 1, 1)
    preds_t = preds_t_norm * ys + ym

    # ---- inverse transform → physical units ----
    preds_phys = inverse_transform_y(
        preds_t, y_indices=y_indices,
        pos_channels=pos_channels, signed_channels=signed_channels,
        eps=eps, s_c=s_c,
    )
    preds_phys *= masks   # enforce gaps = 0 in physical space

    # ---- save predictions ----
    pred_path = out_dir / "pred_Y_img_test.npy"
    np.save(pred_path, preds_phys)
    print(f"Saved predictions: {pred_path}  shape={preds_phys.shape}")

    # ---- evaluate against ground truth ----
    Y_full = np.load(test_y_path, mmap_mode="r")
    Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

    mae_c  = masked_mae_per_channel_np(preds_phys, Y_sel, masks)
    rmse_c = masked_rmse_per_channel_np(preds_phys, Y_sel, masks)

    metrics = {
        "checkpoint":        str(ckpt_path),
        "view":              ckpt["view"],
        "view_tag":          view_tag,
        "arch":              "unet_small_replpad",
        "base":              base,
        "c_in":              c_in,
        "c_out":             c_out,
        "y_indices":         y_indices,
        "mae_avg":           float(np.mean(mae_c)),
        "rmse_avg":          float(np.mean(rmse_c)),
        "mae_per_channel":   mae_c.tolist(),
        "rmse_per_channel":  rmse_c.tolist(),
    }

    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test MAE avg  : {metrics['mae_avg']:.6g}")
    print(f"Test RMSE avg : {metrics['rmse_avg']:.6g}")
    print(f"Metrics saved : {metrics_path}")


if __name__ == "__main__":
    main()
