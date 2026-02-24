#!/usr/bin/env python3
"""
infer_unet.py

Drop-in replacement for infer_cnn.py matching train_unet.py.

- Loads checkpoint_best.pt or checkpoint_last.pt produced by train_unet.py
- Normalizes X using stored x_mean/x_std
- Predicts in normalized transformed space
- Denormalizes + inverse-transforms to physical units (per channel type)
- Saves:
  - pred_Y_img_test.npy  (N, C_sel, H, W) in physical units
  - test_metrics.json    per-channel MAE/RMSE (physical units) + averages
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


# ----------------------------
# Dataset: X only (+ mask), with X normalization
# ----------------------------
class NpyXDataset(Dataset):
    def __init__(self, x_path: Path, x_mean: Optional[np.ndarray], x_std: Optional[np.ndarray]):
        self.X = np.load(x_path, mmap_mode="r")  # (N,C,H,W)
        if self.X.ndim != 4:
            raise ValueError("Expected X to be 4D: (N,C,H,W).")
        if self.X.shape[1] < 1:
            raise ValueError("X must have mask in channel 0.")
        self.x_mean = x_mean
        self.x_std = x_std

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        m = x[0:1]

        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x = (x - xm) / xs

        return x, m


# ----------------------------
# Model (same UNetSmall)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(c_in, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _center_crop(x: torch.Tensor, target_hw) -> torch.Tensor:
        th, tw = target_hw
        h, w = x.shape[-2], x.shape[-1]
        dh = h - th
        dw = w - tw
        if dh == 0 and dw == 0:
            return x
        top = dh // 2
        left = dw // 2
        return x[..., top : top + th, left : left + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        e3c = self._center_crop(e3, u3.shape[-2:])
        d3 = self.dec3(torch.cat([u3, e3c], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        e2c = self._center_crop(e2, u2.shape[-2:])
        d2 = self.dec2(torch.cat([u2, e2c], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        e1c = self._center_crop(e1, u1.shape[-2:])
        d1 = self.dec1(torch.cat([u1, e1c], dim=1))

        return self.out(d1)


# ----------------------------
# Masked metrics in physical space (per channel)
# ----------------------------
def masked_mae_per_channel_np(pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # pred,y_true: (N,C,H,W), mask: (N,1,H,W)
    m = mask.astype(np.float32)
    m = np.broadcast_to(m, pred.shape)
    den = m.sum(axis=(0, 2, 3))
    den = np.maximum(den, eps)
    num = (np.abs(pred - y_true) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = mask.astype(np.float32)
    m = np.broadcast_to(m, pred.shape)
    den = m.sum(axis=(0, 2, 3))
    den = np.maximum(den, eps)
    num = (((pred - y_true) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


# ----------------------------
# Inverse transforms
# ----------------------------
def inverse_transform_y(
    y_pred_t: np.ndarray,        # (N,C,H,W) in transformed space (NOT normalized)
    y_indices: Sequence[int],    # original channel indices
    pos_channels: Sequence[int],
    signed_channels: Sequence[int],
    eps: float,
    s_c: np.ndarray,             # (C,) with scale for signed channels, 1.0 for pos
) -> np.ndarray:
    pos_set = set(int(x) for x in pos_channels)
    signed_set = set(int(x) for x in signed_channels)

    out = np.empty_like(y_pred_t, dtype=np.float32)
    for j, orig_c in enumerate(y_indices):
        if int(orig_c) in pos_set:
            out[:, j, :, :] = (np.power(10.0, y_pred_t[:, j, :, :]) - eps).astype(np.float32)
        elif int(orig_c) in signed_set:
            out[:, j, :, :] = (s_c[j] * np.sinh(y_pred_t[:, j, :, :])).astype(np.float32)
        else:
            raise ValueError(f"Channel index {orig_c} not categorized as POS or SIGNED.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt or checkpoint_last.pt from train_unet.py")
    ap.add_argument("--test_dir", required=True, help="Folder containing global_X_img_test.npy and global_Y_img_test.npy")
    ap.add_argument("--out_dir", required=True, help="Folder to write predictions and metrics")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = test_dir / "global_X_img_test.npy"
    Y_path = test_dir / "global_Y_img_test.npy"
    if not X_path.exists() or not Y_path.exists():
        raise FileNotFoundError("Expected global_X_img_test.npy and global_Y_img_test.npy in test_dir.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    c_in = int(ckpt["c_in"])
    c_out = int(ckpt["c_out"])
    base = int(ckpt.get("base", 32))
    arch = str(ckpt.get("arch", "unet_small"))

    y_indices = [int(x) for x in ckpt["y_indices"]]
    pos_channels = ckpt.get("pos_channels", list(range(0, 12)))
    signed_channels = ckpt.get("signed_channels", list(range(12, 22)))
    eps = float(ckpt.get("eps", 1e-3))

    x_mean = np.asarray(ckpt.get("x_mean", None), dtype=np.float32) if ckpt.get("x_mean", None) is not None else None
    x_std = np.asarray(ckpt.get("x_std", None), dtype=np.float32) if ckpt.get("x_std", None) is not None else None
    y_mean = np.asarray(ckpt.get("y_mean", None), dtype=np.float32)
    y_std = np.asarray(ckpt.get("y_std", None), dtype=np.float32)
    s_c = np.asarray(ckpt.get("s_c", None), dtype=np.float32)

    if arch != "unet_small":
        raise RuntimeError(f"Unsupported arch in checkpoint: {arch}")

    device = torch.device(args.device)

    model = UNetSmall(c_in=c_in, c_out=c_out, base=base).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = NpyXDataset(X_path, x_mean=x_mean, x_std=x_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Shapes
    X_shape = np.load(X_path, mmap_mode="r").shape
    N, _, H, W = X_shape

    preds_t_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
    masks = np.zeros((N, 1, H, W), dtype=np.float32)

    idx0 = 0
    for x, m in dl:
        b = int(x.shape[0])
        x = x.to(device)
        m = m.to(device)

        with torch.no_grad():
            pred_n = model(x)               # normalized transformed space
            pred_n = pred_n * m             # enforce gaps = 0 in normalized space

        preds_t_norm[idx0:idx0 + b] = pred_n.detach().cpu().numpy().astype(np.float32)
        masks[idx0:idx0 + b] = m.detach().cpu().numpy().astype(np.float32)
        idx0 += b

    # denormalize to transformed space
    ym = y_mean.reshape(1, -1, 1, 1)
    ys = y_std.reshape(1, -1, 1, 1)
    preds_t = preds_t_norm * ys + ym

    # inverse transform to physical units
    preds_phys = inverse_transform_y(preds_t, y_indices=y_indices, pos_channels=pos_channels, signed_channels=signed_channels, eps=eps, s_c=s_c)

    # apply mask in physical units (gaps = 0)
    preds_phys *= masks

    np.save(out_dir / "pred_Y_img_test.npy", preds_phys)

    # load ground truth physical Y for selected channels
    Y_full = np.load(Y_path, mmap_mode="r")
    Y = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

    mae_c = masked_mae_per_channel_np(preds_phys, Y, masks)
    rmse_c = masked_rmse_per_channel_np(preds_phys, Y, masks)

    metrics = {
        "checkpoint": str(ckpt_path),
        "arch": arch,
        "base": base,
        "c_in": c_in,
        "c_out": c_out,
        "y_indices": y_indices,
        "mae_avg": float(np.mean(mae_c)),
        "rmse_avg": float(np.mean(rmse_c)),
        "mae_per_channel": mae_c.tolist(),
        "rmse_per_channel": rmse_c.tolist(),
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions: {out_dir / 'pred_Y_img_test.npy'}  shape={preds_phys.shape}")
    print(f"Test MAE avg: {metrics['mae_avg']:.6g} | RMSE avg: {metrics['rmse_avg']:.6g}")


if __name__ == "__main__":
    main()

#n.b. **something** indicates that the value inside can be changed based on which tensor/directory you want to analyze

#python scripts/models/UNET_distinct_channels/infer_unet.py --checkpoint scripts/runs/**unet_50_width64**/checkpoint_best.pt 
# --test_dir scripts/tensor/**nopad_nofill_geom**/test --out_dir scripts/runs/**unet_50_width64**/infer_test --batch_size 32