# infer_cnn.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class NpyXDataset(Dataset):
    def __init__(self, x_path: Path, x_mean: Optional[np.ndarray] = None, x_std: Optional[np.ndarray] = None):
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
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))  # copy slice from memmap
        m = x[0:1]  # (1,H,W)

        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x = (x - xm) / xs

        return x, m


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


def masked_mae_np(pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    # pred,y_true: (N,C,H,W), mask: (N,1,H,W)
    m = mask.astype(np.float32)
    m = np.broadcast_to(m, pred.shape)
    num = np.abs(pred - y_true) * m
    den = float(m.sum())
    if den <= eps:
        return float("nan")
    return float(num.sum() / den)


def masked_rmse_np(pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    m = mask.astype(np.float32)
    m = np.broadcast_to(m, pred.shape)
    num = ((pred - y_true) ** 2) * m
    den = float(m.sum())
    if den <= eps:
        return float("nan")
    return float(np.sqrt(num.sum() / den))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt or checkpoint_last.pt")
    ap.add_argument("--test_dir", required=True, help="Folder containing global_X_img_test.npy and global_Y_img_test.npy")
    ap.add_argument("--out_dir", required=True, help="Folder to write predictions and metrics")
    ap.add_argument("--batch_size", type=int, default=8)
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
    arch = str(ckpt.get("arch", "unet_small"))
    base = int(ckpt.get("base", 32))
    y_channels = str(ckpt.get("y_channels", "all"))

    x_mean = ckpt.get("x_mean", None)
    x_std = ckpt.get("x_std", None)

    if x_mean is not None:
        x_mean = np.asarray(x_mean, dtype=np.float32)
    if x_std is not None:
        x_std = np.asarray(x_std, dtype=np.float32)

    # Load Y for metric computation (subset same channels used in training)
    Y_full = np.load(Y_path, mmap_mode="r")  # (N,Cy,H,W)
    if y_channels.strip().lower() not in ("all", "*"):
        ch = tuple(int(p.strip()) for p in y_channels.split(",") if p.strip() != "")
        Y = np.array(Y_full[:, list(ch), :, :], dtype=np.float32)
    else:
        Y = np.array(Y_full, dtype=np.float32)

    device = torch.device(args.device)

    if arch != "unet_small":
        raise RuntimeError(f"Unsupported arch in checkpoint: {arch}")

    model = UNetSmall(c_in=c_in, c_out=c_out, base=base).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = NpyXDataset(X_path, x_mean=x_mean, x_std=x_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Shapes
    X_shape = np.load(X_path, mmap_mode="r").shape
    N, _, H, W = X_shape

    preds_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
    masks = np.zeros((N, 1, H, W), dtype=np.float32)

    idx0 = 0
    for x, m in dl:
        b = int(x.shape[0])
        x = x.to(device)
        m = m.to(device)

        with torch.no_grad():
            pred = model(x)
            pred = pred * m  # enforce gaps = 0
            pred = torch.pow(10.0, pred) - 1e-3
            pred = pred.detach().cpu().numpy().astype(np.float32)

        preds_norm[idx0 : idx0 + b] = pred
        masks[idx0 : idx0 + b] = m.detach().cpu().numpy().astype(np.float32)
        idx0 += b


    preds = preds_norm

    # Save predictions (physical units if y_mean/y_std available)
    np.save(out_dir / "pred_Y_img_test.npy", preds)

    # Metrics in the same space as Y:
    # If training normalized Y, then stored Y on disk is physical; we compare to physical preds (denormalized).
    mae = masked_mae_np(preds, Y, masks)
    rmse = masked_rmse_np(preds, Y, masks)

    metrics = {"mae": mae, "rmse": rmse}
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "c_in": c_in,
                "c_out": c_out,
                "arch": arch,
                "base": base,
                "y_channels": y_channels,
                "normalized_infer_input": bool(x_mean is not None and x_std is not None),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved predictions: {out_dir / 'pred_Y_img_test.npy'}  shape={preds.shape}")
    print(f"Test masked MAE: {mae:.6g} | masked RMSE: {rmse:.6g}")


if __name__ == "__main__":
    main()
