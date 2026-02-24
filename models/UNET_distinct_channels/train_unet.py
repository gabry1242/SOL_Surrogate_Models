#!/usr/bin/env python3
"""
train_unet.py

Drop-in replacement for train_cnn.py with critical fixes:
1) Channel-specific target transforms:
   - te/ti/na (positive channels): log10(y + eps)
   - ua (signed channels): asinh(y / s_c)  (s_c = masked std in physical units)
2) Per-channel target normalization in transformed space (masked mean/std).
3) Channel-balanced loss: mean of per-channel masked MAE (not pixel-aggregated across channels).
4) Per-channel evaluation (MAE/RMSE) + global averages.
5) Baseline kept as UNetSmall (same as before), so improvements come from correct objective.

Expected tensor formats:
X: (N, C_in, H, W) float32, with mask in channel 0 (0/1)
Y: (N, C_out_total, H, W) float32 in physical units

Default channel ordering assumed (common in this project):
0: te
1: ti
2..11: na (10)
12..21: ua (10)

If you trained a different ordering, adjust POS_CHANNELS / SIGNED_CHANNELS below.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Channel definitions (adjust only if your Y ordering differs)
# ----------------------------
POS_CHANNELS = set(range(0, 12))      # te, ti, na(10)
SIGNED_CHANNELS = set(range(12, 22))  # ua(10)


# ----------------------------
# Masked stats helpers (NumPy)
# ----------------------------
def _masked_den(mask_mem: np.ndarray) -> float:
    den = float(np.asarray(mask_mem, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support; cannot compute masked statistics.")
    return den


def masked_channel_stats_x(x_mem: np.ndarray, mask_ch: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_mem: (N,C,H,W) float32
    mask is x_mem[:, mask_ch:mask_ch+1] (0/1)
    returns mean[C], std[C] float32; forces mask channel to mean=0,std=1.
    """
    if x_mem.ndim != 4:
        raise ValueError("X must be 4D (N,C,H,W).")
    _, C, _, _ = x_mem.shape
    m = x_mem[:, mask_ch:mask_ch + 1, :, :].astype(np.float64)
    den = _masked_den(m)

    mean = np.zeros((C,), dtype=np.float64)
    var = np.zeros((C,), dtype=np.float64)

    for c in range(C):
        xc = x_mem[:, c:c + 1, :, :].astype(np.float64)
        mean[c] = float((xc * m).sum() / den)

    for c in range(C):
        xc = x_mem[:, c:c + 1, :, :].astype(np.float64)
        var[c] = float((((xc - mean[c]) ** 2) * m).sum() / den)

    std = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch] = 0.0
    std[mask_ch] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def masked_mean_std_transformed_y(
    y_mem: np.ndarray,
    mask_mem: np.ndarray,
    y_indices: Sequence[int],
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel (mean,std) for Y in TRANSFORMED space, plus per-channel velocity scales s_c.

    - For positive channels: t = log10(max(y,0) + eps)
    - For signed channels:  t = asinh(y / s_c),  where s_c is masked std of y in physical units
      (computed per channel), clamped to >= 1e-12.

    Returns:
      y_mean: (C_sel,) float32 in transformed space
      y_std:  (C_sel,) float32 in transformed space
      s_c:    (C_sel,) float32 scale used for signed channels (1.0 for positive channels)
    """
    if y_mem.ndim != 4:
        raise ValueError("Y must be 4D (N,C,H,W).")
    if mask_mem.ndim != 4 or mask_mem.shape[1] != 1:
        raise ValueError("mask_mem must be 4D (N,1,H,W).")

    m = mask_mem.astype(np.float64)
    den = _masked_den(m)

    C_sel = len(y_indices)
    y_mean = np.zeros((C_sel,), dtype=np.float64)
    y_var = np.zeros((C_sel,), dtype=np.float64)
    s_c = np.ones((C_sel,), dtype=np.float64)

    # 1) compute s_c for signed channels (masked std in physical space)
    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
            mu = float((yc * m).sum() / den)
            var = float((((yc - mu) ** 2) * m).sum() / den)
            s = float(np.sqrt(max(var, 1e-12)))
            s_c[j] = s

    # 2) mean in transformed space
    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)

        if c in POS_CHANNELS:
            t = np.log10(np.maximum(yc, 0.0) + eps)
        elif c in SIGNED_CHANNELS:
            t = np.arcsinh(yc / s_c[j])
        else:
            raise ValueError(f"Channel index {c} not categorized as POS or SIGNED.")

        y_mean[j] = float((t * m).sum() / den)

    # 3) var in transformed space
    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)

        if c in POS_CHANNELS:
            t = np.log10(np.maximum(yc, 0.0) + eps)
        elif c in SIGNED_CHANNELS:
            t = np.arcsinh(yc / s_c[j])
        else:
            raise ValueError(f"Channel index {c} not categorized as POS or SIGNED.")

        y_var[j] = float((((t - y_mean[j]) ** 2) * m).sum() / den)

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)


# ----------------------------
# Dataset
# ----------------------------
class NpyTensorDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        y_indices: Sequence[int],
        x_mean: Optional[np.ndarray],
        x_std: Optional[np.ndarray],
        y_mean: np.ndarray,
        y_std: np.ndarray,
        s_c: np.ndarray,
        eps: float = 1e-3,
    ):
        self.X = np.load(x_path, mmap_mode="r")  # (N,Cx,H,W)
        self.Y = np.load(y_path, mmap_mode="r")  # (N,Cy,H,W)

        if self.X.ndim != 4 or self.Y.ndim != 4:
            raise ValueError("Expected X and Y to be 4D arrays: (N,C,H,W).")
        if self.X.shape[0] != self.Y.shape[0] or self.X.shape[2:] != self.Y.shape[2:]:
            raise ValueError(f"X and Y shape mismatch: X={self.X.shape}, Y={self.Y.shape}")
        if self.X.shape[1] < 1:
            raise ValueError("X must have mask in channel 0.")
        self.y_indices = list(map(int, y_indices))
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)
        self.s_c = np.asarray(s_c, dtype=np.float32)
        self.eps = float(eps)

        if len(self.y_indices) != int(self.y_mean.shape[0]) or len(self.y_indices) != int(self.y_std.shape[0]):
            raise ValueError("y_mean/y_std length must match number of selected y_indices.")
        if len(self.y_indices) != int(self.s_c.shape[0]):
            raise ValueError("s_c length must match number of selected y_indices.")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _transform_y(self, y_sel: torch.Tensor) -> torch.Tensor:
        # y_sel: (C_sel,H,W) physical units
        C_sel = y_sel.shape[0]
        out = torch.empty_like(y_sel)
        for j in range(C_sel):
            orig_c = self.y_indices[j]
            if orig_c in POS_CHANNELS:
                out[j] = torch.log10(torch.clamp(y_sel[j], min=0.0) + self.eps)
            else:
                s = float(self.s_c[j])
                out[j] = torch.asinh(y_sel[j] / s)
        return out

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y_full = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))
        m = x[0:1]  # (1,H,W)

        # X normalization (masked stats)
        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x = (x - xm) / xs

        # select Y channels
        y_sel = y_full[self.y_indices, :, :]  # (C_sel,H,W)

        # transform + normalize per channel
        y_t = self._transform_y(y_sel)
        ym = torch.from_numpy(self.y_mean).view(-1, 1, 1)
        ys = torch.from_numpy(self.y_std).view(-1, 1, 1)
        y_n = (y_t - ym) / ys

        return x, y_n, m


# ----------------------------
# Model (same UNetSmall as before)
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
# Loss / metrics (channel-balanced)
# ----------------------------
def masked_mae_per_channel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: (B,C,H,W)
    mask: (B,1,H,W) or (B,C,H,W)
    returns: (C,) tensor (mean abs error per channel over all pixels and batch)
    """
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    diff = (pred - target).abs() * mask
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)  # (C,)
    num = diff.sum(dim=(0, 2, 3))                 # (C,)
    return num / den


def masked_rmse_per_channel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    diff2 = (pred - target).pow(2) * mask
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    num = diff2.sum(dim=(0, 2, 3))
    return torch.sqrt(num / den)


def channel_balanced_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_c = masked_mae_per_channel(pred, target, mask)
    return per_c.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    mae_acc = None
    rmse_acc = None
    n_batches = 0

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred = model(x)
        pred = pred * m

        mae_c = masked_mae_per_channel(pred, y, m).detach().cpu()
        rmse_c = masked_rmse_per_channel(pred, y, m).detach().cpu()

        if mae_acc is None:
            mae_acc = mae_c.clone()
            rmse_acc = rmse_c.clone()
        else:
            mae_acc += mae_c
            rmse_acc += rmse_c

        n_batches += 1

    if n_batches == 0:
        return {"mae_avg": float("nan"), "rmse_avg": float("nan"), "mae_per_channel": [], "rmse_per_channel": []}

    mae_acc /= n_batches
    rmse_acc /= n_batches

    return {
        "mae_avg": float(mae_acc.mean().item()),
        "rmse_avg": float(rmse_acc.mean().item()),
        "mae_per_channel": mae_acc.numpy().astype(float).tolist(),
        "rmse_per_channel": rmse_acc.numpy().astype(float).tolist(),
    }


# ----------------------------
# Config / utilities
# ----------------------------
@dataclass
class TrainConfig:
    train_dir: str
    test_dir: str
    run_dir: str

    y_channels: str  # e.g. "0,1" or "all"
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float

    base: int
    seed: int
    device: str

    eps: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_channels(s: str, c_out_total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(c_out_total))
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    ch = [int(p) for p in parts]
    if any((c < 0 or c >= c_out_total) for c in ch):
        raise ValueError(f"y_channels out of bounds. Got {ch}, total C_out={c_out_total}")
    return ch


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Folder containing global_X_img_train.npy and global_Y_img_train.npy")
    ap.add_argument("--test_dir", required=True, help="Folder containing global_X_img_test.npy and global_Y_img_test.npy")
    ap.add_argument("--run_dir", required=True, help="Where to write checkpoints and logs")

    ap.add_argument("--y_channels", default="all", help='Target channels, e.g. "0,1" or "all"')
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--base", type=int, default=32, help="UNet base width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--eps", type=float, default=1e-5, help="epsilon for log10 transform on positive channels")

    args = ap.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        run_dir=args.run_dir,
        y_channels=args.y_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base=args.base,
        seed=args.seed,
        device=args.device,
        eps=float(args.eps),
    )

    set_seed(cfg.seed)

    train_dir = Path(cfg.train_dir)
    test_dir = Path(cfg.test_dir)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x_path = train_dir / "global_X_img_train.npy"
    train_y_path = train_dir / "global_Y_img_train.npy"
    test_x_path = test_dir / "global_X_img_test.npy"
    test_y_path = test_dir / "global_Y_img_test.npy"

    if not train_x_path.exists() or not train_y_path.exists():
        raise FileNotFoundError("Expected global_X_img_train.npy and global_Y_img_train.npy in train_dir.")
    if not test_x_path.exists() or not test_y_path.exists():
        raise FileNotFoundError("Expected global_X_img_test.npy and global_Y_img_test.npy in test_dir.")

    # Memmaps for stats
    X_train = np.load(train_x_path, mmap_mode="r")
    Y_train = np.load(train_y_path, mmap_mode="r")

    if X_train.ndim != 4 or Y_train.ndim != 4:
        raise ValueError("Expected X and Y to be 4D arrays.")
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError("Train X and Y must have same N.")
    c_in = int(X_train.shape[1])
    c_out_total = int(Y_train.shape[1])

    y_indices = parse_channels(cfg.y_channels, c_out_total)
    c_out = len(y_indices)

    # X normalization stats (masked)
    x_mean, x_std = masked_channel_stats_x(X_train, mask_ch=0)

    # Mask for Y stats
    mask_train = X_train[:, 0:1, :, :].astype(np.float32)

    # Y mean/std in transformed space + velocity scales
    y_mean, y_std, s_c = masked_mean_std_transformed_y(Y_train, mask_train, y_indices=y_indices, eps=cfg.eps)

    # Datasets
    ds_train = NpyTensorDataset(
        x_path=train_x_path,
        y_path=train_y_path,
        y_indices=y_indices,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        s_c=s_c,
        eps=cfg.eps,
    )
    ds_test = NpyTensorDataset(
        x_path=test_x_path,
        y_path=test_y_path,
        y_indices=y_indices,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        s_c=s_c,
        eps=cfg.eps,
    )

    device = torch.device(cfg.device)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = UNetSmall(c_in=c_in, c_out=c_out, base=cfg.base).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_rmse = float("inf")

    # Save config snapshot
    save_json(run_dir / "config.json", asdict(cfg))

    metrics_hist = {"train": [], "test": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_steps = 0

        for x, y, m in dl_train:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            pred = model(x)
            pred = pred * m

            loss = channel_balanced_loss(pred, y, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())
            n_steps += 1

        train_loss = loss_sum / max(n_steps, 1)

        train_eval = evaluate(model, dl_train, device)
        test_eval = evaluate(model, dl_test, device)

        metrics_hist["train"].append({"epoch": epoch, "loss": train_loss, **train_eval})
        metrics_hist["test"].append({"epoch": epoch, **test_eval})

        # checkpointing by test rmse_avg
        cur_rmse = float(test_eval["rmse_avg"])
        ckpt = {
            "arch": "unet_small",
            "base": int(cfg.base),
            "c_in": int(c_in),
            "c_out": int(c_out),
            "c_out_total": int(c_out_total),
            "y_indices": [int(c) for c in y_indices],
            "pos_channels": sorted(list(POS_CHANNELS)),
            "signed_channels": sorted(list(SIGNED_CHANNELS)),
            "eps": float(cfg.eps),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
            "s_c": s_c.tolist(),
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "epoch": int(epoch),
            "metrics_last": {"train": train_eval, "test": test_eval, "train_loss": train_loss},
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")

        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        # write metrics after each epoch
        save_json(run_dir / "metrics.json", metrics_hist)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6g} | "
            f"test_mae_avg={float(test_eval['mae_avg']):.6g} | test_rmse_avg={float(test_eval['rmse_avg']):.6g}"
        )

    print(f"Done. Best test rmse_avg={best_rmse:.6g}")


if __name__ == "__main__":
    main()

#mock command to execute
#python scripts/models/UNET_distinct_channels/train_unet.py --train_dir scripts/tensor/nopad_nofill_geom/train --val_dir scripts/tensor/nopad_nofill_geom/test --run_dir scripts/runs/unet_50_width64 --y_channels all --epochs 50 --batch_size 32 --base 64
