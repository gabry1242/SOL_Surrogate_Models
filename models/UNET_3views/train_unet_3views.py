#!/usr/bin/env python3
"""
train_unet_3views.py

Train a UNet on ONE of the three per-view 4D tensors produced by
build_3view_tensors.py.  Run the script once per view; each run is fully
independent and saves its own checkpoint + stats.

Tensor naming convention (build_3view_tensors.py output):
  <prefix>_view{0,1,2}_X_img_{split}.npy   (N, C_in,  Hmax, Wmax)
  <prefix>_view{0,1,2}_Y_img_{split}.npy   (N, C_out, Hmax, Wmax)

Channel ordering assumed (adjust POS_CHANNELS / SIGNED_CHANNELS below if yours differs):
  0       : te
  1       : ti
  2..11   : na (10 species)
  12..21  : ua (10 species)

Usage examples
--------------
# Train on view0
python scripts/models/UNET_3views/train_unet_3views.py \
    --view 0 \
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \
    --train_split train \
    --test_prefix  scripts/tensor/3views_4d/test/global3v \
    --test_split   test \
    --run_dir      scripts/runs/unet_3views/view0 \
    --y_channels   all \
    --epochs       50 \
    --batch_size   32 \
    --base         64

# Train on view1 (same command, change --view and --run_dir)
python scripts/models/UNET_3views/train_unet_3views.py \
    --view 1 \
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \
    --train_split train \
    --test_prefix  scripts/tensor/3views_4d/test/global3v \
    --test_split   test \
    --run_dir      scripts/runs/unet_3views/view1 \
    --y_channels   all \
    --epochs       50 \
    --batch_size   32 \
    --base         64

# Train on view2
python scripts/models/UNET_3views/train_unet_3views.py \
    --view 2 \
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \
    --train_split train \
    --test_prefix  scripts/tensor/3views_4d/test/global3v \
    --test_split   test \
    --run_dir      scripts/runs/unet_3views/view2 \
    --y_channels   all \
    --epochs       50 \
    --batch_size   32 \
    --base         64

Output layout (under --run_dir)
--------------------------------
  checkpoint_best.pt   — best test rmse_avg checkpoint
  checkpoint_last.pt   — most recent epoch checkpoint
  config.json          — all hyperparameters
  metrics.json         — per-epoch train/test metrics
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


# ---------------------------------------------------------------------------
# Channel definitions — adjust if your Y channel ordering differs
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))   # te, ti, na(10)  → log10 transform
SIGNED_CHANNELS = set(range(12, 22))  # ua(10)           → asinh transform


# ---------------------------------------------------------------------------
# Masked statistics helpers
# ---------------------------------------------------------------------------

def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support; cannot compute masked statistics.")
    return den


def masked_channel_stats_x(
    x_mem: np.ndarray, mask_ch: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel masked mean and std for X.
    x_mem : (N, C, H, W)
    Returns mean[C], std[C] float32.  Mask channel forced to mean=0, std=1.
    """
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

    std          = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch] = 0.0
    std[mask_ch]  = 1.0
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

    Returns:
      y_mean : (C_sel,) float32
      y_std  : (C_sel,) float32
      s_c    : (C_sel,) float32  (1.0 for positive channels)
    """
    if y_mem.ndim != 4 or mask_mem.ndim != 4 or mask_mem.shape[1] != 1:
        raise ValueError("y_mem must be 4D (N,C,H,W) and mask_mem 4D (N,1,H,W).")
    m   = mask_mem.astype(np.float64)
    den = _masked_den(m)

    C_sel  = len(y_indices)
    y_mean = np.zeros(C_sel, dtype=np.float64)
    y_var  = np.zeros(C_sel, dtype=np.float64)
    s_c    = np.ones(C_sel,  dtype=np.float64)

    # 1. s_c for signed channels
    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc  = y_mem[:, c:c + 1, :, :].astype(np.float64)
            mu  = float((yc * m).sum() / den)
            var = float((((yc - mu) ** 2) * m).sum() / den)
            s_c[j] = float(np.sqrt(max(var, 1e-12)))

    # 2. mean in transformed space
    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_mean[j] = float((t * m).sum() / den)

    # 3. var in transformed space
    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_var[j] = float((((t - y_mean[j]) ** 2) * m).sum() / den)

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ViewTensorDataset(Dataset):
    """
    Loads one view's X/Y tensors and applies all normalization + transforms.
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
        self.X = np.load(x_path, mmap_mode="r")   # (N, C_in,  H, W)
        self.Y = np.load(y_path, mmap_mode="r")   # (N, C_out, H, W)

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
        m      = x[0:1]  # (1, H, W)  mask channel

        # X normalisation
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


# ---------------------------------------------------------------------------
# Model — UNetSmall (identical to train_unet.py)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,  c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32):
        super().__init__()
        self.enc1       = ConvBlock(c_in,       base)
        self.enc2       = ConvBlock(base,        base * 2)
        self.enc3       = ConvBlock(base * 2,    base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4,    base * 8)

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
# Loss / metrics
# ---------------------------------------------------------------------------

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


def channel_balanced_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mae_per_channel(pred, target, mask).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    mae_acc = rmse_acc = None
    n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred     = model(x) * m
        mae_c    = masked_mae_per_channel(pred, y, m).cpu()
        rmse_c   = masked_rmse_per_channel(pred, y, m).cpu()
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train UNetSmall on a single 3-view tensor (view 0, 1, or 2)."
    )

    # ---- view selection ----
    ap.add_argument(
        "--view", type=int, choices=[0, 1, 2], required=True,
        help="Which view to train on: 0, 1, or 2."
    )

    # ---- tensor paths ----
    ap.add_argument(
        "--tensor_prefix", required=True,
        help=(
            "Path prefix used when running build_3view_tensors.py for the TRAIN split. "
            "E.g. scripts/tensor/3views_4d/train/global3v  "
            "→ will load  global3v_view{N}_X_img_{train_split}.npy"
        ),
    )
    ap.add_argument("--train_split", default="train",
                    help="Split tag for the train tensors (default: train).")
    ap.add_argument(
        "--test_prefix", required=True,
        help="Path prefix for the TEST split tensors (same naming convention).",
    )
    ap.add_argument("--test_split", default="test",
                    help="Split tag for the test tensors (default: test).")

    # ---- run output ----
    ap.add_argument("--run_dir", required=True,
                    help="Directory to write checkpoints, config and metrics.")

    # ---- training hyperparameters ----
    ap.add_argument("--y_channels", default="all",
                    help='Output channels to predict, e.g. "0,1" or "all".')
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--base",         type=int,   default=32,
                    help="UNet base channel width.")
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps",          type=float, default=1e-5,
                    help="Epsilon for log10 transform on positive channels.")
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
    X_train = np.load(train_x_path, mmap_mode="r")   # (N, C_in,  H, W)
    Y_train = np.load(train_y_path, mmap_mode="r")   # (N, C_out, H, W)

    c_in        = int(X_train.shape[1])
    c_out_total = int(Y_train.shape[1])

    y_indices = parse_channels(args.y_channels, c_out_total)
    c_out     = len(y_indices)

    print(f"\n=== Training UNet on {view_tag} ===")
    print(f"  Train X: {train_x_path}  {X_train.shape}")
    print(f"  Train Y: {train_y_path}  {Y_train.shape}")
    print(f"  c_in={c_in}  c_out={c_out}  y_indices={y_indices}")

    # ---- normalisation statistics (computed from train split only) ----
    print("Computing X normalisation stats …")
    x_mean, x_std = masked_channel_stats_x(X_train, mask_ch=0)

    mask_train = X_train[:, 0:1, :, :].astype(np.float32)
    print("Computing Y normalisation stats in transformed space …")
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
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # ---- model + optimiser ----
    model = UNetSmall(c_in=c_in, c_out=c_out, base=args.base).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- save config ----
    config = dict(
        view=args.view, view_tag=view_tag,
        tensor_prefix=args.tensor_prefix, train_split=args.train_split,
        test_prefix=args.test_prefix,     test_split=args.test_split,
        run_dir=str(run_dir),
        y_channels=args.y_channels, y_indices=y_indices,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        base=args.base, seed=args.seed, device=args.device, eps=args.eps,
    )
    save_json(run_dir / "config.json", config)

    # ---- training loop ----
    best_rmse      = float("inf")
    metrics_hist   = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_steps = 0.0, 0

        for x, y, m in dl_train:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred     = model(x) * m
            loss     = channel_balanced_loss(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_steps  += 1

        train_loss = loss_sum / max(n_steps, 1)
        train_eval = evaluate(model, dl_train, device)
        test_eval  = evaluate(model, dl_test,  device)

        metrics_hist["train"].append({"epoch": epoch, "loss": train_loss, **train_eval})
        metrics_hist["test"].append( {"epoch": epoch, **test_eval})

        cur_rmse = float(test_eval["rmse_avg"])

        ckpt = {
            "arch":            "unet_small",
            "view":            args.view,
            "view_tag":        view_tag,
            "base":            int(args.base),
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
            "metrics_last":    {"train": train_eval, "test": test_eval, "train_loss": train_loss},
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        save_json(run_dir / "metrics.json", metrics_hist)
        print(
            f"[{view_tag}] Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4g} | "
            f"test_mae={test_eval['mae_avg']:.4g} | "
            f"test_rmse={test_eval['rmse_avg']:.4g}"
        )

    print(f"\nDone. Best test rmse_avg={best_rmse:.6g}  →  {run_dir}/checkpoint_best.pt")


if __name__ == "__main__":
    main()
