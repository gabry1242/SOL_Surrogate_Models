#!/usr/bin/env python3
"""
overfit_unet_3views.py
─────────────────────────────────────────────────────────────────────────────
Proof-of-concept: train UNetSmall to overfit on a single simulation sample
(index 0) composed from the union of view0, view1, and view2.

The 3 views are treated as a mini-batch of size 3 — all from the same
simulation, but different spatial orientations.  If the model can drive
the loss to near-zero it demonstrates learning capacity.

Usage
─────
python overfit_unet_3views.py \
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \
    --split         train \
    --sample_idx    0 \
    --epochs        1000 \
    --base          32 \
    --lr            1e-3 \
    --y_channels    all \
    --plot_channels 0,1

Arguments
─────────
--tensor_prefix   Prefix shared by all view tensors, e.g.:
                    scripts/tensor/3views_4d/train/global3v
                  → loads global3v_view{0,1,2}_X_img_{split}.npy
                           global3v_view{0,1,2}_Y_img_{split}.npy
--split           Split tag (default: train)
--sample_idx      Which simulation index to use (default: 0)
--epochs          Training iterations (default: 1000)
--base            UNet base width (default: 32)
--lr              Learning rate (default: 1e-3)
--y_channels      Comma-separated channel indices OR "all" (default: all)
--plot_channels   Which output channels to include in the final plot,
                  e.g. "0,1,2" (default: first 4, or fewer if c_out < 4)
--log_every       Print loss every N epochs (default: 50)
--device          "cuda" / "cpu" (auto-detected if omitted)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" if interactive
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Channel metadata  (must match your tensor builder)
# ─────────────────────────────────────────────────────────────────────────────

POS_CHANNELS    = set(range(0, 12))   # te, ti, na(10)  → log10 in Y
SIGNED_CHANNELS = set(range(12, 22))  # ua(10)           → asinh  in Y

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CH_NAMES: List[str] = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers  (identical logic to train_unet_3views.py)
# ─────────────────────────────────────────────────────────────────────────────

def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support — check tensor correctness.")
    return den


def compute_x_stats(
    x_arr: np.ndarray,          # (N, C, H, W)
    mask_ch: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel masked mean & std for X; mask channel → mean=0, std=1."""
    _, C, _, _ = x_arr.shape
    m   = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
    den = _masked_den(m)

    mean = np.zeros(C, np.float64)
    var  = np.zeros(C, np.float64)
    for c in range(C):
        xc      = x_arr[:, c:c + 1].astype(np.float64)
        mean[c] = float((xc * m).sum() / den)
    for c in range(C):
        xc     = x_arr[:, c:c + 1].astype(np.float64)
        var[c] = float((((xc - mean[c]) ** 2) * m).sum() / den)

    std             = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch]   = 0.0
    std[mask_ch]    = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_y_stats(
    y_arr: np.ndarray,          # (N, C_full, H, W)
    mask_arr: np.ndarray,       # (N, 1,      H, W)
    y_indices: Sequence[int],
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel (mean, std, s_c) in TRANSFORMED space."""
    m   = mask_arr.astype(np.float64)
    den = _masked_den(m)

    C_sel  = len(y_indices)
    y_mean = np.zeros(C_sel, np.float64)
    y_var  = np.zeros(C_sel, np.float64)
    s_c    = np.ones(C_sel,  np.float64)

    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc      = y_arr[:, c:c + 1].astype(np.float64)
            mu      = float((yc * m).sum() / den)
            var_raw = float((((yc - mu) ** 2) * m).sum() / den)
            s_c[j]  = float(np.sqrt(max(var_raw, 1e-12)))

    for j, c in enumerate(y_indices):
        yc = y_arr[:, c:c + 1].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_mean[j] = float((t * m).sum() / den)

    for j, c in enumerate(y_indices):
        yc = y_arr[:, c:c + 1].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_var[j] = float((((t - y_mean[j]) ** 2) * m).sum() / den)

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)


def normalize_x(
    x: np.ndarray,              # (C, H, W)
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32))
    m = torch.from_numpy(mean).view(-1, 1, 1)
    s = torch.from_numpy(std).view(-1, 1, 1)
    return (t - m) / s


def transform_normalize_y(
    y: np.ndarray,              # (C_full, H, W)
    y_indices: Sequence[int],
    y_mean: np.ndarray,
    y_std: np.ndarray,
    s_c: np.ndarray,
    eps: float = 1e-3,
) -> torch.Tensor:
    C_sel = len(y_indices)
    out   = np.empty((C_sel, y.shape[1], y.shape[2]), dtype=np.float32)
    for j, c in enumerate(y_indices):
        yc = y[c].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / float(s_c[j])))
        out[j] = ((t - float(y_mean[j])) / float(y_std[j])).astype(np.float32)
    return torch.from_numpy(out)


# ─────────────────────────────────────────────────────────────────────────────
# Model  (exact copy from train_unet_3views.py)
# ─────────────────────────────────────────────────────────────────────────────

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
        self.enc1       = ConvBlock(c_in,      base)
        self.enc2       = ConvBlock(base,       base * 2)
        self.enc3       = ConvBlock(base * 2,   base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4,   base * 8)

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1  = nn.Conv2d(base * 2, base,     1)
        self.dec1 = ConvBlock(base * 2, base)
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _crop(feat: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        th, tw = hw
        dh = feat.shape[-2] - th
        dw = feat.shape[-1] - tw
        return feat[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]

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


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).abs().mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(1e-8)
    return num / den


def masked_rmse_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(1e-8)
    return torch.sqrt(num / den)


def channel_balanced_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mae_per_channel(pred, target, mask).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_overfit_result(
    model: nn.Module,
    batch_x: torch.Tensor,        # (3, C_in, H, W)   — the 3 views
    batch_y: torch.Tensor,        # (3, C_out, H, W)
    batch_m: torch.Tensor,        # (3, 1, H, W)
    device: torch.device,
    y_indices: List[int],
    plot_ch_local: List[int],     # local indices into the c_out dimension
    save_path: Optional[Path] = None,
) -> None:
    """
    For each requested channel and each of the 3 views, show:
      GT | Prediction | Absolute Error
    """
    model.eval()
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_m = batch_m.to(device)

    pred = model(batch_x) * batch_m   # (3, C_out, H, W)

    batch_y = batch_y.cpu().numpy()
    pred    = pred.cpu().numpy()
    err     = np.abs(batch_y - pred)

    view_labels = ["View 0", "View 1", "View 2"]

    for lc in plot_ch_local:
        global_ch = y_indices[lc]
        ch_name   = CH_NAMES[global_ch] if global_ch < len(CH_NAMES) else f"ch{global_ch}"

        # 3 rows (views) × 3 cols (GT / Pred / Error)
        fig, axes = plt.subplots(3, 3, figsize=(13, 4 * 3))
        fig.suptitle(
            f"Overfit check — channel {lc} ({ch_name})"
            f"\n[GT | Prediction | Absolute Error]",
            fontsize=13,
        )

        for v in range(3):
            gt   = batch_y[v, lc]
            pr   = pred[v, lc]
            er   = err[v, lc]

            vmin = min(gt.min(), pr.min())
            vmax = max(gt.max(), pr.max())

            ax_gt, ax_pr, ax_er = axes[v]

            im0 = ax_gt.imshow(gt, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
            ax_gt.set_title(f"{view_labels[v]} — GT")
            plt.colorbar(im0, ax=ax_gt, fraction=0.046, pad=0.04)

            im1 = ax_pr.imshow(pr, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
            ax_pr.set_title(f"{view_labels[v]} — Prediction")
            plt.colorbar(im1, ax=ax_pr, fraction=0.046, pad=0.04)

            im2 = ax_er.imshow(er, origin="lower", cmap="inferno", aspect="auto")
            ax_er.set_title(f"{view_labels[v]} — |Error|")
            plt.colorbar(im2, ax=ax_er, fraction=0.046, pad=0.04)

            for ax in (ax_gt, ax_pr, ax_er):
                ax.axis("off")

        plt.tight_layout()

        if save_path is not None:
            out = save_path.parent / f"{save_path.stem}_ch{lc:02d}_{ch_name}{save_path.suffix}"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            print(f"  Saved → {out}")
        else:
            plt.show()

        plt.close(fig)


def plot_loss_curve(
    losses: List[float],
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (masked MAE, log scale)")
    ax.set_title("Overfit training loss — UNet on 3-view single sample")
    ax.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_channels(s: str, c_out_total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(c_out_total))
    ch = [int(p) for p in s.split(",") if p.strip()]
    if any(c < 0 or c >= c_out_total for c in ch):
        raise ValueError(f"Channel index out of bounds: {ch} (max {c_out_total - 1})")
    return ch


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overfit UNet on sample[0] × 3 views (proof-of-concept)."
    )
    ap.add_argument(
        "--tensor_prefix", required=True,
        help=(
            "Shared prefix for all 6 tensor files, e.g. "
            "'scripts/tensor/3views_4d/train/global3v'  "
            "→ loads global3v_view{0,1,2}_X_img_{split}.npy"
        ),
    )
    ap.add_argument("--split",        default="train",
                    help="Split tag (default: train).")
    ap.add_argument("--sample_idx",   type=int, default=0,
                    help="Simulation index to overfit on (default: 0).")
    ap.add_argument("--epochs",       type=int, default=1000)
    ap.add_argument("--base",         type=int, default=32,
                    help="UNet base channel width (default: 32).")
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--y_channels",   default="all",
                    help='Output channels: comma-separated indices or "all".')
    ap.add_argument("--plot_channels", default=None,
                    help=(
                        "Which output channels to plot (local indices into "
                        "selected y_channels). Default: first 4 (or fewer)."
                    ))
    ap.add_argument("--log_every", type=int, default=50,
                    help="Print progress every N epochs (default: 50).")
    ap.add_argument("--eps",       type=float, default=1e-3,
                    help="Epsilon for log10 transform (default: 1e-3).")
    ap.add_argument("--save_dir",  default=None,
                    help=(
                        "Directory to save plots. If omitted, plots are shown "
                        "interactively (requires a display)."
                    ))
    ap.add_argument("--device",    default=None,
                    help="'cuda' or 'cpu'. Auto-detected if omitted.")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n{'='*62}")
    print(f"  Overfit PoC — UNet × 3 views  (sample_idx={args.sample_idx})")
    print(f"  Device : {device}")
    print(f"{'='*62}\n")

    # ── Resolve file paths ──────────────────────────────────────────────────
    pfx = args.tensor_prefix
    splt = args.split
    view_files: List[Tuple[Path, Path]] = []
    for v in range(3):
        xp = Path(f"{pfx}_view{v}_X_img_{splt}.npy")
        yp = Path(f"{pfx}_view{v}_Y_img_{splt}.npy")
        for p in (xp, yp):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected tensor file not found:\n  {p}\n"
                    "Check --tensor_prefix and --split."
                )
        view_files.append((xp, yp))
        print(f"  view{v}  X: {xp.name}   Y: {yp.name}")

    # ── Load all three views (memmap, extract one sample) ──────────────────
    idx = args.sample_idx
    views_x_raw: List[np.ndarray] = []
    views_y_raw: List[np.ndarray] = []
    views_m_raw: List[np.ndarray] = []
    all_x_full:  List[np.ndarray] = []
    all_y_full:  List[np.ndarray] = []

    for v, (xp, yp) in enumerate(view_files):
        Xmem = np.load(xp, mmap_mode="r")   # (N, C_in,   H, W)
        Ymem = np.load(yp, mmap_mode="r")   # (N, C_out,  H, W)

        if idx >= Xmem.shape[0]:
            raise IndexError(
                f"sample_idx={idx} out of range for view{v} "
                f"(N={Xmem.shape[0]})"
            )

        views_x_raw.append(np.array(Xmem[idx], dtype=np.float32))
        views_y_raw.append(np.array(Ymem[idx], dtype=np.float32))
        views_m_raw.append(Xmem[idx, 0:1].astype(np.float32))  # (1,H,W)
        all_x_full.append(Xmem)
        all_y_full.append(Ymem)

    c_in        = views_x_raw[0].shape[0]
    c_out_total = views_y_raw[0].shape[1] if views_y_raw[0].ndim == 3 else views_y_raw[0].shape[0]

    # ── Channel selection ───────────────────────────────────────────────────
    y_indices = parse_channels(args.y_channels, c_out_total)
    c_out     = len(y_indices)
    print(f"\n  c_in={c_in}  c_out={c_out}  (y_indices: {y_indices[:6]}{'…' if c_out>6 else ''})")

    # ── Normalisation stats from full train arrays ──────────────────────────
    # Concatenate across the 3 views' train data for robust stats
    print("  Computing normalisation statistics from all 3 view tensors …")
    X_concat = np.concatenate(all_x_full, axis=0)   # (3N, C_in, H, W)
    Y_concat = np.concatenate(all_y_full, axis=0)   # (3N, C_out, H, W)
    M_concat = X_concat[:, 0:1].astype(np.float32)  # (3N, 1, H, W)

    x_mean, x_std = compute_x_stats(X_concat, mask_ch=0)
    y_mean, y_std, s_c = compute_y_stats(
        Y_concat, M_concat, y_indices=y_indices, eps=args.eps
    )

    # ── Build the fixed batch: 3 views × 1 sample ───────────────────────────
    xs, ys, ms = [], [], []
    for v in range(3):
        xn = normalize_x(views_x_raw[v], x_mean, x_std)   # (C_in, H, W)
        yn = transform_normalize_y(
            views_y_raw[v], y_indices, y_mean, y_std, s_c, eps=args.eps
        )                                                   # (C_sel, H, W)
        m  = torch.from_numpy(views_m_raw[v])              # (1, H, W)
        xs.append(xn); ys.append(yn); ms.append(m)

    # Stack to (3, C, H, W) — this is our entire "dataset"
    batch_x = torch.stack(xs).to(device)   # (3, C_in,  H, W)
    batch_y = torch.stack(ys).to(device)   # (3, C_out, H, W)
    batch_m = torch.stack(ms).to(device)   # (3, 1,     H, W)

    canvas_h, canvas_w = batch_x.shape[-2], batch_x.shape[-1]
    print(f"  Canvas per view: {canvas_h} × {canvas_w}")
    print(f"  Batch shape  X: {tuple(batch_x.shape)}   "
          f"Y: {tuple(batch_y.shape)}   M: {tuple(batch_m.shape)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = UNetSmall(c_in=c_in, c_out=c_out, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  UNetSmall — base={args.base}  params={n_params:,}")

    opt = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine LR schedule helps squeeze out the last bits of overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs …\n")
    losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        pred = model(batch_x) * batch_m
        loss = channel_balanced_loss(pred, batch_y, batch_m)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        if epoch % args.log_every == 0 or epoch == 1:
            with torch.no_grad():
                pred_eval = model(batch_x) * batch_m
                mae_ch    = masked_mae_per_channel(pred_eval, batch_y, batch_m)
                rmse_ch   = masked_rmse_per_channel(pred_eval, batch_y, batch_m)
            print(
                f"  Epoch {epoch:5d}/{args.epochs} | "
                f"loss={loss_val:.5f} | "
                f"MAE_avg={mae_ch.mean().item():.5f} | "
                f"RMSE_avg={rmse_ch.mean().item():.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # ── Final metrics ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_final = model(batch_x) * batch_m
        mae_final  = masked_mae_per_channel(pred_final, batch_y, batch_m)
        rmse_final = masked_rmse_per_channel(pred_final, batch_y, batch_m)

    print(f"\n{'─'*62}")
    print(f"  Final overfit metrics (all 3 views, sample {idx})")
    print(f"  MAE_avg  = {mae_final.mean().item():.6f}")
    print(f"  RMSE_avg = {rmse_final.mean().item():.6f}")
    if c_out <= 22:
        print("  Per-channel MAE:")
        for j, c in enumerate(y_indices):
            nm = CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"
            print(f"    [{j:2d}] {nm:12s}  MAE={mae_final[j].item():.5f}  RMSE={rmse_final[j].item():.5f}")
    print(f"{'─'*62}\n")

    # ── Determine which channels to plot ────────────────────────────────────
    if args.plot_channels is not None:
        plot_ch_local = parse_channels(args.plot_channels, c_out)
    else:
        plot_ch_local = list(range(min(4, c_out)))

    # ── Save dir ─────────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot predictions ─────────────────────────────────────────────────────
    print(f"  Plotting channels (local indices): {plot_ch_local} …")
    plot_overfit_result(
        model     = model,
        batch_x   = batch_x.cpu(),
        batch_y   = batch_y.cpu(),
        batch_m   = batch_m.cpu(),
        device    = device,
        y_indices = y_indices,
        plot_ch_local = plot_ch_local,
        save_path = (save_dir / "overfit.png") if save_dir else None,
    )

    # ── Plot loss curve ───────────────────────────────────────────────────────
    print("  Plotting loss curve …")
    plot_loss_curve(
        losses,
        save_path = (save_dir / "loss_curve.png") if save_dir else None,
    )

    print("\n  Done.\n")


if __name__ == "__main__":
    main()