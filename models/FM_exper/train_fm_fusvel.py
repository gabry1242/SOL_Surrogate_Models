#!/usr/bin/env python3
"""
overfit_unet_3views_velocity.py
─────────────────────────────────────────────────────────────────────────────
Same proof-of-concept as overfit_unet_3views.py but with VelocityUNet
(FiLM-conditioned, dual-encoder architecture) instead of UNetSmall.

The model is used for direct regression (not flow matching):
  • y_t  = zeros  (no noisy intermediate state needed)
  • t    = 1.0    (constant; FiLM still fires, just on a fixed embedding)

Everything else — normalisation, loss, plotting — is identical to the
original script.  Only the ── CHANGED ── sections below differ.

Usage
─────
python overfit_unet_3views_velocity.py \
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \
    --split         train \
    --sample_idx    0 \
    --epochs        1000 \
    --base          32 \
    --lr            1e-3 \
    --y_channels    all \
    --plot_channels 0,1 \
    --save_dir      scripts/runs/overfit_velocity
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Channel metadata
# ─────────────────────────────────────────────────────────────────────────────

POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CH_NAMES: List[str] = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers  (unchanged from overfit_unet_3views.py)
# ─────────────────────────────────────────────────────────────────────────────

def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support — check tensor correctness.")
    return den


def compute_x_stats(x_arr: np.ndarray, mask_ch: int = 0):
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
    std           = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch] = 0.0
    std[mask_ch]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_y_stats(y_arr, mask_arr, y_indices, eps=1e-3):
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


def normalize_x(x, mean, std):
    t = torch.from_numpy(x.astype(np.float32))
    return (t - torch.from_numpy(mean).view(-1, 1, 1)) / torch.from_numpy(std).view(-1, 1, 1)


def transform_normalize_y(y, y_indices, y_mean, y_std, s_c, eps=1e-3):
    C_sel = len(y_indices)
    out   = np.empty((C_sel, y.shape[1], y.shape[2]), dtype=np.float32)
    for j, c in enumerate(y_indices):
        yc = y[c].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / float(s_c[j])))
        out[j] = ((t - float(y_mean[j])) / float(y_std[j])).astype(np.float32)
    return torch.from_numpy(out)


# ─────────────────────────────────────────────────────────────────────────────
# ── CHANGED: VelocityUNet replaces UNetSmall ─────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        t = t.view(-1).float()
        a = t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, ci, co, t_dim):
        super().__init__()
        self.c1   = nn.Conv2d(ci, co, 3, padding=1)
        self.c2   = nn.Conv2d(co, co, 3, padding=1)
        self.act  = nn.GELU()
        self.film = nn.Linear(t_dim, co * 2)

    def forward(self, x, t_emb):
        h = self.act(self.c1(x))
        h = self.act(self.c2(h))
        s, sh = self.film(t_emb).chunk(2, -1)
        return h * (1 + s[..., None, None]) + sh[..., None, None]


class CondEncoder(nn.Module):
    def __init__(self, c_in, base):
        super().__init__()
        self.e1   = ConvBlock(c_in,    base)
        self.e2   = ConvBlock(base,    base * 2)
        self.e3   = ConvBlock(base*2,  base * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(self.pool(f1))
        f3 = self.e3(self.pool(f2))
        return f1, f2, f3


class VelocityUNet(nn.Module):
    """
    FiLM-conditioned dual-encoder UNet.

    Original signature: forward(x_cond, y_t, t)
      x_cond : conditioning input   (B, c_in,  H, W)
      y_t    : noisy/intermediate y (B, c_out, H, W)  ← zeros for regression
      t      : flow timestep        (B,)               ← ones  for regression

    For direct regression the caller always passes:
      y_t = torch.zeros(B, c_out, H, W)
      t   = torch.ones(B)
    The FiLM blocks still activate; they just see a constant time embedding.
    """
    def __init__(self, c_in: int, c_out: int, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out

        self.time_emb = SinusoidalTimeEmb(t_dim)
        self.cond_enc = CondEncoder(c_in, base)

        self.enc1       = ConvBlock(c_in + c_out, base)
        self.enc2       = ConvBlock(base,          base * 2)
        self.enc3       = ConvBlock(base * 2,      base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4,      base * 8)

        self.up3  = nn.Conv2d(base * 8,  base * 4, 1)
        self.dec3 = FiLMBlock(base * 4 + base * 4 + base * 4, base * 4, t_dim)
        self.up2  = nn.Conv2d(base * 4,  base * 2, 1)
        self.dec2 = FiLMBlock(base * 2 + base * 2 + base * 2, base * 2, t_dim)
        self.up1  = nn.Conv2d(base * 2,  base,     1)
        self.dec1 = FiLMBlock(base      + base      + base,    base,     t_dim)
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _match(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        th, tw = hw
        h, w   = x.shape[-2:]
        if h == th and w == tw:
            return x
        if h >= th and w >= tw:
            dh, dw = h - th, w - tw
            return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]
        return F.interpolate(x, (th, tw), mode="bilinear", align_corners=False)

    def forward(
        self,
        x_cond: torch.Tensor,   # (B, c_in,  H, W)
        y_t:    torch.Tensor,   # (B, c_out, H, W)  — zeros for regression
        t:      torch.Tensor,   # (B,)              — ones  for regression
    ) -> torch.Tensor:
        te         = self.time_emb(t)
        c1, c2, c3 = self.cond_enc(x_cond)

        inp = torch.cat([x_cond, y_t], 1)
        e1  = self.enc1(inp)
        e2  = self.enc2(self.pool(e1))
        e3  = self.enc3(self.pool(e2))
        b   = self.bottleneck(self.pool(e3))

        u3  = self.up3(F.interpolate(b,  e3.shape[-2:], mode="bilinear", align_corners=False))
        d3  = self.dec3(torch.cat([u3,
                                   self._match(e3, u3.shape[-2:]),
                                   self._match(c3, u3.shape[-2:])], 1), te)
        u2  = self.up2(F.interpolate(d3, e2.shape[-2:], mode="bilinear", align_corners=False))
        d2  = self.dec2(torch.cat([u2,
                                   self._match(e2, u2.shape[-2:]),
                                   self._match(c2, u2.shape[-2:])], 1), te)
        u1  = self.up1(F.interpolate(d2, e1.shape[-2:], mode="bilinear", align_corners=False))
        d1  = self.dec1(torch.cat([u1,
                                   self._match(e1, u1.shape[-2:]),
                                   self._match(c1, u1.shape[-2:])], 1), te)
        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# ── CHANGED: forward helper — wraps the 3-arg call for regression use ────────
# ─────────────────────────────────────────────────────────────────────────────

def model_forward(
    model:   VelocityUNet,
    x:       torch.Tensor,   # (B, c_in,  H, W)
    c_out:   int,
) -> torch.Tensor:
    """
    Regression wrapper: always passes y_t=zeros and t=ones.
    Returns predictions with the same shape as the target Y.
    """
    B = x.shape[0]
    y_t = torch.zeros(B, c_out, x.shape[-2], x.shape[-1], device=x.device)
    t   = torch.ones(B, device=x.device)
    return model(x, y_t, t)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae_per_channel(pred, target, mask):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).abs().mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(1e-8)
    return num / den


def masked_rmse_per_channel(pred, target, mask):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(1e-8)
    return torch.sqrt(num / den)


def channel_balanced_loss(pred, target, mask):
    return masked_mae_per_channel(pred, target, mask).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_overfit_result(model, batch_x, batch_y, batch_m, device,
                        y_indices, plot_ch_local, c_out, save_path=None):
    model.eval()
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_m = batch_m.to(device)

    # ── CHANGED: use model_forward ──
    pred = model_forward(model, batch_x, c_out) * batch_m

    batch_y = batch_y.cpu().numpy()
    pred    = pred.cpu().numpy()
    err     = np.abs(batch_y - pred)
    view_labels = ["View 0", "View 1", "View 2"]

    for lc in plot_ch_local:
        global_ch = y_indices[lc]
        ch_name   = CH_NAMES[global_ch] if global_ch < len(CH_NAMES) else f"ch{global_ch}"

        fig, axes = plt.subplots(3, 3, figsize=(13, 4 * 3))
        fig.suptitle(
            f"Overfit check — channel {lc} ({ch_name})\n[GT | Prediction | Absolute Error]",
            fontsize=13,
        )
        for v in range(3):
            gt, pr, er = batch_y[v, lc], pred[v, lc], err[v, lc]
            vmin, vmax = min(gt.min(), pr.min()), max(gt.max(), pr.max())
            ax_gt, ax_pr, ax_er = axes[v]

            im0 = ax_gt.imshow(gt, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
            ax_gt.set_title(f"{view_labels[v]} — GT");  plt.colorbar(im0, ax=ax_gt, fraction=0.046)

            im1 = ax_pr.imshow(pr, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
            ax_pr.set_title(f"{view_labels[v]} — Prediction"); plt.colorbar(im1, ax=ax_pr, fraction=0.046)

            im2 = ax_er.imshow(er, origin="lower", cmap="inferno", aspect="auto")
            ax_er.set_title(f"{view_labels[v]} — |Error|"); plt.colorbar(im2, ax=ax_er, fraction=0.046)

            for ax in (ax_gt, ax_pr, ax_er): ax.axis("off")

        plt.tight_layout()
        if save_path is not None:
            out = save_path.parent / f"{save_path.stem}_ch{lc:02d}_{ch_name}{save_path.suffix}"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            print(f"  Saved → {out}")
        else:
            plt.show()
        plt.close(fig)


def plot_loss_curve(losses, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (masked MAE, log scale)")
    ax.set_title("Overfit training loss — VelocityUNet on 3-view single sample")
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
        raise ValueError(f"Channel index out of bounds: {ch}")
    return ch


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--split",         default="train")
    ap.add_argument("--sample_idx",    type=int, default=0)
    ap.add_argument("--epochs",        type=int, default=1000)
    ap.add_argument("--base",          type=int, default=32)
    ap.add_argument("--t_dim",         type=int, default=128)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight_decay",  type=float, default=0.0)
    ap.add_argument("--y_channels",    default="all")
    ap.add_argument("--plot_channels", default=None)
    ap.add_argument("--log_every",     type=int, default=50)
    ap.add_argument("--eps",           type=float, default=1e-3)
    ap.add_argument("--save_dir",      default=None)
    ap.add_argument("--device",        default=None)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'='*62}")
    print(f"  Overfit PoC — VelocityUNet × 3 views  (sample={args.sample_idx})")
    print(f"  Device : {device}")
    print(f"{'='*62}\n")

    # ── Load tensors ─────────────────────────────────────────────────────────
    pfx  = args.tensor_prefix
    splt = args.split
    idx  = args.sample_idx

    views_x_raw, views_y_raw, views_m_raw = [], [], []
    all_x_full,  all_y_full               = [], []

    for v in range(3):
        xp = Path(f"{pfx}_view{v}_X_img_{splt}.npy")
        yp = Path(f"{pfx}_view{v}_Y_img_{splt}.npy")
        for p in (xp, yp):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
        Xmem = np.load(xp, mmap_mode="r")
        Ymem = np.load(yp, mmap_mode="r")
        views_x_raw.append(np.array(Xmem[idx], dtype=np.float32))
        views_y_raw.append(np.array(Ymem[idx], dtype=np.float32))
        views_m_raw.append(Xmem[idx, 0:1].astype(np.float32))
        all_x_full.append(Xmem); all_y_full.append(Ymem)
        print(f"  view{v}  X:{xp.name}  Y:{yp.name}")

    c_in        = views_x_raw[0].shape[0]
    c_out_total = views_y_raw[0].shape[0]

    y_indices = parse_channels(args.y_channels, c_out_total)
    c_out     = len(y_indices)
    print(f"\n  c_in={c_in}  c_out={c_out}  y_indices={y_indices[:6]}{'…' if c_out>6 else ''}")

    # ── Normalisation ─────────────────────────────────────────────────────────
    print("  Computing normalisation statistics …")
    X_concat = np.concatenate(all_x_full, axis=0)
    Y_concat = np.concatenate(all_y_full, axis=0)
    M_concat = X_concat[:, 0:1].astype(np.float32)
    x_mean, x_std    = compute_x_stats(X_concat)
    y_mean, y_std, s_c = compute_y_stats(Y_concat, M_concat, y_indices, args.eps)

    # ── Build 3-view batch ────────────────────────────────────────────────────
    xs, ys, ms = [], [], []
    for v in range(3):
        xs.append(normalize_x(views_x_raw[v], x_mean, x_std))
        ys.append(transform_normalize_y(views_y_raw[v], y_indices, y_mean, y_std, s_c, args.eps))
        ms.append(torch.from_numpy(views_m_raw[v]))

    batch_x = torch.stack(xs).to(device)   # (3, C_in,  H, W)
    batch_y = torch.stack(ys).to(device)   # (3, C_out, H, W)
    batch_m = torch.stack(ms).to(device)   # (3, 1,     H, W)
    print(f"  Batch  X:{tuple(batch_x.shape)}  Y:{tuple(batch_y.shape)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # ── CHANGED: VelocityUNet instead of UNetSmall ────────────────────────────
    model    = VelocityUNet(c_in=c_in, c_out=c_out, base=args.base, t_dim=args.t_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  VelocityUNet — base={args.base}  t_dim={args.t_dim}  params={n_params:,}")

    opt       = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs …\n")
    losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        # ── CHANGED: use model_forward wrapper ────────────────────────────────
        pred = model_forward(model, batch_x, c_out) * batch_m
        loss = channel_balanced_loss(pred, batch_y, batch_m)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        losses.append(float(loss.item()))

        if epoch % args.log_every == 0 or epoch == 1:
            with torch.no_grad():
                pe    = model_forward(model, batch_x, c_out) * batch_m
                mae_c = masked_mae_per_channel(pe, batch_y, batch_m)
                rms_c = masked_rmse_per_channel(pe, batch_y, batch_m)
            print(
                f"  Epoch {epoch:5d}/{args.epochs} | "
                f"loss={loss.item():.5f} | "
                f"MAE={mae_c.mean().item():.5f} | "
                f"RMSE={rms_c.mean().item():.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # ── Final metrics ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pf   = model_forward(model, batch_x, c_out) * batch_m
        mae  = masked_mae_per_channel(pf, batch_y, batch_m)
        rmse = masked_rmse_per_channel(pf, batch_y, batch_m)

    print(f"\n{'─'*62}")
    print(f"  Final overfit metrics (all 3 views, sample {idx})")
    print(f"  MAE_avg={mae.mean().item():.6f}   RMSE_avg={rmse.mean().item():.6f}")
    for j, c in enumerate(y_indices):
        nm = CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"
        print(f"    [{j:2d}] {nm:12s}  MAE={mae[j].item():.5f}  RMSE={rmse[j].item():.5f}")
    print(f"{'─'*62}\n")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_ch_local = (parse_channels(args.plot_channels, c_out)
                     if args.plot_channels else list(range(min(4, c_out))))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir: save_dir.mkdir(parents=True, exist_ok=True)

    plot_overfit_result(
        model, batch_x.cpu(), batch_y.cpu(), batch_m.cpu(),
        device, y_indices, plot_ch_local, c_out,
        save_path=(save_dir / "overfit.png") if save_dir else None,
    )
    plot_loss_curve(
        losses,
        save_path=(save_dir / "loss_curve.png") if save_dir else None,
    )
    print("  Done.\n")


if __name__ == "__main__":
    main()