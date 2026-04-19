#!/usr/bin/env python3
"""
train_fm_fusvel_full.py
─────────────────────────────────────────────────────────────────────────────
Full-dataset training with VelocityUNet (FiLM-conditioned, dual-encoder)
used for direct regression (y_t=zeros, t=ones).

Trains on ALL simulations across ALL 3 views.  Each (simulation, view) pair
is an independent training sample.  Normalization statistics are computed
once over the full training set and saved inside the checkpoint for
reproducible inference.

Usage
─────
python train_fm_fusvel_full.py \
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \
    --split         train \
    --epochs        200 \
    --batch_size    16 \
    --base          32 \
    --lr            1e-3 \
    --y_channels    all \
    --save_dir      scripts/runs/fusvel_full

Output  (inside --save_dir)
───────
  checkpoint_best.pt   — lowest training loss
  checkpoint_last.pt   — final epoch
  config.json          — full configuration snapshot
  metrics.json         — per-epoch training metrics
  norm_stats.npz       — normalisation stats (also inside checkpoints)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
# Normalisation helpers  (identical to PoC)
# ─────────────────────────────────────────────────────────────────────────────

def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support — check tensor correctness.")
    return den


def compute_x_stats(x_arr: np.ndarray, mask_ch: int = 0):
    """Masked mean/std per channel of X, using channel `mask_ch` as the mask."""
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
    """Masked mean/std of the *transformed* Y channels."""
    m   = mask_arr.astype(np.float64)
    den = _masked_den(m)
    C_sel  = len(y_indices)
    y_mean = np.zeros(C_sel, np.float64)
    y_var  = np.zeros(C_sel, np.float64)
    s_c    = np.ones(C_sel,  np.float64)
    # s_c: pre-transform scale for signed channels (from raw values)
    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc      = y_arr[:, c:c + 1].astype(np.float64)
            mu      = float((yc * m).sum() / den)
            var_raw = float((((yc - mu) ** 2) * m).sum() / den)
            s_c[j]  = float(np.sqrt(max(var_raw, 1e-12)))
    # mean in transformed space
    for j, c in enumerate(y_indices):
        yc = y_arr[:, c:c + 1].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_mean[j] = float((t * m).sum() / den)
    # variance in transformed space
    for j, c in enumerate(y_indices):
        yc = y_arr[:, c:c + 1].astype(np.float64)
        t  = (np.log10(np.maximum(yc, 0.0) + eps)
              if c in POS_CHANNELS else np.arcsinh(yc / s_c[j]))
        y_var[j] = float((((t - y_mean[j]) ** 2) * m).sum() / den)
    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewDataset(Dataset):
    """
    Flattens (N_sim, 3_views) into a single dataset of length N_sim * 3.
    Each item returns (x_norm, y_norm, mask) tensors for one view of one sim.
    """

    def __init__(
        self,
        x_paths: List[Path],       # one per view
        y_paths: List[Path],       # one per view
        y_indices: List[int],
        x_mean: np.ndarray,
        x_std:  np.ndarray,
        y_mean: np.ndarray,
        y_std:  np.ndarray,
        s_c:    np.ndarray,
        eps:    float = 1e-3,
    ):
        self.y_indices = y_indices
        self.x_mean = x_mean
        self.x_std  = x_std
        self.y_mean = y_mean
        self.y_std  = y_std
        self.s_c    = s_c
        self.eps    = eps

        # memory-map all views
        self.X_views = [np.load(p, mmap_mode="r") for p in x_paths]
        self.Y_views = [np.load(p, mmap_mode="r") for p in y_paths]
        self.n_sims  = self.X_views[0].shape[0]
        self.n_views = len(x_paths)

        # sanity checks
        for v in range(self.n_views):
            assert self.X_views[v].shape[0] == self.n_sims
            assert self.Y_views[v].shape[0] == self.n_sims

    def __len__(self) -> int:
        return self.n_sims * self.n_views

    def __getitem__(self, idx: int):
        view = idx % self.n_views
        sim  = idx // self.n_views

        # load single sample
        x_raw = np.array(self.X_views[view][sim], dtype=np.float32)   # (C_in, H, W)
        y_raw = np.array(self.Y_views[view][sim], dtype=np.float32)   # (C_out_all, H, W)
        mask  = x_raw[0:1].copy()                                     # (1, H, W)

        # normalise X
        x_norm = (x_raw - self.x_mean[:, None, None]) / self.x_std[:, None, None]

        # transform + normalise Y (selected channels only)
        C_sel = len(self.y_indices)
        y_norm = np.empty((C_sel, x_raw.shape[1], x_raw.shape[2]), dtype=np.float32)
        for j, c in enumerate(self.y_indices):
            yc = y_raw[c].astype(np.float64)
            t  = (np.log10(np.maximum(yc, 0.0) + self.eps)
                  if c in POS_CHANNELS else np.arcsinh(yc / float(self.s_c[j])))
            y_norm[j] = ((t - float(self.y_mean[j])) / float(self.y_std[j])).astype(np.float32)

        return (
            torch.from_numpy(x_norm),
            torch.from_numpy(y_norm),
            torch.from_numpy(mask),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model components  (identical to PoC — GELU activations)
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
    def forward(self, x):
        return self.net(x)


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
    FiLM-conditioned dual-encoder UNet used for direct regression.
    forward(x_cond, y_t, t) — for regression, y_t=zeros and t=ones.
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
        y_t:    torch.Tensor,   # (B, c_out, H, W)
        t:      torch.Tensor,   # (B,)
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
# Forward helper — regression wrapper
# ─────────────────────────────────────────────────────────────────────────────

def model_forward(
    model: VelocityUNet,
    x:     torch.Tensor,
    c_out: int,
) -> torch.Tensor:
    """Always passes y_t=zeros, t=ones for direct regression."""
    B = x.shape[0]
    y_t = torch.zeros(B, c_out, x.shape[-2], x.shape[-1], device=x.device)
    t   = torch.ones(B, device=x.device)
    return model(x, y_t, t)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_prefix", required=True,
                    help="Prefix for tensor files, e.g. scripts/tensor/3views_4d/train/global3v")
    ap.add_argument("--split",         default="train")
    ap.add_argument("--epochs",        type=int,   default=200)
    ap.add_argument("--batch_size",    type=int,   default=16)
    ap.add_argument("--base",          type=int,   default=32)
    ap.add_argument("--t_dim",         type=int,   default=128)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--y_channels",    default="all",
                    help='Channels to predict: "all" or comma-separated indices')
    ap.add_argument("--log_every",     type=int,   default=1)
    ap.add_argument("--eps",           type=float, default=1e-3)
    ap.add_argument("--save_dir",      required=True)
    ap.add_argument("--device",        default=None)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--num_workers",   type=int,   default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Full-dataset training — VelocityUNet × 3 views (regression)")
    print(f"  Device : {device}")
    print(f"{'='*70}\n")

    # ── Resolve tensor paths ─────────────────────────────────────────────────
    pfx  = args.tensor_prefix
    splt = args.split

    x_paths, y_paths = [], []
    for v in range(3):
        xp = Path(f"{pfx}_view{v}_X_img_{splt}.npy")
        yp = Path(f"{pfx}_view{v}_Y_img_{splt}.npy")
        for p in (xp, yp):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
        x_paths.append(xp)
        y_paths.append(yp)
        print(f"  view{v}  X:{xp.name}  Y:{yp.name}")

    # peek at shapes
    X0_peek = np.load(x_paths[0], mmap_mode="r")
    Y0_peek = np.load(y_paths[0], mmap_mode="r")
    N_sim   = X0_peek.shape[0]
    c_in    = X0_peek.shape[1]
    c_out_total = Y0_peek.shape[1]

    y_indices = parse_channels(args.y_channels, c_out_total)
    c_out     = len(y_indices)
    print(f"\n  N_sim={N_sim}  c_in={c_in}  c_out={c_out}")
    print(f"  y_indices={y_indices[:8]}{'…' if c_out > 8 else ''}")

    # ── Compute normalisation statistics over full training set ──────────────
    print("  Computing normalisation statistics …")
    all_x = [np.load(p, mmap_mode="r") for p in x_paths]
    all_y = [np.load(p, mmap_mode="r") for p in y_paths]
    X_concat = np.concatenate(all_x, axis=0)
    Y_concat = np.concatenate(all_y, axis=0)
    M_concat = X_concat[:, 0:1].astype(np.float32)

    x_mean, x_std      = compute_x_stats(X_concat)
    y_mean, y_std, s_c  = compute_y_stats(Y_concat, M_concat, y_indices, args.eps)
    del X_concat, Y_concat, M_concat

    # save norm stats separately for convenience
    np.savez(
        save_dir / "norm_stats.npz",
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std, s_c=s_c,
    )
    print(f"  Saved norm_stats.npz")

    # ── Build dataset and dataloader ─────────────────────────────────────────
    ds = MultiViewDataset(
        x_paths, y_paths, y_indices,
        x_mean, x_std, y_mean, y_std, s_c,
        eps=args.eps,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"  Dataset size: {len(ds)}  ({N_sim} sims × 3 views)")
    print(f"  Batches/epoch: {len(dl)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model    = VelocityUNet(c_in=c_in, c_out=c_out, base=args.base, t_dim=args.t_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  VelocityUNet — base={args.base}  t_dim={args.t_dim}  params={n_params:,}")

    opt       = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Save config ──────────────────────────────────────────────────────────
    config = {
        "tensor_prefix": str(pfx),
        "split":         splt,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "base":          args.base,
        "t_dim":         args.t_dim,
        "lr":            args.lr,
        "weight_decay":  args.weight_decay,
        "c_in":          c_in,
        "c_out":         c_out,
        "y_indices":     y_indices,
        "y_channels":    args.y_channels,
        "eps":           args.eps,
        "seed":          args.seed,
        "n_params":      n_params,
        "n_sims":        N_sim,
        "pos_channels":  sorted(POS_CHANNELS),
        "signed_channels": sorted(SIGNED_CHANNELS),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs …\n")
    best_loss  = float("inf")
    metrics_history = {"train": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mae  = None
        n_batches  = 0
        t0 = time.time()

        for batch_x, batch_y, batch_m in dl:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_m = batch_m.to(device, non_blocking=True)

            pred = model_forward(model, batch_x, c_out) * batch_m
            loss = channel_balanced_loss(pred, batch_y, batch_m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                mae_c = masked_mae_per_channel(pred, batch_y, batch_m)
                if epoch_mae is None:
                    epoch_mae = mae_c.detach()
                else:
                    epoch_mae += mae_c.detach()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_mae  = (epoch_mae / n_batches).cpu()
        dt       = time.time() - t0

        # record metrics
        entry = {
            "epoch": epoch,
            "loss":  avg_loss,
            "mae_avg": float(avg_mae.mean()),
            "lr":    scheduler.get_last_lr()[0],
            "time_s": dt,
        }
        metrics_history["train"].append(entry)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"loss={avg_loss:.5f} | "
                f"MAE_avg={float(avg_mae.mean()):.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{dt:.1f}s"
            )

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(
                model, save_dir / "checkpoint_best.pt",
                epoch, args, config, x_mean, x_std, y_mean, y_std, s_c,
                y_indices, c_in, c_out, avg_loss,
            )

    # save last
    _save_checkpoint(
        model, save_dir / "checkpoint_last.pt",
        args.epochs, args, config, x_mean, x_std, y_mean, y_std, s_c,
        y_indices, c_in, c_out, avg_loss,
    )

    # save metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n{'─'*70}")
    print(f"  Training complete.  Best loss = {best_loss:.6f}")
    print(f"  Checkpoints saved to {save_dir}")
    print(f"{'─'*70}\n")


def _save_checkpoint(
    model, path, epoch, args, config,
    x_mean, x_std, y_mean, y_std, s_c,
    y_indices, c_in, c_out, loss,
):
    torch.save({
        "model_state":     model.state_dict(),
        "epoch":           epoch,
        "loss":            loss,
        "c_in":            c_in,
        "c_out":           c_out,
        "base":            args.base,
        "t_dim":           args.t_dim,
        "y_indices":       y_indices,
        "x_mean":          x_mean,
        "x_std":           x_std,
        "y_mean":          y_mean,
        "y_std":           y_std,
        "s_c":             s_c,
        "eps":             args.eps,
        "pos_channels":    sorted(POS_CHANNELS),
        "signed_channels": sorted(SIGNED_CHANNELS),
        "config":          config,
    }, path)


if __name__ == "__main__":
    main()
