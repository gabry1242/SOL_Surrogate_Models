#!/usr/bin/env python3
"""
train_fm.py  —  Step 2 of the Flow Matching 3-view pipeline

Trains a single shared-weight conditional flow matching model on all 3 views.
Each view is treated as an independent sample — the model learns view-agnostic
features because all views share the same (Hmax, Wmax) canvas and the mask
channel tells the model which pixels are active.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOW MATCHING IN ONE PARAGRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We define a straight-line path from noise y₀∼N(0,I) to data y₁:
  yₜ = (1−t)·y₀ + t·y₁,   t ∈ [0,1]
The ground-truth velocity along this path is v* = y₁ − y₀.
We train a neural network vθ(yₜ, t, x) to predict this velocity.
At inference, start from noise and integrate: y_{t+dt} = yₜ + vθ·dt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Velocity UNet with:
  • Conditioning encoder: processes X alone → multi-scale feature maps
  • These feature maps are REINJECTED into the velocity UNet decoder at
    each resolution via concatenation (activation reinjection)
  • FiLM modulation: time embedding modulates each decoder block
  • Input to velocity UNet: concat(X_norm, yₜ) along channel dim

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

X and Y tensors are ALREADY normalized by build_tensors.py.
No transforms or normalization happen here — the dataset just loads .npy files.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python train_fm.py \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --train_split train --test_split test \\
      --run_dir scripts/runs/fm3v/run1 \\
      --epochs 100 --batch_size 16 --base 64 --lr 3e-4
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ViewDataset(Dataset):
    """Loads one view's pre-normalized X and Y tensors."""
    def __init__(self, x_path: Path, y_path: Path):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")
        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.shape[2:] == self.Y.shape[2:]

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))
        m = (x[0:1] > 0.5).float()   # mask channel → binary
        return x, y, m


# ═══════════════════════════════════════════════════════════════════════════
# Time embedding
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))

    def forward(self, t):
        t = t.view(-1).float()
        a = t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


# ═══════════════════════════════════════════════════════════════════════════
# Network blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GELU())
    def forward(self, x): return self.net(x)


class FiLMBlock(nn.Module):
    """ConvBlock + FiLM modulation from time embedding."""
    def __init__(self, ci, co, t_dim):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.act = nn.GELU()
        self.film = nn.Linear(t_dim, co * 2)

    def forward(self, x, t_emb):
        h = self.act(self.c1(x))
        h = self.act(self.c2(h))
        s, sh = self.film(t_emb).chunk(2, -1)
        return h * (1 + s[..., None, None]) + sh[..., None, None]


# ═══════════════════════════════════════════════════════════════════════════
# Conditioning Encoder (activation reinjection)
# ═══════════════════════════════════════════════════════════════════════════

class CondEncoder(nn.Module):
    """Extracts multi-scale features from X for reinjection into decoder."""
    def __init__(self, c_in, base):
        super().__init__()
        self.e1 = ConvBlock(c_in, base)
        self.e2 = ConvBlock(base, base*2)
        self.e3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(self.pool(f1))
        f3 = self.e3(self.pool(f2))
        return f1, f2, f3


# ═══════════════════════════════════════════════════════════════════════════
# Velocity UNet
# ═══════════════════════════════════════════════════════════════════════════

class VelocityUNet(nn.Module):
    def __init__(self, c_in, c_out, base=32, t_dim=128):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out

        self.time_emb = SinusoidalTimeEmb(t_dim)
        self.cond_enc = CondEncoder(c_in, base)

        # Encoder (processes concat(X, yₜ))
        self.enc1 = ConvBlock(c_in + c_out, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)

        # Decoder with skip + cond injection + FiLM
        # Each level: up_channels + enc_skip + cond_skip
        self.up3 = nn.Conv2d(base*8, base*4, 1)
        self.dec3 = FiLMBlock(base*4 + base*4 + base*4, base*4, t_dim)
        self.up2 = nn.Conv2d(base*4, base*2, 1)
        self.dec2 = FiLMBlock(base*2 + base*2 + base*2, base*2, t_dim)
        self.up1 = nn.Conv2d(base*2, base, 1)
        self.dec1 = FiLMBlock(base + base + base, base, t_dim)
        self.out = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _match(x, hw):
        th, tw = hw; h, w = x.shape[-2:]
        if h == th and w == tw: return x
        if h >= th and w >= tw:
            dh, dw = h-th, w-tw
            return x[..., dh//2:dh//2+th, dw//2:dw//2+tw]
        return F.interpolate(x, (th, tw), mode="bilinear", align_corners=False)

    def forward(self, x_cond, y_t, t):
        te = self.time_emb(t)
        c1, c2, c3 = self.cond_enc(x_cond)

        inp = torch.cat([x_cond, y_t], 1)
        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        u3 = self.up3(F.interpolate(b, e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, self._match(e3,u3.shape[-2:]),
                                       self._match(c3,u3.shape[-2:])], 1), te)
        u2 = self.up2(F.interpolate(d3, e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, self._match(e2,u2.shape[-2:]),
                                       self._match(c2,u2.shape[-2:])], 1), te)
        u1 = self.up1(F.interpolate(d2, e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, self._match(e1,u1.shape[-2:]),
                                       self._match(c1,u1.shape[-2:])], 1), te)
        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════════
# Loss and metrics
# ═══════════════════════════════════════════════════════════════════════════

def masked_mse(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum((0,2,3))
    den = mask.sum((0,2,3)).clamp_min(1e-8)
    return (num / den).mean()

def masked_mae_ch(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    num = (pred - target).abs().mul(mask).sum((0,2,3))
    den = mask.sum((0,2,3)).clamp_min(1e-8)
    return num / den

def masked_rmse_ch(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum((0,2,3))
    den = mask.sum((0,2,3)).clamp_min(1e-8)
    return (num / den).sqrt()

@torch.no_grad()
def euler_sample(model, x, mask, steps=50, device="cpu"):
    B, _, H, W = x.shape
    y = torch.randn(B, model.c_out, H, W, device=device) * mask
    dt = 1.0 / steps
    model.eval()
    for s in range(steps):
        t = torch.full((B,), s/steps, device=device)
        y = (y + model(x, y, t) * dt) * mask
    return y

@torch.no_grad()
def evaluate(model, loader, device, steps=50):
    model.eval()
    mae_sum = rmse_sum = None; n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = euler_sample(model, x, m, steps, device)
        mae = masked_mae_ch(pred, y, m).cpu()
        rmse = masked_rmse_ch(pred, y, m).cpu()
        mae_sum  = mae  if mae_sum  is None else mae_sum  + mae
        rmse_sum = rmse if rmse_sum is None else rmse_sum + rmse
        n += 1
    if n == 0: return {"mae_avg": float("nan"), "rmse_avg": float("nan")}
    mae_sum /= n; rmse_sum /= n
    return {"mae_avg": float(mae_sum.mean()), "rmse_avg": float(rmse_sum.mean()),
            "mae_per_ch": mae_sum.numpy().tolist(),
            "rmse_per_ch": rmse_sum.numpy().tolist()}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--t_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    pfx = args.tensor_prefix
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load datasets: all 3 views concatenated ──
    def load_view_ds(split, view):
        xp = Path(f"{pfx}_view{view}_X_{split}.npy")
        yp = Path(f"{pfx}_view{view}_Y_{split}.npy")
        if not xp.exists(): raise FileNotFoundError(str(xp))
        return ViewDataset(xp, yp)

    train_ds = ConcatDataset([load_view_ds(args.train_split, v) for v in range(3)])
    test_ds  = ConcatDataset([load_view_ds(args.test_split, v) for v in range(3)])

    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Get shapes
    x0, y0, m0 = train_ds[0]
    c_in, c_out = x0.shape[0], y0.shape[0]

    print(f"\n{'='*60}")
    print(f"Flow Matching Training")
    print(f"  Train: {len(train_ds)} samples (3 views × N)")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  c_in={c_in}  c_out={c_out}  canvas={x0.shape[1]}×{x0.shape[2]}")
    print(f"{'='*60}")

    # ── Model ──
    model = VelocityUNet(c_in, c_out, args.base, args.t_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Save config ──
    config = dict(arch="fm_unet_shared", tensor_prefix=pfx,
                  c_in=c_in, c_out=c_out, base=args.base, t_dim=args.t_dim,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, weight_decay=args.weight_decay,
                  seed=args.seed, n_params=n_params,
                  eval_every=args.eval_every, eval_steps=args.eval_steps)
    save_json(run_dir / "config.json", config)

    # ── Training loop ──
    best_rmse = float("inf")
    hist = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_steps = 0.0, 0

        for x, y1, m in train_dl:
            x, y1, m = x.to(device), y1.to(device), m.to(device)
            B = x.shape[0]

            t = torch.rand(B, device=device)
            y0 = torch.randn_like(y1) * m
            t_e = t.view(B, 1, 1, 1)
            yt = ((1 - t_e) * y0 + t_e * y1) * m
            v_target = (y1 - y0) * m

            v_pred = model(x, yt, t) * m
            loss = masked_mse(v_pred, v_target, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item(); n_steps += 1

        scheduler.step()
        train_loss = loss_sum / max(n_steps, 1)
        hist["train"].append({"epoch": epoch, "loss": train_loss})

        # ── Evaluate ──
        do_eval = (epoch % args.eval_every == 0) or epoch == 1 or epoch == args.epochs
        if do_eval:
            te = evaluate(model, test_dl, device, args.eval_steps)
            hist["test"].append({"epoch": epoch, **te})
            cur_rmse = te["rmse_avg"]
            print(f"  Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4g} | "
                  f"test_mae={te['mae_avg']:.4g} | test_rmse={te['rmse_avg']:.4g}")
        else:
            cur_rmse = float("inf")
            print(f"  Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4g}")

        # ── Checkpoint ──
        ckpt = dict(arch="fm_unet_shared", c_in=c_in, c_out=c_out,
                    base=args.base, t_dim=args.t_dim,
                    model_state=model.state_dict(), opt_state=opt.state_dict(),
                    epoch=epoch, eval_steps=args.eval_steps,
                    tensor_prefix=pfx)
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        save_json(run_dir / "metrics.json", hist)

    print(f"\nDone. Best test RMSE = {best_rmse:.6g}")


if __name__ == "__main__":
    main()


