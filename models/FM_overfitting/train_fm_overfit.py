#!/usr/bin/env python3
"""
train_fm_te_ti.py  —  Flow Matching on Te + Ti only

Uses the SAME tensors from build_tensors.py — just selects Y channels 0,1
(Te and Ti in normalized log-space).  No tensor rebuild needed.

The model has c_out=2 instead of 22, so all capacity goes to temperatures.


python scripts/models/FM_T/train_fm_te_ti.py --tensor_prefix scripts/tensor/fm3v/global3v --train_split train --test_split test --run_dir scripts/runs/fm3v/te_ti_run1 --epochs 150 --batch_size 16 --base 64 --lr 3e-4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python train_fm_te_ti.py \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --train_split train --test_split test \\
      --run_dir scripts/runs/fm3v/te_ti_run1 \\
      --epochs 150 --batch_size 16 --base 64 --lr 3e-4
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset


# ═══════════════════════════════════════════════════════════════════════════
# Dataset — selects only channels 0,1 from Y
# ═══════════════════════════════════════════════════════════════════════════

Y_CHANNELS = [0, 1]  # Te, Ti

class ViewDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        self.X = np.load(x_path, mmap_mode="r")   # (N, 9, H, W)
        self.Y = np.load(y_path, mmap_mode="r")   # (N, 22, H, W)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y_full = np.array(self.Y[idx], dtype=np.float32)
        y = torch.from_numpy(y_full[Y_CHANNELS])  # (2, H, W)
        m = (x[0:1] > 0.5).float()
        return x, y, m


# ═══════════════════════════════════════════════════════════════════════════
# Model (identical architecture, just c_out=2)
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

class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GELU())
    def forward(self, x): return self.net(x)

class FiLMBlock(nn.Module):
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

class CondEncoder(nn.Module):
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

class VelocityUNet(nn.Module):
    def __init__(self, c_in, c_out, base=32, t_dim=128):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.time_emb = SinusoidalTimeEmb(t_dim)
        self.cond_enc = CondEncoder(c_in, base)

        self.enc1 = ConvBlock(c_in + c_out, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)

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
# Loss and evaluation
# ═══════════════════════════════════════════════════════════════════════════

def masked_mse(pred, target, mask):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)

    loss = (pred - target).pow(2) * mask
    return loss.sum() / mask.sum().clamp_min(1e-8)

def masked_mae_ch(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    return (pred - target).abs().mul(mask).sum((0,2,3)).div(
        mask.sum((0,2,3)).clamp_min(1e-8))

def masked_rmse_ch(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    return ((pred - target).pow(2).mul(mask).sum((0,2,3)).div(
        mask.sum((0,2,3)).clamp_min(1e-8))).sqrt()

@torch.no_grad()
def euler_sample_single(model, x, y0, mask, steps=50, device="cpu"):
    y = y0.clone().to(device)
    dt = 1.0 / steps
    model.eval()
    for s in range(steps):
        t = torch.full((x.shape[0],), s / steps, device=device)
        y = (y + model(x, y, t) * dt) * mask
    return y

@torch.no_grad()
def evaluate(model, loader, device, steps=50):
    model.eval()
    mae_sum = rmse_sum = None; n = 0
    for x, y, m in loader:
        x, y, m = next(iter(loader))
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = euler_sample_single(model, x, 0.03*torch.randn_like(y), m, steps=50, device=device)
        mae = masked_mae_ch(pred, y, m).cpu()
        rmse = masked_rmse_ch(pred, y, m).cpu()
        mae_sum  = mae  if mae_sum  is None else mae_sum  + mae
        rmse_sum = rmse if rmse_sum is None else rmse_sum + rmse
        n += 1
    if n == 0: return {"mae_avg": float("nan"), "rmse_avg": float("nan")}
    mae_sum /= n; rmse_sum /= n
    return {"mae_avg": float(mae_sum.mean()), "rmse_avg": float(rmse_sum.mean()),
            "mae_Te": float(mae_sum[0]), "mae_Ti": float(mae_sum[1]),
            "rmse_Te": float(rmse_sum[0]), "rmse_Ti": float(rmse_sum[1])}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)



import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def plot_prediction(model, x, y_true, mask, device, title="overfit_result"):
    model.eval()

    x = x.to(device)
    y_true = y_true.to(device)
    mask = mask.to(device)

    # initialize from noise (same as training idea)
    y = 0.01 * torch.randn_like(y_true)

    # simple Euler rollout (same as your sampling)
    steps = 100
    dt = 1.0 / steps

    for s in range(steps):
        t = torch.full((x.shape[0],), s / steps, device=device)
        v = model(x, y, t)
        y = y + v * dt
        y = y * mask

    y_pred = y

    # move to CPU for plotting
    y_true = y_true.squeeze().cpu().numpy()
    y_pred = y_pred.squeeze().cpu().numpy()

    n_ch = y_true.shape[0]

    fig, axes = plt.subplots(n_ch, 3, figsize=(12, 4 * n_ch))

    if n_ch == 1:
        axes = [axes]

    for c in range(n_ch):
        gt = y_true[c]
        pred = y_pred[c]
        err = abs(gt - pred)

        im0 = axes[c][0].imshow(gt)
        axes[c][0].set_title(f"GT channel {c}")
        plt.colorbar(im0, ax=axes[c][0])

        im1 = axes[c][1].imshow(pred)
        axes[c][1].set_title(f"Pred channel {c}")
        plt.colorbar(im1, ax=axes[c][1])

        im2 = axes[c][2].imshow(err)
        axes[c][2].set_title(f"Abs error {c}")
        plt.colorbar(im2, ax=axes[c][2])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
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

    # ── Load datasets ──
    def load_ds(split, view):
        xp = Path(f"{pfx}_view{view}_X_{split}.npy")
        yp = Path(f"{pfx}_view{view}_Y_{split}.npy")
        if not xp.exists(): raise FileNotFoundError(str(xp))
        return ViewDataset(xp, yp)

    train_ds = ConcatDataset([load_ds(args.train_split, v) for v in range(3)])
    test_ds  = ConcatDataset([load_ds(args.test_split, v) for v in range(3)])

    train_ds = torch.utils.data.Subset(train_ds, [0])
    test_ds  = torch.utils.data.Subset(test_ds, [0])

    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    x0, y0, m0 = train_ds[0]
    c_in  = x0.shape[0]   # 9
    c_out = y0.shape[0]   # 2 (Te, Ti)

    print(f"\n{'='*60}")
    print(f"Flow Matching — Te + Ti only")
    print(f"  Train: {len(train_ds)} samples  |  Test: {len(test_ds)} samples")
    print(f"  c_in={c_in}  c_out={c_out}  canvas={x0.shape[1]}x{x0.shape[2]}")
    print(f"  Y channels: {Y_CHANNELS} (Te, Ti)")
    print(f"{'='*60}")

    # ── Model ──
    model = VelocityUNet(c_in, c_out, args.base, args.t_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    config = dict(arch="fm_unet_te_ti", y_channels=Y_CHANNELS,
                  tensor_prefix=pfx, c_in=c_in, c_out=c_out,
                  base=args.base, t_dim=args.t_dim,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, weight_decay=args.weight_decay,
                  seed=args.seed, n_params=n_params,
                  eval_every=args.eval_every, eval_steps=args.eval_steps)
    save_json(run_dir / "config.json", config)

    # ── Training loop ──
    best_rmse = float("inf")
    hist = {"train": [], "test": []}

    fixed_y0 = (torch.randn_like(y0)).to(device).unsqueeze(0)

    x = x0.to(device).unsqueeze(0)
    y1 = y0.to(device).unsqueeze(0)
    m = m0.to(device).unsqueeze(0)

    for epoch in range(1, args.epochs + 1):
        model.train()

        B = 8  # or 16

        t = torch.rand(B, device=device)
        t = t[:, None, None, None]

        y1_b = y1.expand(B, -1, -1, -1)
        x_b  = x.expand(B, -1, -1, -1)
        m_b  = m.expand(B, -1, -1, -1)

        # 🔥 resample noise
        y_init = torch.randn_like(y1_b) * 0.1

        yt = ((1 - t) * y_init + t * y1_b) * m_b
        v_target = (y1_b - y_init) * m_b

        v_pred = model(x_b, yt, t.squeeze())

        v_pred = v_pred * m_b   # 🔥 CRITICAL

        loss = masked_mse(v_pred, v_target, m_b)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ── metrics ──
        with torch.no_grad():
            mae_ch = masked_mae_ch(v_pred, v_target, m)   # (2,)
            rmse_ch = masked_rmse_ch(v_pred, v_target, m) # (2,)

            rmse = rmse_ch.mean().item()

            v_err = (v_pred - v_target).abs().mean().item()

        # ── logging dict ──
        log_entry = {
            "epoch": epoch,
            "loss": loss.item(),
            "rmse": rmse,
            "rmse_te": rmse_ch[0].item(),
            "rmse_ti": rmse_ch[1].item(),
            "mae_te": mae_ch[0].item(),
            "mae_ti": mae_ch[1].item(),
            "v_err": v_err,
        }

        hist["train"].append(log_entry)

        # ── print ──
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.6f} | "
            f"RMSE: {rmse:.6f} | "
            f"MAE Te: {mae_ch[0].item():.6f} | "
            f"MAE Ti: {mae_ch[1].item():.6f} | "
            f"v_err: {v_err:.6f}"
        )

        # ── save metrics ──
        save_json(run_dir / "metrics.json", hist)

        # ── checkpointing (optional but recommended) ──
        ckpt = {
            "arch": "fm_unet_te_ti",
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

    x, y, m = x0, y0, m0

    plot_prediction(
        model,
        x.unsqueeze(0),
        y.unsqueeze(0),
        m.unsqueeze(0),
        device,
        title="Single-sample overfit result"
    )
if __name__ == "__main__":
    main()
