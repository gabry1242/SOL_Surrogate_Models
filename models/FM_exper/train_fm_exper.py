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

# ─────────────────────────────────────────────────────────────
# Single-sample 3-view dataset
# ─────────────────────────────────────────────────────────────

class SingleSample3ViewDataset(Dataset):
    def __init__(self, tensor_prefix, split="train", sample_idx=0):
        self.views = []

        for v in range(3):
            xp = Path(f"{tensor_prefix}_view{v}_X_{split}.npy")
            yp = Path(f"{tensor_prefix}_view{v}_Y_{split}.npy")

            X = np.load(xp, mmap_mode="r")
            Y = np.load(yp, mmap_mode="r")

            x = torch.from_numpy(np.array(X[sample_idx], dtype=np.float32))
            y_full = np.array(Y[sample_idx], dtype=np.float32)
            y = torch.from_numpy(y_full[Y_CHANNELS])

            m = (x[0:1] > 0.5).float()

            self.views.append((x, y, m))

    def __len__(self):
        return 1  # only one sample

    def __getitem__(self, idx):
        return self.views  # list of 3 views

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
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    return ((pred - target).pow(2) * mask).sum((0,2,3)).div(
        mask.sum((0,2,3)).clamp_min(1e-8)).mean()

def masked_mae_ch(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    return (pred - target).abs().mul(mask).sum((0,2,3)).div(
        mask.sum((0,2,3)).clamp_min(1e-8))

def masked_rmse_ch(pred, target, mask):
    if mask.shape[1] == 1: mask = mask.expand_as(pred)
    return ((pred - target).pow(2).mul(mask).sum((0,2,3)).div(
        mask.sum((0,2,3)).clamp_min(1e-8))).sqrt()

@torch.no_grad()
def euler_sample(model, x, mask, steps=50, device="cpu"):
    B, _, H, W = x.shape
    y = torch.randn(B, model.c_out, H, W, device=device) * mask
    dt = 1.0 / steps
    model.eval()
    for s in range(steps):
        t = torch.full((B,), s / steps, device=device)
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


    train_ds = SingleSample3ViewDataset(
        tensor_prefix=args.tensor_prefix,
        split=args.train_split,
        sample_idx=0
    )

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
    views0 = train_ds[0]        # list of 3 views
    x0, y0, m0 = views0[0]  
    c_in  = x0.shape[0]   # 9
    c_out = y0.shape[0]   # 2 (Te, Ti)

    print(f"\n{'='*60}")
    print(f"Flow Matching — Te + Ti only")

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for views in train_dl:
            views = views[0]  # VERY IMPORTANT

            X, Y, M = views  # tensors with shape (3, ...)

            total_loss = 0.0

            for i in range(3):
                x = X[i:i+1].to(device)
                y1 = Y[i:i+1].to(device)
                m = M[i:i+1].to(device)

                B = x.shape[0]

                t = torch.rand(B, device=device)
                y0 = torch.randn_like(y1) * m
                t_e = t.view(B, 1, 1, 1)

                yt = ((1 - t_e) * y0 + t_e * y1) * m
                v_target = (y1 - y0) * m

                v_pred = model(x, yt, t) * m
                loss = masked_mse(v_pred, v_target, m)

                total_loss += loss

            total_loss = total_loss / 3.0

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()

            loss_sum += total_loss.item()
        # scheduler.step()
        print(f"Epoch {epoch:03d} | loss={loss_sum:.6f}")
    
    import matplotlib.pyplot as plt

    @torch.no_grad()
    def plot_overfit_result(model, dataset, device, steps=50):
        model.eval()

        views = dataset[0]  # list of 3 views

        fig, axes = plt.subplots(3, 6, figsize=(18, 9))

        for v, (x, y_true, m) in enumerate(views):
            x = x.unsqueeze(0).to(device)
            y_true = y_true.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)

            # ── Sampling (Flow Matching Euler) ──
            B, _, H, W = x.shape
            y = torch.randn(B, model.c_out, H, W, device=device) * m

            dt = 1.0 / steps
            for s in range(steps):
                t = torch.full((B,), s / steps, device=device)
                y = (y + model(x, y, t) * dt) * m

            y_pred = y[0].cpu().numpy()
            y_true = y_true[0].cpu().numpy()

            # ── Plot Te (channel 0) ──
            axes[v, 0].imshow(y_true[0], origin="lower")
            axes[v, 0].set_title(f"View {v} Te GT")

            axes[v, 1].imshow(y_pred[0], origin="lower")
            axes[v, 1].set_title(f"View {v} Te Pred")

            axes[v, 2].imshow(abs(y_true[0] - y_pred[0]), origin="lower")
            axes[v, 2].set_title(f"View {v} Te Error")

            # ── Plot Ti (channel 1) ──
            axes[v, 3].imshow(y_true[1], origin="lower")
            axes[v, 3].set_title(f"View {v} Ti GT")

            axes[v, 4].imshow(y_pred[1], origin="lower")
            axes[v, 4].set_title(f"View {v} Ti Pred")

            axes[v, 5].imshow(abs(y_true[1] - y_pred[1]), origin="lower")
            axes[v, 5].set_title(f"View {v} Ti Error")

            for j in range(6):
                axes[v, j].axis("off")

        plt.tight_layout()
        plt.show()
    plot_overfit_result(model, train_ds, device)

if __name__ == "__main__":
    main()
