#!/usr/bin/env python3
"""
train_fm_full22.py — Flow Matching on FULL 22 channels

Predicts:
  Te, Ti
  10 species densities (na_*)
  10 species velocities (ua_*)

Output shape:
  (N, 22, H, W)
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset


# ═══════════════════════════════════════════════════════════════════════════
# Channel names (22 total)
# ═══════════════════════════════════════════════════════════════════════════

SPECIES = ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]

CH_NAMES = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)

CHANNEL_GROUPS = {
    "Te": list(range(0, 2)),
    "Ua": list(range(2, 12)),
    "Dn": list(range(12, 22)),
    "All": list(range(0, 22)),
}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset (FULL Y, no slicing)
# ═══════════════════════════════════════════════════════════════════════════

class ViewDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, y_idx=None):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")
        self.y_idx = y_idx

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))

        y = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))

        if self.y_idx is not None:
            y = y[self.y_idx]   # <<< KEY CHANGE

        m = (x[0:1] > 0.5).float()

        return x, y, m

# ═══════════════════════════════════════════════════════════════════════════
# Model (UNCHANGED ARCHITECTURE)
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, t):
        t = t.view(-1).float()
        a = t[:, None] * self.freq[None, :]
        emb = torch.cat([a.sin(), a.cos()], -1)
        return self.mlp(emb)


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


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
        s, b = self.film(t_emb).chunk(2, -1)
        return h * (1 + s[..., None, None]) + b[..., None, None]


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
    def __init__(self, c_in, c_out, base=64, t_dim=128):
        super().__init__()

        self.time_emb = SinusoidalTimeEmb(t_dim)
        self.cond_enc = CondEncoder(c_in, base)

        self.enc1 = ConvBlock(c_in + c_out, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)

        self.up3 = nn.Conv2d(base*8, base*4, 1)
        self.dec3 = FiLMBlock(base*12, base*4, t_dim)

        self.up2 = nn.Conv2d(base*4, base*2, 1)
        self.dec2 = FiLMBlock(base*6, base*2, t_dim)

        self.up1 = nn.Conv2d(base*2, base, 1)
        self.dec1 = FiLMBlock(base*3, base, t_dim)

        self.out = nn.Conv2d(base, c_out, 1)

    def forward(self, x_cond, y_t, t):
        te = self.time_emb(t)

        c1, c2, c3 = self.cond_enc(x_cond)

        inp = torch.cat([x_cond, y_t], 1)

        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        u3 = self.up3(F.interpolate(b, e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, e3, c3], 1), te)

        u2 = self.up2(F.interpolate(d3, e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, e2, c2], 1), te)

        u1 = self.up1(F.interpolate(d2, e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, e1, c1], 1), te)

        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

def masked_mse(pred, target, mask):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1e-8)


def masked_mae(pred, target, mask):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    return ((pred - target).abs() * mask).sum((0,2,3)) / mask.sum((0,2,3)).clamp_min(1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Plot (FULL 22 CHANNELS)
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

def ensure_4d(x):
    # remove ALL accidental batch stacking
    while x.dim() > 4:
        x = x.squeeze(0)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Plot (3-view aware, one figure per output channel)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_prediction_multiview(
    model,
    x_fused,        # (1, 3*C_in, H, W)  — fused input as used in training
    xs_list,        # list of 3 tensors, each (1, C_in, H, W) — per-view inputs
    y_true,         # (1, C_out, H, W)   — ground truth (from view 0, same for all)
    mask_fused,     # (1, 1,     H, W)   — fused mask (union)
    device,
    selected_ch_names,
    c_in_per_view,  # int — number of input channels per view
    title="run",
):
    model.eval()

    def to_bchw(t):
        while t.dim() > 4:
            t = t.squeeze(0)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t

    x_fused   = to_bchw(x_fused).to(device)
    y_true    = to_bchw(y_true).to(device)
    mask_fused = to_bchw(mask_fused).to(device)

    # ── Build per-view masks from channel 0 of each view's input ──────────
    #    Channel 0 of each view-X is the binary mask for that view.
    view_masks = []
    for v in range(3):
        x_v = to_bchw(xs_list[v]).to(device)          # (1, C_in, H, W)
        m_v = (x_v[:, 0:1] > 0.5).float()             # (1, 1,    H, W)
        view_masks.append(m_v)

    # ── Run flow-matching ODE (single pass on fused input) ────────────────
    steps = 100
    dt    = 1.0 / steps
    y     = 0.01 * torch.randn_like(y_true)

    for s in range(steps):
        t_val = torch.full((x_fused.shape[0],), s / steps, device=device)
        v     = model(x_fused, y, t_val)
        y     = y + v * dt
        y     = y * mask_fused          # zero out background

    y_pred = y  # (1, C_out, H, W)

    # ── To numpy ──────────────────────────────────────────────────────────
    y_true_np  = y_true.squeeze(0).cpu().numpy()   # (C_out, H, W)
    y_pred_np  = y_pred.squeeze(0).cpu().numpy()
    vm_np      = [m.squeeze().cpu().numpy() for m in view_masks]  # 3 x (H,W)

    n_ch = y_true_np.shape[0]

    # ── One figure per output channel, columns = views ────────────────────
    for c in range(n_ch):
        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        fig.suptitle(f"{title}  |  channel {c}: {selected_ch_names[c]}", fontsize=13)

        row_labels = ["GT", "Prediction", "Abs Error"]

        for v in range(3):
            m   = vm_np[v]                      # (H, W) binary
            gt  = y_true_np[c] * m
            pr  = y_pred_np[c] * m
            err = np.abs(gt - pr)

            # shared scale for GT / Pred (ignore masked zeros for range)
            vals = gt[m > 0.5] if m.sum() > 0 else np.array([0.0])
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmin == vmax:
                vmax = vmin + 1e-8

            im0 = axes[0, v].imshow(gt,  vmin=vmin, vmax=vmax, origin="lower")
            axes[0, v].set_title(f"GT  — view {v}")
            plt.colorbar(im0, ax=axes[0, v], fraction=0.046)

            im1 = axes[1, v].imshow(pr,  vmin=vmin, vmax=vmax, origin="lower")
            axes[1, v].set_title(f"Pred — view {v}")
            plt.colorbar(im1, ax=axes[1, v], fraction=0.046)

            im2 = axes[2, v].imshow(err, cmap="inferno", origin="lower")
            axes[2, v].set_title(f"Abs Err — view {v}")
            plt.colorbar(im2, ax=axes[2, v], fraction=0.046)

        for i, rl in enumerate(row_labels):
            axes[i, 0].set_ylabel(rl, fontsize=11)

        plt.tight_layout()
        plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--channels", default="All", choices=["Te", "Ua", "Dn", "All"])
    args = ap.parse_args()

    set_seed(42)

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pfx = args.tensor_prefix

    y_idx = CHANNEL_GROUPS[args.channels]
    selected_ch_names = [CH_NAMES[i] for i in y_idx]
    # ── Load datasets ──
    def load_ds(split, view):
        xp = Path(f"{pfx}_view{view}_X_{split}.npy")
        yp = Path(f"{pfx}_view{view}_Y_{split}.npy")

        if not xp.exists():
            raise FileNotFoundError(str(xp))

        return ViewDataset(xp, yp, y_idx=y_idx)


    class MultiViewDataset(torch.utils.data.Dataset):
        """
        Fuses 3 independent ViewDataset objects WITHOUT modifying ViewDataset.
        """

        def __init__(self, split):
            self.views = [load_ds(split, v) for v in range(3)]

            # assume identical length across views
            self.n = len(self.views[0])

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            xs, ys, ms = [], [], []

            for v in self.views:
                x, y, m = v[idx]
                xs.append(x)
                ys.append(y)
                ms.append(m)

            # fuse input channels
            x_fused = torch.cat(xs, dim=0)

            # same target across views → pick one
            y_fused = ys[0]

            # fuse masks (union of valid cells)
            m_fused = torch.stack(ms, dim=0).amax(dim=0)

            return x_fused, y_fused, m_fused, xs


    train_ds = MultiViewDataset(args.train_split)
    test_ds  = MultiViewDataset(args.test_split)

    train_ds = torch.utils.data.Subset(train_ds, [0])
    test_ds  = torch.utils.data.Subset(test_ds, [0])

    x0, y0, m0, xs0 = train_ds[0]
    c_in = x0.shape[0]
    c_out = len(y_idx)

    model = VelocityUNet(c_in, c_out, args.base).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    x = x0.unsqueeze(0).to(device)
    y = y0.unsqueeze(0).to(device)
    m = m0.unsqueeze(0).to(device)
    xs_list_plot = [xv.unsqueeze(0) for xv in xs0]   # 3 x (1, C_in, H, W)

    for epoch in range(args.epochs):
        model.train()

        B = 4
        t = torch.rand(B, device=device)

        x_b = x.expand(B, -1, -1, -1)
        y1 = y.expand(B, -1, -1, -1)
        m_b = m.expand(B, -1, -1, -1)

        y0_noise = torch.randn_like(y1) * 0.1

        yt = (1-t[:,None,None,None]) * y0_noise + t[:,None,None,None] * y1
        vt = (y1 - y0_noise)

        v_pred = model(x_b, yt, t)

        loss = masked_mse(v_pred, vt, m_b)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Epoch {epoch} | loss {loss.item():.6f}")

    raw_item = train_ds.dataset[train_ds.indices[0]]   # (x_fused, y_fused, m_fused, xs)
    xs_list_plot = [xv.unsqueeze(0) for xv in raw_item[3]]   # 3 x (1, C_in, H, W)

    plot_prediction_multiview(
        model,
        x_fused           = x,
        xs_list           = xs_list_plot,
        y_true            = y,
        mask_fused        = m,
        device            = device,
        selected_ch_names = selected_ch_names,
        c_in_per_view     = c_in // 3,
        title             = args.run_dir,
    )
if __name__ == "__main__":
    main()