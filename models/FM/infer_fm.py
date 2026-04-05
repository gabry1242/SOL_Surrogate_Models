#!/usr/bin/env python3
"""
infer_fm.py  —  Step 3 of the Flow Matching 3-view pipeline

Loads a trained checkpoint, generates predictions for each view via Euler
ODE integration, converts back to physical units, and saves results + metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVERSE TRANSFORM PIPELINE (per sample)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Euler integration: noise → normalized transformed space   (model output)
2. Denormalize:       y_t = y_norm * y_std + y_mean          (transformed)
3. CLAMP:             clamp y_t to safe range                (prevents overflow)
4. Inverse transform:
     positive channels (te, ti, na): physical = 10^y_t − eps_log
     signed channels   (ua):         physical = s_c · sinh(y_t)

The clamp at step 3 is the key fix — it prevents 10^(huge number) = Inf.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Single-sample inference
  python infer_fm.py \\
      --checkpoint scripts/runs/fm3v/run1/checkpoint_best.pt \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --norm_stats scripts/tensor/fm3v/global3v_norm_stats.npz \\
      --split test --out_dir scripts/runs/fm3v/run1/infer_test \\
      --n_steps 100

  # Multi-sample (uncertainty)
  python infer_fm.py \\
      --checkpoint scripts/runs/fm3v/run1/checkpoint_best.pt \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --norm_stats scripts/tensor/fm3v/global3v_norm_stats.npz \\
      --split test --out_dir scripts/runs/fm3v/run1/infer_test_5s \\
      --n_steps 100 --n_samples 5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES (per view, under --out_dir)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  view{0,1,2}_pred_phys.npy       (N, 22, H, W)  physical units
  view{0,1,2}_pred_norm.npy       (N, 22, H, W)  normalized transformed
  view{0,1,2}_pred_std_phys.npy   (N, 22, H, W)  if n_samples > 1
  metrics.json                     per-view, per-channel MAE/RMSE
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════════════════════════════════════════
# Channel definitions
# ═══════════════════════════════════════════════════════════════════════════

POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))
SPECIES = ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]
CH_NAMES = ["Te","Ti"] + [f"na_{s}" for s in SPECIES] + [f"ua_{s}" for s in SPECIES]

# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ViewXDataset(Dataset):
    def __init__(self, x_path):
        self.X = np.load(x_path, mmap_mode="r")
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.X[i], dtype=np.float32))
        return x, (x[0:1] > 0.5).float()

# ═══════════════════════════════════════════════════════════════════════════
# Model (must match train_fm.py exactly)
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
        h = self.act(self.c1(x)); h = self.act(self.c2(h))
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
        f1 = self.e1(x); f2 = self.e2(self.pool(f1)); f3 = self.e3(self.pool(f2))
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
        self.dec3 = FiLMBlock(base*12, base*4, t_dim)
        self.up2 = nn.Conv2d(base*4, base*2, 1)
        self.dec2 = FiLMBlock(base*6, base*2, t_dim)
        self.up1 = nn.Conv2d(base*2, base, 1)
        self.dec1 = FiLMBlock(base*3, base, t_dim)
        self.out = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _m(x, hw):
        th,tw = hw; h,w = x.shape[-2:]
        if h==th and w==tw: return x
        if h>=th and w>=tw:
            dh,dw=h-th,w-tw; return x[...,dh//2:dh//2+th,dw//2:dw//2+tw]
        return F.interpolate(x,(th,tw),mode="bilinear",align_corners=False)

    def forward(self, x_cond, y_t, t):
        te = self.time_emb(t)
        c1,c2,c3 = self.cond_enc(x_cond)
        inp = torch.cat([x_cond, y_t], 1)
        e1=self.enc1(inp); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); b=self.bottleneck(self.pool(e3))
        u3=self.up3(F.interpolate(b,e3.shape[-2:],mode="bilinear",align_corners=False))
        d3=self.dec3(torch.cat([u3,self._m(e3,u3.shape[-2:]),self._m(c3,u3.shape[-2:])],1),te)
        u2=self.up2(F.interpolate(d3,e2.shape[-2:],mode="bilinear",align_corners=False))
        d2=self.dec2(torch.cat([u2,self._m(e2,u2.shape[-2:]),self._m(c2,u2.shape[-2:])],1),te)
        u1=self.up1(F.interpolate(d2,e1.shape[-2:],mode="bilinear",align_corners=False))
        d1=self.dec1(torch.cat([u1,self._m(e1,u1.shape[-2:]),self._m(c1,u1.shape[-2:])],1),te)
        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════════
# Inverse transform (with safe clamping)
# ═══════════════════════════════════════════════════════════════════════════

def inverse_transform(y_transformed, s_c, eps_log):
    """
    y_transformed: (N, 22, H, W) in transformed space (NOT normalized).
    Returns physical-space values with safe clamping.
    """
    out = np.zeros_like(y_transformed, dtype=np.float64)
    for c in range(y_transformed.shape[1]):
        yt = y_transformed[:, c].astype(np.float64)
        if c in POS_CHANNELS:
            # Clamp to prevent 10^(huge) overflow.  37 → 10^37 ≈ 1e37, safe in float64
            yt_clamped = np.clip(yt, -10, 37)
            out[:, c] = 10.0 ** yt_clamped - eps_log
        elif c in SIGNED_CHANNELS:
            sp = c - 12
            yt_clamped = np.clip(yt, -30, 30)
            out[:, c] = float(s_c[sp]) * np.sinh(yt_clamped)
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Build ground truth in physical space from raw data
# ═══════════════════════════════════════════════════════════════════════════

def build_ground_truth_physical(y_norm, y_mean, y_std, s_c, eps_log):
    """Denormalize + inverse transform ground truth."""
    ym = y_mean.reshape(1, -1, 1, 1)
    ys = y_std.reshape(1, -1, 1, 1)
    y_t = y_norm * ys + ym
    return inverse_transform(y_t, s_c, eps_log)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def masked_metrics_np(pred, truth, mask):
    m = np.broadcast_to(mask, pred.shape).astype(np.float32)
    den = np.maximum(m.sum(axis=(0,2,3)), 1e-8)
    mae = ((np.abs(pred - truth) * m).sum(axis=(0,2,3)) / den)
    rmse = np.sqrt(((pred - truth)**2 * m).sum(axis=(0,2,3)) / den)
    return mae, rmse


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    c_in, c_out = ckpt["c_in"], ckpt["c_out"]
    base, t_dim = ckpt["base"], ckpt["t_dim"]

    model = VelocityUNet(c_in, c_out, base, t_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Load norm stats ──
    ns = np.load(args.norm_stats, allow_pickle=True)
    y_mean  = ns["y_mean"].astype(np.float32)
    y_std   = ns["y_std"].astype(np.float32)
    s_c     = ns["s_c"].astype(np.float64)
    eps_log = float(ns["eps_log"])

    pfx = args.tensor_prefix
    all_metrics = {}

    for view in range(3):
        x_path = Path(f"{pfx}_view{view}_X_{args.split}.npy")
        y_path = Path(f"{pfx}_view{view}_Y_{args.split}.npy")
        if not x_path.exists():
            print(f"  Skipping view{view}: {x_path} not found")
            continue

        print(f"\n--- view{view} ---")
        ds = ViewXDataset(x_path)
        dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type=="cuda"))

        Y_gt_norm = np.load(y_path, mmap_mode="r")
        N, _, H, W = Y_gt_norm.shape

        # ── Generate samples ──
        all_phys = []
        for s in range(args.n_samples):
            pred_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
            masks = np.zeros((N, 1, H, W), dtype=np.float32)
            idx0 = 0
            for x, m in dl:
                b = x.shape[0]; x, m = x.to(device), m.to(device)
                y = torch.randn(b, c_out, H, W, device=device) * m
                dt = 1.0 / args.n_steps
                with torch.no_grad():
                    for step in range(args.n_steps):
                        t = torch.full((b,), step/args.n_steps, device=device)
                        y = (y + model(x, y, t) * dt) * m
                pred_norm[idx0:idx0+b] = y.cpu().numpy()
                masks[idx0:idx0+b] = m.cpu().numpy()
                idx0 += b

            # Denormalize → transformed → physical
            ym = y_mean.reshape(1,-1,1,1); ys = y_std.reshape(1,-1,1,1)
            pred_t = pred_norm * ys + ym
            pred_phys = inverse_transform(pred_t, s_c, eps_log) * masks
            all_phys.append(pred_phys)

            if args.n_samples == 1:
                np.save(out_dir / f"view{view}_pred_norm.npy", pred_norm)

        # ── Aggregate ──
        if args.n_samples == 1:
            final = all_phys[0]
        else:
            stacked = np.stack(all_phys, 0)
            final = stacked.mean(0)
            std = stacked.std(0)
            np.save(out_dir / f"view{view}_pred_std_phys.npy", std)

        np.save(out_dir / f"view{view}_pred_phys.npy", final)

        # ── Ground truth in physical space ──
        gt_norm = np.array(Y_gt_norm[:], dtype=np.float32)
        gt_phys = build_ground_truth_physical(gt_norm, y_mean, y_std, s_c, eps_log) * masks

        # ── Metrics ──
        mae, rmse = masked_metrics_np(final, gt_phys, masks)
        vm = {f"mae_{CH_NAMES[c]}": float(mae[c]) for c in range(22)}
        vm.update({f"rmse_{CH_NAMES[c]}": float(rmse[c]) for c in range(22)})
        vm["mae_avg"] = float(mae.mean()); vm["rmse_avg"] = float(rmse.mean())
        all_metrics[f"view{view}"] = vm

        print(f"  MAE avg:  {vm['mae_avg']:.4g}")
        print(f"  RMSE avg: {vm['rmse_avg']:.4g}")
        print(f"  Te MAE:   {vm['mae_Te']:.4g}  Ti MAE: {vm['mae_Ti']:.4g}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
