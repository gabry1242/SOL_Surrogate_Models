#!/usr/bin/env python3
"""
infer_fm_te_ti.py  —  Inference for Te + Ti flow matching model

Loads checkpoint, generates predictions via Euler integration, converts
back to electron-volts, computes metrics in both normalized and physical space.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVERSE TRANSFORM (Te/Ti only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  normalized → denormalize → transformed space → physical (eV)

  denormalize:  y_t = y_norm * y_std + y_mean
  inverse:      physical_eV = 10^(y_t) - eps_log

  Te/Ti range: ~1e-5 to ~1e6 eV → log10 range: [0, 6] (with eps_log=1)
  Safe and well-behaved, no overflow risk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python infer_fm_te_ti.py \\
      --checkpoint scripts/runs/fm3v/te_ti_run1/checkpoint_best.pt \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --norm_stats scripts/tensor/fm3v/global3v_norm_stats.npz \\
      --split test \\
      --out_dir scripts/runs/fm3v/te_ti_run1/infer_test \\
      --n_steps 100

  # Multi-sample for uncertainty
  python infer_fm_te_ti.py \\
      --checkpoint scripts/runs/fm3v/te_ti_run1/checkpoint_best.pt \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --norm_stats scripts/tensor/fm3v/global3v_norm_stats.npz \\
      --split test \\
      --out_dir scripts/runs/fm3v/te_ti_run1/infer_test_5s \\
      --n_steps 100 --n_samples 5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES (per view)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  view{0,1,2}_pred_eV.npy       (N, 2, H, W)  Te,Ti in eV
  view{0,1,2}_truth_eV.npy      (N, 2, H, W)  ground truth in eV
  view{0,1,2}_pred_norm.npy     (N, 2, H, W)  normalized (model output)
  view{0,1,2}_pred_std_eV.npy   (N, 2, H, W)  if n_samples > 1
  metrics.json
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
# Model (must match train_fm_te_ti.py)
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    def forward(self, t):
        t = t.view(-1).float(); a = t[:,None]*self.freq[None,:]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))

class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci,co,3,padding=1), nn.GELU(),
                                  nn.Conv2d(co,co,3,padding=1), nn.GELU())
    def forward(self, x): return self.net(x)

class FiLMBlock(nn.Module):
    def __init__(self, ci, co, t_dim):
        super().__init__()
        self.c1=nn.Conv2d(ci,co,3,padding=1); self.c2=nn.Conv2d(co,co,3,padding=1)
        self.act=nn.GELU(); self.film=nn.Linear(t_dim, co*2)
    def forward(self, x, t_emb):
        h=self.act(self.c1(x)); h=self.act(self.c2(h))
        s,sh=self.film(t_emb).chunk(2,-1)
        return h*(1+s[...,None,None])+sh[...,None,None]

class CondEncoder(nn.Module):
    def __init__(self, c_in, base):
        super().__init__()
        self.e1=ConvBlock(c_in,base); self.e2=ConvBlock(base,base*2)
        self.e3=ConvBlock(base*2,base*4); self.pool=nn.MaxPool2d(2)
    def forward(self, x):
        f1=self.e1(x); f2=self.e2(self.pool(f1)); f3=self.e3(self.pool(f2))
        return f1,f2,f3

class VelocityUNet(nn.Module):
    def __init__(self, c_in, c_out, base=32, t_dim=128):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.time_emb=SinusoidalTimeEmb(t_dim); self.cond_enc=CondEncoder(c_in,base)
        self.enc1=ConvBlock(c_in+c_out,base); self.enc2=ConvBlock(base,base*2)
        self.enc3=ConvBlock(base*2,base*4); self.pool=nn.MaxPool2d(2)
        self.bottleneck=ConvBlock(base*4,base*8)
        self.up3=nn.Conv2d(base*8,base*4,1); self.dec3=FiLMBlock(base*12,base*4,t_dim)
        self.up2=nn.Conv2d(base*4,base*2,1); self.dec2=FiLMBlock(base*6,base*2,t_dim)
        self.up1=nn.Conv2d(base*2,base,1);   self.dec1=FiLMBlock(base*3,base,t_dim)
        self.out=nn.Conv2d(base,c_out,1)
    @staticmethod
    def _m(x,hw):
        th,tw=hw; h,w=x.shape[-2:]
        if h==th and w==tw: return x
        if h>=th and w>=tw: dh,dw=h-th,w-tw; return x[...,dh//2:dh//2+th,dw//2:dw//2+tw]
        return F.interpolate(x,(th,tw),mode="bilinear",align_corners=False)
    def forward(self, x_cond, y_t, t):
        te=self.time_emb(t); c1,c2,c3=self.cond_enc(x_cond)
        inp=torch.cat([x_cond,y_t],1)
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
# Inverse transform: normalized → eV
# ═══════════════════════════════════════════════════════════════════════════

def to_eV(y_norm, y_mean_sel, y_std_sel, eps_log):
    """Convert normalized predictions to electron-volts.
    y_norm: (N, 2, H, W) in normalized space
    Returns (N, 2, H, W) in eV, float64.
    """
    ym = y_mean_sel.reshape(1, -1, 1, 1)
    ys = y_std_sel.reshape(1, -1, 1, 1)
    y_t = (y_norm * ys + ym).astype(np.float64)       # transformed space (log10)
    y_t_clamped = np.clip(y_t, -2, 8)                  # Te/Ti max ~1e6 eV → log10=6, generous margin
    return 10.0 ** y_t_clamped - float(eps_log)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def masked_metrics(pred, truth, mask):
    """Per-channel MAE and RMSE in float64."""
    p = pred.astype(np.float64); t = truth.astype(np.float64)
    m = np.broadcast_to(mask, pred.shape).astype(np.float64)
    den = np.maximum(m.sum(axis=(0,2,3)), 1e-8)
    mae = (np.abs(p - t) * m).sum(axis=(0,2,3)) / den
    rmse = np.sqrt(((p - t)**2 * m).sum(axis=(0,2,3)) / den)
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
    y_channels = ckpt["y_channels"]  # [0, 1]

    model = VelocityUNet(c_in, c_out, base, t_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Load norm stats (only Te, Ti channels) ──
    ns = np.load(args.norm_stats, allow_pickle=True)
    y_mean_full = ns["y_mean"].astype(np.float32)
    y_std_full  = ns["y_std"].astype(np.float32)
    eps_log     = float(ns["eps_log"])

    y_mean_sel = y_mean_full[y_channels]  # (2,)
    y_std_sel  = y_std_full[y_channels]   # (2,)

    print(f"\n{'='*60}")
    print(f"Inference — Te + Ti")
    print(f"  y_channels: {y_channels}")
    print(f"  y_mean: Te={y_mean_sel[0]:.4f}  Ti={y_mean_sel[1]:.4f}")
    print(f"  y_std:  Te={y_std_sel[0]:.4f}  Ti={y_std_sel[1]:.4f}")
    print(f"  eps_log: {eps_log}")
    print(f"  n_steps={args.n_steps}  n_samples={args.n_samples}")
    print(f"{'='*60}")

    pfx = args.tensor_prefix
    all_metrics = {}

    for view in range(3):
        x_path = Path(f"{pfx}_view{view}_X_{args.split}.npy")
        y_path = Path(f"{pfx}_view{view}_Y_{args.split}.npy")
        if not x_path.exists():
            print(f"  Skipping view{view}"); continue

        print(f"\n--- view{view} ---")
        ds = ViewXDataset(x_path)
        dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))

        Y_gt_full = np.load(y_path, mmap_mode="r")   # (N, 22, H, W)
        N, _, H, W = Y_gt_full.shape
        Y_gt_norm = np.array(Y_gt_full[:, y_channels, :, :], dtype=np.float32)  # (N, 2, H, W)

        # ── Generate samples ──
        all_eV = []
        all_norm = []

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
                        t = torch.full((b,), step / args.n_steps, device=device)
                        y = (y + model(x, y, t) * dt) * m
                pred_norm[idx0:idx0+b] = y.cpu().numpy()
                masks[idx0:idx0+b] = m.cpu().numpy()
                idx0 += b

            pred_eV = to_eV(pred_norm, y_mean_sel, y_std_sel, eps_log) * masks
            all_eV.append(pred_eV)
            all_norm.append(pred_norm)

        # ── Aggregate ──
        if args.n_samples == 1:
            final_eV = all_eV[0]
            final_norm = all_norm[0]
        else:
            stacked = np.stack(all_eV, 0)
            final_eV = stacked.mean(0)
            std_eV = stacked.std(0)
            np.save(out_dir / f"view{view}_pred_std_eV.npy", std_eV.astype(np.float32))
            final_norm = np.stack(all_norm, 0).mean(0)

        np.save(out_dir / f"view{view}_pred_eV.npy", final_eV.astype(np.float32))
        np.save(out_dir / f"view{view}_pred_norm.npy", final_norm.astype(np.float32))

        # ── Ground truth in eV ──
        gt_eV = to_eV(Y_gt_norm, y_mean_sel, y_std_sel, eps_log) * masks
        np.save(out_dir / f"view{view}_truth_eV.npy", gt_eV.astype(np.float32))

        # ── Metrics in physical space (eV) ──
        mae_eV, rmse_eV = masked_metrics(final_eV, gt_eV, masks)

        # ── Metrics in normalized space ──
        mae_norm, rmse_norm = masked_metrics(final_norm, Y_gt_norm, masks)

        vm = {
            "mae_eV_Te": float(mae_eV[0]), "mae_eV_Ti": float(mae_eV[1]),
            "rmse_eV_Te": float(rmse_eV[0]), "rmse_eV_Ti": float(rmse_eV[1]),
            "mae_norm_Te": float(mae_norm[0]), "mae_norm_Ti": float(mae_norm[1]),
            "rmse_norm_Te": float(rmse_norm[0]), "rmse_norm_Ti": float(rmse_norm[1]),
            "mae_eV_avg": float(mae_eV.mean()), "rmse_eV_avg": float(rmse_eV.mean()),
            "mae_norm_avg": float(mae_norm.mean()), "rmse_norm_avg": float(rmse_norm.mean()),
        }
        all_metrics[f"view{view}"] = vm

        print(f"  Normalized space:  MAE Te={vm['mae_norm_Te']:.4f}  Ti={vm['mae_norm_Ti']:.4f}")
        print(f"  Physical (eV):     MAE Te={vm['mae_eV_Te']:.2f}    Ti={vm['mae_eV_Ti']:.2f}")
        print(f"  Physical (eV):    RMSE Te={vm['rmse_eV_Te']:.2f}   Ti={vm['rmse_eV_Ti']:.2f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
