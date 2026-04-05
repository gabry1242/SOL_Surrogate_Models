#!/usr/bin/env python3
"""
infer_fm_3views.py

Run inference with a Flow Matching checkpoint produced by train_fm_3views.py.

=============================================================================
INFERENCE PROCEDURE
=============================================================================

1. Load checkpoint (contains all normalization stats, model weights, view tag).
2. For each test sample:
   a. Normalize X using stored x_mean / x_std.
   b. Sample noise y_0 ~ N(0, I) masked to active region.
   c. Integrate the learned velocity field from t=0 to t=1 using Euler steps:
        y_{t+dt} = y_t + v_θ(x, y_t, t) * dt
   d. The result y_1 is in normalized transformed space.
   e. Denormalize: y_t = y_1 * y_std + y_mean  (back to transformed space).
   f. Inverse transform: physical units (10^y for positive, s_c*sinh(y) for signed).
3. Optionally generate multiple samples (--n_samples) for uncertainty quantification.
4. Save predictions and per-channel masked MAE/RMSE.

=============================================================================
USAGE
=============================================================================

# Single-sample inference (deterministic-style, analogous to UNet inference)
python infer_fm_3views.py \\
    --checkpoint scripts/runs/fm_3views/view0/checkpoint_best.pt \\
    --test_prefix scripts/tensor/3views_4d/test/global3v \\
    --test_split  test \\
    --out_dir     scripts/runs/fm_3views/view0/infer_test \\
    --n_steps     100

# Multi-sample inference (5 samples per input, saves mean + std + all samples)
python infer_fm_3views.py \\
    --checkpoint scripts/runs/fm_3views/view0/checkpoint_best.pt \\
    --test_prefix scripts/tensor/3views_4d/test/global3v \\
    --test_split  test \\
    --out_dir     scripts/runs/fm_3views/view0/infer_test_multi \\
    --n_steps     100 \\
    --n_samples   5

=============================================================================
OUTPUT FILES
=============================================================================

  pred_Y_img_test.npy         (N, C_sel, H, W)  — single sample or mean of samples
  test_metrics.json           per-channel MAE/RMSE
  pred_Y_std_test.npy         (N, C_sel, H, W)  — std across samples (if n_samples > 1)
  pred_Y_all_samples_test.npy (n_samples, N, C_sel, H, W) — all samples (if n_samples > 1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math


# ---------------------------------------------------------------------------
# Channel category sets (must match train_fm_3views.py)
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


# ---------------------------------------------------------------------------
# Dataset (X-only, with normalization)
# ---------------------------------------------------------------------------

class ViewXDataset(Dataset):
    def __init__(self, x_path: Path, x_mean: Optional[np.ndarray], x_std: Optional[np.ndarray]):
        self.X      = np.load(x_path, mmap_mode="r")
        self.x_mean = x_mean
        self.x_std  = x_std
        if self.X.ndim != 4:
            raise ValueError("Expected X to be 4D: (N,C,H,W).")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        m = x[0:1]
        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x  = (x - xm) / xs
        return x, m


# ---------------------------------------------------------------------------
# Model components (must match train_fm_3views.py exactly)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1).float()
        angles = t[:, None] * self.freq[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.mlp(emb)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.act   = nn.ReLU(inplace=True)
        self.film  = nn.Linear(t_dim, c_out * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        film_params = self.film(t_emb)
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + scale) + shift


class ConditioningEncoder(nn.Module):
    def __init__(self, c_in: int, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(c_in, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        return f1, f2, f3


class VelocityUNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.c_in  = c_in
        self.c_out = c_out
        self.time_embed = SinusoidalTimeEmbedding(t_dim)
        self.cond_encoder = ConditioningEncoder(c_in, base=base)

        self.enc1       = ConvBlock(c_in + c_out, base)
        self.enc2       = ConvBlock(base,         base * 2)
        self.enc3       = ConvBlock(base * 2,     base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4,     base * 8)

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = FiLMConvBlock(base * 4 + base * 4 + base * 4, base * 4, t_dim)
        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = FiLMConvBlock(base * 2 + base * 2 + base * 2, base * 2, t_dim)
        self.up1  = nn.Conv2d(base * 2, base, 1)
        self.dec1 = FiLMConvBlock(base + base + base, base, t_dim)
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _match_size(x, target_hw):
        th, tw = target_hw
        h, w = x.shape[-2], x.shape[-1]
        if h == th and w == tw:
            return x
        if h >= th and w >= tw:
            dh, dw = h - th, w - tw
            return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(self, x_cond, y_t, t):
        t_emb = self.time_embed(t)
        c1, c2, c3 = self.cond_encoder(x_cond)

        inp = torch.cat([x_cond, y_t], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        u3 = self.up3(F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, self._match_size(e3, u3.shape[-2:]),
                                       self._match_size(c3, u3.shape[-2:])], 1), t_emb)

        u2 = self.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, self._match_size(e2, u2.shape[-2:]),
                                       self._match_size(c2, u2.shape[-2:])], 1), t_emb)

        u1 = self.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, self._match_size(e1, u1.shape[-2:]),
                                       self._match_size(c1, u1.shape[-2:])], 1), t_emb)

        return self.out(d1)


# ---------------------------------------------------------------------------
# Inverse transform (transformed space → physical units)
# ---------------------------------------------------------------------------

def inverse_transform_y(
    y_t:             np.ndarray,
    y_indices:       Sequence[int],
    pos_channels:    Sequence[int],
    signed_channels: Sequence[int],
    eps:             float,
    s_c:             np.ndarray,
) -> np.ndarray:
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)
    out        = np.empty_like(y_t, dtype=np.float32)

    for j, orig_c in enumerate(y_indices):
        if int(orig_c) in pos_set:
            out[:, j] = (10.0 ** y_t[:, j] - eps).astype(np.float32)
        elif int(orig_c) in signed_set:
            out[:, j] = (float(s_c[j]) * np.sinh(y_t[:, j])).astype(np.float32)
        else:
            raise ValueError(f"Channel {orig_c} not categorised as POS or SIGNED.")

    return out


# ---------------------------------------------------------------------------
# Euler ODE integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model: VelocityUNet,
    x_cond: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int = 100,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    B = x_cond.shape[0]
    c_out = model.c_out
    H, W = x_cond.shape[-2], x_cond.shape[-1]

    y = torch.randn(B, c_out, H, W, device=device) * mask
    dt = 1.0 / n_steps

    model.eval()
    for step in range(n_steps):
        t_val = step / n_steps
        t = torch.full((B,), t_val, device=device)
        v = model(x_cond, y, t)
        y = y + v * dt
        y = y * mask

    return y


# ---------------------------------------------------------------------------
# Masked metrics (NumPy, physical space)
# ---------------------------------------------------------------------------

def masked_mae_per_channel_np(pred, y_true, mask, eps=1e-8):
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred - y_true) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(pred, y_true, mask, eps=1e-8):
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred - y_true) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer with a Flow Matching checkpoint from train_fm_3views.py."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_prefix", required=True)
    ap.add_argument("--test_split",  default="test")
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--n_steps",     type=int, default=100,
                    help="Number of Euler integration steps.")
    ap.add_argument("--n_samples",   type=int, default=1,
                    help="Number of samples per input (>1 enables uncertainty estimation).")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    view_tag         = str(ckpt["view_tag"])
    c_in             = int(ckpt["c_in"])
    c_out            = int(ckpt["c_out"])
    base             = int(ckpt.get("base", 64))
    t_dim            = int(ckpt.get("t_dim", 128))
    y_indices        = [int(c) for c in ckpt["y_indices"]]
    pos_channels     = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels  = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))
    eps              = float(ckpt.get("eps", 1e-5))

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32) if ckpt.get("x_mean") is not None else None
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32) if ckpt.get("x_std")  is not None else None
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)

    # ---- resolve tensor paths ----
    test_x_path = Path(f"{args.test_prefix}_{view_tag}_X_img_{args.test_split}.npy")
    test_y_path = Path(f"{args.test_prefix}_{view_tag}_Y_img_{args.test_split}.npy")

    for p in (test_x_path, test_y_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Flow Matching Inference: {view_tag} ===")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Test X     : {test_x_path}")
    print(f"  c_in={c_in}  c_out={c_out}  y_indices={y_indices}")
    print(f"  n_steps={args.n_steps}  n_samples={args.n_samples}")

    # ---- model ----
    device = torch.device(args.device)
    model  = VelocityUNet(c_in=c_in, c_out=c_out, base=base, t_dim=t_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- data loader ----
    ds = ViewXDataset(test_x_path, x_mean=x_mean, x_std=x_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=(device.type == "cuda"))

    N, _, H, W = np.load(test_x_path, mmap_mode="r").shape

    # ---- generate samples ----
    all_samples_norm = []

    for s in range(args.n_samples):
        print(f"  Generating sample {s+1}/{args.n_samples} ...")
        preds_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
        masks      = np.zeros((N, 1,    H, W), dtype=np.float32)

        idx0 = 0
        for x, m in dl:
            b = int(x.shape[0])
            x, m = x.to(device), m.to(device)
            pred = euler_sample(model, x, m, n_steps=args.n_steps, device=device)
            preds_norm[idx0:idx0 + b] = pred.cpu().numpy()
            masks[idx0:idx0 + b]      = m.cpu().numpy()
            idx0 += b

        all_samples_norm.append(preds_norm)

    # ---- denormalize + inverse transform ----
    ym = y_mean.reshape(1, -1, 1, 1)
    ys = y_std.reshape( 1, -1, 1, 1)

    all_samples_phys = []
    for preds_norm in all_samples_norm:
        preds_t = preds_norm * ys + ym
        preds_phys = inverse_transform_y(
            preds_t, y_indices=y_indices,
            pos_channels=pos_channels, signed_channels=signed_channels,
            eps=eps, s_c=s_c,
        )
        preds_phys *= masks
        all_samples_phys.append(preds_phys)

    # ---- aggregate ----
    if args.n_samples == 1:
        final_pred = all_samples_phys[0]
    else:
        stacked = np.stack(all_samples_phys, axis=0)   # (n_samples, N, C, H, W)
        final_pred = stacked.mean(axis=0)
        pred_std   = stacked.std(axis=0)

        # Save uncertainty and all samples
        np.save(out_dir / "pred_Y_std_test.npy", pred_std)
        np.save(out_dir / "pred_Y_all_samples_test.npy", stacked)
        print(f"  Saved std:         {out_dir / 'pred_Y_std_test.npy'}  shape={pred_std.shape}")
        print(f"  Saved all samples: {out_dir / 'pred_Y_all_samples_test.npy'}  shape={stacked.shape}")

    # ---- save predictions ----
    pred_path = out_dir / "pred_Y_img_test.npy"
    np.save(pred_path, final_pred)
    print(f"  Saved predictions: {pred_path}  shape={final_pred.shape}")

    # ---- evaluate against ground truth ----
    Y_full = np.load(test_y_path, mmap_mode="r")
    Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

    mae_c  = masked_mae_per_channel_np(final_pred, Y_sel, masks)
    rmse_c = masked_rmse_per_channel_np(final_pred, Y_sel, masks)

    metrics = {
        "checkpoint":        str(ckpt_path),
        "view":              ckpt["view"],
        "view_tag":          view_tag,
        "arch":              "flow_matching_unet",
        "base":              base,
        "t_dim":             t_dim,
        "c_in":              c_in,
        "c_out":             c_out,
        "y_indices":         y_indices,
        "n_steps":           args.n_steps,
        "n_samples":         args.n_samples,
        "mae_avg":           float(np.mean(mae_c)),
        "rmse_avg":          float(np.mean(rmse_c)),
        "mae_per_channel":   mae_c.tolist(),
        "rmse_per_channel":  rmse_c.tolist(),
    }

    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Test MAE avg  : {metrics['mae_avg']:.6g}")
    print(f"  Test RMSE avg : {metrics['rmse_avg']:.6g}")
    print(f"  Metrics saved : {metrics_path}")


if __name__ == "__main__":
    main()
