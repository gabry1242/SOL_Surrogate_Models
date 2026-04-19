#!/usr/bin/env python3
"""
infer_fm_fusvel.py
─────────────────────────────────────────────────────────────────────────────
Inference for the VelocityUNet regression model trained by
train_fm_fusvel_full.py.

Procedure
─────────
1. Load checkpoint (normalization stats, model weights, config).
2. For each test sample and each view:
   a. Normalize X using stored x_mean / x_std.
   b. Run forward pass with y_t=zeros, t=ones (direct regression).
   c. Denormalize: y_phys = y_pred * y_std + y_mean  → transformed space.
   d. Inverse transform: physical units (10^y for POS, s_c*sinh(y) for SIGNED).
      — POS channels are clamped before exponentiation to prevent float overflow.
      — All inverse transforms are computed in float64 for precision.
3. Save per-view predictions and combined metrics.

Metrics reported
────────────────
  Physical MAE / RMSE  — standard absolute metrics in original physical units.
  Log-space MAE        — |log10(pred+eps) − log10(truth+eps)|, the metric the
                         model actually optimises.  A log-MAE of 0.3 means the
                         prediction is within a factor ~2 of truth.
  Relative error (MRE) — |pred − truth| / max(|truth|, floor), mean across
                         masked pixels.  Directly interpretable as fractional
                         error.

Usage
─────
python infer_fm_fusvel.py \
    --checkpoint scripts/runs/fusvel_full/checkpoint_best.pt \
    --test_prefix scripts/tensor/3views_4d/test/global3v \
    --test_split  test \
    --out_dir     scripts/runs/fusvel_full/infer_test \
    --batch_size  32

Output  (inside --out_dir)
───────
  pred_Y_img_test_view0.npy   (N, C_sel, H, W)  physical units
  pred_Y_img_test_view1.npy
  pred_Y_img_test_view2.npy
  test_metrics.json            per-view, per-channel MAE/RMSE/log-MAE/MRE
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Channel metadata (must match training script)
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
# Dataset  (X-only with normalisation)
# ─────────────────────────────────────────────────────────────────────────────

class ViewXDataset(Dataset):
    """Loads a single view's X tensor, returns normalised (x, mask) pairs."""

    def __init__(self, x_path: Path, x_mean: np.ndarray, x_std: np.ndarray):
        self.X      = np.load(x_path, mmap_mode="r")
        self.x_mean = x_mean
        self.x_std  = x_std
        if self.X.ndim != 4:
            raise ValueError(f"Expected 4D X tensor, got shape {self.X.shape}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x_raw = np.array(self.X[idx], dtype=np.float32)
        mask  = x_raw[0:1].copy()
        x_norm = (x_raw - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        return torch.from_numpy(x_norm), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Model components  (must be identical to training script)
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

    def forward(self, x_cond, y_t, t):
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
# Inverse transform  (transformed space → physical units)
# ─────────────────────────────────────────────────────────────────────────────
#
# FIXES applied:
#   1. POS channels: clamp to [-MAX_LOG10, MAX_LOG10] before 10^y to prevent
#      float overflow (float32 max ≈ 3.4e38 → log10 ≈ 38.5).
#   2. SIGNED channels: clamp before sinh to prevent overflow.
#   3. All arithmetic done in float64, cast to float32 only at the end.
#

MAX_LOG10  = 37.0    # safely below float32 overflow threshold
MAX_SINH   = 85.0    # sinh(85) ≈ 1.1e36, safely in float64 range


def inverse_transform_y(
    y_t:             np.ndarray,       # (N, C_sel, H, W) — in transformed space
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
            clamped = np.clip(y_t[:, j].astype(np.float64), -MAX_LOG10, MAX_LOG10)
            out[:, j] = (10.0 ** clamped - eps).astype(np.float32)
        elif int(orig_c) in signed_set:
            clamped = np.clip(y_t[:, j].astype(np.float64), -MAX_SINH, MAX_SINH)
            out[:, j] = (float(s_c[j]) * np.sinh(clamped)).astype(np.float32)
        else:
            raise ValueError(f"Channel {orig_c} not categorised as POS or SIGNED.")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Masked metrics  (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae_per_channel_np(pred, y_true, mask, eps=1e-8):
    """Physical-space mean absolute error per channel."""
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred.astype(np.float64) - y_true.astype(np.float64)) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(pred, y_true, mask, eps=1e-8):
    """Physical-space root mean squared error per channel."""
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred.astype(np.float64) - y_true.astype(np.float64)) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


def masked_log_mae_per_channel_np(
    pred, y_true, mask,
    y_indices, pos_channels, signed_channels,
    eps_log=1e-3, eps_den=1e-8,
):
    """
    Log-space MAE per channel.

    For POS channels:    |log10(pred + eps) - log10(truth + eps)|
    For SIGNED channels: |arcsinh(pred) - arcsinh(truth)|

    This is the metric the model actually learns in — a log-MAE of 0.1 means
    the prediction is within 10^0.1 ≈ 1.26× of truth (for POS channels).
    """
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)
    C_sel      = len(y_indices)
    result     = np.zeros(C_sel, dtype=np.float64)
    m          = np.broadcast_to(mask.astype(np.float64), pred.shape)

    for j, orig_c in enumerate(y_indices):
        den = max(float(m[:, j:j+1].sum()), eps_den)
        p64 = pred[:, j].astype(np.float64)
        t64 = y_true[:, j].astype(np.float64)

        if int(orig_c) in pos_set:
            log_p = np.log10(np.maximum(p64, 0.0) + eps_log)
            log_t = np.log10(np.maximum(t64, 0.0) + eps_log)
            ae    = np.abs(log_p - log_t) * m[:, j]
        elif int(orig_c) in signed_set:
            ae = np.abs(np.arcsinh(p64) - np.arcsinh(t64)) * m[:, j]
        else:
            ae = np.zeros_like(p64)

        result[j] = float(ae.sum() / den)

    return result


def masked_relative_error_per_channel_np(pred, y_true, mask, rel_floor=1e-8, eps_den=1e-8):
    """
    Mean relative error (MRE) per channel:
        mean( |pred - truth| / max(|truth|, floor) )
    Averaged over masked pixels.
    """
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps_den)
    p64 = pred.astype(np.float64)
    t64 = y_true.astype(np.float64)
    ae  = np.abs(p64 - t64)
    ref = np.maximum(np.abs(t64), rel_floor)
    num = ((ae / ref) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# JSON-safe float helper
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> float:
    """Convert to float, replacing inf/nan with JSON-safe values."""
    f = float(v)
    if np.isfinite(f):
        return f
    if np.isnan(f):
        return None  # JSON null
    return 1e38 if f > 0 else -1e38


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer with a VelocityUNet regression checkpoint."
    )
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--test_prefix",  required=True,
                    help="Tensor prefix for test, e.g. scripts/tensor/3views_4d/test/global3v")
    ap.add_argument("--test_split",   default="test")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--batch_size",   type=int, default=32)
    ap.add_argument("--device",       default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    c_in            = int(ckpt["c_in"])
    c_out           = int(ckpt["c_out"])
    base            = int(ckpt.get("base", 32))
    t_dim           = int(ckpt.get("t_dim", 128))
    y_indices       = [int(c) for c in ckpt["y_indices"]]
    pos_channels    = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))
    eps             = float(ckpt.get("eps", 1e-3))

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)

    print(f"\n{'='*70}")
    print(f"  VelocityUNet Regression Inference")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  c_in={c_in}  c_out={c_out}  base={base}  t_dim={t_dim}")
    print(f"  y_indices={y_indices[:8]}{'…' if c_out > 8 else ''}")
    print(f"  Device : {device}")
    print(f"{'='*70}\n")

    # ── Build model and load weights ─────────────────────────────────────────
    model = VelocityUNet(c_in=c_in, c_out=c_out, base=base, t_dim=t_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-view inference ───────────────────────────────────────────────────
    all_view_metrics = {}

    for v in range(3):
        view_tag = f"view{v}"
        test_x_path = Path(f"{args.test_prefix}_{view_tag}_X_img_{args.test_split}.npy")
        test_y_path = Path(f"{args.test_prefix}_{view_tag}_Y_img_{args.test_split}.npy")

        for p in (test_x_path, test_y_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        print(f"  ── {view_tag} ──")
        print(f"    X: {test_x_path.name}   Y: {test_y_path.name}")

        # data loader
        ds = ViewXDataset(test_x_path, x_mean=x_mean, x_std=x_std)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

        N = len(ds)
        X_peek = np.load(test_x_path, mmap_mode="r")
        _, _, H, W = X_peek.shape

        # run inference
        preds_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
        masks      = np.zeros((N, 1,    H, W), dtype=np.float32)

        idx0 = 0
        with torch.no_grad():
            for x, m in dl:
                b = x.shape[0]
                x, m = x.to(device), m.to(device)

                y_t = torch.zeros(b, c_out, H, W, device=device)
                t   = torch.ones(b, device=device)
                pred = model(x, y_t, t) * m

                preds_norm[idx0:idx0 + b] = pred.cpu().numpy()
                masks[idx0:idx0 + b]      = m.cpu().numpy()
                idx0 += b

        # denormalize → transformed space
        ym = y_mean.reshape(1, -1, 1, 1)
        ys = y_std.reshape( 1, -1, 1, 1)
        preds_t = preds_norm * ys + ym

        # count overflow pixels BEFORE clamping (diagnostic)
        pos_set = set(int(c) for c in pos_channels)
        n_clamped = 0
        for j, orig_c in enumerate(y_indices):
            if int(orig_c) in pos_set:
                n_clamped += int((preds_t[:, j] > MAX_LOG10).sum())
                n_clamped += int((preds_t[:, j] < -MAX_LOG10).sum())

        # inverse transform → physical units  (clamped, float64)
        preds_phys = inverse_transform_y(
            preds_t, y_indices=y_indices,
            pos_channels=pos_channels,
            signed_channels=signed_channels,
            eps=eps, s_c=s_c,
        )
        preds_phys *= masks

        # save predictions
        pred_path = out_dir / f"pred_Y_img_{args.test_split}_{view_tag}.npy"
        np.save(pred_path, preds_phys)
        print(f"    Saved predictions: {pred_path}  shape={preds_phys.shape}")

        # ── Evaluate against ground truth ────────────────────────────────────
        Y_full = np.load(test_y_path, mmap_mode="r")
        Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

        # physical-space metrics  (now overflow-safe thanks to clamping)
        mae_c  = masked_mae_per_channel_np(preds_phys, Y_sel, masks)
        rmse_c = masked_rmse_per_channel_np(preds_phys, Y_sel, masks)

        # log-space MAE  (what the model actually optimises)
        log_mae_c = masked_log_mae_per_channel_np(
            preds_phys, Y_sel, masks,
            y_indices=y_indices,
            pos_channels=pos_channels,
            signed_channels=signed_channels,
            eps_log=eps,
        )

        # mean relative error
        mre_c = masked_relative_error_per_channel_np(preds_phys, Y_sel, masks)

        # ── Console summary ──────────────────────────────────────────────────
        print(f"    Physical MAE_avg  = {float(np.mean(mae_c)):.6g}")
        print(f"    Physical RMSE_avg = {float(np.mean(rmse_c)):.6g}")
        print(f"    Log-space MAE_avg = {float(np.mean(log_mae_c)):.4f}")
        print(f"    Relative err avg  = {float(np.mean(mre_c)):.4f}")
        if n_clamped > 0:
            print(f"    ⚠ Clamped {n_clamped} pixels that would have overflowed")

        # ── Per-channel detail ───────────────────────────────────────────────
        print(f"    {'channel':<12s} {'phys_MAE':>12s} {'log_MAE':>10s} {'MRE':>10s}")
        for j, c in enumerate(y_indices):
            nm = CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"
            print(f"    {nm:<12s} {mae_c[j]:>12.4g} {log_mae_c[j]:>10.4f} {mre_c[j]:>10.4f}")

        all_view_metrics[view_tag] = {
            "mae_avg":              _safe_float(np.mean(mae_c)),
            "rmse_avg":             _safe_float(np.mean(rmse_c)),
            "log_mae_avg":          _safe_float(np.mean(log_mae_c)),
            "mre_avg":              _safe_float(np.mean(mre_c)),
            "mae_per_channel":      [_safe_float(v) for v in mae_c],
            "rmse_per_channel":     [_safe_float(v) for v in rmse_c],
            "log_mae_per_channel":  [_safe_float(v) for v in log_mae_c],
            "mre_per_channel":      [_safe_float(v) for v in mre_c],
            "n_clamped_pixels":     n_clamped,
        }

    # ── Save combined metrics ────────────────────────────────────────────────
    all_mae      = np.mean([m["mae_avg"]     for m in all_view_metrics.values()])
    all_rmse     = np.mean([m["rmse_avg"]    for m in all_view_metrics.values()])
    all_log_mae  = np.mean([m["log_mae_avg"] for m in all_view_metrics.values()])
    all_mre      = np.mean([m["mre_avg"]     for m in all_view_metrics.values()])

    metrics = {
        "checkpoint":         str(ckpt_path),
        "arch":               "velocity_unet_regression",
        "base":               base,
        "t_dim":              t_dim,
        "c_in":               c_in,
        "c_out":              c_out,
        "y_indices":          y_indices,
        "mae_avg_global":     _safe_float(all_mae),
        "rmse_avg_global":    _safe_float(all_rmse),
        "log_mae_avg_global": _safe_float(all_log_mae),
        "mre_avg_global":     _safe_float(all_mre),
        "per_view":           all_view_metrics,
    }

    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'─'*70}")
    print(f"  Global Physical MAE  = {all_mae:.6g}")
    print(f"  Global Physical RMSE = {all_rmse:.6g}")
    print(f"  Global Log-space MAE = {all_log_mae:.4f}")
    print(f"  Global Relative err  = {all_mre:.4f}")
    print(f"  Metrics saved: {metrics_path}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()
