#!/usr/bin/env python3
"""
infer_fm_fusvel_v2.py
─────────────────────────────────────────────────────────────────────────────
Inference for the VelocityUNet OT-CFM model trained by
train_fm_fusvel_full_v2.py.

Key difference from v1: this uses ODE integration from noise → prediction
instead of a single forward pass with t=1.

Procedure
─────────
1. Load checkpoint (normalization stats, model weights, config).
2. For each test sample and each view:
   a. Normalize X using stored x_mean / x_std.
   b. Sample noise y_0 ~ N(0, I) * mask.
   c. Integrate the learned velocity field from t=0 to t=1 using Euler:
        y_{t+dt} = y_t + v_θ(x, y_t, t, view) * dt
   d. The result y_1 is in normalized transformed space.
   e. Denormalize: y_t = y_1 * y_std + y_mean  → transformed space.
   f. Inverse transform: physical units (10^y for POS, s_c*sinh(y) for SIGNED).
3. Optionally generate multiple samples (--n_samples) for uncertainty.
4. Save per-view predictions and combined metrics.

Usage
─────
python infer_fm_fusvel_v2.py \\
    --checkpoint scripts/runs/fusvel_full_v2/checkpoint_best.pt \\
    --test_prefix scripts/tensor/3views_4d/test/global3v \\
    --test_split  test \\
    --out_dir     scripts/runs/fusvel_full_v2/infer_test \\
    --batch_size  32 \\
    --ode_steps   50 \\
    --n_samples   1
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

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x_raw = np.array(self.X[idx], dtype=np.float32)
        mask  = x_raw[0:1].copy()
        x_norm = (x_raw - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        return torch.from_numpy(x_norm), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Model components  (must match training script exactly)
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
            nn.Conv2d(ci, co, 3, padding=1),
            nn.GroupNorm(min(8, co), co),
            nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1),
            nn.GroupNorm(min(8, co), co),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, ci, co, t_dim):
        super().__init__()
        self.c1   = nn.Conv2d(ci, co, 3, padding=1)
        self.gn1  = nn.GroupNorm(min(8, co), co)
        self.c2   = nn.Conv2d(co, co, 3, padding=1)
        self.gn2  = nn.GroupNorm(min(8, co), co)
        self.act  = nn.GELU()
        self.film = nn.Linear(t_dim, co * 2)

    def forward(self, x, t_emb):
        h = self.act(self.gn1(self.c1(x)))
        h = self.gn2(self.c2(h))
        s, sh = self.film(t_emb).chunk(2, -1)
        return self.act(h * (1 + s[..., None, None]) + sh[..., None, None])


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
    def __init__(self, c_in: int, c_out: int, base: int = 32,
                 t_dim: int = 128, n_views: int = 3):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out

        self.view_emb_dim = 4
        self.view_emb = nn.Embedding(n_views, self.view_emb_dim)

        self.time_emb = SinusoidalTimeEmb(t_dim)
        self.cond_enc = CondEncoder(c_in + self.view_emb_dim, base)

        self.enc1       = ConvBlock(c_in + self.view_emb_dim + c_out, base)
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

    def forward(self, x_cond, y_t, t, view_id):
        te = self.time_emb(t)

        B, _, H, W = x_cond.shape
        v_emb = self.view_emb(view_id)
        v_emb_spatial = v_emb[:, :, None, None].expand(B, -1, H, W)
        x_with_view = torch.cat([x_cond, v_emb_spatial], dim=1)

        c1, c2, c3 = self.cond_enc(x_with_view)

        inp = torch.cat([x_with_view, y_t], 1)
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
# ODE integration (Euler method)
# ─────────────────────────────────────────────────────────────────────────────

def euler_integrate(
    model: VelocityUNet,
    x_cond: torch.Tensor,       # (B, c_in, H, W)
    mask: torch.Tensor,          # (B, 1, H, W)
    view_id: torch.Tensor,       # (B,)
    c_out: int,
    n_steps: int = 50,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Integrate the learned velocity field from t=0 to t=1 using Euler steps.
    Returns y_1 in normalised transformed space.
    """
    B, _, H, W = x_cond.shape
    dt = 1.0 / n_steps

    # Start from noise, masked to active region
    y_t = torch.randn(B, c_out, H, W, device=device) * mask

    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=device)
        v = model(x_cond, y_t, t, view_id)
        y_t = y_t + v * dt
        # Re-mask after each step to prevent drift in inactive regions
        y_t = y_t * mask

    return y_t


# ─────────────────────────────────────────────────────────────────────────────
# Inverse transform  (transformed space → physical units)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LOG10  = 37.0
MAX_SINH   = 85.0


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
            clamped = np.clip(y_t[:, j].astype(np.float64), -MAX_LOG10, MAX_LOG10)
            out[:, j] = (10.0 ** clamped - eps).astype(np.float32)
        elif int(orig_c) in signed_set:
            clamped = np.clip(y_t[:, j].astype(np.float64), -MAX_SINH, MAX_SINH)
            out[:, j] = (float(s_c[j]) * np.sinh(clamped)).astype(np.float32)
        else:
            raise ValueError(f"Channel {orig_c} not categorised.")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Masked metrics  (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae_per_channel_np(pred, y_true, mask, eps=1e-8):
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred.astype(np.float64) - y_true.astype(np.float64)) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(pred, y_true, mask, eps=1e-8):
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred.astype(np.float64) - y_true.astype(np.float64)) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


def masked_log_mae_per_channel_np(
    pred, y_true, mask,
    y_indices, pos_channels, signed_channels,
    eps_log=1e-3, eps_den=1e-8,
):
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
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps_den)
    p64 = pred.astype(np.float64)
    t64 = y_true.astype(np.float64)
    ae  = np.abs(p64 - t64)
    ref = np.maximum(np.abs(t64), rel_floor)
    num = ((ae / ref) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> float:
    f = float(v)
    if np.isfinite(f):
        return f
    if np.isnan(f):
        return None
    return 1e38 if f > 0 else -1e38


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer with a VelocityUNet OT-CFM checkpoint."
    )
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--test_prefix",  required=True)
    ap.add_argument("--test_split",   default="test")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--batch_size",   type=int, default=32)
    ap.add_argument("--device",       default=None)
    ap.add_argument("--ode_steps",    type=int, default=50,
                    help="Number of Euler integration steps (more = better quality)")
    ap.add_argument("--n_samples",    type=int, default=1,
                    help="Number of samples per input (>1 for uncertainty)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)

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

    method = ckpt.get("method", "OT-CFM")

    print(f"\n{'='*70}")
    print(f"  VelocityUNet {method} Inference")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  c_in={c_in}  c_out={c_out}  base={base}  t_dim={t_dim}")
    print(f"  ODE steps  : {args.ode_steps}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  Device     : {device}")
    print(f"{'='*70}\n")

    # ── Build model and load weights ─────────────────────────────────────────
    model = VelocityUNet(c_in=c_in, c_out=c_out, base=base, t_dim=t_dim, n_views=3).to(device)
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

        ds = ViewXDataset(test_x_path, x_mean=x_mean, x_std=x_std)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

        N = len(ds)
        X_peek = np.load(test_x_path, mmap_mode="r")
        _, _, H, W = X_peek.shape

        # Accumulate predictions across samples
        all_preds_phys = []

        for sample_idx in range(args.n_samples):
            preds_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
            masks      = np.zeros((N, 1,    H, W), dtype=np.float32)
            idx0 = 0

            with torch.no_grad():
                for x, m in dl:
                    b = x.shape[0]
                    x, m = x.to(device), m.to(device)
                    view_ids = torch.full((b,), v, dtype=torch.long, device=device)

                    # ODE integration from noise to prediction
                    pred = euler_integrate(
                        model, x, m, view_ids,
                        c_out=c_out, n_steps=args.ode_steps, device=device,
                    )

                    preds_norm[idx0:idx0 + b] = pred.cpu().numpy()
                    masks[idx0:idx0 + b]      = m.cpu().numpy()
                    idx0 += b

            # denormalize → transformed space
            ym = y_mean.reshape(1, -1, 1, 1)
            ys = y_std.reshape( 1, -1, 1, 1)
            preds_t = preds_norm * ys + ym

            # inverse transform → physical units
            preds_phys = inverse_transform_y(
                preds_t, y_indices=y_indices,
                pos_channels=pos_channels,
                signed_channels=signed_channels,
                eps=eps, s_c=s_c,
            )
            preds_phys *= masks
            all_preds_phys.append(preds_phys)

        # Average across samples (or just use the single sample)
        if args.n_samples > 1:
            stacked = np.stack(all_preds_phys, axis=0)  # (n_samples, N, C, H, W)
            preds_phys_mean = stacked.mean(axis=0)
            preds_phys_std  = stacked.std(axis=0)
            # Save std for uncertainty visualization
            np.save(out_dir / f"pred_Y_std_{args.test_split}_{view_tag}.npy", preds_phys_std)
            preds_phys = preds_phys_mean
        else:
            preds_phys = all_preds_phys[0]

        # save predictions
        pred_path = out_dir / f"pred_Y_img_{args.test_split}_{view_tag}.npy"
        np.save(pred_path, preds_phys)
        print(f"    Saved: {pred_path}  shape={preds_phys.shape}")

        # ── Evaluate ─────────────────────────────────────────────────────────
        Y_full = np.load(test_y_path, mmap_mode="r")
        Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

        mae_c  = masked_mae_per_channel_np(preds_phys, Y_sel, masks)
        rmse_c = masked_rmse_per_channel_np(preds_phys, Y_sel, masks)
        log_mae_c = masked_log_mae_per_channel_np(
            preds_phys, Y_sel, masks,
            y_indices=y_indices,
            pos_channels=pos_channels,
            signed_channels=signed_channels,
            eps_log=eps,
        )
        mre_c = masked_relative_error_per_channel_np(preds_phys, Y_sel, masks)

        print(f"    Physical MAE_avg  = {float(np.mean(mae_c)):.6g}")
        print(f"    Physical RMSE_avg = {float(np.mean(rmse_c)):.6g}")
        print(f"    Log-space MAE_avg = {float(np.mean(log_mae_c)):.4f}")
        print(f"    Relative err avg  = {float(np.mean(mre_c)):.4f}")

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
        }

    # ── Save combined metrics ────────────────────────────────────────────────
    all_mae      = np.mean([m["mae_avg"]     for m in all_view_metrics.values()])
    all_rmse     = np.mean([m["rmse_avg"]    for m in all_view_metrics.values()])
    all_log_mae  = np.mean([m["log_mae_avg"] for m in all_view_metrics.values()])
    all_mre      = np.mean([m["mre_avg"]     for m in all_view_metrics.values()])

    metrics = {
        "checkpoint":         str(args.checkpoint),
        "arch":               "velocity_unet_otcfm",
        "method":             "OT-CFM",
        "ode_steps":          args.ode_steps,
        "n_samples":          args.n_samples,
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
