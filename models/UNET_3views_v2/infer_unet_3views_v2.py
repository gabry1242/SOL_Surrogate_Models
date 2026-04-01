#!/usr/bin/env python3
"""
infer_unet_3views_v2.py

Inference with a Boundary-Aware UNet checkpoint from train_unet_3views_v2.py.

Key differences from infer_unet_3views.py:
  1. Loads BoundaryAwareUNet with boundary specs from the checkpoint.
  2. Uses per-category eps for the inverse transform (te/ti: 1e-2, na: 1e8).
  3. Clamps transformed-space predictions to [train_min - margin, train_max + margin]
     before applying the inverse transform, preventing 10^25-type explosions
     on density channels.

Usage:
  python infer_unet_3views_v2.py \\
      --checkpoint scripts/runs/unet_3views_v2/view0/checkpoint_best.pt \\
      --test_prefix scripts/tensor/3views_4d/test/global3v \\
      --test_split test \\
      --layout_npz scripts/tensor/3views_4d/test/global3v_layout_map_3views.npz \\
      --out_dir scripts/runs/unet_3views_v2/view0/infer_test \\
      --batch_size 32

Output files (under --out_dir):
  pred_Y_img_test.npy   (N, C_sel, H, W) float32 in physical units
  test_metrics.json     per-channel MAE/RMSE + averages
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


# ---------------------------------------------------------------------------
# Channel definitions (must match train_unet_3views_v2.py)
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


# ---------------------------------------------------------------------------
# Dataset (X-only, with normalisation)
# ---------------------------------------------------------------------------

class ViewXDataset(Dataset):
    def __init__(self, x_path: Path, x_mean: Optional[np.ndarray],
                 x_std: Optional[np.ndarray]):
        self.X      = np.load(x_path, mmap_mode="r")
        self.x_mean = x_mean
        self.x_std  = x_std

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
# Model — BoundaryAwareUNet (identical to train_unet_3views_v2.py)
# ---------------------------------------------------------------------------

class BoundaryAwareConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int,
                 boundary_specs: List[Tuple[int, int, int, int, int, int]]):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in,  c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.boundary_specs = boundary_specs

    def _apply_boundary_padding(self, x: torch.Tensor) -> torch.Tensor:
        for src_cs, src_ce, dst_cs, dst_ce, rs, re in self.boundary_specs:
            x[:, :, rs:re, dst_cs:dst_ce] = x[:, :, rs:re, src_cs:src_ce].clone()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self._apply_boundary_padding(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self._apply_boundary_padding(x)
        x = self.relu(x)
        return x


class BoundaryAwareUNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32,
                 boundary_specs_per_level: Optional[Dict[int, list]] = None):
        super().__init__()
        specs = boundary_specs_per_level or {}

        self.enc1       = BoundaryAwareConvBlock(c_in,      base,      specs.get(0, []))
        self.enc2       = BoundaryAwareConvBlock(base,      base * 2,  specs.get(1, []))
        self.enc3       = BoundaryAwareConvBlock(base * 2,  base * 4,  specs.get(2, []))
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = BoundaryAwareConvBlock(base * 4,  base * 8,  specs.get(3, []))

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = BoundaryAwareConvBlock(base * 8, base * 4, specs.get(2, []))
        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = BoundaryAwareConvBlock(base * 4, base * 2, specs.get(1, []))
        self.up1  = nn.Conv2d(base * 2, base,     1)
        self.dec1 = BoundaryAwareConvBlock(base * 2, base,     specs.get(0, []))
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _crop(x: torch.Tensor, hw) -> torch.Tensor:
        th, tw = hw
        dh, dw = x.shape[-2] - th, x.shape[-1] - tw
        return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        u3 = self.up3(F.interpolate(b,  size=e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, self._crop(e3, u3.shape[-2:])], 1))

        u2 = self.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, self._crop(e2, u2.shape[-2:])], 1))

        u1 = self.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, self._crop(e1, u1.shape[-2:])], 1))

        return self.out(d1)


# ---------------------------------------------------------------------------
# Inverse transform (transformed space → physical units) with clamping
# ---------------------------------------------------------------------------

def inverse_transform_y(
    y_t:             np.ndarray,          # (N, C_sel, H, W) normalised → denorm first
    y_indices:       Sequence[int],
    pos_channels:    Sequence[int],
    signed_channels: Sequence[int],
    eps_per_channel: np.ndarray,          # (C_sel,) per-channel eps
    s_c:             np.ndarray,          # (C_sel,)
    y_min_t:         np.ndarray,          # (C_sel,) training-set min in transformed space
    y_max_t:         np.ndarray,          # (C_sel,) training-set max in transformed space
    clamp_margin:    float = 1.0,         # margin beyond training range before clamp
) -> np.ndarray:
    """
    Inverse-transform predictions from transformed space to physical units.

    For positive channels (log10):
        t_clamped = clamp(t, min_t - margin, max_t + margin)
        y = 10^t_clamped - eps          (then clamp to >= 0)

    For signed channels (asinh):
        t_clamped = clamp(t, min_t - margin, max_t + margin)
        y = s_c * sinh(t_clamped)

    The clamping prevents catastrophic inverse-transform explosions (e.g.,
    10^25 when the model predicts t=25 on a density channel whose training
    range is [10, 21]).
    """
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)
    out        = np.empty_like(y_t, dtype=np.float32)

    for j, orig_c in enumerate(y_indices):
        lo = float(y_min_t[j]) - clamp_margin
        hi = float(y_max_t[j]) + clamp_margin
        t_clamped = np.clip(y_t[:, j], lo, hi)

        if int(orig_c) in pos_set:
            eps_c = float(eps_per_channel[j])
            out[:, j] = np.maximum(10.0 ** t_clamped - eps_c, 0.0).astype(np.float32)
        elif int(orig_c) in signed_set:
            out[:, j] = (float(s_c[j]) * np.sinh(t_clamped)).astype(np.float32)
        else:
            raise ValueError(f"Channel {orig_c} not categorised.")

    return out


# ---------------------------------------------------------------------------
# Masked metrics (NumPy, physical space)
# ---------------------------------------------------------------------------

def masked_mae_per_channel_np(
    pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred - y_true) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)


def masked_rmse_per_channel_np(
    pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    m   = np.broadcast_to(mask.astype(np.float32), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred - y_true) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer with a Boundary-Aware UNet checkpoint."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_prefix", required=True)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--layout_npz", required=True,
                    help="Path to *_layout_map_3views.npz (needed only if checkpoint "
                         "does not contain boundary_specs).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--clamp_margin", type=float, default=1.0,
                    help="Margin beyond training transformed range before clamping "
                         "(default 1.0 — i.e. allow ~1 order of magnitude beyond training).")
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
    base             = int(ckpt.get("base", 32))
    y_indices        = [int(c) for c in ckpt["y_indices"]]
    pos_channels     = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels  = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))

    eps_per_channel  = np.asarray(ckpt["eps_per_channel"], dtype=np.float32)

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32) if ckpt.get("x_mean") is not None else None
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32) if ckpt.get("x_std")  is not None else None
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)
    y_min_t = np.asarray(ckpt["y_min_t"], dtype=np.float32)
    y_max_t = np.asarray(ckpt["y_max_t"], dtype=np.float32)

    # ---- reconstruct boundary specs ----
    bspecs_serial = ckpt["boundary_specs"]
    boundary_specs_per_level: Dict[int, list] = {}
    for k_str, v in bspecs_serial.items():
        boundary_specs_per_level[int(k_str)] = [tuple(s) for s in v]

    # ---- resolve tensor paths ----
    test_x_path = Path(f"{args.test_prefix}_{view_tag}_X_img_{args.test_split}.npy")
    test_y_path = Path(f"{args.test_prefix}_{view_tag}_Y_img_{args.test_split}.npy")

    for p in (test_x_path, test_y_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Inference: {view_tag} (Boundary-Aware UNet) ===")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Test X     : {test_x_path}")
    print(f"  c_in={c_in}  c_out={c_out}  y_indices={y_indices}")
    print(f"  Clamp margin: {args.clamp_margin}")
    for lvl, sp in boundary_specs_per_level.items():
        print(f"  Boundary specs level {lvl}: {len(sp)} ops")

    # ---- model ----
    device = torch.device(args.device)
    model  = BoundaryAwareUNet(
        c_in=c_in, c_out=c_out, base=base,
        boundary_specs_per_level=boundary_specs_per_level,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- data loader ----
    ds = ViewXDataset(test_x_path, x_mean=x_mean, x_std=x_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=(device.type == "cuda"))

    N, _, H, W = np.load(test_x_path, mmap_mode="r").shape
    preds_t_norm = np.zeros((N, c_out, H, W), dtype=np.float32)
    masks        = np.zeros((N, 1,    H, W), dtype=np.float32)

    idx0 = 0
    for x, m in dl:
        b  = int(x.shape[0])
        x  = x.to(device)
        m  = m.to(device)
        with torch.no_grad():
            pred = model(x) * m
        preds_t_norm[idx0:idx0 + b] = pred.cpu().numpy()
        masks[idx0:idx0 + b]        = m.cpu().numpy()
        idx0 += b

    # ---- denormalise → transformed space ----
    ym     = y_mean.reshape(1, -1, 1, 1)
    ys     = y_std.reshape( 1, -1, 1, 1)
    preds_t = preds_t_norm * ys + ym

    # ---- inverse transform → physical units (with clamping) ----
    preds_phys = inverse_transform_y(
        preds_t, y_indices=y_indices,
        pos_channels=pos_channels, signed_channels=signed_channels,
        eps_per_channel=eps_per_channel, s_c=s_c,
        y_min_t=y_min_t, y_max_t=y_max_t,
        clamp_margin=args.clamp_margin,
    )
    preds_phys *= masks   # enforce gaps = 0

    # ---- save predictions ----
    pred_path = out_dir / "pred_Y_img_test.npy"
    np.save(pred_path, preds_phys)
    print(f"Saved predictions: {pred_path}  shape={preds_phys.shape}")

    # ---- evaluate against ground truth ----
    Y_full = np.load(test_y_path, mmap_mode="r")
    Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

    mae_c  = masked_mae_per_channel_np(preds_phys, Y_sel, masks)
    rmse_c = masked_rmse_per_channel_np(preds_phys, Y_sel, masks)

    # ---- channel names for readability ----
    ch_names = []
    for c in y_indices:
        if c == 0:   ch_names.append("te")
        elif c == 1: ch_names.append("ti")
        elif 2 <= c <= 11: ch_names.append(f"na{c-2}")
        elif 12 <= c <= 21: ch_names.append(f"ua{c-12}")
        else: ch_names.append(f"ch{c}")

    metrics = {
        "checkpoint":        str(ckpt_path),
        "view":              ckpt["view"],
        "view_tag":          view_tag,
        "arch":              "boundary_aware_unet",
        "base":              base,
        "c_in":              c_in,
        "c_out":             c_out,
        "y_indices":         y_indices,
        "clamp_margin":      args.clamp_margin,
        "mae_avg":           float(np.mean(mae_c)),
        "rmse_avg":          float(np.mean(rmse_c)),
        "mae_per_channel":   {name: float(v) for name, v in zip(ch_names, mae_c)},
        "rmse_per_channel":  {name: float(v) for name, v in zip(ch_names, rmse_c)},
    }

    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTest MAE avg  : {metrics['mae_avg']:.6g}")
    print(f"Test RMSE avg : {metrics['rmse_avg']:.6g}")
    print(f"\nPer-channel MAE:")
    for name, v in zip(ch_names, mae_c):
        print(f"  {name:8s}: {v:.6g}")
    print(f"\nPer-channel RMSE:")
    for name, v in zip(ch_names, rmse_c):
        print(f"  {name:8s}: {v:.6g}")
    print(f"\nMetrics saved : {metrics_path}")


if __name__ == "__main__":
    main()
