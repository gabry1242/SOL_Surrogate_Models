#!/usr/bin/env python3
"""
viz_mesh_pred.py
─────────────────────────────────────────────────────────────────────────────
Plot reconstructed mesh predictions vs ground truth on the physical
tokamak cross-section.  Each cell is drawn as a color-coded quadrilateral
using crx/cry corner coordinates.

Layout per channel:
    [ Ground Truth ]  [ Prediction ]  [ Relative/Log Error ]
    
    Shared colorbar for GT and Pred, separate colorbar for error.
    Suptitle with error statistics (same style as viz_fm_fusvel.py).

Usage
─────
  python viz_mesh_pred.py \\
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \\
      --truth_dir   test/ \\
      --geom_dir    geometry/ \\
      --idx 0 --channels 0,1 \\
      --out_dir scripts/runs/my_run/viz_mesh

  # all 22 channels
  python viz_mesh_pred.py \\
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \\
      --truth_dir   test/ \\
      --geom_dir    geometry/ \\
      --idx 0 --channels all \\
      --out_dir scripts/runs/my_run/viz_mesh
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

J_PER_EV = 1.602176634e-19

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CHANNEL_NAMES = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)

POS_CHANNELS    = set(range(0, 12))   # Te, Ti, na_*
SIGNED_CHANNELS = set(range(12, 22))  # ua_*


def ch_name(c: int) -> str:
    return CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(truth_dir: Path) -> np.ndarray:
    """Load raw simulation data → (N, 22, 104, 50) in physical units (eV etc)."""
    te = np.load(truth_dir / "te_tmp.npy")
    ti = np.load(truth_dir / "ti_tmp.npy")
    na = np.load(truth_dir / "na_tmp.npy")
    ua = np.load(truth_dir / "ua_tmp.npy")
    N  = te.shape[0]
    full = np.zeros((N, 22, 104, 50), dtype=np.float64)
    full[:, 0]    = te / J_PER_EV
    full[:, 1]    = ti / J_PER_EV
    for s in range(10):
        full[:, 2 + s]  = na[:, :, :, s]
        full[:, 12 + s] = ua[:, :, :, s]
    return full.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh plotter
# ─────────────────────────────────────────────────────────────────────────────

def _build_polys(crx, cry, mask):
    """Pre-build vertex list and index arrays for valid cells."""
    verts = []
    valid_ix = []
    valid_iy = []
    for ix in range(104):
        for iy in range(50):
            if not mask[ix, iy]:
                continue
            corners = np.column_stack([
                crx[ix, iy, :], cry[ix, iy, :]
            ])  # (4, 2)
            verts.append(corners)
            valid_ix.append(ix)
            valid_iy.append(iy)
    return verts, np.array(valid_ix), np.array(valid_iy)


def _plot_mesh(ax, verts, values, cmap, vmin, vmax, title):
    """Draw color-coded quadrilateral mesh on ax."""
    pc = PolyCollection(
        verts, array=values, cmap=cmap,
        edgecolors="face", linewidths=0.1,
    )
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("R [m]", fontsize=8)
    ax.set_ylabel("Z [m]", fontsize=8)
    return pc


def _save_or_show(fig, path: Optional[Path]):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"    Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting logic
# ─────────────────────────────────────────────────────────────────────────────

def plot_channel(
    t_field: np.ndarray,       # (104, 50)  ground truth
    p_field: np.ndarray,       # (104, 50)  prediction
    crx: np.ndarray,           # (104, 50, 4)
    cry: np.ndarray,           # (104, 50, 4)
    orig_c: int,               # original channel index (for naming / error type)
    idx: int,                  # simulation index
    out_dir: Optional[Path],
    eps_log: float = 1e-3,
):
    """One figure per channel: Ground Truth | Prediction | Error."""

    # Mask: cells that have data in either truth or prediction
    active = (np.abs(t_field) > 0) | (np.abs(p_field) > 0)
    if not active.any():
        print(f"    Skipping {ch_name(orig_c)}: all zeros")
        return

    # Pre-build polygon vertices for valid cells
    verts, vix, viy = _build_polys(crx, cry, active)
    t_vals = t_field[vix, viy].astype(np.float64)
    p_vals = p_field[vix, viy].astype(np.float64)

    # ── Shared color range for GT and pred ───────────────────────────────
    vmin = min(t_vals.min(), p_vals.min())
    vmax = max(t_vals.max(), p_vals.max())
    cmap = "viridis"

    # ── Compute error image ──────────────────────────────────────────────
    if int(orig_c) in POS_CHANNELS:
        # Log-space error for positive quantities
        log_p = np.log10(np.maximum(p_vals, 0.0) + eps_log)
        log_t = np.log10(np.maximum(t_vals, 0.0) + eps_log)
        err_vals    = np.abs(log_p - log_t)
        err_label   = "Log₁₀ Error"
        err_cmap    = "inferno"
        mean_err    = float(err_vals.mean())
        # also compute percentage for title
        rel_pct     = np.abs(p_vals - t_vals) / np.maximum(np.abs(t_vals), 1e-30) * 100.0
        median_pct  = float(np.median(rel_pct))
        title_stats = f"log₁₀-MAE = {mean_err:.3f}  (median rel = {median_pct:.1f}%)"
    else:
        # Relative error (%) for signed quantities
        ref         = np.maximum(np.abs(t_vals), 1e-30)
        err_vals    = np.abs(p_vals - t_vals) / ref * 100.0
        err_label   = "Relative Error (%)"
        err_cmap    = "inferno"
        median_pct  = float(np.median(err_vals))
        title_stats = f"median relative error = {median_pct:.1f}%"

    # ── Figure: 3 panels ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    pc0 = _plot_mesh(axes[0], verts, t_vals, cmap, vmin, vmax, "Ground Truth")
    plt.colorbar(pc0, ax=axes[0], fraction=0.046, pad=0.04)

    pc1 = _plot_mesh(axes[1], verts, p_vals, cmap, vmin, vmax, "Prediction")
    plt.colorbar(pc1, ax=axes[1], fraction=0.046, pad=0.04)

    pc2 = _plot_mesh(axes[2], verts, err_vals, err_cmap, None, None, err_label)
    plt.colorbar(pc2, ax=axes[2], fraction=0.046, pad=0.04)

    name = ch_name(orig_c)
    fig.suptitle(
        f"Sample {idx} | {name} | {title_stats}",
        fontsize=12,
    )

    path = out_dir / f"mesh_idx{idx:04d}_{name}.png" if out_dir else None
    _save_or_show(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_channels(s: str, max_c: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(max_c))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Plot reconstructed mesh predictions vs ground truth "
                    "on the physical tokamak cross-section."
    )
    ap.add_argument("--pred_path", required=True,
                    help="Path to reconstructed prediction .npy (N, C, 104, 50)")
    ap.add_argument("--truth_dir", required=True,
                    help="Directory with te_tmp.npy, ti_tmp.npy, na_tmp.npy, ua_tmp.npy")
    ap.add_argument("--geom_dir", required=True,
                    help="Directory with crx.npy, cry.npy")
    ap.add_argument("--y_indices", default=None,
                    help="Original channel indices, e.g. '0,1,2,...,21'. "
                         "Default: assumes all 22 in order.")
    ap.add_argument("--idx", type=int, default=0,
                    help="Simulation index to visualize")
    ap.add_argument("--channels", default="0,1",
                    help="Which prediction channels to plot. 'all' or '0,1,12,13'")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for PNGs (prints to stdout if omitted)")
    ap.add_argument("--eps_log", type=float, default=1e-3,
                    help="Epsilon for log10 transform of positive channels")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    pred = np.load(args.pred_path)  # (N, C_sel, 104, 50)
    N, C_sel = pred.shape[0], pred.shape[1]
    print(f"\n  Predictions : {args.pred_path}  shape={pred.shape}")

    crx = np.load(Path(args.geom_dir) / "crx.npy")  # (104, 50, 4)
    cry = np.load(Path(args.geom_dir) / "cry.npy")
    print(f"  Geometry    : crx={crx.shape}  cry={cry.shape}")

    truth_full = load_ground_truth(Path(args.truth_dir))  # (N, 22, 104, 50)
    print(f"  Ground truth: {args.truth_dir}  shape={truth_full.shape}")

    # ── Channel mapping ──────────────────────────────────────────────────
    if args.y_indices is not None:
        y_indices = [int(x.strip()) for x in args.y_indices.split(",")]
    else:
        y_indices = list(range(min(C_sel, 22)))

    if len(y_indices) != C_sel:
        print(f"  Warning: y_indices length ({len(y_indices)}) != pred channels ({C_sel})")
        y_indices = list(range(C_sel))

    channels = parse_channels(args.channels, C_sel)
    print(f"  Plotting channels: {channels} → {[ch_name(y_indices[c]) for c in channels]}")
    print()

    # ── Plot each channel ────────────────────────────────────────────────
    for c in channels:
        if c >= C_sel:
            continue
        orig_c = y_indices[c]
        plot_channel(
            t_field  = truth_full[args.idx, orig_c],
            p_field  = pred[args.idx, c],
            crx      = crx,
            cry      = cry,
            orig_c   = orig_c,
            idx      = args.idx,
            out_dir  = out_dir,
            eps_log  = args.eps_log,
        )

    print("  Done.\n")


if __name__ == "__main__":
    main()
