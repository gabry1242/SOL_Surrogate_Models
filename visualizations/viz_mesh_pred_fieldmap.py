#!/usr/bin/env python3
"""
viz_mesh_pred_fieldmap.py
─────────────────────────────────────────────────────────────────────────────
Side-by-side Ground Truth vs Prediction on the physical tokamak
cross-section, using the same Polygon / PatchCollection rendering style
as make_field_maps.py (LogNorm for positive fields, symmetric linear
for velocities, robust quantile-based color limits).

Layout per channel:
    [ Ground Truth ]  [ Prediction ]  [ Error ]

    Shared colorbar for GT and Pred (LogNorm or symmetric linear).
    Separate colorbar for the error panel.

Usage
─────
  python viz_mesh_pred_fieldmap.py \
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \
      --truth_dir   test/ \
      --geom_dir    geometry/ \
      --idx 0 --channels all \
      --out_dir scripts/runs/my_run/viz_mesh_fieldmap

  # Specific channels only
  python viz_mesh_pred_fieldmap.py \
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \
      --truth_dir   test/ \
      --geom_dir    geometry/ \
      --idx 0 --channels 0,1,3,12 \
      --out_dir scripts/runs/my_run/viz_mesh_fieldmap
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
from matplotlib.collections import PatchCollection


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NX, NY = 104, 50
J_PER_EV = 1.602176634e-19

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]

CHANNEL_NAMES: List[str] = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)

POS_CHANNELS    = set(range(0, 12))   # Te, Ti, na_*
SIGNED_CHANNELS = set(range(12, 22))  # ua_*


def ch_name(c: int) -> str:
    return CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"


def ch_unit(c: int) -> str:
    if c < len(CHANNEL_NAMES):
        name = CHANNEL_NAMES[c]
        if name in ("Te", "Ti"):
            return "eV"
        if name.startswith("na_"):
            return "m⁻³"
        if name.startswith("ua_"):
            return "m/s"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: build polygon patches from crx / cry
# ─────────────────────────────────────────────────────────────────────────────

def build_cells(crx: np.ndarray, cry: np.ndarray):
    """
    Build a list of matplotlib Polygon patches, one per grid cell,
    using the same corner ordering as make_field_maps.py: [0, 1, 3, 2].

    Returns
    -------
    cells : list of Polygon  (length NX*NY, row-major order)
    xlim  : [xmin, xmax]
    ylim  : [ymin, ymax]
    """
    cells = []
    for ix in range(NX):
        for iy in range(NY):
            x = crx[ix, iy, :]
            y = cry[ix, iy, :]
            corners = np.array([
                [x[0], y[0]],
                [x[1], y[1]],
                [x[3], y[3]],
                [x[2], y[2]],
            ])
            cells.append(patches.Polygon(corners, closed=True))

    xlim = [float(np.min(crx)), float(np.max(crx))]
    ylim = [float(np.min(cry)), float(np.max(cry))]
    return cells, xlim, ylim


# ─────────────────────────────────────────────────────────────────────────────
# Robust color limits (same logic as make_field_maps.py)
# ─────────────────────────────────────────────────────────────────────────────

def robust_pos_limits(a: np.ndarray, qlo: float = 0.01, qhi: float = 0.99):
    """Quantile-based limits for positive (log-scale) fields."""
    a = a[np.isfinite(a)]
    a = a[a > 0]
    if a.size == 0:
        return 1e-30, 1.0
    return float(np.quantile(a, qlo)), float(np.quantile(a, qhi))


def robust_sym_limits(a: np.ndarray, q: float = 0.99):
    """Symmetric quantile-based limits for signed (velocity) fields."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    m = float(np.quantile(np.abs(a), q))
    m = max(m, 1e-12)
    return -m, m


# ─────────────────────────────────────────────────────────────────────────────
# Core field plotter (PatchCollection style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_field(
    ax,
    cells,
    xlim, ylim,
    z_flat: np.ndarray,
    title: str,
    norm,
    cmap: str = "viridis",
):
    """
    Draw color-coded polygons on *ax* using a pre-built PatchCollection.

    Parameters
    ----------
    cells : list of Polygon patches (length NX*NY)
    z_flat : 1-D array of values, same length as cells
    norm : matplotlib Normalize / LogNorm / TwoSlopeNorm
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]", fontsize=9)
    ax.set_ylabel("Z [m]", fontsize=9)
    ax.set_title(title, fontsize=10)

    # PatchCollection needs its own copy of the patches
    cell_copies = [copy.copy(p) for p in cells]
    pc = PatchCollection(cell_copies, antialiaseds=False, norm=norm, rasterized=True)
    pc.set_array(z_flat)
    pc.set_cmap(cmap)
    ax.add_collection(pc)
    return pc


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(truth_dir: Path) -> np.ndarray:
    """Load raw simulation data → (N, 22, 104, 50) in physical units."""
    te = np.load(truth_dir / "te_tmp.npy")
    ti = np.load(truth_dir / "ti_tmp.npy")
    na = np.load(truth_dir / "na_tmp.npy")
    ua = np.load(truth_dir / "ua_tmp.npy")
    N = te.shape[0]
    full = np.zeros((N, 22, NX, NY), dtype=np.float64)
    full[:, 0]  = te / J_PER_EV
    full[:, 1]  = ti / J_PER_EV
    for s in range(10):
        full[:, 2 + s]  = na[:, :, :, s]
        full[:, 12 + s] = ua[:, :, :, s]
    return full.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Plot one channel: GT | Pred | Error
# ─────────────────────────────────────────────────────────────────────────────

def plot_channel(
    t_field: np.ndarray,       # (104, 50)
    p_field: np.ndarray,       # (104, 50)
    cells, xlim, ylim,
    orig_c: int,
    idx: int,
    out_dir: Optional[Path],
    eps_log: float = 1e-3,
):
    """One figure per channel: Ground Truth | Prediction | Error."""

    t_flat = t_field.ravel().astype(np.float64)
    p_flat = p_field.ravel().astype(np.float64)

    # ── Choose norm and color limits ─────────────────────────────────────
    is_positive = int(orig_c) in POS_CHANNELS

    if is_positive:
        # Shared robust limits across both GT and pred
        combined = np.concatenate([t_flat, p_flat])
        vmin, vmax = robust_pos_limits(combined)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = "viridis"
    else:
        combined = np.concatenate([t_flat, p_flat])
        vmin, vmax = robust_sym_limits(combined)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "RdBu_r"

    # ── Compute error ────────────────────────────────────────────────────
    active = (np.abs(t_flat) > 0) | (np.abs(p_flat) > 0)

    if is_positive:
        log_p = np.log10(np.maximum(p_flat, 0.0) + eps_log)
        log_t = np.log10(np.maximum(t_flat, 0.0) + eps_log)
        err_flat    = np.abs(log_p - log_t)
        err_label   = "Log₁₀ Error"
        err_cmap    = "inferno"
        mean_err    = float(err_flat[active].mean()) if active.any() else 0.0
        rel_pct     = np.abs(p_flat - t_flat) / np.maximum(np.abs(t_flat), 1e-30) * 100.0
        median_pct  = float(np.median(rel_pct[active])) if active.any() else 0.0
        title_stats = f"log₁₀-MAE = {mean_err:.3f}  (median rel = {median_pct:.1f}%)"
    else:
        ref         = np.maximum(np.abs(t_flat), 1e-30)
        err_flat    = np.abs(p_flat - t_flat) / ref * 100.0
        err_label   = "Relative Error (%)"
        err_cmap    = "inferno"
        median_pct  = float(np.median(err_flat[active])) if active.any() else 0.0
        title_stats = f"median relative error = {median_pct:.1f}%"

    # Error norm: robust upper limit
    err_active = err_flat[active] if active.any() else err_flat
    err_vmax = float(np.quantile(err_active, 0.99)) if err_active.size > 0 else 1.0
    err_vmax = max(err_vmax, 1e-6)
    err_norm = mcolors.Normalize(vmin=0.0, vmax=err_vmax)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    pc0 = plot_field(axes[0], cells, xlim, ylim, t_flat,
                     "Ground Truth", norm, cmap)
    plt.colorbar(pc0, ax=axes[0], label=ch_unit(orig_c),
                 fraction=0.046, pad=0.04)

    pc1 = plot_field(axes[1], cells, xlim, ylim, p_flat,
                     "Prediction", norm, cmap)
    plt.colorbar(pc1, ax=axes[1], label=ch_unit(orig_c),
                 fraction=0.046, pad=0.04)

    pc2 = plot_field(axes[2], cells, xlim, ylim, err_flat,
                     err_label, err_norm, err_cmap)
    plt.colorbar(pc2, ax=axes[2], label=err_label,
                 fraction=0.046, pad=0.04)

    name = ch_name(orig_c)
    fig.suptitle(
        f"Sample {idx}  |  {name} [{ch_unit(orig_c)}]  |  {title_stats}",
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"fieldmap_idx{idx:04d}_{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"    Saved: {path}")
    plt.close(fig)


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
        description="Plot mesh predictions vs ground truth using the "
                    "make_field_maps.py polygon / PatchCollection style."
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
                    help="Output directory for PNGs")
    ap.add_argument("--eps_log", type=float, default=1e-3,
                    help="Epsilon for log10 transform of positive channels")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    # ── Load geometry ────────────────────────────────────────────────────
    crx = np.load(Path(args.geom_dir) / "crx.npy")
    cry = np.load(Path(args.geom_dir) / "cry.npy")
    print(f"\n  Geometry: crx={crx.shape}  cry={cry.shape}")

    cells, xlim, ylim = build_cells(crx, cry)
    print(f"  Built {len(cells)} polygon cells")

    # ── Load predictions ─────────────────────────────────────────────────
    pred = np.load(args.pred_path)
    N, C_sel = pred.shape[0], pred.shape[1]
    print(f"  Predictions: {args.pred_path}  shape={pred.shape}")

    # ── Load ground truth ────────────────────────────────────────────────
    truth_full = load_ground_truth(Path(args.truth_dir))
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
            cells    = cells,
            xlim     = xlim,
            ylim     = ylim,
            orig_c   = orig_c,
            idx      = args.idx,
            out_dir  = out_dir,
            eps_log  = args.eps_log,
        )

    print("  Done.\n")


if __name__ == "__main__":
    main()
