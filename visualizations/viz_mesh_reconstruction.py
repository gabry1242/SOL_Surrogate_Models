#!/usr/bin/env python3
"""
viz_mesh_reconstruction.py
─────────────────────────────────────────────────────────────────────────────
Visualize reconstructed mesh predictions vs ground truth on the original
(104, 50) logical grid.

Supports two rendering modes:
  1. Logical grid   — simple imshow on the (104, 50) array  (default)
  2. Physical grid  — plots each cell as a quadrilateral using crx/cry
                      corner coordinates (--geom_dir geometry/)

Plot types:
  pred-vs-truth   Side-by-side ground truth | prediction with shared colorbar
  error-map       Absolute and relative error maps
  scatter          Scatter plot of pred vs truth for each channel
  profile          1D profiles along a selected row or column of the mesh
  all              All of the above

Usage
─────
  # Logical grid, all channels, simulation 0
  python viz_mesh_reconstruction.py pred-vs-truth \\
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \\
      --truth_dir   test/ \\
      --idx 0 --channels 0,1 \\
      --out_dir scripts/runs/my_run/infer_test/viz_mesh

  # Physical geometry overlay
  python viz_mesh_reconstruction.py pred-vs-truth \\
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \\
      --truth_dir   test/ \\
      --geom_dir    geometry/ \\
      --idx 0 --channels 0,1 \\
      --out_dir scripts/runs/my_run/infer_test/viz_mesh

  # All plots at once
  python viz_mesh_reconstruction.py all \\
      --pred_path   scripts/runs/my_run/infer_test/pred_Y_img_test_mesh.npy \\
      --truth_dir   test/ \\
      --geom_dir    geometry/ \\
      --idx 0 --channels all \\
      --out_dir scripts/runs/my_run/infer_test/viz_mesh
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

J_PER_EV = 1.602176634e-19

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]

# Full 22-channel output ordering (must match build_3view_tensors.py)
CHANNEL_NAMES: List[str] = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)

# Channel categories
POS_CHANNELS  = set(range(0, 12))    # Te, Ti, na_*
SIGNED_CHANNELS = set(range(12, 22)) # ua_*


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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(truth_dir: Path, channels: List[int]) -> np.ndarray:
    """
    Load ground-truth fields from the raw simulation directory and
    assemble them into an (N, C_sel, 104, 50) array matching the
    channel ordering of the prediction tensor.

    Channel ordering: [Te, Ti, na_0..na_9, ua_0..ua_9]  (22 total)
    """
    te = np.load(truth_dir / "te_tmp.npy")   # (N, 104, 50)
    ti = np.load(truth_dir / "ti_tmp.npy")
    na = np.load(truth_dir / "na_tmp.npy")   # (N, 104, 50, 10)
    ua = np.load(truth_dir / "ua_tmp.npy")

    N = te.shape[0]

    # Build full (N, 22, 104, 50) array
    full = np.zeros((N, 22, 104, 50), dtype=np.float64)

    # Te, Ti are stored in Joules; the tensor builder converts to eV
    full[:, 0] = te / J_PER_EV
    full[:, 1] = ti / J_PER_EV

    # na: channels 2..11
    for s in range(10):
        full[:, 2 + s] = na[:, :, :, s]

    # ua: channels 12..21
    for s in range(10):
        full[:, 12 + s] = ua[:, :, :, s]

    # Select requested channels
    return full[:, channels].astype(np.float32)


def load_geometry(geom_dir: Path):
    """Load crx, cry corner coordinate arrays."""
    crx = np.load(geom_dir / "crx.npy")  # (104, 50, 4)
    cry = np.load(geom_dir / "cry.npy")
    return crx, cry


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig, path: Optional[Path], show: bool):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def _get_valid_mask(field: np.ndarray) -> np.ndarray:
    """Return a boolean mask of cells with non-zero values."""
    return np.abs(field) > 0


def _choose_cmap(c: int) -> str:
    """Choose a colormap based on channel type."""
    if c in SIGNED_CHANNELS:
        return "RdBu_r"
    return "inferno"


def _choose_norm(values: np.ndarray, c: int):
    """Choose a matplotlib norm based on channel type."""
    vmin, vmax = float(values.min()), float(values.max())
    if c in SIGNED_CHANNELS:
        # Symmetric around zero for velocity
        vlim = max(abs(vmin), abs(vmax))
        if vlim < 1e-30:
            vlim = 1.0
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    else:
        # Positive quantities: use log scale if range is large
        if vmin > 0 and vmax / max(vmin, 1e-30) > 100:
            return mcolors.LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ─────────────────────────────────────────────────────────────────────────────
# Physical geometry plotter
# ─────────────────────────────────────────────────────────────────────────────

def plot_field_physical(
    ax,
    field: np.ndarray,       # (104, 50)
    crx: np.ndarray,         # (104, 50, 4)
    cry: np.ndarray,         # (104, 50, 4)
    norm,
    cmap: str,
    title: str,
    mask: Optional[np.ndarray] = None,
):
    """Plot a field on the physical geometry using quadrilateral patches."""
    IX, IY = field.shape

    if mask is None:
        mask = _get_valid_mask(field)

    verts = []
    colors = []

    for ix in range(IX):
        for iy in range(IY):
            if not mask[ix, iy]:
                continue
            # Four corners: build polygon
            corners = np.array([
                [crx[ix, iy, k], cry[ix, iy, k]] for k in range(4)
            ])
            verts.append(corners)
            colors.append(field[ix, iy])

    colors = np.array(colors)
    pc = PolyCollection(verts, array=colors, cmap=cmap, norm=norm,
                        edgecolors="face", linewidths=0.1)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("R [m]", fontsize=8)
    ax.set_ylabel("Z [m]", fontsize=8)
    return pc


def plot_field_logical(
    ax,
    field: np.ndarray,       # (104, 50)
    norm,
    cmap: str,
    title: str,
    mask: Optional[np.ndarray] = None,
):
    """Plot a field on the logical (ix, iy) grid using imshow."""
    display = field.copy()
    if mask is not None:
        display[~mask] = np.nan

    im = ax.imshow(display.T, origin="lower", aspect="auto",
                   cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ix", fontsize=8)
    ax.set_ylabel("iy", fontsize=8)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Prediction vs Ground Truth
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_truth(
    pred: np.ndarray,            # (N, C_sel, 104, 50)
    truth: np.ndarray,           # (N, C_sel, 104, 50)
    channels: List[int],
    channel_indices: List[int],  # original channel indices for naming
    idx: int,
    crx: Optional[np.ndarray],
    cry: Optional[np.ndarray],
    out_dir: Optional[Path],
    show: bool,
):
    use_phys = crx is not None and cry is not None

    for j, c_orig in enumerate(channel_indices):
        if j >= pred.shape[1]:
            continue

        t_field = truth[idx, j]
        p_field = pred[idx, j]

        mask = _get_valid_mask(t_field) | _get_valid_mask(p_field)
        if not mask.any():
            print(f"  Skipping {ch_name(c_orig)}: all zeros")
            continue

        # Shared normalization
        all_vals = np.concatenate([t_field[mask], p_field[mask]])
        cmap = _choose_cmap(c_orig)
        norm = _choose_norm(all_vals, c_orig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5) if use_phys else (15, 4))

        if use_phys:
            pc1 = plot_field_physical(axes[0], t_field, crx, cry, norm, cmap,
                                      f"Ground Truth — {ch_name(c_orig)}", mask)
            pc2 = plot_field_physical(axes[1], p_field, crx, cry, norm, cmap,
                                      f"Prediction — {ch_name(c_orig)}", mask)
        else:
            pc1 = plot_field_logical(axes[0], t_field, norm, cmap,
                                     f"Ground Truth — {ch_name(c_orig)}", mask)
            pc2 = plot_field_logical(axes[1], p_field, norm, cmap,
                                     f"Prediction — {ch_name(c_orig)}", mask)

        fig.colorbar(pc2, ax=axes[:2], shrink=0.8, label=ch_unit(c_orig))

        # Error panel
        err = np.abs(p_field - t_field)
        err_cmap = "hot"
        err_norm = mcolors.Normalize(vmin=0, vmax=float(err[mask].max()) if mask.any() else 1)

        if use_phys:
            pc3 = plot_field_physical(axes[2], err, crx, cry, err_norm, err_cmap,
                                      f"|Error| — {ch_name(c_orig)}", mask)
        else:
            pc3 = plot_field_logical(axes[2], err, err_norm, err_cmap,
                                     f"|Error| — {ch_name(c_orig)}", mask)
        fig.colorbar(pc3, ax=axes[2], shrink=0.8, label=ch_unit(c_orig))

        # Summary stats
        mae = float(np.mean(err[mask]))
        rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
        ref = np.maximum(np.abs(t_field[mask]), 1e-30)
        mre = float(np.mean(err[mask] / ref))

        fig.suptitle(
            f"Sample {idx} | {ch_name(c_orig)} [{ch_unit(c_orig)}] — "
            f"MAE={mae:.4g}  RMSE={rmse:.4g}  MRE={mre:.3%}",
            fontsize=11,
        )

        path = (out_dir / f"pred_vs_truth_idx{idx:04d}_{ch_name(c_orig)}.png"
                if out_dir else None)
        _save_or_show(fig, path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Error Maps (absolute + relative)
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_maps(
    pred: np.ndarray,
    truth: np.ndarray,
    channels: List[int],
    channel_indices: List[int],
    idx: int,
    crx: Optional[np.ndarray],
    cry: Optional[np.ndarray],
    out_dir: Optional[Path],
    show: bool,
):
    use_phys = crx is not None and cry is not None

    for j, c_orig in enumerate(channel_indices):
        if j >= pred.shape[1]:
            continue

        t_field = truth[idx, j]
        p_field = pred[idx, j]
        mask = _get_valid_mask(t_field) | _get_valid_mask(p_field)
        if not mask.any():
            continue

        abs_err = np.abs(p_field - t_field)
        ref = np.maximum(np.abs(t_field), 1e-30)
        rel_err = abs_err / ref

        fig, axes = plt.subplots(1, 2, figsize=(14, 5) if use_phys else (12, 4))

        # Absolute error
        ae_norm = mcolors.Normalize(vmin=0, vmax=float(abs_err[mask].max()))
        if use_phys:
            pc1 = plot_field_physical(axes[0], abs_err, crx, cry, ae_norm, "hot",
                                      f"|Error| — {ch_name(c_orig)}", mask)
        else:
            pc1 = plot_field_logical(axes[0], abs_err, ae_norm, "hot",
                                     f"|Error| — {ch_name(c_orig)}", mask)
        fig.colorbar(pc1, ax=axes[0], shrink=0.8, label=ch_unit(c_orig))

        # Relative error (capped at 100% for display)
        re_vals = rel_err[mask]
        re_cap = min(float(np.percentile(re_vals, 99)), 2.0)
        re_norm = mcolors.Normalize(vmin=0, vmax=re_cap)
        if use_phys:
            pc2 = plot_field_physical(axes[1], rel_err, crx, cry, re_norm, "YlOrRd",
                                      f"Relative Error — {ch_name(c_orig)}", mask)
        else:
            pc2 = plot_field_logical(axes[1], rel_err, re_norm, "YlOrRd",
                                     f"Relative Error — {ch_name(c_orig)}", mask)
        fig.colorbar(pc2, ax=axes[1], shrink=0.8, label="fraction")

        mae = float(np.mean(abs_err[mask]))
        mre = float(np.mean(re_vals))
        fig.suptitle(
            f"Sample {idx} | {ch_name(c_orig)} — MAE={mae:.4g}  MRE={mre:.3%}",
            fontsize=11,
        )

        path = (out_dir / f"error_map_idx{idx:04d}_{ch_name(c_orig)}.png"
                if out_dir else None)
        _save_or_show(fig, path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Scatter (pred vs truth per channel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(
    pred: np.ndarray,
    truth: np.ndarray,
    channels: List[int],
    channel_indices: List[int],
    idx: int,
    out_dir: Optional[Path],
    show: bool,
    max_points: int = 5000,
):
    for j, c_orig in enumerate(channel_indices):
        if j >= pred.shape[1]:
            continue

        t_vals = truth[idx, j].ravel()
        p_vals = pred[idx, j].ravel()

        # Filter to valid cells
        mask = (np.abs(t_vals) > 0) | (np.abs(p_vals) > 0)
        t_v = t_vals[mask]
        p_v = p_vals[mask]

        if len(t_v) == 0:
            continue

        # Subsample if too many points
        if len(t_v) > max_points:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(t_v), max_points, replace=False)
            t_v = t_v[sel]
            p_v = p_v[sel]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(t_v, p_v, s=2, alpha=0.4, c="steelblue", edgecolors="none")

        # Perfect prediction line
        lo = min(t_v.min(), p_v.min())
        hi = max(t_v.max(), p_v.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", lw=1, alpha=0.5, label="y = x")

        ax.set_xlabel(f"Ground Truth [{ch_unit(c_orig)}]", fontsize=10)
        ax.set_ylabel(f"Prediction [{ch_unit(c_orig)}]", fontsize=10)
        ax.set_title(f"Sample {idx} | {ch_name(c_orig)} scatter", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

        # R² score
        ss_res = np.sum((p_v - t_v) ** 2)
        ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        path = (out_dir / f"scatter_idx{idx:04d}_{ch_name(c_orig)}.png"
                if out_dir else None)
        _save_or_show(fig, path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: 1D Profiles
# ─────────────────────────────────────────────────────────────────────────────

def plot_profiles(
    pred: np.ndarray,
    truth: np.ndarray,
    channels: List[int],
    channel_indices: List[int],
    idx: int,
    profile_axis: str,
    profile_index: int,
    out_dir: Optional[Path],
    show: bool,
):
    """Plot 1D profiles along a row (fixed ix) or column (fixed iy)."""
    for j, c_orig in enumerate(channel_indices):
        if j >= pred.shape[1]:
            continue

        if profile_axis == "ix":
            # Fixed ix, vary iy
            t_line = truth[idx, j, profile_index, :]
            p_line = pred[idx, j, profile_index, :]
            x_label = "iy"
            title_extra = f"ix={profile_index}"
        else:
            # Fixed iy, vary ix
            t_line = truth[idx, j, :, profile_index]
            p_line = pred[idx, j, :, profile_index]
            x_label = "ix"
            title_extra = f"iy={profile_index}"

        # Skip empty profiles
        if np.all(np.abs(t_line) < 1e-30) and np.all(np.abs(p_line) < 1e-30):
            continue

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1],
                                  sharex=True)

        x = np.arange(len(t_line))

        axes[0].plot(x, t_line, "b-", lw=1.5, label="Ground Truth", alpha=0.8)
        axes[0].plot(x, p_line, "r--", lw=1.5, label="Prediction", alpha=0.8)
        axes[0].set_ylabel(f"{ch_name(c_orig)} [{ch_unit(c_orig)}]", fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].set_title(
            f"Sample {idx} | {ch_name(c_orig)} | {title_extra}", fontsize=11
        )
        axes[0].grid(True, alpha=0.3)

        # Error subplot
        err = p_line - t_line
        axes[1].fill_between(x, err, 0, color="gray", alpha=0.4)
        axes[1].plot(x, err, "k-", lw=0.8)
        axes[1].axhline(0, color="k", lw=0.5, ls="--")
        axes[1].set_xlabel(x_label, fontsize=10)
        axes[1].set_ylabel("Error", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        path = (out_dir / f"profile_{profile_axis}{profile_index}_idx{idx:04d}_{ch_name(c_orig)}.png"
                if out_dir else None)
        _save_or_show(fig, path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Multi-sample summary (all simulations, per-channel statistics)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_stats(
    pred: np.ndarray,
    truth: np.ndarray,
    channel_indices: List[int],
    out_dir: Optional[Path],
    show: bool,
):
    """Bar chart of MAE, RMSE, MRE aggregated over all simulations."""
    N, C_sel = pred.shape[0], pred.shape[1]
    maes  = np.zeros(C_sel)
    rmses = np.zeros(C_sel)
    mres  = np.zeros(C_sel)

    for j in range(C_sel):
        mask = _get_valid_mask(truth[:, j]) | _get_valid_mask(pred[:, j])
        if not mask.any():
            continue
        diff = np.abs(pred[:, j][mask] - truth[:, j][mask])
        ref  = np.maximum(np.abs(truth[:, j][mask]), 1e-30)

        maes[j]  = float(np.mean(diff))
        rmses[j] = float(np.sqrt(np.mean(diff ** 2)))
        mres[j]  = float(np.mean(diff / ref))

    names = [ch_name(c) for c in channel_indices[:C_sel]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(names))

    axes[0].bar(x, maes, color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Mean Absolute Error (all sims)")
    axes[0].set_yscale("log")

    axes[1].bar(x, rmses, color="coral", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Root Mean Squared Error (all sims)")
    axes[1].set_yscale("log")

    axes[2].bar(x, mres, color="seagreen", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("MRE")
    axes[2].set_title("Mean Relative Error (all sims)")

    path = out_dir / "summary_metrics.png" if out_dir else None
    _save_or_show(fig, path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Channel parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_channels(s: str, max_c: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(max_c))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualize reconstructed mesh predictions vs ground truth."
    )
    ap.add_argument("mode", choices=[
        "pred-vs-truth", "error-map", "scatter", "profile", "summary", "all"
    ], help="Which plot(s) to generate.")

    ap.add_argument("--pred_path", required=True,
                    help="Path to reconstructed prediction .npy (N, C, 104, 50)")
    ap.add_argument("--truth_dir", required=True,
                    help="Directory containing te_tmp.npy, ti_tmp.npy, na_tmp.npy, ua_tmp.npy")
    ap.add_argument("--geom_dir", default=None,
                    help="Directory containing crx.npy, cry.npy for physical geometry plots")
    ap.add_argument("--y_indices", default=None,
                    help="Original channel indices used during training, e.g. '0,1,2,...,21'. "
                         "If omitted, assumes all 22 channels in order.")
    ap.add_argument("--idx", type=int, default=0,
                    help="Simulation index to visualize")
    ap.add_argument("--channels", default="0,1",
                    help="Which output channels to plot (indices into the prediction tensor). "
                         '"all" or comma-separated, e.g. "0,1,12,13"')
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for PNGs")
    ap.add_argument("--no_show", action="store_true",
                    help="Don't show interactive plots")
    ap.add_argument("--profile_axis", default="ix", choices=["ix", "iy"],
                    help="For profile plots: fix this axis")
    ap.add_argument("--profile_index", type=int, default=52,
                    help="For profile plots: index along the fixed axis")

    args = ap.parse_args()

    show    = not args.no_show
    out_dir = Path(args.out_dir) if args.out_dir else None

    # ── Load predictions ────────────────────────────────────────────────────
    pred = np.load(args.pred_path)  # (N, C_sel, 104, 50)
    N, C_sel = pred.shape[0], pred.shape[1]
    print(f"\n  Predictions: {args.pred_path}  shape={pred.shape}")

    # ── Determine original channel indices ──────────────────────────────────
    if args.y_indices is not None:
        y_indices = [int(x.strip()) for x in args.y_indices.split(",")]
    else:
        y_indices = list(range(min(C_sel, 22)))

    if len(y_indices) != C_sel:
        print(f"  Warning: y_indices length ({len(y_indices)}) != prediction channels ({C_sel})")
        print(f"  Assuming first {C_sel} indices.")
        y_indices = list(range(C_sel))

    # ── Parse channel selection (into prediction tensor indices) ────────────
    channels = parse_channels(args.channels, C_sel)
    # Map to original channel indices for naming
    channel_indices = [y_indices[c] for c in channels if c < len(y_indices)]

    print(f"  Channels to plot: {channels} → original indices {channel_indices}")
    print(f"  Channel names: {[ch_name(c) for c in channel_indices]}")

    # ── Load ground truth ───────────────────────────────────────────────────
    truth_dir = Path(args.truth_dir)
    truth = load_ground_truth(truth_dir, y_indices)
    print(f"  Ground truth: {truth_dir}  shape={truth.shape}")

    if truth.shape[0] != N:
        print(f"  Warning: prediction N={N} != truth N={truth.shape[0]}")
        n_use = min(N, truth.shape[0])
        pred  = pred[:n_use]
        truth = truth[:n_use]

    # ── Load geometry (optional) ────────────────────────────────────────────
    crx, cry = None, None
    if args.geom_dir:
        crx, cry = load_geometry(Path(args.geom_dir))
        print(f"  Geometry: {args.geom_dir}  crx={crx.shape} cry={cry.shape}")
        print(f"  → Physical geometry plots enabled")
    else:
        print(f"  → Logical grid plots (use --geom_dir for physical geometry)")

    # ── Determine plot modes ────────────────────────────────────────────────
    modes = [args.mode] if args.mode != "all" else [
        "pred-vs-truth", "error-map", "scatter", "profile", "summary"
    ]

    # ── Execute ─────────────────────────────────────────────────────────────
    for mode in modes:
        print(f"\n  ── {mode} ──")

        if mode == "pred-vs-truth":
            plot_pred_vs_truth(pred, truth, channels, channel_indices,
                               args.idx, crx, cry, out_dir, show)

        elif mode == "error-map":
            plot_error_maps(pred, truth, channels, channel_indices,
                            args.idx, crx, cry, out_dir, show)

        elif mode == "scatter":
            plot_scatter(pred, truth, channels, channel_indices,
                         args.idx, out_dir, show)

        elif mode == "profile":
            plot_profiles(pred, truth, channels, channel_indices,
                          args.idx, args.profile_axis, args.profile_index,
                          out_dir, show)

        elif mode == "summary":
            plot_summary_stats(pred, truth, channel_indices, out_dir, show)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
