#!/usr/bin/env python3
"""
viz_fm_3views.py

Visualize results from the 3-view pipeline (works with both flow matching
and deterministic UNet outputs).

=============================================================================
WHAT IT PLOTS
=============================================================================

1. pred-vs-truth   — Side-by-side ground truth vs prediction for selected
                     channels, with shared colorbar across views. Optionally
                     cropped to the active region using the layout map.

2. error-maps      — Absolute error |pred - truth| per channel per view,
                     highlighting where the model struggles.

3. uncertainty      — If multi-sample predictions exist (pred_Y_std_test.npy),
                     plots the per-pixel standard deviation across samples.

4. metrics-bar     — Bar chart of per-channel MAE and RMSE from test_metrics.json.

5. loss-curve      — Training loss vs epoch from metrics.json.

=============================================================================
USAGE
=============================================================================

# Plot prediction vs ground truth for simulation index 0, channels te and ti
python viz_fm_3views.py pred-vs-truth \\
    --run_dir   scripts/runs/fm_3views/view0 \\
    --tensor_prefix scripts/tensor/3views_4d/test/global3v \\
    --split test --idx 0 --channels 0,1 \\
    --out_dir scripts/runs/fm_3views/view0/viz

# Plot error maps
python viz_fm_3views.py error-maps \\
    --run_dir   scripts/runs/fm_3views/view0 \\
    --tensor_prefix scripts/tensor/3views_4d/test/global3v \\
    --split test --idx 0 --channels 0,1 \\
    --out_dir scripts/runs/fm_3views/view0/viz

# Plot uncertainty (requires --n_samples > 1 during inference)
python viz_fm_3views.py uncertainty \\
    --run_dir   scripts/runs/fm_3views/view0 \\
    --split test --idx 0 --channels 0,1 \\
    --out_dir scripts/runs/fm_3views/view0/viz

# Plot per-channel metrics bar chart
python viz_fm_3views.py metrics-bar \\
    --run_dir   scripts/runs/fm_3views/view0 \\
    --out_dir scripts/runs/fm_3views/view0/viz

# Plot training loss curve
python viz_fm_3views.py loss-curve \\
    --run_dir   scripts/runs/fm_3views/view0 \\
    --out_dir scripts/runs/fm_3views/view0/viz

# All plots at once
python viz_fm_3views.py all \\
    --run_dir   scripts/runs/fm_3views/view0 \\
    --tensor_prefix scripts/tensor/3views_4d/test/global3v \\
    --split test --idx 0 --channels 0,1 \\
    --out_dir scripts/runs/fm_3views/view0/viz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Channel name lookup ─────────────────────────────────────────────────
CHANNEL_NAMES = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]]
    + [f"ua_{s}" for s in ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]]
)

def ch_name(c: int) -> str:
    if c < len(CHANNEL_NAMES):
        return CHANNEL_NAMES[c]
    return f"ch{c}"


# ── Layout loading ──────────────────────────────────────────────────────
def load_layout(tensor_prefix: str) -> Optional[dict]:
    """Try to load layout map for view cropping."""
    p = Path(f"{tensor_prefix}_layout_map_3views.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    out = {}
    for k in ["W0", "H0", "W1", "H1", "W2", "H2", "Hmax", "Wmax"]:
        if k in z.files:
            out[k] = int(z[k])
    return out


def resolve_crop(layout: Optional[dict], view_id: int, H: int, W: int) -> Tuple[int, int]:
    if layout is None:
        return H, W
    hk, wk = f"H{view_id}", f"W{view_id}"
    if hk in layout and wk in layout:
        return min(layout[hk], H), min(layout[wk], W)
    return H, W


# ── Plotting helpers ────────────────────────────────────────────────────
def _imshow(ax, img, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


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


def parse_channels(s: str, max_c: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(max_c))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Prediction vs Ground Truth
# ═══════════════════════════════════════════════════════════════════════════

def plot_pred_vs_truth(
    pred: np.ndarray,           # (N, C, H, W) physical units
    truth: np.ndarray,          # (N, C, H, W) physical units
    mask: np.ndarray,           # (N, 1, H, W) or (H, W)
    idx: int,
    channels: List[int],
    layout: Optional[dict],
    view_tag: str,
    out_dir: Optional[Path],
    show: bool,
):
    """Side-by-side truth | prediction for each selected channel."""
    H, W = pred.shape[-2], pred.shape[-1]

    # Determine view_id from view_tag
    view_id = int(view_tag.replace("view", ""))
    Hc, Wc = resolve_crop(layout, view_id, H, W)

    # Get mask for this sample
    if mask.ndim == 4:
        m = mask[idx, 0, :Hc, :Wc]
    elif mask.ndim == 2:
        m = mask[:Hc, :Wc]
    else:
        m = mask[0, :Hc, :Wc] if mask.ndim == 3 else np.ones((Hc, Wc))

    for c in channels:
        if c >= pred.shape[1]:
            continue
        t_img = truth[idx, c, :Hc, :Wc] * m
        p_img = pred[idx, c, :Hc, :Wc] * m

        # Shared color range
        valid_t = t_img[m > 0.5]
        valid_p = p_img[m > 0.5]
        if len(valid_t) == 0:
            continue
        vmin = min(valid_t.min(), valid_p.min())
        vmax = max(valid_t.max(), valid_p.max())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im1 = _imshow(axes[0], t_img, f"Ground Truth — {ch_name(c)} ({view_tag})",
                       vmin=vmin, vmax=vmax)
        im2 = _imshow(axes[1], p_img, f"Prediction — {ch_name(c)} ({view_tag})",
                       vmin=vmin, vmax=vmax)
        fig.colorbar(im2, ax=axes, shrink=0.8)
        fig.suptitle(f"Sample {idx} | {ch_name(c)} | {view_tag}", fontsize=11)

        path = out_dir / f"pred_vs_truth_idx{idx:04d}_{ch_name(c)}_{view_tag}.png" if out_dir else None
        _save_or_show(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Error Maps
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_maps(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    idx: int,
    channels: List[int],
    layout: Optional[dict],
    view_tag: str,
    out_dir: Optional[Path],
    show: bool,
):
    """Absolute error |pred - truth| heatmap for each channel."""
    H, W = pred.shape[-2], pred.shape[-1]
    view_id = int(view_tag.replace("view", ""))
    Hc, Wc = resolve_crop(layout, view_id, H, W)

    if mask.ndim == 4:
        m = mask[idx, 0, :Hc, :Wc]
    elif mask.ndim == 2:
        m = mask[:Hc, :Wc]
    else:
        m = mask[0, :Hc, :Wc] if mask.ndim == 3 else np.ones((Hc, Wc))

    for c in channels:
        if c >= pred.shape[1]:
            continue
        t_img = truth[idx, c, :Hc, :Wc] * m
        p_img = pred[idx, c, :Hc, :Wc] * m
        err   = np.abs(p_img - t_img) * m

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        valid = t_img[m > 0.5]
        if len(valid) == 0:
            continue
        vmin, vmax = valid.min(), valid.max()

        _imshow(axes[0], t_img, f"Truth — {ch_name(c)}", vmin=vmin, vmax=vmax)
        _imshow(axes[1], p_img, f"Prediction — {ch_name(c)}", vmin=vmin, vmax=vmax)
        im_err = _imshow(axes[2], err, f"|Error| — {ch_name(c)}", cmap="hot")
        fig.colorbar(im_err, ax=axes[2], shrink=0.8)

        masked_mae = float(err[m > 0.5].mean()) if (m > 0.5).any() else 0.0
        fig.suptitle(f"Sample {idx} | {ch_name(c)} | {view_tag} | masked MAE = {masked_mae:.4g}",
                     fontsize=11)

        path = out_dir / f"error_map_idx{idx:04d}_{ch_name(c)}_{view_tag}.png" if out_dir else None
        _save_or_show(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Uncertainty (multi-sample std)
# ═══════════════════════════════════════════════════════════════════════════

def plot_uncertainty(
    std_arr: np.ndarray,        # (N, C, H, W)
    mask: np.ndarray,
    idx: int,
    channels: List[int],
    layout: Optional[dict],
    view_tag: str,
    out_dir: Optional[Path],
    show: bool,
):
    """Per-pixel standard deviation across samples."""
    H, W = std_arr.shape[-2], std_arr.shape[-1]
    view_id = int(view_tag.replace("view", ""))
    Hc, Wc = resolve_crop(layout, view_id, H, W)

    if mask.ndim == 4:
        m = mask[idx, 0, :Hc, :Wc]
    elif mask.ndim == 2:
        m = mask[:Hc, :Wc]
    else:
        m = mask[0, :Hc, :Wc] if mask.ndim == 3 else np.ones((Hc, Wc))

    for c in channels:
        if c >= std_arr.shape[1]:
            continue
        s_img = std_arr[idx, c, :Hc, :Wc] * m

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im = _imshow(ax, s_img, f"Std across samples — {ch_name(c)} ({view_tag})", cmap="magma")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Uncertainty | Sample {idx} | {ch_name(c)} | {view_tag}", fontsize=11)

        path = out_dir / f"uncertainty_idx{idx:04d}_{ch_name(c)}_{view_tag}.png" if out_dir else None
        _save_or_show(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Per-channel metrics bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(
    metrics: dict,
    out_dir: Optional[Path],
    show: bool,
):
    """Bar chart of per-channel MAE and RMSE."""
    mae_list  = metrics.get("mae_per_channel", [])
    rmse_list = metrics.get("rmse_per_channel", [])
    if not mae_list:
        print("  No per-channel metrics found, skipping.")
        return

    y_indices = metrics.get("y_indices", list(range(len(mae_list))))
    names = [ch_name(c) for c in y_indices]
    n = len(mae_list)

    # Filter out Infinity for plotting
    mae_clean  = [v if np.isfinite(v) else 0 for v in mae_list]
    rmse_clean = [v if np.isfinite(v) else 0 for v in rmse_list]
    has_inf    = any(not np.isfinite(v) for v in mae_list + rmse_list)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 0.6), 8))
    x = np.arange(n)

    axes[0].bar(x, mae_clean, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("MAE (physical)")
    axes[0].set_title(f"Per-channel MAE — {metrics.get('view_tag', '?')} | avg={metrics.get('mae_avg', 0):.4g}")
    axes[0].set_yscale("log")

    axes[1].bar(x, rmse_clean, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("RMSE (physical)")
    axes[1].set_title(f"Per-channel RMSE — {metrics.get('view_tag', '?')} | avg={metrics.get('rmse_avg', 0):.4g}")
    axes[1].set_yscale("log")

    if has_inf:
        fig.text(0.5, 0.01, "Note: Inf values replaced with 0 for display", ha="center",
                 fontsize=8, color="red")

    path = out_dir / f"metrics_bar_{metrics.get('view_tag', 'view')}.png" if out_dir else None
    _save_or_show(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Training loss curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curve(
    metrics_hist: dict,
    view_tag: str,
    out_dir: Optional[Path],
    show: bool,
):
    """Training loss and (if available) test MAE/RMSE vs epoch."""
    train = metrics_hist.get("train", [])
    test  = metrics_hist.get("test", [])

    if not train:
        print("  No training metrics found, skipping.")
        return

    epochs_train = [e["epoch"] for e in train]
    losses       = [e["loss"]  for e in train if "loss" in e]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.plot(epochs_train[:len(losses)], losses, "b-", label="Train loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="b")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="b")

    if test:
        epochs_test = [e["epoch"]   for e in test]
        mae_test    = [e.get("mae_avg", None)  for e in test]
        rmse_test   = [e.get("rmse_avg", None) for e in test]

        ax2 = ax1.twinx()
        valid_mae = [(ep, v) for ep, v in zip(epochs_test, mae_test)
                     if v is not None and np.isfinite(v)]
        valid_rmse = [(ep, v) for ep, v in zip(epochs_test, rmse_test)
                      if v is not None and np.isfinite(v)]
        if valid_mae:
            ax2.plot(*zip(*valid_mae), "r--o", label="Test MAE", markersize=3, alpha=0.8)
        if valid_rmse:
            ax2.plot(*zip(*valid_rmse), "g--s", label="Test RMSE", markersize=3, alpha=0.8)
        ax2.set_ylabel("Test metric", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    fig.suptitle(f"Training curves — {view_tag}", fontsize=12)

    path = out_dir / f"loss_curve_{view_tag}.png" if out_dir else None
    _save_or_show(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Visualize 3-view flow matching (or UNet) results."
    )
    ap.add_argument("mode", choices=[
        "pred-vs-truth", "error-maps", "uncertainty", "metrics-bar", "loss-curve", "all"
    ], help="Which plot(s) to generate.")

    ap.add_argument("--run_dir", required=True,
                    help="Run directory containing infer_test/ and metrics.json")
    ap.add_argument("--tensor_prefix", default=None,
                    help="Tensor prefix for ground truth (needed for pred-vs-truth, error-maps)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--idx", type=int, default=0,
                    help="Simulation index to visualize")
    ap.add_argument("--channels", default="0,1",
                    help='Channels to plot, e.g. "0,1" or "all"')
    ap.add_argument("--infer_subdir", default="infer_test",
                    help="Subdirectory under run_dir containing predictions")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for PNGs (default: run_dir/viz)")
    ap.add_argument("--no_show", action="store_true")

    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    infer_dir = run_dir / args.infer_subdir
    out_dir   = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    show      = not args.no_show

    # ── Load config to get view_tag ──
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        view_tag = config.get("view_tag", "view0")
    else:
        view_tag = "view0"
        config = {}

    modes = [args.mode] if args.mode != "all" else [
        "pred-vs-truth", "error-maps", "uncertainty", "metrics-bar", "loss-curve"
    ]

    # ── Load prediction if needed ──
    pred = mask_arr = truth = None
    if any(m in modes for m in ["pred-vs-truth", "error-maps"]):
        pred_path = infer_dir / "pred_Y_img_test.npy"
        if not pred_path.exists():
            print(f"  Prediction file not found: {pred_path}")
            print(f"  Run inference first, then try again.")
            modes = [m for m in modes if m not in ["pred-vs-truth", "error-maps"]]
        else:
            pred = np.load(pred_path, mmap_mode="r")
            print(f"  Loaded predictions: {pred_path}  shape={pred.shape}")

            # Load ground truth
            if args.tensor_prefix is None:
                tp = config.get("test_prefix", config.get("tensor_prefix"))
                if tp is None:
                    print("  Need --tensor_prefix for ground truth comparison.")
                    modes = [m for m in modes if m not in ["pred-vs-truth", "error-maps"]]
            else:
                tp = args.tensor_prefix

            if tp is not None:
                y_path = Path(f"{tp}_{view_tag}_Y_img_{args.split}.npy")
                x_path = Path(f"{tp}_{view_tag}_X_img_{args.split}.npy")
                if y_path.exists():
                    truth = np.load(y_path, mmap_mode="r")
                    # Select same y_indices as used in training
                    y_indices = config.get("y_indices", list(range(truth.shape[1])))
                    truth_sel = np.array(truth[:, y_indices, :, :], dtype=np.float32)
                    print(f"  Loaded truth: {y_path}  shape={truth.shape} -> selected {truth_sel.shape}")
                    truth = truth_sel
                else:
                    print(f"  Ground truth not found: {y_path}")
                    modes = [m for m in modes if m not in ["pred-vs-truth", "error-maps"]]

                # Load mask from X channel 0
                if x_path.exists():
                    X_test = np.load(x_path, mmap_mode="r")
                    mask_arr = X_test[:, 0:1, :, :].astype(np.float32)
                else:
                    mask_arr = np.ones((pred.shape[0], 1, pred.shape[2], pred.shape[3]),
                                       dtype=np.float32)

    # Layout for cropping
    layout = None
    if args.tensor_prefix:
        layout = load_layout(args.tensor_prefix)
    elif config.get("test_prefix"):
        layout = load_layout(config["test_prefix"])

    # Parse channel selection
    max_c = pred.shape[1] if pred is not None else 22
    channels = parse_channels(args.channels, max_c)

    # ── Execute each mode ──
    for mode in modes:
        print(f"\n--- {mode} ---")

        if mode == "pred-vs-truth" and pred is not None and truth is not None:
            plot_pred_vs_truth(pred, truth, mask_arr, args.idx, channels,
                               layout, view_tag, out_dir, show)

        elif mode == "error-maps" and pred is not None and truth is not None:
            plot_error_maps(pred, truth, mask_arr, args.idx, channels,
                            layout, view_tag, out_dir, show)

        elif mode == "uncertainty":
            std_path = infer_dir / "pred_Y_std_test.npy"
            if std_path.exists():
                std_arr = np.load(std_path, mmap_mode="r")
                if mask_arr is None:
                    # Try loading mask
                    tp = args.tensor_prefix or config.get("test_prefix")
                    if tp:
                        x_path = Path(f"{tp}_{view_tag}_X_img_{args.split}.npy")
                        if x_path.exists():
                            X_test = np.load(x_path, mmap_mode="r")
                            mask_arr = X_test[:, 0:1, :, :].astype(np.float32)
                    if mask_arr is None:
                        mask_arr = np.ones((std_arr.shape[0], 1, std_arr.shape[2],
                                            std_arr.shape[3]), dtype=np.float32)
                plot_uncertainty(std_arr, mask_arr, args.idx, channels,
                                 layout, view_tag, out_dir, show)
            else:
                print(f"  No uncertainty file found: {std_path}")
                print(f"  Run inference with --n_samples > 1 to generate it.")

        elif mode == "metrics-bar":
            metrics_path = infer_dir / "test_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                plot_metrics_bar(metrics, out_dir, show)
            else:
                print(f"  No test_metrics.json found at {metrics_path}")

        elif mode == "loss-curve":
            hist_path = run_dir / "metrics.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    metrics_hist = json.load(f)
                plot_loss_curve(metrics_hist, view_tag, out_dir, show)
            else:
                print(f"  No metrics.json found at {hist_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
