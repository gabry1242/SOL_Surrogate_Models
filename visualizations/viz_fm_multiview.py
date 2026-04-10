#!/usr/bin/env python3
"""
viz_fm_multiview.py — Visualize Flow Matching multiview results.

Plots ground truth vs predictions for all 22 channels across all 3 views,
with error maps displayed side-by-side.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Plot all channels
  python viz_fm_multiview.py \\
      --infer_dir scripts/runs/fm_multiview/infer_test \\
      --channels all \\
      --out_dir scripts/runs/fm_multiview/viz

  # Plot specific channels
  python viz_fm_multiview.py \\
      --infer_dir scripts/runs/fm_multiview/infer_test \\
      --channels 0,1,2,3 \\
      --out_dir scripts/runs/fm_multiview/viz

  # Just plot loss curve
  python viz_fm_multiview.py loss-curve \\
      --run_dir scripts/runs/fm_multiview \\
      --out_dir scripts/runs/fm_multiview/viz

  # Plot metrics bar chart
  python viz_fm_multiview.py metrics-bar \\
      --infer_dir scripts/runs/fm_multiview/infer_test \\
      --out_dir scripts/runs/fm_multiview/viz
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPECIES = ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]
CH_NAMES = ["Te","Ti"] + [f"na_{s}" for s in SPECIES] + [f"ua_{s}" for s in SPECIES]

def ch_name(c): 
    return CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"

def parse_channels(s: str, max_c=22) -> List[int]:
    """Parse comma-separated channel list or 'all'."""
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(max_c))
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def save_fig(fig, path, show=False):
    """Save figure and close."""
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show: 
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Truth | Pred | Error for each channel (3 views side-by-side)
# ═══════════════════════════════════════════════════════════════════════════

def plot_channel_truth_pred_error(pred_dict, truth_dict, mask_dict, channels, out_dir):
    """
    For each channel:
      - Row 1 (view 0): truth | pred | error
      - Row 2 (view 1): truth | pred | error
      - Row 3 (view 2): truth | pred | error
    """
    for c in channels:
        if c >= 22:
            continue
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        for view in range(3):
            if view not in pred_dict or view not in truth_dict:
                continue
            
            pred = pred_dict[view][0, c]      # (H, W)
            truth = truth_dict[view][0, c]    # (H, W)
            mask = mask_dict[view][0, 0]      # (H, W)
            
            # Mask out inactive regions
            pred_masked = pred.copy()
            truth_masked = truth.copy()
            pred_masked[mask < 0.5] = np.nan
            truth_masked[mask < 0.5] = np.nan
            
            # Get shared color range from valid data
            valid_mask = mask > 0.5
            if valid_mask.sum() > 0:
                all_vals = np.concatenate([pred[valid_mask], truth[valid_mask]])
                vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            else:
                vmin, vmax = 0, 1
            
            # Error map
            error = np.abs(pred - truth)
            error[mask < 0.5] = np.nan
            
            # Plot truth
            im0 = axes[view, 0].imshow(truth_masked, vmin=vmin, vmax=vmax, aspect="auto", cmap="viridis")
            axes[view, 0].set_title(f"Truth | view{view}", fontsize=10, fontweight="bold")
            axes[view, 0].set_xticks([])
            axes[view, 0].set_yticks([])
            
            # Plot prediction
            im1 = axes[view, 1].imshow(pred_masked, vmin=vmin, vmax=vmax, aspect="auto", cmap="viridis")
            axes[view, 1].set_title(f"Prediction | view{view}", fontsize=10, fontweight="bold")
            axes[view, 1].set_xticks([])
            axes[view, 1].set_yticks([])
            
            # Plot error
            im2 = axes[view, 2].imshow(error, aspect="auto", cmap="hot")
            mae_val = np.nanmean(error[valid_mask]) if valid_mask.sum() > 0 else 0.0
            axes[view, 2].set_title(f"Error | MAE={mae_val:.4g}", fontsize=10, fontweight="bold")
            axes[view, 2].set_xticks([])
            axes[view, 2].set_yticks([])
        
        fig.suptitle(f"Channel {c}: {ch_name(c)}", fontsize=13, fontweight="bold", y=0.995)
        save_fig(fig, out_dir / f"channel_{c:02d}_{ch_name(c)}.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Metrics bar chart (MAE + RMSE per channel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(metrics, out_dir):
    """Plot MAE and RMSE bar charts for all views."""
    for view_key, vm in metrics.items():
        mae_vals = [vm.get(f"mae_{CH_NAMES[c]}", 0) for c in range(22)]
        rmse_vals = [vm.get(f"rmse_{CH_NAMES[c]}", 0) for c in range(22)]
        
        # Replace inf/nan with 0 for display
        mae_clean = [v if np.isfinite(v) else 0 for v in mae_vals]
        rmse_clean = [v if np.isfinite(v) else 0 for v in rmse_vals]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        x = np.arange(22)
        
        axes[0].bar(x, mae_clean, color="steelblue", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(CH_NAMES, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("MAE", fontsize=11)
        axes[0].set_yscale("log")
        axes[0].set_title(f"MAE — {view_key} | avg={vm.get('mae_avg', 0):.4g}", fontsize=12, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(x, rmse_clean, color="coral", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(CH_NAMES, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("RMSE", fontsize=11)
        axes[1].set_yscale("log")
        axes[1].set_title(f"RMSE — {view_key} | avg={vm.get('rmse_avg', 0):.4g}", fontsize=12, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        
        save_fig(fig, out_dir / f"metrics_bar_{view_key}.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Loss curve (training history)
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curve(hist, out_dir):
    """Plot training loss vs epoch."""
    if not isinstance(hist, list):
        print("  No training metrics found, skipping loss curve.")
        return

    epochs = [e["epoch"] for e in hist]
    losses = [e.get("loss") for e in hist]
    
    # Filter out None values
    valid_data = [(ep, loss) for ep, loss in zip(epochs, losses) if loss is not None]
    if not valid_data:
        print("  No valid loss data, skipping loss curve.")
        return
    
    epochs_clean, losses_clean = zip(*valid_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_clean, losses_clean, "b-", alpha=0.8, linewidth=2, label="Train loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11, color="b")
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor="b")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.suptitle("Training Loss Curve", fontsize=12, fontweight="bold")
    save_fig(fig, out_dir / "loss_curve.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Visualize Flow Matching multiview inference results."
    )
    ap.add_argument("--mode", default="compare",
                    choices=["compare", "metrics-bar", "loss-curve", "all"],
                    help="Which plots to generate")
    ap.add_argument("--infer_dir", default=None,
                    help="Inference output directory (contains view*_pred_phys.npy)")
    ap.add_argument("--run_dir", default=None,
                    help="Training run directory (for loss curve)")
    ap.add_argument("--channels", default="0,1",
                    help="Channels to plot: '0,1,2' or 'all'")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for PNG plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = parse_channels(args.channels, 22)
    
    print(f"\n{'='*70}")
    print(f"Flow Matching MultiView Visualization")
    print(f"  Channels: {channels}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    # ── Load predictions and ground truth ──
    pred_dict = {}
    truth_dict = {}
    mask_dict = {}
    
    if args.mode in ["compare", "all"] and args.infer_dir:
        infer_dir = Path(args.infer_dir)
        
        for view in range(3):
            pred_path = infer_dir / f"view{view}_pred_phys.npy"
            if pred_path.exists():
                pred_dict[view] = np.load(pred_path)
                print(f"  Loaded view{view} predictions: {pred_path.name}  shape={pred_dict[view].shape}")
            else:
                print(f"  WARNING: {pred_path} not found")
        
        # We need ground truth for comparison - load from normalized predictions and convert
        # Actually, we should have saved ground truth during inference
        # For now, we'll compute it from the normalized pred files
        for view in range(3):
            truth_path = infer_dir / f"view{view}_truth_phys.npy"
            mask_path = infer_dir / f"view{view}_mask.npy"
            
            # Check if truth was saved
            if truth_path.exists():
                truth_dict[view] = np.load(truth_path)
            else:
                print(f"  WARNING: {truth_path} not found (need to save ground truth in inference)")
            
            if mask_path.exists():
                mask_dict[view] = np.load(mask_path)
            else:
                # Create mask from prediction (assume first channel is mask in input)
                if view in pred_dict:
                    mask_dict[view] = np.ones_like(pred_dict[view][:, 0:1])

    # ── Generate plots ──
    modes = [args.mode] if args.mode != "all" else ["compare", "metrics-bar", "loss-curve"]

    for mode in modes:
        print(f"\n--- {mode} ---")
        
        if mode == "compare" and pred_dict and truth_dict:
            plot_channel_truth_pred_error(pred_dict, truth_dict, mask_dict, channels, out_dir)
        
        elif mode == "metrics-bar" and args.infer_dir:
            metrics_path = Path(args.infer_dir) / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                plot_metrics_bar(metrics, out_dir)
            else:
                print(f"  {metrics_path} not found")
        
        elif mode == "loss-curve" and args.run_dir:
            hist_path = Path(args.run_dir) / "metrics.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    hist = json.load(f)
                plot_loss_curve(hist, out_dir)
            else:
                print(f"  {hist_path} not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
