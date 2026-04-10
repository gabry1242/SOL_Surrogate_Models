#!/usr/bin/env python3
"""
viz_fm_overfit.py  —  Visualization for Flow Matching overfit PoC

Generates one plot per channel showing:
  • Ground truth (physical space)
  • Prediction (physical space)
  • Absolute error (|pred - gt|)

Also generates a training convergence plot (loss and RMSE vs epoch).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python viz_fm_overfit.py \\
      --run_dir scripts/runs/fm_overfit/te_ti_v0

This expects:
  run_dir/infer/pred_phys.npy
  run_dir/infer/gt_phys.npy
  run_dir/infer/mask.npy
  run_dir/metrics.json
  run_dir/config.json
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CH_NAMES = (["Te", "Ti"]
            + [f"na_{s}" for s in SPECIES]
            + [f"ua_{s}" for s in SPECIES])


def load_config(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        return json.load(f)


def load_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json") as f:
        return json.load(f)


def plot_channel(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    ch_name: str,
    out_path: Path,
):
    """
    Plot ground truth, prediction, and absolute error for one channel.
    gt, pred, mask: all (H, W)
    """
    error = np.abs(pred - gt) * mask

    # Compute common scale from GT for gt/pred panels
    masked_gt = np.ma.masked_where(mask < 0.5, gt)
    masked_pred = np.ma.masked_where(mask < 0.5, pred)
    masked_err = np.ma.masked_where(mask < 0.5, error)

    vmin = float(masked_gt.min()) if masked_gt.count() > 0 else 0
    vmax = float(masked_gt.max()) if masked_gt.count() > 0 else 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Ground truth
    im0 = axes[0].imshow(masked_gt, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"{ch_name} — Ground Truth")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Prediction
    im1 = axes[1].imshow(masked_pred, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"{ch_name} — Prediction")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error
    err_max = float(masked_err.max()) if masked_err.count() > 0 else 1
    im2 = axes[2].imshow(masked_err, vmin=0, vmax=err_max, cmap="hot")
    axes[2].set_title(f"{ch_name} — |Error|")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_convergence(metrics: dict, out_path: Path):
    """Plot training loss and evaluation RMSE vs epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    if "train_loss" in metrics and len(metrics["train_loss"]) > 0:
        epochs = [d["epoch"] for d in metrics["train_loss"]]
        losses = [d["loss"] for d in metrics["train_loss"]]
        ax1.semilogy(epochs, losses, linewidth=0.5, alpha=0.7, color="steelblue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss (velocity MSE)")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)

    # Eval RMSE
    if "eval_rmse_avg" in metrics and len(metrics["eval_rmse_avg"]) > 0:
        epochs = [d["epoch"] for d in metrics["eval_rmse_avg"]]
        rmses  = [d["rmse_avg"] for d in metrics["eval_rmse_avg"]]
        ax2.semilogy(epochs, rmses, "o-", markersize=3, linewidth=1.5, color="coral")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("RMSE (normalized space)")
        ax2.set_title("Evaluation RMSE (Euler sampling)")
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Overfit Convergence", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_rmse_per_channel(metrics: dict, ch_names: list, out_path: Path):
    """Plot per-channel RMSE convergence over training."""
    if "eval_rmse_per_ch" not in metrics or len(metrics["eval_rmse_per_ch"]) == 0:
        return

    epochs = [d["epoch"] for d in metrics["eval_rmse_per_ch"]]
    all_rmses = [d["rmse_per_ch"] for d in metrics["eval_rmse_per_ch"]]
    n_ch = len(all_rmses[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(n_ch):
        vals = [r[j] for r in all_rmses]
        label = ch_names[j] if j < len(ch_names) else f"ch{j}"
        ax.semilogy(epochs, vals, "o-", markersize=2, linewidth=1.2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (normalized space)")
    ax.set_title("Per-Channel RMSE Convergence")
    ax.legend(fontsize=8, ncol=max(1, n_ch // 6))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Visualize FM overfit results")
    ap.add_argument("--run_dir", required=True,
                    help="Run directory containing infer/ and metrics.json")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for plots (default: run_dir/viz)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_dir = run_dir / "infer"

    # ── Load config ────────────────────────────────────────────────────────
    config = load_config(run_dir)
    y_indices = config["y_indices"]
    ch_names  = config.get("ch_names", [CH_NAMES[c] for c in y_indices])
    c_out     = config["c_out"]

    print(f"\n{'='*60}")
    print(f"Visualization — FM Overfit PoC")
    print(f"  Run:      {run_dir}")
    print(f"  Channels: {ch_names}")
    print(f"{'='*60}")

    # ── 1. Convergence plots ───────────────────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(run_dir)
        plot_convergence(metrics, out_dir / "convergence.png")
        plot_rmse_per_channel(metrics, ch_names, out_dir / "rmse_per_channel.png")
    else:
        print("  Warning: metrics.json not found, skipping convergence plots")

    # ── 2. Per-channel comparison plots ────────────────────────────────────
    pred_path = infer_dir / "pred_phys.npy"
    gt_path   = infer_dir / "gt_phys.npy"
    mask_path = infer_dir / "mask.npy"

    if not pred_path.exists():
        print(f"\n  No inference outputs found at {infer_dir}")
        print(f"  Run infer_fm_overfit.py first.")
        return

    pred = np.load(pred_path)   # (1, c_out, H, W)
    gt   = np.load(gt_path)     # (1, c_out, H, W)
    mask = np.load(mask_path)   # (1, 1, H, W)

    mask_2d = mask[0, 0]  # (H, W)

    print(f"\n  Generating {c_out} channel comparison plots ...")
    for j in range(c_out):
        ch_name = ch_names[j] if j < len(ch_names) else f"ch{j}"
        plot_channel(
            gt=gt[0, j],
            pred=pred[0, j],
            mask=mask_2d,
            ch_name=ch_name,
            out_path=out_dir / f"channel_{j:02d}_{ch_name}.png",
        )

    # ── 3. Summary error bar chart ─────────────────────────────────────────
    infer_metrics_path = infer_dir / "metrics.json"
    if infer_metrics_path.exists():
        with open(infer_metrics_path) as f:
            infer_metrics = json.load(f)

        if "physical_space" in infer_metrics:
            phys = infer_metrics["physical_space"]
            rmse_vals = [phys.get(f"rmse_{ch_names[j]}", 0) for j in range(c_out)]
            mae_vals  = [phys.get(f"mae_{ch_names[j]}", 0) for j in range(c_out)]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, c_out * 0.8), 8))

            x = range(c_out)
            ax1.bar(x, rmse_vals, color="coral", alpha=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
            ax1.set_ylabel("RMSE (physical)")
            ax1.set_title("Per-Channel RMSE — Physical Space")
            ax1.set_yscale("log")
            ax1.grid(True, alpha=0.3, axis="y")

            ax2.bar(x, mae_vals, color="steelblue", alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("MAE (physical)")
            ax2.set_title("Per-Channel MAE — Physical Space")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3, axis="y")

            fig.tight_layout()
            fig.savefig(out_dir / "error_summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_dir / 'error_summary.png'}")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
