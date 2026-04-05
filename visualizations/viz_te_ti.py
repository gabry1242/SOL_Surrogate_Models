#!/usr/bin/env python3
"""
viz_te_ti.py  —  Visualize Te/Ti flow matching results

Works with the output files from infer_fm_te_ti.py:
  view{0,1,2}_pred_eV.npy, view{0,1,2}_truth_eV.npy,
  view{0,1,2}_pred_std_eV.npy (if multi-sample)

And training history from train_fm_te_ti.py:
  metrics.json (in run_dir)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  compare      — side-by-side truth vs prediction
  error        — truth | prediction | |error| heatmap
  uncertainty  — per-pixel std across samples (needs --n_samples > 1)
  metrics-bar  — bar chart of MAE/RMSE per view from metrics.json
  loss-curve   — training loss + test metrics vs epoch
  all          — all of the above

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # All plots for all 3 views, simulation index 0
  python viz_te_ti.py all \\
      --run_dir scripts/runs/fm3v/te_ti_run1 \\
      --infer_dir scripts/runs/fm3v/te_ti_run1/infer_test \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --idx 0

  # Just loss curve
  python viz_te_ti.py loss-curve \\
      --run_dir scripts/runs/fm3v/te_ti_run1

  # Compare for a specific view and simulation
  python viz_te_ti.py compare \\
      --infer_dir scripts/runs/fm3v/te_ti_run1/infer_test \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --view 1 --idx 5

  # Uncertainty (requires inference with --n_samples > 1)
  python viz_te_ti.py uncertainty \\
      --infer_dir scripts/runs/fm3v/te_ti_run1/infer_test_5s \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --view 0 --idx 0
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CH = {0: "Te", 1: "Ti"}


# ── Layout ──────────────────────────────────────────────────────────────

def load_layout(pfx):
    p = Path(f"{pfx}_layout.npz")
    if not p.exists(): return None
    z = np.load(p)
    return {k: int(z[k]) for k in z.files if k in
            ["W0","H0","W1","H1","W2","H2","Hmax","Wmax"]}

def crop_hw(layout, view, H, W):
    if layout is None: return H, W
    hk, wk = f"H{view}", f"W{view}"
    return min(layout.get(hk, H), H), min(layout.get(wk, W), W)


# ── Helpers ─────────────────────────────────────────────────────────────

def save_fig(fig, path):
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

def get_mask(pfx, view, split, N):
    xp = Path(f"{pfx}_view{view}_X_{split}.npy")
    if xp.exists():
        X = np.load(xp, mmap_mode="r")
        return (np.array(X[:, 0:1]) > 0.5).astype(np.float32)
    return np.ones((N, 1, 1, 1), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_compare(pred, truth, mask, idx, layout, view, out_dir):
    """Side-by-side truth vs prediction for Te and Ti."""
    H, W = pred.shape[-2:]
    Hc, Wc = crop_hw(layout, view, H, W)
    m = mask[idx, 0, :Hc, :Wc]

    for c in range(2):
        t_img = truth[idx, c, :Hc, :Wc] * m
        p_img = pred[idx, c, :Hc, :Wc] * m
        valid = t_img[m > 0.5]
        if len(valid) == 0: continue
        vmin = min(valid.min(), p_img[m > 0.5].min())
        vmax = max(valid.max(), p_img[m > 0.5].max())

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].imshow(t_img, vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title(f"Truth — {CH[c]}", fontsize=11)
        im = axes[1].imshow(p_img, vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title(f"Prediction — {CH[c]}", fontsize=11)
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.75, label="eV")
        fig.suptitle(f"Sample {idx} | {CH[c]} | view{view}", fontsize=12)
        save_fig(fig, out_dir / f"compare_v{view}_idx{idx}_{CH[c]}.png")


def plot_error(pred, truth, mask, idx, layout, view, out_dir):
    """Truth | Prediction | |Error| for Te and Ti."""
    H, W = pred.shape[-2:]
    Hc, Wc = crop_hw(layout, view, H, W)
    m = mask[idx, 0, :Hc, :Wc]

    for c in range(2):
        t_img = truth[idx, c, :Hc, :Wc] * m
        p_img = pred[idx, c, :Hc, :Wc] * m
        err = np.abs(p_img - t_img) * m
        valid = t_img[m > 0.5]
        if len(valid) == 0: continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmin, vmax = valid.min(), valid.max()
        axes[0].imshow(t_img, vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title("Truth (eV)")
        axes[1].imshow(p_img, vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title("Prediction (eV)")
        im_err = axes[2].imshow(err, cmap="hot", aspect="auto")
        axes[2].set_title("|Error| (eV)")
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im_err, ax=axes[2], shrink=0.75)

        mae_val = float(err[m > 0.5].mean())
        fig.suptitle(f"Sample {idx} | {CH[c]} | view{view} | MAE = {mae_val:.1f} eV",
                     fontsize=12)
        save_fig(fig, out_dir / f"error_v{view}_idx{idx}_{CH[c]}.png")


def plot_uncertainty(std_arr, mask, idx, layout, view, out_dir):
    """Per-pixel std across samples."""
    H, W = std_arr.shape[-2:]
    Hc, Wc = crop_hw(layout, view, H, W)
    m = mask[idx, 0, :Hc, :Wc]

    for c in range(2):
        s_img = std_arr[idx, c, :Hc, :Wc] * m
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(s_img, cmap="magma", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, shrink=0.75, label="std (eV)")
        fig.suptitle(f"Uncertainty | Sample {idx} | {CH[c]} | view{view}", fontsize=12)
        save_fig(fig, out_dir / f"uncert_v{view}_idx{idx}_{CH[c]}.png")


def plot_metrics_bar(metrics, out_dir):
    """Bar chart of MAE and RMSE per view."""
    views = sorted(metrics.keys())
    labels_mae = ["mae_eV_Te", "mae_eV_Ti"]
    labels_rmse = ["rmse_eV_Te", "rmse_eV_Ti"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(views))
    w = 0.35

    # MAE
    for i, lbl in enumerate(labels_mae):
        vals = [metrics[v].get(lbl, 0) for v in views]
        axes[0].bar(x + i*w, vals, w, label=lbl.split("_")[-1])
    axes[0].set_xticks(x + w/2); axes[0].set_xticklabels(views)
    axes[0].set_ylabel("MAE (eV)"); axes[0].legend(); axes[0].set_title("MAE per view")

    # RMSE
    for i, lbl in enumerate(labels_rmse):
        vals = [metrics[v].get(lbl, 0) for v in views]
        axes[1].bar(x + i*w, vals, w, label=lbl.split("_")[-1])
    axes[1].set_xticks(x + w/2); axes[1].set_xticklabels(views)
    axes[1].set_ylabel("RMSE (eV)"); axes[1].legend(); axes[1].set_title("RMSE per view")

    save_fig(fig, out_dir / "metrics_bar.png")

    # Also show normalized-space metrics
    labels_n = ["mae_norm_Te", "mae_norm_Ti"]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for i, lbl in enumerate(labels_n):
        vals = [metrics[v].get(lbl, 0) for v in views]
        ax2.bar(x + i*w, vals, w, label=lbl.split("_")[-1])
    ax2.set_xticks(x + w/2); ax2.set_xticklabels(views)
    ax2.set_ylabel("MAE (normalized)"); ax2.legend()
    ax2.set_title("Normalized-space MAE (model quality)")
    save_fig(fig2, out_dir / "metrics_bar_norm.png")


def plot_loss_curve(hist, out_dir):
    """Training loss + test MAE/RMSE vs epoch."""
    train = hist.get("train", []); test = hist.get("test", [])
    if not train: print("  No training data"); return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs = [e["epoch"] for e in train]
    losses = [e["loss"] for e in train]
    ax1.plot(epochs, losses, "b-", alpha=0.8, label="Train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b")
    ax1.set_yscale("log"); ax1.tick_params(axis="y", labelcolor="b")

    if test:
        ax2 = ax1.twinx()
        te_ep = [e["epoch"] for e in test]
        te_mae = [e.get("mae_Te") for e in test]
        ti_mae = [e.get("mae_Ti") for e in test]
        te_rmse = [e.get("rmse_Te") for e in test]

        valid_te = [(ep,v) for ep,v in zip(te_ep, te_mae) if v is not None and np.isfinite(v)]
        valid_ti = [(ep,v) for ep,v in zip(te_ep, ti_mae) if v is not None and np.isfinite(v)]
        valid_rmse = [(ep,v) for ep,v in zip(te_ep, te_rmse) if v is not None and np.isfinite(v)]

        if valid_te: ax2.plot(*zip(*valid_te), "r--o", ms=3, label="Test MAE Te (norm)")
        if valid_ti: ax2.plot(*zip(*valid_ti), "g--s", ms=3, label="Test MAE Ti (norm)")
        if valid_rmse: ax2.plot(*zip(*valid_rmse), "m--^", ms=3, label="Test RMSE avg (norm)")
        ax2.set_ylabel("Test metric (norm space)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.legend(loc="upper right", fontsize=8)

    ax1.legend(loc="upper left")
    fig.suptitle("Training Curves — Te + Ti", fontsize=12)
    save_fig(fig, out_dir / "loss_curve.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compare","error","uncertainty",
                                      "metrics-bar","loss-curve","all"])
    ap.add_argument("--run_dir", default=None,
                    help="Training run directory (for loss-curve)")
    ap.add_argument("--infer_dir", default=None,
                    help="Inference output directory (for compare/error/uncertainty/metrics)")
    ap.add_argument("--tensor_prefix", default=None,
                    help="Tensor prefix (for masks and layout)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--view", type=int, default=-1,
                    help="Which view (0,1,2). Use -1 for all views.")
    ap.add_argument("--idx", type=int, default=0,
                    help="Simulation index to visualize")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for PNGs (default: run_dir/viz or infer_dir/viz)")
    args = ap.parse_args()

    # Resolve directories
    infer_dir = Path(args.infer_dir) if args.infer_dir else (
        Path(args.run_dir) / "infer_test" if args.run_dir else None)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.run_dir) / "viz" if args.run_dir else
        infer_dir / "viz" if infer_dir else Path("viz"))
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [args.view] if args.view >= 0 else [0, 1, 2]
    layout = load_layout(args.tensor_prefix) if args.tensor_prefix else None

    modes = [args.mode] if args.mode != "all" else [
        "compare", "error", "uncertainty", "metrics-bar", "loss-curve"]

    for mode in modes:
        print(f"\n--- {mode} ---")

        if mode in ("compare", "error", "uncertainty") and infer_dir:
            for view in views:
                pred_path = infer_dir / f"view{view}_pred_eV.npy"
                truth_path = infer_dir / f"view{view}_truth_eV.npy"
                std_path = infer_dir / f"view{view}_pred_std_eV.npy"

                if not pred_path.exists():
                    print(f"  view{view}: {pred_path} not found, skipping")
                    continue

                pred = np.load(pred_path, mmap_mode="r")
                mask = get_mask(args.tensor_prefix, view, args.split, pred.shape[0]) \
                    if args.tensor_prefix else np.ones((pred.shape[0],1,pred.shape[2],pred.shape[3]))

                if mode == "compare" and truth_path.exists():
                    truth = np.load(truth_path, mmap_mode="r")
                    plot_compare(pred, truth, mask, args.idx, layout, view, out_dir)
                elif mode == "error" and truth_path.exists():
                    truth = np.load(truth_path, mmap_mode="r")
                    plot_error(pred, truth, mask, args.idx, layout, view, out_dir)
                elif mode == "uncertainty" and std_path.exists():
                    std_arr = np.load(std_path, mmap_mode="r")
                    plot_uncertainty(std_arr, mask, args.idx, layout, view, out_dir)
                elif mode == "uncertainty":
                    print(f"  view{view}: no std file found (run inference with --n_samples > 1)")

        elif mode == "metrics-bar" and infer_dir:
            mp = infer_dir / "metrics.json"
            if mp.exists():
                with open(mp) as f: plot_metrics_bar(json.load(f), out_dir)
            else:
                print(f"  {mp} not found")

        elif mode == "loss-curve" and args.run_dir:
            hp = Path(args.run_dir) / "metrics.json"
            if hp.exists():
                with open(hp) as f: plot_loss_curve(json.load(f), out_dir)
            else:
                print(f"  {hp} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
