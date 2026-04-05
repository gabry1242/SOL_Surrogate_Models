#!/usr/bin/env python3
"""
viz_fm.py  —  Step 4 of the Flow Matching 3-view pipeline

Visualize results: pred vs truth, error maps, uncertainty, metrics, loss curves.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # All plots for view 0, simulation index 0, channels Te and Ti
  python viz_fm.py all \\
      --run_dir scripts/runs/fm3v/run1 \\
      --infer_dir scripts/runs/fm3v/run1/infer_test \\
      --norm_stats scripts/tensor/fm3v/global3v_norm_stats.npz \\
      --tensor_prefix scripts/tensor/fm3v/global3v \\
      --split test --view 0 --idx 0 --channels 0,1

  # Just loss curve
  python viz_fm.py loss-curve --run_dir scripts/runs/fm3v/run1

  # Just metrics bar chart
  python viz_fm.py metrics-bar --infer_dir scripts/runs/fm3v/run1/infer_test
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPECIES = ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]
CH_NAMES = ["Te","Ti"] + [f"na_{s}" for s in SPECIES] + [f"ua_{s}" for s in SPECIES]

def ch_name(c): return CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"

def load_layout(pfx):
    p = Path(f"{pfx}_layout.npz")
    if not p.exists(): return None
    z = np.load(p)
    return {k: int(z[k]) for k in ["W0","H0","W1","H1","W2","H2","Hmax","Wmax"] if k in z.files}

def crop(layout, view, H, W):
    if layout is None: return H, W
    hk, wk = f"H{view}", f"W{view}"
    return (min(layout.get(hk, H), H), min(layout.get(wk, W), W))

def save_fig(fig, path, show=False):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show: plt.show()
    plt.close(fig)


# ── Plot: pred vs truth ──
def plot_compare(pred, truth, mask, idx, channels, layout, view, out_dir):
    H, W = pred.shape[-2:]
    Hc, Wc = crop(layout, view, H, W)
    m = mask[idx, 0, :Hc, :Wc] if mask.ndim == 4 else np.ones((Hc, Wc))

    for c in channels:
        if c >= pred.shape[1]: continue
        t = truth[idx, c, :Hc, :Wc] * m
        p = pred[idx, c, :Hc, :Wc] * m
        valid_t = t[m > 0.5]; valid_p = p[m > 0.5]
        if len(valid_t) == 0: continue
        vmin = min(valid_t.min(), valid_p.min())
        vmax = max(valid_t.max(), valid_p.max())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(t, vmin=vmin, vmax=vmax, aspect="auto"); axes[0].set_title(f"Truth — {ch_name(c)}")
        im = axes[1].imshow(p, vmin=vmin, vmax=vmax, aspect="auto"); axes[1].set_title(f"Pred — {ch_name(c)}")
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.suptitle(f"idx={idx} | {ch_name(c)} | view{view}", fontsize=11)
        save_fig(fig, out_dir / f"compare_v{view}_idx{idx}_{ch_name(c)}.png")


# ── Plot: error maps ──
def plot_error(pred, truth, mask, idx, channels, layout, view, out_dir):
    H, W = pred.shape[-2:]
    Hc, Wc = crop(layout, view, H, W)
    m = mask[idx, 0, :Hc, :Wc] if mask.ndim == 4 else np.ones((Hc, Wc))

    for c in channels:
        if c >= pred.shape[1]: continue
        t = truth[idx, c, :Hc, :Wc] * m
        p = pred[idx, c, :Hc, :Wc] * m
        err = np.abs(p - t) * m
        if (m > 0.5).sum() == 0: continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        v = t[m > 0.5]; vmin, vmax = v.min(), v.max()
        axes[0].imshow(t, vmin=vmin, vmax=vmax, aspect="auto"); axes[0].set_title("Truth")
        axes[1].imshow(p, vmin=vmin, vmax=vmax, aspect="auto"); axes[1].set_title("Pred")
        im = axes[2].imshow(err, cmap="hot", aspect="auto"); axes[2].set_title("|Error|")
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes[2], shrink=0.8)
        mae_val = float(err[m > 0.5].mean())
        fig.suptitle(f"idx={idx} | {ch_name(c)} | view{view} | MAE={mae_val:.4g}")
        save_fig(fig, out_dir / f"error_v{view}_idx{idx}_{ch_name(c)}.png")


# ── Plot: uncertainty ──
def plot_uncertainty(std_arr, mask, idx, channels, layout, view, out_dir):
    H, W = std_arr.shape[-2:]
    Hc, Wc = crop(layout, view, H, W)
    m = mask[idx, 0, :Hc, :Wc] if mask.ndim == 4 else np.ones((Hc, Wc))

    for c in channels:
        if c >= std_arr.shape[1]: continue
        s = std_arr[idx, c, :Hc, :Wc] * m
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(s, cmap="magma", aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, shrink=0.8)
        fig.suptitle(f"Std | idx={idx} | {ch_name(c)} | view{view}")
        save_fig(fig, out_dir / f"uncert_v{view}_idx{idx}_{ch_name(c)}.png")


# ── Plot: metrics bar chart ──
def plot_metrics_bar(metrics, out_dir):
    for vk, vm in metrics.items():
        mae_vals = [vm.get(f"mae_{CH_NAMES[c]}", 0) for c in range(22)]
        rmse_vals = [vm.get(f"rmse_{CH_NAMES[c]}", 0) for c in range(22)]
        # Replace inf/nan with 0 for display
        mae_clean = [v if np.isfinite(v) else 0 for v in mae_vals]
        rmse_clean = [v if np.isfinite(v) else 0 for v in rmse_vals]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        x = np.arange(22)
        axes[0].bar(x, mae_clean, color="steelblue")
        axes[0].set_xticks(x); axes[0].set_xticklabels(CH_NAMES, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("MAE"); axes[0].set_yscale("log")
        axes[0].set_title(f"MAE — {vk} | avg={vm.get('mae_avg',0):.4g}")

        axes[1].bar(x, rmse_clean, color="coral")
        axes[1].set_xticks(x); axes[1].set_xticklabels(CH_NAMES, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("RMSE"); axes[1].set_yscale("log")
        axes[1].set_title(f"RMSE — {vk} | avg={vm.get('rmse_avg',0):.4g}")
        save_fig(fig, out_dir / f"metrics_bar_{vk}.png")


# ── Plot: loss curve ──
def plot_loss_curve(hist, out_dir):
    train = hist.get("train", []); test = hist.get("test", [])
    if not train: print("  No training metrics"); return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs = [e["epoch"] for e in train]
    losses = [e["loss"] for e in train]
    ax1.plot(epochs, losses, "b-", alpha=0.8, label="Train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b"); ax1.set_yscale("log")

    if test:
        ax2 = ax1.twinx()
        te = [e["epoch"] for e in test]
        mae = [e.get("mae_avg") for e in test]
        rmse = [e.get("rmse_avg") for e in test]
        valid_mae = [(ep,v) for ep,v in zip(te,mae) if v and np.isfinite(v)]
        valid_rmse = [(ep,v) for ep,v in zip(te,rmse) if v and np.isfinite(v)]
        if valid_mae: ax2.plot(*zip(*valid_mae), "r--o", ms=3, label="Test MAE")
        if valid_rmse: ax2.plot(*zip(*valid_rmse), "g--s", ms=3, label="Test RMSE")
        ax2.set_ylabel("Test metric", color="r"); ax2.set_yscale("log"); ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    fig.suptitle("Training Curves")
    save_fig(fig, out_dir / "loss_curve.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compare","error","uncertainty","metrics-bar","loss-curve","all"])
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--infer_dir", default=None)
    ap.add_argument("--tensor_prefix", default=None)
    ap.add_argument("--norm_stats", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--channels", default="0,1")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    infer_dir = Path(args.infer_dir) if args.infer_dir else (Path(args.run_dir)/"infer_test" if args.run_dir else None)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.run_dir)/"viz" if args.run_dir else Path("viz"))
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    layout = load_layout(args.tensor_prefix) if args.tensor_prefix else None

    modes = [args.mode] if args.mode != "all" else ["compare","error","uncertainty","metrics-bar","loss-curve"]

    # Load prediction + truth if needed
    pred = truth = mask_arr = std_arr = None
    if any(m in modes for m in ["compare","error","uncertainty"]) and infer_dir:
        pp = infer_dir / f"view{args.view}_pred_phys.npy"
        if pp.exists():
            pred = np.load(pp, mmap_mode="r")
            # Load ground truth
            if args.tensor_prefix:
                yp = Path(f"{args.tensor_prefix}_view{args.view}_Y_{args.split}.npy")
                xp = Path(f"{args.tensor_prefix}_view{args.view}_X_{args.split}.npy")
                if yp.exists() and args.norm_stats:
                    ns = np.load(args.norm_stats)
                    y_mean = ns["y_mean"]; y_std = ns["y_std"]
                    s_c = ns["s_c"]; eps_log = float(ns["eps_log"])
                    # Import inverse_transform from infer_fm
                    from infer_fm import build_ground_truth_physical
                    y_norm = np.load(yp)
                    truth = build_ground_truth_physical(y_norm, y_mean, y_std, s_c, eps_log)
                if xp.exists():
                    X = np.load(xp, mmap_mode="r")
                    # Mask: channel 0 of X, but X is normalized so check > threshold
                    # The mask channel was kept as raw 0/1 in build_tensors
                    mask_arr = (np.array(X[:, 0:1]) > 0.5).astype(np.float32)
                    truth = truth * mask_arr if truth is not None else None

        sp = infer_dir / f"view{args.view}_pred_std_phys.npy"
        if sp.exists(): std_arr = np.load(sp, mmap_mode="r")

    for mode in modes:
        print(f"\n--- {mode} ---")
        if mode == "compare" and pred is not None and truth is not None:
            plot_compare(pred, truth, mask_arr, args.idx, channels, layout, args.view, out_dir)
        elif mode == "error" and pred is not None and truth is not None:
            plot_error(pred, truth, mask_arr, args.idx, channels, layout, args.view, out_dir)
        elif mode == "uncertainty" and std_arr is not None:
            plot_uncertainty(std_arr, mask_arr, args.idx, channels, layout, args.view, out_dir)
        elif mode == "metrics-bar" and infer_dir:
            mp = infer_dir / "metrics.json"
            if mp.exists():
                with open(mp) as f: plot_metrics_bar(json.load(f), out_dir)
            else: print(f"  {mp} not found")
        elif mode == "loss-curve" and args.run_dir:
            hp = Path(args.run_dir) / "metrics.json"
            if hp.exists():
                with open(hp) as f: plot_loss_curve(json.load(f), out_dir)
            else: print(f"  {hp} not found")

    print("\nDone.")

if __name__ == "__main__":
    main()
