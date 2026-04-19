#!/usr/bin/env python3
"""
viz_fm_fusvel.py
─────────────────────────────────────────────────────────────────────────────
Visualize results from the VelocityUNet regression pipeline.

Plots
─────
1. error-maps      — Ground Truth | Prediction | Relative Error (%)
                     for each selected channel and view.
2. metrics-bar     — Per-channel metrics bar chart:
                       • Log-space MAE (what the model optimises)
                       • Mean Relative Error (MRE)
                       • Physical MAE / RMSE
3. loss-curve      — Training loss vs epoch from metrics.json.
4. all             — All of the above.

Usage
─────
python viz_fm_fusvel.py error-maps \
    --run_dir       scripts/runs/fusvel_full \
    --tensor_prefix scripts/tensor/3views_4d/test/global3v \
    --split test --idx 0 --channels 0,1 \
    --out_dir scripts/runs/fusvel_full/viz

python viz_fm_fusvel.py metrics-bar \
    --run_dir scripts/runs/fusvel_full \
    --out_dir scripts/runs/fusvel_full/viz

python viz_fm_fusvel.py loss-curve \
    --run_dir scripts/runs/fusvel_full \
    --out_dir scripts/runs/fusvel_full/viz

python viz_fm_fusvel.py all \
    --run_dir       scripts/runs/fusvel_full \
    --tensor_prefix scripts/tensor/3views_4d/test/global3v \
    --split test --idx 0 --channels 0,1 \
    --out_dir scripts/runs/fusvel_full/viz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Channel name lookup ──────────────────────────────────────────────────

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CHANNEL_NAMES = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)

POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


def ch_name(c: int) -> str:
    if c < len(CHANNEL_NAMES):
        return CHANNEL_NAMES[c]
    return f"ch{c}"


# ── Layout loading (for optional view cropping) ─────────────────────────

def load_layout(tensor_prefix: str) -> Optional[dict]:
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


# ── Plotting helpers ─────────────────────────────────────────────────────

def _imshow(ax, img, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _save_or_show(fig, path: Optional[Path]):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def parse_channels(s: str, max_c: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(max_c))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ═════════════════════════════════════════════════════════════════════════
# PLOT 1: Error maps  (GT | Prediction | Relative Error %)
# ═════════════════════════════════════════════════════════════════════════

def plot_error_maps(
    pred:       np.ndarray,      # (N, C_sel, H, W)
    truth:      np.ndarray,      # (N, C_sel, H, W)
    mask:       np.ndarray,      # (N, 1, H, W)
    idx:        int,
    channels:   List[int],
    y_indices:  List[int],
    layout:     Optional[dict],
    view_id:    int,
    view_tag:   str,
    out_dir:    Optional[Path],
    eps_log:    float = 1e-3,
):
    H, W = pred.shape[-2], pred.shape[-1]
    Hc, Wc = resolve_crop(layout, view_id, H, W)

    # get mask for this sample
    if mask.ndim == 4:
        m = mask[idx, 0, :Hc, :Wc]
    elif mask.ndim == 2:
        m = mask[:Hc, :Wc]
    else:
        m = mask[0, :Hc, :Wc] if mask.ndim == 3 else np.ones((Hc, Wc))

    for c in channels:
        if c >= pred.shape[1]:
            continue

        t_img = truth[idx, c, :Hc, :Wc].astype(np.float64)
        p_img = pred[idx, c,  :Hc, :Wc].astype(np.float64)
        active = m > 0.5

        valid_t = t_img[active]
        valid_p = p_img[active]
        if len(valid_t) == 0:
            continue

        # determine the original channel index for choosing error type
        orig_c = y_indices[c] if c < len(y_indices) else c

        # ── compute relative error ───────────────────────────────────────
        if int(orig_c) in POS_CHANNELS:
            # for POS channels: log-space error is most informative
            # |log10(pred+eps) - log10(truth+eps)| expressed as factor
            log_p = np.log10(np.maximum(p_img, 0.0) + eps_log)
            log_t = np.log10(np.maximum(t_img, 0.0) + eps_log)
            err_img = np.abs(log_p - log_t) * m
            err_label = "Log₁₀ Error"
            err_cmap  = "inferno"
            # also compute percentage for the title
            rel_pct = np.abs(p_img - t_img) / np.maximum(np.abs(t_img), 1e-30) * 100.0
            median_pct = float(np.median(rel_pct[active])) if active.any() else 0.0
        else:
            # for SIGNED channels: relative error in %
            ref = np.maximum(np.abs(t_img), 1e-30)
            err_img = (np.abs(p_img - t_img) / ref * 100.0) * m
            err_label = "Relative Error (%)"
            err_cmap  = "inferno"
            median_pct = float(np.median(err_img[active])) if active.any() else 0.0

        # ── shared color range for GT and pred ───────────────────────────
        vmin = min(valid_t.min(), valid_p.min())
        vmax = max(valid_t.max(), valid_p.max())

        # mask out inactive region for display
        t_disp = t_img * m
        p_disp = p_img * m

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = _imshow(axes[0], t_disp, "Ground Truth", vmin=vmin, vmax=vmax)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = _imshow(axes[1], p_disp, "Prediction", vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = _imshow(axes[2], err_img, err_label, cmap=err_cmap)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # title with median relative error
        name = ch_name(orig_c)
        if int(orig_c) in POS_CHANNELS:
            mean_log_err = float(err_img[active].mean()) if active.any() else 0.0
            fig.suptitle(
                f"Sample {idx} | {name} | {view_tag} | "
                f"log₁₀-MAE = {mean_log_err:.3f}  (median rel = {median_pct:.1f}%)",
                fontsize=11,
            )
        else:
            fig.suptitle(
                f"Sample {idx} | {name} | {view_tag} | "
                f"median relative error = {median_pct:.1f}%",
                fontsize=11,
            )

        path = (out_dir / f"error_map_idx{idx:04d}_{name}_{view_tag}.png"
                if out_dir else None)
        _save_or_show(fig, path)


# ═════════════════════════════════════════════════════════════════════════
# PLOT 2: Per-channel metrics bar charts
# ═════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(
    metrics:  dict,
    out_dir:  Optional[Path],
):
    """
    Bar charts of per-channel metrics.
    Produces three figures per view + three global figures:
      1. Log-space MAE  (the metric the model learns in)
      2. Mean Relative Error (MRE)
      3. Physical MAE / RMSE
    """
    per_view = metrics.get("per_view", {})
    if not per_view:
        print("  No per-view metrics found, skipping.")
        return

    y_indices = metrics.get("y_indices", [])

    # ── Per-view plots ───────────────────────────────────────────────────
    for view_tag, vm in per_view.items():
        names = [ch_name(c) for c in y_indices] if y_indices else \
                [f"ch{i}" for i in range(len(vm.get("mae_per_channel", [])))]
        n = len(names)
        if n == 0:
            continue
        x = np.arange(n)

        # --- 1. Log-space MAE ---
        log_mae = vm.get("log_mae_per_channel")
        if log_mae and len(log_mae) == n:
            fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
            vals = [v if v is not None and np.isfinite(v) else 0 for v in log_mae]
            colors = ["#2ecc71" if v < 0.3 else "#e67e22" if v < 1.0 else "#e74c3c"
                       for v in vals]
            ax.bar(x, vals, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Log₁₀-space MAE")
            ax.set_title(f"Log-space MAE — {view_tag} | avg={vm.get('log_mae_avg', 0):.4f}\n"
                         f"(green < 0.3 ≈ 2× | orange < 1.0 ≈ 10× | red > 1.0)")
            ax.axhline(0.3, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax.grid(axis="y", alpha=0.3)
            _save_or_show(fig, out_dir / f"log_mae_{view_tag}.png" if out_dir else None)

        # --- 2. Mean Relative Error ---
        mre = vm.get("mre_per_channel")
        if mre and len(mre) == n:
            fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
            vals = [v if v is not None and np.isfinite(v) else 0 for v in mre]
            pcts = [v * 100 for v in vals]  # convert to %
            colors = ["#2ecc71" if v < 0.1 else "#e67e22" if v < 0.5 else "#e74c3c"
                       for v in vals]
            ax.bar(x, pcts, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Mean Relative Error (%)")
            ax.set_title(f"MRE — {view_tag} | avg={vm.get('mre_avg', 0) * 100:.2f}%\n"
                         f"(green < 10% | orange < 50% | red > 50%)")
            ax.axhline(10, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)
            _save_or_show(fig, out_dir / f"mre_{view_tag}.png" if out_dir else None)

        # --- 3. Physical MAE / RMSE ---
        mae_list  = vm.get("mae_per_channel", [])
        rmse_list = vm.get("rmse_per_channel", [])
        if mae_list and len(mae_list) == n:
            mae_clean  = [v if v is not None and np.isfinite(v) else 0 for v in mae_list]
            rmse_clean = [v if v is not None and np.isfinite(v) else 0 for v in rmse_list]

            fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 0.6), 8))

            axes[0].bar(x, mae_clean, color="steelblue")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            axes[0].set_ylabel("MAE (physical)")
            axes[0].set_title(f"Per-channel Physical MAE — {view_tag} | "
                              f"avg={vm.get('mae_avg', 0):.4g}")
            axes[0].set_yscale("log")

            axes[1].bar(x, rmse_clean, color="coral")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            axes[1].set_ylabel("RMSE (physical)")
            axes[1].set_title(f"Per-channel Physical RMSE — {view_tag} | "
                              f"avg={vm.get('rmse_avg', 0):.4g}")
            axes[1].set_yscale("log")

            _save_or_show(fig, out_dir / f"physical_metrics_{view_tag}.png" if out_dir else None)

    # ── Global summary (averaged across views) ───────────────────────────
    n_views = len(per_view)
    if n_views == 0:
        return

    first_vm = next(iter(per_view.values()))
    n = len(first_vm.get("mae_per_channel", []))
    names = [ch_name(c) for c in y_indices] if y_indices else [f"ch{i}" for i in range(n)]
    x = np.arange(n)

    def _avg_field(field):
        arrs = [vm.get(field, []) for vm in per_view.values()]
        if not arrs or not all(len(a) == n for a in arrs):
            return None
        # replace None with 0
        cleaned = [[v if v is not None and np.isfinite(v) else 0 for v in a] for a in arrs]
        return np.mean(cleaned, axis=0)

    # --- Global Log MAE ---
    avg_log_mae = _avg_field("log_mae_per_channel")
    if avg_log_mae is not None:
        fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
        colors = ["#2ecc71" if v < 0.3 else "#e67e22" if v < 1.0 else "#e74c3c"
                   for v in avg_log_mae]
        ax.bar(x, avg_log_mae, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Log₁₀-space MAE")
        ax.set_title(f"Log-space MAE — avg across views | "
                     f"global={metrics.get('log_mae_avg_global', 0):.4f}\n"
                     f"(green < 0.3 ≈ 2× | orange < 1.0 ≈ 10× | red > 1.0)")
        ax.axhline(0.3, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        _save_or_show(fig, out_dir / "log_mae_global.png" if out_dir else None)

    # --- Global MRE ---
    avg_mre = _avg_field("mre_per_channel")
    if avg_mre is not None:
        fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
        pcts = avg_mre * 100
        colors = ["#2ecc71" if v < 10 else "#e67e22" if v < 50 else "#e74c3c"
                   for v in pcts]
        ax.bar(x, pcts, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Relative Error (%)")
        ax.set_title(f"MRE — avg across views | "
                     f"global={metrics.get('mre_avg_global', 0) * 100:.2f}%\n"
                     f"(green < 10% | orange < 50% | red > 50%)")
        ax.axhline(10, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        _save_or_show(fig, out_dir / "mre_global.png" if out_dir else None)

    # --- Global Physical MAE / RMSE ---
    avg_mae  = _avg_field("mae_per_channel")
    avg_rmse = _avg_field("rmse_per_channel")
    if avg_mae is not None and avg_rmse is not None:
        fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 0.6), 8))

        axes[0].bar(x, avg_mae, color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("MAE (physical)")
        axes[0].set_title(f"Physical MAE — avg across views | "
                          f"global={metrics.get('mae_avg_global', 0):.4g}")
        axes[0].set_yscale("log")

        axes[1].bar(x, avg_rmse, color="coral")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("RMSE (physical)")
        axes[1].set_title(f"Physical RMSE — avg across views | "
                          f"global={metrics.get('rmse_avg_global', 0):.4g}")
        axes[1].set_yscale("log")

        _save_or_show(fig, out_dir / "physical_metrics_global.png" if out_dir else None)


# ═════════════════════════════════════════════════════════════════════════
# PLOT 3: Training loss curve
# ═════════════════════════════════════════════════════════════════════════

def plot_loss_curve(
    metrics_hist: dict,
    out_dir:      Optional[Path],
):
    train = metrics_hist.get("train", [])
    if not train:
        print("  No training metrics found, skipping.")
        return

    epochs = [e["epoch"] for e in train]
    losses = [e["loss"]  for e in train if "loss" in e]
    mae_avgs = [e.get("mae_avg") for e in train]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.plot(epochs[:len(losses)], losses, "b-", label="Train loss", alpha=0.8, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss (channel-balanced MAE)", color="b")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, which="both", alpha=0.3)

    valid_mae = [(ep, v) for ep, v in zip(epochs, mae_avgs)
                 if v is not None and np.isfinite(v)]
    if valid_mae:
        ax2 = ax1.twinx()
        ax2.plot(*zip(*valid_mae), "r--", label="MAE avg", alpha=0.7, linewidth=1)
        ax2.set_ylabel("MAE avg", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    fig.suptitle("Training curves — VelocityUNet regression", fontsize=12)

    path = out_dir / "loss_curve.png" if out_dir else None
    _save_or_show(fig, path)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Visualize VelocityUNet regression results."
    )
    ap.add_argument("mode", choices=[
        "error-maps", "metrics-bar", "loss-curve", "all"
    ], help="Which plot(s) to generate.")

    ap.add_argument("--run_dir", required=True,
                    help="Run directory (contains config.json, metrics.json, infer_test/)")
    ap.add_argument("--tensor_prefix", default=None,
                    help="Tensor prefix for ground truth (needed for error-maps)")
    ap.add_argument("--split",    default="test")
    ap.add_argument("--idx",      type=int, default=0,
                    help="Simulation index to visualize")
    ap.add_argument("--channels", default="0,1",
                    help='Channel indices to plot, e.g. "0,1" or "all"')
    ap.add_argument("--views",    default="0,1,2",
                    help='Views to plot, e.g. "0,1,2" or "0"')
    ap.add_argument("--infer_subdir", default="infer_test",
                    help="Subdirectory under run_dir containing predictions")
    ap.add_argument("--out_dir",  default=None,
                    help="Output directory for PNGs (default: run_dir/viz)")
    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    infer_dir = run_dir / args.infer_subdir
    out_dir   = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load config
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    y_indices = config.get("y_indices", [])
    eps       = config.get("eps", 1e-3)
    views_to_plot = [int(v.strip()) for v in args.views.split(",")]

    modes = [args.mode] if args.mode != "all" else ["error-maps", "metrics-bar", "loss-curve"]

    # ── Error maps ───────────────────────────────────────────────────────
    if "error-maps" in modes:
        print("\n--- error-maps ---")

        tp = args.tensor_prefix or config.get("tensor_prefix")
        if tp is None:
            print("  Need --tensor_prefix for error maps. Skipping.")
        else:
            layout = load_layout(tp)

            first_pred_path = None
            for v in views_to_plot:
                p = infer_dir / f"pred_Y_img_{args.split}_view{v}.npy"
                if p.exists():
                    first_pred_path = p
                    break
            if first_pred_path is None:
                print("  No prediction files found. Run inference first.")
            else:
                peek = np.load(first_pred_path, mmap_mode="r")
                max_c = peek.shape[1]
                channels = parse_channels(args.channels, max_c)

                for v in views_to_plot:
                    view_tag = f"view{v}"
                    pred_path = infer_dir / f"pred_Y_img_{args.split}_{view_tag}.npy"
                    if not pred_path.exists():
                        print(f"  Prediction file missing: {pred_path}")
                        continue

                    pred = np.load(pred_path, mmap_mode="r")

                    y_path = Path(f"{tp}_{view_tag}_Y_img_{args.split}.npy")
                    x_path = Path(f"{tp}_{view_tag}_X_img_{args.split}.npy")

                    if not y_path.exists():
                        print(f"  Ground truth missing: {y_path}")
                        continue

                    truth_full = np.load(y_path, mmap_mode="r")
                    yi = y_indices if y_indices else list(range(truth_full.shape[1]))
                    truth = np.array(truth_full[:, yi, :, :], dtype=np.float32)

                    if x_path.exists():
                        X_test = np.load(x_path, mmap_mode="r")
                        mask = X_test[:, 0:1, :, :].astype(np.float32)
                    else:
                        mask = np.ones((pred.shape[0], 1, pred.shape[2], pred.shape[3]),
                                       dtype=np.float32)

                    print(f"  Plotting {view_tag} …")
                    plot_error_maps(
                        pred, truth, mask,
                        idx=args.idx,
                        channels=channels,
                        y_indices=yi,
                        layout=layout,
                        view_id=v,
                        view_tag=view_tag,
                        out_dir=out_dir,
                        eps_log=eps,
                    )

    # ── Metrics bar ──────────────────────────────────────────────────────
    if "metrics-bar" in modes:
        print("\n--- metrics-bar ---")
        metrics_path = infer_dir / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                test_metrics = json.load(f)
            plot_metrics_bar(test_metrics, out_dir)
        else:
            print(f"  No test_metrics.json found at {metrics_path}")

    # ── Loss curve ───────────────────────────────────────────────────────
    if "loss-curve" in modes:
        print("\n--- loss-curve ---")
        hist_path = run_dir / "metrics.json"
        if hist_path.exists():
            with open(hist_path) as f:
                metrics_hist = json.load(f)
            plot_loss_curve(metrics_hist, out_dir)
        else:
            print(f"  No metrics.json found at {hist_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
