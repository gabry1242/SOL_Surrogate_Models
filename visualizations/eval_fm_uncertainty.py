#!/usr/bin/env python3
"""
eval_fm_uncertainty.py
──────────────────────────────────────────────────────────────────────────────
Evaluates whether the Flow-Matching model's stochastic uncertainty
(the per-pixel std across K independent samples) is *informative* and *useful*,
addressing supervisor feedback point 3.

WHY THIS IS FUNDAMENTALLY DIFFERENT FROM A SIMPLE ENSEMBLE
──────────────────────────────────────────────────────────
A deep ensemble varies model WEIGHTS across runs (epistemic / parameter
uncertainty).  The FM stochastic sampler keeps the weights fixed and varies
only the initial noise draw z₀ ~ N(0,I).  Each run is a genuine sample from
the learned conditional distribution p(y|x).  The resulting σ is therefore
a model-internal claim: "multiple plasma field states consistent with x are
plausible, and their spread is σ."

That claim is testable with three complementary lenses:

  1. CALIBRATION (reliability diagram + ECE)
     Under a Gaussian assumption N(μ, σ²), 68% of true values should fall
     within ±1σ, 95% within ±2σ, etc.  Calibration quantifies whether the
     model's confidence intervals contain truth with the claimed probability.
     We compare to a flat-σ baseline (σ_flat = RMSE per channel), which
     always has zero information about *where* the model is uncertain.

  2. UNCERTAINTY–ERROR CORRELATION (Spearman ρ)
     If σ is informative, cells with large σ should have large |ŷ - y_true|.
     Spearman ρ between pixel-wise σ and pixel-wise |error| measures this
     without assuming normality.

  3. PROPER SCORING RULES (NLL and Gaussian CRPS)
     Both NLL and CRPS reward distributions that are simultaneously accurate
     AND sharp.  A flat σ = RMSE baseline can achieve good calibration but
     zero sharpness.  If FM beats the flat baseline in NLL / CRPS, it means
     its σ carries *useful* information beyond the global error level.

PREREQUISITE
────────────
Run inference with --n_samples > 1 (recommend ≥ 10) and --integrator stochastic.
This saves, per view:
    infer_dir/pred_Y_img_{split}_view{v}.npy   ← mean over K samples
    infer_dir/pred_Y_std_{split}_view{v}.npy   ← std  over K samples

Usage
─────
python eval_fm_uncertainty.py \\
    --run_dir       scripts/runs/my_fm_run \\
    --tensor_prefix scripts/tensor/train/global3 \\
    --split         test \\
    --infer_subdir  infer_test \\
    --channels      0,1 \\
    --out_dir       scripts/runs/my_fm_run/viz_uncertainty
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
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Channel metadata
# ─────────────────────────────────────────────────────────────────────────────

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CHANNEL_NAMES: List[str] = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)
POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


def ch_name(c: int) -> str:
    return CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_channels(s: str, max_c: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(max_c))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_layout(tensor_prefix: str) -> Optional[dict]:
    p = Path(f"{tensor_prefix}_layout_map_3views.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: int(z[k]) for k in z.files if k in ["W0","H0","W1","H1","W2","H2","Hmax","Wmax"]}


def resolve_crop(layout: Optional[dict], view_id: int, H: int, W: int) -> Tuple[int, int]:
    if layout is None:
        return H, W
    hk, wk = f"H{view_id}", f"W{view_id}"
    if hk in layout and wk in layout:
        return min(layout[hk], H), min(layout[wk], W)
    return H, W


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                 mask: np.ndarray, eps: float = 1e-8) -> float:
    """Mean NLL under N(mu, sigma²) over active cells.

    NLL = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
    Lower is better.  A flat baseline with σ = RMSE achieves the first
    term only; FM should win by adapting σ spatially.
    """
    active = mask > 0.5
    s = np.maximum(sigma[active], eps)
    err = y[active] - mu[active]
    nll_vals = 0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(s) + (err / s) ** 2)
    return float(np.mean(nll_vals))


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                  mask: np.ndarray, eps: float = 1e-8) -> float:
    """Continuous Ranked Probability Score for N(mu, sigma²).

    Closed form: CRPS = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - mu) / sigma.
    Lower is better, strictly proper scoring rule.
    """
    from scipy.stats import norm as sp_norm
    active = mask > 0.5
    s = np.maximum(sigma[active], eps)
    z = (y[active] - mu[active]) / s
    crps_vals = s * (z * (2.0 * sp_norm.cdf(z) - 1.0) + 2.0 * sp_norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_vals))


def spearman_sigma_error(sigma: np.ndarray, error: np.ndarray,
                         mask: np.ndarray) -> float:
    """Spearman rank correlation between per-pixel σ and |error| over active cells."""
    active = mask > 0.5
    rho, _ = stats.spearmanr(sigma[active].ravel(), error[active].ravel())
    return float(rho)


def calibration_curve(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                      mask: np.ndarray, n_bins: int = 20,
                      eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a reliability / calibration curve.

    For confidence levels α ∈ (0,1), compute the empirical coverage:
        P(|y - μ| < z_α * σ)
    where z_α is the α-quantile of the half-normal (i.e. Φ⁻¹((1+α)/2)).

    Returns (alphas, empirical_coverages).
    A perfectly calibrated model lies on the diagonal.
    """
    from scipy.stats import norm as sp_norm
    active = mask > 0.5
    mu_a    = mu[active].ravel()
    sigma_a = np.maximum(sigma[active].ravel(), eps)
    y_a     = y[active].ravel()

    alphas   = np.linspace(0.05, 0.99, n_bins)
    coverage = np.zeros(n_bins)

    for i, alpha in enumerate(alphas):
        z_alpha = sp_norm.ppf((1.0 + alpha) / 2.0)
        within  = np.abs(y_a - mu_a) < z_alpha * sigma_a
        coverage[i] = within.mean()

    return alphas, coverage


def ece_from_curve(alphas: np.ndarray, coverage: np.ndarray) -> float:
    """Expected Calibration Error = mean |coverage - alpha|."""
    return float(np.mean(np.abs(coverage - alphas)))


def stratified_mae_by_uncertainty(
    sigma: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide active pixels into n_bins uncertainty quantiles, compute mean |error| per bin.

    Returns (bin_centers, mean_abs_errors).
    If σ is informative, mean error should rise monotonically with σ.
    """
    active = mask > 0.5
    s = sigma[active].ravel()
    e = error[active].ravel()
    quantile_edges = np.quantile(s, np.linspace(0.0, 1.0, n_bins + 1))
    bin_centers = 0.5 * (quantile_edges[:-1] + quantile_edges[1:])
    mean_errors = np.zeros(n_bins)
    for k in range(n_bins):
        lo, hi = quantile_edges[k], quantile_edges[k + 1]
        sel = (s >= lo) & (s <= hi) if k == n_bins - 1 else (s >= lo) & (s < hi)
        mean_errors[k] = float(e[sel].mean()) if sel.any() else 0.0
    return bin_centers, mean_errors


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(
    fm_alphas: np.ndarray, fm_coverage: np.ndarray,
    flat_alphas: np.ndarray, flat_coverage: np.ndarray,
    fm_ece: float, flat_ece: float,
    channel_label: str, view_tag: str, out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(fm_alphas,   fm_coverage,   "b-o", ms=4, lw=1.8,
            label=f"FM  (ECE={fm_ece:.3f})")
    ax.plot(flat_alphas, flat_coverage, "r--s", ms=4, lw=1.4,
            label=f"Flat σ=RMSE (ECE={flat_ece:.3f})")
    ax.set_xlabel("Expected confidence α")
    ax.set_ylabel("Empirical coverage P(|y-μ| < z_α·σ)")
    ax.set_title(f"Calibration — {channel_label} | {view_tag}\n"
                 f"FM ECE={fm_ece:.3f}  vs  Flat ECE={flat_ece:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out_dir / f"calibration_{channel_label}_{view_tag}.png")


def plot_sigma_error_scatter(
    sigma: np.ndarray, error: np.ndarray, mask: np.ndarray,
    rho: float, channel_label: str, view_tag: str, out_dir: Path,
    max_pts: int = 5000,
) -> None:
    active = mask > 0.5
    s = sigma[active].ravel()
    e = error[active].ravel()
    if len(s) > max_pts:
        idx = np.random.choice(len(s), max_pts, replace=False)
        s, e = s[idx], e[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(s, e, s=4, alpha=0.3, c="steelblue", rasterized=True)
    ax.set_xlabel("σ (FM std across samples)")
    ax.set_ylabel("|error|  = |ŷ - y_true|")
    ax.set_title(f"σ vs |error| — {channel_label} | {view_tag}\n"
                 f"Spearman ρ = {rho:.3f}")
    ax.grid(alpha=0.3)
    _save(fig, out_dir / f"sigma_error_scatter_{channel_label}_{view_tag}.png")


def plot_stratified_mae(
    bin_centers: np.ndarray, mean_errors: np.ndarray,
    channel_label: str, view_tag: str, out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(bin_centers)), mean_errors, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels(
        [f"Q{k+1}" for k in range(len(bin_centers))], fontsize=8
    )
    ax.set_xlabel("Uncertainty quantile (Q1=low σ → Q10=high σ)")
    ax.set_ylabel("Mean |error|")
    ax.set_title(f"Stratified MAE by uncertainty — {channel_label} | {view_tag}\n"
                 f"(Monotone rise → σ is a good error predictor)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir / f"stratified_mae_{channel_label}_{view_tag}.png")


def plot_spatial_uncertainty(
    mu: np.ndarray, sigma: np.ndarray, truth: np.ndarray,
    error: np.ndarray, mask: np.ndarray,
    sim_idx: int, channel_label: str, view_tag: str, out_dir: Path,
) -> None:
    """Four-panel spatial map: GT | Prediction | σ | |error|."""
    m = mask[sim_idx, 0]
    t  = truth[sim_idx] * m
    p  = mu   [sim_idx] * m
    s  = sigma [sim_idx] * m
    e  = error [sim_idx] * m

    vmin = min(t[m > 0.5].min(), p[m > 0.5].min())
    vmax = max(t[m > 0.5].max(), p[m > 0.5].max())

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    kw = dict(aspect="auto", origin="lower")
    ims = [
        axes[0].imshow(t, cmap="viridis",  vmin=vmin, vmax=vmax, **kw),
        axes[1].imshow(p, cmap="viridis",  vmin=vmin, vmax=vmax, **kw),
        axes[2].imshow(s, cmap="plasma",  **kw),
        axes[3].imshow(e, cmap="inferno", **kw),
    ]
    titles = ["Ground truth", "FM mean (μ)", "FM std (σ)", "|error|  |ŷ-y|"]
    for ax, im, ti in zip(axes, ims, titles):
        ax.set_title(ti, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Spatial uncertainty — sim {sim_idx} | {channel_label} | {view_tag}",
        fontsize=11,
    )
    _save(fig, out_dir / f"spatial_uncertainty_{channel_label}_{view_tag}_sim{sim_idx}.png")


def plot_nll_crps_comparison(
    results: dict, out_dir: Path,
) -> None:
    """Bar chart comparing FM vs flat-σ baseline for NLL and CRPS, across channels."""
    channels = list(results.keys())
    fm_nll    = [results[c]["nll_fm"]   for c in channels]
    flat_nll  = [results[c]["nll_flat"] for c in channels]
    fm_crps   = [results[c]["crps_fm"]  for c in channels]
    flat_crps = [results[c]["crps_flat"] for c in channels]

    x = np.arange(len(channels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(channels) * 0.9 + 2), 5))

    for ax, fm_vals, flat_vals, ylabel, title in zip(
        axes,
        [fm_nll, fm_crps],
        [flat_nll, flat_crps],
        ["NLL (lower is better)", "CRPS (lower is better)"],
        ["Negative Log-Likelihood", "Gaussian CRPS"],
    ):
        ax.bar(x - w/2, fm_vals,   w, label="FM (adaptive σ)", color="steelblue", alpha=0.85)
        ax.bar(x + w/2, flat_vals, w, label="Flat σ=RMSE",     color="coral",     alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "FM uncertainty vs flat-σ baseline — proper scoring rules\n"
        "(FM below flat → FM uncertainty is more informative than constant σ)",
        fontsize=10,
    )
    _save(fig, out_dir / "nll_crps_comparison.png")


def plot_summary_dashboard(summary_rows: list, out_dir: Path) -> None:
    """Summary table figure of key metrics across channels and views."""
    if not summary_rows:
        return
    import matplotlib.patches as mpatches

    keys = ["channel", "view", "spearman_rho", "ece_fm", "ece_flat",
            "nll_fm", "nll_flat", "crps_fm", "crps_flat"]
    n = len(summary_rows)

    fig, ax = plt.subplots(figsize=(16, max(3, n * 0.4 + 2)))
    ax.axis("off")

    header = ["Channel", "View", "Spearman ρ", "ECE(FM)", "ECE(Flat)",
              "NLL(FM)", "NLL(Flat)", "CRPS(FM)", "CRPS(Flat)"]
    rows = [
        [
            r["channel"], r["view"],
            f"{r['spearman_rho']:.3f}",
            f"{r['ece_fm']:.3f}", f"{r['ece_flat']:.3f}",
            f"{r['nll_fm']:.3f}", f"{r['nll_flat']:.3f}",
            f"{r['crps_fm']:.4g}", f"{r['crps_flat']:.4g}",
        ]
        for r in summary_rows
    ]

    tbl = ax.table(cellText=rows, colLabels=header,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    # color cells green/red based on whether FM beats flat
    for i, row in enumerate(summary_rows):
        for j, (fm_k, flat_k) in enumerate([
            ("ece_fm", "ece_flat"), ("nll_fm", "nll_flat"), ("crps_fm", "crps_flat")
        ]):
            col_idx = [3, 5, 7][j]   # header column
            fm_val   = row[fm_k]
            flat_val = row[flat_k]
            color = "#c8f7c8" if fm_val <= flat_val else "#f7c8c8"
            tbl[i + 1, col_idx].set_facecolor(color)

    green_patch = mpatches.Patch(color="#c8f7c8", label="FM ≤ flat (FM wins)")
    red_patch   = mpatches.Patch(color="#f7c8c8", label="FM > flat (flat wins)")
    ax.legend(handles=[green_patch, red_patch], loc="upper right")

    fig.suptitle(
        "Uncertainty quality summary  ·  FM stochastic σ vs flat-σ baseline\n"
        "Spearman ρ: how well σ predicts |error|  (higher = better)\n"
        "ECE / NLL / CRPS: lower = better calibration / sharpness",
        fontsize=10,
    )
    _save(fig, out_dir / "uncertainty_summary_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate FM stochastic uncertainty quality."
    )
    ap.add_argument("--run_dir",       required=True,
                    help="FM run directory (contains config.json)")
    ap.add_argument("--tensor_prefix", required=True,
                    help="Tensor prefix for the split, e.g. scripts/tensor/test/global3")
    ap.add_argument("--split",         default="test")
    ap.add_argument("--infer_subdir",  default="infer_test",
                    help="Subdirectory inside run_dir with prediction npy files")
    ap.add_argument("--channels",      default="0,1",
                    help='Output channel indices to evaluate, e.g. "0,1" or "all"')
    ap.add_argument("--views",         default="0,1,2")
    ap.add_argument("--sim_idx",       type=int, default=0,
                    help="Simulation index used for spatial uncertainty plot")
    ap.add_argument("--out_dir",       default=None,
                    help="Output directory for plots (default: run_dir/viz_uncertainty)")
    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    infer_dir = run_dir / args.infer_subdir
    out_dir   = Path(args.out_dir) if args.out_dir else run_dir / "viz_uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    y_indices = config.get("y_indices", [])

    layout = load_layout(args.tensor_prefix)

    views_to_eval = [int(v.strip()) for v in args.views.split(",")]

    summary_rows   = []
    per_ch_results = {}   # keyed by ch_label → metrics averaged across views

    for v in views_to_eval:
        view_tag = f"view{v}"

        # ── Load prediction mean ──────────────────────────────────────────
        pred_mean_path = infer_dir / f"pred_Y_img_{args.split}_{view_tag}.npy"
        pred_std_path  = infer_dir / f"pred_Y_std_{args.split}_{view_tag}.npy"

        if not pred_mean_path.exists():
            print(f"  Prediction mean missing: {pred_mean_path} — skipping {view_tag}")
            continue
        if not pred_std_path.exists():
            print(f"  MISSING: {pred_std_path}")
            print(f"  Re-run inference with --n_samples > 1 and --integrator stochastic")
            print(f"  to generate per-pixel uncertainty estimates.  Skipping {view_tag}.")
            continue

        pred_mean = np.load(pred_mean_path, mmap_mode="r")   # (N, C, H, W)
        pred_std  = np.load(pred_std_path,  mmap_mode="r")   # (N, C, H, W)

        # ── Load ground truth ─────────────────────────────────────────────
        y_path = Path(f"{args.tensor_prefix}_{view_tag}_Y_img_{args.split}.npy")
        x_path = Path(f"{args.tensor_prefix}_{view_tag}_X_img_{args.split}.npy")

        if not y_path.exists():
            print(f"  Ground truth missing: {y_path} — skipping {view_tag}")
            continue

        truth_full = np.load(y_path, mmap_mode="r")
        yi         = y_indices if y_indices else list(range(truth_full.shape[1]))
        truth      = np.array(truth_full[:, yi, :, :], dtype=np.float32)

        if x_path.exists():
            X_test = np.load(x_path, mmap_mode="r")
            mask   = X_test[:, 0:1, :, :].astype(np.float32)
        else:
            mask = np.ones((truth.shape[0], 1, truth.shape[2], truth.shape[3]),
                           dtype=np.float32)

        N, C, H, W = pred_mean.shape
        Hc, Wc = resolve_crop(layout, v, H, W)

        # crop to actual view content
        mu    = np.array(pred_mean[:, :, :Hc, :Wc], dtype=np.float64)
        sigma = np.array(pred_std [:, :, :Hc, :Wc], dtype=np.float64)
        truth_c = truth[:, :, :Hc, :Wc].astype(np.float64)
        mask_c  = mask [:, :, :Hc, :Wc].astype(np.float64)

        # parse channels
        channels = parse_channels(args.channels, C)

        print(f"\n{'─'*70}")
        print(f"  {view_tag}  | shape=({N},{C},{Hc},{Wc})")

        for c in channels:
            if c >= C:
                continue

            orig_c      = yi[c] if c < len(yi) else c
            ch_label    = ch_name(orig_c)

            mu_c    = mu   [:, c]               # (N, H, W)
            sigma_c = sigma[:, c]               # (N, H, W)
            truth_c_ch = truth_c[:, c]          # (N, H, W)
            mask_c_ch  = mask_c[:, 0]           # (N, H, W)  broadcast mask

            error_c = np.abs(mu_c - truth_c_ch)

            # ── Global mask (flatten across all sims) ──────────────────
            mu_flat    = mu_c.ravel()
            sigma_flat = sigma_c.ravel()
            truth_flat = truth_c_ch.ravel()
            error_flat = error_c.ravel()
            mask_flat  = mask_c_ch.ravel()

            # ── 1. Spearman ρ(σ, |error|) ──────────────────────────────
            rho = spearman_sigma_error(sigma_flat, error_flat, mask_flat)

            # ── 2. Calibration (FM σ) ───────────────────────────────────
            fm_alphas, fm_coverage = calibration_curve(mu_flat, sigma_flat,
                                                       truth_flat, mask_flat)
            fm_ece = ece_from_curve(fm_alphas, fm_coverage)

            # ── 3. Flat-σ baseline (σ = per-channel RMSE) ─────────────
            active = mask_flat > 0.5
            rmse_flat = float(np.sqrt(np.mean((mu_flat[active] - truth_flat[active]) ** 2)))
            sigma_flat_arr = np.full_like(sigma_flat, fill_value=rmse_flat)

            flat_alphas, flat_coverage = calibration_curve(mu_flat, sigma_flat_arr,
                                                           truth_flat, mask_flat)
            flat_ece = ece_from_curve(flat_alphas, flat_coverage)

            # ── 4. NLL ──────────────────────────────────────────────────
            nll_fm   = gaussian_nll(mu_flat, sigma_flat,     truth_flat, mask_flat)
            nll_flat = gaussian_nll(mu_flat, sigma_flat_arr, truth_flat, mask_flat)

            # ── 5. CRPS ─────────────────────────────────────────────────
            crps_fm   = gaussian_crps(mu_flat, sigma_flat,     truth_flat, mask_flat)
            crps_flat = gaussian_crps(mu_flat, sigma_flat_arr, truth_flat, mask_flat)

            # ── 6. Stratified MAE by σ quantile ─────────────────────────
            bin_centers, mean_mae_per_bin = stratified_mae_by_uncertainty(
                sigma_flat, error_flat, mask_flat
            )

            print(f"\n    Channel {ch_label} ({view_tag})")
            print(f"      Spearman ρ(σ, |err|)  = {rho:.4f}  (>0 → σ tracks error)")
            print(f"      ECE  FM / Flat         = {fm_ece:.4f} / {flat_ece:.4f}")
            print(f"      NLL  FM / Flat         = {nll_fm:.4f} / {nll_flat:.4f}")
            print(f"      CRPS FM / Flat         = {crps_fm:.4g} / {crps_flat:.4g}")

            # ── Plots ────────────────────────────────────────────────────
            plot_calibration(
                fm_alphas, fm_coverage, flat_alphas, flat_coverage,
                fm_ece, flat_ece, ch_label, view_tag, out_dir,
            )
            plot_sigma_error_scatter(
                sigma_flat, error_flat, mask_flat,
                rho, ch_label, view_tag, out_dir,
            )
            plot_stratified_mae(
                bin_centers, mean_mae_per_bin, ch_label, view_tag, out_dir,
            )
            if args.sim_idx < N:
                plot_spatial_uncertainty(
                    mu_c, sigma_c, truth_c_ch, error_c, mask_c_ch[..., np.newaxis],
                    args.sim_idx, ch_label, view_tag, out_dir,
                )

            row = dict(
                channel=ch_label, view=view_tag,
                spearman_rho=rho,
                ece_fm=fm_ece, ece_flat=flat_ece,
                nll_fm=nll_fm, nll_flat=nll_flat,
                crps_fm=crps_fm, crps_flat=crps_flat,
                rmse_baseline=rmse_flat,
            )
            summary_rows.append(row)

            key = ch_label
            if key not in per_ch_results:
                per_ch_results[key] = []
            per_ch_results[key].append(row)

    # ── Cross-channel NLL/CRPS bar chart (averaged across views) ─────────
    if per_ch_results:
        def _avg(rows, k):
            return float(np.mean([r[k] for r in rows]))

        agg = {ch: {k: _avg(rows, k) for k in
                    ["spearman_rho","ece_fm","ece_flat","nll_fm","nll_flat","crps_fm","crps_flat"]}
               for ch, rows in per_ch_results.items()}
        plot_nll_crps_comparison(agg, out_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    plot_summary_dashboard(summary_rows, out_dir)

    # ── Save JSON metrics ─────────────────────────────────────────────────
    metrics_out = {
        "per_channel_view": summary_rows,
        "per_channel_averaged": {
            ch: {k: _avg(rows, k) for k in rows[0].keys() if k not in ("channel","view")}
            for ch, rows in per_ch_results.items()
        } if per_ch_results else {},
    }
    json_path = out_dir / "uncertainty_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics JSON: {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
