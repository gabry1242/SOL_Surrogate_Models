#!/usr/bin/env python3
"""
scan_gas_puff_hidden.py
────────────────────────────────────────────────────────────────────────────
Identical to scan_gas_puff.py but adapted for the SYNTHETIC dataset in which
D_perp and Chi_perp are HIDDEN inputs (not present in X_tmp / X_legacy).

Key differences vs the original script
───────────────────────────────────────
  • build_x_img_batch  : reads n_params from params_arr.shape[1] instead of
                         hardcoding range(8).  Works for any number of inputs.
  • build_params_2d /
    build_params_1d    : build a 6-column params array
                         [R, B, Pin, D_puff, N_puff, Dcore].
                         D_perp and Chi_perp columns are gone.
  • CLI               : --D_perp and --Chi_perp arguments removed.
  • Sanity check      : only checks the 6 visible parameters.
  • Dead code removed : duplicate unreachable return block that was present in
                        the original scan_to_te_mesh function.

Everything else (inference, plotting, OOD scan, KDE, violin, …) is unchanged.

Outputs
───────
  fig_10_1A_te_ot_2D.png           2D heatmap of Te,ot over (D_puff, N_puff)
  fig_10_1B_te_1D_scan.png         3-panel: mean + std + CV vs D_puff
  fig_10_1B_uncertainty_sidebyside.png  side-by-side quick overview
  fig_uncertainty_2D.png           std and CV maps over 2D grid
  fig_sample_kde_1D.png            KDE at key D_puff points (detects bimodality)
  fig_violin_1D.png                violin plot — all samples at every D_puff
  fig_kde_convergence.png          KDE shape vs n_samples (convergence check)
  fig_scatter_all_samples.png      all sample dots at every D_puff
  fig_peak_convergence.png         peak positions vs n_samples
  fig_ood_extension.png            OOD scan extending D_puff beyond training range
  fig_te_na_anticorrelation.png    Te vs density anti-correlation (if na in model)
  *.npy                            raw arrays for further analysis

Usage
─────
python scan_gas_puff_hidden.py \\
    --scripts_dir  thesis/scripts \\
    --checkpoint   scripts/runs/<synthetic_run>/checkpoint_best.pt \\
    --tensor_prefix scripts/tensor/train/global3 \\
    --layout_path  scripts/tensor/train/global3_layout_map_3views.npz \\
    --unroller     scripts/tensor/unrolled_strip_clockwise_adjpreserve.py \\
    --b2fgmtry     geometry/b2fgmtry \\
    --out_dir      scripts/runs/<synthetic_run>/gas_puff_scan \\
    --n_dpuff 40  --n_npuff 35  --n_samples 50 \\
    --integrator stochastic

Quick sanity check before running — verify c_in matches 6 params:
    python -c "import torch; c=torch.load('checkpoint_best.pt',map_location='cpu'); print('c_in:', c['c_in'])"
    # expect 7 (mask + 6 params) or 8 (mask + 6 params + 1 geometry channel)
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
PARENT_DIR  = SCRIPTS_DIR.parent
UTIL_DIR    = PARENT_DIR.parent
PARENT_DIR  = PARENT_DIR / "models" / "FM_fin"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_fm_fusvel_v8 import (
    VelocityUNet,
    VelocityUNetV8,
    LayoutInfo,
    stochastic_integrate_xpred,
    euler_integrate_xpred,
    euler_integrate_velocity,
    direct_forward,
    stochastic_integrate_xpred_3v,
    euler_integrate_xpred_3v,
    euler_integrate_velocity_3v,
    direct_forward_3v,
    inverse_transform_y,
    POS_CHANNELS,
    SIGNED_CHANNELS,
)
from reconstruct_mesh import (
    build_strip,
    compute_view_layouts,
    build_inverse_maps,
    reconstruct_to_mesh,
    SLOT_TO_SPECNAME,
    SLOT_TF,
)
from utils import Geometry


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_separatrix_indices(geo: Geometry) -> Tuple[int, int, int]:
    """Return (ix_ot, ix_omp, iy_sep)."""
    return geo.nx, geo.jxa, geo.jsep + 2


# ─────────────────────────────────────────────────────────────────────────────
# X_img construction for arbitrary scan parameters
#
# FIX: n_params is now read from params_arr.shape[1] instead of being
# hardcoded as 8.  This makes the function correct for both the original
# 8-parameter dataset and the 6-parameter synthetic dataset.
# ─────────────────────────────────────────────────────────────────────────────

def build_x_img_batch(
    params_arr: np.ndarray,          # (N_scan, n_params)  — 6 for synthetic, 8 for original
    mask_v:     np.ndarray,          # (H, W) static binary mask for this view
    geom_v:     Optional[np.ndarray],# (2, H, W) geometry channels, or None
    c_in:       int,
    x_mean:     np.ndarray,          # (c_in,)
    x_std:      np.ndarray,          # (c_in,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Build normalised X and raw mask tensors for a batch of scan inputs.

    Mirrors ViewXDataset.__getitem__ from the training script:
      channel 0       : static mask
      channels 1..n   : scalar_k × mask  (n = params_arr.shape[1])
      channels n+1..  : geometry centroids (optional)

    Returns
    -------
    X_norm : (N_scan, c_in, H, W) float32
    masks  : (N_scan, 1,    H, W) float32
    """
    N_scan   = params_arr.shape[0]
    n_params = params_arr.shape[1]          # ← FIX: was hardcoded as 8
    H, W     = mask_v.shape
    X_raw    = np.zeros((N_scan, c_in, H, W), dtype=np.float32)

    # Channel 0: mask
    X_raw[:, 0] = mask_v[None]

    # Channels 1..n_params: param_k × mask
    for k in range(n_params):               # ← FIX: was range(8)
        X_raw[:, 1 + k] = (
            params_arr[:, k, None, None].astype(np.float32) * mask_v[None]
        )

    # Channels n_params+1 .. n_params+2: geometry (same for all scan points)
    geom_start = 1 + n_params
    if geom_v is not None and c_in >= geom_start + 2:
        X_raw[:, geom_start:geom_start + 2] = geom_v[None]

    # Extract raw mask BEFORE normalisation
    masks = X_raw[:, 0:1].copy()   # (N_scan, 1, H, W)

    # Per-channel normalisation
    X_norm = (X_raw - x_mean[None, :, None, None]) / x_std[None, :, None, None]
    return X_norm.astype(np.float32), masks.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Inference: one view, all scan points, one stochastic sample
# ─────────────────────────────────────────────────────────────────────────────

def infer_view_samples(
    model:          VelocityUNet,
    X_norm:         np.ndarray,    # (N_scan, c_in, H, W)
    masks:          np.ndarray,    # (N_scan, 1, H, W)
    view_id:        int,
    c_out:          int,
    n_steps:        int,
    mode:           str,
    integrator:     str,
    y_indices:      list,
    pos_channels,
    signed_channels,
    y_mean:         np.ndarray,
    y_std:          np.ndarray,
    eps:            float,
    s_c:            np.ndarray,
    batch_size:     int,
    device:         torch.device,
) -> np.ndarray:
    """Run ONE stochastic sample for all N_scan inputs on one view."""
    N_scan = X_norm.shape[0]
    H, W   = X_norm.shape[2], X_norm.shape[3]
    preds_norm = np.zeros((N_scan, c_out, H, W), dtype=np.float32)

    for b0 in range(0, N_scan, batch_size):
        b1   = min(b0 + batch_size, N_scan)
        x_b  = torch.from_numpy(X_norm[b0:b1]).to(device)
        m_b  = torch.from_numpy(masks[b0:b1]).to(device)
        bsz  = x_b.shape[0]
        vids = torch.full((bsz,), view_id, dtype=torch.long, device=device)

        with torch.no_grad():
            if mode == "direct":
                pred = direct_forward(
                    model, x_b, m_b, vids, c_out=c_out, device=device)
            elif mode == "xpred" and integrator == "stochastic":
                pred = stochastic_integrate_xpred(
                    model, x_b, m_b, vids, c_out=c_out,
                    n_steps=n_steps, device=device)
            elif mode == "xpred":
                pred = euler_integrate_xpred(
                    model, x_b, m_b, vids, c_out=c_out,
                    n_steps=n_steps, device=device)
            else:  # cfm
                pred = euler_integrate_velocity(
                    model, x_b, m_b, vids, c_out=c_out,
                    n_steps=n_steps, device=device)

        preds_norm[b0:b1] = pred.cpu().numpy()

    preds_phys = np.empty_like(preds_norm)
    for b0 in range(0, N_scan, batch_size):
        b1 = min(b0 + batch_size, N_scan)
        pt = (preds_norm[b0:b1] * y_std[None, :, None, None]
              + y_mean[None, :, None, None])
        preds_phys[b0:b1] = inverse_transform_y(
            pt, y_indices, pos_channels, signed_channels, eps, s_c)
        preds_phys[b0:b1] *= masks[b0:b1]

    del preds_norm
    return preds_phys


# ─────────────────────────────────────────────────────────────────────────────
# Inference: all 3 views simultaneously (v9 / reinjection path)
# ─────────────────────────────────────────────────────────────────────────────

def infer_3views_samples(
    model,
    X0_norm, masks0,
    X1_norm, masks1,
    X2_norm, masks2,
    c_out:          int,
    n_steps:        int,
    mode:           str,
    integrator:     str,
    y_indices:      list,
    pos_channels,
    signed_channels,
    y_mean:         np.ndarray,
    y_std:          np.ndarray,
    eps:            float,
    s_c:            np.ndarray,
    batch_size:     int,
    device:         torch.device,
):
    """Run ONE sample for all N_scan inputs across ALL 3 views simultaneously."""
    N_scan = X0_norm.shape[0]
    H0, W0 = X0_norm.shape[2], X0_norm.shape[3]
    H1, W1 = X1_norm.shape[2], X1_norm.shape[3]
    H2, W2 = X2_norm.shape[2], X2_norm.shape[3]

    preds0_norm = np.zeros((N_scan, c_out, H0, W0), dtype=np.float32)
    preds1_norm = np.zeros((N_scan, c_out, H1, W1), dtype=np.float32)
    preds2_norm = np.zeros((N_scan, c_out, H2, W2), dtype=np.float32)

    for b0 in range(0, N_scan, batch_size):
        b1 = min(b0 + batch_size, N_scan)
        x0_b = torch.from_numpy(X0_norm[b0:b1]).to(device)
        m0_b = torch.from_numpy(masks0 [b0:b1]).to(device)
        x1_b = torch.from_numpy(X1_norm[b0:b1]).to(device)
        m1_b = torch.from_numpy(masks1 [b0:b1]).to(device)
        x2_b = torch.from_numpy(X2_norm[b0:b1]).to(device)
        m2_b = torch.from_numpy(masks2 [b0:b1]).to(device)

        with torch.no_grad():
            if mode == "direct":
                p0, p1, p2 = direct_forward_3v(
                    model, x0_b, m0_b, x1_b, m1_b, x2_b, m2_b,
                    c_out=c_out, device=device)
            elif mode == "xpred" and integrator == "stochastic":
                p0, p1, p2 = stochastic_integrate_xpred_3v(
                    model, x0_b, m0_b, x1_b, m1_b, x2_b, m2_b,
                    c_out=c_out, n_steps=n_steps, device=device)
            elif mode == "xpred":
                p0, p1, p2 = euler_integrate_xpred_3v(
                    model, x0_b, m0_b, x1_b, m1_b, x2_b, m2_b,
                    c_out=c_out, n_steps=n_steps, device=device)
            else:  # cfm
                p0, p1, p2 = euler_integrate_velocity_3v(
                    model, x0_b, m0_b, x1_b, m1_b, x2_b, m2_b,
                    c_out=c_out, n_steps=n_steps, device=device)

        preds0_norm[b0:b1] = p0.cpu().numpy()
        preds1_norm[b0:b1] = p1.cpu().numpy()
        preds2_norm[b0:b1] = p2.cpu().numpy()

    def _denorm(preds_norm, masks):
        preds_phys = np.empty_like(preds_norm)
        for b0 in range(0, N_scan, batch_size):
            b1 = min(b0 + batch_size, N_scan)
            pt = (preds_norm[b0:b1] * y_std[None, :, None, None]
                  + y_mean[None, :, None, None])
            preds_phys[b0:b1] = inverse_transform_y(
                pt, y_indices, pos_channels, signed_channels, eps, s_c)
            preds_phys[b0:b1] *= masks[b0:b1]
        return preds_phys

    p0_phys = _denorm(preds0_norm, masks0)
    p1_phys = _denorm(preds1_norm, masks1)
    p2_phys = _denorm(preds2_norm, masks2)
    del preds0_norm, preds1_norm, preds2_norm
    return p0_phys, p1_phys, p2_phys


# ─────────────────────────────────────────────────────────────────────────────
# Core scan runner: params → Te mesh statistics
# Uses Welford's online algorithm so only 2 × (N_scan, 104, 50) float64 arrays
# are held in RAM at any time instead of n_samples × 3 × (N_scan, C_out, H, W).
# ─────────────────────────────────────────────────────────────────────────────

def scan_to_te_mesh(
    params_arr, model, view_masks, view_geoms, c_in, c_out,
    x_mean, x_std, y_mean, y_std, s_c, eps,
    y_indices, pos_channels, signed_channels,
    j_Te, inv_maps, layout,
    mode, integrator, n_samples, n_steps, batch_size, device,
    j_extra: int = -1,
    is_v9: bool = False,
):
    """Run scan pipeline; optionally track a second channel (j_extra) for density."""
    N_scan      = params_arr.shape[0]
    track_extra = (j_extra >= 0 and j_extra != j_Te)

    print("  Pre-building X tensors for all views ...")
    view_X, view_M = {}, {}
    for v in range(3):
        view_X[v], view_M[v] = build_x_img_batch(
            params_arr, view_masks[v], view_geoms[v], c_in, x_mean, x_std)

    # Channel slice to keep before reconstruct_to_mesh
    if track_extra:
        keep_lo  = min(j_Te, j_extra)
        keep_hi  = max(j_Te, j_extra) + 1
        j_Te_sl  = j_Te    - keep_lo
        j_ex_sl  = j_extra - keep_lo
    else:
        keep_lo, keep_hi = j_Te, j_Te + 1
        j_Te_sl  = 0
        j_ex_sl  = -1

    wf_mean = np.zeros((N_scan, 104, 50), dtype=np.float64)
    wf_M2   = np.zeros((N_scan, 104, 50), dtype=np.float64)
    if track_extra:
        wf_mean_ex = np.zeros((N_scan, 104, 50), dtype=np.float64)
        wf_M2_ex   = np.zeros((N_scan, 104, 50), dtype=np.float64)

    te_ot_samp  = np.zeros((n_samples, N_scan), dtype=np.float32)
    te_omp_samp = np.zeros((n_samples, N_scan), dtype=np.float32)
    ex_ot_samp  = np.zeros((n_samples, N_scan), dtype=np.float32) if track_extra else None
    ex_omp_samp = np.zeros((n_samples, N_scan), dtype=np.float32) if track_extra else None

    for s in range(n_samples):
        print(f"  Sample {s+1}/{n_samples} ...", end="\r", flush=True)
        pred_views = {}

        if is_v9:
            p0, p1, p2 = infer_3views_samples(
                model,
                view_X[0], view_M[0],
                view_X[1], view_M[1],
                view_X[2], view_M[2],
                c_out=c_out, n_steps=n_steps,
                mode=mode, integrator=integrator,
                y_indices=y_indices, pos_channels=pos_channels,
                signed_channels=signed_channels,
                y_mean=y_mean, y_std=y_std, eps=eps, s_c=s_c,
                batch_size=batch_size, device=device,
            )
            for v, pred_full in enumerate([p0, p1, p2]):
                pred_views[f"view{v}"] = pred_full[:, keep_lo:keep_hi, :, :]
            del p0, p1, p2
        else:
            for v in range(3):
                pred_full = infer_view_samples(
                    model, view_X[v], view_M[v], view_id=v, c_out=c_out,
                    n_steps=n_steps, mode=mode, integrator=integrator,
                    y_indices=y_indices, pos_channels=pos_channels,
                    signed_channels=signed_channels,
                    y_mean=y_mean, y_std=y_std, eps=eps, s_c=s_c,
                    batch_size=batch_size, device=device,
                )
                pred_views[f"view{v}"] = pred_full[:, keep_lo:keep_hi, :, :]
                del pred_full

        mesh_out, _ = reconstruct_to_mesh(pred_views, inv_maps, layout)
        del pred_views

        te_s   = mesh_out[:, j_Te_sl, :, :].astype(np.float64)
        delta  = te_s - wf_mean
        wf_mean += delta / (s + 1)
        wf_M2   += delta * (te_s - wf_mean)

        te_ot_samp[s]  = te_s[:, inv_maps["_ix_ot"],  inv_maps["_iy_sep"]]
        te_omp_samp[s] = te_s[:, inv_maps["_ix_omp"], inv_maps["_iy_sep"]]

        if track_extra:
            ex_s = mesh_out[:, j_ex_sl, :, :].astype(np.float64)
            dx   = ex_s - wf_mean_ex
            wf_mean_ex += dx / (s + 1)
            wf_M2_ex   += dx * (ex_s - wf_mean_ex)
            ex_ot_samp[s]  = ex_s[:, inv_maps["_ix_ot"],  inv_maps["_iy_sep"]]
            ex_omp_samp[s] = ex_s[:, inv_maps["_ix_omp"], inv_maps["_iy_sep"]]
        del mesh_out

    print()
    te_mean = wf_mean.astype(np.float32)
    te_std  = np.sqrt(wf_M2 / max(n_samples - 1, 1)).astype(np.float32)
    if track_extra:
        ex_mean = wf_mean_ex.astype(np.float32)
        ex_std  = np.sqrt(wf_M2_ex / max(n_samples - 1, 1)).astype(np.float32)
    else:
        ex_mean = ex_std = None

    return te_mean, te_std, te_ot_samp, te_omp_samp, ex_mean, ex_std, ex_ot_samp, ex_omp_samp


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid builders — 6-parameter version (D_perp / Chi_perp removed)
#
# Column order matches X_legacy (synthetic dataset):
#   [R, B, Pin, D_puff, N_puff, Dcore]
#
# D_puff, N_puff and Dcore are stored as log10 in X_tmp / X_legacy.
# ─────────────────────────────────────────────────────────────────────────────

def build_params_2d(dpuff_arr, npuff_arr, fixed):
    """(N_dpuff × N_npuff, 6) params array for a 2D Cartesian grid."""
    ND, NN = len(dpuff_arr), len(npuff_arr)
    DG, NG = np.meshgrid(dpuff_arr, npuff_arr, indexing='ij')  # (ND, NN)
    D_flat = DG.flatten()
    N_flat = NG.flatten()
    params = np.empty((len(D_flat), 6), dtype=np.float64)   # ← was 8
    params[:, 0] = fixed['R']
    params[:, 1] = fixed['B']
    params[:, 2] = fixed['Pin']
    params[:, 3] = np.log10(D_flat)            # log10-encoded
    params[:, 4] = np.log10(N_flat)            # log10-encoded
    params[:, 5] = np.log10(fixed['Dcore'])    # log10-encoded
    # D_perp (col 6) and Chi_perp (col 7) are HIDDEN — not included
    return params.astype(np.float32), ND, NN


def build_params_1d(dpuff_arr, npuff_fixed, fixed):
    """(N_dpuff, 6) params array for a 1D D_puff scan at fixed N_puff."""
    N      = len(dpuff_arr)
    params = np.empty((N, 6), dtype=np.float64)              # ← was 8
    params[:, 0] = fixed['R']
    params[:, 1] = fixed['B']
    params[:, 2] = fixed['Pin']
    params[:, 3] = np.log10(dpuff_arr)         # log10-encoded
    params[:, 4] = np.log10(npuff_fixed)       # log10-encoded
    params[:, 5] = np.log10(fixed['Dcore'])    # log10-encoded
    # D_perp (col 6) and Chi_perp (col 7) are HIDDEN — not included
    return params.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# KDE helper
# ─────────────────────────────────────────────────────────────────────────────

def kde_log(samples_1d: np.ndarray, n_pts: int = 200):
    """Gaussian KDE in log10-space.  Returns (x_grid, density)."""
    log_s = np.log10(np.maximum(samples_1d, 1e-10))
    bw    = 1.06 * log_s.std() * len(log_s) ** (-0.2)
    bw    = max(bw, 0.01)
    x_log = np.linspace(log_s.min() - 0.5, log_s.max() + 0.5, n_pts)
    diff  = x_log[:, None] - log_s[None, :]
    density = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1)
    density /= (density.max() + 1e-30)
    return 10.0 ** x_log, density


# ─────────────────────────────────────────────────────────────────────────────
# Main scan runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scan(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print(
            "  ⚠️  WARNING: running on CPU.  Each UNet forward pass takes ~100ms on CPU\n"
            "     vs ~5-10ms on GPU.  With n_samples=50 and n_steps=20 this scan will\n"
            "     take several hours on CPU.  Pass --device cuda if a GPU is available."
        )

    # ── 1.  Load checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)

    c_in            = int(ckpt["c_in"])
    c_out           = int(ckpt["c_out"])
    base            = int(ckpt.get("base", 32))
    t_dim           = int(ckpt.get("t_dim", 128))
    y_indices       = [int(c) for c in ckpt["y_indices"]]
    pos_channels    = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))
    eps             = float(ckpt.get("eps", 1e-3))
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)
    mode   = str(ckpt.get("mode", "cfm"))
    if args.mode:
        mode = args.mode

    print(f"  Checkpoint : c_in={c_in}  c_out={c_out}  mode={mode}")
    print(f"  y_indices  : {y_indices}")

    # Validate c_in is consistent with 6 params (+ mask + optional geometry)
    expected_c_in_min = 1 + 6   # mask + 6 params (no geometry)
    expected_c_in_max = 1 + 6 + 2  # mask + 6 params + 2 geometry channels
    if not (expected_c_in_min <= c_in <= expected_c_in_max):
        print(
            f"\n  ⚠️  WARNING: c_in={c_in} is outside the expected range "
            f"[{expected_c_in_min}, {expected_c_in_max}] for a 6-parameter model.\n"
            f"     If this model was trained on the ORIGINAL 8-parameter dataset,\n"
            f"     use scan_gas_puff.py instead of this script.\n"
        )

    # Locate the Te channel (index 0) in the prediction array
    if 0 not in y_indices:
        raise ValueError(
            "Te (channel index 0) is not in the model's y_indices. "
            "This scan requires Te predictions."
        )
    j_Te = y_indices.index(0)
    print(f"  Te is at prediction position j_Te={j_Te}")

    # Density channel for anti-correlation analysis
    na_species_idx = args.na_species
    na_y_idx       = 2 + na_species_idx
    j_extra        = na_y_idx if na_y_idx in y_indices else -1
    _na_name = {0:'D0 neutral', 1:'D1 ion', 2:'N0', 3:'N1', 4:'N2',
                5:'N3', 6:'N4', 7:'N5', 8:'N6', 9:'N7'}.get(na_species_idx,
                f'species {na_species_idx}')
    if j_extra >= 0:
        print(f"  Density: na species {na_species_idx} ({_na_name}) → y_index={na_y_idx}, "
              f"j_extra={j_extra}")
    else:
        print(f"  na species {na_species_idx} (y_index {na_y_idx}) not in model outputs "
              f"— density plots disabled.  Model y_indices: {y_indices}")

    # ── 2.  Build model ──────────────────────────────────────────────────────
    saved_config = ckpt.get("config", {})
    is_v9 = saved_config.get("reinjection") is not None
    if is_v9:
        if not hasattr(args, "layout_path") or args.layout_path is None:
            raise ValueError(
                "This is a v9 (reinjection) checkpoint. "
                "Pass --layout_path pointing to *_layout_map_3views.npz."
            )
        print("  Detected v9 checkpoint — loading LayoutInfo for reinjection …")
        layout_info = LayoutInfo(args.layout_path)
        model = VelocityUNetV8(
            layout=layout_info, c_in=c_in, c_out=c_out,
            base=base, t_dim=t_dim, n_views=3,
        ).to(device)
        print("  Instantiated VelocityUNetV8 (reinjection active)")
    else:
        model = VelocityUNet(
            c_in=c_in, c_out=c_out, base=base, t_dim=t_dim, n_views=3
        ).to(device)
        print("  Instantiated VelocityUNet (v7, no reinjection)")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── 3.  Load geometry ────────────────────────────────────────────────────
    geo = Geometry(args.b2fgmtry)
    ix_ot, ix_omp, iy_sep = get_separatrix_indices(geo)
    print(f"  Geometry   : ix_ot={ix_ot}  ix_omp={ix_omp}  iy_sep={iy_sep}")

    # ── 4.  Load view masks and geometry channels ─────────────────────────────
    view_masks: Dict[int, np.ndarray] = {}
    view_geoms: Dict[int, Optional[np.ndarray]] = {}
    for v in range(3):
        x_path = Path(
            f"{args.tensor_prefix}_view{v}_X_img_{args.tensor_split}.npy"
        )
        if not x_path.exists():
            raise FileNotFoundError(f"Tensor not found: {x_path}")
        X_ref = np.load(x_path, mmap_mode="r")
        view_masks[v] = np.array(X_ref[0, 0], dtype=np.float32)
        # Geometry channels sit at positions (1 + n_params) and (1 + n_params + 1)
        # For the 6-param model that is channels 7 and 8; for the 8-param model
        # it was 9 and 10.  We use c_in to decide whether to load them.
        geom_start = 1 + 6   # mask + 6 params
        view_geoms[v] = (
            np.array(X_ref[0, geom_start:geom_start + 2], dtype=np.float32)
            if c_in >= geom_start + 2 else None
        )
        print(f"  Loaded view {v} mask: {view_masks[v].shape}  "
              f"(geometry={'yes' if view_geoms[v] is not None else 'no'})")

    # ── 5.  Build 3-view inverse mapping ─────────────────────────────────────
    layout  = dict(np.load(args.layout_path, allow_pickle=True))
    gap_px  = int(layout["gap_px"])

    unroller_path = Path(args.unroller)
    if unroller_path.suffix == ".py":
        sys.path.insert(0, str(unroller_path.parent.resolve()))
        mod_name = unroller_path.stem
    else:
        mod_name = args.unroller
    u            = importlib.import_module(mod_name)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}
    strip_dicts  = {
        slot: build_strip(u, name_to_spec[specname], tf=SLOT_TF[slot])
        for slot, specname in SLOT_TO_SPECNAME.items()
    }
    origins_view, _ = compute_view_layouts(strip_dicts, gap_px=gap_px)
    inv_maps        = build_inverse_maps(strip_dicts, origins_view)
    inv_maps["_ix_ot"]  = ix_ot
    inv_maps["_ix_omp"] = ix_omp
    inv_maps["_iy_sep"] = iy_sep

    # ── 6.  Define scan grids ─────────────────────────────────────────────────
    # FIX: D_perp and Chi_perp removed from the fixed dict — they are hidden.
    fixed = dict(
        R=args.R, B=args.B, Pin=args.Pin, Dcore=args.Dcore,
    )
    dpuff_vals = np.logspace(
        np.log10(args.dpuff_min), np.log10(args.dpuff_max), args.n_dpuff)

    npuff_vals = np.logspace(
        np.log10(args.npuff_min), np.log10(args.npuff_max), args.n_npuff)
    if not np.isclose(npuff_vals, args.npuff_fixed, rtol=1e-6).any():
        npuff_vals = np.sort(np.append(npuff_vals, args.npuff_fixed))
        print(f"  npuff_fixed={args.npuff_fixed:.1e} inserted into N_puff grid "
              f"→ {len(npuff_vals)} N_puff points")

    _log_step = (np.log10(args.dpuff_max) - np.log10(args.dpuff_min)) / (args.n_dpuff - 1)
    dpuff_ood_ext = np.logspace(
        np.log10(args.dpuff_max) + _log_step,
        np.log10(args.dpuff_ood_max),
        args.n_dpuff_ood,
    )
    dpuff_ood = np.concatenate([dpuff_vals, dpuff_ood_ext])

    print(f"\n  2D grid  : {args.n_dpuff} x {len(npuff_vals)} = "
          f"{args.n_dpuff * len(npuff_vals)} points")
    print(f"  D_puff   : {args.dpuff_min:.1e} – {args.dpuff_max:.1e}")
    print(f"  N_puff   : {args.npuff_min:.1e} – {args.npuff_max:.1e}")
    print(f"  N_puff0  : {args.npuff_fixed:.1e}  → log10 = {np.log10(args.npuff_fixed):.2f}")
    print(f"  Dcore    : {args.Dcore:.3e}  → log10 = {np.log10(args.Dcore):.2f}")
    print(f"  Pin      : {args.Pin:.3e} W")
    print(f"  N samples: {args.n_samples}")
    print(f"  NOTE     : D_perp and Chi_perp are HIDDEN — not passed to model")

    # ── Sanity check: only the 6 visible parameters ──────────────────────────
    # FIX: D_perp and Chi_perp rows removed from the sanity check.
    print(f"\n  Sanity check — encoded values vs training ranges:")
    print(f"  {'param':<14} {'value':>15}  {'train_min':>10}  {'train_max':>10}  {'in range?':>12}")
    checks = [
        ("R",         args.R,                     1.00,  10.0),
        ("B",         args.B,                     1.00,  10.0),
        ("Pin",       args.Pin,                   5e6,   1e8 ),
        ("D_puff_lo", np.log10(args.dpuff_min),   20.0,  24.0),
        ("D_puff_hi", np.log10(args.dpuff_max),   20.0,  24.0),
        ("N_puff",    np.log10(args.npuff_fixed),  18.0,  23.0),
        ("Dcore",     np.log10(args.Dcore),        19.0,  24.0),
    ]
    for name, val, lo, hi in checks:
        flag = "OK" if lo <= val <= hi else "*** OUT OF RANGE ***"
        print(f"  {name:<14} {val:>15.4g}  {lo:>10.4g}  {hi:>10.4g}  {flag:>12}")
    print()

    scan_kw = dict(
        model=model, view_masks=view_masks, view_geoms=view_geoms,
        c_in=c_in, c_out=c_out,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        s_c=s_c, eps=eps,
        y_indices=y_indices,
        pos_channels=pos_channels, signed_channels=signed_channels,
        j_Te=j_Te, j_extra=j_extra, inv_maps=inv_maps, layout=layout,
        mode=mode, integrator=args.integrator,
        n_samples=args.n_samples, n_steps=args.ode_steps,
        batch_size=args.batch_size, device=device,
        is_v9=is_v9,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 7.  2D grid scan ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  2D gas puff scan ...")
    print("="*60)
    params_2d, ND, NN = build_params_2d(dpuff_vals, npuff_vals, fixed)
    te_mean_2d_full, te_std_2d_full, te_ot_samp_2d, te_omp_samp_2d, \
        na_mean_2d_full, na_std_2d_full, na_ot_samp_2d, na_omp_samp_2d = \
        scan_to_te_mesh(params_2d, **scan_kw)

    te_ot_mean_2d = te_mean_2d_full[:, ix_ot, iy_sep].reshape(ND, NN)
    te_ot_std_2d  = te_std_2d_full[:, ix_ot, iy_sep].reshape(ND, NN)
    np.save(out_dir / "te_ot_mean_2d.npy", te_ot_mean_2d)
    np.save(out_dir / "te_ot_std_2d.npy",  te_ot_std_2d)

    # ── 8.  Extract 1D slice from 2D results ─────────────────────────────────
    i_n = int(np.argmin(np.abs(npuff_vals - args.npuff_fixed)))
    print(f"\n  Extracting 1D slice at N_puff={npuff_vals[i_n]:.2e} "
          f"(index {i_n} / {NN}, requested {args.npuff_fixed:.1e})")

    te_mean_1d_full = te_mean_2d_full.reshape(ND, NN, 104, 50)[:, i_n, :, :]
    te_std_1d_full  = te_std_2d_full.reshape(ND, NN, 104, 50)[:, i_n, :, :]

    te_ot_mean_1d  = te_mean_1d_full[:, ix_ot,  iy_sep]
    te_omp_mean_1d = te_mean_1d_full[:, ix_omp, iy_sep]
    te_ot_std_1d   = te_std_1d_full[:,  ix_ot,  iy_sep]
    te_omp_std_1d  = te_std_1d_full[:,  ix_omp, iy_sep]

    te_ot_samp_1d  = te_ot_samp_2d.reshape(args.n_samples, ND, NN)[:, :, i_n]
    te_omp_samp_1d = te_omp_samp_2d.reshape(args.n_samples, ND, NN)[:, :, i_n]
    print(f"  KDE samples extracted from 2D scan  (shape: {te_ot_samp_1d.shape})")

    if na_ot_samp_2d is not None:
        na_ot_mean_1d  = na_mean_2d_full.reshape(ND, NN, 104, 50)[:, i_n, ix_ot,  iy_sep]
        na_omp_mean_1d = na_mean_2d_full.reshape(ND, NN, 104, 50)[:, i_n, ix_omp, iy_sep]
        na_ot_samp_1d  = na_ot_samp_2d.reshape(args.n_samples, ND, NN)[:, :, i_n]
        na_omp_samp_1d = na_omp_samp_2d.reshape(args.n_samples, ND, NN)[:, :, i_n]
    else:
        na_ot_mean_1d = na_omp_mean_1d = na_ot_samp_1d = na_omp_samp_1d = None

    del te_mean_2d_full, te_std_2d_full, te_ot_samp_2d, te_omp_samp_2d
    if na_mean_2d_full is not None:
        del na_mean_2d_full, na_std_2d_full, na_ot_samp_2d, na_omp_samp_2d

    # ── 9.  OOD extended scan ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  OOD extension ({len(dpuff_ood_ext)} new points beyond {args.dpuff_max:.0e}) ...")
    print("="*60)
    params_ood_ext = build_params_1d(dpuff_ood_ext, args.npuff_fixed, fixed)
    te_mean_ext, te_std_ext, te_ot_samp_ext, te_omp_samp_ext, _, _, _, _ = \
        scan_to_te_mesh(params_ood_ext, **scan_kw)

    te_ot_mean_ood  = np.concatenate([te_ot_mean_1d,  te_mean_ext[:, ix_ot,  iy_sep]])
    te_omp_mean_ood = np.concatenate([te_omp_mean_1d, te_mean_ext[:, ix_omp, iy_sep]])
    te_ot_std_ood   = np.concatenate([te_ot_std_1d,   te_std_ext[:,  ix_ot,  iy_sep]])
    te_omp_std_ood  = np.concatenate([te_omp_std_1d,  te_std_ext[:,  ix_omp, iy_sep]])

    # Save raw arrays
    np.save(out_dir / "te_ot_mean_1d.npy",     te_ot_mean_1d)
    np.save(out_dir / "te_omp_mean_1d.npy",    te_omp_mean_1d)
    np.save(out_dir / "te_ot_std_1d.npy",      te_ot_std_1d)
    np.save(out_dir / "te_omp_std_1d.npy",     te_omp_std_1d)
    np.save(out_dir / "te_ot_samples_1d.npy",  te_ot_samp_1d)
    np.save(out_dir / "te_omp_samples_1d.npy", te_omp_samp_1d)
    np.save(out_dir / "dpuff_vals.npy",         dpuff_vals)
    np.save(out_dir / "npuff_vals.npy",         npuff_vals)
    np.save(out_dir / "dpuff_ood.npy",          dpuff_ood)
    print(f"\n  Raw arrays saved to {out_dir}")

    # ── 10.  Plotting ─────────────────────────────────────────────────────────
    print("\n  Generating figures ...")
    DG, NG = np.meshgrid(dpuff_vals, npuff_vals, indexing='ij')

    # ─ 2D heatmap Te,ot ──────────────────────────────────────────────────────
    if NN == 1:
        print("    Skipping fig_10_1A_te_ot_2D.png: n_npuff=1")
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        pos_vals = te_ot_mean_2d[te_ot_mean_2d > 0]
        vmin = float(np.percentile(pos_vals, 2))  if len(pos_vals) else 1e-2
        vmax = float(np.percentile(pos_vals, 98)) if len(pos_vals) else 1e4
        c = ax.pcolormesh(
            DG, NG, te_ot_mean_2d,
            norm=LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, vmin*2)),
            cmap='viridis', shading='nearest',
        )
        plt.colorbar(c, ax=ax, label='Te,ot  [eV]')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(args.dpuff_min, args.dpuff_max)
        ax.set_ylim(args.npuff_min, args.npuff_max)
        ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
        ax.set_ylabel('N$_{puff}$  [atoms/s]', fontsize=11)
        ax.set_title('FM model  -  Te,ot  (mean prediction)\n'
                     '[synthetic dataset — D_perp / Chi_perp hidden]', fontsize=10)
        ax.axhline(args.npuff_fixed, color='black', lw=1.2, ls='--',
                   label=f'N_puff = {args.npuff_fixed:.0e}  (1D slice)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_10_1A_te_ot_2D.png", dpi=150)
        plt.close(fig)
        print("    Saved: fig_10_1A_te_ot_2D.png")

    # ─ 3-panel 1D scan: mean + std + CV ──────────────────────────────────────
    te_ot_std_1d_plot  = np.load(out_dir / "te_ot_std_1d.npy")
    _omp_std_path = out_dir / "te_omp_std_1d.npy"
    te_omp_std_1d_plot = (np.load(_omp_std_path) if _omp_std_path.exists()
                          else np.zeros_like(te_omp_mean_1d))
    cv_ot_1d  = te_ot_std_1d_plot  / (te_ot_mean_1d  + 1e-10)
    cv_omp_1d = te_omp_std_1d_plot / (te_omp_mean_1d + 1e-10)

    _log_te = np.log10(np.maximum(te_ot_mean_1d, 1e-10))
    _trans  = int(np.argmax(np.abs(np.diff(_log_te))))

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 11),
                             gridspec_kw={"height_ratios": [2.5, 1.5, 1.5]},
                             sharex=True)

    ax = axes[0]
    ax.plot(dpuff_vals, te_ot_mean_1d,  color='C0', lw=2.2, label='Te,ot  (FM mean)')
    ax.plot(dpuff_vals, te_omp_mean_1d, color='C1', lw=2.2, label='Te,omp  (FM mean)')
    ax.axvline(dpuff_vals[_trans], color='gray', lw=1.0, ls=':', alpha=0.7,
               label=f'transition ~{dpuff_vals[_trans]:.1e}')
    ax.set_yscale('log')
    ax.set_ylabel('Electron temperature  [eV]', fontsize=10)
    ax.set_title(
        f'FM model  |  N_puff = {args.npuff_fixed:.0e} atoms/s  |  '
        f'n_samples = {args.n_samples}\n'
        f'[synthetic dataset — D_perp / Chi_perp hidden]',
        fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=5e-1)

    ax = axes[1]
    ax.plot(dpuff_vals, te_ot_std_1d_plot,  color='C0', lw=1.8, label='std(Te,ot)  [eV]')
    if _omp_std_path.exists():
        ax.plot(dpuff_vals, te_omp_std_1d_plot, color='C1', lw=1.8, label='std(Te,omp)  [eV]')
    ax.axvline(dpuff_vals[_trans], color='gray', lw=1.0, ls=':', alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylabel('std(Te)  [eV]', fontsize=10)
    ax.set_title('Absolute uncertainty  (should peak near detachment transition)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    _peak_i = int(np.argmax(te_ot_std_1d_plot))
    ax.annotate(
        f'peak = {te_ot_std_1d_plot[_peak_i]:.0f} eV\n'
        f'at D = {dpuff_vals[_peak_i]:.1e}',
        xy=(dpuff_vals[_peak_i], te_ot_std_1d_plot[_peak_i]),
        xytext=(dpuff_vals[min(_peak_i+3, len(dpuff_vals)-1)],
                te_ot_std_1d_plot[_peak_i] * 0.3),
        fontsize=7.5, color='C0',
        arrowprops=dict(arrowstyle='->', color='C0', lw=0.8),
    )

    ax = axes[2]
    ax.plot(dpuff_vals, cv_ot_1d * 100,  color='C0', lw=1.8, label='CV  Te,ot  [%]')
    if _omp_std_path.exists():
        ax.plot(dpuff_vals, cv_omp_1d * 100, color='C1', lw=1.8, label='CV  Te,omp  [%]')
    ax.axvline(dpuff_vals[_trans], color='gray', lw=1.0, ls=':', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('CV = std / mean  [%]', fontsize=10)
    ax.set_title('Relative uncertainty (coefficient of variation)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(dpuff_vals[0], dpuff_vals[-1])
    _peak_cv_i = int(np.argmax(cv_ot_1d))
    ax.annotate(
        f'peak CV = {cv_ot_1d[_peak_cv_i]*100:.2f}%\n'
        f'at D = {dpuff_vals[_peak_cv_i]:.1e}',
        xy=(dpuff_vals[_peak_cv_i], cv_ot_1d[_peak_cv_i]*100),
        xytext=(dpuff_vals[min(_peak_cv_i+3, len(dpuff_vals)-1)],
                cv_ot_1d[_peak_cv_i]*100*0.5),
        fontsize=7.5, color='C0',
        arrowprops=dict(arrowstyle='->', color='C0', lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_dir / "fig_10_1B_te_1D_scan.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_10_1B_te_1D_scan.png")

    # ─ Side-by-side quick overview ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes2[0]
    ax.plot(dpuff_vals, te_ot_mean_1d,  color='C0', lw=2.2, label='Te,ot  (FM mean)')
    ax.plot(dpuff_vals, te_omp_mean_1d, color='C1', lw=2.2, label='Te,omp  (FM mean)')
    ax.axvline(dpuff_vals[_trans], color='gray', lw=1.0, ls=':', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('Electron temperature  [eV]', fontsize=11)
    ax.set_title('Mean Te prediction', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=5e-1)

    ax  = axes2[1]
    ax2 = ax.twinx()
    ax.plot(dpuff_vals, te_ot_std_1d_plot,  color='C0', lw=2.2, label='std(Te,ot)  [eV]')
    if _omp_std_path.exists():
        ax.plot(dpuff_vals, te_omp_std_1d_plot, color='C1', lw=2.2, label='std(Te,omp)  [eV]')
    ax2.plot(dpuff_vals, cv_ot_1d * 100, color='C0', lw=1.5, ls='--',
             alpha=0.7, label='CV Te,ot [%]')
    ax.axvline(dpuff_vals[_trans], color='gray', lw=1.0, ls=':', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('std(Te)  [eV]', fontsize=11)
    ax2.set_ylabel('CV = std / mean  [%]', fontsize=10)
    ax.set_title('FM uncertainty along 1D slice', fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig2.suptitle(
        f'N_puff = {args.npuff_fixed:.0e} atoms/s  |  n_samples = {args.n_samples}  '
        f'[synthetic, hidden D_perp/Chi_perp]',
        fontsize=10)
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_10_1B_uncertainty_sidebyside.png", dpi=150)
    plt.close(fig2)
    print("    Saved: fig_10_1B_uncertainty_sidebyside.png")

    # ─ 2D uncertainty maps ────────────────────────────────────────────────────
    if NN == 1:
        print("    Skipping 2D uncertainty map: n_npuff=1")
    else:
        cv_2d = np.where(
            te_ot_mean_2d > 0,
            te_ot_std_2d / (te_ot_mean_2d + 1e-10),
            0.0,
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        std_pos = te_ot_std_2d[te_ot_std_2d > 0]
        im0 = axes[0].pcolormesh(
            DG, NG, te_ot_std_2d,
            norm=LogNorm(
                vmin=max(float(np.percentile(std_pos, 2))  if len(std_pos) else 1e-3, 1e-3),
                vmax=float(np.percentile(std_pos, 98)) if len(std_pos) else 1e3,
            ),
            cmap='hot_r', shading='nearest',
        )
        plt.colorbar(im0, ax=axes[0], label='std(Te,ot)  [eV]')
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
        axes[0].set_xlim(args.dpuff_min, args.dpuff_max)
        axes[0].set_ylim(args.npuff_min, args.npuff_max)
        axes[0].set_xlabel('D$_{puff}$  [atoms/s]')
        axes[0].set_ylabel('N$_{puff}$  [atoms/s]')
        axes[0].set_title('FM uncertainty:  std(Te,ot)')
        axes[0].axhline(args.npuff_fixed, color='white', lw=1, ls='--', alpha=0.7)

        im1 = axes[1].pcolormesh(
            DG, NG, np.clip(cv_2d, 0, 1.5),
            vmin=0, vmax=min(float(cv_2d.max()), 1.5),
            cmap='hot_r', shading='nearest',
        )
        plt.colorbar(im1, ax=axes[1], label='CV = std / mean')
        axes[1].set_xscale('log'); axes[1].set_yscale('log')
        axes[1].set_xlim(args.dpuff_min, args.dpuff_max)
        axes[1].set_ylim(args.npuff_min, args.npuff_max)
        axes[1].set_xlabel('D$_{puff}$  [atoms/s]')
        axes[1].set_ylabel('N$_{puff}$  [atoms/s]')
        axes[1].set_title('FM uncertainty:  CV = std / mean')
        axes[1].axhline(args.npuff_fixed, color='white', lw=1, ls='--', alpha=0.7)

        fig.suptitle('FM uncertainty over gas puff parameter space\n'
                     '[synthetic dataset — D_perp / Chi_perp hidden]', fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_uncertainty_2D.png", dpi=150)
        plt.close(fig)
        print("    Saved: fig_uncertainty_2D.png")

    # ─ KDE at selected D_puff points ─────────────────────────────────────────
    log_te_ot = np.log10(np.maximum(te_ot_mean_1d, 1e-10))
    delta_log = np.abs(np.diff(log_te_ot))
    trans_idx = int(np.argmax(delta_log))

    n_pts = len(dpuff_vals)
    tw    = min(args.kde_n_transition, n_pts // 2)
    tz_lo = max(0, trans_idx - tw)
    tz_hi = min(n_pts - 1, trans_idx + tw)
    tz_count   = min(args.kde_n_transition, tz_hi - tz_lo + 1)
    trans_zone = [int(round(x)) for x in np.linspace(tz_lo, tz_hi, tz_count)]

    background = sorted(set([
        0,
        max(0, tz_lo - max(1, tz_lo // 2)),
        min(n_pts - 1, tz_hi + max(1, (n_pts - 1 - tz_hi) // 2)),
        n_pts - 1,
    ]))
    background = [i for i in background if i < tz_lo or i > tz_hi]
    sel_idx = sorted(set(background + trans_zone))
    sel_idx = [i for i in sel_idx if 0 <= i < n_pts]

    cmap_pts    = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(sel_idx)))
    trans_color = 'gray'

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for color, idx in zip(cmap_pts, sel_idx):
        samp = te_ot_samp_1d[:, idx]
        samp = samp[samp > 0]
        if len(samp) < 3:
            continue
        dp_val          = dpuff_vals[idx]
        x_grid, density = kde_log(samp)
        lw  = 2.2 if idx == trans_idx else 1.6
        col = trans_color if idx == trans_idx else color
        ax.plot(x_grid, density, color=col, lw=lw,
                label=f'D={dp_val:.1e}' + (' ←transition' if idx == trans_idx else ''))
        ax.axvline(np.median(samp), color=col, lw=0.7, ls=':', alpha=0.5)

    # Optional ground truth overlay
    if args.test_X_path and args.test_te_path:
        try:
            X_test  = np.load(args.test_X_path)    # (N_test, 6), col3=log10(Dpuff)
            te_test = np.load(args.test_te_path)    # (N_test, 104, 50)
            dpuff_test = 10.0 ** X_test[:, 3]
            npuff_test = 10.0 ** X_test[:, 4]
            J_PER_EV   = 1.602176634e-19
            te_ot_test = te_test[:, ix_ot, iy_sep] / J_PER_EV
            n_mask = np.abs(np.log10(npuff_test) - np.log10(args.npuff_fixed)) <= 1.0
            d_mask = (dpuff_test >= args.dpuff_min) & (dpuff_test <= args.dpuff_max)
            valid  = n_mask & d_mask & (te_ot_test > 0)
            if valid.any():
                ax.scatter(te_ot_test[valid], np.full(valid.sum(), 0.05),
                           s=20, marker='|', color='black', alpha=0.6, zorder=5,
                           label=f'Ground truth ({valid.sum()} test sims)')
                print(f"    Ground truth overlay: {valid.sum()} test sims plotted")
        except Exception as e:
            print(f"    Ground truth overlay skipped: {e}")

    ax.set_xscale('log')
    ax.set_xlabel('Te,ot  [eV]', fontsize=11)
    ax.set_ylabel('Normalised KDE  (log-space bandwidth)', fontsize=10)
    ax.set_title(
        f'Sample distribution of Te,ot at selected D_puff values\n'
        f'(N_puff = {args.npuff_fixed:.0e},  n_samples = {args.n_samples},  '
        f'⚡ transition ≈ {dpuff_vals[trans_idx]:.1e})\n'
        f'Bimodal KDE = bifurcation detected from hidden Chi_perp',
        fontsize=9,
    )
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sample_kde_1D.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_sample_kde_1D.png")

    # ─ Violin plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xscale('log'); ax.set_yscale('log')

    log_dpuff = np.log10(dpuff_vals)
    log_step  = (log_dpuff[-1] - log_dpuff[0]) / max(len(log_dpuff) - 1, 1)
    half_w    = log_step * 0.42

    cmap_violin = plt.cm.coolwarm(np.linspace(0.0, 1.0, n_pts))
    for i in range(n_pts):
        samp = te_ot_samp_1d[:, i]; samp = samp[samp > 0]
        if len(samp) < 3: continue
        x_grid, density = kde_log(samp, n_pts=120)
        w    = density * half_w
        ld   = log_dpuff[i]
        x_lo = 10 ** (ld - w); x_hi = 10 ** (ld + w)
        alpha = 0.65 if i == trans_idx else 0.45
        ax.fill_betweenx(x_grid, x_lo, x_hi,
                         alpha=alpha, color=cmap_violin[i], linewidth=0.0)

    ax.plot(dpuff_vals, te_ot_mean_1d, color='black', lw=1.8, label='FM mean Te,ot', zorder=4)
    q05 = np.percentile(te_ot_samp_1d, 5,  axis=0)
    q95 = np.percentile(te_ot_samp_1d, 95, axis=0)
    ax.plot(dpuff_vals, np.maximum(q05, 1e-2), color='black', lw=0.8, ls='--',
            alpha=0.5, label='5th–95th pct')
    ax.plot(dpuff_vals, q95, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(dpuff_vals[trans_idx], color='gray', lw=1.0, ls=':',
               label=f'transition ≈ {dpuff_vals[trans_idx]:.1e}')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('Te,ot  [eV]', fontsize=11)
    ax.set_xlim(dpuff_vals[0], dpuff_vals[-1])
    ax.set_ylim(bottom=max(1e-2, float(np.percentile(te_ot_samp_1d[te_ot_samp_1d>0], 1)) * 0.5))
    ax.set_title(
        f'FM violin plot — Te,ot sample distribution at each D_puff\n'
        f'N_puff = {args.npuff_fixed:.0e}  |  n_samples = {args.n_samples}  '
        f'|  colour = D_puff (blue→red)',
        fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_violin_1D.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_violin_1D.png")

    # ─ KDE convergence check ─────────────────────────────────────────────────
    samp_at_trans = te_ot_samp_1d[:, trans_idx]
    samp_at_trans = samp_at_trans[samp_at_trans > 0]
    n_avail       = len(samp_at_trans)

    fracs    = [0.25, 0.5, 0.75, 1.0]
    n_subs   = [max(5, int(n_avail * f)) for f in fracs]
    colors_c = ['#bbbbcc', '#7777bb', '#3333aa', '#000080']
    lws_c    = [1.0, 1.4, 1.8, 2.2]

    pre_std = te_ot_std_1d[:trans_idx]
    pre_idx = int(np.argmax(pre_std)) if len(pre_std) else max(0, trans_idx - 3)
    samp_pre = te_ot_samp_1d[:, pre_idx]; samp_pre = samp_pre[samp_pre > 0]

    fig, axes_c = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes_c[0]
    for n_sub, col, lw_c in zip(n_subs, colors_c, lws_c):
        sub = samp_at_trans[:n_sub]
        x_grid, density = kde_log(sub)
        ax.plot(x_grid, density, color=col, lw=lw_c,
                label=f'n={n_sub}  ({int(100*n_sub/n_avail)}%)')
    ax.set_xscale('log')
    ax.set_xlabel('Te,ot  [eV]', fontsize=10)
    ax.set_ylabel('Normalised KDE', fontsize=10)
    ax.set_title(f'KDE convergence at transition  (D_puff ≈ {dpuff_vals[trans_idx]:.1e})\n'
                 f'Bimodal shape = two Chi_perp branches', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes_c[1]
    for n_sub, col, lw_c in zip(n_subs, colors_c, lws_c):
        sub = samp_pre[:min(n_sub, len(samp_pre))]
        if len(sub) < 3: continue
        x_grid, density = kde_log(sub)
        ax.plot(x_grid, density, color=col, lw=lw_c, label=f'n={n_sub}')
    ax.set_xscale('log')
    ax.set_xlabel('Te,ot  [eV]', fontsize=10)
    ax.set_ylabel('Normalised KDE', fontsize=10)
    ax.set_title(f'KDE convergence at pre-transition peak  (D_puff ≈ {dpuff_vals[pre_idx]:.1e})',
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Sampling convergence check  |  N_puff = {args.npuff_fixed:.0e}  '
        f'|  total n_samples = {args.n_samples}',
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_kde_convergence.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_kde_convergence.png")

    # ─ Scatter: all samples at every D_puff ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_yscale('log')
    cmap_sc = plt.cm.coolwarm(np.linspace(0, 1, len(dpuff_vals)))
    rng = np.random.default_rng(0)
    for i, (dp, col) in enumerate(zip(dpuff_vals, cmap_sc)):
        ys = te_ot_samp_1d[:, i]; ys = ys[ys > 0]
        if len(ys) == 0: continue
        jitter = dp * 0.003 * (rng.random(len(ys)) - 0.5)
        ax.scatter(np.full(len(ys), dp) + jitter, ys,
                   s=2, alpha=0.25, color=col, linewidths=0, rasterized=True)
    ax.plot(dpuff_vals, te_ot_mean_1d, color='black', lw=2, zorder=5, label='FM mean')
    q05 = np.percentile(te_ot_samp_1d, 5,  axis=0)
    q95 = np.percentile(te_ot_samp_1d, 95, axis=0)
    ax.plot(dpuff_vals, np.maximum(q05, 1e-2), color='black', lw=1, ls='--',
            alpha=0.5, label='5th–95th pct')
    ax.plot(dpuff_vals, q95, color='black', lw=1, ls='--', alpha=0.5)
    ax.axvline(dpuff_vals[trans_idx], color='gray', lw=1, ls=':',
               label=f'transition ≈ {dpuff_vals[trans_idx]:.1e}')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('Te,ot  [eV]', fontsize=11)
    ax.set_xlim(dpuff_vals[0], dpuff_vals[-1])
    ax.set_title(
        f'All {args.n_samples} FM samples at each D_puff  (N_puff={args.npuff_fixed:.0e})\n'
        f'Two visible clusters near transition → bifurcation from hidden Chi_perp',
        fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_scatter_all_samples.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_scatter_all_samples.png")

    # ─ Peak convergence ───────────────────────────────────────────────────────
    def peak_info_fixed(samp, n_eval):
        from scipy.signal import argrelmax
        sub = samp[:n_eval]; sub = sub[sub > 0]
        if len(sub) < 5: return np.nan, np.nan
        x_g, dens = kde_log(sub, n_pts=200)
        peaks_idx, = argrelmax(dens, order=4)
        if len(peaks_idx) == 0:
            return float(x_g[np.argmax(dens)]), np.nan
        peak_te = sorted(float(x_g[p]) for p in peaks_idx)
        p1 = peak_te[0]
        p2 = peak_te[-1] if len(peak_te) >= 2 else np.nan
        return p1, p2

    samp_tr  = te_ot_samp_1d[:, trans_idx]; samp_tr  = samp_tr[samp_tr > 0]
    samp_pre_c = te_ot_samp_1d[:, pre_idx]; samp_pre_c = samp_pre_c[samp_pre_c > 0]
    ns_range = np.unique(np.round(
        np.logspace(np.log10(10), np.log10(len(samp_tr)), 40)).astype(int))
    ns_range = ns_range[ns_range <= len(samp_tr)]

    p1_tr, p2_tr, std_conv = [], [], []
    for n in ns_range:
        p1, p2 = peak_info_fixed(samp_tr, n)
        p1_tr.append(p1); p2_tr.append(p2)
        sub = samp_tr[:n]; sub = sub[sub > 0]
        std_conv.append(float(sub.std()) if len(sub) > 1 else np.nan)

    p1_pre, p2_pre = [], []
    for n in ns_range:
        p1, p2 = peak_info_fixed(samp_pre_c, n)
        p1_pre.append(p1); p2_pre.append(p2)

    fig, axes_p = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, p1, p2, title in [
        (axes_p[0], p1_tr,    p2_tr,
         f'Transition  D≈{dpuff_vals[trans_idx]:.1e}'),
        (axes_p[1], p1_pre,   p2_pre,
         f'Pre-transition  D≈{dpuff_vals[pre_idx]:.1e}'),
    ]:
        ax.semilogx(ns_range, p1, color='steelblue', lw=1.8,
                    label='Peak 1 (low Te, detached)')
        ax.semilogx(ns_range, p2, color='tomato',    lw=1.8, ls='--',
                    label='Peak 2 (high Te, sheath-limited)')
        ax.set_xlabel('n_samples'); ax.set_ylabel('Te,ot  [eV]')
        ax.set_title(title + '\n(peaks sorted by Te value)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes_p[2].semilogx(ns_range, std_conv, color='black', lw=1.8)
    axes_p[2].set_xlabel('n_samples'); axes_p[2].set_ylabel('std(Te,ot)  [eV]')
    axes_p[2].set_title(f'std convergence at transition\nD≈{dpuff_vals[trans_idx]:.1e}',
                        fontsize=9)
    axes_p[2].grid(True, alpha=0.3)
    fig.suptitle(
        f'Sampling convergence — peak positions and std vs n_samples\n'
        f'N_puff={args.npuff_fixed:.0e}  |  total n_samples={args.n_samples}',
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_peak_convergence.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_peak_convergence.png")

    # ─ OOD extension ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(dpuff_ood, te_ot_mean_ood,  color='C0', lw=2.0, label='Te,ot  (mean)')
    ax.plot(dpuff_ood, te_omp_mean_ood, color='C1', lw=2.0, label='Te,omp  (mean)')
    ax.fill_between(dpuff_ood,
                    np.maximum(te_ot_mean_ood  - te_ot_std_ood,  1e-2),
                    te_ot_mean_ood + te_ot_std_ood,
                    color='C0', alpha=0.25, label='±1σ  Te,ot')
    ax.fill_between(dpuff_ood,
                    np.maximum(te_omp_mean_ood - te_omp_std_ood, 1e-2),
                    te_omp_mean_ood + te_omp_std_ood,
                    color='C1', alpha=0.25)
    ax.axvline(args.dpuff_max, color='red', lw=1.5, ls=':',
               label=f'training limit ({args.dpuff_max:.0e})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('Te  [eV]', fontsize=11)
    ax.set_title('OOD extension: Te vs D_puff', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=5e-1)

    ax  = axes[1]
    ax2 = ax.twinx()
    ax.plot(dpuff_ood, te_ot_std_ood,  color='C0', lw=2.0, label='std(Te,ot)')
    ax.plot(dpuff_ood, te_omp_std_ood, color='C1', lw=2.0, label='std(Te,omp)')
    cv_ot_ood = te_ot_std_ood / (te_ot_mean_ood + 1e-10)
    ax2.plot(dpuff_ood, cv_ot_ood, color='gray', lw=1.5, ls='--',
             label='CV = std/mean  (Te,ot)')
    ax.axvline(args.dpuff_max, color='red', lw=1.5, ls=':', label='training limit')
    ax.set_xscale('log')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('std(Te)  [eV]', fontsize=11)
    ax2.set_ylabel('CV = std / mean', fontsize=10)
    ax.set_title('OOD: uncertainty grows beyond training range', fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        f'OOD extension  |  N_puff fixed = {args.npuff_fixed:.0e}  |  '
        f'n_samples = {args.n_samples}',
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ood_extension.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_ood_extension.png")

    # ─ Density anti-correlation ───────────────────────────────────────────────
    if na_ot_samp_1d is not None:
        fig, axes_d = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes_d[0]
        ax.semilogy(dpuff_vals, te_ot_mean_1d, color='steelblue', lw=2,
                    label='Te,ot  [eV]')
        ax2 = ax.twinx()
        ax2.semilogy(dpuff_vals, np.maximum(na_ot_mean_1d, 1e-10),
                     color='tomato', lw=2, label=f'na,ot  ({_na_name})  [m⁻³]')
        ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=10)
        ax.set_ylabel('Te,ot  [eV]', color='steelblue', fontsize=10)
        ax2.set_ylabel('na,ot  [m⁻³]', color='tomato', fontsize=10)
        ax.axvline(dpuff_vals[trans_idx], color='gray', lw=1, ls=':')
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, lab1+lab2, fontsize=8)
        ax.set_title(f'Te vs {_na_name} density — outer target', fontsize=9)

        ax = axes_d[1]
        ax.semilogy(dpuff_vals, te_omp_mean_1d, color='steelblue', lw=2,
                    label='Te,omp  [eV]')
        ax3 = ax.twinx()
        ax3.semilogy(dpuff_vals, np.maximum(na_omp_mean_1d, 1e-10),
                     color='darkorange', lw=2, label=f'na,omp  ({_na_name})  [m⁻³]')
        ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=10)
        ax.set_ylabel('Te,omp  [eV]', color='steelblue', fontsize=10)
        ax3.set_ylabel('na,omp  [m⁻³]', color='darkorange', fontsize=10)
        ax.axvline(dpuff_vals[trans_idx], color='gray', lw=1, ls=':')
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax3.get_legend_handles_labels()
        ax.legend(lines1+lines2, lab1+lab2, fontsize=8)
        ax.set_title(f'Te vs {_na_name} density — outer midplane', fontsize=9)

        ax = axes_d[2]
        te_sc = te_ot_samp_1d[:, trans_idx]
        na_sc = na_ot_samp_1d[:, trans_idx]
        valid = (te_sc > 0) & (na_sc > 0)
        sc = ax.scatter(te_sc[valid], na_sc[valid],
                        s=8, alpha=0.4, c=np.arange(valid.sum()), cmap='plasma')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Te,ot  [eV]', fontsize=10)
        ax.set_ylabel(f'na,ot  [m⁻³]  ({_na_name})', fontsize=10)
        ax.set_title(f'(Te, na) at transition D≈{dpuff_vals[trans_idx]:.1e}\n'
                     f'{valid.sum()} samples — expect 2 clusters (anti-correlated)', fontsize=9)
        ax.grid(True, which='both', alpha=0.25)
        plt.colorbar(sc, ax=ax, label='sample index')

        fig.suptitle(
            f'Temperature–density anti-correlation  '
            f'(N_puff={args.npuff_fixed:.0e},  n_samples={args.n_samples},  '
            f'species: {_na_name})',
            fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_te_na_anticorrelation.png", dpi=150)
        plt.close(fig)
        print("    Saved: fig_te_na_anticorrelation.png")
    else:
        print("    Skipping fig_te_na_anticorrelation.png: species not in model outputs")

    print(f"\n{'='*60}")
    print(f"  All outputs written to: {out_dir}")
    print(f"  Figures:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"    {f.name}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Gas puff scan + FM uncertainty analysis for the SYNTHETIC dataset "
            "(D_perp and Chi_perp hidden).  Use scan_gas_puff.py for the original "
            "8-parameter SOLPS dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    ap.add_argument("--scripts_dir",   required=True,
                    help="Path to the scripts/ folder containing "
                         "infer_fm_fusvel_v8.py, reconstruct_mesh.py, utils.py")
    ap.add_argument("--checkpoint",    required=True,
                    help="Path to checkpoint_best.pt")
    ap.add_argument("--tensor_prefix", required=True,
                    help="Prefix for X_img tensors, e.g. scripts/tensor/train/global3")
    ap.add_argument("--layout_path",   required=True,
                    help="Path to *_layout_map_3views.npz")
    ap.add_argument("--unroller",      required=True,
                    help="Path to unrolled_strip_clockwise_adjpreserve.py")
    ap.add_argument("--b2fgmtry",      required=True,
                    help="Path to b2fgmtry geometry file")
    ap.add_argument("--out_dir",       required=True,
                    help="Output directory for figures and .npy arrays")

    # ── Fixed parameters (6 visible inputs — D_perp / Chi_perp removed) ──────
    ap.add_argument("--R",     type=float, default=6.2,
                    help="Major radius R [m]")
    ap.add_argument("--B",     type=float, default=5.3,
                    help="Toroidal field B [T]")
    ap.add_argument("--Pin",   type=float, default=100e6,
                    help="Input power Pin [W].  100 MW → 100e6.")
    ap.add_argument("--Dcore", type=float, default=9.1e21,
                    help="Core fueling Dcore [atoms/s]")
    # NOTE: --D_perp and --Chi_perp intentionally removed.
    #       They are hidden in the synthetic dataset and must NOT be passed
    #       to the model.

    # ── Scan grid ─────────────────────────────────────────────────────────────
    ap.add_argument("--n_dpuff",       type=int,   default=40)
    ap.add_argument("--n_npuff",       type=int,   default=35)
    ap.add_argument("--dpuff_min",     type=float, default=1e20)
    ap.add_argument("--dpuff_max",     type=float, default=1e24)
    ap.add_argument("--npuff_min",     type=float, default=1e18)
    ap.add_argument("--npuff_max",     type=float, default=1e23)
    ap.add_argument("--npuff_fixed",   type=float, default=1e20,
                    help="Fixed N_puff value for the 1D D_puff scan [atoms/s]")
    ap.add_argument("--dpuff_ood_max", type=float, default=1e25)
    ap.add_argument("--n_dpuff_ood",   type=int,   default=20)

    # ── KDE / visualisation ───────────────────────────────────────────────────
    ap.add_argument("--kde_n_transition", type=int, default=6)
    ap.add_argument("--na_species",       type=int, default=0,
                    help="Species index for density anti-correlation plot "
                         "(0=D0 neutral, 1=D1 ion, 2-9=nitrogen species)")

    # ── Optional ground truth overlay ─────────────────────────────────────────
    ap.add_argument("--test_X_path",  default=None,
                    help="Path to synthetic test/X_legacy.npy (6-column).  "
                         "When provided together with --test_te_path, ground truth "
                         "Te,ot values are overlaid on the KDE figure as vertical ticks.")
    ap.add_argument("--test_te_path", default=None,
                    help="Path to synthetic test/te_tmp.npy  (N_test × 104 × 50).")

    # ── Inference settings ────────────────────────────────────────────────────
    ap.add_argument("--n_samples",    type=int,   default=50,
                    help="Number of stochastic samples per input point.  "
                         "50 recommended for bifurcation detection.")
    ap.add_argument("--ode_steps",    type=int,   default=20)
    ap.add_argument("--batch_size",   type=int,   default=64)
    ap.add_argument("--mode",         default=None,
                    choices=["cfm", "xpred", "direct"])
    ap.add_argument("--integrator",   default="stochastic",
                    choices=["euler", "stochastic"],
                    help="'stochastic' is required for meaningful uncertainty estimates.")
    ap.add_argument("--tensor_split", default="train",
                    help="Split name used when loading the reference mask tensor")
    ap.add_argument("--device",       default=None)
    ap.add_argument("--seed",         type=int, default=42)

    args = ap.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
