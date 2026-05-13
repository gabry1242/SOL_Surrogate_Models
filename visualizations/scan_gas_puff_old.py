#!/usr/bin/env python3
"""
scan_gas_puff.py
────────────────────────────────────────────────────────────────────────────
Reproduce Figure 10.1A/B (Dasbach 2025) and perform uncertainty analysis
using a trained Flow Matching surrogate model.

Three outputs
─────────────
  1. fig_10_1A_te_ot_2D.png       — 2D heatmap of Te,ot over (D_puff, N_puff)
  2. fig_10_1B_te_1D_scan.png     — 1D scan: Te,ot & Te,omp vs D_puff, with ±1σ bands
  3. fig_uncertainty_2D.png       — std and CV maps over (D_puff, N_puff)
  4. fig_sample_kde_1D.png        — per-sample KDE at key D_puff points (detects multimodality)
  5. fig_ood_extension.png        — OOD scan extending D_puff beyond training range

Fixed ITER parameters (thesis p.180):
  R=6.2m  B=5.3T  D_perp=0.3m²/s  Chi_perp=1.0m²/s
  Dcore=9.1e21 atoms/s  Pin=100MW

⚠️  Pin units: the script expects SI watts (--Pin 100e6).
    Verify against your X_tmp.npy before running.  If your training data
    stored Pin in MW, pass --Pin 100 instead.

Usage
─────
python scan_gas_puff.py \\
    --checkpoint    scripts/runs/my_run/checkpoint_best.pt \\
    --tensor_prefix scripts/tensor/train/global3 \\
    --layout_path   scripts/tensor/train/global3_layout_map_3views.npz \\
    --unroller      scripts/tensor/unrolled_strip_clockwise_adjpreserve.py \\
    --b2fgmtry      geometry/b2fgmtry \\
    --out_dir       scripts/runs/my_run/gas_puff_scan \\
    --n_dpuff 40  --n_npuff 35  --n_samples 30
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
# Resolve the directory containing infer_fm_fusvel_v6.py / utils.py
# early, before argparse runs, so the imports below work.
#
# Pass   --scripts_dir C:\path\to\scripts   on the command line.
# If omitted, the script falls back to its own parent directory.
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_scripts_dir() -> Path:
    """Read --scripts_dir from sys.argv without running the full argparse."""
    for i, tok in enumerate(sys.argv):
        if tok in ("--scripts_dir", "--scripts-dir") and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
    # fallback: try own parent (works when script lives directly in scripts/)
    return Path(__file__).resolve().parent.parent

SCRIPTS_DIR = Path(__file__).resolve().parent
PARENT_DIR  = SCRIPTS_DIR.parent
UTIL_DIR = PARENT_DIR.parent
PARENT_DIR = PARENT_DIR / "models" / "FM_fin"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # visualizations/ itself

from infer_fm_fusvel_v6 import (          # model definition + samplers
    VelocityUNet,
    stochastic_integrate_xpred,
    euler_integrate_xpred,
    euler_integrate_velocity,
    direct_forward,
    inverse_transform_y,
    POS_CHANNELS,
    SIGNED_CHANNELS,
)
from reconstruct_mesh import (             # 3-view → (104,50) mesh
    build_strip,
    compute_view_layouts,
    build_inverse_maps,
    reconstruct_to_mesh,
    SLOT_TO_SPECNAME,
    SLOT_TF,
)
from utils import Geometry                 # b2fgmtry reader + geometry indices


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers  (mirrors evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_separatrix_indices(geo: Geometry) -> Tuple[int, int, int]:
    """Return (ix_ot, ix_omp, iy_sep).

    ix_ot  = outer target poloidal index     (geo.nx)
    ix_omp = outer midplane poloidal index   (geo.jxa)
    iy_sep = first radial cell outside sep.  (geo.jsep + 2)
    """
    return geo.nx, geo.jxa, geo.jsep + 2


# ─────────────────────────────────────────────────────────────────────────────
# X_img construction for arbitrary scan parameters
# ─────────────────────────────────────────────────────────────────────────────

def build_x_img_batch(
    params_arr: np.ndarray,         # (N_scan, 8) raw physical values
    mask_v:     np.ndarray,         # (H, W) static binary mask for this view
    geom_v:     Optional[np.ndarray],# (2, H, W) geometry channels, or None
    c_in:       int,
    x_mean:     np.ndarray,         # (c_in,)
    x_std:      np.ndarray,         # (c_in,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Build normalised X and raw mask tensors for a batch of scan inputs.

    Mirrors ViewXDataset.__getitem__ from the training script:
      - channel 0 : static mask (mean=0, std=1 → unchanged by normalisation)
      - channels 1-8 : scalar params × mask
      - channels 9-10: geometry centroids (optional, same for all scan points)

    Returns
    -------
    X_norm : (N_scan, c_in, H, W) float32 — normalised, ready for model
    masks  : (N_scan, 1,    H, W) float32 — raw mask (for prediction masking)
    """
    N_scan    = params_arr.shape[0]
    H, W      = mask_v.shape
    X_raw     = np.zeros((N_scan, c_in, H, W), dtype=np.float32)

    # Channel 0: mask (broadcasted across all scan points)
    X_raw[:, 0] = mask_v[None]

    # Channels 1–8: param_k × mask  (shape broadcasting: scalar × (H,W))
    for k in range(8):
        X_raw[:, 1 + k] = params_arr[:, k, None, None].astype(np.float32) * mask_v[None]

    # Channels 9–10: geometry (same for all points, already masked in source)
    if geom_v is not None and c_in >= 11:
        X_raw[:, 9:11] = geom_v[None]

    # Extract mask BEFORE normalisation (ViewXDataset does it in this order)
    masks = X_raw[:, 0:1].copy()   # (N_scan, 1, H, W)

    # Normalise per-channel.  The mask channel has mean=0, std=1 (forced by
    # compute_x_stats during training), so it is unchanged by this step.
    X_norm = (X_raw - x_mean[None, :, None, None]) / x_std[None, :, None, None]
    return X_norm.astype(np.float32), masks.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Inference: one view, all scan points, all stochastic samples
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
    """Run ONE stochastic sample for all N_scan inputs on one view.

    Returns
    -------
    preds_phys : (N_scan, C_out, H, W) float32 in physical units
    """
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

    # Denormalise batch-by-batch to avoid one giant allocation
    preds_phys = np.empty_like(preds_norm)
    for b0 in range(0, N_scan, batch_size):
        b1   = min(b0 + batch_size, N_scan)
        pt   = (preds_norm[b0:b1] * y_std[None, :, None, None]
                + y_mean[None, :, None, None])
        preds_phys[b0:b1] = inverse_transform_y(
            pt, y_indices, pos_channels, signed_channels, eps, s_c)
        preds_phys[b0:b1] *= masks[b0:b1]

    del preds_norm
    return preds_phys   # (N_scan, C_out, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Core scan runner: params → Te mesh statistics
# Uses Welford's online algorithm so only 2 × (N_scan, 104, 50) arrays are
# held in RAM at any time instead of n_samples × 3 × (N_scan, C_out, H, W).
# ─────────────────────────────────────────────────────────────────────────────

def scan_to_te_mesh(
    params_arr: np.ndarray,      # (N_scan, 8)
    model, view_masks, view_geoms, c_in, c_out,
    x_mean, x_std, y_mean, y_std, s_c, eps,
    y_indices, pos_channels, signed_channels,
    j_Te, inv_maps, layout,
    mode, integrator, n_samples, n_steps, batch_size, device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full scan pipeline for params_arr.

    Memory strategy
    ---------------
    Each sample is processed one at a time:
      1. Run 3 views sequentially → (N_scan, C_out, H, W) each, freed after use
      2. Reconstruct to (N_scan, C_out, 104, 50), extract Te → (N_scan, 104, 50)
      3. Update Welford running mean/variance, discard spatial array
      4. Save scalar values at sepa/sepm for KDE plots

    Peak RAM ≈ 3 × (N_scan × C_out × H × W) × float32  (one view buffer each)

    Returns
    -------
    te_mean         : (N_scan, 104, 50) float32
    te_std          : (N_scan, 104, 50) float32
    te_ot_samples   : (n_samples, N_scan) float32 — scalars at sepa
    te_omp_samples  : (n_samples, N_scan) float32 — scalars at sepm
    """
    N_scan   = params_arr.shape[0]
    ix_ot    = inv_maps.get("ix_ot",  None)   # resolved by caller via geometry
    # Pre-build per-view X tensors once (reused across all samples)
    print("  Pre-building X tensors for all views ...")
    view_X: Dict[int, np.ndarray]     = {}
    view_M: Dict[int, np.ndarray]     = {}
    for v in range(3):
        view_X[v], view_M[v] = build_x_img_batch(
            params_arr, view_masks[v], view_geoms[v], c_in, x_mean, x_std)

    # Welford accumulators for Te field — float64 for numerical stability
    wf_mean = np.zeros((N_scan, 104, 50), dtype=np.float64)
    wf_M2   = np.zeros((N_scan, 104, 50), dtype=np.float64)

    # Scalar sample storage (tiny — just floats at two grid points)
    te_ot_samp  = np.zeros((n_samples, N_scan), dtype=np.float32)
    te_omp_samp = np.zeros((n_samples, N_scan), dtype=np.float32)

    for s in range(n_samples):
        print(f"  Sample {s+1}/{n_samples} — running 3 views ...", end="\r", flush=True)

        # Run each view and reconstruct immediately
        pred_views = {}
        for v in range(3):
            pred_views[f"view{v}"] = infer_view_samples(
                model, view_X[v], view_M[v],
                view_id=v, c_out=c_out,
                n_steps=n_steps, mode=mode, integrator=integrator,
                y_indices=y_indices,
                pos_channels=pos_channels, signed_channels=signed_channels,
                y_mean=y_mean, y_std=y_std, eps=eps, s_c=s_c,
                batch_size=batch_size, device=device,
            )

        # Reconstruct to (N_scan, C_out, 104, 50)
        mesh_out, _ = reconstruct_to_mesh(pred_views, inv_maps, layout)
        del pred_views   # free the 3 view buffers immediately

        # Extract Te channel → (N_scan, 104, 50)
        te_s = mesh_out[:, j_Te, :, :].astype(np.float64)
        del mesh_out

        # Welford online update
        delta      = te_s - wf_mean
        wf_mean   += delta / (s + 1)
        wf_M2     += delta * (te_s - wf_mean)

        # Store scalars — these are resolved by the caller with geometry indices
        # We store the full (N_scan, 104, 50) slice temporarily for scalar extraction
        # but pass the geometry indices via the layout dict if available,
        # otherwise store the full field for one sample and extract outside.
        te_ot_samp[s]  = te_s[:, inv_maps["_ix_ot"],  inv_maps["_iy_sep"]]
        te_omp_samp[s] = te_s[:, inv_maps["_ix_omp"], inv_maps["_iy_sep"]]

    print()
    te_mean = wf_mean.astype(np.float32)
    te_std  = np.sqrt(wf_M2 / max(n_samples - 1, 1)).astype(np.float32)
    return te_mean, te_std, te_ot_samp, te_omp_samp
    te_std     = te_samples.std(axis=0)
    return te_mean, te_std, te_samples


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid builders
# ─────────────────────────────────────────────────────────────────────────────

def build_params_2d(dpuff_arr, npuff_arr, fixed):
    """(N_dpuff x N_npuff, 8) params array for a 2D Cartesian grid.

    Column order matches X_tmp:
      [R, B, Pin, D_puff, N_puff, Dcore, D_perp, Chi_perp]

    NOTE: D_puff, N_puff and Dcore are stored as log10 in X_tmp.npy
    (confirmed: D_puff range 20-24, N_puff 18-23, Dcore 19-24).
    dpuff_arr/npuff_arr are passed in physical units and converted here.
    fixed['Dcore'] is also in physical units (e.g. 9.1e21 atoms/s).
    """
    ND, NN = len(dpuff_arr), len(npuff_arr)
    DG, NG = np.meshgrid(dpuff_arr, npuff_arr, indexing='ij')  # (ND, NN)
    D_flat = DG.flatten()
    N_flat = NG.flatten()
    params  = np.empty((len(D_flat), 8), dtype=np.float64)
    params[:, 0] = fixed['R']
    params[:, 1] = fixed['B']
    params[:, 2] = fixed['Pin']
    params[:, 3] = np.log10(D_flat)            # log10-encoded
    params[:, 4] = np.log10(N_flat)            # log10-encoded
    params[:, 5] = np.log10(fixed['Dcore'])    # log10-encoded
    params[:, 6] = fixed['D_perp']
    params[:, 7] = fixed['Chi_perp']
    return params.astype(np.float32), ND, NN


def build_params_1d(dpuff_arr, npuff_fixed, fixed):
    """(N_dpuff, 8) params array for a 1D D_puff scan at fixed N_puff.

    dpuff_arr and npuff_fixed are in physical units (atoms/s).
    Dcore is in physical units.  All three are log10-encoded to match X_tmp.
    """
    N  = len(dpuff_arr)
    params = np.empty((N, 8), dtype=np.float64)
    params[:, 0] = fixed['R']
    params[:, 1] = fixed['B']
    params[:, 2] = fixed['Pin']
    params[:, 3] = np.log10(dpuff_arr)         # log10-encoded
    params[:, 4] = np.log10(npuff_fixed)       # log10-encoded
    params[:, 5] = np.log10(fixed['Dcore'])    # log10-encoded
    params[:, 6] = fixed['D_perp']
    params[:, 7] = fixed['Chi_perp']
    return params.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# KDE helper for sample distribution plots
# ─────────────────────────────────────────────────────────────────────────────

def kde_log(samples_1d: np.ndarray, n_pts: int = 200):
    """Gaussian KDE in log10-space.  Returns (x_grid, density)."""
    log_s = np.log10(np.maximum(samples_1d, 1e-10))
    bw    = 1.06 * log_s.std() * len(log_s) ** (-0.2)     # Silverman's rule
    bw    = max(bw, 0.01)
    x_log = np.linspace(log_s.min() - 0.5, log_s.max() + 0.5, n_pts)
    # Gaussian KDE manually (no scipy dependency)
    diff  = x_log[:, None] - log_s[None, :]               # (n_pts, n_samp)
    density = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1)
    density /= (density.max() + 1e-30)                     # normalise to 1
    return 10.0 ** x_log, density


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_scan(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")

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

    # Locate the Te channel (index 0) in the prediction array
    if 0 not in y_indices:
        raise ValueError(
            "Te (channel index 0) is not in the model's y_indices. "
            "This scan requires Te predictions."
        )
    j_Te = y_indices.index(0)
    print(f"  Te is at prediction position j_Te={j_Te}")

    # ── 2.  Build model ──────────────────────────────────────────────────────
    model = VelocityUNet(
        c_in=c_in, c_out=c_out, base=base, t_dim=t_dim, n_views=3
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── 3.  Load geometry (b2fgmtry) ────────────────────────────────────────
    geo = Geometry(args.b2fgmtry)
    ix_ot, ix_omp, iy_sep = get_separatrix_indices(geo)
    print(f"  Geometry   : ix_ot={ix_ot}  ix_omp={ix_omp}  iy_sep={iy_sep}")

    # ── 4.  Load view masks and geometry channels from existing tensor ────────
    # Channel 0 of X_img is the static binary mask — identical for every
    # simulation in the dataset.  We read it once from index [0].
    view_masks: Dict[int, np.ndarray] = {}
    view_geoms: Dict[int, Optional[np.ndarray]] = {}
    for v in range(3):
        x_path = Path(
            f"{args.tensor_prefix}_view{v}_X_img_{args.tensor_split}.npy"
        )
        if not x_path.exists():
            raise FileNotFoundError(f"Tensor not found: {x_path}")
        X_ref = np.load(x_path, mmap_mode="r")
        view_masks[v] = np.array(X_ref[0, 0], dtype=np.float32)  # (H, W)
        view_geoms[v] = (
            np.array(X_ref[0, 9:11], dtype=np.float32)           # (2, H, W)
            if c_in >= 11 else None
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
    # Embed geometry indices so scan_to_te_mesh can extract scalars internally
    inv_maps["_ix_ot"]  = ix_ot
    inv_maps["_ix_omp"] = ix_omp
    inv_maps["_iy_sep"] = iy_sep

    # ── 6.  Define scan grids ─────────────────────────────────────────────────
    fixed = dict(
        R=args.R, B=args.B, Pin=args.Pin,
        Dcore=args.Dcore, D_perp=args.D_perp, Chi_perp=args.Chi_perp,
    )
    dpuff_vals = np.logspace(
        np.log10(args.dpuff_min), np.log10(args.dpuff_max), args.n_dpuff)
    npuff_vals = np.logspace(
        np.log10(args.npuff_min), np.log10(args.npuff_max), args.n_npuff)
    dpuff_ood  = np.logspace(
        np.log10(args.dpuff_min), np.log10(args.dpuff_ood_max), args.n_dpuff_ood)

    print(f"\n  2D grid  : {args.n_dpuff} x {args.n_npuff} = "
          f"{args.n_dpuff * args.n_npuff} points")
    print(f"  D_puff   : {args.dpuff_min:.1e} - {args.dpuff_max:.1e}  (stored as log10: {np.log10(args.dpuff_min):.1f} - {np.log10(args.dpuff_max):.1f})")
    print(f"  N_puff   : {args.npuff_min:.1e} - {args.npuff_max:.1e}  (stored as log10: {np.log10(args.npuff_min):.1f} - {np.log10(args.npuff_max):.1f})")
    print(f"  N_puff0  : {args.npuff_fixed:.1e}  -> log10 = {np.log10(args.npuff_fixed):.2f}")
    print(f"  Dcore    : {args.Dcore:.3e}  -> log10 = {np.log10(args.Dcore):.2f}")
    print(f"  Pin      : {args.Pin:.3e} W")
    print(f"  N samples: {args.n_samples}")
    print(f"  OOD max  : {args.dpuff_ood_max:.1e}")

    # Sanity check: verify encoded values fall inside training range
    print(f"\n  Sanity check — X_tmp training ranges:")
    print(f"  {'param':<12} {'encoded_value':>15}  {'train_min':>10}  {'train_max':>10}  {'in range?':>10}")
    checks = [
        ("R",        args.R,                       1.00,  10.0),
        ("B",        args.B,                       1.00,  10.0),
        ("Pin",      args.Pin,                     5e6,   1e8 ),
        ("D_puff_lo",np.log10(args.dpuff_min),     20.0,  24.0),
        ("D_puff_hi",np.log10(args.dpuff_max),     20.0,  24.0),
        ("N_puff",   np.log10(args.npuff_fixed),   18.0,  23.0),
        ("Dcore",    np.log10(args.Dcore),         19.0,  24.0),
        ("D_perp",   args.D_perp,                  0.1,   2.0 ),
        ("Chi_perp", args.Chi_perp,                0.1,   2.0 ),
    ]
    for name, val, lo, hi in checks:
        flag = "OK" if lo <= val <= hi else "*** OUT OF RANGE ***"
        print(f"  {name:<12} {val:>15.4g}  {lo:>10.4g}  {hi:>10.4g}  {flag:>10}")
    print()

    # Shared kwargs for all scan_to_te_mesh calls
    scan_kw = dict(
        model=model, view_masks=view_masks, view_geoms=view_geoms,
        c_in=c_in, c_out=c_out,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        s_c=s_c, eps=eps,
        y_indices=y_indices,
        pos_channels=pos_channels, signed_channels=signed_channels,
        j_Te=j_Te, inv_maps=inv_maps, layout=layout,
        mode=mode, integrator=args.integrator,
        n_samples=args.n_samples, n_steps=args.ode_steps,
        batch_size=args.batch_size, device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 7.  2D grid scan ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  2D gas puff scan ...")
    print("="*60)
    params_2d, ND, NN = build_params_2d(dpuff_vals, npuff_vals, fixed)
    te_mean_2d, te_std_2d, te_ot_samp_2d, _ = scan_to_te_mesh(params_2d, **scan_kw)
    # Scalar extractions — already at (sepa, sepm); reshape grid
    te_ot_mean_2d = te_mean_2d[:, ix_ot,  iy_sep].reshape(ND, NN)  # (ND, NN)
    te_ot_std_2d  = te_std_2d[:, ix_ot,  iy_sep].reshape(ND, NN)

    np.save(out_dir / "te_ot_mean_2d.npy", te_ot_mean_2d)
    np.save(out_dir / "te_ot_std_2d.npy",  te_ot_std_2d)

    # ── 8.  1D scan (training range, fixed N_puff) ───────────────────────────
    print("\n" + "="*60)
    print("  1D D_puff scan (training range) ...")
    print("="*60)
    params_1d = build_params_1d(dpuff_vals, args.npuff_fixed, fixed)
    te_mean_1d, te_std_1d, te_ot_samp_1d, te_omp_samp_1d = scan_to_te_mesh(
        params_1d, **scan_kw)
    te_ot_mean_1d  = te_mean_1d[:, ix_ot,  iy_sep]   # (ND,)
    te_omp_mean_1d = te_mean_1d[:, ix_omp, iy_sep]
    te_ot_std_1d   = te_std_1d[:, ix_ot,  iy_sep]
    te_omp_std_1d  = te_std_1d[:, ix_omp, iy_sep]
    # te_ot_samp_1d / te_omp_samp_1d : (n_samples, ND) — already scalar

    # ── 9.  OOD extended scan ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  OOD extended D_puff scan ...")
    print("="*60)
    params_ood = build_params_1d(dpuff_ood, args.npuff_fixed, fixed)
    te_mean_ood, te_std_ood, te_ot_samp_ood, te_omp_samp_ood = scan_to_te_mesh(
        params_ood, **scan_kw)
    te_ot_mean_ood  = te_mean_ood[:, ix_ot,  iy_sep]
    te_omp_mean_ood = te_mean_ood[:, ix_omp, iy_sep]
    te_ot_std_ood   = te_std_ood[:, ix_ot,  iy_sep]
    te_omp_std_ood  = te_std_ood[:, ix_omp, iy_sep]

    # Save raw arrays for further analysis
    np.save(out_dir / "te_ot_mean_1d.npy",    te_ot_mean_1d)
    np.save(out_dir / "te_omp_mean_1d.npy",   te_omp_mean_1d)
    np.save(out_dir / "te_ot_std_1d.npy",     te_ot_std_1d)
    np.save(out_dir / "te_ot_samples_1d.npy", te_ot_samp_1d)
    np.save(out_dir / "dpuff_vals.npy",        dpuff_vals)
    np.save(out_dir / "npuff_vals.npy",        npuff_vals)
    np.save(out_dir / "dpuff_ood.npy",         dpuff_ood)
    print(f"\n  Raw arrays saved to {out_dir}")

    # ── 10.  Plotting ─────────────────────────────────────────────────────────
    print("\n  Generating figures ...")
    DG, NG = np.meshgrid(dpuff_vals, npuff_vals, indexing='ij')  # (ND, NN)

    # ─ Figure 10.1A equivalent: 2D heatmap Te,ot ─────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    pos_vals = te_ot_mean_2d[te_ot_mean_2d > 0]
    vmin = float(np.percentile(pos_vals, 2))  if len(pos_vals) else 1e-2
    vmax = float(np.percentile(pos_vals, 98)) if len(pos_vals) else 1e4
    c    = ax.pcolormesh(
        DG, NG, te_ot_mean_2d,
        norm=LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, vmin*2)),
        cmap='viridis', shading='nearest',
    )
    plt.colorbar(c, ax=ax, label='Te,ot  [eV]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(args.dpuff_min, args.dpuff_max)
    ax.set_ylim(args.npuff_min, args.npuff_max)
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('N$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_title('FM model  -  Te,ot  (mean prediction)', fontsize=11)
    ax.axhline(args.npuff_fixed, color='black', lw=1.2, ls='--',
               label=f'N_puff = {args.npuff_fixed:.0e}  (1D slice)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_10_1A_te_ot_2D.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_10_1A_te_ot_2D.png")

    # ─ Figure 10.1B equivalent: 1D Te,ot and Te,omp vs D_puff ────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(dpuff_vals, te_ot_mean_1d,  color='C0', lw=2,
            label='Te,ot  (FM mean)')
    ax.plot(dpuff_vals, te_omp_mean_1d, color='C1', lw=2,
            label='Te,omp  (FM mean)')
    ax.fill_between(dpuff_vals,
                    np.maximum(te_ot_mean_1d  - te_ot_std_1d,  1e-2),
                    te_ot_mean_1d  + te_ot_std_1d,
                    color='C0', alpha=0.25, label='±1σ  Te,ot')
    ax.fill_between(dpuff_vals,
                    np.maximum(te_omp_mean_1d - te_omp_std_1d, 1e-2),
                    te_omp_mean_1d + te_omp_std_1d,
                    color='C1', alpha=0.25, label='±1σ  Te,omp')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('D$_{puff}$  [atoms/s]', fontsize=11)
    ax.set_ylabel('Electron temperature  [eV]', fontsize=11)
    ax.set_title(
        f'FM model  –  N_puff = {args.npuff_fixed:.0e} atoms/s', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=5e-1)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_10_1B_te_1D_scan.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_10_1B_te_1D_scan.png")

    # ─ Uncertainty maps: std and CV over 2D grid ─────────────────────────────
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

    fig.suptitle('FM uncertainty over gas puff parameter space', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_uncertainty_2D.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_uncertainty_2D.png")

    # ─ KDE of sample distribution at selected D_puff points ──────────────────
    # Detect the detachment transition: steepest drop in log(Te,ot)
    log_te_ot   = np.log10(np.maximum(te_ot_mean_1d, 1e-10))
    delta_log   = np.abs(np.diff(log_te_ot))
    trans_idx   = int(np.argmax(delta_log))  # point just before steepest drop

    # Select a spread of D_puff points including the transition region
    n_pts   = args.n_dpuff
    sel_idx = sorted(set([
        0,
        n_pts // 5,
        n_pts // 2,
        max(0, trans_idx - 1),
        trans_idx,
        min(n_pts - 1, trans_idx + 1),
        min(n_pts - 1, trans_idx + 2),
        n_pts - 1,
    ]))
    sel_idx = [i for i in sel_idx if 0 <= i < n_pts]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    cmap_pts = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(sel_idx)))
    for color, idx in zip(cmap_pts, sel_idx):
        samp = te_ot_samp_1d[:, idx]
        samp = samp[samp > 0]
        if len(samp) < 3:
            continue
        dp_val   = dpuff_vals[idx]
        x_grid, density = kde_log(samp)
        ax.plot(x_grid, density, color=color, lw=1.8,
                label=f'D={dp_val:.1e}')
        # Mark the mean
        ax.axvline(te_ot_mean_1d[idx], color=color, lw=0.8, ls=':', alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel('Te,ot  [eV]', fontsize=11)
    ax.set_ylabel('Normalised KDE  (log-space bandwidth)', fontsize=10)
    ax.set_title(
        f'Sample distribution of Te,ot at selected D_puff values\n'
        f'(N_puff = {args.npuff_fixed:.0e}, n_samples = {args.n_samples})',
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    # Annotate the transition region
    if trans_idx < n_pts:
        ax.set_title(ax.get_title() +
                     f'\n⚡ transition near D_puff ≈ {dpuff_vals[trans_idx]:.1e}',
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sample_kde_1D.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_sample_kde_1D.png")

    # ─ OOD extension ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left panel: Te,ot and Te,omp over extended range
    ax = axes[0]
    ax.plot(dpuff_ood, te_ot_mean_ood,  color='C0', lw=2.0, label='Te,ot  (mean)')
    ax.plot(dpuff_ood, te_omp_mean_ood, color='C1', lw=2.0, label='Te,omp  (mean)')
    ax.fill_between(dpuff_ood,
                    np.maximum(te_ot_mean_ood  - te_ot_std_ood,  1e-2),
                    te_ot_mean_ood  + te_ot_std_ood,
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

    # Right panel: uncertainty vs D_puff (OOD)
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
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ood_extension.png", dpi=150)
    plt.close(fig)
    print("    Saved: fig_ood_extension.png")

    # ── Done ─────────────────────────────────────────────────────────────────
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
        description="Gas puff scan + FM uncertainty analysis (Fig 10.1 reproduction).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    ap.add_argument("--scripts_dir",  required=True,
                    help="Path to the scripts/ folder containing "
                         "infer_fm_fusvel_v6.py, reconstruct_mesh.py, utils.py")
    ap.add_argument("--checkpoint",    required=True,
                    help="Path to checkpoint_best.pt")
    ap.add_argument("--tensor_prefix", required=True,
                    help="Prefix for X_img tensors, "
                         "e.g. scripts/tensor/train/global3")
    ap.add_argument("--layout_path",   required=True,
                    help="Path to *_layout_map_3views.npz")
    ap.add_argument("--unroller",      required=True,
                    help="Path to unrolled_strip_clockwise_adjpreserve.py")
    ap.add_argument("--b2fgmtry",      required=True,
                    help="Path to b2fgmtry geometry file")
    ap.add_argument("--out_dir",       required=True,
                    help="Output directory for figures and .npy arrays")

    # ── Fixed ITER parameters (thesis p.180) ──────────────────────────────────
    ap.add_argument("--R",        type=float, default=6.2,
                    help="Major radius R [m]")
    ap.add_argument("--B",        type=float, default=5.3,
                    help="Toroidal field B [T]")
    ap.add_argument("--Pin",      type=float, default=100e6,
                    help="Input power Pin [W].  100 MW → 100e6.  "
                         "⚠️  Verify units match your X_tmp.npy.")
    ap.add_argument("--Dcore",    type=float, default=9.1e21,
                    help="Core fueling Dcore [atoms/s]")
    ap.add_argument("--D_perp",   type=float, default=0.3,
                    help="Cross-field particle transport D_perp [m^2/s]")
    ap.add_argument("--Chi_perp", type=float, default=1.0,
                    help="Cross-field heat transport Chi_perp [m^2/s]")

    # ── Scan grid ─────────────────────────────────────────────────────────────
    ap.add_argument("--n_dpuff",       type=int,   default=40,
                    help="Number of D_puff points in the scan")
    ap.add_argument("--n_npuff",       type=int,   default=35,
                    help="Number of N_puff points in the scan")
    ap.add_argument("--dpuff_min",     type=float, default=1e20,
                    help="Minimum D_puff [atoms/s]")
    ap.add_argument("--dpuff_max",     type=float, default=1e24,
                    help="Maximum D_puff for training-range scan [atoms/s]")
    ap.add_argument("--npuff_min",     type=float, default=1e18,
                    help="Minimum N_puff [atoms/s]")
    ap.add_argument("--npuff_max",     type=float, default=1e23,
                    help="Maximum N_puff [atoms/s]")
    ap.add_argument("--npuff_fixed",   type=float, default=1e20,
                    help="Fixed N_puff value for the 1D D_puff scan [atoms/s]")
    ap.add_argument("--dpuff_ood_max", type=float, default=1e25,
                    help="Maximum D_puff for OOD extended scan [atoms/s]")
    ap.add_argument("--n_dpuff_ood",   type=int,   default=50,
                    help="Number of D_puff points for OOD scan")

    # ── Inference settings ────────────────────────────────────────────────────
    ap.add_argument("--n_samples",    type=int,   default=30,
                    help="Number of stochastic samples per input point.  "
                         "More → lower variance in uncertainty estimates.  "
                         "30–50 is a good range.")
    ap.add_argument("--ode_steps",    type=int,   default=50,
                    help="Number of ODE integration steps per sample")
    ap.add_argument("--batch_size",   type=int,   default=64,
                    help="Batch size for model inference")
    ap.add_argument("--mode",         default=None,
                    choices=["cfm", "xpred", "direct"],
                    help="Override inference mode (auto-detected from checkpoint)")
    ap.add_argument("--integrator",   default="stochastic",
                    choices=["euler", "stochastic"],
                    help="Integration method.  'stochastic' is required for "
                         "meaningful uncertainty estimates.")
    ap.add_argument("--tensor_split", default="train",
                    help="Split name used when loading the reference mask tensor "
                         "('train' or 'test')")
    ap.add_argument("--device",       default=None,
                    help="Torch device (auto-detected if omitted)")
    ap.add_argument("--seed",         type=int, default=42)

    args = ap.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
