#!/usr/bin/env python3
"""
plot_improvements.py
────────────────────────────────────────────────────────────────────────────
Post-processing companion to scan_gas_puff.py.
Reads cached .npy arrays from a previous scan run and produces seven
additional / improved figures that give a much richer picture of the
FM model's uncertainty behaviour.

NEW FIGURES
───────────
  A  fig_stochasticity_check.png   — Are the 30 samples actually different?
                                     Strip plot of repeated forward passes
                                     at one input point (validates FM noise)

  B  fig_1d_percentile.png         — 1D scan with 5th/50th/95th percentile
                                     bands instead of ±1σ; also overlays
                                     every individual sample as a faint line
                                     (spaghetti plot)

  C  fig_sample_strip_1d.png       — All 30 sample dots per D_puff value
                                     stacked vertically. Far more honest than
                                     KDE with only 30 points.

  D  fig_2D_with_test_scatter.png  — 2D heatmap of Te,ot mean overlaid with
                                     test-set ground-truth scatter points.
                                     Requires --test_data_dir.

  E  fig_1d_with_test_scatter.png  — 1D scan lines + test-set scatter at the
                                     fixed N_puff slice. Requires
                                     --test_data_dir.

  F  fig_spatial_field_gallery.png — Full 104×50 Te field (mean and std) at
                                     four representative (D_puff, N_puff)
                                     operating points: sheath-limited,
                                     pre-transition, at-transition, detached.
                                     Needs model re-inference.

  G  fig_spatial_uncertainty_transition.png
                                   — Pixel-wise std field at the transition
                                     D_puff. Shows WHERE spatially the model
                                     is most uncertain.

  H  fig_npuff_slice_uncertainty.png
                                   — std(Te,ot) vs N_puff at fixed D_puff
                                     near the transition. Investigates the
                                     horizontal band artefact in uncertainty
                                     2D maps.

USAGE
─────
python scripts/visualizations/plot_improvements.py \\
    --scripts_dir   scripts \\
    --out_dir       D:/thesis/solps-nn/FM_predictions/gas_puff_scan \\
    --checkpoint    scripts/runs/fusvel_v7_xpred_selfcond_continue/checkpoint_best.pt \\
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \\
    --layout_path   scripts/tensor/3views_4d/train/global3v_layout_map_3views.npz \\
    --unroller      unrolled_strip_clockwise_adjpreserve.py \\
    --b2fgmtry      geometry/b2fgmtry \\
    --test_data_dir train          # optional: path to train/ or test/ folder
                                   # containing X_tmp.npy and te_tmp.npy
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution (identical to scan_gas_puff.py)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_scripts_dir() -> Path:
    for i, tok in enumerate(sys.argv):
        if tok in ("--scripts_dir", "--scripts-dir") and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
    return Path(__file__).resolve().parent.parent

SCRIPTS_DIR = Path(__file__).resolve().parent
PARENT_DIR  = SCRIPTS_DIR.parent
UTIL_DIR = PARENT_DIR.parent
PARENT_DIR = PARENT_DIR / "models" / "FM_fin"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(UTIL_DIR))

from infer_fm_fusvel_v6 import (
    VelocityUNet, stochastic_integrate_xpred,
    euler_integrate_xpred, euler_integrate_velocity, direct_forward,
    inverse_transform_y, POS_CHANNELS, SIGNED_CHANNELS,
)
from reconstruct_mesh import (
    build_strip, compute_view_layouts, build_inverse_maps,
    reconstruct_to_mesh, SLOT_TO_SPECNAME, SLOT_TF,
)
from utils import Geometry


# ─────────────────────────────────────────────────────────────────────────────
# Shared model infrastructure (mirrors scan_gas_puff.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_stats(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
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
    model  = VelocityUNet(c_in=c_in, c_out=c_out, base=base, t_dim=t_dim,
                          n_views=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    j_Te = y_indices.index(0) if 0 in y_indices else None
    return (model, c_in, c_out, y_indices, j_Te, pos_channels, signed_channels,
            eps, x_mean, x_std, y_mean, y_std, s_c, mode)


def load_inv_maps(layout_path, unroller_arg, gap_px, ix_ot, ix_omp, iy_sep):
    layout = dict(np.load(layout_path, allow_pickle=True))
    up     = Path(unroller_arg)
    sys.path.insert(0, str(up.parent.resolve()) if up.suffix == ".py"
                    else str(Path(".").resolve()))
    u            = importlib.import_module(up.stem if up.suffix == ".py"
                                           else unroller_arg)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}
    strip_dicts  = {slot: build_strip(u, name_to_spec[spec], tf=SLOT_TF[slot])
                    for slot, spec in SLOT_TO_SPECNAME.items()}
    origins_view, _ = compute_view_layouts(strip_dicts, gap_px=gap_px)
    inv_maps        = build_inverse_maps(strip_dicts, origins_view)
    inv_maps["_ix_ot"]  = ix_ot
    inv_maps["_ix_omp"] = ix_omp
    inv_maps["_iy_sep"] = iy_sep
    return inv_maps, layout


def load_view_masks(tensor_prefix, tensor_split, c_in):
    view_masks, view_geoms = {}, {}
    for v in range(3):
        p  = Path(f"{tensor_prefix}_view{v}_X_img_{tensor_split}.npy")
        if not p.exists():
            raise FileNotFoundError(f"Tensor not found: {p}")
        X  = np.load(p, mmap_mode="r")
        view_masks[v] = np.array(X[0, 0], dtype=np.float32)
        view_geoms[v] = (np.array(X[0, 9:11], dtype=np.float32)
                         if c_in >= 11 else None)
    return view_masks, view_geoms


def build_x_img_batch(params_arr, mask_v, geom_v, c_in, x_mean, x_std):
    N, H, W = params_arr.shape[0], *mask_v.shape
    X_raw = np.zeros((N, c_in, H, W), dtype=np.float32)
    X_raw[:, 0] = mask_v[None]
    for k in range(8):
        X_raw[:, 1+k] = params_arr[:, k, None, None].astype(np.float32) * mask_v[None]
    if geom_v is not None and c_in >= 11:
        X_raw[:, 9:11] = geom_v[None]
    masks  = X_raw[:, 0:1].copy()
    X_norm = (X_raw - x_mean[None, :, None, None]) / x_std[None, :, None, None]
    return X_norm.astype(np.float32), masks.astype(np.float32)


def run_one_sample(model, view_X, view_M, inv_maps, layout,
                   c_out, j_Te, y_indices, pos_channels, signed_channels,
                   y_mean, y_std, eps, s_c, mode, integrator,
                   n_steps, batch_size, device):
    """Single stochastic forward pass → (N_scan, 104, 50) Te field."""
    pred_views = {}
    for v in range(3):
        X_norm, masks = view_X[v], view_M[v]
        N_scan = X_norm.shape[0]
        preds_norm = np.zeros((N_scan, c_out, *X_norm.shape[2:]), dtype=np.float32)
        for b0 in range(0, N_scan, batch_size):
            b1   = min(b0 + batch_size, N_scan)
            x_b  = torch.from_numpy(X_norm[b0:b1]).to(device)
            m_b  = torch.from_numpy(masks[b0:b1]).to(device)
            vids = torch.full((b1-b0,), v, dtype=torch.long, device=device)
            with torch.no_grad():
                if mode == "direct":
                    pred = direct_forward(model, x_b, m_b, vids,
                                          c_out=c_out, device=device)
                elif mode == "xpred" and integrator == "stochastic":
                    pred = stochastic_integrate_xpred(model, x_b, m_b, vids,
                                                      c_out=c_out, n_steps=n_steps,
                                                      device=device)
                elif mode == "xpred":
                    pred = euler_integrate_xpred(model, x_b, m_b, vids,
                                                 c_out=c_out, n_steps=n_steps,
                                                 device=device)
                else:
                    pred = euler_integrate_velocity(model, x_b, m_b, vids,
                                                    c_out=c_out, n_steps=n_steps,
                                                    device=device)
            preds_norm[b0:b1] = pred.cpu().numpy()
        # denormalise
        preds_phys = np.empty_like(preds_norm)
        for b0 in range(0, N_scan, batch_size):
            b1 = min(b0 + batch_size, N_scan)
            pt = (preds_norm[b0:b1] * y_std[None, :, None, None]
                  + y_mean[None, :, None, None])
            preds_phys[b0:b1] = inverse_transform_y(
                pt, y_indices, pos_channels, signed_channels, eps, s_c)
            preds_phys[b0:b1] *= masks[b0:b1]
        pred_views[f"view{v}"] = preds_phys

    mesh_out, _ = reconstruct_to_mesh(pred_views, inv_maps, layout)
    return mesh_out[:, j_Te, :, :]   # (N_scan, 104, 50)


def make_params_row(dpuff, npuff, fixed):
    """Single row of params (1, 8) with log10 encoding where needed."""
    p = np.array([[
        fixed["R"], fixed["B"], fixed["Pin"],
        np.log10(dpuff), np.log10(npuff), np.log10(fixed["Dcore"]),
        fixed["D_perp"], fixed["Chi_perp"],
    ]], dtype=np.float32)
    return p


def make_params_1d(dpuff_arr, npuff_fixed, fixed):
    N = len(dpuff_arr)
    p = np.empty((N, 8), dtype=np.float64)
    p[:, 0] = fixed["R"];   p[:, 1] = fixed["B"];   p[:, 2] = fixed["Pin"]
    p[:, 3] = np.log10(dpuff_arr); p[:, 4] = np.log10(npuff_fixed)
    p[:, 5] = np.log10(fixed["Dcore"])
    p[:, 6] = fixed["D_perp"]; p[:, 7] = fixed["Chi_perp"]
    return p.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A  —  Stochasticity verification
# ─────────────────────────────────────────────────────────────────────────────

def fig_stochasticity_check(args, model, view_masks, view_geoms, inv_maps,
                             layout, c_in, c_out, j_Te, y_indices,
                             pos_channels, signed_channels,
                             y_mean, y_std, eps, s_c, mode,
                             ix_ot, ix_omp, iy_sep, fixed, device, out_dir):
    """
    Run the model N_REPEAT times on a single fixed input point and plot
    the distribution of Te,ot values.  If all values are identical the
    stochastic integrator is effectively deterministic for this model.
    """
    print("\n  [A] Stochasticity check ...")
    N_REPEAT = args.n_samples
    dpuff_vals = np.load(out_dir / "dpuff_vals.npy")
    # Use the transition point (steepest drop in mean Te,ot)
    te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
    delta = np.abs(np.diff(np.log10(np.maximum(te_ot_mean_1d, 1e-10))))
    trans_idx = int(np.argmax(delta))
    dpuff_test = dpuff_vals[trans_idx]
    npuff_test = args.npuff_fixed
    print(f"     Test point: D_puff={dpuff_test:.2e}  N_puff={npuff_test:.1e}")

    params = make_params_row(dpuff_test, npuff_test, fixed)
    view_X, view_M = {}, {}
    for v in range(3):
        view_X[v], view_M[v] = build_x_img_batch(
            params, view_masks[v], view_geoms[v], c_in,
            np.load(out_dir / ".x_mean.npy"), np.load(out_dir / ".x_std.npy"))

    te_ot_stoch, te_ot_euler = [], []
    te_omp_stoch, te_omp_euler = [], []

    for rep in range(N_REPEAT):
        print(f"     Repeat {rep+1}/{N_REPEAT}", end="\r", flush=True)
        te_s = run_one_sample(model, view_X, view_M, inv_maps, layout,
                              c_out, j_Te, y_indices, pos_channels,
                              signed_channels, y_mean, y_std, eps, s_c,
                              mode, "stochastic", args.ode_steps,
                              args.batch_size, device)
        te_ot_stoch.append(float(te_s[0, ix_ot,  iy_sep]))
        te_omp_stoch.append(float(te_s[0, ix_omp, iy_sep]))

        te_e = run_one_sample(model, view_X, view_M, inv_maps, layout,
                              c_out, j_Te, y_indices, pos_channels,
                              signed_channels, y_mean, y_std, eps, s_c,
                              mode, "euler", args.ode_steps,
                              args.batch_size, device)
        te_ot_euler.append(float(te_e[0, ix_ot,  iy_sep]))
        te_omp_euler.append(float(te_e[0, ix_omp, iy_sep]))
    print()

    te_ot_stoch  = np.array(te_ot_stoch)
    te_ot_euler  = np.array(te_ot_euler)
    te_omp_stoch = np.array(te_omp_stoch)
    te_omp_euler = np.array(te_omp_euler)   # FIX: was never converted

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, vals_s, vals_e, loc_name in [
        (axes[0], te_ot_stoch,  te_ot_euler,  "Te,ot  (sepa)"),
        (axes[1], te_omp_stoch, te_omp_euler, "Te,omp  (sepm)"),   # FIX
    ]:
        jitter = np.random.uniform(-0.15, 0.15, N_REPEAT)
        ax.scatter(np.ones(N_REPEAT)*0 + jitter, vals_s,
                   color="C0", alpha=0.6, s=30, label="Stochastic integrator")
        ax.scatter(np.ones(N_REPEAT)*1 + jitter, vals_e,
                   color="C1", alpha=0.6, s=30, label="Euler (deterministic)")
        ax.axhline(vals_s.mean(), color="C0", lw=2, ls="--",
                   label=f"Stoch mean={vals_s.mean():.2f}")
        ax.axhline(vals_e.mean(), color="C1", lw=2, ls="--",
                   label=f"Euler mean={vals_e.mean():.2f}")
        # annotate std
        ax.text(0, vals_s.mean()*1.05,
                f"std={vals_s.std():.3f}\nCV={vals_s.std()/max(abs(vals_s.mean()),1e-9):.4f}",
                ha="center", fontsize=8, color="C0")
        ax.text(1, vals_e.mean()*1.05,
                f"std={vals_e.std():.3f}\nCV={vals_e.std()/max(abs(vals_e.mean()),1e-9):.4f}",
                ha="center", fontsize=8, color="C1")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stochastic", "Euler (deterministic)"])
        ax.set_ylabel(f"{loc_name}  [eV]")
        ax.set_title(f"{loc_name}  —  {N_REPEAT} repeated passes\n"
                     f"D_puff={dpuff_test:.1e}  N_puff={npuff_test:.1e}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "FM stochasticity check: are repeated forward passes actually different?\n"
        "(If all dots cluster at the same value, the sampler is effectively deterministic)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_A_stochasticity_check.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_A_stochasticity_check.png")

    # Save for downstream
    np.save(out_dir / "stoch_check_te_ot.npy", te_ot_stoch)
    np.save(out_dir / "stoch_check_te_ot_euler.npy", te_ot_euler)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B  —  1D scan with percentile envelopes + spaghetti
# ─────────────────────────────────────────────────────────────────────────────

def fig_1d_percentile(args, out_dir):
    print("\n  [B] 1D percentile scan ...")
    dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
    te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
    te_omp_mean_1d= np.load(out_dir / "te_omp_mean_1d.npy")
    samp_path     = out_dir / "te_ot_samples_1d.npy"
    omp_samp_path = out_dir / "te_omp_samples_1d.npy"

    fig, ax = plt.subplots(figsize=(8, 5))

    # ─ spaghetti: individual sample traces ───────────────────────────────────
    if samp_path.exists():
        samp = np.load(samp_path)   # (n_samples, ND)
        for s in range(min(samp.shape[0], 30)):
            ax.plot(dpuff_vals, np.maximum(samp[s], 1e-2),
                    color="C0", alpha=0.06, lw=0.8)
        # percentile bands
        p5  = np.percentile(samp, 5,  axis=0)
        p50 = np.percentile(samp, 50, axis=0)
        p95 = np.percentile(samp, 95, axis=0)
        ax.fill_between(dpuff_vals, np.maximum(p5, 1e-2), np.maximum(p95, 1e-2),
                        color="C0", alpha=0.20, label="Te,ot  5th–95th pct")
        ax.plot(dpuff_vals, np.maximum(p50, 1e-2),
                color="C0", lw=1.5, ls="--", alpha=0.7, label="Te,ot  median")

    if omp_samp_path.exists():
        samp_omp = np.load(omp_samp_path)
        p5o  = np.percentile(samp_omp, 5,  axis=0)
        p95o = np.percentile(samp_omp, 95, axis=0)
        p50o = np.percentile(samp_omp, 50, axis=0)
        ax.fill_between(dpuff_vals, np.maximum(p5o, 1e-2), np.maximum(p95o, 1e-2),
                        color="C1", alpha=0.20, label="Te,omp  5th–95th pct")
        ax.plot(dpuff_vals, np.maximum(p50o, 1e-2),
                color="C1", lw=1.5, ls="--", alpha=0.7, label="Te,omp  median")

    # ─ mean lines ─────────────────────────────────────────────────────────────
    ax.plot(dpuff_vals, te_ot_mean_1d,  color="C0", lw=2.5, label="Te,ot  mean")
    ax.plot(dpuff_vals, te_omp_mean_1d, color="C1", lw=2.5, label="Te,omp  mean")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(dpuff_vals[0], dpuff_vals[-1])
    ax.set_ylim(bottom=5e-1)
    ax.set_xlabel("D$_{puff}$  [atoms/s]", fontsize=11)
    ax.set_ylabel("Electron temperature  [eV]", fontsize=11)
    ax.set_title(f"FM model  —  N_puff = {args.npuff_fixed:.0e} atoms/s\n"
                 f"Mean + spaghetti + 5th/95th percentile  (n={args.n_samples})",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_B_1d_percentile.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_B_1d_percentile.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C  —  Strip plot of all samples per D_puff value
# ─────────────────────────────────────────────────────────────────────────────

def fig_sample_strip(args, out_dir):
    print("\n  [C] Sample strip plot ...")
    samp_path = out_dir / "te_ot_samples_1d.npy"
    if not samp_path.exists():
        print("     SKIP: te_ot_samples_1d.npy not found")
        return
    samp       = np.load(samp_path)         # (n_samples, ND)
    dpuff_vals = np.load(out_dir / "dpuff_vals.npy")
    te_ot_mean = np.load(out_dir / "te_ot_mean_1d.npy")
    ND         = len(dpuff_vals)
    n_samp     = samp.shape[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color each dot by its D_puff value
    cmap  = plt.cm.plasma
    norms = Normalize(vmin=np.log10(dpuff_vals[0]),
                      vmax=np.log10(dpuff_vals[-1]))

    for i in range(ND):
        dp_log = np.log10(dpuff_vals[i])
        color  = cmap(norms(dp_log))
        jitter = np.random.uniform(-0.2, 0.2, n_samp) * (dpuff_vals[i] * 0.02)
        ax.scatter(dpuff_vals[i] + jitter,
                   np.maximum(samp[:, i], 1e-2),
                   color=color, alpha=0.5, s=14, zorder=3)

    ax.plot(dpuff_vals, te_ot_mean, color="black", lw=2.5,
            zorder=5, label="Mean")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(dpuff_vals[0]*0.8, dpuff_vals[-1]*1.2)
    ax.set_ylim(bottom=5e-1)
    ax.set_xlabel("D$_{puff}$  [atoms/s]", fontsize=11)
    ax.set_ylabel("Te,ot  [eV]", fontsize=11)
    ax.set_title(f"All {n_samp} FM samples at each D_puff value  "
                 f"(N_puff={args.npuff_fixed:.0e})\n"
                 "Spread of dots = FM uncertainty; tight cluster = overconfident model",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(
        vmin=np.log10(dpuff_vals[0]), vmax=np.log10(dpuff_vals[-1])))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="log$_{10}$(D_puff)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_C_sample_strip_1d.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_C_sample_strip_1d.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D  —  2D heatmap overlaid with test-set ground-truth scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig_2d_with_test_scatter(args, out_dir, ix_ot, ix_omp, iy_sep):
    print("\n  [D] 2D map + test scatter ...")
    if not args.test_data_dir:
        print("     SKIP: --test_data_dir not provided")
        return
    test_dir = Path(args.test_data_dir)
    X_test_path  = test_dir / "X_tmp.npy"
    te_test_path = test_dir / "te_tmp.npy"
    if not X_test_path.exists() or not te_test_path.exists():
        print(f"     SKIP: X_tmp.npy or te_tmp.npy not found in {test_dir}")
        return

    te_ot_mean_2d = np.load(out_dir / "te_ot_mean_2d.npy")   # (ND, NN)
    dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
    npuff_vals    = np.load(out_dir / "npuff_vals.npy")
    ND, NN        = len(dpuff_vals), len(npuff_vals)
    DG, NG        = np.meshgrid(dpuff_vals, npuff_vals, indexing="ij")

    # Load test data
    J_PER_EV = 1.602176634e-19
    X_test  = np.load(X_test_path)    # (N_test, 8)
    te_test = np.load(te_test_path) / J_PER_EV   # convert J → eV
    # Columns 3, 4 are log10(D_puff), log10(N_puff)
    dp_test    = 10.0 ** X_test[:, 3]
    np_test    = 10.0 ** X_test[:, 4]
    te_ot_test = te_test[:, ix_ot, iy_sep]   # ground truth Te at sepa [eV]

    # Filter out guard-cell artefacts (unphysical values outside expected range)
    phys_mask_d = (te_ot_test > 0.5) & (te_ot_test < 15000.0)
    dp_test    = dp_test[phys_mask_d]
    np_test    = np_test[phys_mask_d]
    te_ot_test = te_ot_test[phys_mask_d]
    print(f"     Figure D: {phys_mask_d.sum()} / {len(phys_mask_d)} simulations "
          f"pass physical filter  "
          f"({'train' if 'train' in str(test_dir) else 'test'} set)")

    pos_vals = te_ot_mean_2d[te_ot_mean_2d > 0]
    vmin = max(float(np.percentile(pos_vals, 2)), 1e-2)
    vmax = float(np.percentile(pos_vals, 98))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, show_gt, title_sfx in [
        (axes[0], False, "FM mean prediction"),
        (axes[1], True,  "FM mean + ground-truth scatter"),
    ]:
        c = ax.pcolormesh(DG, NG, te_ot_mean_2d,
                          norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap="viridis", shading="nearest")
        plt.colorbar(c, ax=ax, label="Te,ot  [eV]")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(dpuff_vals[0], dpuff_vals[-1])
        ax.set_ylim(npuff_vals[0], npuff_vals[-1])
        ax.set_xlabel("D$_{puff}$  [atoms/s]")
        ax.set_ylabel("N$_{puff}$  [atoms/s]")
        ax.axhline(args.npuff_fixed, color="white", lw=1, ls="--", alpha=0.6)
        if show_gt:
            # Small markers so the underlying heatmap stays visible
            sc = ax.scatter(dp_test, np_test, c=te_ot_test,
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            cmap="viridis", edgecolors="none",
                            s=8, alpha=0.7, zorder=5,
                            label="Ground truth (test set)")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title_sfx, fontsize=10)

    fig.suptitle("Te,ot  —  FM model vs test-set ground truth", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_D_2D_with_test_scatter.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_D_2D_with_test_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E  —  1D scan + test-set scatter at fixed N_puff slice
# ─────────────────────────────────────────────────────────────────────────────

def fig_1d_with_test_scatter(args, out_dir, ix_ot, ix_omp, iy_sep):
    print("\n  [E] 1D scan + test scatter ...")
    if not args.test_data_dir:
        print("     SKIP: --test_data_dir not provided")
        return
    test_dir = Path(args.test_data_dir)
    X_test_path  = test_dir / "X_tmp.npy"
    te_test_path = test_dir / "te_tmp.npy"
    if not X_test_path.exists() or not te_test_path.exists():
        print(f"     SKIP: {test_dir}")
        return

    dpuff_vals     = np.load(out_dir / "dpuff_vals.npy")
    te_ot_mean_1d  = np.load(out_dir / "te_ot_mean_1d.npy")
    te_omp_mean_1d = np.load(out_dir / "te_omp_mean_1d.npy")
    te_ot_std_1d   = np.load(out_dir / "te_ot_std_1d.npy")

    X_test  = np.load(X_test_path)
    J_PER_EV = 1.602176634e-19
    te_test = np.load(te_test_path) / J_PER_EV   # convert J → eV
    dp_test   = 10.0 ** X_test[:, 3]
    np_puff   = 10.0 ** X_test[:, 4]

    # Narrow window: 0.3 log-decade keeps points close to the fixed slice
    # so the scatter is actually comparable to the model line
    log_npuff_fixed = np.log10(args.npuff_fixed)
    window = 0.3
    mask_np = np.abs(np.log10(np_puff) - log_npuff_fixed) < window
    print(f"     N_puff filter: log10={log_npuff_fixed:.1f} ±{window}  "
          f"→ {mask_np.sum()} / {len(mask_np)} test points pass")
    dp_sel   = dp_test[mask_np]
    np_sel   = np_puff[mask_np]
    te_ot_sel  = te_test[mask_np, ix_ot,  iy_sep]   # [eV]
    te_omp_sel = te_test[mask_np, ix_omp, iy_sep]   # [eV]
    if mask_np.sum() > 0:
        print(f"     Te,ot  range in selection: {te_ot_sel.min():.1f} – {te_ot_sel.max():.1f} eV")
        print(f"     Te,omp range in selection: {te_omp_sel.min():.1f} – {te_omp_sel.max():.1f} eV")

    # Filter out unphysical guard-cell values (outside the range seen in training)
    # Guard cells in SOLPS raw data can have extreme values at boundary rows
    phys_min, phys_max = 0.5, 15000.0   # eV — reasonable physical range
    phys_mask = ((te_ot_sel > phys_min)  & (te_ot_sel  < phys_max) &
                 (te_omp_sel > phys_min) & (te_omp_sel < phys_max))
    dp_sel     = dp_sel[phys_mask]
    np_sel     = np_sel[phys_mask]
    te_ot_sel  = te_ot_sel[phys_mask]
    te_omp_sel = te_omp_sel[phys_mask]
    print(f"     After physical filter ({phys_min}–{phys_max} eV): "
          f"{phys_mask.sum()} points remain")

    samp_path = out_dir / "te_ot_samples_1d.npy"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ym, ys, te_gt, loc, color in [
        (axes[0], te_ot_mean_1d,  te_ot_std_1d,  te_ot_sel,  "Te,ot  (sepa)",  "C0"),
        (axes[1], te_omp_mean_1d, te_ot_std_1d*0, te_omp_sel, "Te,omp  (sepm)", "C1"),
    ]:
        if samp_path.exists() and loc == "Te,ot  (sepa)":
            samp = np.load(samp_path)
            p5  = np.percentile(samp, 5,  axis=0)
            p95 = np.percentile(samp, 95, axis=0)
            ax.fill_between(dpuff_vals, np.maximum(p5, 1e-2), np.maximum(p95, 1e-2),
                            color=color, alpha=0.18, label="5th–95th pct")

        ax.plot(dpuff_vals, ym, color=color, lw=2.5, label=f"{loc}  (FM mean)")
        # Test scatter
        sc = ax.scatter(dp_sel, np.maximum(te_gt, 1e-2),
                        c=np.log10(np.maximum(np_sel, 1e-10)),
                        cmap="cool", s=25, zorder=5, edgecolors="none",
                        alpha=0.8, label="Ground truth (test set)")
        plt.colorbar(sc, ax=ax, label="log$_{10}$(N_puff) of test point")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(dpuff_vals[0], dpuff_vals[-1])
        ax.set_ylim(bottom=5e-1)
        ax.set_xlabel("D$_{puff}$  [atoms/s]")
        ax.set_ylabel("Te  [eV]")
        ax.set_title(f"{loc}  —  model vs ground truth\n"
                     f"(test points within 1.0 log-decade of N_puff={args.npuff_fixed:.0e})",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("FM prediction vs test-set ground truth along 1D D_puff slice",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_E_1d_with_test_scatter.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_E_1d_with_test_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F  —  Spatial field gallery (mean Te across 104×50 at 4 op-points)
# ─────────────────────────────────────────────────────────────────────────────

def fig_spatial_field_gallery(args, model, view_masks, view_geoms, inv_maps,
                               layout, c_in, c_out, j_Te, y_indices,
                               pos_channels, signed_channels,
                               y_mean, y_std, eps, s_c, mode,
                               ix_ot, ix_omp, iy_sep, fixed, device, out_dir):
    """
    Four operating points: sheath-limited / pre-transition / at-transition / detached.
    For each: show Welford mean and std of Te field across n_samples passes.
    """
    print("\n  [F] Spatial field gallery ...")

    te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
    dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
    delta         = np.abs(np.diff(np.log10(np.maximum(te_ot_mean_1d, 1e-10))))
    trans_idx     = int(np.argmax(delta))
    ND            = len(dpuff_vals)

    # 4 operating points: sheath-limited, ~2 decades before trans, transition, detached
    op_indices = [
        0,
        max(0, trans_idx - 4),
        trans_idx,
        min(ND - 1, trans_idx + 3),
    ]
    op_labels = ["Sheath-limited", "Pre-transition", "At transition", "Detached"]
    npuff_fixed = args.npuff_fixed
    n_samp = min(args.n_samples, 20)   # cap to keep runtime reasonable

    # Load x normalisation stats (saved during scan)
    x_mean = np.load(out_dir / ".x_mean.npy")
    x_std  = np.load(out_dir / ".x_std.npy")

    # Build Welford mean/std for each op-point
    fields_mean = []
    fields_std  = []
    for op_i, label in zip(op_indices, op_labels):
        dp = dpuff_vals[op_i]
        params = make_params_row(dp, npuff_fixed, fixed)
        vX, vM = {}, {}
        for v in range(3):
            vX[v], vM[v] = build_x_img_batch(
                params, view_masks[v], view_geoms[v], c_in, x_mean, x_std)

        wf_mean = np.zeros((104, 50), dtype=np.float64)
        wf_M2   = np.zeros((104, 50), dtype=np.float64)
        for s in range(n_samp):
            print(f"     {label}  sample {s+1}/{n_samp}", end="\r", flush=True)
            te_s = run_one_sample(model, vX, vM, inv_maps, layout,
                                  c_out, j_Te, y_indices, pos_channels,
                                  signed_channels, y_mean, y_std, eps, s_c,
                                  mode, "stochastic", args.ode_steps,
                                  args.batch_size, device)[0]  # (104,50)
            delta = te_s - wf_mean
            wf_mean += delta / (s + 1)
            wf_M2   += delta * (te_s - wf_mean)
        print()
        fields_mean.append(wf_mean.astype(np.float32))
        fields_std.append(np.sqrt(wf_M2 / max(n_samp - 1, 1)).astype(np.float32))

    # ─ Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.35)

    # Global colour limits across all panels
    all_means = np.concatenate([f.ravel() for f in fields_mean])
    all_std   = np.concatenate([f.ravel() for f in fields_std])
    pos_m = all_means[all_means > 0]
    pos_s = all_std[all_std > 0]
    mean_vmin = max(float(np.percentile(pos_m, 1)) if len(pos_m) else 1e-2, 1e-2)
    mean_vmax = float(np.percentile(pos_m, 99)) if len(pos_m) else 1e4
    std_vmin  = max(float(np.percentile(pos_s, 1)) if len(pos_s) else 1e-4, 1e-4)
    std_vmax  = float(np.percentile(pos_s, 99)) if len(pos_s) else 1e2

    for col, (label, fmean, fstd, op_i) in enumerate(
            zip(op_labels, fields_mean, fields_std, op_indices)):
        dp = dpuff_vals[op_i]

        # Row 0: mean Te field
        ax0 = fig.add_subplot(gs[0, col])
        pos = fmean.copy(); pos[pos <= 0] = np.nan
        im0 = ax0.imshow(pos.T, origin="lower", aspect="auto",
                         norm=LogNorm(vmin=mean_vmin, vmax=mean_vmax),
                         cmap="inferno")
        ax0.axhline(iy_sep,  color="white", lw=1.0, ls="--", alpha=0.8)
        ax0.axvline(ix_ot,   color="cyan",  lw=1.0, ls=":",  alpha=0.8)
        ax0.axvline(ix_omp,  color="lime",  lw=1.0, ls=":",  alpha=0.8)
        ax0.set_title(f"{label}\nD={dp:.1e}", fontsize=9)
        if col == 0:
            ax0.set_ylabel("Radial index", fontsize=8)
        plt.colorbar(im0, ax=ax0, label="Te [eV]" if col == 3 else "")

        # Row 1: std field
        ax1 = fig.add_subplot(gs[1, col])
        pos_s2 = fstd.copy(); pos_s2[pos_s2 <= 0] = np.nan
        im1 = ax1.imshow(pos_s2.T, origin="lower", aspect="auto",
                         norm=LogNorm(vmin=std_vmin, vmax=std_vmax),
                         cmap="hot_r")
        ax1.axhline(iy_sep, color="white", lw=1.0, ls="--", alpha=0.8)
        ax1.axvline(ix_ot,  color="cyan",  lw=1.0, ls=":",  alpha=0.8)
        ax1.axvline(ix_omp, color="lime",  lw=1.0, ls=":",  alpha=0.8)
        if col == 0:
            ax1.set_ylabel("Radial index", fontsize=8)
        ax1.set_xlabel("Poloidal index", fontsize=8)
        plt.colorbar(im1, ax=ax1, label="std(Te) [eV]" if col == 3 else "")

    # Legend for the guide lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="white", ls="--", label=f"iy_sep={iy_sep}"),
        Line2D([0],[0], color="cyan",  ls=":",  label=f"ix_ot={ix_ot}"),
        Line2D([0],[0], color="lime",  ls=":",  label=f"ix_omp={ix_omp}"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.8)
    fig.suptitle(
        f"Spatial Te field: mean (top) and std (bottom) across {n_samp} FM samples\n"
        f"N_puff = {npuff_fixed:.0e}  |  white dashed = separatrix  |  "
        f"cyan = outer target  |  lime = outer midplane",
        fontsize=10)
    fig.savefig(out_dir / "fig_F_spatial_field_gallery.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_F_spatial_field_gallery.png")

    return fields_mean, fields_std, op_indices


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE G  —  Pixel-wise uncertainty map at the transition point
# ─────────────────────────────────────────────────────────────────────────────

def fig_spatial_uncertainty_transition(args, model, view_masks, view_geoms,
                                       inv_maps, layout, c_in, c_out, j_Te,
                                       y_indices, pos_channels, signed_channels,
                                       y_mean, y_std, eps, s_c, mode,
                                       ix_ot, ix_omp, iy_sep,
                                       fixed, device, out_dir,
                                       gallery_fields=None,
                                       gallery_op_indices=None):
    """
    Show the pixel-wise std(Te) field at the transition D_puff and also
    the relative uncertainty = std / mean at each pixel.
    """
    print("\n  [G] Spatial uncertainty at transition ...")
    dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
    te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
    delta         = np.abs(np.diff(np.log10(np.maximum(te_ot_mean_1d, 1e-10))))
    trans_idx     = int(np.argmax(delta))
    dp_trans      = dpuff_vals[trans_idx]

    # Try to reuse from gallery (index 2 = "At transition")
    if (gallery_fields is not None and gallery_op_indices is not None
            and len(gallery_fields) >= 3):
        fmean = gallery_fields[2]
        fstd  = gallery_fields[2]  # wrong — gallery has mean/std separately
    # Always re-run for clarity (only 1 scan point, fast)
    x_mean = np.load(out_dir / ".x_mean.npy")
    x_std  = np.load(out_dir / ".x_std.npy")
    params = make_params_row(dp_trans, args.npuff_fixed, fixed)
    vX, vM = {}, {}
    for v in range(3):
        vX[v], vM[v] = build_x_img_batch(
            params, view_masks[v], view_geoms[v], c_in, x_mean, x_std)

    n_samp = args.n_samples
    wf_mean = np.zeros((104, 50), dtype=np.float64)
    wf_M2   = np.zeros((104, 50), dtype=np.float64)
    for s in range(n_samp):
        print(f"     Transition sample {s+1}/{n_samp}", end="\r", flush=True)
        te_s = run_one_sample(model, vX, vM, inv_maps, layout,
                              c_out, j_Te, y_indices, pos_channels,
                              signed_channels, y_mean, y_std, eps, s_c,
                              mode, "stochastic", args.ode_steps,
                              args.batch_size, device)[0]
        delta = te_s - wf_mean
        wf_mean += delta / (s + 1)
        wf_M2   += delta * (te_s - wf_mean)
    print()

    fmean = wf_mean.astype(np.float32)
    fstd  = np.sqrt(wf_M2 / max(n_samp - 1, 1)).astype(np.float32)
    fcv   = np.where(fmean > 0, fstd / (fmean + 1e-10), 0.0)

    np.save(out_dir / "spatial_mean_transition.npy", fmean)
    np.save(out_dir / "spatial_std_transition.npy",  fstd)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pos_mean = fmean.copy(); pos_mean[pos_mean <= 0] = np.nan
    pos_std  = fstd.copy();  pos_std[pos_std   <= 0] = np.nan

    # Panel 1: mean Te
    im0 = axes[0].imshow(pos_mean.T, origin="lower", aspect="auto",
                         norm=LogNorm(vmin=max(np.nanpercentile(pos_mean, 1), 1e-2),
                                      vmax=np.nanpercentile(pos_mean, 99)),
                         cmap="inferno")
    plt.colorbar(im0, ax=axes[0], label="Mean Te  [eV]")
    axes[0].set_title(f"Mean Te field  —  D_puff={dp_trans:.1e}", fontsize=10)

    # Panel 2: std(Te)
    im1 = axes[1].imshow(pos_std.T, origin="lower", aspect="auto",
                         norm=LogNorm(vmin=max(np.nanpercentile(pos_std, 1), 1e-4),
                                      vmax=np.nanpercentile(pos_std, 99)),
                         cmap="hot_r")
    plt.colorbar(im1, ax=axes[1], label="std(Te)  [eV]")
    axes[1].set_title(f"Pixel-wise std(Te)  —  {n_samp} samples", fontsize=10)

    # Panel 3: CV = std/mean
    cv_clip = np.clip(fcv, 0, 1.0)
    im2 = axes[2].imshow(cv_clip.T, origin="lower", aspect="auto",
                         vmin=0, vmax=min(float(cv_clip.max()), 1.0),
                         cmap="hot_r")
    plt.colorbar(im2, ax=axes[2], label="CV = std / mean")
    axes[2].set_title("Coefficient of variation (relative uncertainty)", fontsize=10)

    for ax in axes:
        ax.axhline(iy_sep, color="white", lw=1.2, ls="--", alpha=0.9,
                   label=f"sep (iy={iy_sep})")
        ax.axvline(ix_ot,  color="cyan",  lw=1.2, ls=":",  alpha=0.9,
                   label=f"OT (ix={ix_ot})")
        ax.axvline(ix_omp, color="lime",  lw=1.2, ls=":",  alpha=0.9,
                   label=f"OMP (ix={ix_omp})")
        ax.set_xlabel("Poloidal index", fontsize=9)
        ax.set_ylabel("Radial index",   fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Spatial uncertainty at the detachment transition  "
        f"(D_puff={dp_trans:.1e}, N_puff={args.npuff_fixed:.0e})\n"
        "Where on the 2D grid is the FM model most uncertain?",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_G_spatial_uncertainty_transition.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_G_spatial_uncertainty_transition.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE H  —  N_puff slice: std vs N_puff at fixed D_puff (band investigation)
# ─────────────────────────────────────────────────────────────────────────────

def fig_npuff_slice_uncertainty(args, model, view_masks, view_geoms,
                                inv_maps, layout, c_in, c_out, j_Te,
                                y_indices, pos_channels, signed_channels,
                                y_mean, y_std, eps, s_c, mode,
                                ix_ot, ix_omp, iy_sep,
                                fixed, device, out_dir):
    """
    Fix D_puff at the transition value and sweep N_puff from 10^18 to 10^23.
    Plot mean and std of Te,ot vs N_puff. This investigates the horizontal
    band of high uncertainty visible in the 2D uncertainty map.
    """
    print("\n  [H] N_puff slice uncertainty ...")

    te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
    dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
    delta         = np.abs(np.diff(np.log10(np.maximum(te_ot_mean_1d, 1e-10))))
    trans_idx     = int(np.argmax(delta))
    dp_fixed      = dpuff_vals[trans_idx]

    n_npuff_pts  = 30
    npuff_sweep  = np.logspace(18, 23, n_npuff_pts)
    n_samp       = args.n_samples
    x_mean = np.load(out_dir / ".x_mean.npy")
    x_std  = np.load(out_dir / ".x_std.npy")

    # Batch all N_puff points together for efficiency
    params = np.empty((n_npuff_pts, 8), dtype=np.float32)
    params[:, 0] = fixed["R"];   params[:, 1] = fixed["B"]
    params[:, 2] = fixed["Pin"]; params[:, 3] = np.log10(dp_fixed)
    params[:, 4] = np.log10(npuff_sweep)
    params[:, 5] = np.log10(fixed["Dcore"])
    params[:, 6] = fixed["D_perp"]; params[:, 7] = fixed["Chi_perp"]

    vX, vM = {}, {}
    for v in range(3):
        vX[v], vM[v] = build_x_img_batch(
            params, view_masks[v], view_geoms[v], c_in, x_mean, x_std)

    te_ot_samp_h  = np.zeros((n_samp, n_npuff_pts), dtype=np.float32)
    te_omp_samp_h = np.zeros((n_samp, n_npuff_pts), dtype=np.float32)

    for s in range(n_samp):
        print(f"     N_puff slice sample {s+1}/{n_samp}", end="\r", flush=True)
        te_s = run_one_sample(model, vX, vM, inv_maps, layout,
                              c_out, j_Te, y_indices, pos_channels,
                              signed_channels, y_mean, y_std, eps, s_c,
                              mode, "stochastic", args.ode_steps,
                              args.batch_size, device)   # (n_npuff_pts, 104, 50)
        te_ot_samp_h[s]  = te_s[:, ix_ot,  iy_sep]
        te_omp_samp_h[s] = te_s[:, ix_omp, iy_sep]
    print()

    te_ot_mean_h  = te_ot_samp_h.mean(axis=0)
    te_ot_std_h   = te_ot_samp_h.std(axis=0)
    te_omp_mean_h = te_omp_samp_h.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean Te,ot and Te,omp vs N_puff
    ax = axes[0]
    p5  = np.percentile(te_ot_samp_h, 5,  axis=0)
    p95 = np.percentile(te_ot_samp_h, 95, axis=0)
    ax.plot(npuff_sweep, te_ot_mean_h,  color="C0", lw=2.5, label="Te,ot  mean")
    ax.plot(npuff_sweep, te_omp_mean_h, color="C1", lw=2.5, label="Te,omp  mean")
    ax.fill_between(npuff_sweep, np.maximum(p5, 1e-2), np.maximum(p95, 1e-2),
                    color="C0", alpha=0.25, label="Te,ot  5th–95th pct")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N$_{puff}$  [atoms/s]", fontsize=11)
    ax.set_ylabel("Te  [eV]", fontsize=11)
    ax.set_title(f"Te vs N_puff  (D_puff = {dp_fixed:.1e} fixed)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=5e-1)

    # Right: std and CV vs N_puff
    ax  = axes[1]
    ax2 = ax.twinx()
    ax.plot(npuff_sweep, te_ot_std_h, color="C0", lw=2.5, label="std(Te,ot)")
    cv_h = te_ot_std_h / (te_ot_mean_h + 1e-10)
    ax2.plot(npuff_sweep, cv_h, color="gray", lw=2, ls="--", label="CV = std/mean")
    ax.set_xscale("log")
    ax.set_xlabel("N$_{puff}$  [atoms/s]", fontsize=11)
    ax.set_ylabel("std(Te,ot)  [eV]", fontsize=11)
    ax2.set_ylabel("CV = std / mean", fontsize=10)
    ax.set_title(f"Uncertainty vs N_puff  (D_puff = {dp_fixed:.1e} fixed)\n"
                 "Investigates the horizontal band in the 2D uncertainty map",
                 fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"N_puff sweep at D_puff = {dp_fixed:.1e}  (transition point)  |  "
        f"n_samples = {n_samp}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_H_npuff_slice_uncertainty.png", dpi=150)
    plt.close(fig)
    print("     Saved: fig_H_npuff_slice_uncertainty.png")


# ─────────────────────────────────────────────────────────────────────────────
# Physical mesh rendering helpers  (adapted from viz_mesh_pred.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mesh_polys(crx: np.ndarray, cry: np.ndarray,
                      mask: np.ndarray):
    """Return (verts, ix_arr, iy_arr) for all active cells.

    crx/cry : (104, 50, 4)   corner R/Z coordinates
    mask    : (104, 50) bool  True = cell is active
    """
    verts, vix, viy = [], [], []
    for ix in range(104):
        for iy in range(50):
            if not mask[ix, iy]:
                continue
            corners = np.column_stack([crx[ix, iy], cry[ix, iy]])  # (4,2)
            verts.append(corners)
            vix.append(ix)
            viy.append(iy)
    return verts, np.array(vix, dtype=int), np.array(viy, dtype=int)


def _render_on_mesh(ax, verts, values, cmap, norm, title,
                    xlabel="R [m]", ylabel="Z [m]"):
    """Draw a colour-coded quad mesh on ax using PolyCollection."""
    pc = PolyCollection(verts, array=values, cmap=cmap,
                        norm=norm, edgecolors="face", linewidths=0.0)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return pc


def _mark_geometry_on_mesh(ax, crx, cry, ix_ot, ix_omp, iy_sep):
    """Overlay thin lines marking the outer target, OMP and separatrix."""
    # Separatrix: cells at iy = iy_sep across all poloidal positions
    sep_R = crx[:, iy_sep, 0]
    sep_Z = cry[:, iy_sep, 0]
    ax.plot(sep_R, sep_Z, color="white", lw=0.8, ls="--", alpha=0.8,
            label=f"sep (iy={iy_sep})")
    # Outer target column (ix = ix_ot)
    ot_R = crx[ix_ot, :, 0]
    ot_Z = cry[ix_ot, :, 0]
    ax.plot(ot_R, ot_Z, color="cyan", lw=0.8, ls=":", alpha=0.8,
            label=f"OT (ix={ix_ot})")
    # OMP column (ix = ix_omp)
    omp_R = crx[ix_omp, :, 0]
    omp_Z = cry[ix_omp, :, 0]
    ax.plot(omp_R, omp_Z, color="lime", lw=0.8, ls=":", alpha=0.8,
            label=f"OMP (ix={ix_omp})")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE I-a  —  Gallery of mean Te on physical geometry (F equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def fig_physical_mesh_gallery(args, model, view_masks, view_geoms, inv_maps,
                               layout, c_in, c_out, j_Te, y_indices,
                               pos_channels, signed_channels,
                               y_mean, y_std, eps, s_c, mode,
                               ix_ot, ix_omp, iy_sep, fixed, device, out_dir):
    """
    Same 4 operating points as Figure F, but rendered on the true R-Z
    tokamak geometry using crx/cry corner coordinates.

    Layout: 2 rows × 4 columns
      Row 0: mean Te on real geometry
      Row 1: std(Te) on real geometry
    """
    print("\n  [I-a] Physical mesh gallery ...")

    if not args.geom_dir:
        print("     SKIP: --geom_dir not provided")
        return

    geom_dir = Path(args.geom_dir)
    crx = np.load(geom_dir / "crx.npy")   # (104, 50, 4)
    cry = np.load(geom_dir / "cry.npy")

    # Reuse saved spatial fields from Figure F if they exist, otherwise re-infer
    gallery_cache = out_dir / "_gallery_fields.npz"
    if gallery_cache.exists():
        print("     Loading cached gallery fields ...")
        cache = np.load(gallery_cache)
        fields_mean  = [cache[f"mean_{i}"] for i in range(4)]
        fields_std   = [cache[f"std_{i}"]  for i in range(4)]
        op_dpuff     = cache["op_dpuff"]
        op_labels    = ["Sheath-limited", "Pre-transition",
                        "At transition",  "Detached"]
    else:
        # Re-infer (same logic as fig_spatial_field_gallery)
        te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
        dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
        delta         = np.abs(np.diff(np.log10(
                            np.maximum(te_ot_mean_1d, 1e-10))))
        trans_idx     = int(np.argmax(delta))
        ND            = len(dpuff_vals)
        op_indices    = [0, max(0, trans_idx-4), trans_idx,
                         min(ND-1, trans_idx+3)]
        op_labels     = ["Sheath-limited", "Pre-transition",
                         "At transition",  "Detached"]
        op_dpuff      = dpuff_vals[op_indices]
        x_mean = np.load(out_dir / ".x_mean.npy")
        x_std  = np.load(out_dir / ".x_std.npy")
        n_samp = min(args.n_samples, 20)
        fields_mean, fields_std = [], []
        for dp, label in zip(op_dpuff, op_labels):
            params = make_params_row(dp, args.npuff_fixed, fixed)
            vX, vM = {}, {}
            for v in range(3):
                vX[v], vM[v] = build_x_img_batch(
                    params, view_masks[v], view_geoms[v], c_in, x_mean, x_std)
            wf_mean = np.zeros((104, 50), dtype=np.float64)
            wf_M2   = np.zeros((104, 50), dtype=np.float64)
            for s in range(n_samp):
                print(f"     {label}  {s+1}/{n_samp}", end="\r", flush=True)
                te_s = run_one_sample(
                    model, vX, vM, inv_maps, layout, c_out, j_Te, y_indices,
                    pos_channels, signed_channels, y_mean, y_std, eps, s_c,
                    mode, "stochastic", args.ode_steps, args.batch_size,
                    device)[0]
                delta_w = te_s - wf_mean
                wf_mean += delta_w / (s + 1)
                wf_M2   += delta_w * (te_s - wf_mean)
            print()
            fields_mean.append(wf_mean.astype(np.float32))
            fields_std.append(
                np.sqrt(wf_M2 / max(n_samp-1, 1)).astype(np.float32))
        # Cache for Figure I-b to reuse
        np.savez_compressed(gallery_cache,
                            **{f"mean_{i}": fields_mean[i] for i in range(4)},
                            **{f"std_{i}":  fields_std[i]  for i in range(4)},
                            op_dpuff=op_dpuff)

    # Build polygon vertices once (shared across all panels)
    mask_active = np.abs(fields_mean[0]) > 0   # (104,50) bool
    verts, vix, viy = _build_mesh_polys(crx, cry, mask_active)

    # Global colour limits
    all_mean_vals = np.concatenate([f[vix, viy] for f in fields_mean])
    all_std_vals  = np.concatenate([f[vix, viy] for f in fields_std])
    pos_m = all_mean_vals[all_mean_vals > 0]
    pos_s = all_std_vals[all_std_vals > 0]
    mean_norm = LogNorm(
        vmin=max(float(np.percentile(pos_m, 1)) if len(pos_m) else 1e-2, 1e-2),
        vmax=float(np.percentile(pos_m, 99)) if len(pos_m) else 1e4)
    std_norm  = LogNorm(
        vmin=max(float(np.percentile(pos_s, 1)) if len(pos_s) else 1e-4, 1e-4),
        vmax=float(np.percentile(pos_s, 99)) if len(pos_s) else 1e2)

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.30,
                            left=0.06, right=0.96, top=0.88, bottom=0.08)

    for col, (label, fmean, fstd, dp) in enumerate(
            zip(op_labels, fields_mean, fields_std, op_dpuff)):

        mean_vals = fmean[vix, viy]
        std_vals  = fstd[vix, viy]
        mean_vals = np.where(mean_vals > 0, mean_vals, np.nan)

        # Row 0: mean Te on real geometry
        ax0 = fig.add_subplot(gs[0, col])
        pc0 = _render_on_mesh(ax0, verts, mean_vals, "inferno", mean_norm,
                              f"{label}\nD={dp:.1e}")
        _mark_geometry_on_mesh(ax0, crx, cry, ix_ot, ix_omp, iy_sep)
        if col == 3:
            plt.colorbar(pc0, ax=ax0, label="Mean Te [eV]", fraction=0.04)

        # Row 1: std(Te) on real geometry
        ax1 = fig.add_subplot(gs[1, col])
        std_plot = np.where(std_vals > 0, std_vals, np.nan)
        pc1 = _render_on_mesh(ax1, verts, std_plot, "hot_r", std_norm,
                              "")
        _mark_geometry_on_mesh(ax1, crx, cry, ix_ot, ix_omp, iy_sep)
        if col == 3:
            plt.colorbar(pc1, ax=ax1, label="std(Te) [eV]", fraction=0.04)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color="white", ls="--", label=f"sep iy={iy_sep}"),
        Line2D([0],[0], color="cyan",  ls=":",  label=f"OT ix={ix_ot}"),
        Line2D([0],[0], color="lime",  ls=":",  label=f"OMP ix={ix_omp}"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.8)
    fig.suptitle(
        f"Physical geometry: mean Te (top) and std(Te) (bottom) at 4 operating points\n"
        f"N_puff = {args.npuff_fixed:.0e} atoms/s  |  "
        f"inferno = Te scale  |  hot_r = uncertainty scale",
        fontsize=11)
    fig.savefig(out_dir / "fig_Ia_physical_mesh_gallery.png", dpi=180)
    plt.close(fig)
    print("     Saved: fig_Ia_physical_mesh_gallery.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE I-b  —  Transition point: mean / std / CV on physical geometry (G equiv)
# ─────────────────────────────────────────────────────────────────────────────

def fig_physical_mesh_transition(args, model, view_masks, view_geoms, inv_maps,
                                  layout, c_in, c_out, j_Te, y_indices,
                                  pos_channels, signed_channels,
                                  y_mean, y_std, eps, s_c, mode,
                                  ix_ot, ix_omp, iy_sep, fixed, device,
                                  out_dir):
    """
    Render the spatial uncertainty at the detachment transition point
    (mean Te, std(Te), CV=std/mean) on the true R-Z tokamak geometry.
    This is the physical-geometry equivalent of Figure G.
    """
    print("\n  [I-b] Physical mesh transition uncertainty ...")

    if not args.geom_dir:
        print("     SKIP: --geom_dir not provided")
        return

    geom_dir = Path(args.geom_dir)
    crx = np.load(geom_dir / "crx.npy")
    cry = np.load(geom_dir / "cry.npy")

    # Load or compute the transition spatial fields
    mean_path = out_dir / "spatial_mean_transition.npy"
    std_path  = out_dir / "spatial_std_transition.npy"

    if mean_path.exists() and std_path.exists():
        print("     Loading cached transition fields ...")
        fmean = np.load(mean_path)
        fstd  = np.load(std_path)
        te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
        dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
        delta         = np.abs(np.diff(np.log10(
                            np.maximum(te_ot_mean_1d, 1e-10))))
        dp_trans      = float(dpuff_vals[int(np.argmax(delta))])
    else:
        # Re-infer
        te_ot_mean_1d = np.load(out_dir / "te_ot_mean_1d.npy")
        dpuff_vals    = np.load(out_dir / "dpuff_vals.npy")
        delta         = np.abs(np.diff(np.log10(
                            np.maximum(te_ot_mean_1d, 1e-10))))
        trans_idx     = int(np.argmax(delta))
        dp_trans      = float(dpuff_vals[trans_idx])
        x_mean = np.load(out_dir / ".x_mean.npy")
        x_std  = np.load(out_dir / ".x_std.npy")
        params = make_params_row(dp_trans, args.npuff_fixed, fixed)
        vX, vM = {}, {}
        for v in range(3):
            vX[v], vM[v] = build_x_img_batch(
                params, view_masks[v], view_geoms[v], c_in, x_mean, x_std)
        wf_mean = np.zeros((104, 50), dtype=np.float64)
        wf_M2   = np.zeros((104, 50), dtype=np.float64)
        n_samp  = args.n_samples
        for s in range(n_samp):
            print(f"     Transition sample {s+1}/{n_samp}", end="\r", flush=True)
            te_s = run_one_sample(
                model, vX, vM, inv_maps, layout, c_out, j_Te, y_indices,
                pos_channels, signed_channels, y_mean, y_std, eps, s_c,
                mode, "stochastic", args.ode_steps, args.batch_size,
                device)[0]
            dw = te_s - wf_mean
            wf_mean += dw / (s + 1)
            wf_M2   += dw * (te_s - wf_mean)
        print()
        fmean = wf_mean.astype(np.float32)
        fstd  = np.sqrt(wf_M2 / max(n_samp-1, 1)).astype(np.float32)
        np.save(mean_path, fmean)
        np.save(std_path,  fstd)

    fcv = np.where(fmean > 0, fstd / (fmean + 1e-10), 0.0).astype(np.float32)

    # Build polygons
    mask_active = fmean > 0
    verts, vix, viy = _build_mesh_polys(crx, cry, mask_active)

    mean_vals = fmean[vix, viy]
    std_vals  = fstd[vix, viy]
    cv_vals   = np.clip(fcv[vix, viy], 0, 1.5)

    pos_m  = mean_vals[mean_vals > 0]
    pos_s  = std_vals[std_vals   > 0]
    mean_norm = LogNorm(
        vmin=max(float(np.percentile(pos_m, 1)) if len(pos_m) else 1e-2, 1e-2),
        vmax=float(np.percentile(pos_m, 99)) if len(pos_m) else 1e4)
    std_norm  = LogNorm(
        vmin=max(float(np.percentile(pos_s, 1)) if len(pos_s) else 1e-4, 1e-4),
        vmax=float(np.percentile(pos_s, 99)) if len(pos_s) else 1e3)
    cv_norm   = Normalize(vmin=0, vmax=min(float(cv_vals.max()), 1.5))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    pc0 = _render_on_mesh(axes[0], verts, mean_vals, "inferno", mean_norm,
                          f"Mean Te  —  D_puff={dp_trans:.1e}")
    plt.colorbar(pc0, ax=axes[0], label="Mean Te [eV]", fraction=0.04)
    _mark_geometry_on_mesh(axes[0], crx, cry, ix_ot, ix_omp, iy_sep)
    axes[0].legend(fontsize=7, loc="upper left")

    pc1 = _render_on_mesh(axes[1], verts,
                          np.where(std_vals > 0, std_vals, np.nan),
                          "hot_r", std_norm,
                          f"Pixel-wise std(Te)  —  {args.n_samples} samples")
    plt.colorbar(pc1, ax=axes[1], label="std(Te) [eV]", fraction=0.04)
    _mark_geometry_on_mesh(axes[1], crx, cry, ix_ot, ix_omp, iy_sep)

    pc2 = _render_on_mesh(axes[2], verts, cv_vals, "hot_r", cv_norm,
                          "CV = std / mean  (relative uncertainty)")
    plt.colorbar(pc2, ax=axes[2], label="CV = std / mean", fraction=0.04)
    _mark_geometry_on_mesh(axes[2], crx, cry, ix_ot, ix_omp, iy_sep)

    fig.suptitle(
        f"Physical geometry: FM uncertainty at the detachment transition\n"
        f"D_puff = {dp_trans:.1e}  |  N_puff = {args.npuff_fixed:.0e}  |  "
        f"white dashed = separatrix  |  cyan = outer target  |  lime = OMP",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_Ib_physical_mesh_transition.png", dpi=180)
    plt.close(fig)
    print("     Saved: fig_Ib_physical_mesh_transition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    print(f"\n  Output dir : {out_dir}")
    print(f"  Device     : {device}")

    # ── Verify required cached files exist ────────────────────────────────────
    required = ["te_ot_mean_2d.npy", "te_ot_std_2d.npy",
                "te_ot_mean_1d.npy", "te_omp_mean_1d.npy",
                "te_ot_std_1d.npy",  "te_ot_samples_1d.npy",
                "dpuff_vals.npy",    "npuff_vals.npy"]
    missing = [f for f in required if not (out_dir / f).exists()]
    if missing:
        print(f"\n  ERROR: The following files are missing from --out_dir:\n"
              f"  {missing}\n"
              f"  Run scan_gas_puff.py first to generate them.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n  Loading model ...")
    (model, c_in, c_out, y_indices, j_Te,
     pos_channels, signed_channels,
     eps, x_mean, x_std, y_mean, y_std, s_c, mode) = load_model_and_stats(
         args.checkpoint, device)

    if j_Te is None:
        print("  ERROR: Te (channel 0) not in y_indices — cannot proceed.")
        sys.exit(1)

    # Save normalisation stats so figure functions can access them
    np.save(out_dir / ".x_mean.npy", x_mean)
    np.save(out_dir / ".x_std.npy",  x_std)

    if args.mode:
        mode = args.mode
    print(f"  c_in={c_in}  c_out={c_out}  mode={mode}  j_Te={j_Te}")

    # ── Load geometry ──────────────────────────────────────────────────────────
    geo = Geometry(args.b2fgmtry)
    ix_ot  = geo.nx
    ix_omp = geo.jxa
    iy_sep = geo.jsep + 2
    print(f"  Geometry: ix_ot={ix_ot}  ix_omp={ix_omp}  iy_sep={iy_sep}")

    # ── Load view masks ────────────────────────────────────────────────────────
    view_masks, view_geoms = load_view_masks(
        args.tensor_prefix, args.tensor_split, c_in)

    # ── Build reconstruction maps ─────────────────────────────────────────────
    layout   = dict(np.load(args.layout_path, allow_pickle=True))
    gap_px   = int(layout["gap_px"])
    inv_maps, layout = load_inv_maps(
        args.layout_path, args.unroller, gap_px, ix_ot, ix_omp, iy_sep)

    # ── Fixed ITER parameters ─────────────────────────────────────────────────
    fixed = dict(R=args.R, B=args.B, Pin=args.Pin,
                 Dcore=args.Dcore, D_perp=args.D_perp, Chi_perp=args.Chi_perp)

    # Shared kwargs for model-inference figures
    infer_kw = dict(
        model=model, view_masks=view_masks, view_geoms=view_geoms,
        inv_maps=inv_maps, layout=layout,
        c_in=c_in, c_out=c_out, j_Te=j_Te,
        y_indices=y_indices, pos_channels=pos_channels,
        signed_channels=signed_channels,
        y_mean=y_mean, y_std=y_std, eps=eps, s_c=s_c, mode=mode,
        ix_ot=ix_ot, ix_omp=ix_omp, iy_sep=iy_sep,
        fixed=fixed, device=device, out_dir=out_dir,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Run each figure section, catching errors independently so one failure
    # does not prevent the others from running.
    # ─────────────────────────────────────────────────────────────────────────

    sections = [
        ("A — stochasticity check",
         lambda: fig_stochasticity_check(args, **infer_kw)),
        ("B — 1D percentile + spaghetti",
         lambda: fig_1d_percentile(args, out_dir)),
        ("C — sample strip plot",
         lambda: fig_sample_strip(args, out_dir)),
        ("D — 2D heatmap + test scatter",
         lambda: fig_2d_with_test_scatter(args, out_dir, ix_ot, ix_omp, iy_sep)),
        ("E — 1D scan + test scatter",
         lambda: fig_1d_with_test_scatter(args, out_dir, ix_ot, ix_omp, iy_sep)),
        ("F — spatial field gallery",
         lambda: fig_spatial_field_gallery(args, **infer_kw)),
        ("G — spatial uncertainty at transition",
         lambda: fig_spatial_uncertainty_transition(args, **infer_kw)),
        ("H — N_puff slice uncertainty",
         lambda: fig_npuff_slice_uncertainty(args, **infer_kw)),
        ("I-a — physical mesh gallery",
         lambda: fig_physical_mesh_gallery(args, **infer_kw)),
        ("I-b — physical mesh transition",
         lambda: fig_physical_mesh_transition(args, **infer_kw)),
    ]

    results = {}
    for name, fn in sections:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  !! SECTION {name} FAILED: {e}")
            traceback.print_exc()
            results[name] = None

    print(f"\n{'='*60}")
    print(f"  All done.  Figures written to: {out_dir}")
    for f in sorted(out_dir.glob("fig_[A-Z]*.png")):
        print(f"    {f.name}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FM uncertainty analysis — improvements to scan_gas_puff.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Path args ──────────────────────────────────────────────────────────────
    ap.add_argument("--scripts_dir",    required=True)
    ap.add_argument("--out_dir",        required=True,
                    help="Directory where scan_gas_puff.py wrote its .npy files")
    ap.add_argument("--checkpoint",     required=True)
    ap.add_argument("--tensor_prefix",  required=True)
    ap.add_argument("--layout_path",    required=True)
    ap.add_argument("--unroller",       required=True)
    ap.add_argument("--b2fgmtry",       required=True)
    ap.add_argument("--test_data_dir",  default=None,
                    help="Path to train/ or test/ folder with X_tmp.npy and "
                         "te_tmp.npy.  Required for figures D and E.")
    ap.add_argument("--geom_dir",       default=None,
                    help="Path to geometry/ folder containing crx.npy and cry.npy. "
                         "Required for figures I-a and I-b.")
    # ── ITER fixed parameters ──────────────────────────────────────────────────
    ap.add_argument("--R",        type=float, default=6.2)
    ap.add_argument("--B",        type=float, default=5.3)
    ap.add_argument("--Pin",      type=float, default=100e6,
                    help="Pin in Watts (100 MW = 100e6)")
    ap.add_argument("--Dcore",    type=float, default=9.1e21)
    ap.add_argument("--D_perp",   type=float, default=0.3)
    ap.add_argument("--Chi_perp", type=float, default=1.0)
    ap.add_argument("--npuff_fixed", type=float, default=1e20)
    # ── Inference ─────────────────────────────────────────────────────────────
    ap.add_argument("--n_samples",    type=int,   default=30)
    ap.add_argument("--ode_steps",    type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=64)
    ap.add_argument("--mode",         default=None,
                    choices=["cfm", "xpred", "direct"])
    ap.add_argument("--tensor_split", default="train")
    ap.add_argument("--device",       default=None)
    ap.add_argument("--seed",         type=int, default=42)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
