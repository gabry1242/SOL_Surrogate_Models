#!/usr/bin/env python3
"""
viz_infer_3views.py

Visualize inference results from infer_unet_3inputs.py.

For each selected sample and each output channel, produces one figure with:
  - 3 rows  : view0 (top), view1 (mid), view2 (bottom)
  - 3 cols  : Ground Truth | Prediction | Absolute Error

Both pred and ground truth are in physical units (already inverse-transformed
by infer_unet_3inputs.py).

Loads:
  <pred_dir>/pred_Y_img_test.npy       (N, 3, C_sel, H, W)  — predictions
  <tensor_dir>/<prefix>_Y_img_test.npy (N, 3, C_out, H, W)  — ground truth
  <tensor_dir>/<prefix>_X_img_test.npy (N, 3, C_in,  H, W)  — for mask
  <tensor_dir>/<prefix>_layout_map_3views.npz                — for crop sizes (optional)

Outputs PNGs to <out_dir>/<idx>/<channel_name>.png

Usage:
  python scripts/visualizations/viz_infer_3views.py \
    --pred_dir   scripts/runs/unet3_width64/infer_test \
    --tensor_dir scripts/tensor/3images/test \
    --out_dir    scripts/runs/unet3_width64/infer_test/viz \
    --prefix     global3 \
    --idx        0 \
    --y_indices  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21

  # visualize multiple samples:
  --idx 0,1,2,5
  # visualize all samples in test set:
  --idx all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------
# Default channel names matching train_unet_3inputs.py ordering:
#   0: te, 1: ti, 2-11: na (10 species), 12-21: ua (10 species)
_SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]

DEFAULT_CHANNEL_NAMES = (
    ["te", "ti"]
    + [f"na_{s}" for s in _SPECIES]
    + [f"ua_{s}" for s in _SPECIES]
)

DEFAULT_CHANNEL_UNITS = (
    ["J", "J"]                    # te, ti in Joules (physical units from infer)
    + ["m⁻³"] * 10                # na
    + ["m/s"]  * 10                # ua
)

# Positive channels (te, ti, na) benefit from log-scale display
# Signed channels (ua) use linear/symlog
POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_idx_list(s: str, N: int) -> List[int]:
    s = s.strip().lower()
    if s == "all":
        return list(range(N))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ch_list(s: str, C: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", ""):
        return list(range(C))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_layout(tensor_dir: Path, prefix: str) -> Optional[dict]:
    p = tensor_dir / f"{prefix}_layout_map_3views.npz"
    if not p.exists():
        return None
    z   = np.load(p, allow_pickle=True)
    out = {}
    for k in ["W0", "H0", "W1", "H1", "W2", "H2", "Hmax", "Wmax"]:
        if k in z.files:
            out[k] = int(z[k])
    return out


def _view_crop(layout: Optional[dict], view_id: int, H: int, W: int) -> Tuple[int, int]:
    if layout is None:
        return H, W
    hk = f"H{view_id}"
    wk = f"W{view_id}"
    Hc = min(layout[hk], H) if hk in layout else H
    Wc = min(layout[wk], W) if wk in layout else W
    return Hc, Wc


def _get_mask(X: np.ndarray, idx: int) -> np.ndarray:
    """Returns (3, H, W) float32 mask from X channel 0 per view."""
    return X[idx, :, 0, :, :].astype(np.float32)


def _apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """img: (H,W), mask: (H,W) — set gap pixels to NaN for display."""
    out = img.copy().astype(np.float64)
    out[mask < 0.5] = np.nan
    return out


def _colormap_for_channel(orig_c: int) -> str:
    if orig_c in POS_CHANNELS:
        return "inferno"
    return "RdBu_r"


def _norm_for_channel(
    orig_c: int,
    gt_img: np.ndarray,
    pr_img: np.ndarray,
) -> mcolors.Normalize:
    """
    Compute a shared colormap normalization from the valid (non-NaN) pixels
    of both ground truth and prediction.
    For signed channels uses symmetric limits around zero.
    """
    valid = np.concatenate([
        gt_img[~np.isnan(gt_img)].ravel(),
        pr_img[~np.isnan(pr_img)].ravel(),
    ])
    if valid.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)

    vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))

    if orig_c in SIGNED_CHANNELS:
        # symmetric around zero for velocity fields
        lim = max(abs(vmin), abs(vmax))
        return mcolors.Normalize(vmin=-lim, vmax=lim)

    # positive channels: guard against log of zero
    vmin = max(vmin, 0.0)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _error_norm(err_img: np.ndarray) -> mcolors.Normalize:
    valid = err_img[~np.isnan(err_img)].ravel()
    if valid.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    return mcolors.Normalize(vmin=0, vmax=float(np.nanpercentile(valid, 98)))


# ---------------------------------------------------------------------------
# Single figure: one channel, one sample, all 3 views
# ---------------------------------------------------------------------------
VIEW_LABELS = ["View 0 (top)", "View 1 (mid)", "View 2 (bottom)"]

def plot_channel(
    gt:        np.ndarray,   # (3, H, W)  ground truth, physical units
    pred:      np.ndarray,   # (3, H, W)  prediction,   physical units
    mask3:     np.ndarray,   # (3, H, W)  binary mask
    layout:    Optional[dict],
    orig_c:    int,          # original channel index (for colormap choice)
    ch_name:   str,
    ch_unit:   str,
    idx:       int,
    out_path:  Path,
) -> None:
    """
    Produces a 3×3 figure:
      rows = views (0,1,2)
      cols = Ground Truth | Prediction | |Error|
    """
    n_views = 3
    fig, axes = plt.subplots(
        n_views, 3,
        figsize=(15, 4 * n_views),
        squeeze=False,
    )
    fig.suptitle(
        f"Sample {idx}  |  Channel: {ch_name}  [{ch_unit}]",
        fontsize=14, fontweight="bold", y=1.01,
    )

    H, W = gt.shape[1], gt.shape[2]

    for v in range(n_views):
        Hc, Wc = _view_crop(layout, v, H, W)
        m = mask3[v, :Hc, :Wc]

        gt_img   = _apply_mask(gt  [v, :Hc, :Wc], m)
        pred_img = _apply_mask(pred[v, :Hc, :Wc], m)
        err_img  = _apply_mask(np.abs(gt[v, :Hc, :Wc] - pred[v, :Hc, :Wc]), m)

        cmap     = _colormap_for_channel(orig_c)
        norm     = _norm_for_channel(orig_c, gt_img, pred_img)
        err_norm = _error_norm(err_img)

        # compute masked MAE for this view/channel
        valid_mask = m > 0.5
        if valid_mask.sum() > 0:
            mae_val  = float(np.nanmean(np.abs(
                gt[v, :Hc, :Wc][valid_mask] - pred[v, :Hc, :Wc][valid_mask]
            )))
            rmse_val = float(np.sqrt(np.nanmean((
                gt[v, :Hc, :Wc][valid_mask] - pred[v, :Hc, :Wc][valid_mask]
            ) ** 2)))
            metric_str = f"MAE={mae_val:.3g}  RMSE={rmse_val:.3g}"
        else:
            metric_str = "no valid pixels"

        # --- col 0: ground truth ---
        ax = axes[v, 0]
        im = ax.imshow(gt_img, cmap=cmap, norm=norm, origin="upper")
        ax.set_title(f"{VIEW_LABELS[v]}\nGround Truth", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- col 1: prediction ---
        ax = axes[v, 1]
        im = ax.imshow(pred_img, cmap=cmap, norm=norm, origin="upper")
        ax.set_title(f"{VIEW_LABELS[v]}\nPrediction\n{metric_str}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- col 2: absolute error ---
        ax = axes[v, 2]
        im = ax.imshow(err_img, cmap="hot_r", norm=err_norm, origin="upper")
        ax.set_title(f"{VIEW_LABELS[v]}\n|Error|", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir",   required=True,
                    help="Folder containing pred_Y_img_test.npy "
                         "(output of infer_unet_3inputs.py)")
    ap.add_argument("--tensor_dir", required=True,
                    help="Folder containing ground-truth tensors "
                         "(global3_Y_img_test.npy, global3_X_img_test.npy)")
    ap.add_argument("--out_dir",    required=True,
                    help="Where to write PNG figures")
    ap.add_argument("--prefix",     default="global3",
                    help="Tensor file prefix (default: global3)")
    ap.add_argument("--idx",        default="0",
                    help='Sample indices to visualize: "0", "0,1,5", or "all"')
    ap.add_argument("--y_indices",  default="all",
                    help='Which output channels to plot (original channel indices). '
                         '"all" plots all channels present in pred tensor. '
                         'e.g. "0,1" for te and ti only.')
    ap.add_argument("--no_mask",    action="store_true",
                    help="Do not apply mask (show full canvas including gaps)")
    args = ap.parse_args()

    pred_dir   = Path(args.pred_dir)
    tensor_dir = Path(args.tensor_dir)
    out_dir    = Path(args.out_dir)
    pfx        = args.prefix

    # ---- load arrays ----
    pred_path = pred_dir / "pred_Y_img_test.npy"
    gt_path   = tensor_dir / f"{pfx}_Y_img_test.npy"
    x_path    = tensor_dir / f"{pfx}_X_img_test.npy"

    for p in [pred_path, gt_path, x_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print("Loading tensors (mmap) …")
    pred_arr = np.load(pred_path, mmap_mode="r")  # (N, 3, C_sel, H, W)
    gt_arr   = np.load(gt_path,   mmap_mode="r")  # (N, 3, C_out, H, W)
    x_arr    = np.load(x_path,    mmap_mode="r")  # (N, 3, C_in,  H, W)

    if pred_arr.ndim != 5:
        raise ValueError(f"pred tensor must be 5D (N,3,C,H,W), got {pred_arr.shape}")
    if gt_arr.ndim != 5:
        raise ValueError(f"gt tensor must be 5D (N,3,C,H,W), got {gt_arr.shape}")

    N, _, C_sel, H, W = pred_arr.shape
    C_out_total        = gt_arr.shape[2]

    layout = _load_layout(tensor_dir, pfx)

    # ---- resolve sample indices ----
    sample_indices = _parse_idx_list(args.idx, N)
    bad = [i for i in sample_indices if i < 0 or i >= N]
    if bad:
        raise ValueError(f"Sample indices out of range [0,{N-1}]: {bad}")

    # ---- resolve channel indices ----
    # y_indices here refers to which of the C_sel pred channels to plot,
    # AND maps back to original channel index for naming/colourmap.
    # If the user passes original channel indices, we use them directly.
    # If pred has C_sel < C_out_total, we need to know the mapping.
    # We read it from the checkpoint if available, otherwise assume 0..C_sel-1.
    ckpt_y_indices = None
    ckpt_path_guess = pred_dir.parent / "checkpoint_best.pt"
    if ckpt_path_guess.exists():
        try:
            import torch
            ckpt = torch.load(ckpt_path_guess, map_location="cpu", weights_only=False)
            ckpt_y_indices = [int(x) for x in ckpt.get("y_indices", [])]
            print(f"Loaded y_indices from checkpoint: {ckpt_y_indices}")
        except Exception as e:
            print(f"Could not load checkpoint for y_indices: {e}")

    if ckpt_y_indices is None:
        # assume pred channel j corresponds to original channel j
        ckpt_y_indices = list(range(C_sel))

    # user filter on top of that
    user_ch = _parse_ch_list(args.y_indices, len(ckpt_y_indices))
    # user_ch are indices INTO ckpt_y_indices (i.e. into the C_sel pred axis)
    # validate
    bad_ch = [c for c in user_ch if c < 0 or c >= len(ckpt_y_indices)]
    if bad_ch:
        raise ValueError(
            f"--y_indices values {bad_ch} out of range [0,{len(ckpt_y_indices)-1}]"
        )

    # build channel name/unit lookup
    ch_names = DEFAULT_CHANNEL_NAMES  # list of 22 names
    ch_units = DEFAULT_CHANNEL_UNITS  # list of 22 units

    print(f"\nSamples to visualize : {sample_indices}")
    print(f"Channels to visualize: {[ckpt_y_indices[j] for j in user_ch]}")
    print(f"Output directory     : {out_dir}\n")

    # ---- main loop ----
    total_figs = len(sample_indices) * len(user_ch)
    done = 0

    for idx in sample_indices:
        # load this sample's data into RAM (avoid repeated mmap seeks)
        pred_sample = np.array(pred_arr[idx], dtype=np.float32)  # (3, C_sel, H, W)
        gt_sample   = np.array(gt_arr  [idx], dtype=np.float32)  # (3, C_out, H, W)
        mask3       = _get_mask(x_arr, idx)                       # (3, H, W)

        if args.no_mask:
            mask3 = np.ones_like(mask3)

        sample_out = out_dir / f"idx{idx:05d}"

        for j in user_ch:
            orig_c   = ckpt_y_indices[j]
            ch_name  = ch_names[orig_c] if orig_c < len(ch_names) else f"ch{orig_c}"
            ch_unit  = ch_units[orig_c] if orig_c < len(ch_units) else "?"

            # pred channel j  ↔  gt channel orig_c
            pred_ch = pred_sample[:, j,      :, :]  # (3, H, W)
            gt_ch   = gt_sample  [:, orig_c, :, :]  # (3, H, W)

            out_path = sample_out / f"ch{orig_c:02d}_{ch_name}.png"

            plot_channel(
                gt=gt_ch,
                pred=pred_ch,
                mask3=mask3,
                layout=layout,
                orig_c=orig_c,
                ch_name=ch_name,
                ch_unit=ch_unit,
                idx=idx,
                out_path=out_path,
            )

            done += 1
            print(f"[{done:3d}/{total_figs}] idx={idx}  ch={orig_c} ({ch_name})  → {out_path}")

    print(f"\nDone. {done} figures saved to {out_dir}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Example commands:
#
# Visualize sample 0, all 22 channels:
#   python scripts/visualizations/viz_infer_3views.py \
#     --pred_dir   scripts/runs/unet3_width64/infer_test \
#     --tensor_dir scripts/tensor/3images/test \
#     --out_dir    scripts/runs/unet3_width64/infer_test/viz \
#     --prefix     global3 \
#     --idx        0
#
# Visualize samples 0,1,2 — te and ti only (channels 0,1):
#   python scripts/visualizations/viz_infer_3views.py \
#     --pred_dir   scripts/runs/unet3_width64/infer_test \
#     --tensor_dir scripts/tensor/3images/test \
#     --out_dir    scripts/runs/unet3_width64/infer_test/viz \
#     --prefix     global3 \
#     --idx        0,1,2 \
#     --y_indices  0,1
#
# Visualize all test samples (warning: generates N*22 figures):
#   python scripts/visualizations/viz_infer_3views.py \
#     --pred_dir   scripts/runs/unet3_width64/infer_test \
#     --tensor_dir scripts/tensor/3images/test \
#     --out_dir    scripts/runs/unet3_width64/infer_test/viz \
#     --prefix     global3 \
#     --idx        all \
#     --y_indices  0,1
# ---------------------------------------------------------------------------
