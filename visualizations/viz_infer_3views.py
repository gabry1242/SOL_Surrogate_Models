#!/usr/bin/env python3
"""
viz_infer_3views.py

Visualize inference results from infer_unet_3views.py.

For each selected sample and each output channel, produces one figure with:
  - 3 rows : view0 (top) | view1 (mid) | view2 (bottom)
  - 3 cols : Ground Truth | Prediction | |Absolute Error|
  Plus per-view MAE and RMSE in subplot titles.

Tensor layout (as produced by build_3view_tensors.py + infer_unet_3views.py):
  Each view has its own 4D file:
    <tensor_prefix>_view{0,1,2}_X_img_<split>.npy  (N, C_in,  H, W)
    <tensor_prefix>_view{0,1,2}_Y_img_<split>.npy  (N, C_out, H, W)
  Predictions (one per view run):
    <pred_dir_view0>/pred_Y_img_test.npy            (N, C_sel, H, W)
    <pred_dir_view1>/pred_Y_img_test.npy
    <pred_dir_view2>/pred_Y_img_test.npy
  Optional layout metadata (for canvas crop):
    <tensor_prefix>_layout_map_3views.npz

Usage examples
--------------
# Visualize sample 0, all predicted channels, all 3 views:
python scripts/visualizations/viz_infer_3views.py \
    --tensor_prefix  scripts/tensor/3views_4d/test/global3v \
    --tensor_split   test \
    --pred_view0     scripts/runs/unet_3views/view0/infer_test \
    --pred_view1     scripts/runs/unet_3views/view1/infer_test \
    --pred_view2     scripts/runs/unet_3views/view2/infer_test \
    --out_dir        scripts/runs/unet_3views/viz \
    --idx            0

# te and ti only (original channels 0 and 1), samples 0,1,2:
python scripts/visualizations/viz_infer_3views.py \
    --tensor_prefix  scripts/tensor/3views_4d/test/global3v \
    --tensor_split   test \
    --pred_view0     scripts/runs/unet_3views/view0/infer_test \
    --pred_view1     scripts/runs/unet_3views/view1/infer_test \
    --pred_view2     scripts/runs/unet_3views/view2/infer_test \
    --out_dir        scripts/runs/unet_3views/viz \
    --idx            0,1,2 \
    --y_indices      0,1

# Only view0 and view1 (omit --pred_view2 to skip view2):
python scripts/visualizations/viz_infer_3views.py \
    --tensor_prefix  scripts/tensor/3views_4d/test/global3v \
    --tensor_split   test \
    --pred_view0     scripts/runs/unet_3views/view0/infer_test \
    --pred_view1     scripts/runs/unet_3views/view1/infer_test \
    --out_dir        scripts/runs/unet_3views/viz \
    --idx            0

Output layout (under --out_dir)
--------------------------------
  idx00000/
    ch00_te.png
    ch01_ti.png
    ...
  idx00001/
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Channel metadata  (matches train_unet_3views.py ordering)
# ---------------------------------------------------------------------------
_SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]

CHANNEL_NAMES = (
    ["te", "ti"]
    + [f"na_{s}" for s in _SPECIES]   # 2..11
    + [f"ua_{s}" for s in _SPECIES]   # 12..21
)
CHANNEL_UNITS = (
    ["J", "J"]
    + ["m⁻³"] * 10
    + ["m/s"]  * 10
)

POS_CHANNELS    = set(range(0, 12))   # te, ti, na  → positive physical quantities
SIGNED_CHANNELS = set(range(12, 22))  # ua           → can be negative


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


def _load_layout(tensor_prefix: str) -> Optional[Dict]:
    """Load the shared layout .npz (crop sizes per view)."""
    p = Path(f"{tensor_prefix}_layout_map_3views.npz")
    if not p.exists():
        return None
    z   = np.load(p, allow_pickle=True)
    out = {}
    for k in ["W0", "H0", "W1", "H1", "W2", "H2", "Hmax", "Wmax"]:
        if k in z.files:
            out[k] = int(z[k])
    return out


def _view_crop(layout: Optional[Dict], view_id: int, H: int, W: int) -> Tuple[int, int]:
    """Return (Hcrop, Wcrop) for the valid (non-padded) region of a view canvas."""
    if layout is None:
        return H, W
    Hc = min(layout.get(f"H{view_id}", H), H)
    Wc = min(layout.get(f"W{view_id}", W), W)
    return Hc, Wc


def _apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set gap/padding pixels to NaN so imshow renders them as blank."""
    out = img.astype(np.float64)
    out[mask < 0.5] = np.nan
    return out


def _load_y_indices_from_checkpoint(pred_dir: Path) -> Optional[List[int]]:
    """Try to read y_indices stored inside the companion checkpoint."""
    for ckpt_name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        ckpt_path = pred_dir.parent / ckpt_name
        if ckpt_path.exists():
            try:
                import torch
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ids  = [int(x) for x in ckpt.get("y_indices", [])]
                if ids:
                    print(f"  y_indices read from {ckpt_path}: {ids}")
                    return ids
            except Exception as e:
                print(f"  Warning: could not read checkpoint {ckpt_path}: {e}")
    return None


def _shared_norm(
    orig_c:   int,
    gt_imgs:  List[np.ndarray],   # list of (Hc,Wc) arrays (may contain NaN)
    pr_imgs:  List[np.ndarray],
) -> mcolors.Normalize:
    """Shared colormap normalisation across all views for one channel."""
    all_vals = np.concatenate([
        v[~np.isnan(v)].ravel()
        for v in gt_imgs + pr_imgs
        if v is not None and v.size > 0
    ])
    if all_vals.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)

    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    if orig_c in SIGNED_CHANNELS:
        lim = max(abs(vmin), abs(vmax), 1e-12)
        return mcolors.Normalize(vmin=-lim, vmax=lim)

    return mcolors.Normalize(vmin=max(vmin, 0.0), vmax=max(vmax, 1e-12))


def _error_norm(err_imgs: List[np.ndarray]) -> mcolors.Normalize:
    """Shared error colormap normalisation across all views."""
    all_vals = np.concatenate([
        v[~np.isnan(v)].ravel()
        for v in err_imgs
        if v is not None and v.size > 0
    ])
    if all_vals.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    return mcolors.Normalize(vmin=0, vmax=float(np.nanpercentile(all_vals, 98)))


def _metric_str(
    gt_crop: np.ndarray, pr_crop: np.ndarray, mask_crop: np.ndarray
) -> str:
    valid = mask_crop > 0.5
    if valid.sum() == 0:
        return "no valid pixels"
    diff  = gt_crop[valid] - pr_crop[valid]
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    return f"MAE={mae:.3g}  RMSE={rmse:.3g}"


# ---------------------------------------------------------------------------
# Single figure: one channel, one sample, N active views (rows) × 3 cols
# ---------------------------------------------------------------------------

VIEW_LABELS = ["View 0  (top)", "View 1  (mid)", "View 2  (bottom)"]


def plot_channel(
    active_views: List[int],          # e.g. [0, 1, 2]
    gt_data:      Dict[int, np.ndarray],   # view_id → (H,W) ground truth
    pr_data:      Dict[int, np.ndarray],   # view_id → (H,W) prediction
    mask_data:    Dict[int, np.ndarray],   # view_id → (H,W) mask
    orig_c:       int,
    ch_name:      str,
    ch_unit:      str,
    idx:          int,
    out_path:     Path,
) -> None:
    n_rows = len(active_views)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, 4.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Sample {idx}  |  Channel: {ch_name}  [{ch_unit}]",
        fontsize=13, fontweight="bold",
    )

    # pre-collect masked images for shared normalisation
    gt_masked  = [_apply_mask(gt_data[v],  mask_data[v]) for v in active_views]
    pr_masked  = [_apply_mask(pr_data[v],  mask_data[v]) for v in active_views]
    err_masked = [_apply_mask(np.abs(gt_data[v] - pr_data[v]), mask_data[v])
                  for v in active_views]

    cmap      = "inferno" if orig_c in POS_CHANNELS else "RdBu_r"
    norm      = _shared_norm(orig_c, gt_masked, pr_masked)
    err_norm_ = _error_norm(err_masked)

    for row, v in enumerate(active_views):
        gt_img   = gt_masked[row]
        pr_img   = pr_masked[row]
        err_img  = err_masked[row]
        metrics  = _metric_str(gt_data[v], pr_data[v], mask_data[v])
        vlabel   = VIEW_LABELS[v] if v < len(VIEW_LABELS) else f"View {v}"

        # col 0 — ground truth
        ax = axes[row, 0]
        im = ax.imshow(gt_img, cmap=cmap, norm=norm, origin="upper",
                       interpolation="nearest")
        ax.set_title(f"{vlabel}\nGround Truth", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # col 1 — prediction
        ax = axes[row, 1]
        im = ax.imshow(pr_img, cmap=cmap, norm=norm, origin="upper",
                       interpolation="nearest")
        ax.set_title(f"{vlabel}\nPrediction\n{metrics}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # col 2 — absolute error
        ax = axes[row, 2]
        im = ax.imshow(err_img, cmap="hot_r", norm=err_norm_, origin="upper",
                       interpolation="nearest")
        ax.set_title(f"{vlabel}\n|Error|", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize per-view UNet inference results (4D tensors)."
    )

    # ---- tensor paths ----
    ap.add_argument(
        "--tensor_prefix", required=True,
        help=(
            "Prefix used when building the test tensors with build_3view_tensors.py. "
            "E.g. scripts/tensor/3views_4d/test/global3v"
        ),
    )
    ap.add_argument("--tensor_split", default="test",
                    help="Split tag for the ground-truth tensors (default: test).")

    # ---- prediction directories (one per view, all optional) ----
    ap.add_argument("--pred_view0", default=None,
                    help="Folder containing view0 pred_Y_img_test.npy "
                         "(output of infer_unet_3views.py for view 0).")
    ap.add_argument("--pred_view1", default=None,
                    help="Folder containing view1 pred_Y_img_test.npy.")
    ap.add_argument("--pred_view2", default=None,
                    help="Folder containing view2 pred_Y_img_test.npy.")

    # ---- output ----
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write PNG figures.")

    # ---- selection ----
    ap.add_argument("--idx", default="0",
                    help='Sample indices: "0", "0,1,5", or "all".')
    ap.add_argument(
        "--y_indices", default="all",
        help=(
            'Original channel indices to plot, e.g. "0,1" for te+ti, or "all". '
            'If a view was trained on a subset, only channels present in that '
            'view\'s checkpoint are plotted for that row.'
        ),
    )
    ap.add_argument("--no_mask", action="store_true",
                    help="Skip masking (show full padded canvas).")
    args = ap.parse_args()

    # ---- figure out which views are active ----
    pred_dirs: Dict[int, Path] = {}
    for v, arg_val in [(0, args.pred_view0),
                       (1, args.pred_view1),
                       (2, args.pred_view2)]:
        if arg_val is not None:
            pred_dirs[v] = Path(arg_val)

    if not pred_dirs:
        raise ValueError("Provide at least one of --pred_view0 / --pred_view1 / --pred_view2.")

    layout = _load_layout(args.tensor_prefix)
    if layout:
        print(f"Layout loaded: {layout}")
    else:
        print("No layout file found — full canvas will be shown without cropping.")

    # ---- load per-view arrays (mmap) ----
    # Structure:
    #   gt_arrs[v]   : (N, C_out, H, W)  ground truth
    #   x_arrs[v]    : (N, C_in,  H, W)  for mask (channel 0)
    #   pred_arrs[v] : (N, C_sel, H, W)  model predictions
    #   y_idx_map[v] : List[int]          orig channel index for each pred channel

    gt_arrs:   Dict[int, np.ndarray] = {}
    x_arrs:    Dict[int, np.ndarray] = {}
    pred_arrs: Dict[int, np.ndarray] = {}
    y_idx_map: Dict[int, List[int]]  = {}

    for v, pred_dir in pred_dirs.items():
        view_tag   = f"view{v}"
        gt_path    = Path(f"{args.tensor_prefix}_{view_tag}_Y_img_{args.tensor_split}.npy")
        x_path     = Path(f"{args.tensor_prefix}_{view_tag}_X_img_{args.tensor_split}.npy")
        pred_path  = pred_dir / "pred_Y_img_test.npy"

        for p in (gt_path, x_path, pred_path):
            if not p.exists():
                raise FileNotFoundError(str(p))

        print(f"Loading view {v} …")
        gt_arrs[v]   = np.load(gt_path,   mmap_mode="r")
        x_arrs[v]    = np.load(x_path,    mmap_mode="r")
        pred_arrs[v] = np.load(pred_path, mmap_mode="r")

        if pred_arrs[v].ndim != 4:
            raise ValueError(
                f"pred_Y_img_test.npy for view {v} must be 4D (N,C,H,W), "
                f"got {pred_arrs[v].shape}. "
                "Did you use infer_unet_3views.py?"
            )

        C_sel = pred_arrs[v].shape[1]
        # try to read y_indices from companion checkpoint
        ids = _load_y_indices_from_checkpoint(pred_dir)
        y_idx_map[v] = ids if ids is not None else list(range(C_sel))

        print(f"  gt:   {gt_arrs[v].shape}")
        print(f"  pred: {pred_arrs[v].shape}  →  y_indices: {y_idx_map[v]}")

    # ---- N consistency check ----
    Ns = [arr.shape[0] for arr in pred_arrs.values()]
    if len(set(Ns)) > 1:
        raise ValueError(f"Prediction arrays have inconsistent N: { {v: Ns[i] for i,v in enumerate(pred_dirs)} }")
    N = Ns[0]

    # ---- sample indices ----
    sample_indices = _parse_idx_list(args.idx, N)
    bad = [i for i in sample_indices if i < 0 or i >= N]
    if bad:
        raise ValueError(f"--idx values {bad} out of range [0,{N-1}]")

    # ---- determine which original channels to plot ----
    # Union of all original channels across active views, then filter by --y_indices.
    all_orig_channels: List[int] = sorted(set(
        c for ids in y_idx_map.values() for c in ids
    ))
    user_filter = _parse_ch_list(args.y_indices, max(all_orig_channels) + 1
                                  if all_orig_channels else 1)
    # user_filter interpreted as original channel indices
    channels_to_plot = [c for c in all_orig_channels if c in set(user_filter)]

    if not channels_to_plot:
        raise ValueError(
            f"No channels to plot after applying --y_indices={args.y_indices}. "
            f"Available original channels across views: {all_orig_channels}"
        )

    out_dir    = Path(args.out_dir)
    active_views = sorted(pred_dirs.keys())

    print(f"\nSamples         : {sample_indices}")
    print(f"Channels to plot: {channels_to_plot}")
    print(f"Active views    : {active_views}")
    print(f"Output dir      : {out_dir}\n")

    total_figs = len(sample_indices) * len(channels_to_plot)
    done = 0

    for idx in sample_indices:
        sample_out = out_dir / f"idx{idx:05d}"

        for orig_c in channels_to_plot:
            ch_name = CHANNEL_NAMES[orig_c] if orig_c < len(CHANNEL_NAMES) else f"ch{orig_c}"
            ch_unit = CHANNEL_UNITS[orig_c] if orig_c < len(CHANNEL_UNITS) else "?"

            # collect per-view data for this (idx, orig_c)
            gt_data:   Dict[int, np.ndarray] = {}
            pr_data:   Dict[int, np.ndarray] = {}
            mask_data: Dict[int, np.ndarray] = {}
            views_with_channel: List[int]    = []

            for v in active_views:
                if orig_c not in y_idx_map[v]:
                    # this view's model wasn't trained on this channel — skip row
                    continue

                pred_ch_idx = y_idx_map[v].index(orig_c)

                H, W  = gt_arrs[v].shape[2], gt_arrs[v].shape[3]
                Hc, Wc = _view_crop(layout, v, H, W)

                gt_crop   = np.array(gt_arrs[v]  [idx, orig_c,     :Hc, :Wc], dtype=np.float32)
                pr_crop   = np.array(pred_arrs[v] [idx, pred_ch_idx,:Hc, :Wc], dtype=np.float32)
                mask_crop = np.array(x_arrs[v]    [idx, 0,          :Hc, :Wc], dtype=np.float32)

                if args.no_mask:
                    mask_crop = np.ones_like(mask_crop)

                gt_data[v]   = gt_crop
                pr_data[v]   = pr_crop
                mask_data[v] = mask_crop
                views_with_channel.append(v)

            if not views_with_channel:
                print(f"  [skip] idx={idx} ch={orig_c} ({ch_name}): no view has this channel")
                continue

            out_path = sample_out / f"ch{orig_c:02d}_{ch_name}.png"
            print(f"[{done+1:3d}/{total_figs}] idx={idx}  ch={orig_c} ({ch_name})"
                  f"  views={views_with_channel}")

            plot_channel(
                active_views=views_with_channel,
                gt_data=gt_data,
                pr_data=pr_data,
                mask_data=mask_data,
                orig_c=orig_c,
                ch_name=ch_name,
                ch_unit=ch_unit,
                idx=idx,
                out_path=out_path,
            )
            done += 1

    print(f"\nDone. {done} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
