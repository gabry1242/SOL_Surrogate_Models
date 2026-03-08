#!/usr/bin/env python3
"""
viz_3view_tensors.py

Visualize one view at a time from the 3 separate 4D tensors produced by
build_3view_tensors.py.

Loads:
  <prefix>_view{v}_X_img_<split>.npy   (N, C_in,  H, W)
  <prefix>_view{v}_Y_img_<split>.npy   (N, C_out, H, W)
Optional:
  <prefix>_layout_map_3views.npz       (for true local crop sizes)

Outputs PNGs into --out_dir and/or shows interactively.

Usage examples:
  python viz_3view_tensors.py --split train --prefix scripts/tensor/3views_4d/train/global3v --view 0 --idx 0 --x_channels 0,1 --y_channels 0,1
  python viz_3view_tensors.py --split train --prefix scripts/tensor/3views_4d/train/global3v --view 1 --idx 5 --y_channels all --apply_mask
  python viz_3view_tensors.py --split test  --prefix scripts/tensor/3views_4d/test/global3v  --view 2 --idx 0 --no_save
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if s.lower() == "all":
        return [-1]
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _load_layout(prefix: str) -> Optional[dict]:
    p = Path(f"{prefix}_layout_map_3views.npz")
    if not p.exists():
        return None
    z    = np.load(p, allow_pickle=True)
    keys = set(z.files)
    out  = {}
    for k in ["W0", "H0", "W1", "H1", "W2", "H2", "Hmax", "Wmax"]:
        if k in keys:
            out[k] = int(z[k])
    return out


def _resolve_crop(layout: Optional[dict], view_id: int, H: int, W: int) -> Tuple[int, int]:
    """Return the true local canvas size for this view (before zero-padding)."""
    if layout is None:
        return H, W
    Hk = f"H{view_id}"
    Wk = f"W{view_id}"
    Hc = min(layout[Hk], H) if Hk in layout else H
    Wc = min(layout[Wk], W) if Wk in layout else W
    return Hc, Wc


def _imshow(ax, img: np.ndarray, title: str, vmin=None, vmax=None):
    ax.imshow(img, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_or_show(fig, out_path: Optional[Path], show: bool):
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"  Saved: {out_path}")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split",      choices=["train", "test"], required=True)
    ap.add_argument("--prefix",     required=True,
                    help="Path prefix, e.g. scripts/tensor/3views_4d/train/global3v")
    ap.add_argument("--view",       type=int, choices=[0, 1, 2], required=True,
                    help="Which view to visualize: 0=top, 1=mid, 2=bot")
    ap.add_argument("--idx",        type=int, default=0,
                    help="Simulation index to visualize")
    ap.add_argument("--x_channels", default="0",
                    help='X channels to show. Comma list or "all". 0=mask.')
    ap.add_argument("--y_channels", default="0,1",
                    help='Y channels to show. Comma list or "all".')
    ap.add_argument("--apply_mask", action="store_true",
                    help="Multiply shown channels by mask before display.")
    ap.add_argument("--out_dir",    default="viz_3view_out")
    ap.add_argument("--no_save",    action="store_true")
    ap.add_argument("--no_show",    action="store_true")
    args = ap.parse_args()

    v      = args.view
    split  = args.split
    idx    = args.idx
    prefix = args.prefix

    X_path = Path(f"{prefix}_view{v}_X_img_{split}.npy")
    Y_path = Path(f"{prefix}_view{v}_Y_img_{split}.npy")

    if not X_path.exists():
        raise FileNotFoundError(f"X tensor not found: {X_path}")
    if not Y_path.exists():
        raise FileNotFoundError(f"Y tensor not found: {Y_path}")

    X = np.load(X_path, mmap_mode="r")   # (N, C_in,  H, W)
    Y = np.load(Y_path, mmap_mode="r")   # (N, C_out, H, W)

    N, C_in,  H,  W  = X.shape
    _, C_out, Hy, Wy = Y.shape

    if idx < 0 or idx >= N:
        raise ValueError(f"--idx {idx} out of range [0, {N-1}]")

    layout = _load_layout(prefix)
    Hc, Wc = _resolve_crop(layout, v, H, W)

    # mask from X channel 0
    mask = np.array(X[idx, 0, :Hc, :Wc], dtype=np.float32)   # (Hc, Wc)

    x_sel = _parse_int_list(args.x_channels)
    y_sel = _parse_int_list(args.y_channels)
    if x_sel == [-1]: x_sel = list(range(C_in))
    if y_sel == [-1]: y_sel = list(range(C_out))

    out_dir = Path(args.out_dir)
    save    = not args.no_save
    show    = not args.no_show
    tag     = f"view{v}_idx{idx:05d}"

    view_label = ["top (A|B)", "mid (C|D|E)", "bot (F)"][v]
    print(f"\nView {v} ({view_label})  |  idx={idx}  |  local canvas={Hc}×{Wc}  |  padded={H}×{W}")
    print(f"  C_in={C_in}  C_out={C_out}")

    # ---- 1. Mask --------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _imshow(ax, mask, f"mask  view{v}  idx={idx}", vmin=0, vmax=1)
    out_path = out_dir / f"{tag}_mask.png" if save else None
    _save_or_show(fig, out_path, show)

    # ---- 2. X channels --------------------------------------------------
    for c in x_sel:
        if c < 0 or c >= C_in:
            print(f"  Skipping X channel {c} (out of range)")
            continue
        img = np.array(X[idx, c, :Hc, :Wc], dtype=np.float32)
        if args.apply_mask:
            img *= mask
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        _imshow(ax, img, f"X  c={c}  view{v}  idx={idx}")
        out_path = out_dir / f"{tag}_X_c{c:02d}.png" if save else None
        _save_or_show(fig, out_path, show)

    # ---- 3. Y channels --------------------------------------------------
    for c in y_sel:
        if c < 0 or c >= C_out:
            print(f"  Skipping Y channel {c} (out of range)")
            continue
        img = np.array(Y[idx, c, :Hc, :Wc], dtype=np.float32)
        if args.apply_mask:
            img *= mask
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        _imshow(ax, img, f"Y  c={c}  view{v}  idx={idx}")
        out_path = out_dir / f"{tag}_Y_c{c:02d}.png" if save else None
        _save_or_show(fig, out_path, show)

    # ---- 4. te/ti quick overview (if Y has ≥ 2 channels) ----------------
    if C_out >= 2:
        te = np.array(Y[idx, 0, :Hc, :Wc], dtype=np.float32)
        ti = np.array(Y[idx, 1, :Hc, :Wc], dtype=np.float32)
        if args.apply_mask:
            te *= mask
            ti *= mask
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        _imshow(axes[0], te, f"Y te (c=0)  view{v}  idx={idx}")
        _imshow(axes[1], ti, f"Y ti (c=1)  view{v}  idx={idx}")
        out_path = out_dir / f"{tag}_Y_te_ti.png" if save else None
        _save_or_show(fig, out_path, show)


if __name__ == "__main__":
    main()
