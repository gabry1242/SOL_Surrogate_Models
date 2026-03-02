#!/usr/bin/env python3
"""
viz_tensor_3views.py

Visualize 3-view tensors created by tensor_3_images.py.

Loads:
- <prefix>_X_img_<split>.npy        (N,3,Cin,H,W)
- <prefix>_Y_img_<split>.npy        (N,3,Cout,H,W)
Optional:
- <prefix>_mask_views_k{k}.npy      (N,3,H,W) or (3,H,W)
- <prefix>_layout_map_3views.npz    (for view sizes W0/H0, W1/H1, W2/H2)

Outputs PNGs into --out_dir and/or shows interactively.

Usage examples:
  python scripts/visualizations/viz_tensor_3views.py --split train --prefix scripts/tensor/3images/train/global3 --idx 0 --view all --x_channels 0,1,2 --y_channels 0,1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if s.lower() == "all":
        return [-1]
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _load_layout(prefix: str) -> Optional[dict]:
    p = Path(f"{prefix}_layout_map_3views.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    keys = set(z.files)
    out = {}
    for k in ["W0", "H0", "W1", "H1", "W2", "H2", "Hmax", "Wmax"]:
        if k in keys:
            out[k] = int(z[k])
    return out


def _resolve_view_crop(layout: Optional[dict], view_id: int, H: int, W: int) -> Tuple[int, int]:
    if layout is None:
        return H, W
    if view_id == 0 and "H0" in layout and "W0" in layout:
        return min(layout["H0"], H), min(layout["W0"], W)
    if view_id == 1 and "H1" in layout and "W1" in layout:
        return min(layout["H1"], H), min(layout["W1"], W)
    if view_id == 2 and "H2" in layout and "W2" in layout:
        return min(layout["H2"], H), min(layout["W2"], W)
    return H, W


def _get_mask(prefix: str, split: str, idx: int, X: np.ndarray) -> np.ndarray:
    """
    Prefer explicit mask file if present; otherwise use X[:,0] channel (mask channel).
    Returns (3,H,W) float32 in {0,1}.
    """
    # try N,3,H,W
    cand = list(Path(prefix).parent.glob(Path(prefix).name + "_mask_views_k*.npy"))
    if cand:
        m = np.load(cand[0])
        if m.ndim == 4:
            return m[idx].astype(np.float32)
        if m.ndim == 3:
            return m.astype(np.float32)
    return X[idx, :, 0, :, :].astype(np.float32)


def _imshow(ax, img, title: str, vmin=None, vmax=None):
    ax.imshow(img, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_or_show(fig, out_path: Optional[Path], show: bool):
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--prefix", required=True, help="Path prefix without suffix, e.g. scripts/tensor/3images/train/global3")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--view", default="all", choices=["0", "1", "2", "all"])
    ap.add_argument("--x_channels", default="0", help='Comma list or "all". X channels (within Cin). 0 is mask.')
    ap.add_argument("--y_channels", default="0,1", help='Comma list or "all". Y channels (within Cout).')
    ap.add_argument("--apply_mask", action="store_true", help="Multiply shown channels by mask.")
    ap.add_argument("--out_dir", default="scripts/visualizations/out_tensor_3views")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    split = args.split
    prefix = args.prefix
    idx = int(args.idx)

    X_path = Path(f"{prefix}_X_img_{split}.npy")
    Y_path = Path(f"{prefix}_Y_img_{split}.npy")
    if not X_path.exists():
        raise FileNotFoundError(str(X_path))
    if not Y_path.exists():
        raise FileNotFoundError(str(Y_path))

    X = np.load(X_path, mmap_mode="r")  # (N,3,Cin,H,W)
    Y = np.load(Y_path, mmap_mode="r")  # (N,3,Cout,H,W)

    N, V, Cin, H, W = X.shape
    _, _, Cout, Hy, Wy = Y.shape
    if (Hy, Wy) != (H, W):
        raise ValueError(f"X/Y spatial mismatch: X=({H},{W}) Y=({Hy},{Wy})")
    if idx < 0 or idx >= N:
        raise ValueError(f"--idx out of range: {idx} not in [0,{N-1}]")

    layout = _load_layout(prefix)
    mask3 = _get_mask(prefix, split, idx, X)  # (3,H,W)

    x_sel = _parse_int_list(args.x_channels)
    y_sel = _parse_int_list(args.y_channels)

    if x_sel == [-1]:
        x_sel = list(range(Cin))
    if y_sel == [-1]:
        y_sel = list(range(Cout))

    if args.view == "all":
        views = [0, 1, 2]
    else:
        views = [int(args.view)]

    out_dir = Path(args.out_dir)
    save = not args.no_save
    show = not args.no_show

    # 1) mask overview per view
    fig, axes = plt.subplots(1, len(views), figsize=(5 * len(views), 4))
    if len(views) == 1:
        axes = [axes]
    for j, v in enumerate(views):
        Hc, Wc = _resolve_view_crop(layout, v, H, W)
        _imshow(axes[j], mask3[v, :Hc, :Wc], f"mask view {v} (idx={idx})", vmin=0, vmax=1)
    out_path = out_dir / f"idx{idx:05d}_mask_views_{'_'.join(map(str,views))}.png" if save else None
    _save_or_show(fig, out_path, show)

    # 2) X channels
    for c in x_sel:
        if c < 0 or c >= Cin:
            continue
        fig, axes = plt.subplots(1, len(views), figsize=(5 * len(views), 4))
        if len(views) == 1:
            axes = [axes]
        for j, v in enumerate(views):
            Hc, Wc = _resolve_view_crop(layout, v, H, W)
            img = np.array(X[idx, v, c, :Hc, :Wc], dtype=np.float32)
            if args.apply_mask:
                img *= mask3[v, :Hc, :Wc]
            _imshow(axes[j], img, f"X c={c} view {v}")
        out_path = out_dir / f"idx{idx:05d}_X_c{c:02d}_views_{'_'.join(map(str,views))}.png" if save else None
        _save_or_show(fig, out_path, show)

    # 3) Y channels
    for c in y_sel:
        if c < 0 or c >= Cout:
            continue
        fig, axes = plt.subplots(1, len(views), figsize=(5 * len(views), 4))
        if len(views) == 1:
            axes = [axes]
        for j, v in enumerate(views):
            Hc, Wc = _resolve_view_crop(layout, v, H, W)
            img = np.array(Y[idx, v, c, :Hc, :Wc], dtype=np.float32)
            if args.apply_mask:
                img *= mask3[v, :Hc, :Wc]
            _imshow(axes[j], img, f"Y c={c} view {v}")
        out_path = out_dir / f"idx{idx:05d}_Y_c{c:02d}_views_{'_'.join(map(str,views))}.png" if save else None
        _save_or_show(fig, out_path, show)

    # 4) quick diff sanity if te/ti are first two channels (optional)
    if Cout >= 2:
        fig, axes = plt.subplots(2, len(views), figsize=(5 * len(views), 7))
        if len(views) == 1:
            axes = np.array(axes).reshape(2, 1)
        for j, v in enumerate(views):
            Hc, Wc = _resolve_view_crop(layout, v, H, W)
            te_img = np.array(Y[idx, v, 0, :Hc, :Wc], dtype=np.float32)
            ti_img = np.array(Y[idx, v, 1, :Hc, :Wc], dtype=np.float32)
            if args.apply_mask:
                te_img *= mask3[v, :Hc, :Wc]
                ti_img *= mask3[v, :Hc, :Wc]
            _imshow(axes[0, j], te_img, f"Y te (c=0) view {v}")
            _imshow(axes[1, j], ti_img, f"Y ti (c=1) view {v}")
        out_path = out_dir / f"idx{idx:05d}_Y_te_ti_views_{'_'.join(map(str,views))}.png" if save else None
        _save_or_show(fig, out_path, show)


if __name__ == "__main__":
    main()