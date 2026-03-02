#!/usr/bin/env python3
"""
viz_tensor_3views.py

Viewer for tensors produced by tensor_3_images.py / tensor_3_images_fixed.py.

Loads:
  <prefix>_X_img_<split>.npy
  <prefix>_Y_img_<split>.npy
Optional:
  <mask_file>  (typically <prefix>_mask_views_k<k>.npy OR <prefix>_mask_views_fromX_<split>.npy)

Supports:
- pack="channels": X (N, 3*C_in, H, W), Y (N, 3*C_out, H, W)
- pack="views":    X (N, 3, C_in, H, W), Y (N, 3, C_out, H, W)

Plots per view:
- mask
- selected X channels
- selected Y channels
- optional error maps if --pred_y is provided
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


VIEW_NAMES = ["top", "mid", "bottom"]


def _parse_int_list(s: str):
    s = s.strip().lower()
    if s == "all":
        return "all"
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def _detect_pack(arr: np.ndarray) -> str:
    if arr.ndim == 4:
        return "channels"
    if arr.ndim == 5:
        return "views"
    raise ValueError(f"Unsupported tensor rank: {arr.ndim}, shape={arr.shape}")


def _slice_view(arr: np.ndarray, pack: str, view_idx: int, c_per_view: int) -> np.ndarray:
    """
    Returns (N, C, H, W) for the requested view.
    """
    if pack == "views":
        return arr[:, view_idx]
    c0 = view_idx * c_per_view
    c1 = (view_idx + 1) * c_per_view
    return arr[:, c0:c1]


def _imshow(ax, img2d, title, vmin=None, vmax=None):
    ax.imshow(img2d, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tensor_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--view", choices=VIEW_NAMES + ["all"], default="all")

    ap.add_argument("--x_channels", default="0,9,10", help='Per-view X channels to show, e.g. "0,9,10" or "all"')
    ap.add_argument("--y_channels", default="0,1", help='Per-view Y channels to show, e.g. "0,1" or "all"')

    ap.add_argument("--mask_file", default="", help="Optional mask file inside tensor_dir")
    ap.add_argument("--pred_y", default="", help="Optional pred_Y file path (same shape as Y) to plot |pred-gt|")
    ap.add_argument("--save", default="", help="If set, save figure to this path instead of showing.")
    args = ap.parse_args()

    tensor_dir = Path(args.tensor_dir)
    x_path = tensor_dir / f"{args.prefix}_X_img_{args.split}.npy"
    y_path = tensor_dir / f"{args.prefix}_Y_img_{args.split}.npy"

    if not x_path.exists():
        raise FileNotFoundError(str(x_path))
    if not y_path.exists():
        raise FileNotFoundError(str(y_path))

    X = np.load(x_path)
    Y = np.load(y_path)

    pack_x = _detect_pack(X)
    pack_y = _detect_pack(Y)
    if pack_x != pack_y:
        raise ValueError(f"X/Y pack mismatch: X is {pack_x}, Y is {pack_y}")
    pack = pack_x

    N = int(X.shape[0])
    if not (0 <= args.idx < N):
        raise IndexError(f"--idx {args.idx} out of range [0, {N-1}]")

    if pack == "views":
        if X.shape[1] != 3 or Y.shape[1] != 3:
            raise ValueError(f"Expected 3 views, got X.shape={X.shape}, Y.shape={Y.shape}")
        c_in = int(X.shape[2])
        c_out = int(Y.shape[2])
        H = int(X.shape[3])
        W = int(X.shape[4])
    else:
        if X.shape[1] % 3 != 0 or Y.shape[1] % 3 != 0:
            raise ValueError(f"Expected channels divisible by 3. X.shape={X.shape}, Y.shape={Y.shape}")
        c_in = int(X.shape[1] // 3)
        c_out = int(Y.shape[1] // 3)
        H = int(X.shape[2])
        W = int(X.shape[3])

    # mask loading
    masks = None
    if args.mask_file:
        mpath = tensor_dir / args.mask_file
        if not mpath.exists():
            raise FileNotFoundError(str(mpath))
        masks = np.load(mpath)
        # allowed shapes:
        # (3,H,W) OR (H,W) (then used for all views)
        if masks.ndim == 3:
            if masks.shape[0] != 3:
                raise ValueError(f"mask_file must have shape (3,H,W). Got {masks.shape}")
            if masks.shape[1] != H or masks.shape[2] != W:
                raise ValueError(f"mask_file spatial mismatch. mask={masks.shape}, expected (3,{H},{W})")
        elif masks.ndim == 2:
            if masks.shape != (H, W):
                raise ValueError(f"mask_file spatial mismatch. mask={masks.shape}, expected ({H},{W})")
        else:
            raise ValueError(f"mask_file must be 2D or 3D. Got {masks.ndim}D")

    # optional predictions
    P = None
    if args.pred_y:
        ppath = Path(args.pred_y)
        if not ppath.exists():
            raise FileNotFoundError(str(ppath))
        P = np.load(ppath)
        if P.shape != Y.shape:
            raise ValueError(f"--pred_y shape {P.shape} must match Y shape {Y.shape}")

    x_sel = _parse_int_list(args.x_channels)
    y_sel = _parse_int_list(args.y_channels)

    view_indices = [0, 1, 2] if args.view == "all" else [VIEW_NAMES.index(args.view)]

    plots = []  # (title, img2d, is_mask)
    for vi in view_indices:
        Xv = _slice_view(X, pack, vi, c_in)[args.idx]  # (C,H,W)
        Yv = _slice_view(Y, pack, vi, c_out)[args.idx]  # (C,H,W)

        if masks is None:
            mv = (Xv[0] > 0).astype(np.float32)  # derive from X mask channel
        else:
            if masks.ndim == 3:
                mv = masks[vi].astype(np.float32)
            else:
                mv = masks.astype(np.float32)

        plots.append((f"{VIEW_NAMES[vi]}: mask", mv, True))

        # X channels
        if x_sel == "all":
            x_channels = list(range(c_in))
        else:
            x_channels = [c for c in x_sel if 0 <= c < c_in]
        for ch in x_channels:
            plots.append((f"{VIEW_NAMES[vi]}: X[{ch}]", Xv[ch] * (mv > 0), False))

        # Y channels
        if y_sel == "all":
            y_channels = list(range(c_out))
        else:
            y_channels = [c for c in y_sel if 0 <= c < c_out]
        for ch in y_channels:
            plots.append((f"{VIEW_NAMES[vi]}: Y[{ch}]", Yv[ch] * (mv > 0), False))

        # error channels
        if P is not None:
            Pv = _slice_view(P, pack, vi, c_out)[args.idx]
            for ch in y_channels:
                err = np.abs(Pv[ch] - Yv[ch]) * (mv > 0)
                plots.append((f"{VIEW_NAMES[vi]}: |err| Y[{ch}]", err, False))

    n = len(plots)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.2 * ncols, 3.6 * nrows))

    for i, (title, img, is_mask) in enumerate(plots, 1):
        ax = fig.add_subplot(nrows, ncols, i)
        if is_mask:
            _imshow(ax, img, title, vmin=0, vmax=1)
        else:
            _imshow(ax, img, title)

    fig.suptitle(
        f"{args.prefix} {args.split} idx={args.idx} pack={pack}  X={X.shape} Y={Y.shape}",
        fontsize=12,
    )
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()