from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def masked_metrics(yhat: np.ndarray, y: np.ndarray, m: np.ndarray, eps: float = 1e-8):
    """
    yhat, y: (H,W) float
    m: (H,W) in {0,1}
    """
    m = m.astype(np.float32)
    denom = m.sum() + eps
    diff = (yhat - y)
    mae = (np.abs(diff) * m).sum() / denom
    rmse = np.sqrt(((diff ** 2) * m).sum() / denom)
    return float(mae), float(rmse)


def masked_imshow(ax, img: np.ndarray, mask: np.ndarray, title: str):
    """
    Render img but hide gaps (mask==0) by setting them to NaN.
    """
    show = img.astype(np.float32).copy()
    show[mask == 0] = np.nan
    im = ax.imshow(show, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="Folder with global_X_img_test.npy and global_Y_img_test.npy")
    ap.add_argument("--pred_path", required=True, help="Path to pred_Y_img_test.npy")
    ap.add_argument("--out_dir", required=True, help="Where to save PNGs")
    ap.add_argument("--index", type=int, default=0, help="Test sample index to visualize")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(test_dir / "global_X_img_test.npy", mmap_mode="r")
    Y = np.load(test_dir / "global_Y_img_test.npy", mmap_mode="r")
    P = np.load(args.pred_path, mmap_mode="r")                    

    i = args.index
    if i < 0 or i >= X.shape[0]:
        raise ValueError(f"index out of range: {i} (N={X.shape[0]})")

    mask = X[i, 0]  # (H,W) 0/1
    # channels 0,1 = te, ti in your training setup
    y_te = Y[i, 0]
    y_ti = Y[i, 1]
    p_te = P[i, 0]
    p_ti = P[i, 1]

    e_te = np.abs(p_te - y_te)
    e_ti = np.abs(p_ti - y_ti)

    te_mae, te_rmse = masked_metrics(p_te, y_te, mask)
    ti_mae, ti_rmse = masked_metrics(p_ti, y_ti, mask)

    # ---- te figure ----
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    masked_imshow(ax1, y_te, mask, f"te true (idx={i})")
    masked_imshow(ax2, p_te, mask, f"te pred (MAE={te_mae:.6g}, RMSE={te_rmse:.6g})")
    masked_imshow(ax3, e_te, mask, "te abs error")

    masked_imshow(ax4, y_ti, mask, f"ti true (idx={i})")
    masked_imshow(ax5, p_ti, mask, f"ti pred (MAE={ti_mae:.6g}, RMSE={ti_rmse:.6g})")
    masked_imshow(ax6, e_ti, mask, "ti abs error")

    fig.tight_layout()
    out_path = out_dir / f"pred_vs_true_idx_{i}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- numeric summary ----
    txt = (
        f"idx={i}\n"
        f"te: MAE={te_mae:.8g} RMSE={te_rmse:.8g}\n"
        f"ti: MAE={ti_mae:.8g} RMSE={ti_rmse:.8g}\n"
        f"valid_pixels={int(mask.sum())} / total_pixels={mask.size}\n"
    )
    (out_dir / f"metrics_idx_{i}.txt").write_text(txt, encoding="utf-8")

    print(f"Saved: {out_path}")
    print(txt)


if __name__ == "__main__":
    main()

#python scripts/visualizations/viz_predictions.py --test_dir scripts/tensor/test 
# --pred_path scripts/runs/unet_3gap/infer_test/pred_Y_img_test.npy --out_dir scripts/runs/unet_3gap/viz --index 0