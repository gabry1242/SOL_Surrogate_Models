# viz_predictions_all_channels.py
#
# Extension of viz_predictions.py to handle all 22 output channels:
# 0: te
# 1: ti
# 2..11: na[0..9]
# 12..21: ua[0..9]
#
# It saves:
# - one PNG per selected channel: true / pred / abs error
# - a per-channel metrics table (MAE, RMSE, valid_pixels)
#
# Run example:
# python scripts/visualizations/viz_predictions_all_channels.py --test_dir scripts/tensor/test --pred_path scripts/runs/unet_3gap/infer_test/pred_Y_img_test.npy 
# --out_dir scripts/runs/unet_3gap/viz_all --index 0 --group all
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def masked_metrics(yhat: np.ndarray, y: np.ndarray, m: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    """
    yhat, y: (H,W) float
    m: (H,W) in {0,1}
    """
    m = m.astype(np.float32)
    denom = float(m.sum()) + eps
    diff = (yhat - y)
    mae = float((np.abs(diff) * m).sum() / denom)
    rmse = float(np.sqrt(((diff ** 2) * m).sum() / denom))
    return mae, rmse


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


def channel_labels(c_out: int) -> List[str]:
    """
    Default labels for the common 22-channel layout.
    Falls back to ch_<idx> if c_out is not 22.
    """
    if c_out == 22:
        labs = ["te", "ti"]
        labs += [f"na[{k}]" for k in range(10)]
        labs += [f"ua[{k}]" for k in range(10)]
        return labs
    return [f"ch_{i}" for i in range(c_out)]


def select_channels(group: str, c_out: int) -> List[int]:
    """
    group:
      - te, ti
      - na (densities), ua (velocities)
      - all (all channels)
      - other20 (channels excluding te/ti; requires c_out>=22)
    """
    g = group.strip().lower()
    if g == "all":
        return list(range(c_out))
    if g == "te":
        return [0]
    if g == "ti":
        return [1]
    if g == "na":
        if c_out < 12:
            raise ValueError(f"Requested group 'na' but c_out={c_out} < 12.")
        return list(range(2, min(12, c_out)))
    if g == "ua":
        if c_out < 22:
            raise ValueError(f"Requested group 'ua' but c_out={c_out} < 22.")
        return list(range(12, min(22, c_out)))
    if g == "other20":
        if c_out < 22:
            raise ValueError(f"Requested group 'other20' but c_out={c_out} < 22.")
        return list(range(2, 22))
    raise ValueError(f"Unknown --group '{group}'. Use: te, ti, na, ua, all, other20")


def parse_channel_list(s: str, c_out: int) -> List[int]:
    """
    Comma-separated list of channel indices, e.g. "2,3,12".
    """
    if s is None or s.strip() == "":
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if p == "":
            continue
        i = int(p)
        if i < 0 or i >= c_out:
            raise ValueError(f"channel index out of range: {i} (c_out={c_out})")
        out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="Folder with global_X_img_test.npy and global_Y_img_test.npy")
    ap.add_argument("--pred_path", required=True, help="Path to pred_Y_img_test.npy")
    ap.add_argument("--out_dir", required=True, help="Where to save PNGs and per-channel metrics")
    ap.add_argument("--index", type=int, default=0, help="Test sample index to visualize")
    ap.add_argument("--group", default="other20", help="te|ti|na|ua|all|other20")
    ap.add_argument("--channels", default="", help='Optional explicit channel list, e.g. "2,3,12". Overrides --group.')
    ap.add_argument("--max_channels", type=int, default=0, help="If >0, limit number of channels processed (debug).")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(test_dir / "global_X_img_test.npy", mmap_mode="r")
    Y = np.load(test_dir / "global_Y_img_test.npy", mmap_mode="r")
    P = np.load(args.pred_path, mmap_mode="r")

    i = int(args.index)
    if i < 0 or i >= X.shape[0]:
        raise ValueError(f"index out of range: {i} (N={X.shape[0]})")

    if Y.shape[0] != P.shape[0]:
        raise ValueError(f"Pred/true N mismatch: Y={Y.shape}, P={P.shape}")
    if Y.shape[2:] != P.shape[2:]:
        raise ValueError(f"Pred/true spatial mismatch: Y={Y.shape}, P={P.shape}")

    c_out_true = int(Y.shape[1])
    c_out_pred = int(P.shape[1])
    if c_out_true != c_out_pred:
        raise ValueError(f"Channel mismatch: Y has {c_out_true} channels, pred has {c_out_pred} channels.")

    c_out = c_out_true
    labels = channel_labels(c_out)

    mask = X[i, 0].astype(np.uint8)  # (H,W) 0/1
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    chosen = parse_channel_list(args.channels, c_out)
    if not chosen:
        chosen = select_channels(args.group, c_out)

    if args.max_channels and args.max_channels > 0:
        chosen = chosen[: int(args.max_channels)]

    metrics_rows: List[Tuple[int, str, float, float, int]] = []

    for ch in chosen:
        y = np.array(Y[i, ch], dtype=np.float32)
        p = np.array(P[i, ch], dtype=np.float32)
        e = np.abs(p - y)

        mae, rmse = masked_metrics(p, y, mask)
        valid = int(mask.sum())
        lab = labels[ch] if ch < len(labels) else f"ch_{ch}"
        metrics_rows.append((ch, lab, mae, rmse, valid))

        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        masked_imshow(ax1, y, mask, f"{lab} true (idx={i}, ch={ch})")
        masked_imshow(ax2, p, mask, f"{lab} pred (MAE={mae:.6g}, RMSE={rmse:.6g})")
        masked_imshow(ax3, e, mask, f"{lab} abs error")

        fig.tight_layout()
        out_path = out_dir / f"pred_vs_true_idx_{i}_ch_{ch:02d}_{lab.replace('[','_').replace(']','')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Write metrics table
    lines = []
    lines.append(f"idx={i}")
    lines.append(f"valid_pixels={int(mask.sum())} / total_pixels={mask.size}")
    lines.append("ch\tlabel\tmae\trmse\tvalid_pixels")
    for ch, lab, mae, rmse, valid in metrics_rows:
        lines.append(f"{ch}\t{lab}\t{mae:.10g}\t{rmse:.10g}\t{valid}")
    (out_dir / f"metrics_idx_{i}_channels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {len(metrics_rows)} channel figures to: {out_dir}")
    print(f"Saved metrics table: {out_dir / f'metrics_idx_{i}_channels.tsv'}")


if __name__ == "__main__":
    main()