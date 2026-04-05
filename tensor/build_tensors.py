#!/usr/bin/env python3
"""
build_tensors.py  —  Step 1 of the Flow Matching 3-view pipeline

Builds 3 independent 4D tensors (one per view) from raw simulation data,
with the same 3-view layout as the original build_3view_tensors.py:

  view0 (top):    A = TOP_left  | gap |  B = TOP_right
  view1 (mid):    C = MID_left  | gap |  D = MID_center  | gap |  E = MID_right
  view2 (bot):    [gap_left]  F = Bottom_center  [gap_right]

Adjacency padding copies k border pixels across strip boundaries (within-view
horizontal, view2-F periodic wrap, and cross-view vertical padding between
view0↔view1 and view2↔view1).  This is IDENTICAL to the original code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS DIFFERENT FROM THE ORIGINAL build_3view_tensors.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Y tensors are stored ALREADY TRANSFORMED (log10 / asinh) in physical-
   transformed space (NOT normalized — normalization happens in the dataset).
   This means the Y tensor values are bounded and well-behaved (no 1e27
   density values that later overflow when you do 10^x).

2. Normalization statistics (y_mean, y_std, s_c for signed channels,
   x_mean, x_std) are computed from the TRAIN split and saved as a
   separate .npz file.  The test split reuses the train stats.

3. The X input channels are also pre-normalized (z-scored) and stored
   that way, so the dataset/model doesn't need to do any normalization.

4. A single norm_stats.npz file is produced when --split=train and must
   be passed via --norm_stats when building the test split.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  <pfx>_view{0,1,2}_X_<split>.npy    (N, C_in,  Hmax, Wmax)  float32
  <pfx>_view{0,1,2}_Y_<split>.npy    (N, C_out, Hmax, Wmax)  float32
  <pfx>_layout.npz                    layout metadata + static masks
  <pfx>_norm_stats.npz                normalization stats (train only)
  <pfx>_meta_<split>.npz              channel names, shapes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Build TRAIN (computes and saves normalization stats)
  python build_tensors.py --split train --data_root . \\
      --out_prefix scripts/tensor/fm3v/global3v \\
      --gap_px 15 --k 3

  # Build TEST (loads normalization stats from train)
  python build_tensors.py --split test --data_root . \\
      --out_prefix scripts/tensor/fm3v/global3v \\
      --gap_px 15 --k 3 \\
      --norm_stats scripts/tensor/fm3v/global3v_norm_stats.npz
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SLOT_TO_SPECNAME = {
    "TOP_right":     "L_top_E_right_fullrep",
    "TOP_left":      "F_top_M_left_fullrep",
    "MID_left":      "N_top_G_right_fullrep",
    "MID_right":     "D_top_K_left_fullrep",
    "MID_center":    "A_top_J_bottom_H_left",
    "Bottom_center": "C_top_I_bottom_B_left",
}

SLOT_TF = {k: "I" for k in SLOT_TO_SPECNAME}  # All identity transforms

VIEW_SLOTS = {
    "view0": ["TOP_left", "TOP_right"],
    "view1": ["MID_left", "MID_center", "MID_right"],
    "view2": ["Bottom_center"],
}

J_PER_EV = 1.602176634e-19

# Channel layout in Y tensor:
#   0     : te   (positive → log10)
#   1     : ti   (positive → log10)
#   2..11 : na   (positive → log10)   10 species
#  12..21 : ua   (signed  → asinh)   10 species
POS_CHANNELS    = set(range(0, 12))   # te, ti, na
SIGNED_CHANNELS = set(range(12, 22))  # ua

SPECIES = ["D0","D1","N0","N1","N2","N3","N4","N5","N6","N7"]
Y_CHANNEL_NAMES = ["te","ti"] + [f"na_{s}" for s in SPECIES] + [f"ua_{s}" for s in SPECIES]
X_CHANNEL_NAMES = ["mask"] + [f"x{i}" for i in range(8)]


# ═══════════════════════════════════════════════════════════════════════════
# Strip builder (IDENTICAL to original)
# ═══════════════════════════════════════════════════════════════════════════

def apply_transform_to_grids(pix_ix, pix_iy, mask, tf):
    if tf == "I":
        return pix_ix, pix_iy, mask
    if tf == "FU":
        return pix_ix[:, ::-1], pix_iy[:, ::-1], mask[:, ::-1]
    if tf == "FV":
        return pix_ix[::-1, :], pix_iy[::-1, :], mask[::-1, :]
    if tf == "R90":
        return np.rot90(pix_ix, k=3), np.rot90(pix_iy, k=3), np.rot90(mask, k=3)
    if tf == "R180":
        return np.rot90(pix_ix, k=2), np.rot90(pix_iy, k=2), np.rot90(mask, k=2)
    if tf == "R270":
        return np.rot90(pix_ix, k=1), np.rot90(pix_iy, k=1), np.rot90(mask, k=1)
    raise ValueError(tf)


def build_strip(u, spec, tf: str) -> dict:
    spec = u.normalize_rect_spec(spec)
    req = [lab for lab in [spec.get("top"), spec.get("bottom"),
                           spec.get("left"), spec.get("right")] if lab is not None]
    target_set = u.pick_component_by_labels(req)
    bottom_cells = None
    if spec.get("bottom") is not None:
        bottom_cells = [c for c in u.LABEL_TO_CELLS[spec["bottom"]] if c in target_set]
    coords, (W, H) = u.unroll_component_adjacency_preserving(target_set, bottom_cells=bottom_cells)
    pix_ix = np.full((H, W), -1, dtype=np.int32)
    pix_iy = np.full((H, W), -1, dtype=np.int32)
    mask   = np.zeros((H, W), dtype=np.uint8)
    for (ix, iy), (uu, vv) in coords.items():
        pix_ix[vv, uu] = ix
        pix_iy[vv, uu] = iy
        mask[vv, uu]   = 1
    pix_ix_t, pix_iy_t, mask_t = apply_transform_to_grids(pix_ix, pix_iy, mask, tf)
    return dict(W=int(mask_t.shape[1]), H=int(mask_t.shape[0]), tf=tf,
                mask=mask_t.astype(np.uint8),
                pix_cell_ix=pix_ix_t.astype(np.int32),
                pix_cell_iy=pix_iy_t.astype(np.int32))


# ═══════════════════════════════════════════════════════════════════════════
# View layout (IDENTICAL to original)
# ═══════════════════════════════════════════════════════════════════════════

def compute_view_layouts(strip_dicts, gap_px):
    A, B = strip_dicts["TOP_left"], strip_dicts["TOP_right"]
    C, D, E = strip_dicts["MID_left"], strip_dicts["MID_center"], strip_dicts["MID_right"]
    F = strip_dicts["Bottom_center"]

    h0 = max(A["H"], B["H"])
    W0, H0 = A["W"] + gap_px + B["W"], gap_px + h0 + gap_px

    h1 = max(C["H"], D["H"], E["H"])
    W1, H1 = C["W"] + gap_px + D["W"] + gap_px + E["W"], gap_px + h1 + gap_px

    W2, H2 = gap_px + F["W"] + gap_px, gap_px + F["H"] + gap_px

    origins = {
        "view0": {"TOP_left": (0, gap_px), "TOP_right": (A["W"]+gap_px, gap_px)},
        "view1": {"MID_left": (0, gap_px), "MID_center": (C["W"]+gap_px, gap_px),
                  "MID_right": (C["W"]+gap_px+D["W"]+gap_px, gap_px)},
        "view2": {"Bottom_center": (gap_px, gap_px)},
    }
    sizes = {"view0": (W0, H0), "view1": (W1, H1), "view2": (W2, H2)}
    return origins, sizes


def build_view_mask(strip_dicts, origins, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    for slot, (u0, v0) in origins.items():
        sd = strip_dicts[slot]
        mask[v0:v0+sd["H"], u0:u0+sd["W"]] |= sd["mask"]
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Field pasters (IDENTICAL to original)
# ═══════════════════════════════════════════════════════════════════════════

def paste_field_view(strip_dicts, origins, mask_view, field_2d):
    H, W = mask_view.shape
    out = np.zeros((H, W), dtype=np.float32)
    for slot, (u0, v0) in origins.items():
        sd = strip_dicts[slot]
        m = sd["mask"].astype(bool)
        ix, iy = sd["pix_cell_ix"][m], sd["pix_cell_iy"][m]
        sub = out[v0:v0+sd["H"], u0:u0+sd["W"]]
        sub[m] = field_2d[ix, iy].astype(np.float32)
        out[v0:v0+sd["H"], u0:u0+sd["W"]] = sub
    out *= mask_view.astype(np.float32)
    return out


def paste_field_view_3d(strip_dicts, origins, mask_view, field_3d):
    S = field_3d.shape[-1]
    H, W = mask_view.shape
    out = np.zeros((S, H, W), dtype=np.float32)
    for s in range(S):
        out[s] = paste_field_view(strip_dicts, origins, mask_view, field_3d[:,:,s])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Adjacency padding (IDENTICAL to original)
# ═══════════════════════════════════════════════════════════════════════════

def _copy_cols(dst, src, du0, du1, su0, su1, v0, v1):
    dst[:, v0:v1, du0:du1] = src[:, v0:v1, su0:su1]

def apply_padding_view(Xv, Yv, mask, strip_dicts, origins, gap_px, k, which_view):
    if k <= 0: return
    Cv_in = Xv.shape[0]
    stack = np.concatenate([Xv, Yv], axis=0)
    def bbox(slot):
        u0, v0 = origins[slot]; sd = strip_dicts[slot]
        return u0, v0, sd["W"], sd["H"]
    def pad_lr(left_slot, right_slot):
        uL,vL,wL,hL = bbox(left_slot); uR,vR,wR,hR = bbox(right_slot)
        h = min(hL, hR); v0_ = vL; v1_ = vL + h
        gl = uL + wL; gr = uR
        _copy_cols(stack,stack, gl,gl+k, uR,uR+k, v0_,v1_)
        _copy_cols(stack,stack, gr-k,gr, uL+wL-k,uL+wL, v0_,v1_)
        mask[v0_:v1_, gl:gl+k] = 1; mask[v0_:v1_, gr-k:gr] = 1
    def pad_F_wrap():
        uF,vF,wF,hF = bbox("Bottom_center")
        v0_,v1_ = vF, vF+hF
        _copy_cols(stack,stack, gap_px-k,gap_px, uF+wF-k,uF+wF, v0_,v1_)
        _copy_cols(stack,stack, gap_px+wF,gap_px+wF+k, uF,uF+k, v0_,v1_)
        mask[v0_:v1_, gap_px-k:gap_px] = 1
        mask[v0_:v1_, gap_px+wF:gap_px+wF+k] = 1
    if which_view == "view0": pad_lr("TOP_left","TOP_right")
    elif which_view == "view1": pad_lr("MID_left","MID_center"); pad_lr("MID_center","MID_right")
    elif which_view == "view2": pad_F_wrap()
    Xv[:] = stack[:Cv_in]; Yv[:] = stack[Cv_in:]
    Xv[0] = mask.astype(np.float32)


def apply_vertical_padding_cross_views(X0,Y0,m0,o0, X1,Y1,m1,o1, X2,Y2,m2,o2,
                                        strip_dicts, gap_px, k):
    if k <= 0: return
    def bbox(origins, slot):
        u0,v0 = origins[slot]; sd = strip_dicts[slot]
        return u0,v0,sd["W"],sd["H"]
    def stack_xy(Xv,Yv):
        return np.concatenate([Xv,Yv], axis=0)
    triplets = [
        ("TOP_left","MID_left",(X0,Y0,m0,o0),(X1,Y1,m1,o1)),
        ("TOP_right","MID_right",(X0,Y0,m0,o0),(X1,Y1,m1,o1)),
        ("Bottom_center","MID_center",(X2,Y2,m2,o2),(X1,Y1,m1,o1)),
    ]
    for top_slot,bot_slot,(Xt,Yt,mt,ot),(Xb,Yb,mb,ob) in triplets:
        uT,vT,wT,hT = bbox(ot,top_slot); uB,vB,wB,hB = bbox(ob,bot_slot)
        w = min(wT,wB); uT0,uT1 = uT,uT+w; uB0,uB1 = uB,uB+w
        gts = vT+hT; gte = gts+gap_px; gbe = vB; gbs = gbe-gap_px
        St = stack_xy(Xt,Yt); Sb = stack_xy(Xb,Yb)
        St[:,gts:gts+k, uT0:uT1] = Sb[:,vB:vB+k, uB0:uB1]
        Sb[:,gbe-k:gbe, uB0:uB1] = St[:,vT+hT-k:vT+hT, uT0:uT1]
        mt[gts:gts+k, uT0:uT1] = 1; mb[gbe-k:gbe, uB0:uB1] = 1
        Cv_t = Xt.shape[0]; Xt[:] = St[:Cv_t]; Yt[:] = St[Cv_t:]
        Xt[0] = mt.astype(np.float32)
        Cv_b = Xb.shape[0]; Xb[:] = Sb[:Cv_b]; Yb[:] = Sb[Cv_b:]
        Xb[0] = mb.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Normalization statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_norm_stats(X_views, Y_views, masks, c_in, c_out):
    """
    Compute per-channel masked mean/std for X and Y across all 3 views.
    X stats exclude mask channel (forced to mean=0, std=1).
    Y is already in transformed space (log10/asinh).

    Also computes s_c (physical-space std for signed channels) — but since
    Y is already transformed, s_c is embedded in the transform and recorded
    as 1.0 (the actual s_c was used when building Y).

    Returns dict with x_mean, x_std, y_mean, y_std (all float32 1D arrays).
    """
    # Gather all masked values per channel
    x_sums  = np.zeros(c_in, dtype=np.float64)
    x_sq    = np.zeros(c_in, dtype=np.float64)
    y_sums  = np.zeros(c_out, dtype=np.float64)
    y_sq    = np.zeros(c_out, dtype=np.float64)
    count   = 0.0

    for v in range(3):
        X, Y, M = X_views[v], Y_views[v], masks[v]  # (N,C,H,W), (N,C,H,W), (H,W)
        m = M.astype(np.float64)
        n_active = float(m.sum())
        N = X.shape[0]
        for i in range(N):
            count += n_active
            for c in range(c_in):
                xc = X[i, c].astype(np.float64)
                x_sums[c] += (xc * m).sum()
                x_sq[c]   += ((xc ** 2) * m).sum()
            for c in range(c_out):
                yc = Y[i, c].astype(np.float64)
                y_sums[c] += (yc * m).sum()
                y_sq[c]   += ((yc ** 2) * m).sum()

    x_mean = (x_sums / count).astype(np.float32)
    x_var  = (x_sq / count - (x_sums / count) ** 2).clip(1e-12)
    x_std  = np.sqrt(x_var).astype(np.float32)
    x_mean[0] = 0.0; x_std[0] = 1.0  # mask channel

    y_mean = (y_sums / count).astype(np.float32)
    y_var  = (y_sq / count - (y_sums / count) ** 2).clip(1e-12)
    y_std  = np.sqrt(y_var).astype(np.float32)

    return dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train","test"], required=True)
    ap.add_argument("--unroller_module", default="unrolled_strip_clockwise_adjpreserve")
    ap.add_argument("--gap_px", type=int, default=15)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_prefix", default="global3v")
    ap.add_argument("--norm_stats", default=None,
                    help="Path to norm_stats.npz from train split (required for --split test)")
    ap.add_argument("--limit_n", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    gap_px, k = args.gap_px, args.k
    pfx = args.out_prefix

    # Ensure output directory exists
    Path(pfx).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Build strip mappings ──
    u = importlib.import_module(args.unroller_module)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}
    strip_dicts = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=SLOT_TF[slot])

    # ── 2. Compute layouts ──
    origins_view, view_sizes = compute_view_layouts(strip_dicts, gap_px)
    W0,H0 = view_sizes["view0"]; W1,H1 = view_sizes["view1"]; W2,H2 = view_sizes["view2"]
    Hmax, Wmax = max(H0,H1,H2), max(W0,W1,W2)

    m0_local = build_view_mask(strip_dicts, origins_view["view0"], W0, H0)
    m1_local = build_view_mask(strip_dicts, origins_view["view1"], W1, H1)
    m2_local = build_view_mask(strip_dicts, origins_view["view2"], W2, H2)

    def embed(a):
        out = np.zeros((Hmax, Wmax), dtype=a.dtype); out[:a.shape[0],:a.shape[1]] = a; return out
    m0_base, m1_base, m2_base = embed(m0_local), embed(m1_local), embed(m2_local)

    # ── 3. Load raw simulation data ──
    X_raw  = np.load(split_dir/"X_tmp.npy")
    te_raw = np.load(split_dir/"te_tmp.npy")
    ti_raw = np.load(split_dir/"ti_tmp.npy")
    na_raw = np.load(split_dir/"na_tmp.npy")
    ua_raw = np.load(split_dir/"ua_tmp.npy")

    N = X_raw.shape[0]
    if args.limit_n > 0:
        N = min(N, args.limit_n)
        X_raw = X_raw[:N]; te_raw = te_raw[:N]; ti_raw = ti_raw[:N]
        na_raw = na_raw[:N]; ua_raw = ua_raw[:N]

    # ── 4. Compute s_c for signed channels (ua) from PHYSICAL space ──
    # s_c = per-species std of ua across all cells and simulations
    # We need this BEFORE transforming Y so we can do asinh(ua / s_c)
    if args.norm_stats:
        # Test split: load s_c from train
        ns = np.load(args.norm_stats, allow_pickle=True)
        s_c = ns["s_c"].astype(np.float64)
    else:
        # Train split: compute s_c from raw ua data
        s_c = np.zeros(10, dtype=np.float64)
        for sp in range(10):
            vals = ua_raw[:, :, :, sp].astype(np.float64).ravel()
            s_c[sp] = max(vals.std(), 1e-12)

    # ── 5. Channel counts ──
    c_in  = 1 + 8    # mask + 8 scalar parameters
    c_out = 22       # te + ti + 10*na + 10*ua

    # ── 6. Allocate output arrays ──
    X0 = np.zeros((N, c_in, Hmax, Wmax), dtype=np.float32)
    Y0 = np.zeros((N, c_out, Hmax, Wmax), dtype=np.float32)
    X1 = np.zeros((N, c_in, Hmax, Wmax), dtype=np.float32)
    Y1 = np.zeros((N, c_out, Hmax, Wmax), dtype=np.float32)
    X2 = np.zeros((N, c_in, Hmax, Wmax), dtype=np.float32)
    Y2 = np.zeros((N, c_out, Hmax, Wmax), dtype=np.float32)

    # ── 7. Fill X: mask channel + broadcast scalars ──
    X0[:, 0] = m0_base[None]; X1[:, 0] = m1_base[None]; X2[:, 0] = m2_base[None]
    for kk in range(8):
        vals = X_raw[:, kk].astype(np.float32)[:, None, None]
        X0[:, 1+kk] = vals * m0_base[None]
        X1[:, 1+kk] = vals * m1_base[None]
        X2[:, 1+kk] = vals * m2_base[None]

    # ── 8. Fill Y: paste fields in TRANSFORMED space ──
    # Transform BEFORE pasting so the stored Y is already safe
    EPS_LOG = 1.0  # Use eps=1 for log10 to compress the range
    # With eps=1: log10(1e10 + 1) ≈ 10, log10(1e-5 + 1) ≈ 0
    # Range becomes [0, 27] instead of [-5, 27] — more compact

    print(f"Pasting Y fields (transformed) for {N} simulations ...")

    for i in range(N):
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{N}")

        # te (channel 0): log10(te_eV + eps)
        te_eV = (te_raw[i] / J_PER_EV).astype(np.float64)
        te_t  = np.log10(np.maximum(te_eV, 0.0) + EPS_LOG).astype(np.float32)
        for v, (sd, org, ml, Yi, Hi, Wi) in enumerate([
            (strip_dicts, origins_view["view0"], m0_local, Y0, H0, W0),
            (strip_dicts, origins_view["view1"], m1_local, Y1, H1, W1),
            (strip_dicts, origins_view["view2"], m2_local, Y2, H2, W2),
        ]):
            Yi[i, 0, :Hi, :Wi] = paste_field_view(sd, org, ml, te_t)

        # ti (channel 1): log10(ti_eV + eps)
        ti_eV = (ti_raw[i] / J_PER_EV).astype(np.float64)
        ti_t  = np.log10(np.maximum(ti_eV, 0.0) + EPS_LOG).astype(np.float32)
        for v, (sd, org, ml, Yi, Hi, Wi) in enumerate([
            (strip_dicts, origins_view["view0"], m0_local, Y0, H0, W0),
            (strip_dicts, origins_view["view1"], m1_local, Y1, H1, W1),
            (strip_dicts, origins_view["view2"], m2_local, Y2, H2, W2),
        ]):
            Yi[i, 1, :Hi, :Wi] = paste_field_view(sd, org, ml, ti_t)

        # na (channels 2..11): log10(na + eps)
        for sp in range(10):
            na_sp = na_raw[i, :, :, sp].astype(np.float64)
            na_t  = np.log10(np.maximum(na_sp, 0.0) + EPS_LOG).astype(np.float32)
            for v, (sd, org, ml, Yi, Hi, Wi) in enumerate([
                (strip_dicts, origins_view["view0"], m0_local, Y0, H0, W0),
                (strip_dicts, origins_view["view1"], m1_local, Y1, H1, W1),
                (strip_dicts, origins_view["view2"], m2_local, Y2, H2, W2),
            ]):
                Yi[i, 2+sp, :Hi, :Wi] = paste_field_view(sd, org, ml, na_t)

        # ua (channels 12..21): asinh(ua / s_c)
        for sp in range(10):
            ua_sp = ua_raw[i, :, :, sp].astype(np.float64)
            ua_t  = np.arcsinh(ua_sp / s_c[sp]).astype(np.float32)
            for v, (sd, org, ml, Yi, Hi, Wi) in enumerate([
                (strip_dicts, origins_view["view0"], m0_local, Y0, H0, W0),
                (strip_dicts, origins_view["view1"], m1_local, Y1, H1, W1),
                (strip_dicts, origins_view["view2"], m2_local, Y2, H2, W2),
            ]):
                Yi[i, 12+sp, :Hi, :Wi] = paste_field_view(sd, org, ml, ua_t)

    # ── 9. Apply adjacency padding ──
    print("Applying adjacency padding ...")
    if k > 0:
        for i in range(N):
            mi0, mi1, mi2 = m0_base.copy(), m1_base.copy(), m2_base.copy()
            apply_padding_view(X0[i],Y0[i],mi0, strip_dicts,origins_view["view0"],gap_px,k,"view0")
            apply_padding_view(X1[i],Y1[i],mi1, strip_dicts,origins_view["view1"],gap_px,k,"view1")
            apply_padding_view(X2[i],Y2[i],mi2, strip_dicts,origins_view["view2"],gap_px,k,"view2")
            apply_vertical_padding_cross_views(
                X0[i],Y0[i],mi0,origins_view["view0"],
                X1[i],Y1[i],mi1,origins_view["view1"],
                X2[i],Y2[i],mi2,origins_view["view2"],
                strip_dicts, gap_px, k)
            Y0[i] *= mi0[None]; Y1[i] *= mi1[None]; Y2[i] *= mi2[None]
            for kk in range(8):
                X0[i,1+kk] = X_raw[i,kk].astype(np.float32)*mi0
                X1[i,1+kk] = X_raw[i,kk].astype(np.float32)*mi1
                X2[i,1+kk] = X_raw[i,kk].astype(np.float32)*mi2
    else:
        Y0 *= m0_base[None,None]; Y1 *= m1_base[None,None]; Y2 *= m2_base[None,None]

    # ── 10. Compute or load normalization stats ──
    if args.norm_stats:
        print(f"Loading normalization stats from {args.norm_stats}")
        ns = dict(np.load(args.norm_stats, allow_pickle=True))
    else:
        print("Computing normalization stats (train split) ...")
        ns = compute_norm_stats(
            [X0, X1, X2], [Y0, Y1, Y2],
            [m0_base, m1_base, m2_base], c_in, c_out)
        ns["s_c"] = s_c.astype(np.float32)
        ns["eps_log"] = np.float32(EPS_LOG)
        stats_path = f"{pfx}_norm_stats.npz"
        np.savez_compressed(stats_path, **ns)
        print(f"  Saved: {stats_path}")

    # ── 11. Apply normalization to X and Y IN-PLACE ──
    x_mean = ns["x_mean"].reshape(1, -1, 1, 1)
    x_std  = ns["x_std"].reshape(1, -1, 1, 1)
    y_mean = ns["y_mean"].reshape(1, -1, 1, 1)
    y_std  = ns["y_std"].reshape(1, -1, 1, 1)

    for Xv, Yv, m_base in [(X0,Y0,m0_base),(X1,Y1,m1_base),(X2,Y2,m2_base)]:
        Xv[:] = ((Xv - x_mean) / x_std) * m_base[None, None]
        # Re-set mask channel to raw 0/1 (it was normalized above but we want it binary)
        Xv[:, 0] = m_base[None]
        Yv[:] = ((Yv - y_mean) / y_std) * m_base[None, None]

    # ── 12. Save ──
    split = args.split
    for tag, arr in [("view0_X",X0),("view0_Y",Y0),("view1_X",X1),("view1_Y",Y1),
                     ("view2_X",X2),("view2_Y",Y2)]:
        p = f"{pfx}_{tag}_{split}.npy"
        np.save(p, arr)
        print(f"  Saved {tag}: {p}  shape={arr.shape}")

    # Layout
    layout_path = f"{pfx}_layout.npz"
    np.savez_compressed(layout_path,
        gap_px=np.int32(gap_px), k=np.int32(k),
        Hmax=np.int32(Hmax), Wmax=np.int32(Wmax),
        W0=np.int32(W0), H0=np.int32(H0),
        W1=np.int32(W1), H1=np.int32(H1),
        W2=np.int32(W2), H2=np.int32(H2),
        mask_view0=m0_base, mask_view1=m1_base, mask_view2=m2_base)
    print(f"  Saved layout: {layout_path}")

    # Meta
    meta_path = f"{pfx}_meta_{split}.npz"
    np.savez_compressed(meta_path,
        split=split, N=N, Hmax=Hmax, Wmax=Wmax,
        C_in=c_in, C_out=c_out, gap_px=gap_px, k=k,
        x_channels=X_CHANNEL_NAMES, y_channels=Y_CHANNEL_NAMES,
        eps_log=np.float32(EPS_LOG))
    print(f"  Saved meta: {meta_path}")

    print(f"\n{'='*60}")
    print(f"Split={split} | N={N} | canvas={Hmax}x{Wmax} | C_in={c_in} | C_out={c_out}")
    print(f"  view0: {H0}x{W0}  | view1: {H1}x{W1}  | view2: {H2}x{W2}")
    print(f"  Y stored in TRANSFORMED+NORMALIZED space (safe, no overflow)")
    print(f"  eps_log={EPS_LOG} | gap_px={gap_px} | k={k}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
