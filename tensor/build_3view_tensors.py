#!/usr/bin/env python3
"""
build_3view_tensors.py

Build 3 independent 4D tensors — one per view — each following the exact same
logic as build_global_tensor.py (single canvas, standard 4D shape).

Views:
  view0 (top):    A=TOP_left | gap | B=TOP_right
  view1 (mid):    C=MID_left | gap | D=MID_center | gap | E=MID_right
  view2 (bot):    F=Bottom_center  (with left/right gap padding)

Each view produces its own independent rectangular canvas with size (Hv, Wv),
padded to a common (Hmax, Wmax) so all three tensors share the same spatial
dimensions and can be fed to the same model without reshaping.

Outputs (all float32):
  <out_prefix>_view0_X_img_<split>.npy   (N, C_in,  Hmax, Wmax)
  <out_prefix>_view0_Y_img_<split>.npy   (N, C_out, Hmax, Wmax)
  <out_prefix>_view1_X_img_<split>.npy   (N, C_in,  Hmax, Wmax)
  <out_prefix>_view1_Y_img_<split>.npy   (N, C_out, Hmax, Wmax)
  <out_prefix>_view2_X_img_<split>.npy   (N, C_in,  Hmax, Wmax)
  <out_prefix>_view2_Y_img_<split>.npy   (N, C_out, Hmax, Wmax)
  <out_prefix>_layout_map_3views.npz     layout metadata
  <out_prefix>_meta_<split>.npz          shapes / channel names

Slot transforms use identity ("I") as in tensor_3_images.py.
Adjacency padding (k pixels) is written into the gap regions, exactly as in
tensor_3_images.py (horizontal within-view + F periodic wrap + cross-view
vertical padding written into the gap rows present on each canvas).

Usage example:
  python build_3view_tensors.py \
    --split train \
    --data_root . \
    --out_prefix scripts/tensor/3views_4d/train/global3v \
    --gap_px 15 --k 3 --include_geometry
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Slot / spec / transform definitions  (identity transforms, same as
# tensor_3_images.py — different from build_global_tensor.py which uses R180
# for TOP slots)
# ---------------------------------------------------------------------------
SLOT_TO_SPECNAME = {
    "TOP_right":     "L_top_E_right_fullrep",
    "TOP_left":      "F_top_M_left_fullrep",
    "MID_left":      "N_top_G_right_fullrep",
    "MID_right":     "D_top_K_left_fullrep",
    "MID_center":    "A_top_J_bottom_H_left",
    "Bottom_center": "C_top_I_bottom_B_left",
}

SLOT_TF = {
    "TOP_left":      "I",
    "TOP_right":     "I",
    "MID_left":      "I",
    "MID_right":     "I",
    "MID_center":    "I",
    "Bottom_center": "I",
}

# Which slots belong to each view
VIEW_SLOTS = {
    "view0": ["TOP_left", "TOP_right"],
    "view1": ["MID_left", "MID_center", "MID_right"],
    "view2": ["Bottom_center"],
}

J_PER_EV = 1.602176634e-19


# ---------------------------------------------------------------------------
# Strip builder  (identical to build_global_tensor.py / tensor_3_images.py)
# ---------------------------------------------------------------------------
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
    Ht, Wt = mask_t.shape
    return dict(
        W=int(Wt), H=int(Ht), tf=tf,
        mask=mask_t.astype(np.uint8),
        pix_cell_ix=pix_ix_t.astype(np.int32),
        pix_cell_iy=pix_iy_t.astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Per-view canvas layout  (same geometry as tensor_3_images.py)
# ---------------------------------------------------------------------------
def compute_view_layouts(
    strip_dicts: Dict, gap_px: int
) -> Tuple[Dict[str, Dict[str, Tuple[int, int]]], Dict[str, Tuple[int, int]]]:
    """
    Returns:
      origins_view  : {"view0": {slot: (u0,v0)}, ...}
      view_sizes    : {"view0": (W, H), ...}   — local canvas sizes (before padding)
    """
    A = strip_dicts["TOP_left"]
    B = strip_dicts["TOP_right"]
    C = strip_dicts["MID_left"]
    D = strip_dicts["MID_center"]
    E = strip_dicts["MID_right"]
    F = strip_dicts["Bottom_center"]

    # view0:  [gap_top] A | gap | B [gap_bot]
    h0   = max(A["H"], B["H"])
    w0   = A["W"] + gap_px + B["W"]
    H0   = gap_px + h0 + gap_px      # top + bot gap rows for cross-view padding
    W0   = w0

    # view1:  [gap_top] C | gap | D | gap | E [gap_bot]
    h1   = max(C["H"], D["H"], E["H"])
    w1   = C["W"] + gap_px + D["W"] + gap_px + E["W"]
    H1   = gap_px + h1 + gap_px
    W1   = w1

    # view2:  [gap_top] [gap_left] F [gap_right] [gap_bot]
    h2   = F["H"]
    w2   = gap_px + F["W"] + gap_px
    H2   = gap_px + h2 + gap_px
    W2   = w2

    origins_view = {
        "view0": {
            "TOP_left":  (0,                gap_px),
            "TOP_right": (A["W"] + gap_px,  gap_px),
        },
        "view1": {
            "MID_left":   (0,                                    gap_px),
            "MID_center": (C["W"] + gap_px,                      gap_px),
            "MID_right":  (C["W"] + gap_px + D["W"] + gap_px,    gap_px),
        },
        "view2": {
            "Bottom_center": (gap_px, gap_px),
        },
    }

    view_sizes = {
        "view0": (int(W0), int(H0)),
        "view1": (int(W1), int(H1)),
        "view2": (int(W2), int(H2)),
    }

    return origins_view, view_sizes


# ---------------------------------------------------------------------------
# Mask builder for one view canvas
# ---------------------------------------------------------------------------
def build_view_mask(
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    W: int, H: int,
) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for slot, (u0, v0) in origins.items():
        sd = strip_dicts[slot]
        mask[v0:v0 + sd["H"], u0:u0 + sd["W"]] |= sd["mask"]
    return mask


# ---------------------------------------------------------------------------
# Field pasters  (same as build_global_tensor.py but scoped to one view)
# ---------------------------------------------------------------------------
def paste_field_view(
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    mask_view: np.ndarray,
    field_2d: np.ndarray,
) -> np.ndarray:
    """field_2d: (104,50) → (H,W) float32, gaps=0."""
    H, W = mask_view.shape
    out  = np.zeros((H, W), dtype=np.float32)
    for slot, (u0, v0) in origins.items():
        sd  = strip_dicts[slot]
        m   = sd["mask"].astype(bool)
        ix  = sd["pix_cell_ix"][m]
        iy  = sd["pix_cell_iy"][m]
        sub = out[v0:v0 + sd["H"], u0:u0 + sd["W"]]
        sub[m] = field_2d[ix, iy].astype(np.float32)
        out[v0:v0 + sd["H"], u0:u0 + sd["W"]] = sub
    out *= mask_view.astype(np.float32)
    return out


def paste_field_view_3d(
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    mask_view: np.ndarray,
    field_3d: np.ndarray,
) -> np.ndarray:
    """field_3d: (104,50,S) → (S,H,W) float32."""
    S = field_3d.shape[-1]
    H, W = mask_view.shape
    out = np.zeros((S, H, W), dtype=np.float32)
    for s in range(S):
        out[s] = paste_field_view(strip_dicts, origins, mask_view, field_3d[:, :, s])
    return out


def build_geometry_channels_view(
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    mask_view: np.ndarray,
    geom_dir: str,
) -> np.ndarray:
    """Returns (2, H, W): centroid_x, centroid_y."""
    crx = np.load(os.path.join(geom_dir, "crx.npy"))
    cry = np.load(os.path.join(geom_dir, "cry.npy"))
    cx  = crx.mean(axis=2)
    cy  = cry.mean(axis=2)
    gx  = paste_field_view(strip_dicts, origins, mask_view, cx)
    gy  = paste_field_view(strip_dicts, origins, mask_view, cy)
    geom = np.stack([gx, gy], axis=0).astype(np.float32)
    geom *= mask_view.astype(np.float32)[None]
    return geom


# ---------------------------------------------------------------------------
# Adjacency padding  (ported from tensor_3_images.py, operates on 4D
# X/Y tensors  (C,H,W) + mask (H,W) for a single sample)
# ---------------------------------------------------------------------------
def _copy_cols(dst, src, dst_u0, dst_u1, src_u0, src_u1, v0, v1):
    dst[:, v0:v1, dst_u0:dst_u1] = src[:, v0:v1, src_u0:src_u1]


def _copy_rows(dst, src, u0, u1, dst_v0, dst_v1, src_v0, src_v1):
    dst[:, dst_v0:dst_v1, u0:u1] = src[:, src_v0:src_v1, u0:u1]


def apply_padding_view(
    Xv: np.ndarray,          # (C_in, H, W)
    Yv: np.ndarray,          # (C_out, H, W)
    mask: np.ndarray,        # (H, W) uint8
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    gap_px: int,
    k: int,
    which_view: str,
) -> None:
    """In-place: write k-column/row adjacency padding into the gap regions."""
    if k <= 0:
        return
    if 2 * k > gap_px:
        raise ValueError(f"Need 2*k <= gap_px. Got k={k}, gap_px={gap_px}.")

    Cv_in  = Xv.shape[0]
    stack  = np.concatenate([Xv.astype(np.float32), Yv.astype(np.float32)], axis=0)

    def bbox(slot):
        u0, v0 = origins[slot]
        sd = strip_dicts[slot]
        return u0, v0, sd["W"], sd["H"]

    def pad_lr(left_slot, right_slot):
        uL, vL, wL, hL = bbox(left_slot)
        uR, vR, wR, hR = bbox(right_slot)
        h         = min(hL, hR)
        v0_       = vL
        v1_       = vL + h
        gap_left  = uL + wL
        gap_right = uR
        _copy_cols(stack, stack, gap_left,         gap_left + k,  uR,       uR + k,   v0_, v1_)
        _copy_cols(stack, stack, gap_right - k,    gap_right,     uL + wL - k, uL + wL, v0_, v1_)
        mask[v0_:v1_, gap_left:gap_left + k]           = 1
        mask[v0_:v1_, gap_right - k:gap_right]         = 1

    def pad_F_wrap():
        uF, vF, wF, hF = bbox("Bottom_center")
        v0_ = vF
        v1_ = vF + hF
        # right side of left gap ← last k cols of F
        _copy_cols(stack, stack, gap_px - k, gap_px,         uF + wF - k, uF + wF, v0_, v1_)
        # left side of right gap ← first k cols of F
        _copy_cols(stack, stack, gap_px + wF, gap_px + wF + k, uF, uF + k,          v0_, v1_)
        mask[v0_:v1_, gap_px - k:gap_px]               = 1
        mask[v0_:v1_, gap_px + wF:gap_px + wF + k]     = 1

    if which_view == "view0":
        pad_lr("TOP_left", "TOP_right")
    elif which_view == "view1":
        pad_lr("MID_left",   "MID_center")
        pad_lr("MID_center", "MID_right")
    elif which_view == "view2":
        pad_F_wrap()
    else:
        raise ValueError(which_view)

    Xv[:] = stack[:Cv_in].astype(np.float32)
    Yv[:] = stack[Cv_in:].astype(np.float32)
    Xv[0] = mask.astype(np.float32)


def apply_vertical_padding_cross_views(
    X0, Y0, m0, origins0,   # view0 tensors (C,H,W) + mask (H,W)
    X1, Y1, m1, origins1,   # view1
    X2, Y2, m2, origins2,   # view2
    strip_dicts: Dict,
    gap_px: int,
    k: int,
) -> None:
    """
    Cross-view vertical padding (in-place):
      TOP_left   (view0) ↔ MID_left   (view1)
      TOP_right  (view0) ↔ MID_right  (view1)
      Bottom_center (view2) ↔ MID_center (view1)
    Gap rows are those that exist at the bottom of view0 / top of view1, etc.
    """
    if k <= 0:
        return
    if 2 * k > gap_px:
        raise ValueError(f"Need 2*k <= gap_px. Got k={k}, gap_px={gap_px}.")

    def bbox(origins, slot):
        u0, v0 = origins[slot]
        sd = strip_dicts[slot]
        return u0, v0, sd["W"], sd["H"]

    def stack_xy(Xv, Yv):
        return np.concatenate([Xv.astype(np.float32), Yv.astype(np.float32)], axis=0)

    triplets = [
        ("TOP_left",      "MID_left",   (X0, Y0, m0, origins0), (X1, Y1, m1, origins1)),
        ("TOP_right",     "MID_right",  (X0, Y0, m0, origins0), (X1, Y1, m1, origins1)),
        ("Bottom_center", "MID_center", (X2, Y2, m2, origins2), (X1, Y1, m1, origins1)),
    ]

    for top_slot, bot_slot, (Xt, Yt, mt, ot), (Xb, Yb, mb, ob) in triplets:
        uT, vT, wT, hT = bbox(ot, top_slot)
        uB, vB, wB, hB = bbox(ob, bot_slot)

        w   = min(wT, wB)
        uT0 = uT;  uT1 = uT + w
        uB0 = uB;  uB1 = uB + w

        # gap rows in the TOP view canvas (below the block)
        gap_top_start = vT + hT
        gap_top_end   = gap_top_start + gap_px
        # gap rows in the BOTTOM view canvas (above the block)
        gap_bot_end   = vB
        gap_bot_start = gap_bot_end - gap_px

        if gap_top_end > Xt.shape[1] or gap_bot_start < 0:
            raise RuntimeError("Canvas does not contain expected vertical gap rows.")

        St = stack_xy(Xt, Yt)
        Sb = stack_xy(Xb, Yb)

        # fill top gap: first k rows from bottom block
        St[:, gap_top_start:gap_top_start + k, uT0:uT1] = Sb[:, vB:vB + k, uB0:uB1]
        # fill bot gap: last k rows from top block
        Sb[:, gap_bot_end - k:gap_bot_end,     uB0:uB1] = St[:, vT + hT - k:vT + hT, uT0:uT1]

        mt[gap_top_start:gap_top_start + k, uT0:uT1] = 1
        mb[gap_bot_end - k:gap_bot_end,     uB0:uB1] = 1

        Cv_in_t = Xt.shape[0]
        Xt[:] = St[:Cv_in_t];  Yt[:] = St[Cv_in_t:]
        Xt[0] = mt.astype(np.float32)

        Cv_in_b = Xb.shape[0]
        Xb[:] = Sb[:Cv_in_b];  Yb[:] = Sb[Cv_in_b:]
        Xb[0] = mb.astype(np.float32)


# ---------------------------------------------------------------------------
# Output-target selection  (same as build_global_tensor.py)
# ---------------------------------------------------------------------------
def parse_out_targets(s: str) -> List[str]:
    if s is None or s.strip() == "":
        return ["te", "ti", "na", "ua"]
    items = [t.strip().lower() for t in s.split(",") if t.strip()]
    allowed = {"te", "ti", "na", "ua"}
    bad = [t for t in items if t not in allowed]
    if bad:
        raise ValueError(f"Unknown out_targets: {bad}. Allowed: {sorted(allowed)}")
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",          choices=["train", "test"], required=True)
    ap.add_argument("--unroller_module", default="unrolled_strip_clockwise_adjpreserve")
    ap.add_argument("--gap_px",         type=int, default=15)
    ap.add_argument("--k",              type=int, default=3,
                    help="Adjacency padding half-width (0 = no padding)")
    ap.add_argument("--geom_dir",       default="geometry")
    ap.add_argument("--data_root",      default=".")
    ap.add_argument("--out_prefix",     default="global3v")
    ap.add_argument("--include_geometry", action="store_true")
    ap.add_argument("--out_targets",    default="te,ti,na,ua")
    ap.add_argument("--include_fnixap", action="store_true")
    ap.add_argument("--limit_n",        type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    gap_px    = int(args.gap_px)
    k         = int(args.k)

    # ------------------------------------------------------------------ #
    # 1.  Build strip mappings                                             #
    # ------------------------------------------------------------------ #
    u            = importlib.import_module(args.unroller_module)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}

    missing = [nm for nm in SLOT_TO_SPECNAME.values() if nm not in name_to_spec]
    if missing:
        raise RuntimeError(f"Missing RECT_SPECS names: {missing}")

    strip_dicts: Dict = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=SLOT_TF[slot])

    # ------------------------------------------------------------------ #
    # 2.  Compute per-view canvas layouts                                  #
    # ------------------------------------------------------------------ #
    origins_view, view_sizes = compute_view_layouts(strip_dicts, gap_px=gap_px)

    W0, H0 = view_sizes["view0"]
    W1, H1 = view_sizes["view1"]
    W2, H2 = view_sizes["view2"]

    # Pad all canvases to the same spatial size so downstream models see
    # identically-shaped tensors for all three views.
    Hmax = int(max(H0, H1, H2))
    Wmax = int(max(W0, W1, W2))

    # Local (unpadded) masks — used for pasting
    m0_local = build_view_mask(strip_dicts, origins_view["view0"], W0, H0)
    m1_local = build_view_mask(strip_dicts, origins_view["view1"], W1, H1)
    m2_local = build_view_mask(strip_dicts, origins_view["view2"], W2, H2)

    def embed(arr2d: np.ndarray) -> np.ndarray:
        """Embed a (H, W) array into (Hmax, Wmax) by zero-padding bottom/right."""
        out = np.zeros((Hmax, Wmax), dtype=arr2d.dtype)
        H, W = arr2d.shape
        out[:H, :W] = arr2d
        return out

    # Common-size masks (used as the static base; per-sample copies updated in loop)
    m0_base = embed(m0_local)
    m1_base = embed(m1_local)
    m2_base = embed(m2_local)

    # ------------------------------------------------------------------ #
    # 3.  Load simulation data                                             #
    # ------------------------------------------------------------------ #
    X_raw  = np.load(split_dir / "X_tmp.npy")      # (N,8)
    te_raw = np.load(split_dir / "te_tmp.npy")     # (N,104,50)
    ti_raw = np.load(split_dir / "ti_tmp.npy")
    na_raw = np.load(split_dir / "na_tmp.npy")     # (N,104,50,10)
    ua_raw = np.load(split_dir / "ua_tmp.npy")
    fnixap = None
    if args.include_fnixap:
        fnixap = np.load(split_dir / "fnixap_tmp.npy").astype(np.float32)

    N = int(X_raw.shape[0])
    if args.limit_n and args.limit_n > 0:
        N = min(N, int(args.limit_n))
        X_raw  = X_raw[:N]
        te_raw = te_raw[:N]
        ti_raw = ti_raw[:N]
        na_raw = na_raw[:N]
        ua_raw = ua_raw[:N]
        if fnixap is not None:
            fnixap = fnixap[:N]

    # ------------------------------------------------------------------ #
    # 4.  Channel counts                                                   #
    # ------------------------------------------------------------------ #
    c_in = 1 + 8 + (2 if args.include_geometry else 0)

    targets   = parse_out_targets(args.out_targets)
    out_parts: List[Tuple[str, int]] = []
    out_names: List[str]             = []
    if "te" in targets: out_parts.append(("te", 1))
    if "ti" in targets: out_parts.append(("ti", 1))
    if "na" in targets: out_parts.append(("na", 10))
    if "ua" in targets: out_parts.append(("ua", 10))
    c_out = int(sum(ch for _, ch in out_parts))
    for name, ch in out_parts:
        if ch == 1:
            out_names.append(name)
        else:
            out_names.extend([f"{name}{s}" for s in range(ch)])

    # ------------------------------------------------------------------ #
    # 5.  Pre-compute static geometry channels (view-local, then embedded) #
    # ------------------------------------------------------------------ #
    geom0 = geom1 = geom2 = None
    if args.include_geometry:
        g0_local = build_geometry_channels_view(
            strip_dicts, origins_view["view0"], m0_local, args.geom_dir)  # (2,H0,W0)
        g1_local = build_geometry_channels_view(
            strip_dicts, origins_view["view1"], m1_local, args.geom_dir)
        g2_local = build_geometry_channels_view(
            strip_dicts, origins_view["view2"], m2_local, args.geom_dir)

        geom0 = np.zeros((2, Hmax, Wmax), dtype=np.float32)
        geom1 = np.zeros((2, Hmax, Wmax), dtype=np.float32)
        geom2 = np.zeros((2, Hmax, Wmax), dtype=np.float32)
        geom0[:, :H0, :W0] = g0_local
        geom1[:, :H1, :W1] = g1_local
        geom2[:, :H2, :W2] = g2_local

    # ------------------------------------------------------------------ #
    # 6.  Allocate output arrays                                           #
    # ------------------------------------------------------------------ #
    X0_img = np.zeros((N, c_in,  Hmax, Wmax), dtype=np.float32)
    Y0_img = np.zeros((N, c_out, Hmax, Wmax), dtype=np.float32)
    X1_img = np.zeros((N, c_in,  Hmax, Wmax), dtype=np.float32)
    Y1_img = np.zeros((N, c_out, Hmax, Wmax), dtype=np.float32)
    X2_img = np.zeros((N, c_in,  Hmax, Wmax), dtype=np.float32)
    Y2_img = np.zeros((N, c_out, Hmax, Wmax), dtype=np.float32)

    # Static mask channel (channel 0): same for every simulation before padding
    X0_img[:, 0] = m0_base.astype(np.float32)[None]
    X1_img[:, 0] = m1_base.astype(np.float32)[None]
    X2_img[:, 0] = m2_base.astype(np.float32)[None]

    # Broadcast scalar channels (1..8) with mask — same per simulation for the
    # mask shape, values vary per sample
    for kk in range(8):
        X0_img[:, 1 + kk] = (X_raw[:, kk].astype(np.float32)[:, None, None]) * m0_base[None]
        X1_img[:, 1 + kk] = (X_raw[:, kk].astype(np.float32)[:, None, None]) * m1_base[None]
        X2_img[:, 1 + kk] = (X_raw[:, kk].astype(np.float32)[:, None, None]) * m2_base[None]

    if args.include_geometry:
        X0_img[:, 9:11] = geom0[None]
        X1_img[:, 9:11] = geom1[None]
        X2_img[:, 9:11] = geom2[None]

    # ------------------------------------------------------------------ #
    # 7.  Paste Y fields per simulation                                    #
    # ------------------------------------------------------------------ #
    ch0 = 0
    for name, ch in out_parts:
        if name in ("te", "ti"):
            raw = te_raw if name == "te" else ti_raw
            for i in range(N):
                field = (raw[i] / J_PER_EV).astype(np.float32)

                y0_local = paste_field_view(strip_dicts, origins_view["view0"], m0_local, field)
                y1_local = paste_field_view(strip_dicts, origins_view["view1"], m1_local, field)
                y2_local = paste_field_view(strip_dicts, origins_view["view2"], m2_local, field)

                Y0_img[i, ch0, :H0, :W0] = y0_local
                Y1_img[i, ch0, :H1, :W1] = y1_local
                Y2_img[i, ch0, :H2, :W2] = y2_local
            ch0 += 1

        elif name in ("na", "ua"):
            raw = na_raw if name == "na" else ua_raw
            for i in range(N):
                field3 = raw[i]  # (104,50,10)

                y0_local = paste_field_view_3d(strip_dicts, origins_view["view0"], m0_local, field3)  # (10,H0,W0)
                y1_local = paste_field_view_3d(strip_dicts, origins_view["view1"], m1_local, field3)
                y2_local = paste_field_view_3d(strip_dicts, origins_view["view2"], m2_local, field3)

                Y0_img[i, ch0:ch0 + 10, :H0, :W0] = y0_local
                Y1_img[i, ch0:ch0 + 10, :H1, :W1] = y1_local
                Y2_img[i, ch0:ch0 + 10, :H2, :W2] = y2_local
            ch0 += 10

    # ------------------------------------------------------------------ #
    # 8.  Apply adjacency padding per simulation                           #
    # ------------------------------------------------------------------ #
    if k > 0:
        for i in range(N):
            # Fresh per-sample mask copies (padding extends them)
            mi0 = m0_base.copy()
            mi1 = m1_base.copy()
            mi2 = m2_base.copy()

            # Within-view horizontal + F wrap
            apply_padding_view(X0_img[i], Y0_img[i], mi0,
                               strip_dicts, origins_view["view0"], gap_px, k, "view0")
            apply_padding_view(X1_img[i], Y1_img[i], mi1,
                               strip_dicts, origins_view["view1"], gap_px, k, "view1")
            apply_padding_view(X2_img[i], Y2_img[i], mi2,
                               strip_dicts, origins_view["view2"], gap_px, k, "view2")

            # Cross-view vertical padding
            apply_vertical_padding_cross_views(
                X0_img[i], Y0_img[i], mi0, origins_view["view0"],
                X1_img[i], Y1_img[i], mi1, origins_view["view1"],
                X2_img[i], Y2_img[i], mi2, origins_view["view2"],
                strip_dicts, gap_px=gap_px, k=k,
            )

            # Enforce gaps = 0 in Y after padding
            Y0_img[i] *= mi0.astype(np.float32)[None]
            Y1_img[i] *= mi1.astype(np.float32)[None]
            Y2_img[i] *= mi2.astype(np.float32)[None]

            # Re-apply mask to scalar X channels after padding extended the mask
            for kk in range(8):
                X0_img[i, 1 + kk] = X_raw[i, kk].astype(np.float32) * mi0.astype(np.float32)
                X1_img[i, 1 + kk] = X_raw[i, kk].astype(np.float32) * mi1.astype(np.float32)
                X2_img[i, 1 + kk] = X_raw[i, kk].astype(np.float32) * mi2.astype(np.float32)

    else:
        # No padding: just enforce gaps=0 with static mask
        Y0_img *= m0_base.astype(np.float32)[None, None]
        Y1_img *= m1_base.astype(np.float32)[None, None]
        Y2_img *= m2_base.astype(np.float32)[None, None]

    # ------------------------------------------------------------------ #
    # 9.  Save outputs                                                     #
    # ------------------------------------------------------------------ #
    pfx  = args.out_prefix
    split = args.split

    paths = {
        "view0_X": f"{pfx}_view0_X_img_{split}.npy",
        "view0_Y": f"{pfx}_view0_Y_img_{split}.npy",
        "view1_X": f"{pfx}_view1_X_img_{split}.npy",
        "view1_Y": f"{pfx}_view1_Y_img_{split}.npy",
        "view2_X": f"{pfx}_view2_X_img_{split}.npy",
        "view2_Y": f"{pfx}_view2_Y_img_{split}.npy",
    }

    np.save(paths["view0_X"], X0_img)
    np.save(paths["view0_Y"], Y0_img)
    np.save(paths["view1_X"], X1_img)
    np.save(paths["view1_Y"], Y1_img)
    np.save(paths["view2_X"], X2_img)
    np.save(paths["view2_Y"], Y2_img)

    if fnixap is not None:
        fnixap_path = f"{pfx}_fnixap_{split}.npy"
        np.save(fnixap_path, fnixap)
        print(f"Saved fnixap:    {fnixap_path}  shape={fnixap.shape}")

    # Layout map
    layout_path = f"{pfx}_layout_map_3views.npz"
    np.savez_compressed(
        layout_path,
        gap_px=np.int32(gap_px),
        k=np.int32(k),
        Hmax=np.int32(Hmax), Wmax=np.int32(Wmax),
        W0=np.int32(W0), H0=np.int32(H0),
        W1=np.int32(W1), H1=np.int32(H1),
        W2=np.int32(W2), H2=np.int32(H2),
        mask_view0=m0_base,
        mask_view1=m1_base,
        mask_view2=m2_base,
    )

    # Meta
    in_ch_names = (["mask"] + [f"x{kk}" for kk in range(8)] +
                   (["centroid_x", "centroid_y"] if args.include_geometry else []))
    meta = dict(
        split=split, N=N,
        Hmax=Hmax, Wmax=Wmax,
        C_in=c_in, C_out=c_out,
        gap_px=gap_px, k=k,
        views=["view0", "view1", "view2"],
        in_channels=in_ch_names,
        out_channels=out_names,
    )
    meta_path = f"{pfx}_meta_{split}.npz"
    np.savez_compressed(meta_path, **{mk: np.array(mv, dtype=object)
                                      for mk, mv in meta.items()})

    # Summary
    print(f"\n{'='*60}")
    print(f"Split: {split}  |  N={N}  |  canvas={Hmax}×{Wmax}  |  C_in={c_in}  C_out={c_out}")
    print(f"  view0 local: {H0}×{W0}   slots: TOP_left, TOP_right")
    print(f"  view1 local: {H1}×{W1}   slots: MID_left, MID_center, MID_right")
    print(f"  view2 local: {H2}×{W2}   slots: Bottom_center")
    print(f"{'='*60}")
    for tag, p in paths.items():
        shape = X0_img.shape if "view0" in tag and "_X" in tag else \
                Y0_img.shape if "view0" in tag and "_Y" in tag else \
                X1_img.shape if "view1" in tag and "_X" in tag else \
                Y1_img.shape if "view1" in tag and "_Y" in tag else \
                X2_img.shape if "view2" in tag and "_X" in tag else \
                Y2_img.shape
        print(f"  Saved {tag:10s}: {p}   shape={shape}")
    print(f"  Saved layout:     {layout_path}")
    print(f"  Saved meta:       {meta_path}")


if __name__ == "__main__":
    main()
