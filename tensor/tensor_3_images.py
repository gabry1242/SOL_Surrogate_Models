#!/usr/bin/env python3
"""
tensor_3_images.py

Build 3-view tensors (A|gap|B), (C|gap|D|gap|E), (gap|F|gap) with adjacency padding.

Blocks/slots mapping (same unroller + same per-slot transforms as build_global_tensor.py):
A = TOP_left
B = TOP_right
C = MID_left
D = MID_center
E = MID_right
F = Bottom_center

Adjacency padding (k pixels) is written INTO the gap regions (gap_px wide/high):
- Horizontal adjacency: fill k columns on each side of the gap with neighbor content.
- Vertical adjacency: fill k rows on each side of the gap with neighbor content.
- Fleft adj Fright: periodic wrap of F into its left/right gap columns.

Outputs:
- <out_prefix>_layout_map_3views.npz     (origins + view sizes + masks)
- <out_prefix>_mask_views_k{k}.npy       uint8 (3,Hmax,Wmax)
- <out_prefix>_X_img_<split>.npy         float32 (N,3,C_in,Hmax,Wmax)
- <out_prefix>_Y_img_<split>.npy         float32 (N,3,C_out,Hmax,Wmax)
- <out_prefix>_meta_<split>.npz

This is a direct adaptation of build_global_tensor.py logic (same unrolling/paste),
but switches to 3 views and adds adjacency padding. :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -----------------------
# Slot layout: (slot -> RECT_SPECS name)  (same as build_global_tensor.py)
# -----------------------
SLOT_TO_SPECNAME = {
    "TOP_right": "L_top_E_right_fullrep",
    "TOP_left": "F_top_M_left_fullrep",
    "MID_left": "N_top_G_right_fullrep",
    "MID_right": "D_top_K_left_fullrep",
    "MID_center": "A_top_J_bottom_H_left",
    "Bottom_center": "C_top_I_bottom_B_left",
}

# -----------------------
# Fixed transforms per slot (same as build_global_tensor.py)
# -----------------------
SLOT_TF = {
    "TOP_left": "I",
    "TOP_right": "I",
    "MID_left": "I",
    "MID_right": "I",
    "MID_center": "I",
    "Bottom_center": "I",
}

# -----------------------
# Transform helpers (same as build_global_tensor.py)
# -----------------------
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


# -----------------------
# Build one strip mapping (pixel->cell indices) using the unroller (same as build_global_tensor.py)
# -----------------------
def build_strip(u, spec, tf: str):
    spec = u.normalize_rect_spec(spec)

    req = [lab for lab in [spec.get("top"), spec.get("bottom"), spec.get("left"), spec.get("right")] if lab is not None]
    target_set = u.pick_component_by_labels(req)

    bottom_cells = None
    if spec.get("bottom") is not None:
        bottom_cells = [c for c in u.LABEL_TO_CELLS[spec["bottom"]] if c in target_set]

    coords, (W, H) = u.unroll_component_adjacency_preserving(target_set, bottom_cells=bottom_cells)

    pix_ix = np.full((H, W), -1, dtype=np.int32)
    pix_iy = np.full((H, W), -1, dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)

    for (ix, iy), (uu, vv) in coords.items():
        pix_ix[vv, uu] = ix
        pix_iy[vv, uu] = iy
        mask[vv, uu] = 1

    pix_ix_t, pix_iy_t, mask_t = apply_transform_to_grids(pix_ix, pix_iy, mask, tf)

    Ht, Wt = mask_t.shape
    return dict(
        W=int(Wt),
        H=int(Ht),
        tf=tf,
        mask=mask_t.astype(np.uint8),
        pix_cell_ix=pix_ix_t.astype(np.int32),
        pix_cell_iy=pix_iy_t.astype(np.int32),
    )


# -----------------------
# View layout
# -----------------------
def compute_3view_layout(strip_dicts: Dict, gap_px: int) -> Tuple[Dict, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Define three separate canvases (local coordinates), then embed them into common (Hmax,Wmax).

    View0 (top):   A | gap | B, plus bottom gap (for A<->C and B<->E vertical padding)
    View1 (mid):   top gap + (C | gap | D | gap | E) + bottom gap
    View2 (bot):   top gap + (left gap | F | right gap)

    Returns:
      origins_view: dict(view_name -> dict(slot -> (u0,v0)))
      sizes: (W0,H0),(W1,H1),(W2,H2)  local sizes (before embedding)
    """
    A = strip_dicts["TOP_left"]
    B = strip_dicts["TOP_right"]
    C = strip_dicts["MID_left"]
    D = strip_dicts["MID_center"]
    E = strip_dicts["MID_right"]
    F = strip_dicts["Bottom_center"]

    # local sizes
    h0 = max(A["H"], B["H"])
    w0 = A["W"] + gap_px + B["W"]
    H0 = gap_px + h0 + gap_px  # bottom gap

    h1 = max(C["H"], D["H"], E["H"])
    w1 = C["W"] + gap_px + D["W"] + gap_px + E["W"]
    H1 = gap_px + h1 + gap_px  # top + bottom gap

    h2 = F["H"]
    w2 = gap_px + F["W"] + gap_px
    H2 = gap_px + h2 + gap_px # top gap

    origins_view = {
        "view0": {
            "TOP_left": (0, gap_px),
            "TOP_right": (A["W"] + gap_px, gap_px),
        },
        "view1": {
            "MID_left": (0, gap_px),
            "MID_center": (C["W"] + gap_px, gap_px),
            "MID_right": (C["W"] + gap_px + D["W"] + gap_px, gap_px),
        },
        "view2": {
            "Bottom_center": (gap_px, gap_px),
        },
    }

    return origins_view, (int(w0), int(H0)), (int(w1), int(H1)), (int(w2), int(H2))


def embed_into_common(mask_local: np.ndarray, Hmax: int, Wmax: int) -> np.ndarray:
    out = np.zeros((Hmax, Wmax), dtype=mask_local.dtype)
    H, W = mask_local.shape
    out[:H, :W] = mask_local
    return out


# -----------------------
# Paste helpers
# -----------------------
def paste_mask_view(strip_dicts: Dict, origins: Dict[str, Tuple[int, int]], W: int, H: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for slot, (u0, v0) in origins.items():
        sd = strip_dicts[slot]
        h, w = sd["H"], sd["W"]
        mask[v0 : v0 + h, u0 : u0 + w] |= sd["mask"]
    return mask


def paste_field_view(strip_dicts: Dict, origins: Dict[str, Tuple[int, int]], mask_view: np.ndarray, field_2d: np.ndarray) -> np.ndarray:
    """
    field_2d: (104,50)
    """
    H, W = mask_view.shape
    out = np.zeros((H, W), dtype=np.float32)
    for slot, (u0, v0) in origins.items():
        sd = strip_dicts[slot]
        m = sd["mask"].astype(bool)
        pix_ix = sd["pix_cell_ix"]
        pix_iy = sd["pix_cell_iy"]

        sub = out[v0 : v0 + sd["H"], u0 : u0 + sd["W"]]
        ix = pix_ix[m]
        iy = pix_iy[m]
        sub[m] = field_2d[ix, iy].astype(np.float32)
        out[v0 : v0 + sd["H"], u0 : u0 + sd["W"]] = sub

    # enforce gaps = 0
    out *= mask_view.astype(np.float32)
    return out


def paste_field_view_3d(strip_dicts: Dict, origins: Dict[str, Tuple[int, int]], mask_view: np.ndarray, field_3d: np.ndarray) -> np.ndarray:
    """
    field_3d: (104,50,S)
    returns: (S,H,W)
    """
    S = field_3d.shape[-1]
    H, W = mask_view.shape
    out = np.zeros((S, H, W), dtype=np.float32)
    for s in range(S):
        out[s] = paste_field_view(strip_dicts, origins, mask_view, field_3d[:, :, s])
    return out


def build_geometry_channels_view(strip_dicts: Dict, origins: Dict[str, Tuple[int, int]], mask_view: np.ndarray, geom_dir: str) -> np.ndarray:
    """
    Returns (2,H,W): centroid_x, centroid_y pasted, gaps=0.
    """
    crx = np.load(os.path.join(geom_dir, "crx.npy"))
    cry = np.load(os.path.join(geom_dir, "cry.npy"))
    cx = crx.mean(axis=2)
    cy = cry.mean(axis=2)
    gx = paste_field_view(strip_dicts, origins, mask_view, cx)
    gy = paste_field_view(strip_dicts, origins, mask_view, cy)
    geom = np.stack([gx, gy], axis=0).astype(np.float32)
    geom *= mask_view.astype(np.float32)[None, :, :]
    return geom


# -----------------------
# Padding primitives (operate on (C,H,W) + mask (H,W))
# -----------------------
def _copy_cols(dst: np.ndarray, src: np.ndarray, dst_u0: int, dst_u1: int, src_u0: int, src_u1: int, v0: int, v1: int):
    # inclusive-exclusive
    dst[:, v0:v1, dst_u0:dst_u1] = src[:, v0:v1, src_u0:src_u1]


def _copy_rows(dst: np.ndarray, src: np.ndarray, u0: int, u1: int, dst_v0: int, dst_v1: int, src_v0: int, src_v1: int):
    dst[:, dst_v0:dst_v1, u0:u1] = src[:, src_v0:src_v1, u0:u1]


def apply_padding_view(
    Xv: np.ndarray,  # (C_in,H,W)
    Yv: np.ndarray,  # (C_out,H,W)
    mask: np.ndarray,  # (H,W) uint8
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    gap_px: int,
    k: int,
    which_view: str,
):
    """
    Writes padding INTO the gap regions by copying k rows/cols from neighbor blocks.

    Implemented adjacencies (as requested):
      A rightadj B / B leftadj A           (view0)
      A bottomadj C / C topadj A           (view0<->view1 uses view0 bottom gap and view1 top gap)
      B bottomadj E / E topadj B           (view0<->view1)
      C rightadj D / D leftadj C           (view1)
      D rightadj E / E leftadj D           (view1)
      D topadj F / F bottomadj D           (view1<->view2 uses view1 bottom gap and view2 top gap)
      Fleft adj Fright                     (view2 periodic into its left/right gap)
    """
    if k <= 0:
        return

    if 2 * k > gap_px:
        raise ValueError(f"Need 2*k <= gap_px. Got k={k}, gap_px={gap_px}.")

    # stack channels so padding copies mask+scalars+geom the same way
    # we will update mask explicitly too (safer for uint8)
    Cv_in = Xv.shape[0]
    Cv_out = Yv.shape[0]

    # Convenience: work on a float stack for copying, then put back
    stack = np.concatenate([Xv.astype(np.float32), Yv.astype(np.float32)], axis=0)  # (Ctot,H,W)

    def bbox(slot: str) -> Tuple[int, int, int, int]:
        u0, v0 = origins[slot]
        sd = strip_dicts[slot]
        return u0, v0, sd["W"], sd["H"]

    def pad_lr(left_slot: str, right_slot: str):
        uL, vL, wL, hL = bbox(left_slot)
        uR, vR, wR, hR = bbox(right_slot)
        if vL != vR:
            raise RuntimeError(f"pad_lr expects aligned top edges: {left_slot},{right_slot}")

        # overlap height: min of block heights (padding only where both exist)
        h = min(hL, hR)
        v0 = vL
        v1 = vL + h

        # gap is from (uL+wL) .. (uR-1)
        gap_left = uL + wL
        gap_right = uR
        if gap_right - gap_left != gap_px:
            raise RuntimeError(f"Expected gap_px between {left_slot} and {right_slot}, got {gap_right-gap_left}")

        # Fill:
        # - left side of gap: first k cols from right block (its left edge)
        _copy_cols(stack, stack, gap_left, gap_left + k, uR, uR + k, v0, v1)
        # - right side of gap: last k cols from left block (its right edge)
        _copy_cols(stack, stack, gap_right - k, gap_right, uL + wL - k, uL + wL, v0, v1)

        # mask: mark these copied pixels valid (where source was valid)
        # use X mask channel 0 as validity reference (after copy it is already correct),
        # but force uint8 mask to 1 where any copied-from-valid happened.
        mask[v0:v1, gap_left:gap_left + k] = 1
        mask[v0:v1, gap_right - k:gap_right] = 1

    def pad_tb(top_slot: str, bot_slot: str, top_gap_in_view: bool, bot_gap_in_view: bool):
        """
        top_slot bottomadj bot_slot with a vertical gap of size gap_px between them.
        We assume the gap rows exist in the current view tensor around the relevant boundary.
        """
        uT, vT, wT, hT = bbox(top_slot)
        uB, vB, wB, hB = bbox(bot_slot)

        if uT != uB:
            raise RuntimeError(f"pad_tb expects aligned left edges: {top_slot},{bot_slot}")

        w = min(wT, wB)
        u0 = uT
        u1 = uT + w

        # gap is from (vT+hT) .. (vB-1)
        gap_top = vT + hT
        gap_bot = vB
        if gap_bot - gap_top != gap_px:
            raise RuntimeError(f"Expected gap_px between {top_slot} and {bot_slot}, got {gap_bot-gap_top}")

        # Fill:
        # - top side of gap: first k rows from bottom block (its top edge)
        _copy_rows(stack, stack, u0, u1, gap_top, gap_top + k, vB, vB + k)
        # - bottom side of gap: last k rows from top block (its bottom edge)
        _copy_rows(stack, stack, u0, u1, gap_bot - k, gap_bot, vT + hT - k, vT + hT)

        mask[gap_top:gap_top + k, u0:u1] = 1
        mask[gap_bot - k:gap_bot, u0:u1] = 1

    def pad_F_wrap():
        # Bottom_center in view2 has left and right gaps of size gap_px.
        uF, vF, wF, hF = bbox("Bottom_center")
        # left gap: [0, gap_px)
        # right gap: [gap_px+wF, gap_px+wF+gap_px)
        left_gap_u0 = 0
        left_gap_u1 = gap_px
        right_gap_u0 = gap_px + wF
        right_gap_u1 = gap_px + wF + gap_px

        v0 = vF
        v1 = vF + hF

        # Copy last k cols of F into rightmost k cols of left gap
        _copy_cols(stack, stack, left_gap_u1 - k, left_gap_u1, uF + wF - k, uF + wF, v0, v1)
        # Copy first k cols of F into leftmost k cols of right gap
        _copy_cols(stack, stack, right_gap_u0, right_gap_u0 + k, uF, uF + k, v0, v1)

        mask[v0:v1, left_gap_u1 - k:left_gap_u1] = 1
        mask[v0:v1, right_gap_u0:right_gap_u0 + k] = 1

    # Apply per-view operations
    if which_view == "view0":
        pad_lr("TOP_left", "TOP_right")
        # vertical padding is handled by view0 bottom gap + view1 top gap; done in view0/view1 separately.
    elif which_view == "view1":
        pad_lr("MID_left", "MID_center")
        pad_lr("MID_center", "MID_right")
        # vertical padding with view0/view2 handled separately in those views.
    elif which_view == "view2":
        pad_F_wrap()
    else:
        raise ValueError(which_view)

    # write back
    Xv[:, :, :] = stack[:Cv_in].astype(np.float32)
    Yv[:, :, :] = stack[Cv_in:].astype(np.float32)
    Xv[0, :, :] = mask.astype(np.float32)  # force mask channel consistent


def apply_vertical_padding_cross_views(
    X0: np.ndarray, Y0: np.ndarray, m0: np.ndarray, origins0: Dict[str, Tuple[int, int]],  # view0
    X1: np.ndarray, Y1: np.ndarray, m1: np.ndarray, origins1: Dict[str, Tuple[int, int]],  # view1
    X2: np.ndarray, Y2: np.ndarray, m2: np.ndarray, origins2: Dict[str, Tuple[int, int]],  # view2
    strip_dicts: Dict,
    gap_px: int,
    k: int,
):
    """
    Implements:
      A bottomadj C  (TOP_left bottomadj MID_left)
      B bottomadj E  (TOP_right bottomadj MID_right)
      D topadj F     (MID_center topadj Bottom_center)  [as requested; note this is "topadj" in your list]

    These are written into the explicit gap rows that exist in the view canvases:
      view0 has bottom gap rows
      view1 has top+bottom gap rows
      view2 has top gap rows
    """
    if k <= 0:
        return
    if 2 * k > gap_px:
        raise ValueError(f"Need 2*k <= gap_px. Got k={k}, gap_px={gap_px}.")

    def bbox(origins: Dict[str, Tuple[int, int]], slot: str) -> Tuple[int, int, int, int]:
        u0, v0 = origins[slot]
        sd = strip_dicts[slot]
        return u0, v0, sd["W"], sd["H"]

    # stack helper
    def stack_xy(Xv, Yv):
        return np.concatenate([Xv.astype(np.float32), Yv.astype(np.float32)], axis=0)

    # A (view0) bottomadj C (view1)
    # A is TOP_left in view0, C is MID_left in view1; gap is:
    # - view0 bottom gap: rows [h0, h0+gap_px)
    # - view1 top gap:    rows [0, gap_px)
    for top_slot, bot_slot, (Xt, Yt, mt, ot), (Xb, Yb, mb, ob) in [
        ("TOP_left", "MID_left", (X0, Y0, m0, origins0), (X1, Y1, m1, origins1)),
        ("TOP_right", "MID_right", (X0, Y0, m0, origins0), (X1, Y1, m1, origins1)),
        ("Bottom_center", "MID_center", (X2, Y2, m2, origins2), (X1, Y1, m1, origins1)),
    ]:
        uT, vT, wT, hT = bbox(ot, top_slot)
        uB, vB, wB, hB = bbox(ob, bot_slot)

        # Horizontal overlap within each local packing
        w = min(wT, wB)
        uT0, uT1 = uT, uT + w
        uB0, uB1 = uB, uB + w

        # Identify the gap rows in the TOP view just below top block
        gap_top_view_start = vT + hT
        gap_top_view_end = gap_top_view_start + gap_px

        # Identify the gap rows in the BOTTOM view just above bottom block
        gap_bot_view_end = vB
        gap_bot_view_start = gap_bot_view_end - gap_px

        # Sanity
        if gap_top_view_end > Xt.shape[1] or gap_bot_view_start < 0:
            raise RuntimeError("View canvases do not contain expected vertical gap regions.")

        St = stack_xy(Xt, Yt)
        Sb = stack_xy(Xb, Yb)

        # Fill:
        # - in TOP gap: first k rows from bottom block (bottom view, top edge)
        St[:, gap_top_view_start:gap_top_view_start + k, uT0:uT1] = Sb[:, vB:vB + k, uB0:uB1]
        # - in BOTTOM gap: last k rows from top block (top view, bottom edge)
        Sb[:, gap_bot_view_end - k:gap_bot_view_end, uB0:uB1] = St[:, vT + hT - k:vT + hT, uT0:uT1]

        mt[gap_top_view_start:gap_top_view_start + k, uT0:uT1] = 1
        mb[gap_bot_view_end - k:gap_bot_view_end, uB0:uB1] = 1

        # write back + force mask channel
        Cv_in = Xt.shape[0]
        Xt[:, :, :] = St[:Cv_in]
        Yt[:, :, :] = St[Cv_in:]
        Xt[0, :, :] = mt.astype(np.float32)

        Cv_in_b = Xb.shape[0]
        Xb[:, :, :] = Sb[:Cv_in_b]
        Yb[:, :, :] = Sb[Cv_in_b:]
        Xb[0, :, :] = mb.astype(np.float32)


# -----------------------
# Targets selection (same names as build_global_tensor.py)
# -----------------------
def parse_out_targets(s: str) -> List[str]:
    if s is None or s.strip() == "":
        return ["te", "ti", "na", "ua"]
    items = [t.strip().lower() for t in s.split(",") if t.strip()]
    allowed = {"te", "ti", "na", "ua"}
    bad = [t for t in items if t not in allowed]
    if bad:
        raise ValueError(f"Unknown out_targets: {bad}. Allowed: {sorted(allowed)}")
    return items


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--unroller_module", default="unrolled_strip_clockwise_adjpreserve")
    ap.add_argument("--gap_px", type=int, default=15)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--geom_dir", default="geometry")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_prefix", default="global3")

    ap.add_argument("--include_geometry", action="store_true")
    ap.add_argument("--out_targets", default="te,ti,na,ua")
    ap.add_argument("--include_fnixap", action="store_true")
    ap.add_argument("--limit_n", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split

    gap_px = int(args.gap_px)
    k = int(args.k)

    # -------- build strip mappings --------
    u = importlib.import_module(args.unroller_module)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}

    missing = [nm for nm in SLOT_TO_SPECNAME.values() if nm not in name_to_spec]
    if missing:
        raise RuntimeError(f"Missing RECT_SPECS names: {missing}")

    strip_dicts = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        tf = SLOT_TF[slot]
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=tf)

    # -------- compute 3-view layout (local) + common embedding size --------
    origins_view, (W0, H0), (W1, H1), (W2, H2) = compute_3view_layout(strip_dicts, gap_px=gap_px)
    Hmax = int(max(H0, H1, H2))
    Wmax = int(max(W0, W1, W2))

    # local masks
    m0_local = paste_mask_view(strip_dicts, origins_view["view0"], W0, H0)
    m1_local = paste_mask_view(strip_dicts, origins_view["view1"], W1, H1)
    m2_local = paste_mask_view(strip_dicts, origins_view["view2"], W2, H2)

    # embed masks to common size
    m0 = embed_into_common(m0_local, Hmax, Wmax)
    m1 = embed_into_common(m1_local, Hmax, Wmax)
    m2 = embed_into_common(m2_local, Hmax, Wmax)

    mask_views = np.stack([m0, m1, m2], axis=0).astype(np.uint8)  # (3,Hmax,Wmax)

    # -------- load split data --------
    X = np.load(split_dir / "X_tmp.npy")          # (N,8)
    te = np.load(split_dir / "te_tmp.npy")        # (N,104,50)
    ti = np.load(split_dir / "ti_tmp.npy")        # (N,104,50)
    na = np.load(split_dir / "na_tmp.npy")        # (N,104,50,10)
    ua = np.load(split_dir / "ua_tmp.npy")        # (N,104,50,10)
    fnixap = None
    if args.include_fnixap:
        fnixap = np.load(split_dir / "fnixap_tmp.npy").astype(np.float32)

    N = int(X.shape[0])
    if args.limit_n and args.limit_n > 0:
        N = min(N, int(args.limit_n))
        X = X[:N]
        te = te[:N]
        ti = ti[:N]
        na = na[:N]
        ua = ua[:N]
        if fnixap is not None:
            fnixap = fnixap[:N]

    # -------- build input tensor X_img (N,3,C_in,Hmax,Wmax) --------
    c_in = 1 + 8 + (2 if args.include_geometry else 0)
    X_img = np.zeros((N, 3, c_in, Hmax, Wmax), dtype=np.float32)

    # base mask channel
    X_img[:, 0, 0, :, :] = m0.astype(np.float32)[None, :, :]
    X_img[:, 1, 0, :, :] = m1.astype(np.float32)[None, :, :]
    X_img[:, 2, 0, :, :] = m2.astype(np.float32)[None, :, :]

    # broadcast scalars with mask
    for kk in range(8):
        X_img[:, 0, 1 + kk, :, :] = (X[:, kk].astype(np.float32)[:, None, None]) * m0.astype(np.float32)[None, :, :]
        X_img[:, 1, 1 + kk, :, :] = (X[:, kk].astype(np.float32)[:, None, None]) * m1.astype(np.float32)[None, :, :]
        X_img[:, 2, 1 + kk, :, :] = (X[:, kk].astype(np.float32)[:, None, None]) * m2.astype(np.float32)[None, :, :]

    if args.include_geometry:
        g0_local = build_geometry_channels_view(strip_dicts, origins_view["view0"], m0_local, geom_dir=args.geom_dir)  # (2,H0,W0)
        g1_local = build_geometry_channels_view(strip_dicts, origins_view["view1"], m1_local, geom_dir=args.geom_dir)  # (2,H1,W1)
        g2_local = build_geometry_channels_view(strip_dicts, origins_view["view2"], m2_local, geom_dir=args.geom_dir)  # (2,H2,W2)

        g0 = np.zeros((2, Hmax, Wmax), dtype=np.float32); g0[:, :H0, :W0] = g0_local
        g1 = np.zeros((2, Hmax, Wmax), dtype=np.float32); g1[:, :H1, :W1] = g1_local
        g2 = np.zeros((2, Hmax, Wmax), dtype=np.float32); g2[:, :H2, :W2] = g2_local

        X_img[:, 0, 1 + 8:1 + 8 + 2, :, :] = g0[None, :, :, :]
        X_img[:, 1, 1 + 8:1 + 8 + 2, :, :] = g1[None, :, :, :]
        X_img[:, 2, 1 + 8:1 + 8 + 2, :, :] = g2[None, :, :, :]

    # -------- build output tensor Y_img (N,3,C_out,Hmax,Wmax) --------
    targets = parse_out_targets(args.out_targets)
    out_parts: List[Tuple[str, int]] = []
    out_names: List[str] = []

    if "te" in targets:
        out_parts.append(("te", 1))
    if "ti" in targets:
        out_parts.append(("ti", 1))
    if "na" in targets:
        out_parts.append(("na", 10))
    if "ua" in targets:
        out_parts.append(("ua", 10))

    c_out = int(sum(ch for _, ch in out_parts))
    Y_img = np.zeros((N, 3, c_out, Hmax, Wmax), dtype=np.float32)

    J_PER_EV = 1.602176634e-19

    # paste per simulation
    ch0 = 0
    for name, ch in out_parts:
        if name in ("te", "ti"):
            for i in range(N):
                field = te[i] if name == "te" else ti[i]
                field = (field / J_PER_EV).astype(np.float32)

                # view0
                y0_local = paste_field_view(strip_dicts, origins_view["view0"], m0_local, field)
                y0 = np.zeros((Hmax, Wmax), dtype=np.float32); y0[:H0, :W0] = y0_local
                # view1
                y1_local = paste_field_view(strip_dicts, origins_view["view1"], m1_local, field)
                y1 = np.zeros((Hmax, Wmax), dtype=np.float32); y1[:H1, :W1] = y1_local
                # view2
                y2_local = paste_field_view(strip_dicts, origins_view["view2"], m2_local, field)
                y2 = np.zeros((Hmax, Wmax), dtype=np.float32); y2[:H2, :W2] = y2_local

                Y_img[i, 0, ch0, :, :] = y0
                Y_img[i, 1, ch0, :, :] = y1
                Y_img[i, 2, ch0, :, :] = y2

            out_names.append(name)
            ch0 += 1

        elif name in ("na", "ua"):
            for i in range(N):
                field3 = na[i] if name == "na" else ua[i]  # (104,50,10)
                y0_local = paste_field_view_3d(strip_dicts, origins_view["view0"], m0_local, field3)  # (10,H0,W0)
                y1_local = paste_field_view_3d(strip_dicts, origins_view["view1"], m1_local, field3)  # (10,H1,W1)
                y2_local = paste_field_view_3d(strip_dicts, origins_view["view2"], m2_local, field3)  # (10,H2,W2)

                y0 = np.zeros((10, Hmax, Wmax), dtype=np.float32); y0[:, :H0, :W0] = y0_local
                y1 = np.zeros((10, Hmax, Wmax), dtype=np.float32); y1[:, :H1, :W1] = y1_local
                y2 = np.zeros((10, Hmax, Wmax), dtype=np.float32); y2[:, :H2, :W2] = y2_local

                Y_img[i, 0, ch0:ch0 + 10, :, :] = y0
                Y_img[i, 1, ch0:ch0 + 10, :, :] = y1
                Y_img[i, 2, ch0:ch0 + 10, :, :] = y2

            out_names.extend([f"{name}{s}" for s in range(10)])
            ch0 += 10
        else:
            raise RuntimeError(name)

    # -------- apply adjacency padding (per simulation) --------
    out_mask_views = np.zeros((N, 3, Hmax, Wmax), dtype=np.uint8)

    for i in range(N):
        # copy base masks for this sample (will be updated by padding)
        mi0 = m0.copy()
        mi1 = m1.copy()
        mi2 = m2.copy()

        # per-view in-place padding (within-view horizontal + F wrap)
        apply_padding_view(
            X_img[i, 0], Y_img[i, 0], mi0,
            strip_dicts, origins_view["view0"], gap_px=gap_px, k=k, which_view="view0",
        )
        apply_padding_view(
            X_img[i, 1], Y_img[i, 1], mi1,
            strip_dicts, origins_view["view1"], gap_px=gap_px, k=k, which_view="view1",
        )
        apply_padding_view(
            X_img[i, 2], Y_img[i, 2], mi2,
            strip_dicts, origins_view["view2"], gap_px=gap_px, k=k, which_view="view2",
        )

        # cross-view vertical padding (A<->C, B<->E, D<->F)
        apply_vertical_padding_cross_views(
            X_img[i, 0], Y_img[i, 0], mi0, origins_view["view0"],
            X_img[i, 1], Y_img[i, 1], mi1, origins_view["view1"],
            X_img[i, 2], Y_img[i, 2], mi2, origins_view["view2"],
            strip_dicts, gap_px=gap_px, k=k,
        )

        # store final masks
        out_mask_views[i, 0] = mi0
        out_mask_views[i, 1] = mi1
        out_mask_views[i, 2] = mi2

        # enforce output gaps are 0 outside mask
        Y_img[i, 0] *= mi0.astype(np.float32)[None, :, :]
        Y_img[i, 1] *= mi1.astype(np.float32)[None, :, :]
        Y_img[i, 2] *= mi2.astype(np.float32)[None, :, :]

        # enforce scalar channels stay consistent with mask after padding
        # (copying already did it, but mask may have expanded; safest to re-mask scalars)
        for kk in range(8):
            X_img[i, 0, 1 + kk] = (X[i, kk].astype(np.float32)) * mi0.astype(np.float32)
            X_img[i, 1, 1 + kk] = (X[i, kk].astype(np.float32)) * mi1.astype(np.float32)
            X_img[i, 2, 1 + kk] = (X[i, kk].astype(np.float32)) * mi2.astype(np.float32)

    # -------- save outputs --------
    out_prefix = Path(args.out_prefix)

    out_x = f"{out_prefix}_X_img_{args.split}.npy"
    out_y = f"{out_prefix}_Y_img_{args.split}.npy"
    out_mask = f"{out_prefix}_mask_views_k{k}.npy"
    out_layout = f"{out_prefix}_layout_map_3views.npz"

    np.save(out_x, X_img)
    np.save(out_y, Y_img)
    np.save(out_mask, out_mask_views.astype(np.uint8))

    if fnixap is not None:
        out_f = f"{out_prefix}_fnixap_{args.split}.npy"
        np.save(out_f, fnixap)

    # layout metadata for debugging/visualization
    np.savez_compressed(
        out_layout,
        gap_px=np.int32(gap_px),
        k=np.int32(k),
        Hmax=np.int32(Hmax),
        Wmax=np.int32(Wmax),
        W0=np.int32(W0), H0=np.int32(H0),
        W1=np.int32(W1), H1=np.int32(H1),
        W2=np.int32(W2), H2=np.int32(H2),
        mask_view0=embed_into_common(m0_local, Hmax, Wmax),
        mask_view1=embed_into_common(m1_local, Hmax, Wmax),
        mask_view2=embed_into_common(m2_local, Hmax, Wmax),
        origins_view=np.array(origins_view, dtype=object),
        slot_to_specname=np.array(SLOT_TO_SPECNAME, dtype=object),
        slot_tf=np.array(SLOT_TF, dtype=object),
    )

    meta = {
        "split": args.split,
        "N": int(N),
        "Hmax": int(Hmax),
        "Wmax": int(Wmax),
        "C_in": int(c_in),
        "C_out": int(c_out),
        "gap_px": int(gap_px),
        "k": int(k),
        "views": ["view0", "view1", "view2"],
        "in_channels": ["mask"] + [f"x{k}" for k in range(8)] + (["centroid_x", "centroid_y"] if args.include_geometry else []),
        "out_channels": out_names,
        "slots": list(SLOT_TO_SPECNAME.keys()),
    }
    meta_path = f"{out_prefix}_meta_{args.split}.npz"
    np.savez_compressed(meta_path, **{k: np.array(v, dtype=object) for k, v in meta.items()})

    print(f"Saved X tensor:      {out_x}    shape={X_img.shape}")
    print(f"Saved Y tensor:      {out_y}    shape={Y_img.shape}")
    print(f"Saved masks:         {out_mask} shape={out_mask_views.shape}")
    print(f"Saved layout map:    {out_layout}")
    print(f"Saved meta:          {meta_path}")
    if fnixap is not None:
        print(f"Saved fnixap:        {out_f}    shape={fnixap.shape}")


if __name__ == "__main__":
    main()