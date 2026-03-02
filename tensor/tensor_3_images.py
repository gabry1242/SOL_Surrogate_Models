#!/usr/bin/env python3
"""
tensor_3_images_fixed.py

3 views (top/mid/bottom), same (H, W), U-NET friendly.

VIEW "top":    TOP_left + TOP_right
VIEW "mid":    MID_left + MID_center + MID_right
VIEW "bottom": Bottom_center only

Padding + adjacency (applied to BOTH Y and geometry channels; masks updated accordingly):

A) Intra-view horizontal gap fill (k px)
   - top:    TOP_left <-> TOP_right  (fills into the center gap band near edges)
   - mid:    MID_left <-> MID_center, MID_center <-> MID_right
   - bottom: Bottom_center wrap left/right (copies BC right edge into left pad band and vice versa)

B) Inter-view vertical adjacency exchange bands (k rows), bidirectional
   - Left column:  TOP_left bottom band  <->  MID_left top band
   - Right column: TOP_right bottom band <->  MID_right top band
   - Center standard adjacency: MID_center bottom band <-> Bottom_center top band
   - Center wrap adjacency (your request): Bottom_center bottom band <-> MID_center top band

Notes:
- Each view reserves BOTH a top pad band and a bottom pad band of height k.
- Masks are reset per-sample (critical).
- out_prefix may contain subfolders; parents are created.
- Output pack:
    --pack channels (default): (N, 3*C, H, W)
    --pack views:             (N, 3, C, H, W)

Based on your current tensor_3_images.py code structure. :contentReference[oaicite:0]{index=0}
"""

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


# -----------------------
# Slot layout: (slot -> RECT_SPECS name)
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
# Fixed transforms per slot
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
# Transform helpers
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
# Build one strip mapping (pixel->cell indices) using the unroller
# -----------------------
def build_strip(u, spec, tf):
    spec = u.normalize_rect_spec(spec)

    req = [
        lab
        for lab in [spec.get("top"), spec.get("bottom"), spec.get("left"), spec.get("right")]
        if lab is not None
    ]
    target_set = u.pick_component_by_labels(req)

    bottom_cells = None
    if spec.get("bottom") is not None:
        bottom_cells = [c for c in u.LABEL_TO_CELLS[spec["bottom"]] if c in target_set]

    coords, (W, H) = u.unroll_component_adjacency_preserving(
        target_set, bottom_cells=bottom_cells
    )

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
        pix_cell_ix=pix_ix_t,
        pix_cell_iy=pix_iy_t,
    )


# -----------------------
# View layouts
# -----------------------
def compute_common_widths(strip_dicts, gap_px: int):
    TL = strip_dicts["TOP_left"]
    TR = strip_dicts["TOP_right"]
    ML = strip_dicts["MID_left"]
    MR = strip_dicts["MID_right"]
    MC = strip_dicts["MID_center"]
    BC = strip_dicts["Bottom_center"]

    wL = max(TL["W"], ML["W"])
    wC = max(MC["W"], BC["W"])
    wR = max(TR["W"], MR["W"])
    W = int(wL + gap_px + wC + gap_px + wR)
    return int(wL), int(wC), int(wR), int(W)


def compute_common_height(strip_dicts, k: int):
    TL = strip_dicts["TOP_left"]
    TR = strip_dicts["TOP_right"]
    ML = strip_dicts["MID_left"]
    MR = strip_dicts["MID_right"]
    MC = strip_dicts["MID_center"]
    BC = strip_dicts["Bottom_center"]

    hT = max(TL["H"], TR["H"])
    hM = max(ML["H"], MC["H"], MR["H"])
    hB = BC["H"]

    # Reserve top pad k and bottom pad k for every view.
    H = int(k + max(hT, hM, hB) + k)
    return int(hT), int(hM), int(hB), int(H)


def compute_view_origins(strip_dicts, gap_px: int, k: int):
    wL, wC, wR, W = compute_common_widths(strip_dicts, gap_px)
    _, _, _, H = compute_common_height(strip_dicts, k)

    xL = 0
    xC = wL + gap_px
    xR = wL + gap_px + wC + gap_px
    y0 = k

    origins = {"top": {}, "mid": {}, "bottom": {}}

    origins["top"]["TOP_left"] = (xL, y0)
    origins["top"]["TOP_right"] = (xR, y0)

    origins["mid"]["MID_left"] = (xL, y0)
    origins["mid"]["MID_center"] = (xC, y0)
    origins["mid"]["MID_right"] = (xR, y0)

    origins["bottom"]["Bottom_center"] = (xC, y0)

    return origins, W, H, (wL, wC, wR)


def paste_view_mask(strip_dicts, origins_view, W: int, H: int):
    m = np.zeros((H, W), dtype=np.uint8)
    for slot, (u0, v0) in origins_view.items():
        sd = strip_dicts[slot]
        m[v0 : v0 + sd["H"], u0 : u0 + sd["W"]] |= sd["mask"]
    return m


def paste_field_view(strip_dicts, origins_view, field_2d, H: int, W: int):
    out = np.zeros((H, W), dtype=np.float32)
    for slot, (u0, v0) in origins_view.items():
        sd = strip_dicts[slot]
        m = sd["mask"].astype(bool)
        pix_ix = sd["pix_cell_ix"]
        pix_iy = sd["pix_cell_iy"]

        sub = out[v0 : v0 + sd["H"], u0 : u0 + sd["W"]]
        ix = pix_ix[m]
        iy = pix_iy[m]
        sub[m] = field_2d[ix, iy].astype(np.float32)
        out[v0 : v0 + sd["H"], u0 : u0 + sd["W"]] = sub
    return out


def paste_field_view_3d(strip_dicts, origins_view, field_3d, H: int, W: int):
    S = field_3d.shape[-1]
    out = np.zeros((S, H, W), dtype=np.float32)
    for s in range(S):
        out[s] = paste_field_view(strip_dicts, origins_view, field_3d[:, :, s], H, W)
    return out


# -----------------------
# Intra-view gap fill (writes into arr and updates maskv)
# -----------------------
def fill_gap_lr_view(arr, maskv, left_slot, right_slot, strip_dicts, origins_view, k: int):
    H, W = maskv.shape
    (lu0, lv0) = origins_view[left_slot]
    (ru0, rv0) = origins_view[right_slot]
    lW, lH = int(strip_dicts[left_slot]["W"]), int(strip_dicts[left_slot]["H"])
    rW, rH = int(strip_dicts[right_slot]["W"]), int(strip_dicts[right_slot]["H"])

    left_end = lu0 + lW
    right_start = ru0
    gap = right_start - left_end
    if gap <= 0:
        return

    k1 = min(k, gap)
    k2 = min(k, gap)

    y0 = max(lv0, rv0)
    y1 = min(lv0 + lH, rv0 + rH)
    if y1 <= y0:
        return

    lmask = np.zeros((H, W), dtype=np.uint8)
    rmask = np.zeros((H, W), dtype=np.uint8)
    lmask[lv0 : lv0 + lH, lu0 : lu0 + lW] = strip_dicts[left_slot]["mask"].astype(np.uint8)
    rmask[rv0 : rv0 + rH, ru0 : ru0 + rW] = strip_dicts[right_slot]["mask"].astype(np.uint8)

    # right of LEFT: copy from RIGHT left edge
    dst_x0 = left_end
    dst_x1 = left_end + k1
    src_x0 = ru0
    src_x1 = ru0 + k1
    src_m = rmask[y0:y1, src_x0:src_x1]

    if arr.ndim == 2:
        arr[y0:y1, dst_x0:dst_x1] = arr[y0:y1, src_x0:src_x1] * src_m
    else:
        arr[:, y0:y1, dst_x0:dst_x1] = arr[:, y0:y1, src_x0:src_x1] * src_m[None, :, :]
    maskv[y0:y1, dst_x0:dst_x1] = np.maximum(maskv[y0:y1, dst_x0:dst_x1], src_m)

    # left of RIGHT: copy from LEFT right edge
    dst_x0b = right_start - k2
    dst_x1b = right_start
    src_x0b = left_end - k2
    src_x1b = left_end
    src_m2 = lmask[y0:y1, src_x0b:src_x1b]

    if arr.ndim == 2:
        arr[y0:y1, dst_x0b:dst_x1b] = arr[y0:y1, src_x0b:src_x1b] * src_m2
    else:
        arr[:, y0:y1, dst_x0b:dst_x1b] = arr[:, y0:y1, src_x0b:src_x1b] * src_m2[None, :, :]
    maskv[y0:y1, dst_x0b:dst_x1b] = np.maximum(maskv[y0:y1, dst_x0b:dst_x1b], src_m2)


def wrap_pad_lr_for_center_block(arr, maskv, u0: int, v0: int, Wc: int, Hc: int, k: int):
    if k <= 0:
        return
    H, W = maskv.shape
    y0 = v0
    y1 = v0 + Hc
    if y1 <= y0:
        return

    lp_x0 = max(0, u0 - k)
    lp_x1 = u0
    rp_x0 = u0 + Wc
    rp_x1 = min(W, u0 + Wc + k)

    srcR_x0 = u0 + Wc - (lp_x1 - lp_x0)
    srcR_x1 = u0 + Wc
    srcL_x0 = u0
    srcL_x1 = u0 + (rp_x1 - rp_x0)

    if lp_x1 > lp_x0:
        src_m = maskv[y0:y1, srcR_x0:srcR_x1]
        if arr.ndim == 2:
            arr[y0:y1, lp_x0:lp_x1] = arr[y0:y1, srcR_x0:srcR_x1] * src_m
        else:
            arr[:, y0:y1, lp_x0:lp_x1] = arr[:, y0:y1, srcR_x0:srcR_x1] * src_m[None, :, :]
        maskv[y0:y1, lp_x0:lp_x1] = np.maximum(maskv[y0:y1, lp_x0:lp_x1], src_m)

    if rp_x1 > rp_x0:
        src_m2 = maskv[y0:y1, srcL_x0:srcL_x1]
        if arr.ndim == 2:
            arr[y0:y1, rp_x0:rp_x1] = arr[y0:y1, srcL_x0:srcL_x1] * src_m2
        else:
            arr[:, y0:y1, rp_x0:rp_x1] = arr[:, y0:y1, srcL_x0:srcL_x1] * src_m2[None, :, :]
        maskv[y0:y1, rp_x0:rp_x1] = np.maximum(maskv[y0:y1, rp_x0:rp_x1], src_m2)


# -----------------------
# Inter-view band exchange (bidirectional), updates destination mask
# -----------------------
def copy_top_band_from_bottom_band(dst_arr, dst_mask, src_arr, src_mask, x0: int, x1: int, k: int):
    # src bottom [H-k:H) -> dst top [0:k)
    if k <= 0:
        return
    H, W = dst_mask.shape
    assert src_mask.shape == (H, W)

    src_y0, src_y1 = H - k, H
    dst_y0, dst_y1 = 0, k

    src_m = src_mask[src_y0:src_y1, x0:x1]
    if dst_arr.ndim == 2:
        dst_arr[dst_y0:dst_y1, x0:x1] = src_arr[src_y0:src_y1, x0:x1] * src_m
    else:
        dst_arr[:, dst_y0:dst_y1, x0:x1] = src_arr[:, src_y0:src_y1, x0:x1] * src_m[None, :, :]
    dst_mask[dst_y0:dst_y1, x0:x1] = np.maximum(dst_mask[dst_y0:dst_y1, x0:x1], src_m)


def copy_bottom_band_from_top_band(dst_arr, dst_mask, src_arr, src_mask, x0: int, x1: int, k: int):
    # src top [0:k) -> dst bottom [H-k:H)
    if k <= 0:
        return
    H, W = dst_mask.shape
    assert src_mask.shape == (H, W)

    src_y0, src_y1 = 0, k
    dst_y0, dst_y1 = H - k, H

    src_m = src_mask[src_y0:src_y1, x0:x1]
    if dst_arr.ndim == 2:
        dst_arr[dst_y0:dst_y1, x0:x1] = src_arr[src_y0:src_y1, x0:x1] * src_m
    else:
        dst_arr[:, dst_y0:dst_y1, x0:x1] = src_arr[:, src_y0:src_y1, x0:x1] * src_m[None, :, :]
    dst_mask[dst_y0:dst_y1, x0:x1] = np.maximum(dst_mask[dst_y0:dst_y1, x0:x1], src_m)


# -----------------------
# Geometry channels (optional)
# -----------------------
def build_geometry_channels_for_view(strip_dicts, origins_view, H: int, W: int, geom_dir: str):
    crx = np.load(os.path.join(geom_dir, "crx.npy"))
    cry = np.load(os.path.join(geom_dir, "cry.npy"))
    cx = crx.mean(axis=2)
    cy = cry.mean(axis=2)

    gx = paste_field_view(strip_dicts, origins_view, cx, H, W)
    gy = paste_field_view(strip_dicts, origins_view, cy, H, W)
    geom = np.stack([gx, gy], axis=0).astype(np.float32)
    return geom


# -----------------------
# Output target selection
# -----------------------
def parse_out_targets(s: str):
    if s is None or s.strip() == "":
        return ["te", "ti", "na", "ua"]
    items = [t.strip().lower() for t in s.split(",") if t.strip()]
    allowed = {"te", "ti", "na", "ua"}
    bad = [t for t in items if t not in allowed]
    if bad:
        raise ValueError(f"Unknown out_targets: {bad}. Allowed: {sorted(allowed)}")
    return items


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def save_layout_map_3views(path_npz: Path, strip_dicts, origins_all, W: int, H: int, gap_px: int, k: int):
    out = {
        "W": np.int32(W),
        "H": np.int32(H),
        "gap_px": np.int32(gap_px),
        "k": np.int32(k),
    }
    for vname, origins_view in origins_all.items():
        for slot, (u0, v0) in origins_view.items():
            sd = strip_dicts[slot]
            out[f"{vname}__{slot}__u0"] = np.int32(u0)
            out[f"{vname}__{slot}__v0"] = np.int32(v0)
            out[f"{vname}__{slot}__W"] = np.int32(sd["W"])
            out[f"{vname}__{slot}__H"] = np.int32(sd["H"])
            out[f"{vname}__{slot}__tf"] = np.array(sd["tf"])
            out[f"{vname}__{slot}__mask"] = sd["mask"].astype(np.uint8)
            out[f"{vname}__{slot}__pix_ix"] = sd["pix_cell_ix"].astype(np.int32)
            out[f"{vname}__{slot}__pix_iy"] = sd["pix_cell_iy"].astype(np.int32)

    _ensure_parent(path_npz)
    np.savez_compressed(str(path_npz), **out)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--unroller_module", default="unrolled_strip_clockwise_adjpreserve")
    ap.add_argument("--gap_px", type=int, default=10)
    ap.add_argument("--geom_dir", default="geometry")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_prefix", default="global3")
    ap.add_argument("--out_root", type=str, default=None)

    ap.add_argument("--include_geometry", action="store_true")
    ap.add_argument("--out_targets", default="te,ti,na,ua")
    ap.add_argument("--include_fnixap", action="store_true")

    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit_n", type=int, default=0)

    ap.add_argument("--pack", choices=["channels", "views"], default="channels")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split

    out_root = Path(args.out_root).resolve() if args.out_root is not None else data_root
    out_dir = out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- build strip dicts ----
    u = importlib.import_module(args.unroller_module)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}
    missing = [nm for nm in SLOT_TO_SPECNAME.values() if nm not in name_to_spec]
    if missing:
        raise RuntimeError(f"Missing RECT_SPECS names: {missing}")

    strip_dicts = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=SLOT_TF[slot])

    k = int(args.k)
    gap_px = int(args.gap_px)

    origins_all, W, H, (wL, wC, wR) = compute_view_origins(strip_dicts, gap_px=gap_px, k=k)

    o_top = origins_all["top"]
    o_mid = origins_all["mid"]
    o_bot = origins_all["bottom"]

    # base masks (static placement only)
    mask_top_base = paste_view_mask(strip_dicts, o_top, W, H)
    mask_mid_base = paste_view_mask(strip_dicts, o_mid, W, H)
    mask_bot_base = paste_view_mask(strip_dicts, o_bot, W, H)

    # layout map (static)
    layout_path = out_dir / f"{args.out_prefix}_layout_3views.npz"
    save_layout_map_3views(layout_path, strip_dicts, origins_all, W=W, H=H, gap_px=gap_px, k=k)
    print(f"Saved layout map: {layout_path}  (H={H}, W={W})")

    # save base masks (static)
    base_mask_path = out_dir / f"{args.out_prefix}_mask_views_base.npy"
    _ensure_parent(base_mask_path)
    np.save(str(base_mask_path), np.stack([mask_top_base, mask_mid_base, mask_bot_base], axis=0).astype(np.uint8))

    # ---- load data ----
    X = np.load(split_dir / "X_tmp.npy")
    te = np.load(split_dir / "te_tmp.npy")
    ti = np.load(split_dir / "ti_tmp.npy")
    na = np.load(split_dir / "na_tmp.npy")
    ua = np.load(split_dir / "ua_tmp.npy")

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

    # ---- output channels ----
    targets = parse_out_targets(args.out_targets)
    out_parts: List[Tuple[str, int]] = []
    if "te" in targets:
        out_parts.append(("te", 1))
    if "ti" in targets:
        out_parts.append(("ti", 1))
    if "na" in targets:
        out_parts.append(("na", 10))
    if "ua" in targets:
        out_parts.append(("ua", 10))
    c_out = int(sum(ch for _, ch in out_parts))

    c_in = 1 + 8 + (2 if args.include_geometry else 0)

    if args.pack == "views":
        X_img = np.zeros((N, 3, c_in, H, W), dtype=np.float32)
        Y_img = np.zeros((N, 3, c_out, H, W), dtype=np.float32)
    else:
        X_img = np.zeros((N, 3 * c_in, H, W), dtype=np.float32)
        Y_img = np.zeros((N, 3 * c_out, H, W), dtype=np.float32)

    # column spans for vertical band exchanges
    xL0, xL1 = 0, wL
    xC0, xC1 = wL + gap_px, wL + gap_px + wC
    xR0 = wL + gap_px + wC + gap_px
    xR1 = xR0 + wR

    # BC bbox for wrap padding
    bc_u0, bc_v0 = o_bot["Bottom_center"]
    bc_W, bc_H = int(strip_dicts["Bottom_center"]["W"]), int(strip_dicts["Bottom_center"]["H"])

    J_PER_EV = 1.602176634e-19

    # ---- per-sample build ----
    for i in range(N):
        # reset filled masks per-sample (critical)
        mask_top = mask_top_base.copy()
        mask_mid = mask_mid_base.copy()
        mask_bot = mask_bot_base.copy()

        # Y buffers
        Y_top = np.zeros((c_out, H, W), dtype=np.float32)
        Y_mid = np.zeros((c_out, H, W), dtype=np.float32)
        Y_bot = np.zeros((c_out, H, W), dtype=np.float32)

        ch0 = 0
        for name, ch in out_parts:
            if name in ("te", "ti"):
                field = (te[i] if name == "te" else ti[i]) / J_PER_EV
                Y_top[ch0] = paste_field_view(strip_dicts, o_top, field, H, W)
                Y_mid[ch0] = paste_field_view(strip_dicts, o_mid, field, H, W)
                Y_bot[ch0] = paste_field_view(strip_dicts, o_bot, field, H, W)
                ch0 += 1
            elif name == "na":
                Y_top[ch0:ch0+10] = paste_field_view_3d(strip_dicts, o_top, na[i], H, W)
                Y_mid[ch0:ch0+10] = paste_field_view_3d(strip_dicts, o_mid, na[i], H, W)
                Y_bot[ch0:ch0+10] = paste_field_view_3d(strip_dicts, o_bot, na[i], H, W)
                ch0 += 10
            elif name == "ua":
                Y_top[ch0:ch0+10] = paste_field_view_3d(strip_dicts, o_top, ua[i], H, W)
                Y_mid[ch0:ch0+10] = paste_field_view_3d(strip_dicts, o_mid, ua[i], H, W)
                Y_bot[ch0:ch0+10] = paste_field_view_3d(strip_dicts, o_bot, ua[i], H, W)
                ch0 += 10
            else:
                raise RuntimeError(name)

        # ---------- intra-view fills ----------
        fill_gap_lr_view(Y_top, mask_top, "TOP_left", "TOP_right", strip_dicts, o_top, k)
        fill_gap_lr_view(Y_mid, mask_mid, "MID_left", "MID_center", strip_dicts, o_mid, k)
        fill_gap_lr_view(Y_mid, mask_mid, "MID_center", "MID_right", strip_dicts, o_mid, k)
        wrap_pad_lr_for_center_block(Y_bot, mask_bot, bc_u0, bc_v0, bc_W, bc_H, k)

        # ---------- inter-view vertical adjacency (bidirectional) ----------
        # TL bottom <-> ML top
        copy_top_band_from_bottom_band(Y_mid, mask_mid, Y_top, mask_top, xL0, xL1, k)
        copy_bottom_band_from_top_band(Y_top, mask_top, Y_mid, mask_mid, xL0, xL1, k)

        # TR bottom <-> MR top
        copy_top_band_from_bottom_band(Y_mid, mask_mid, Y_top, mask_top, xR0, xR1, k)
        copy_bottom_band_from_top_band(Y_top, mask_top, Y_mid, mask_mid, xR0, xR1, k)

        # Standard center adjacency: MC bottom <-> BC top
        copy_top_band_from_bottom_band(Y_bot, mask_bot, Y_mid, mask_mid, xC0, xC1, k)
        copy_bottom_band_from_top_band(Y_mid, mask_mid, Y_bot, mask_bot, xC0, xC1, k)

        # Wrap center adjacency (your request): BC bottom <-> MC top
        copy_top_band_from_bottom_band(Y_mid, mask_mid, Y_bot, mask_bot, xC0, xC1, k)
        copy_bottom_band_from_top_band(Y_bot, mask_bot, Y_mid, mask_mid, xC0, xC1, k)

        # ---------- build X ----------
        X_top = np.zeros((c_in, H, W), dtype=np.float32)
        X_mid = np.zeros((c_in, H, W), dtype=np.float32)
        X_bot = np.zeros((c_in, H, W), dtype=np.float32)

        mt = mask_top.astype(np.float32)
        mm = mask_mid.astype(np.float32)
        mb = mask_bot.astype(np.float32)

        X_top[0] = mt
        X_mid[0] = mm
        X_bot[0] = mb

        for kk in range(8):
            val = float(X[i, kk])
            X_top[1 + kk] = val * mt
            X_mid[1 + kk] = val * mm
            X_bot[1 + kk] = val * mb

        if args.include_geometry:
            geom_top = build_geometry_channels_for_view(strip_dicts, o_top, H, W, geom_dir=args.geom_dir)
            geom_mid = build_geometry_channels_for_view(strip_dicts, o_mid, H, W, geom_dir=args.geom_dir)
            geom_bot = build_geometry_channels_for_view(strip_dicts, o_bot, H, W, geom_dir=args.geom_dir)

            # intra-view fills on geometry
            fill_gap_lr_view(geom_top, mask_top, "TOP_left", "TOP_right", strip_dicts, o_top, k)
            fill_gap_lr_view(geom_mid, mask_mid, "MID_left", "MID_center", strip_dicts, o_mid, k)
            fill_gap_lr_view(geom_mid, mask_mid, "MID_center", "MID_right", strip_dicts, o_mid, k)
            wrap_pad_lr_for_center_block(geom_bot, mask_bot, bc_u0, bc_v0, bc_W, bc_H, k)

            # vertical exchanges on geometry
            copy_top_band_from_bottom_band(geom_mid, mask_mid, geom_top, mask_top, xL0, xL1, k)
            copy_bottom_band_from_top_band(geom_top, mask_top, geom_mid, mask_mid, xL0, xL1, k)

            copy_top_band_from_bottom_band(geom_mid, mask_mid, geom_top, mask_top, xR0, xR1, k)
            copy_bottom_band_from_top_band(geom_top, mask_top, geom_mid, mask_mid, xR0, xR1, k)

            copy_top_band_from_bottom_band(geom_bot, mask_bot, geom_mid, mask_mid, xC0, xC1, k)
            copy_bottom_band_from_top_band(geom_mid, mask_mid, geom_bot, mask_bot, xC0, xC1, k)

            copy_top_band_from_bottom_band(geom_mid, mask_mid, geom_bot, mask_bot, xC0, xC1, k)
            copy_bottom_band_from_top_band(geom_bot, mask_bot, geom_mid, mask_mid, xC0, xC1, k)

            X_top[1 + 8:1 + 8 + 2] = geom_top * mt[None, :, :]
            X_mid[1 + 8:1 + 8 + 2] = geom_mid * mm[None, :, :]
            X_bot[1 + 8:1 + 8 + 2] = geom_bot * mb[None, :, :]

        # ---------- store ----------
        if args.pack == "views":
            X_img[i, 0] = X_top
            X_img[i, 1] = X_mid
            X_img[i, 2] = X_bot

            Y_img[i, 0] = Y_top
            Y_img[i, 1] = Y_mid
            Y_img[i, 2] = Y_bot
        else:
            X_img[i, 0*c_in:1*c_in] = X_top
            X_img[i, 1*c_in:2*c_in] = X_mid
            X_img[i, 2*c_in:3*c_in] = X_bot

            Y_img[i, 0*c_out:1*c_out] = Y_top
            Y_img[i, 1*c_out:2*c_out] = Y_mid
            Y_img[i, 2*c_out:3*c_out] = Y_bot

    # ---- save outputs ----
    x_path = out_dir / f"{args.out_prefix}_X_img_{args.split}.npy"
    y_path = out_dir / f"{args.out_prefix}_Y_img_{args.split}.npy"
    _ensure_parent(x_path)
    _ensure_parent(y_path)
    np.save(str(x_path), X_img)
    np.save(str(y_path), Y_img)

    # save static base masks already done; also save one representative "filled mask" by recomputing from X[mask]
    # to avoid saving last-sample artifacts.
    filled_mask_path = out_dir / f"{args.out_prefix}_mask_views_fromX_{args.split}.npy"
    _ensure_parent(filled_mask_path)
    if args.pack == "views":
        masks_fromX = np.stack(
            [
                (X_img[0, 0, 0] > 0).astype(np.uint8),
                (X_img[0, 1, 0] > 0).astype(np.uint8),
                (X_img[0, 2, 0] > 0).astype(np.uint8),
            ],
            axis=0,
        )
    else:
        masks_fromX = np.stack(
            [
                (X_img[0, 0*c_in + 0] > 0).astype(np.uint8),
                (X_img[0, 1*c_in + 0] > 0).astype(np.uint8),
                (X_img[0, 2*c_in + 0] > 0).astype(np.uint8),
            ],
            axis=0,
        )
    np.save(str(filled_mask_path), masks_fromX)

    if fnixap is not None:
        f_path = out_dir / f"{args.out_prefix}_fnixap_{args.split}.npy"
        _ensure_parent(f_path)
        np.save(str(f_path), fnixap)

    print(f"Saved X: {x_path}  shape={X_img.shape}")
    print(f"Saved Y: {y_path}  shape={Y_img.shape}")
    print(f"Saved base masks: {base_mask_path}")
    print(f"Saved filled masks (from sample 0 X[mask]): {filled_mask_path}")


if __name__ == "__main__":
    main()