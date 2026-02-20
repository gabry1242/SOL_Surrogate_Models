#!/usr/bin/env python3
"""
build_global_tensor.py (filled_gap_3)

Creates the same global tensors as the baseline build_global_tensor.py,
but additionally fills a thin band of previously-masked pixels inside the gaps
between adjacent blocks.

Gap filling rule (k pixels, default k=3):
- For every adjacency between two placed blocks (left-right or top-bottom),
  copy the nearest k columns/rows from the opposite block into the gap,
  and set the global mask to 1 at those filled locations.

This keeps your global canvas size unchanged; only the mask and the tensors
are modified.

Outputs (same naming as baseline):
- global_layout_map.npz
- <out_prefix>_X_img_<split>.npy
- <out_prefix>_Y_img_<split>.npy
- optionally fnixap_<split>.npy

Baseline file this is derived from: build_global_tensor.py. fileciteturn1file13
"""

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, Tuple

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
    "TOP_left": "R180",
    "TOP_right": "R180",
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
        pix_cell_ix=pix_ix_t,
        pix_cell_iy=pix_iy_t,
    )

# -----------------------
# Fixed layout (same as baseline)
# -----------------------
def compute_layout(strip_dicts, gap_px: int):
    TL = strip_dicts["TOP_left"]
    TR = strip_dicts["TOP_right"]
    ML = strip_dicts["MID_left"]
    MC = strip_dicts["MID_center"]
    MR = strip_dicts["MID_right"]
    BC = strip_dicts["Bottom_center"]

    wL = max(TL["W"], ML["W"])
    wC = max(MC["W"], BC["W"])
    wR = max(TR["W"], MR["W"])

    hT = max(TL["H"], TR["H"])
    hM = max(ML["H"], MC["H"], MR["H"])
    hB = BC["H"]

    Wg = wL + gap_px + wC + gap_px + wR
    Hg = hT + gap_px + hM + gap_px + hB

    origins = {}
    origins["TOP_left"] = (0, 0)
    origins["TOP_right"] = (wL + gap_px + wC + gap_px, 0)

    origins["MID_left"] = (0, hT + gap_px)
    origins["MID_center"] = (wL + gap_px, hT + gap_px)
    origins["MID_right"] = (wL + gap_px + wC + gap_px, hT + gap_px)

    origins["Bottom_center"] = (wL + gap_px, hT + gap_px + hM + gap_px)
    return origins, int(Wg), int(Hg)


def paste_global_mask(strip_dicts, origins, Wg, Hg):
    maskg = np.zeros((Hg, Wg), dtype=np.uint8)
    for slot, sd in strip_dicts.items():
        u0, v0 = origins[slot]
        H, W = sd["H"], sd["W"]
        maskg[v0 : v0 + H, u0 : u0 + W] |= sd["mask"]
    return maskg

# -----------------------
# Save layout map (reusable)
# -----------------------
def save_layout_map(path_npz, strip_dicts, origins, Wg, Hg, maskg, gap_px):
    out = {
        "Wg": np.int32(Wg),
        "Hg": np.int32(Hg),
        "gap_px": np.int32(gap_px),
        "maskg": maskg.astype(np.uint8),
    }
    for slot, sd in strip_dicts.items():
        u0, v0 = origins[slot]
        out[f"{slot}__u0"] = np.int32(u0)
        out[f"{slot}__v0"] = np.int32(v0)
        out[f"{slot}__mask"] = sd["mask"].astype(np.uint8)
        out[f"{slot}__pix_ix"] = sd["pix_cell_ix"].astype(np.int32)
        out[f"{slot}__pix_iy"] = sd["pix_cell_iy"].astype(np.int32)
        out[f"{slot}__tf"] = np.array(sd["tf"])
        out[f"{slot}__W"] = np.int32(sd["W"])
        out[f"{slot}__H"] = np.int32(sd["H"])

    np.savez_compressed(path_npz, **out)

# -----------------------
# Paste fields into global canvas
# -----------------------
def paste_field_global(maskg, slot_dicts, origins, field_2d):
    """field_2d: (104,50) for one simulation and one channel."""
    Hg, Wg = maskg.shape
    out = np.zeros((Hg, Wg), dtype=np.float32)

    for slot, sd in slot_dicts.items():
        u0, v0 = origins[slot]
        m = sd["mask"].astype(bool)
        pix_ix = sd["pix_cell_ix"]
        pix_iy = sd["pix_cell_iy"]

        sub = out[v0 : v0 + sd["H"], u0 : u0 + sd["W"]]
        ix = pix_ix[m]
        iy = pix_iy[m]
        sub[m] = field_2d[ix, iy].astype(np.float32)
        out[v0 : v0 + sd["H"], u0 : u0 + sd["W"]] = sub

    return out


def paste_field_global_3d(maskg, slot_dicts, origins, field_3d):
    """field_3d: (104,50,S) for one simulation."""
    Hg, Wg = maskg.shape
    S = field_3d.shape[-1]
    out = np.zeros((S, Hg, Wg), dtype=np.float32)
    for s in range(S):
        out[s] = paste_field_global(maskg, slot_dicts, origins, field_3d[:, :, s])
    return out

# -----------------------
# Geometry channels (optional)
# -----------------------
def build_geometry_channels(maskg, slot_dicts, origins, geom_dir: str):
    crx = np.load(os.path.join(geom_dir, "crx.npy"))
    cry = np.load(os.path.join(geom_dir, "cry.npy"))

    cx = crx.mean(axis=2)
    cy = cry.mean(axis=2)

    gx = paste_field_global(maskg, slot_dicts, origins, cx)
    gy = paste_field_global(maskg, slot_dicts, origins, cy)
    geom = np.stack([gx, gy], axis=0).astype(np.float32)

    m = maskg.astype(np.float32)
    geom *= m[None, :, :]
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

# -----------------------
# Gap filling
# -----------------------
def _slot_bbox(slot: str, strip_dicts: Dict, origins: Dict[str, Tuple[int, int]]):
    u0, v0 = origins[slot]
    W = int(strip_dicts[slot]["W"])
    H = int(strip_dicts[slot]["H"])
    return u0, v0, W, H


def _slot_mask_global(slot: str, strip_dicts: Dict, origins: Dict[str, Tuple[int, int]], Hg: int, Wg: int):
    u0, v0, W, H = _slot_bbox(slot, strip_dicts, origins)
    g = np.zeros((Hg, Wg), dtype=np.uint8)
    g[v0 : v0 + H, u0 : u0 + W] = strip_dicts[slot]["mask"].astype(np.uint8)
    return g


def fill_gap_lr(
    arr: np.ndarray,
    maskg: np.ndarray,
    left_slot: str,
    right_slot: str,
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    k: int,
):
    """Fill k columns inside the gap between left_slot and right_slot.

    Writes into arr in-place and updates maskg in-place.

    arr shape can be (Hg,Wg) or (C,Hg,Wg).
    """
    Hg, Wg = maskg.shape
    lu0, lv0, lW, lH = _slot_bbox(left_slot, strip_dicts, origins)
    ru0, rv0, rW, rH = _slot_bbox(right_slot, strip_dicts, origins)

    left_end = lu0 + lW
    right_start = ru0
    gap = right_start - left_end
    if gap <= 0:
        return

    k1 = min(k, gap)
    k2 = min(k, gap)

    # y overlap based on placed rectangles
    y0 = max(lv0, rv0)
    y1 = min(lv0 + lH, rv0 + rH)
    if y1 <= y0:
        return

    lmask_g = _slot_mask_global(left_slot, strip_dicts, origins, Hg, Wg)
    rmask_g = _slot_mask_global(right_slot, strip_dicts, origins, Hg, Wg)

    # Part A: immediately to the right of LEFT block, copy from RIGHT block's left edge
    dst_x0 = left_end
    dst_x1 = left_end + k1
    src_x0 = ru0
    src_x1 = ru0 + k1

    src_m = rmask_g[y0:y1, src_x0:src_x1]
    if arr.ndim == 2:
        arr[y0:y1, dst_x0:dst_x1] = arr[y0:y1, src_x0:src_x1] * src_m
    else:
        arr[:, y0:y1, dst_x0:dst_x1] = arr[:, y0:y1, src_x0:src_x1] * src_m[None, :, :]
    maskg[y0:y1, dst_x0:dst_x1] = np.maximum(maskg[y0:y1, dst_x0:dst_x1], src_m)

    # Part B: immediately to the left of RIGHT block, copy from LEFT block's right edge
    dst_x0b = right_start - k2
    dst_x1b = right_start
    src_x0b = left_end - k2
    src_x1b = left_end

    src_m2 = lmask_g[y0:y1, src_x0b:src_x1b]
    if arr.ndim == 2:
        arr[y0:y1, dst_x0b:dst_x1b] = arr[y0:y1, src_x0b:src_x1b] * src_m2
    else:
        arr[:, y0:y1, dst_x0b:dst_x1b] = arr[:, y0:y1, src_x0b:src_x1b] * src_m2[None, :, :]
    maskg[y0:y1, dst_x0b:dst_x1b] = np.maximum(maskg[y0:y1, dst_x0b:dst_x1b], src_m2)


def fill_gap_tb(
    arr: np.ndarray,
    maskg: np.ndarray,
    top_slot: str,
    bottom_slot: str,
    strip_dicts: Dict,
    origins: Dict[str, Tuple[int, int]],
    k: int,
):
    """Fill k rows inside the gap between top_slot and bottom_slot.

    Writes into arr in-place and updates maskg in-place.

    arr shape can be (Hg,Wg) or (C,Hg,Wg).
    """
    Hg, Wg = maskg.shape
    tu0, tv0, tW, tH = _slot_bbox(top_slot, strip_dicts, origins)
    bu0, bv0, bW, bH = _slot_bbox(bottom_slot, strip_dicts, origins)

    top_end = tv0 + tH
    bottom_start = bv0
    gap = bottom_start - top_end
    if gap <= 0:
        return

    k1 = min(k, gap)
    k2 = min(k, gap)

    # x overlap based on placed rectangles
    x0 = max(tu0, bu0)
    x1 = min(tu0 + tW, bu0 + bW)
    if x1 <= x0:
        return

    tmask_g = _slot_mask_global(top_slot, strip_dicts, origins, Hg, Wg)
    bmask_g = _slot_mask_global(bottom_slot, strip_dicts, origins, Hg, Wg)

    # Part A: immediately below TOP block, copy from BOTTOM block's top edge
    dst_y0 = top_end
    dst_y1 = top_end + k1
    src_y0 = bv0
    src_y1 = bv0 + k1

    src_m = bmask_g[src_y0:src_y1, x0:x1]
    if arr.ndim == 2:
        arr[dst_y0:dst_y1, x0:x1] = arr[src_y0:src_y1, x0:x1] * src_m
    else:
        arr[:, dst_y0:dst_y1, x0:x1] = arr[:, src_y0:src_y1, x0:x1] * src_m[None, :, :]
    maskg[dst_y0:dst_y1, x0:x1] = np.maximum(maskg[dst_y0:dst_y1, x0:x1], src_m)

    # Part B: immediately above BOTTOM block, copy from TOP block's bottom edge
    dst_y0b = bottom_start - k2
    dst_y1b = bottom_start
    src_y0b = top_end - k2
    src_y1b = top_end

    src_m2 = tmask_g[src_y0b:src_y1b, x0:x1]
    if arr.ndim == 2:
        arr[dst_y0b:dst_y1b, x0:x1] = arr[src_y0b:src_y1b, x0:x1] * src_m2
    else:
        arr[:, dst_y0b:dst_y1b, x0:x1] = arr[:, src_y0b:src_y1b, x0:x1] * src_m2[None, :, :]
    maskg[dst_y0b:dst_y1b, x0:x1] = np.maximum(maskg[dst_y0b:dst_y1b, x0:x1], src_m2)


def apply_gap_filling_to_global_channels(global_ch: np.ndarray, maskg: np.ndarray, strip_dicts, origins, k: int):
    """Apply gap filling to a (C,Hg,Wg) array in-place."""
    # Horizontal adjacencies
    fill_gap_lr(global_ch, maskg, "MID_left", "MID_center", strip_dicts, origins, k)
    fill_gap_lr(global_ch, maskg, "MID_center", "MID_right", strip_dicts, origins, k)

    # The top row has a large empty center; fill k cols near both TL and TR to mimic continuity
    fill_gap_lr(global_ch, maskg, "TOP_left", "TOP_right", strip_dicts, origins, k)

    # Vertical adjacencies
    fill_gap_tb(global_ch, maskg, "TOP_left", "MID_left", strip_dicts, origins, k)
    fill_gap_tb(global_ch, maskg, "TOP_right", "MID_right", strip_dicts, origins, k)
    fill_gap_tb(global_ch, maskg, "MID_center", "Bottom_center", strip_dicts, origins, k)


def apply_gap_filling_to_global_2d(global_2d: np.ndarray, maskg: np.ndarray, strip_dicts, origins, k: int):
    """Apply gap filling to a (Hg,Wg) array in-place."""
    apply_gap_filling_to_global_channels(global_2d[None, :, :], maskg, strip_dicts, origins, k)

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
    ap.add_argument("--out_prefix", default="global")
    ap.add_argument("--layout_map", default="global_layout_map.npz")
    ap.add_argument("--out_root", type=str, default=None)

    ap.add_argument("--include_geometry", action="store_true", help="Add 2 input channels: centroid_x, centroid_y.")
    ap.add_argument("--out_targets", default="te,ti,na,ua", help="Comma-separated subset among: te,ti,na,ua")
    ap.add_argument("--include_fnixap", action="store_true", help="Also save fnixap_<split>.npy")

    ap.add_argument("--fill_k", type=int, default=3, help="Number of rows/cols to fill inside gaps.")
    ap.add_argument("--limit_n", type=int, default=0, help="If >0, process only first N simulations (debug).")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root) if args.out_root is not None else data_root
    split_dir = data_root / args.split

    out_root = Path(args.out_root).resolve() if args.out_root is not None else data_root
    out_dir = out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    # -------- build layout mapping (constant) --------
    u = importlib.import_module(args.unroller_module)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}

    missing = [nm for nm in SLOT_TO_SPECNAME.values() if nm not in name_to_spec]
    if missing:
        raise RuntimeError(f"Missing RECT_SPECS names: {missing}")

    strip_dicts = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        tf = SLOT_TF[slot]
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=tf)

    origins, Wg, Hg = compute_layout(strip_dicts, gap_px=args.gap_px)
    maskg = paste_global_mask(strip_dicts, origins, Wg, Hg)

    save_layout_map(args.layout_map, strip_dicts, origins, Wg, Hg, maskg, args.gap_px)
    print(f"Saved layout map: {args.layout_map}  (Hg={Hg}, Wg={Wg})")

    # -------- load split data --------
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

    # -------- build (optionally filled) global mask --------
    maskg_filled = maskg.copy()

    # We will build Y first using the original mask placement, then fill gaps on the pasted global fields.
    # (Filling needs source values, so it must happen after pasting.)

    # -------- build output tensor Y_img --------
    targets = parse_out_targets(args.out_targets)
    out_parts = []
    out_names = []
    if "te" in targets:
        out_parts.append(("te", 1))
    if "ti" in targets:
        out_parts.append(("ti", 1))
    if "na" in targets:
        out_parts.append(("na", 10))
    if "ua" in targets:
        out_parts.append(("ua", 10))

    c_out = int(sum(ch for _, ch in out_parts))
    Y_img = np.zeros((N, c_out, Hg, Wg), dtype=np.float32)

    J_PER_EV = 1.602176634e-19

    ch0 = 0
    for name, ch in out_parts:
        if name in ("te", "ti"):
            for i in range(N):
                field = te[i] if name == "te" else ti[i]
                field = field / J_PER_EV
                Y_img[i, ch0, :, :] = paste_field_global(maskg, strip_dicts, origins, field)

                # fill gaps for this channel
                apply_gap_filling_to_global_2d(Y_img[i, ch0, :, :], maskg_filled, strip_dicts, origins, int(args.fill_k))

            out_names.append(name)
            ch0 += 1

        elif name == "na":
            for i in range(N):
                g = paste_field_global_3d(maskg, strip_dicts, origins, na[i])  # (10,Hg,Wg)
                apply_gap_filling_to_global_channels(g, maskg_filled, strip_dicts, origins, int(args.fill_k))
                Y_img[i, ch0 : ch0 + 10, :, :] = g
            out_names.append(name)
            ch0 += 10

        elif name == "ua":
            for i in range(N):
                g = paste_field_global_3d(maskg, strip_dicts, origins, ua[i])  # (10,Hg,Wg)
                apply_gap_filling_to_global_channels(g, maskg_filled, strip_dicts, origins, int(args.fill_k))
                Y_img[i, ch0 : ch0 + 10, :, :] = g
            out_names.append(name)
            ch0 += 10

        else:
            raise RuntimeError(name)

    # -------- build input tensor X_img --------
    c_in = 1 + 8 + (2 if args.include_geometry else 0)
    X_img = np.zeros((N, c_in, Hg, Wg), dtype=np.float32)

    m = maskg_filled.astype(np.float32)
    X_img[:, 0, :, :] = m[None, :, :]

    # broadcast scalars with filled mask
    for k in range(8):
        X_img[:, 1 + k, :, :] = (X[:, k].astype(np.float32)[:, None, None]) * m[None, :, :]

    if args.include_geometry:
        # build geom on baseline mask, then fill gaps consistently (copying from neighbors)
        geom = build_geometry_channels(maskg, strip_dicts, origins, geom_dir=args.geom_dir)  # (2,Hg,Wg)
        apply_gap_filling_to_global_channels(geom, maskg_filled, strip_dicts, origins, int(args.fill_k))
        X_img[:, 1 + 8 : 1 + 8 + 2, :, :] = geom[None, :, :, :]

    # -------- save tensors --------
    x_path = out_dir / f"{args.out_prefix}_X_img_{args.split}.npy"
    y_path = out_dir / f"{args.out_prefix}_Y_img_{args.split}.npy"

    np.save(x_path, X_img)
    np.save(y_path, Y_img)

    if fnixap is not None:
        np.save(out_dir / f"fnixap_{args.split}.npy", fnixap)

    np.save(out_dir / f"{args.out_prefix}_maskg_filled_k{int(args.fill_k)}.npy",
            maskg_filled.astype(np.uint8))

    print(f"Saved X: {x_path}  shape={X_img.shape}")
    print(f"Saved Y: {y_path}  shape={Y_img.shape}")
    print(f"Filled mask saved: {out_dir / (args.out_prefix + '_maskg_filled_k' + str(int(args.fill_k)) + '.npy')}")

if __name__ == "__main__":
    main()
