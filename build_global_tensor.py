#!/usr/bin/env python3
"""
plotting_grid.py  (modified)

Build a fixed global rectangular canvas by tiling multiple unrolled strips (from unroller),
then generate per-simulation CNN tensors:

Outputs:
- global_layout_map.npz      (constant, reusable mapping + mask + slot origins)
- X_img_<split>.npy          float32, (N, C_in, H, W)
- Y_img_<split>.npy          float32, (N, C_out, H, W)
- fnixap_<split>.npy         float32, (N,)    [optional]

CNN input design:
- channel 0: global valid-cell mask (1 for cell, 0 for gap)
- channels 1..8: the 8 scalars from X_tmp broadcast over all valid pixels (masked)
Optional:
- geometry channels: centroid_x, centroid_y at each pixel (masked)

CNN output design (configurable):
- default: te, ti, na(10), ua(10) -> 22 channels
- can be reduced via --out_targets

Run examples:
  python plotting_grid.py --split train --out_prefix out_train
  python plotting_grid.py --split test  --out_prefix out_test --limit_n 50
"""

import argparse
import importlib
import os
from pathlib import Path
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
        pix_cell_ix=pix_ix_t,
        pix_cell_iy=pix_iy_t,
    )

# -----------------------
# Fixed layout (same as original)
# -----------------------
def compute_layout(strip_dicts, gap_px):
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
        maskg[v0:v0+H, u0:u0+W] |= sd["mask"]
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
    # store per-slot origin + per-slot maps
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
def paste_field_global(maskg, layout, slot_dicts, origins, field_2d):
    """
    field_2d: array (104,50) for one simulation and one channel.
    Returns global (Hg,Wg) float32 with zeros in gaps.
    """
    Hg, Wg = maskg.shape
    out = np.zeros((Hg, Wg), dtype=np.float32)

    for slot, sd in slot_dicts.items():
        u0, v0 = origins[slot]
        m = sd["mask"].astype(bool)
        pix_ix = sd["pix_cell_ix"]
        pix_iy = sd["pix_cell_iy"]

        # gather values for occupied pixels
        sub = out[v0:v0+sd["H"], u0:u0+sd["W"]]
        ix = pix_ix[m]
        iy = pix_iy[m]
        sub[m] = field_2d[ix, iy].astype(np.float32)
        out[v0:v0+sd["H"], u0:u0+sd["W"]] = sub

    # gaps already 0
    return out

def paste_field_global_3d(maskg, slot_dicts, origins, field_3d):
    """
    field_3d: array (104,50,S) for one simulation, multiple species channels.
    Returns global (S,Hg,Wg) float32
    """
    Hg, Wg = maskg.shape
    S = field_3d.shape[-1]
    out = np.zeros((S, Hg, Wg), dtype=np.float32)
    for s in range(S):
        out[s] = paste_field_global(maskg, None, slot_dicts, origins, field_3d[:, :, s])
    return out

# -----------------------
# Geometry channels (optional)
# -----------------------
def build_geometry_channels(maskg, slot_dicts, origins, geom_dir="geometry"):
    """
    Build centroid_x and centroid_y global channels using crx/cry.
    Returns (2,Hg,Wg) float32.
    """
    crx = np.load(os.path.join(geom_dir, "crx.npy"))
    cry = np.load(os.path.join(geom_dir, "cry.npy"))

    # centroid per cell (104,50)
    cx = crx.mean(axis=2)
    cy = cry.mean(axis=2)

    gx = paste_field_global(maskg, None, slot_dicts, origins, cx)
    gy = paste_field_global(maskg, None, slot_dicts, origins, cy)
    geom = np.stack([gx, gy], axis=0).astype(np.float32)

    # mask gaps
    m = maskg.astype(np.float32)
    geom *= m[None, :, :]
    return geom

# -----------------------
# Output target selection
# -----------------------
def parse_out_targets(s: str):
    """
    Comma-separated list among: te,ti,na,ua
    Default: te,ti,na,ua
    """
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
    ap.add_argument("--gap_px", type=int, default=10)
    ap.add_argument("--geom_dir", default="geometry")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_prefix", default="out")
    ap.add_argument("--layout_map", default="global_layout_map.npz")

    ap.add_argument("--include_geometry", action="store_true",
                    help="Add 2 input channels: centroid_x, centroid_y.")
    ap.add_argument("--out_targets", default="te,ti,na,ua",
                    help="Comma-separated subset among: te,ti,na,ua")
    ap.add_argument("--include_fnixap", action="store_true",
                    help="Also save fnixap_<split>.npy")

    ap.add_argument("--limit_n", type=int, default=0,
                    help="If >0, process only first N simulations (debug).")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split

    # -------- build layout mapping (constant) --------
    u = importlib.import_module(args.unroller_module)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}

    missing = [nm for nm in SLOT_TO_SPECNAME.values() if nm not in name_to_spec]
    if missing:
        raise RuntimeError(f"Missing RECT_SPECS names: {missing}")

    # unroll + fixed orientation for each slot
    strip_dicts = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        tf = SLOT_TF[slot]
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=tf)

    origins, Wg, Hg = compute_layout(strip_dicts, gap_px=args.gap_px)
    maskg = paste_global_mask(strip_dicts, origins, Wg, Hg)

    # save mapping
    save_layout_map(args.layout_map, strip_dicts, origins, Wg, Hg, maskg, args.gap_px)
    print(f"Saved layout map: {args.layout_map}  (Hg={Hg}, Wg={Wg})")

    # -------- load split data --------
    X = np.load(split_dir / "X_tmp.npy")          # (N,8)
    te = np.load(split_dir / "te_tmp.npy")        # (N,104,50)
    ti = np.load(split_dir / "ti_tmp.npy")        # (N,104,50)
    na = np.load(split_dir / "na_tmp.npy")        # (N,104,50,10)
    ua = np.load(split_dir / "ua_tmp.npy")        # (N,104,50,10)
    fnixap = None
    if args.include_fnixap:
        fnixap = np.load(split_dir / "fnixap_tmp.npy").astype(np.float32)

    N = X.shape[0]
    if args.limit_n and args.limit_n > 0:
        N = min(N, args.limit_n)
        X = X[:N]
        te = te[:N]
        ti = ti[:N]
        na = na[:N]
        ua = ua[:N]
        if fnixap is not None:
            fnixap = fnixap[:N]

    # -------- build input tensor X_img --------
    # channels: [mask] + 8 broadcast scalars + optional geometry(2)
    c_in = 1 + 8 + (2 if args.include_geometry else 0)
    X_img = np.zeros((N, c_in, Hg, Wg), dtype=np.float32)

    m = maskg.astype(np.float32)
    X_img[:, 0, :, :] = m[None, :, :]

    # broadcast scalars with mask
    for k in range(8):
        X_img[:, 1 + k, :, :] = (X[:, k].astype(np.float32)[:, None, None]) * m[None, :, :]

    if args.include_geometry:
        geom = build_geometry_channels(maskg, strip_dicts, origins, geom_dir=args.geom_dir)  # (2,Hg,Wg)
        X_img[:, 1 + 8:1 + 8 + 2, :, :] = geom[None, :, :, :]

    # -------- build output tensor Y_img --------
    targets = parse_out_targets(args.out_targets)
    out_parts = []
    out_names = []

    # te, ti are single-channel
    if "te" in targets:
        out_parts.append(("te", 1))
    if "ti" in targets:
        out_parts.append(("ti", 1))
    if "na" in targets:
        out_parts.append(("na", 10))
    if "ua" in targets:
        out_parts.append(("ua", 10))

    c_out = sum(ch for _, ch in out_parts)
    Y_img = np.zeros((N, c_out, Hg, Wg), dtype=np.float32)

    # paste per simulation
    ch0 = 0
    for name, ch in out_parts:
        if name in ("te", "ti"):
            for i in range(N):
                field = te[i] if name == "te" else ti[i]
                J_PER_EV = 1.602176634e-19
                field = field / J_PER_EV
                Y_img[i, ch0, :, :] = paste_field_global(maskg, None, strip_dicts, origins, field)
            out_names.append(name)
            ch0 += 1
        elif name in ("na", "ua"):
            for i in range(N):
                field3 = na[i] if name == "na" else ua[i]  # (104,50,10)
                pasted = paste_field_global_3d(maskg, strip_dicts, origins, field3)  # (10,Hg,Wg)
                Y_img[i, ch0:ch0+10, :, :] = pasted
            out_names.extend([f"{name}{s}" for s in range(10)])
            ch0 += 10
        else:
            raise RuntimeError(name)

    # mask outputs in gaps (already zero, but enforce)
    Y_img *= m[None, None, :, :]

    # -------- save outputs --------
    out_prefix = Path(args.out_prefix)
    out_x = f"{out_prefix}_X_img_{args.split}.npy"
    out_y = f"{out_prefix}_Y_img_{args.split}.npy"

    np.save(out_x, X_img)
    np.save(out_y, Y_img)

    if fnixap is not None:
        out_f = f"{out_prefix}_fnixap_{args.split}.npy"
        np.save(out_f, fnixap)

    # metadata
    meta = {
        "split": args.split,
        "N": int(N),
        "Hg": int(Hg),
        "Wg": int(Wg),
        "C_in": int(c_in),
        "C_out": int(c_out),
        "in_channels": ["mask"] + [f"x{k}" for k in range(8)] + (["centroid_x", "centroid_y"] if args.include_geometry else []),
        "out_channels": out_names,
        "gap_px": int(args.gap_px),
        "slots": list(SLOT_TO_SPECNAME.keys()),
        "slot_to_specname": SLOT_TO_SPECNAME,
        "slot_tf": SLOT_TF,
        "layout_map": args.layout_map,
    }
    meta_path = f"{out_prefix}_meta_{args.split}.npz"
    np.savez_compressed(meta_path, **{k: np.array(v, dtype=object) for k, v in meta.items()})
    print(f"Saved X tensor: {out_x}  shape={X_img.shape}")
    print(f"Saved Y tensor: {out_y}  shape={Y_img.shape}")
    if fnixap is not None:
        print(f"Saved fnixap:  {out_f}  shape={fnixap.shape}")
    print(f"Saved meta:    {meta_path}")

if __name__ == "__main__":
    main()
