#!/usr/bin/env python3
"""
reconstruct_mesh.py
─────────────────────────────────────────────────────────────────────────────
Reconstruct model predictions from the 3-view tensor representation back
to the original (104, 50) logical mesh shape.

The build_3view_tensors.py script maps each logical cell (ix, iy) to a
pixel (u, v) in one of the three view canvases, via the strip_dicts
pix_cell_ix / pix_cell_iy arrays.  This script inverts that mapping:
for every pixel in each view, it reads the predicted value and writes it
back to the (ix, iy) position in the mesh.

When a logical cell is covered by multiple views (which should not happen
by design, but is handled defensively), the values are averaged.

Inputs
──────
  --pred_prefix   Prefix for prediction files, e.g.
                  scripts/runs/my_run/infer_test/pred_Y_img_test
                  → expects {prefix}_view0.npy, {prefix}_view1.npy, {prefix}_view2.npy
  --layout_path   Path to the layout_map_3views.npz saved by build_3view_tensors.py
  --unroller_module  Python module with RECT_SPECS (default: unrolled_strip_clockwise_adjpreserve)
  --out_path      Output .npy path (default: {pred_prefix}_mesh.npy)

Outputs
───────
  (N, C, 104, 50)  float32 array with predictions mapped back to the
                     original logical mesh.  Cells not covered by any
                     view are left as 0.

Usage
─────
  python reconstruct_mesh.py \\
      --pred_prefix scripts/runs/my_run/infer_test/pred_Y_img_test \\
      --layout_path scripts/tensor/3views_4d/train/global3v_layout_map_3views.npz \\
      --unroller_module unrolled_strip_clockwise_adjpreserve \\
      --out_path scripts/runs/my_run/infer_test/pred_Y_mesh_test.npy
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Slot / spec definitions  (must match build_3view_tensors.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

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

VIEW_SLOTS = {
    "view0": ["TOP_left", "TOP_right"],
    "view1": ["MID_left", "MID_center", "MID_right"],
    "view2": ["Bottom_center"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Strip builder  (same as build_3view_tensors.py)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Compute view layouts  (must match build_3view_tensors.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_view_layouts(strip_dicts, gap_px):
    A = strip_dicts["TOP_left"]
    B = strip_dicts["TOP_right"]
    C = strip_dicts["MID_left"]
    D = strip_dicts["MID_center"]
    E = strip_dicts["MID_right"]
    F = strip_dicts["Bottom_center"]

    h0 = max(A["H"], B["H"])
    w0 = A["W"] + gap_px + B["W"]
    H0 = gap_px + h0 + gap_px
    W0 = w0

    h1 = max(C["H"], D["H"], E["H"])
    w1 = C["W"] + gap_px + D["W"] + gap_px + E["W"]
    H1 = gap_px + h1 + gap_px
    W1 = w1

    h2 = F["H"]
    w2 = gap_px + F["W"] + gap_px
    H2 = gap_px + h2 + gap_px
    W2 = w2

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


# ─────────────────────────────────────────────────────────────────────────────
# Build the inverse mapping: for each view, a list of
#   (pixel_row, pixel_col) → (ix, iy)
# ─────────────────────────────────────────────────────────────────────────────

def build_inverse_maps(
    strip_dicts: Dict,
    origins_view: Dict,
    mesh_shape: Tuple[int, int] = (104, 50),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each view, build arrays:
      pix_to_ix[v, u] = logical ix  (-1 if not a valid cell)
      pix_to_iy[v, u] = logical iy  (-1 if not a valid cell)
      valid[v, u]     = True if this pixel maps to a mesh cell

    These are in the LOCAL view coordinate system (before Hmax/Wmax padding).
    """
    inv_maps = {}

    for view_name, slot_origins in origins_view.items():
        # Determine local canvas size from the strips + origins
        max_v = 0
        max_u = 0
        for slot, (u0, v0) in slot_origins.items():
            sd = strip_dicts[slot]
            max_v = max(max_v, v0 + sd["H"])
            max_u = max(max_u, u0 + sd["W"])

        pix_to_ix = np.full((max_v, max_u), -1, dtype=np.int32)
        pix_to_iy = np.full((max_v, max_u), -1, dtype=np.int32)

        for slot, (u0, v0) in slot_origins.items():
            sd = strip_dicts[slot]
            m  = sd["mask"].astype(bool)
            ix = sd["pix_cell_ix"]
            iy = sd["pix_cell_iy"]

            rows, cols = np.where(m)
            for r, c in zip(rows, cols):
                cell_ix = ix[r, c]
                cell_iy = iy[r, c]
                if cell_ix >= 0 and cell_iy >= 0:
                    pix_to_ix[v0 + r, u0 + c] = cell_ix
                    pix_to_iy[v0 + r, u0 + c] = cell_iy

        inv_maps[view_name] = {
            "pix_to_ix": pix_to_ix,
            "pix_to_iy": pix_to_iy,
        }

    return inv_maps


# ─────────────────────────────────────────────────────────────────────────────
# Core reconstruction function
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_to_mesh(
    pred_views: Dict[str, np.ndarray],
    inv_maps: Dict[str, Dict[str, np.ndarray]],
    layout: Dict,
    mesh_shape: Tuple[int, int] = (104, 50),
) -> np.ndarray:
    """
    Reconstruct predictions from 3 view tensors back to the mesh.

    Parameters
    ----------
    pred_views : dict
        {"view0": (N, C, Hmax, Wmax), "view1": ..., "view2": ...}
    inv_maps : dict
        Output of build_inverse_maps().
    layout : dict
        Loaded layout_map_3views.npz with Hmax, Wmax, H0, W0, etc.
    mesh_shape : tuple
        (104, 50) — the logical grid dimensions.

    Returns
    -------
    mesh_out : (N, C, 104, 50)  float32
    """
    # Get a reference array to determine N and C
    ref = pred_views["view0"]
    N, C = ref.shape[0], ref.shape[1]
    IX, IY = mesh_shape

    mesh_out = np.zeros((N, C, IX, IY), dtype=np.float64)
    count    = np.zeros((IX, IY), dtype=np.int32)

    view_crop = {
        "view0": (int(layout["H0"]), int(layout["W0"])),
        "view1": (int(layout["H1"]), int(layout["W1"])),
        "view2": (int(layout["H2"]), int(layout["W2"])),
    }

    for view_name in ["view0", "view1", "view2"]:
        pred_v = pred_views[view_name]  # (N, C, Hmax, Wmax)
        Hv, Wv = view_crop[view_name]
        pix_ix = inv_maps[view_name]["pix_to_ix"]  # (Hlocal, Wlocal)
        pix_iy = inv_maps[view_name]["pix_to_iy"]

        # Only iterate over valid pixels (where ix >= 0)
        valid = (pix_ix >= 0) & (pix_iy >= 0)
        rows, cols = np.where(valid)

        cell_ix = pix_ix[rows, cols]
        cell_iy = pix_iy[rows, cols]

        # Vectorised scatter: for each valid pixel, add its value
        # to the mesh and increment the count
        for r, c, ix, iy in zip(rows, cols, cell_ix, cell_iy):
            mesh_out[:, :, ix, iy] += pred_v[:, :, r, c].astype(np.float64)

        # Update counts (same for all simulations since geometry is fixed)
        for ix, iy in zip(cell_ix, cell_iy):
            count[ix, iy] += 1

    # Average where multiple views contributed to the same cell
    for ix in range(IX):
        for iy in range(IY):
            if count[ix, iy] > 1:
                mesh_out[:, :, ix, iy] /= count[ix, iy]

    return mesh_out.astype(np.float32), count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct 3-view predictions back to the (104, 50) mesh."
    )
    ap.add_argument("--pred_prefix", required=True,
                    help="Prefix for prediction files. Expects "
                         "{prefix}_view0.npy, {prefix}_view1.npy, {prefix}_view2.npy")
    ap.add_argument("--layout_path", required=True,
                    help="Path to layout_map_3views.npz from build_3view_tensors.py")
    ap.add_argument("--unroller_module", default="unrolled_strip_clockwise_adjpreserve",
                    help="Python module containing RECT_SPECS and unrolling logic")
    ap.add_argument("--gap_px", type=int, default=None,
                    help="Gap pixels (auto-read from layout if omitted)")
    ap.add_argument("--out_path", default=None,
                    help="Output .npy path (default: {pred_prefix}_mesh.npy)")
    args = ap.parse_args()

    # ── Load layout ─────────────────────────────────────────────────────────
    layout = dict(np.load(args.layout_path, allow_pickle=True))
    gap_px = int(layout["gap_px"]) if args.gap_px is None else args.gap_px

    print(f"\n{'='*60}")
    print(f"  Reconstruct 3-view predictions → (104, 50) mesh")
    print(f"  Layout : {args.layout_path}")
    print(f"  Gap px : {gap_px}")
    print(f"  Hmax={int(layout['Hmax'])}  Wmax={int(layout['Wmax'])}")
    print(f"{'='*60}\n")

    # ── Build strip dicts (same as build_3view_tensors.py) ──────────────────
    # Add the directory containing the unroller module to sys.path
    unroller_path = Path(args.unroller_module)
    if unroller_path.suffix == ".py":
        # Given as a file path
        sys.path.insert(0, str(unroller_path.parent.resolve()))
        mod_name = unroller_path.stem
    else:
        mod_name = args.unroller_module

    u = importlib.import_module(mod_name)
    name_to_spec = {d.get("name"): d for d in u.RECT_SPECS}

    strip_dicts = {}
    for slot, specname in SLOT_TO_SPECNAME.items():
        strip_dicts[slot] = build_strip(u, name_to_spec[specname], tf=SLOT_TF[slot])

    # ── Compute view layouts ────────────────────────────────────────────────
    origins_view, view_sizes = compute_view_layouts(strip_dicts, gap_px=gap_px)

    # ── Build inverse maps ──────────────────────────────────────────────────
    inv_maps = build_inverse_maps(strip_dicts, origins_view)

    # ── Load predictions ────────────────────────────────────────────────────
    pred_views = {}
    for v in range(3):
        vname = f"view{v}"
        pred_path = Path(f"{args.pred_prefix}_{vname}.npy")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        pred_views[vname] = np.load(pred_path)
        print(f"  Loaded {vname}: {pred_path}  shape={pred_views[vname].shape}")

    # ── Reconstruct ─────────────────────────────────────────────────────────
    mesh_out, count = reconstruct_to_mesh(pred_views, inv_maps, layout)

    n_covered = int((count > 0).sum())
    n_total   = 104 * 50
    n_overlap = int((count > 1).sum())
    print(f"\n  Mesh coverage: {n_covered}/{n_total} cells covered")
    if n_overlap > 0:
        print(f"  Overlap: {n_overlap} cells covered by multiple views (averaged)")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = args.out_path if args.out_path else f"{args.pred_prefix}_mesh.npy"
    np.save(out_path, mesh_out)
    print(f"\n  Saved: {out_path}  shape={mesh_out.shape}")

    # Also save the coverage count for diagnostics
    count_path = str(Path(out_path).with_name(
        Path(out_path).stem + "_coverage.npy"
    ))
    np.save(count_path, count)
    print(f"  Saved: {count_path}  shape={count.shape}")
    print()


if __name__ == "__main__":
    main()
