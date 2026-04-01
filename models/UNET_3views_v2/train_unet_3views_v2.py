#!/usr/bin/env python3
"""
train_unet_3views_v2.py

Boundary-Aware UNet for 3-view SOL plasma surrogate modeling.

Key improvements over train_unet_3views.py:
  1. Adjacency padding reinjection after every convolutional layer (within-view).
     This ensures that strip-edge pixels always see their true physical neighbours
     through gap regions, at every network depth — not just at the input.
  2. Per-category epsilon for the log10 transform:
       te/ti (ch 0-1):  eps = 1e-2  (0.01 eV is physically negligible)
       na    (ch 2-11): eps = 1e8   (1e8 m⁻³ is negligible in SOL)
       ua    (ch 12-21): N/A — uses asinh, no eps needed
  3. Stores per-channel transformed-space min/max from training data in the
     checkpoint, so that inference can clamp predictions before inverse transform
     (prevents 10^25-type explosions on density channels).

Strategy A (within-view only):
  Each view is processed by its own BoundaryAwareUNet instance.  Cross-view
  vertical padding is handled at the tensor-construction stage only (by
  build_3view_tensors.py).  This isolates the effect of boundary-aware
  convolutions from weight sharing and is the recommended starting point.

Physical adjacencies maintained per view:
  View 0:  A right ↔ B left  (one horizontal gap)
  View 1:  C right ↔ D left, D right ↔ E left  (two horizontal gaps)
  View 2:  F left ↔ F right  (periodic wrap through left/right gaps)

Usage:
  python train_unet_3views_v2.py \\
      --view 0 \\
      --tensor_prefix scripts/tensor/3views_4d/train/global3v \\
      --train_split train \\
      --test_prefix  scripts/tensor/3views_4d/test/global3v \\
      --test_split   test \\
      --layout_npz   scripts/tensor/3views_4d/train/global3v_layout_map_3views.npz \\
      --run_dir      scripts/runs/unet_3views_v2/view0 \\
      --y_channels   all \\
      --epochs 50 --batch_size 32 --base 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------
POS_CHANNELS    = set(range(0, 12))   # te, ti, na(10) → log10 transform
SIGNED_CHANNELS = set(range(12, 22))  # ua(10)          → asinh transform

# Per-category epsilon for log10(max(y,0) + eps).
# te/ti values are in eV (range ~0.01–1e5):  eps=1e-2 is safe.
# na values are particle densities (range ~1e10–1e21 m⁻³): eps=1e8 is safe.
EPS_TEMP    = 1e-2   # for channels 0,1  (te, ti)
EPS_DENSITY = 1e8    # for channels 2..11 (na)


def eps_for_channel(c: int) -> float:
    """Return the appropriate epsilon for a positive-channel index."""
    if c in (0, 1):
        return EPS_TEMP
    elif 2 <= c <= 11:
        return EPS_DENSITY
    else:
        raise ValueError(f"Channel {c} is not a positive channel.")


# ---------------------------------------------------------------------------
# Boundary spec computation from layout metadata
#
# A "boundary spec" is a tuple:
#   (src_col_start, src_col_end, dst_col_start, dst_col_end, row_start, row_end)
# describing a column-copy operation that reinjects strip-edge pixels into a
# gap region.  For each horizontal adjacency there are TWO specs:
#   1. right-edge of left strip → left side of gap   (right-to-gap)
#   2. left-edge of right strip → right side of gap  (left-to-gap)
# ---------------------------------------------------------------------------

def _compute_strip_origins_and_sizes(
    layout: dict, view_id: int
) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (u0, v0, w, h) for each strip in the given view,
    derived deterministically from the layout metadata and the same
    formulae used by build_3view_tensors.py → compute_view_layouts().

    The layout dict must contain: gap_px, mask_view0/1/2, W0,H0,W1,H1,W2,H2.
    We reconstruct strip widths/heights from the view-local canvas sizes and
    gap_px, matching the code in build_3view_tensors.py exactly.
    """
    gap = int(layout["gap_px"])

    # Recover per-view local canvas sizes (before padding to Hmax×Wmax).
    # These are the values W0,H0 … W2,H2 stored in the layout NPZ.
    Wv = int(layout[f"W{view_id}"])
    Hv = int(layout[f"H{view_id}"])

    # The local canvas has gap_px rows at top and bottom (cross-view gap rows).
    # The strip content sits between rows gap .. gap+h_strip, where h_strip is
    # the maximum strip height in that view.
    v0 = gap  # top-gap offset for all views

    if view_id == 0:
        # View 0:  [A  | gap | B]   no left/right outer gaps
        # W0 = wA + gap + wB,  H0 = gap + max(hA,hB) + gap
        # We need wA.  The mask tells us: the leftmost strip starts at u=0
        # and extends until the first all-zero column.
        mask = np.asarray(layout["mask_view0"])
        wA = _find_strip_width_from_left(mask, v0)
        hA = _find_strip_height(mask, 0, v0)
        wB = Wv - gap - wA
        hB = _find_strip_height(mask, wA + gap, v0)
        return [(0, v0, wA, hA), (wA + gap, v0, wB, hB)]

    elif view_id == 1:
        # View 1:  [C | gap | D | gap | E]
        mask = np.asarray(layout["mask_view1"])
        wC = _find_strip_width_from_left(mask, v0)
        hC = _find_strip_height(mask, 0, v0)
        # After C + gap, D starts
        d_start = wC + gap
        wD = _find_strip_width_from_pos(mask, d_start, v0)
        hD = _find_strip_height(mask, d_start, v0)
        e_start = d_start + wD + gap
        wE = Wv - e_start
        hE = _find_strip_height(mask, e_start, v0)
        return [
            (0,       v0, wC, hC),
            (d_start, v0, wD, hD),
            (e_start, v0, wE, hE),
        ]

    elif view_id == 2:
        # View 2:  [gap_left | F | gap_right]
        # W2 = gap + wF + gap,  H2 = gap + hF + gap
        mask = np.asarray(layout["mask_view2"])
        wF = Wv - 2 * gap
        hF = _find_strip_height(mask, gap, v0)
        return [(gap, v0, wF, hF)]

    else:
        raise ValueError(f"Unknown view_id {view_id}")


def _find_strip_width_from_left(mask: np.ndarray, v0: int) -> int:
    """Width of the leftmost strip (starts at column 0)."""
    row = mask[v0, :]  # pick a row inside the strip region
    # Find first zero after initial ones
    for u in range(len(row)):
        if row[u] == 0:
            return u
    return len(row)


def _find_strip_width_from_pos(mask: np.ndarray, u_start: int, v0: int) -> int:
    """Width of a strip starting at column u_start."""
    row = mask[v0, :]
    for u in range(u_start, len(row)):
        if row[u] == 0:
            return u - u_start
    return len(row) - u_start


def _find_strip_height(mask: np.ndarray, u: int, v0: int) -> int:
    """Height of a strip starting at row v0, sampled at column u."""
    col = mask[:, u]
    for v in range(v0, len(col)):
        if col[v] == 0:
            return v - v0
    return len(col) - v0


def compute_boundary_specs_for_view(
    layout: dict, view_id: int, k: int, scale: int = 1
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Compute the list of column-copy specs for a given view at a given
    spatial scale (1 = full resolution, 2 = half, 4 = quarter, 8 = eighth).

    Each spec is:
        (src_col_start, src_col_end, dst_col_start, dst_col_end, row_start, row_end)

    At coarser scales all coordinates are integer-divided by `scale` and k
    is similarly reduced.  If the scaled k becomes 0, the spec list is empty
    (no padding to reinject at that resolution).
    """
    gap = int(layout["gap_px"])
    strips = _compute_strip_origins_and_sizes(layout, view_id)

    k_s = k // scale
    if k_s <= 0:
        return []

    gap_s = gap // scale
    if gap_s < 2 * k_s:
        # Not enough room in the gap for padding at this scale
        return []

    specs: List[Tuple[int, int, int, int, int, int]] = []

    if view_id == 0:
        # View 0: A | gap | B  — one horizontal boundary
        uA, vA, wA, hA = strips[0]
        uB, vB, wB, hB = strips[1]
        _add_horizontal_specs(specs, uA, vA, wA, hA, uB, vB, wB, hB, gap, k, scale)

    elif view_id == 1:
        # View 1: C | gap | D | gap | E  — two horizontal boundaries
        uC, vC, wC, hC = strips[0]
        uD, vD, wD, hD = strips[1]
        uE, vE, wE, hE = strips[2]
        _add_horizontal_specs(specs, uC, vC, wC, hC, uD, vD, wD, hD, gap, k, scale)
        _add_horizontal_specs(specs, uD, vD, wD, hD, uE, vE, wE, hE, gap, k, scale)

    elif view_id == 2:
        # View 2: [gap_left] F [gap_right]  — periodic wrap
        uF, vF, wF, hF = strips[0]
        _add_periodic_wrap_specs(specs, uF, vF, wF, hF, gap, k, scale)

    return specs


def _add_horizontal_specs(
    specs: list,
    uL: int, vL: int, wL: int, hL: int,   # left strip
    uR: int, vR: int, wR: int, hR: int,   # right strip
    gap: int, k: int, scale: int,
) -> None:
    """
    Add two copy specs for a horizontal boundary between a left and right strip.
    The gap sits between columns (uL+wL) and uR.

    Spec 1: copy the last k_s columns of the left strip into the first k_s columns
             of the gap  (so the gap's left side mirrors the left strip's right edge).
    Spec 2: copy the first k_s columns of the right strip into the last k_s columns
             of the gap  (so the gap's right side mirrors the right strip's left edge).

    IMPORTANT: We compute the gap boundary position ("anchor") in downsampled
    coordinates first, then offset by k_s from that anchor.  This guarantees
    that source and destination slices always have exactly the same width (k_s).
    Dividing original src/dst endpoints independently can produce mismatched
    widths due to integer-division rounding (e.g. (26-3)//2=11 vs 26//2=13
    gives src width 2, but k_s=3//2=1 gives dst width 1).
    """
    s = scale
    k_s = k // s
    if k_s <= 0:
        return

    h = min(hL, hR)
    rs = vL // s
    re = (vL + h) // s
    if re <= rs:
        return

    # Anchor: gap boundary positions in downsampled coordinates.
    # left_end  = first gap column (just past the left strip's right edge)
    # right_start = first column of the right strip (just past the gap)
    left_end    = (uL + wL) // s
    right_start = uR // s

    # Spec 1: last k_s cols of left strip → first k_s cols of gap
    #   source = [left_end - k_s, left_end)      inside left strip
    #   dest   = [left_end,       left_end + k_s) inside gap
    src1_cs = left_end - k_s
    src1_ce = left_end
    dst1_cs = left_end
    dst1_ce = left_end + k_s

    # Spec 2: first k_s cols of right strip → last k_s cols of gap
    #   source = [right_start,       right_start + k_s) inside right strip
    #   dest   = [right_start - k_s, right_start)       inside gap
    src2_cs = right_start
    src2_ce = right_start + k_s
    dst2_cs = right_start - k_s
    dst2_ce = right_start

    if src1_cs >= 0 and dst1_ce > dst1_cs:
        specs.append((src1_cs, src1_ce, dst1_cs, dst1_ce, rs, re))
    if dst2_cs >= 0 and src2_ce > src2_cs:
        specs.append((src2_cs, src2_ce, dst2_cs, dst2_ce, rs, re))


def _add_periodic_wrap_specs(
    specs: list,
    uF: int, vF: int, wF: int, hF: int,
    gap: int, k: int, scale: int,
) -> None:
    """
    Add two copy specs for the periodic wrap of strip F (view 2).
    The canvas is: [gap_left (gap px)] [F (wF px)] [gap_right (gap px)]
    F starts at column uF = gap.

    Spec 1: F's right edge (last k_s cols) → right side of the left gap.
    Spec 2: F's left edge (first k_s cols) → left side of the right gap.

    Uses k_s symmetrically from downsampled boundary positions (same fix as
    _add_horizontal_specs — avoids integer-division width mismatches).
    """
    s = scale
    k_s = k // s
    if k_s <= 0:
        return

    rs = vF // s
    re = (vF + hF) // s
    if re <= rs:
        return

    # Downsampled boundary positions
    f_start = uF // s              # first column of F
    f_end   = (uF + wF) // s       # first column after F (= start of right gap)
    gap_s   = gap // s             # left gap ends at column gap_s (= f_start)

    # Spec 1: F's right edge (last k_s cols) → left gap's right side
    src1_cs = f_end - k_s
    src1_ce = f_end
    dst1_cs = gap_s - k_s
    dst1_ce = gap_s

    # Spec 2: F's left edge (first k_s cols) → right gap's left side
    src2_cs = f_start
    src2_ce = f_start + k_s
    dst2_cs = f_end
    dst2_ce = f_end + k_s

    if dst1_cs >= 0 and dst1_ce > dst1_cs:
        specs.append((src1_cs, src1_ce, dst1_cs, dst1_ce, rs, re))
    if src2_ce > src2_cs and dst2_ce > dst2_cs:
        specs.append((src2_cs, src2_ce, dst2_cs, dst2_ce, rs, re))


def compute_all_boundary_specs(
    layout: dict, view_id: int, k: int, n_levels: int = 4
) -> Dict[int, List[Tuple[int, int, int, int, int, int]]]:
    """
    Compute boundary specs at every resolution level of the UNet.

    Level 0 = full resolution (scale 1)
    Level 1 = half            (scale 2)
    Level 2 = quarter         (scale 4)
    Level 3 = eighth          (scale 8)  — bottleneck
    """
    specs = {}
    for level in range(n_levels):
        scale = 2 ** level
        specs[level] = compute_boundary_specs_for_view(layout, view_id, k, scale)
    return specs


# ---------------------------------------------------------------------------
# Masked statistics helpers
# ---------------------------------------------------------------------------

def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support.")
    return den


def masked_channel_stats_x(
    x_mem: np.ndarray, mask_ch: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel masked mean and std for X.  Mask channel forced to 0/1."""
    _, C, _, _ = x_mem.shape
    m   = x_mem[:, mask_ch:mask_ch + 1, :, :].astype(np.float64)
    den = _masked_den(m)

    mean = np.zeros(C, dtype=np.float64)
    var  = np.zeros(C, dtype=np.float64)
    for c in range(C):
        xc      = x_mem[:, c:c + 1, :, :].astype(np.float64)
        mean[c] = float((xc * m).sum() / den)
    for c in range(C):
        xc     = x_mem[:, c:c + 1, :, :].astype(np.float64)
        var[c] = float((((xc - mean[c]) ** 2) * m).sum() / den)

    std           = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch] = 0.0
    std[mask_ch]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def masked_mean_std_transformed_y(
    y_mem: np.ndarray,
    mask_mem: np.ndarray,
    y_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel stats in TRANSFORMED space, using per-category eps.

    Returns:
      y_mean    : (C_sel,) float32
      y_std     : (C_sel,) float32
      s_c       : (C_sel,) float32   (velocity scale; 1.0 for positive channels)
      y_min_t   : (C_sel,) float32   min of transformed values (for clamping)
      y_max_t   : (C_sel,) float32   max of transformed values (for clamping)
    """
    m   = mask_mem.astype(np.float64)
    den = _masked_den(m)
    C_sel = len(y_indices)

    y_mean  = np.zeros(C_sel, dtype=np.float64)
    y_var   = np.zeros(C_sel, dtype=np.float64)
    s_c     = np.ones(C_sel,  dtype=np.float64)
    y_min_t = np.full(C_sel, np.inf,  dtype=np.float64)
    y_max_t = np.full(C_sel, -np.inf, dtype=np.float64)

    # 1. Compute s_c for signed channels (physical-space masked std)
    for j, c in enumerate(y_indices):
        if c in SIGNED_CHANNELS:
            yc  = y_mem[:, c:c + 1, :, :].astype(np.float64)
            mu  = float((yc * m).sum() / den)
            var = float((((yc - mu) ** 2) * m).sum() / den)
            s_c[j] = float(np.sqrt(max(var, 1e-12)))

    # 2. Per-channel: transform, compute mean, var, min, max
    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
        if c in POS_CHANNELS:
            eps_c = float(eps_for_channel(c))
            t = np.log10(np.maximum(yc, 0.0) + eps_c)
        else:
            t = np.arcsinh(yc / s_c[j])

        y_mean[j] = float((t * m).sum() / den)

        # min/max over masked pixels
        t_masked = t[m > 0.5]
        if t_masked.size > 0:
            y_min_t[j] = float(t_masked.min())
            y_max_t[j] = float(t_masked.max())

    # 3. Variance
    for j, c in enumerate(y_indices):
        yc = y_mem[:, c:c + 1, :, :].astype(np.float64)
        if c in POS_CHANNELS:
            eps_c = float(eps_for_channel(c))
            t = np.log10(np.maximum(yc, 0.0) + eps_c)
        else:
            t = np.arcsinh(yc / s_c[j])
        y_var[j] = float((((t - y_mean[j]) ** 2) * m).sum() / den)

    y_std = np.sqrt(np.maximum(y_var, 1e-12))

    return (
        y_mean.astype(np.float32),
        y_std.astype(np.float32),
        s_c.astype(np.float32),
        y_min_t.astype(np.float32),
        y_max_t.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ViewTensorDataset(Dataset):
    """Load one view's X/Y tensors with normalization and channel-aware transforms."""

    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        y_indices: Sequence[int],
        x_mean: Optional[np.ndarray],
        x_std:  Optional[np.ndarray],
        y_mean: np.ndarray,
        y_std:  np.ndarray,
        s_c:    np.ndarray,
    ):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")

        if self.X.shape[0] != self.Y.shape[0] or self.X.shape[2:] != self.Y.shape[2:]:
            raise ValueError(f"X/Y shape mismatch: {self.X.shape} vs {self.Y.shape}")

        self.y_indices = list(map(int, y_indices))
        self.x_mean    = x_mean
        self.x_std     = x_std
        self.y_mean    = np.asarray(y_mean, dtype=np.float32)
        self.y_std     = np.asarray(y_std,  dtype=np.float32)
        self.s_c       = np.asarray(s_c,    dtype=np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _transform_y(self, y_sel: torch.Tensor) -> torch.Tensor:
        """Apply per-channel transform using per-category eps."""
        out = torch.empty_like(y_sel)
        for j, c in enumerate(self.y_indices):
            if c in POS_CHANNELS:
                eps_c = eps_for_channel(c)
                out[j] = torch.log10(torch.clamp(y_sel[j], min=0.0) + eps_c)
            else:
                out[j] = torch.asinh(y_sel[j] / float(self.s_c[j]))
        return out

    def __getitem__(self, idx: int):
        x      = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y_full = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32))
        m      = x[0:1]  # (1, H, W) mask

        # X normalisation
        if self.x_mean is not None and self.x_std is not None:
            xm = torch.from_numpy(self.x_mean).view(-1, 1, 1)
            xs = torch.from_numpy(self.x_std).view(-1, 1, 1)
            x  = (x - xm) / xs

        # Select + transform + normalize Y
        y_sel = y_full[self.y_indices, :, :]
        y_t   = self._transform_y(y_sel)
        ym    = torch.from_numpy(self.y_mean).view(-1, 1, 1)
        ys    = torch.from_numpy(self.y_std).view(-1, 1, 1)
        y_n   = (y_t - ym) / ys

        return x, y_n, m


# ---------------------------------------------------------------------------
# Model — Boundary-Aware UNet
# ---------------------------------------------------------------------------

class BoundaryAwareConvBlock(nn.Module):
    """
    Two 3×3 convolutions, each followed by boundary padding reinjection and ReLU.

    The boundary specs are a list of copy operations that overwrite gap pixels
    with the strip-edge pixels from the physically adjacent strip.  This ensures
    that at every layer depth, convolutional kernels near strip edges see their
    true physical neighbours rather than stale/zero gap values.

    The .clone() used in the copy is differentiable, so gradients flow through
    the boundary and encourage cross-gap consistency.
    """

    def __init__(self, c_in: int, c_out: int,
                 boundary_specs: List[Tuple[int, int, int, int, int, int]]):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in,  c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.boundary_specs = boundary_specs

    def _apply_boundary_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Copy strip-edge pixels into adjacent gap regions (in-place, differentiable)."""
        for src_cs, src_ce, dst_cs, dst_ce, rs, re in self.boundary_specs:
            x[:, :, rs:re, dst_cs:dst_ce] = x[:, :, rs:re, src_cs:src_ce].clone()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self._apply_boundary_padding(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self._apply_boundary_padding(x)
        x = self.relu(x)
        return x


class BoundaryAwareUNet(nn.Module):
    """
    UNet with boundary-aware convolution blocks at every encoder/decoder level.

    boundary_specs_per_level: dict mapping level → list of copy specs.
        Level 0: full resolution   (Hmax, Wmax)
        Level 1: half              (Hmax//2, Wmax//2)
        Level 2: quarter           (Hmax//4, Wmax//4)
        Level 3: eighth            (Hmax//8, Wmax//8) — bottleneck

    The same specs are reused in the decoder at matching resolution levels.
    """

    def __init__(self, c_in: int, c_out: int, base: int = 32,
                 boundary_specs_per_level: Optional[Dict[int, list]] = None):
        super().__init__()
        specs = boundary_specs_per_level or {}

        self.enc1       = BoundaryAwareConvBlock(c_in,      base,      specs.get(0, []))
        self.enc2       = BoundaryAwareConvBlock(base,      base * 2,  specs.get(1, []))
        self.enc3       = BoundaryAwareConvBlock(base * 2,  base * 4,  specs.get(2, []))
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = BoundaryAwareConvBlock(base * 4,  base * 8,  specs.get(3, []))

        self.up3  = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = BoundaryAwareConvBlock(base * 8, base * 4, specs.get(2, []))
        self.up2  = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = BoundaryAwareConvBlock(base * 4, base * 2, specs.get(1, []))
        self.up1  = nn.Conv2d(base * 2, base,     1)
        self.dec1 = BoundaryAwareConvBlock(base * 2, base,     specs.get(0, []))
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _crop(x: torch.Tensor, hw) -> torch.Tensor:
        th, tw = hw
        dh, dw = x.shape[-2] - th, x.shape[-1] - tw
        return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        u3 = self.up3(F.interpolate(b,  size=e3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.dec3(torch.cat([u3, self._crop(e3, u3.shape[-2:])], 1))

        u2 = self.up2(F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([u2, self._crop(e2, u2.shape[-2:])], 1))

        u1 = self.up1(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([u1, self._crop(e1, u1.shape[-2:])], 1))

        return self.out(d1)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def masked_mae_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).abs().mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return num / den


def masked_rmse_per_channel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).pow(2).mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(eps)
    return torch.sqrt(num / den)


def channel_balanced_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mae_per_channel(pred, target, mask).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    mae_acc = rmse_acc = None
    n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred     = model(x) * m
        mae_c    = masked_mae_per_channel(pred, y, m).cpu()
        rmse_c   = masked_rmse_per_channel(pred, y, m).cpu()
        mae_acc  = mae_c  if mae_acc  is None else mae_acc  + mae_c
        rmse_acc = rmse_c if rmse_acc is None else rmse_acc + rmse_c
        n += 1
    if n == 0:
        return {"mae_avg": float("nan"), "rmse_avg": float("nan"),
                "mae_per_channel": [], "rmse_per_channel": []}
    mae_acc  /= n
    rmse_acc /= n
    return {
        "mae_avg":          float(mae_acc.mean()),
        "rmse_avg":         float(rmse_acc.mean()),
        "mae_per_channel":  mae_acc.numpy().tolist(),
        "rmse_per_channel": rmse_acc.numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_channels(s: str, c_out_total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(c_out_total))
    ch = [int(p) for p in s.split(",") if p.strip()]
    if any(c < 0 or c >= c_out_total for c in ch):
        raise ValueError(f"y_channels out of bounds: {ch}, total={c_out_total}")
    return ch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_layout(npz_path: str) -> dict:
    """Load layout metadata from the NPZ produced by build_3view_tensors.py."""
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in z.files:
        v = z[k]
        if v.ndim == 0:
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Boundary-Aware UNet on a single 3-view tensor."
    )

    ap.add_argument("--view", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_prefix", required=True)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--layout_npz", required=True,
                    help="Path to *_layout_map_3views.npz from build_3view_tensors.py.")
    ap.add_argument("--run_dir", required=True)

    ap.add_argument("--y_channels", default="all")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--base",         type=int,   default=32)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- resolve file paths ----
    view_tag     = f"view{args.view}"
    train_x_path = Path(f"{args.tensor_prefix}_{view_tag}_X_img_{args.train_split}.npy")
    train_y_path = Path(f"{args.tensor_prefix}_{view_tag}_Y_img_{args.train_split}.npy")
    test_x_path  = Path(f"{args.test_prefix}_{view_tag}_X_img_{args.test_split}.npy")
    test_y_path  = Path(f"{args.test_prefix}_{view_tag}_Y_img_{args.test_split}.npy")

    for p in (train_x_path, train_y_path, test_x_path, test_y_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # ---- load layout metadata ----
    layout = load_layout(args.layout_npz)
    k_pad  = int(layout["k"])       # adjacency padding half-width

    # Compute boundary specs for each resolution level (4 levels for 3 encoder + bottleneck)
    boundary_specs = compute_all_boundary_specs(layout, args.view, k_pad, n_levels=4)

    print(f"\nBoundary specs for {view_tag}:")
    for lvl, sp in boundary_specs.items():
        print(f"  Level {lvl} (scale {2**lvl}): {len(sp)} copy operations")

    # ---- load data ----
    X_train = np.load(train_x_path, mmap_mode="r")
    Y_train = np.load(train_y_path, mmap_mode="r")

    c_in        = int(X_train.shape[1])
    c_out_total = int(Y_train.shape[1])
    y_indices   = parse_channels(args.y_channels, c_out_total)
    c_out       = len(y_indices)

    print(f"\n=== Training Boundary-Aware UNet on {view_tag} ===")
    print(f"  Train X: {train_x_path}  {X_train.shape}")
    print(f"  Train Y: {train_y_path}  {Y_train.shape}")
    print(f"  c_in={c_in}  c_out={c_out}  y_indices={y_indices}")

    # ---- normalisation statistics ----
    print("Computing X normalisation stats …")
    x_mean, x_std = masked_channel_stats_x(X_train, mask_ch=0)

    mask_train = X_train[:, 0:1, :, :].astype(np.float32)
    print("Computing Y normalisation stats in transformed space (per-category eps) …")
    y_mean, y_std, s_c, y_min_t, y_max_t = masked_mean_std_transformed_y(
        Y_train, mask_train, y_indices=y_indices,
    )

    print(f"  y_min_t: {y_min_t}")
    print(f"  y_max_t: {y_max_t}")

    # ---- per-channel eps array (for checkpoint) ----
    eps_per_channel = np.array(
        [eps_for_channel(c) if c in POS_CHANNELS else 0.0 for c in y_indices],
        dtype=np.float32
    )

    # ---- datasets ----
    ds_kwargs = dict(
        y_indices=y_indices, x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std, s_c=s_c,
    )
    ds_train = ViewTensorDataset(train_x_path, train_y_path, **ds_kwargs)
    ds_test  = ViewTensorDataset(test_x_path,  test_y_path,  **ds_kwargs)

    device = torch.device(args.device)
    pin    = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)

    # ---- model + optimiser ----
    model = BoundaryAwareUNet(
        c_in=c_in, c_out=c_out, base=args.base,
        boundary_specs_per_level=boundary_specs,
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # ---- save config ----
    config = dict(
        view=args.view, view_tag=view_tag,
        tensor_prefix=args.tensor_prefix, train_split=args.train_split,
        test_prefix=args.test_prefix,     test_split=args.test_split,
        layout_npz=args.layout_npz,
        run_dir=str(run_dir),
        y_channels=args.y_channels, y_indices=y_indices,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        base=args.base, seed=args.seed, device=args.device,
        k_pad=k_pad, n_boundary_specs={k: len(v) for k, v in boundary_specs.items()},
    )
    save_json(run_dir / "config.json", config)

    # ---- training loop ----
    best_rmse    = float("inf")
    metrics_hist = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_steps = 0.0, 0

        for x, y, m in dl_train:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred     = model(x) * m
            loss     = channel_balanced_loss(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_steps  += 1

        train_loss = loss_sum / max(n_steps, 1)
        train_eval = evaluate(model, dl_train, device)
        test_eval  = evaluate(model, dl_test,  device)

        metrics_hist["train"].append({"epoch": epoch, "loss": train_loss, **train_eval})
        metrics_hist["test"].append( {"epoch": epoch, **test_eval})

        cur_rmse = float(test_eval["rmse_avg"])

        # -- serialisable boundary specs (list of lists) --
        bspecs_serial = {str(k): [list(s) for s in v] for k, v in boundary_specs.items()}

        ckpt = {
            "arch":                   "boundary_aware_unet",
            "view":                   args.view,
            "view_tag":               view_tag,
            "base":                   int(args.base),
            "c_in":                   int(c_in),
            "c_out":                  int(c_out),
            "c_out_total":            int(c_out_total),
            "y_indices":              [int(c) for c in y_indices],
            "pos_channels":           sorted(POS_CHANNELS),
            "signed_channels":        sorted(SIGNED_CHANNELS),
            "eps_per_channel":        eps_per_channel.tolist(),
            "x_mean":                 x_mean.tolist(),
            "x_std":                  x_std.tolist(),
            "y_mean":                 y_mean.tolist(),
            "y_std":                  y_std.tolist(),
            "s_c":                    s_c.tolist(),
            "y_min_t":               y_min_t.tolist(),
            "y_max_t":               y_max_t.tolist(),
            "boundary_specs":         bspecs_serial,
            "k_pad":                  k_pad,
            "model_state":            model.state_dict(),
            "opt_state":              opt.state_dict(),
            "epoch":                  int(epoch),
            "metrics_last":           {"train": train_eval, "test": test_eval,
                                       "train_loss": train_loss},
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        save_json(run_dir / "metrics.json", metrics_hist)
        print(
            f"[{view_tag}] Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4g} | "
            f"test_mae={test_eval['mae_avg']:.4g} | "
            f"test_rmse={test_eval['rmse_avg']:.4g}"
        )

    print(f"\nDone. Best test rmse_avg={best_rmse:.6g}  →  {run_dir}/checkpoint_best.pt")


if __name__ == "__main__":
    main()
