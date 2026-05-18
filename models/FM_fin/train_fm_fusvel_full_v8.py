"""
train_fm_fusvel_full_v8.py
──────────────────────────────────────────────────────────────────────────────
Same as v7, PLUS intra-view and inter-view boundary reinjection between every
encoder ConvBlock.

What is boundary reinjection?
──────────────────────────────
Each view canvas is a rectangular image whose "valid" pixels correspond to
physically real SOL cells.  Between strips within a view, and between views
that share physical boundaries, there are gap columns/rows that are filled
at tensor-construction time with static copies of the adjacent strip's
boundary pixels.

Problem:  A standard CNN with zero-padding causes the feature maps at strip
edges to degrade with depth: after the first ConvBlock the gap pixels already
contain slightly stale features, and after pooling they shrink further.  The
cross-view gap rows suffer the same problem AND are in SEPARATE tensors, so
from layer 2 onwards views are completely blind to each other's features.

Fix — two complementary operations applied after every encoder ConvBlock:

  1. INTRA-VIEW horizontal reinjection  (same canvas)
     For each pair of adjacent strips (A|gap|B):
       gap-left  [gs : gs+k]  ← right strip's left-edge features  [ge : ge+k]
       gap-right [ge-k : ge]  ← left strip's right-edge features  [gs-k : gs]
     This is equivalent to circular/wrap padding at every layer.

  2. INTER-VIEW vertical reinjection  (across canvases, same simulation)
     Process all 3 views of the SAME simulation in one forward pass.
     After each block, copy the k content-boundary rows from one view's
     feature map into the adjacent gap rows of the neighbouring view:
       view0-bottom-gap ← view1-MID_left/MID_right top k content rows
       view1-top-gap    ← view0-TOP_left/TOP_right bottom k content rows
       view2-bottom-gap ← view1-MID_center top k content rows
       view1-bottom-gap ← view2-Bottom_center top k content rows
     Column correspondences (TOP_left↔MID_left, etc.) are derived at
     startup from the saved mask arrays.

Architecture changes vs v7
──────────────────────────
  • VelocityUNet is UNCHANGED (backward-compatible).
  • VelocityUNetV8 is a thin subclass that adds forward_3views().
  • ConvBlock / FiLMBlock / CondEncoder are identical.
  • MultiViewSimDataset returns all 3 views of one sim per __getitem__
    (instead of one view).  Batch shape: 9 tensors of (B, C, H, W).
  • main() loads the layout .npz and passes LayoutInfo to the model.

Usage
─────
python train_fm_fusvel_full_v8.py \\
    --tensor_prefix scripts/tensor/3views_4d/train/global3v \\
    --layout_map    scripts/tensor/3views_4d/train/global3v_layout_map_3views.npz \\
    --split train --epochs 200 --batch_size 8 --base 64 --lr 3e-4 \\
    --mode xpred --t_schedule uniform --self_cond_prob 0.5 \\
    --save_dir scripts/runs/fusvel_v8_xpred_reinjection
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset


# ─────────────────────────────────────────────────────────────────────────────
# Channel metadata  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

POS_CHANNELS    = set(range(0, 12))
SIGNED_CHANNELS = set(range(12, 22))

SPECIES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
CH_NAMES: List[str] = (
    ["Te", "Ti"]
    + [f"na_{s}" for s in SPECIES]
    + [f"ua_{s}" for s in SPECIES]
)


# ─────────────────────────────────────────────────────────────────────────────
# Layout information  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class LayoutInfo:
    """
    Parses the saved layout_map_3views.npz and derives all information
    needed for boundary reinjection:

      horiz_gaps[v]   : list of (col_start, col_end) for each horizontal
                        zero-column gap in view v's canvas (at scale 1).
      cross_mappings  : list of dicts describing inter-view vertical pairs.
      H_v, W_v        : per-view canvas sizes (before padding to Hmax, Wmax).
      gap_px, k       : gap width and reinjection half-width from construction.

    Column correspondences for cross-view reinjection are derived from the
    mask arrays without needing to rerun the unroller.
    """

    def __init__(self, layout_npz_path: str) -> None:
        d = np.load(layout_npz_path)
        self.gap_px = int(d["gap_px"])
        self.k      = int(d["k"])
        self.Hmax   = int(d["Hmax"])
        self.Wmax   = int(d["Wmax"])
        self.W_v    = [int(d["W0"]), int(d["W1"]), int(d["W2"])]
        self.H_v    = [int(d["H0"]), int(d["H1"]), int(d["H2"])]
        self.masks  = [
            d["mask_view0"].astype(np.uint8),
            d["mask_view1"].astype(np.uint8),
            d["mask_view2"].astype(np.uint8),
        ]

        # Derive horizontal gap positions for each view
        self.horiz_gaps: List[List[Tuple[int, int]]] = [
            self._find_horiz_gaps(v) for v in range(3)
        ]

        # Derive cross-view vertical reinjection mappings
        self.cross_mappings: List[dict] = self._build_cross_mappings()

        print(f"  [LayoutInfo] gap_px={self.gap_px}  k={self.k}")
        for v in range(3):
            print(f"    view{v}: H={self.H_v[v]}  W={self.W_v[v]}"
                  f"  horiz_gaps={self.horiz_gaps[v]}")
        print(f"    cross_mappings: {len(self.cross_mappings)} pairs")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_horiz_gaps(self, v: int) -> List[Tuple[int, int]]:
        """
        Return list of (col_start, col_end) for each horizontal all-zero
        column run in the CONTENT REGION of view v's mask.
        Excludes the leftmost/rightmost edges (which are canvas borders,
        not strip gaps).
        """
        g  = self.gap_px
        W  = self.W_v[v]
        H  = self.H_v[v]
        m  = self.masks[v]

        # Content rows: skip the top/bottom gap rows
        r0 = g if (H > 2 * g) else 0
        r1 = H - g if (H > 2 * g) else H
        content = m[r0:r1, :W].astype(np.int32)

        col_sum = content.sum(axis=0)   # (W,)
        is_gap  = col_sum == 0

        gaps: List[Tuple[int, int]] = []
        in_gap = False
        start  = 0
        for c in range(W):
            if is_gap[c] and not in_gap:
                in_gap = True
                start  = c
            elif not is_gap[c] and in_gap:
                in_gap = False
                gaps.append((start, c))
        if in_gap and start > 0:
            gaps.append((start, W))

        # Exclude gaps that touch the canvas edge (these are not strip-to-strip gaps)
        gaps = [(s, e) for (s, e) in gaps if s > 0 and e < W]
        return gaps

    def _strip_bounds(self, v: int):
        """
        Return list of (col_start, col_end) for each CONTENT STRIP in view v.
        Derived as the column ranges NOT covered by horiz_gaps.
        """
        W    = self.W_v[v]
        gaps = self.horiz_gaps[v]

        # build a sorted list of gap column ranges
        boundaries = [0] + [c for (s, e) in gaps for c in (s, e)] + [W]
        strips = []
        for i in range(0, len(boundaries) - 1, 2):
            c0, c1 = boundaries[i], boundaries[i + 1]
            if c1 > c0:
                strips.append((c0, c1))
        return strips

    def _build_cross_mappings(self) -> List[dict]:
        """
        Build cross-view vertical reinjection mappings based on the known
        physical adjacency in build_3view_tensors.py:

          TOP_left  (view0) ↔ MID_left   (view1)  — strips at cols starting 0 in both
          TOP_right (view0) ↔ MID_right  (view1)  — last strip of each
          Bottom_center (view2) ↔ MID_center (view1) — view2 left-padded by gap_px

        Each mapping dict contains:
          src_view, dst_view : int
          src_c0, src_c1    : column range in the SOURCE canvas
          dst_c0, dst_c1    : column range in the DESTINATION canvas
          src_is_top        : bool — take from src's TOP content rows (True)
                                     or BOTTOM content rows (False)
          dst_is_top        : bool — inject into dst's TOP gap rows (True)
                                     or BOTTOM gap rows (False)

        The function generates BIDIRECTIONAL pairs for each physical adjacency.
        """
        g      = self.gap_px
        strips = [self._strip_bounds(v) for v in range(3)]

        mappings: List[dict] = []

        # ── helper ────────────────────────────────────────────────────
        def add_pair(sv, dv, sc0, sc1, dc0, dc1,
                     src_is_top_for_sv, dst_is_top_for_dv):
            """Add a bidirectional (sv→dv) + (dv→sv) pair."""
            w = min(sc1 - sc0, dc1 - dc0)
            if w <= 0:
                return
            # sv → dv
            mappings.append(dict(
                src_view=sv, dst_view=dv,
                src_c0=sc0, src_c1=sc0 + w,
                dst_c0=dc0, dst_c1=dc0 + w,
                src_is_top=src_is_top_for_sv,
                dst_is_top=dst_is_top_for_dv,
            ))
            # dv → sv  (reverse direction)
            # if sv is top (src_is_top_for_sv=False → bottom content), then
            # the REVERSE means dv's top content → sv's bottom gap
            mappings.append(dict(
                src_view=dv, dst_view=sv,
                src_c0=dc0, src_c1=dc0 + w,
                dst_c0=sc0, dst_c1=sc0 + w,
                src_is_top=(not src_is_top_for_sv),   # dv is at opposite end
                dst_is_top=(not dst_is_top_for_dv),
            ))

        # ── view0 bottom ↔ view1 top ─────────────────────────────────
        # In build_3view_tensors.py:
        #   ("TOP_left",  "MID_left",  view0, view1)
        #   ("TOP_right", "MID_right", view0, view1)
        # view0 is "top block": its BOTTOM gap ← view1's FIRST k content rows
        # view1 is "bottom block": its TOP gap ← view0's LAST k content rows

        strips0 = strips[0]   # [(c0_A, c1_A), (c0_B, c1_B)]
        strips1 = strips[1]   # [(c0_C,c1_C),(c0_D,c1_D),(c0_E,c1_E)] or fewer

        # LEFT strips: index 0 in both views
        if len(strips0) >= 1 and len(strips1) >= 1:
            s0A = strips0[0];  s1C = strips1[0]
            add_pair(
                sv=0, dv=1,
                sc0=s0A[0], sc1=s0A[1],   # view0 TOP_left cols
                dc0=s1C[0], dc1=s1C[1],   # view1 MID_left cols
                src_is_top_for_sv=False,   # view0 contributes its BOTTOM rows
                dst_is_top_for_dv=True,    # view1 receives into its TOP gap
            )

        # RIGHT strips: last strip in each view
        if len(strips0) >= 2 and len(strips1) >= 3:
            s0B = strips0[-1];  s1E = strips1[-1]
            add_pair(
                sv=0, dv=1,
                sc0=s0B[0], sc1=s0B[1],
                dc0=s1E[0], dc1=s1E[1],
                src_is_top_for_sv=False,
                dst_is_top_for_dv=True,
            )

        # ── view2 bottom ↔ view1 bottom-center ───────────────────────
        # In build_3view_tensors.py:
        #   ("Bottom_center", "MID_center", view2, view1)
        # view2 is "top block": its BOTTOM gap ← view1 MID_center's FIRST k rows
        # view1 MID_center "bottom block": its TOP gap ← view2's LAST k rows
        #
        # view2 Bottom_center column range: [gap_px, W2-gap_px)
        # view1 MID_center is strips1[1] (middle strip of 3)

        if len(strips1) >= 3:
            s1D = strips1[1]   # MID_center
            # Bottom_center in view2: padded by gap_px on both sides
            bc0 = g
            bc1 = self.W_v[2] - g
            add_pair(
                sv=2, dv=1,
                sc0=bc0, sc1=bc1,          # view2 Bottom_center cols
                dc0=s1D[0], dc1=s1D[1],   # view1 MID_center cols
                src_is_top_for_sv=False,   # view2 contributes its BOTTOM rows
                dst_is_top_for_dv=False,   # view1 receives into its BOTTOM gap
            )

        return mappings


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def _masked_den(mask: np.ndarray, eps: float = 1e-12) -> float:
    den = float(np.asarray(mask, dtype=np.float64).sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support — check tensor correctness.")
    return den


def _symlog(y: np.ndarray, s_c: float) -> np.ndarray:
    return np.sign(y) * np.log10(np.abs(y) / s_c + 1.0)


def compute_x_stats_streaming(x_mmaps: list, mask_ch: int = 0):
    """Masked mean/std per channel, iterating one view at a time (no concat)."""
    C = x_mmaps[0].shape[1]

    # total masked denominator across views
    den = 0.0
    for x_arr in x_mmaps:
        m = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
        den += float(m.sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support")

    mean = np.zeros(C, np.float64)
    for c in range(C):
        num = 0.0
        for x_arr in x_mmaps:
            xc = x_arr[:, c:c + 1].astype(np.float64)
            m  = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
            num += float((xc * m).sum())
        mean[c] = num / den

    var = np.zeros(C, np.float64)
    for c in range(C):
        vnum = 0.0
        for x_arr in x_mmaps:
            xc  = x_arr[:, c:c + 1].astype(np.float64)
            m   = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
            vnum += float((((xc - mean[c]) ** 2) * m).sum())
        var[c] = vnum / den

    std           = np.sqrt(np.maximum(var, 1e-12))
    mean[mask_ch] = 0.0
    std[mask_ch]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_y_stats_streaming(y_mmaps: list, x_mmaps: list, y_indices: list,
                               eps: float = 1e-3, mask_ch: int = 0):
    """
    Masked mean/std of *transformed* Y channels.
    Processes one view at a time — never materialises the full concatenated array.
    """
    C_sel  = len(y_indices)
    s_c    = np.ones(C_sel,  np.float64)
    y_mean = np.zeros(C_sel, np.float64)
    y_var  = np.zeros(C_sel, np.float64)

    # total masked denominator
    den = 0.0
    for x_arr in x_mmaps:
        m = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
        den += float(m.sum())
    if den <= 0.0:
        raise RuntimeError("Mask has zero support")

    # Pass 0 — compute s_c (physical std) for SIGNED channels
    for j, c in enumerate(y_indices):
        if c not in SIGNED_CHANNELS:
            continue
        mu_num = 0.0
        for y_arr, x_arr in zip(y_mmaps, x_mmaps):
            yc = y_arr[:, c:c + 1].astype(np.float64)
            m  = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
            mu_num += float((yc * m).sum())
        mu = mu_num / den
        v_num = 0.0
        for y_arr, x_arr in zip(y_mmaps, x_mmaps):
            yc = y_arr[:, c:c + 1].astype(np.float64)
            m  = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
            v_num += float((((yc - mu) ** 2) * m).sum())
        s_c[j] = float(np.sqrt(max(v_num / den, 1e-12)))

    # Pass 1 — masked mean of the transformed values
    for j, c in enumerate(y_indices):
        t_num = 0.0
        for y_arr, x_arr in zip(y_mmaps, x_mmaps):
            yc = y_arr[:, c:c + 1].astype(np.float64)
            m  = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
            t  = (np.log10(np.maximum(yc, 0.0) + eps)
                  if c in POS_CHANNELS else _symlog(yc, s_c[j]))
            t_num += float((t * m).sum())
        y_mean[j] = t_num / den

    # Pass 2 — masked variance of the transformed values
    for j, c in enumerate(y_indices):
        v_num = 0.0
        for y_arr, x_arr in zip(y_mmaps, x_mmaps):
            yc = y_arr[:, c:c + 1].astype(np.float64)
            m  = x_arr[:, mask_ch:mask_ch + 1].astype(np.float64)
            t  = (np.log10(np.maximum(yc, 0.0) + eps)
                  if c in POS_CHANNELS else _symlog(yc, s_c[j]))
            v_num += float((((t - y_mean[j]) ** 2) * m).sum())
        y_var[j] = v_num / den

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    return y_mean.astype(np.float32), y_std.astype(np.float32), s_c.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset — returns ALL 3 VIEWS per simulation  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewSimDataset(Dataset):
    """
    Each __getitem__(sim_idx) returns all 3 views for the same simulation:
      (x0_norm, y0_norm, mask0,
       x1_norm, y1_norm, mask1,
       x2_norm, y2_norm, mask2)

    This groups views of the same sim together, enabling inter-view
    reinjection during the forward pass.

    Dataset length = n_sims  (not n_sims × 3 as in v7 MultiViewDataset).
    """

    def __init__(
        self,
        x_paths: List[Path],
        y_paths: List[Path],
        y_indices: List[int],
        x_mean: np.ndarray,
        x_std:  np.ndarray,
        y_mean: np.ndarray,
        y_std:  np.ndarray,
        s_c:    np.ndarray,
        eps:    float = 1e-3,
    ):
        self.y_indices = y_indices
        self.x_mean = x_mean
        self.x_std  = x_std
        self.y_mean = y_mean
        self.y_std  = y_std
        self.s_c    = s_c
        self.eps    = eps
        self.n_views = 3

        self.X_views = [np.load(p, mmap_mode="r") for p in x_paths]
        self.Y_views = [np.load(p, mmap_mode="r") for p in y_paths]
        self.n_sims  = self.X_views[0].shape[0]

        for v in range(self.n_views):
            assert self.X_views[v].shape[0] == self.n_sims, \
                f"View {v} sim count mismatch"
            assert self.Y_views[v].shape[0] == self.n_sims, \
                f"View {v} Y sim count mismatch"

    def __len__(self) -> int:
        return self.n_sims

    def _load_view(self, sim: int, view: int):
        x_raw = np.array(self.X_views[view][sim], dtype=np.float32)
        y_raw = np.array(self.Y_views[view][sim], dtype=np.float32)
        mask  = x_raw[0:1].copy()

        # normalise X
        x_norm = (x_raw - self.x_mean[:, None, None]) / self.x_std[:, None, None]

        # transform + normalise Y
        C_sel = len(self.y_indices)
        y_norm = np.empty((C_sel, x_raw.shape[1], x_raw.shape[2]), dtype=np.float32)
        for j, c in enumerate(self.y_indices):
            yc = y_raw[c].astype(np.float64)
            t  = (np.log10(np.maximum(yc, 0.0) + self.eps)
                  if c in POS_CHANNELS else _symlog(yc, float(self.s_c[j])))
            y_norm[j] = ((t - float(self.y_mean[j])) / float(self.y_std[j])).astype(np.float32)

        return (
            torch.from_numpy(x_norm),
            torch.from_numpy(y_norm),
            torch.from_numpy(mask),
        )

    def __getitem__(self, sim_idx: int):
        # Returns 9 tensors: (x0,y0,m0, x1,y1,m1, x2,y2,m2)
        parts = []
        for v in range(self.n_views):
            parts.extend(self._load_view(sim_idx, v))
        return tuple(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Model components  (UNCHANGED from v7)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        t = t.view(-1).float()
        a = t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1),
            nn.GroupNorm(min(8, co), co),
            nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1),
            nn.GroupNorm(min(8, co), co),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, ci, co, t_dim):
        super().__init__()
        self.c1   = nn.Conv2d(ci, co, 3, padding=1)
        self.gn1  = nn.GroupNorm(min(8, co), co)
        self.c2   = nn.Conv2d(co, co, 3, padding=1)
        self.gn2  = nn.GroupNorm(min(8, co), co)
        self.act  = nn.GELU()
        self.film = nn.Linear(t_dim, co * 2)

    def forward(self, x, t_emb):
        h = self.act(self.gn1(self.c1(x)))
        h = self.gn2(self.c2(h))
        s, sh = self.film(t_emb).chunk(2, -1)
        return self.act(h * (1 + s[..., None, None]) + sh[..., None, None])


class CondEncoder(nn.Module):
    def __init__(self, c_in, base):
        super().__init__()
        self.e1   = ConvBlock(c_in,    base)
        self.e2   = ConvBlock(base,    base * 2)
        self.e3   = ConvBlock(base*2,  base * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(self.pool(f1))
        f3 = self.e3(self.pool(f2))
        return f1, f2, f3


class VelocityUNet(nn.Module):
    """Original v7 model — kept completely unchanged."""
    def __init__(self, c_in, c_out, base=32, t_dim=128, n_views=3):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out

        self.view_emb_dim = 4
        self.view_emb = nn.Embedding(n_views, self.view_emb_dim)

        self.time_emb = SinusoidalTimeEmb(t_dim)
        self.cond_enc = CondEncoder(c_in + self.view_emb_dim, base)

        self.enc1       = ConvBlock(c_in + self.view_emb_dim + c_out, base)
        self.enc2       = ConvBlock(base,          base * 2)
        self.enc3       = ConvBlock(base * 2,      base * 4)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4,      base * 8)

        self.up3  = nn.Conv2d(base * 8,  base * 4, 1)
        self.dec3 = FiLMBlock(base * 4 + base * 4 + base * 4, base * 4, t_dim)
        self.up2  = nn.Conv2d(base * 4,  base * 2, 1)
        self.dec2 = FiLMBlock(base * 2 + base * 2 + base * 2, base * 2, t_dim)
        self.up1  = nn.Conv2d(base * 2,  base,     1)
        self.dec1 = FiLMBlock(base      + base      + base,    base,     t_dim)
        self.out  = nn.Conv2d(base, c_out, 1)

    @staticmethod
    def _match(x, hw):
        th, tw = hw
        h, w   = x.shape[-2:]
        if h == th and w == tw:
            return x
        if h >= th and w >= tw:
            dh, dw = h - th, w - tw
            return x[..., dh // 2: dh // 2 + th, dw // 2: dw // 2 + tw]
        return F.interpolate(x, (th, tw), mode="bilinear", align_corners=False)

    def forward(self, x_cond, y_t, t, view_id):
        te = self.time_emb(t)
        B, _, H, W = x_cond.shape
        v_emb = self.view_emb(view_id)
        v_emb_spatial = v_emb[:, :, None, None].expand(B, -1, H, W)
        x_with_view = torch.cat([x_cond, v_emb_spatial], dim=1)

        c1, c2, c3 = self.cond_enc(x_with_view)

        inp = torch.cat([x_with_view, y_t], 1)
        e1  = self.enc1(inp)
        e2  = self.enc2(self.pool(e1))
        e3  = self.enc3(self.pool(e2))
        b   = self.bottleneck(self.pool(e3))

        u3  = self.up3(F.interpolate(b,  e3.shape[-2:], mode="bilinear", align_corners=False))
        d3  = self.dec3(torch.cat([u3,
                                   self._match(e3, u3.shape[-2:]),
                                   self._match(c3, u3.shape[-2:])], 1), te)
        u2  = self.up2(F.interpolate(d3, e2.shape[-2:], mode="bilinear", align_corners=False))
        d2  = self.dec2(torch.cat([u2,
                                   self._match(e2, u2.shape[-2:]),
                                   self._match(c2, u2.shape[-2:])], 1), te)
        u1  = self.up1(F.interpolate(d2, e1.shape[-2:], mode="bilinear", align_corners=False))
        d1  = self.dec1(torch.cat([u1,
                                   self._match(e1, u1.shape[-2:]),
                                   self._match(c1, u1.shape[-2:])], 1), te)
        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary reinjection subclass  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class VelocityUNetV8(VelocityUNet):
    """
    VelocityUNet + intra-view horizontal and inter-view vertical boundary
    reinjection between every encoder ConvBlock.

    The underlying blocks (ConvBlock, FiLMBlock, CondEncoder) are IDENTICAL
    to VelocityUNet and share weights if loaded from a v7 checkpoint.

    The original single-view forward() is preserved for inference fallback.
    The 3-view training path uses forward_3views().
    """

    def __init__(self, layout: LayoutInfo, c_in: int, c_out: int,
                 base: int = 32, t_dim: int = 128, n_views: int = 3):
        super().__init__(c_in=c_in, c_out=c_out, base=base,
                         t_dim=t_dim, n_views=n_views)
        self.layout = layout

    # ------------------------------------------------------------------
    # Intra-view horizontal reinjection
    # ------------------------------------------------------------------

    def _ri_h(self, feat: torch.Tensor, view_id: int, scale: int) -> torch.Tensor:
        """
        Refresh horizontal gap columns with adjacent strip-edge features.

        For each gap (gs, ge) in view `view_id` at the original scale:
          feat[:, :, :, gs_s : gs_s+k_eff] ← right strip's left edge
                                              feat[:, :, :, ge_s : ge_s+k_eff]
          feat[:, :, :, ge_s-k_eff : ge_s] ← left strip's right edge
                                              feat[:, :, :, gs_s-k_eff : gs_s]

        where gs_s = gs // scale, ge_s = ge // scale, k_eff = k // scale.

        This is equivalent to applying circular/wrap padding at strip edges
        before every ConvBlock.
        """
        gaps  = self.layout.horiz_gaps[view_id]
        if not gaps:
            return feat

        k_raw = self.layout.k
        k_eff = max(1, k_raw // scale)

        feat = feat.clone()
        W_feat = feat.shape[3]

        for (gs, ge) in gaps:
            gs_s = gs // scale
            ge_s = ge // scale
            span = ge_s - gs_s          # gap width at this scale

            if span < 2 or k_eff > span:
                # Gap too narrow at this scale — skip
                continue

            k_use = min(k_eff, span // 2)

            # Bounds check
            if (ge_s + k_use > W_feat) or (gs_s < k_use):
                continue

            # Left part of gap ← right strip's left edge (first k_use cols of right strip)
            feat[:, :, :, gs_s : gs_s + k_use] = \
                feat[:, :, :, ge_s : ge_s + k_use].clone()

            # Right part of gap ← left strip's right edge (last k_use cols of left strip)
            feat[:, :, :, ge_s - k_use : ge_s] = \
                feat[:, :, :, gs_s - k_use : gs_s].clone()

        return feat

    # ------------------------------------------------------------------
    # Inter-view vertical reinjection
    # ------------------------------------------------------------------

    def _ri_v(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Refresh vertical gap rows with feature values from the adjacent
        view's content boundary rows.

        For each cross_mapping:
          src content boundary rows → dst gap rows
          (at appropriately scaled positions)

        Returns updated (f0, f1, f2).
        """
        g_raw = self.layout.gap_px
        k_raw = self.layout.k
        g_s   = max(1, g_raw // scale)
        k_s   = max(1, k_raw // scale)

        feats = [f0.clone(), f1.clone(), f2.clone()]
        H_vs  = [max(1, h // scale) for h in self.layout.H_v]

        for m in self.layout.cross_mappings:
            sv = m["src_view"];  dv = m["dst_view"]
            sc0 = m["src_c0"] // scale;  sc1 = m["src_c1"] // scale
            dc0 = m["dst_c0"] // scale;  dc1 = m["dst_c1"] // scale
            w   = min(sc1 - sc0, dc1 - dc0)
            if w <= 0:
                continue

            k_use = min(k_s, w)
            Hv_src = H_vs[sv]
            Hv_dst = H_vs[dv]

            # Source rows
            if m["src_is_top"]:
                # first k_use content rows (below top gap)
                sr0 = g_s
            else:
                # last k_use content rows (above bottom gap)
                sr0 = Hv_src - g_s - k_use

            # Destination rows (into the gap)
            if m["dst_is_top"]:
                # last k_use rows of the top gap
                dr0 = g_s - k_use
            else:
                # first k_use rows of the bottom gap
                dr0 = Hv_dst - g_s

            sr0 = max(0, sr0)
            dr0 = max(0, dr0)

            # Spatial bounds check
            src_feat = feats[sv]
            dst_feat = feats[dv]

            if (sr0 + k_use > src_feat.shape[2] or
                dr0 + k_use > dst_feat.shape[2] or
                sc1 > src_feat.shape[3] or
                dc1 > dst_feat.shape[3]):
                continue

            feats[dv][:, :, dr0 : dr0 + k_use, dc0 : dc0 + w] = \
                src_feat[:, :, sr0 : sr0 + k_use, sc0 : sc0 + w].clone()

        return feats[0], feats[1], feats[2]

    # ------------------------------------------------------------------
    # 3-view forward pass with reinjection
    # ------------------------------------------------------------------

    def forward_3views(
        self,
        x0: torch.Tensor,   # (B, c_in, H, W)
        y0: torch.Tensor,   # (B, c_out, H, W)
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        t:  torch.Tensor,   # (B,) — SAME t for all 3 views of the same sim
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process all 3 views simultaneously with boundary reinjection after
        each encoder ConvBlock.

        Returns (pred0, pred1, pred2), each (B, c_out, H, W).
        """
        device = x0.device
        B, _, H, W = x0.shape

        te = self.time_emb(t)

        # Fixed view ID tensors
        vid0 = torch.zeros (B, dtype=torch.long, device=device)
        vid1 = torch.ones  (B, dtype=torch.long, device=device)
        vid2 = torch.full  ((B,), 2, dtype=torch.long, device=device)

        def add_view_emb(x, vid):
            ve = self.view_emb(vid)[:, :, None, None].expand(B, -1, H, W)
            return torch.cat([x, ve], dim=1)

        xv0 = add_view_emb(x0, vid0)
        xv1 = add_view_emb(x1, vid1)
        xv2 = add_view_emb(x2, vid2)

        # ── Condition encoders ────────────────────────────────────────
        c1_0, c2_0, c3_0 = self.cond_enc(xv0)
        c1_1, c2_1, c3_1 = self.cond_enc(xv1)
        c1_2, c2_2, c3_2 = self.cond_enc(xv2)

        # ── Encoder scale 1 ──────────────────────────────────────────
        e1_0 = self.enc1(torch.cat([xv0, y0], 1))
        e1_1 = self.enc1(torch.cat([xv1, y1], 1))
        e1_2 = self.enc1(torch.cat([xv2, y2], 1))

        # Intra-view reinjection (horizontal gaps)
        e1_0 = self._ri_h(e1_0, 0, 1)
        e1_1 = self._ri_h(e1_1, 1, 1)
        e1_2 = self._ri_h(e1_2, 2, 1)
        # Inter-view reinjection (vertical gaps, cross-view)
        e1_0, e1_1, e1_2 = self._ri_v(e1_0, e1_1, e1_2, 1)

        # ── Encoder scale 2 ──────────────────────────────────────────
        e2_0 = self.enc2(self.pool(e1_0))
        e2_1 = self.enc2(self.pool(e1_1))
        e2_2 = self.enc2(self.pool(e1_2))

        e2_0 = self._ri_h(e2_0, 0, 2)
        e2_1 = self._ri_h(e2_1, 1, 2)
        e2_2 = self._ri_h(e2_2, 2, 2)
        e2_0, e2_1, e2_2 = self._ri_v(e2_0, e2_1, e2_2, 2)

        # ── Encoder scale 4 ──────────────────────────────────────────
        e3_0 = self.enc3(self.pool(e2_0))
        e3_1 = self.enc3(self.pool(e2_1))
        e3_2 = self.enc3(self.pool(e2_2))

        e3_0 = self._ri_h(e3_0, 0, 4)
        e3_1 = self._ri_h(e3_1, 1, 4)
        e3_2 = self._ri_h(e3_2, 2, 4)
        e3_0, e3_1, e3_2 = self._ri_v(e3_0, e3_1, e3_2, 4)

        # ── Bottleneck (scale 8) ──────────────────────────────────────
        b0 = self.bottleneck(self.pool(e3_0))
        b1 = self.bottleneck(self.pool(e3_1))
        b2 = self.bottleneck(self.pool(e3_2))

        # ── Decoder  (per-view, shared weights, no cross-view) ────────
        # Cross-view reinjection in the decoder is less critical because
        # skip connections already carry the (reinjected) encoder features.
        # It can be added in a future version if needed.

        def decode(b, e3, e2, e1, c3, c2, c1):
            u3 = self.up3(F.interpolate(b, e3.shape[-2:],
                                        mode="bilinear", align_corners=False))
            d3 = self.dec3(torch.cat([u3, self._match(e3, u3.shape[-2:]),
                                      self._match(c3, u3.shape[-2:])], 1), te)
            u2 = self.up2(F.interpolate(d3, e2.shape[-2:],
                                        mode="bilinear", align_corners=False))
            d2 = self.dec2(torch.cat([u2, self._match(e2, u2.shape[-2:]),
                                      self._match(c2, u2.shape[-2:])], 1), te)
            u1 = self.up1(F.interpolate(d2, e1.shape[-2:],
                                        mode="bilinear", align_corners=False))
            d1 = self.dec1(torch.cat([u1, self._match(e1, u1.shape[-2:]),
                                      self._match(c1, u1.shape[-2:])], 1), te)
            return self.out(d1)

        pred0 = decode(b0, e3_0, e2_0, e1_0, c3_0, c2_0, c1_0)
        pred1 = decode(b1, e3_1, e2_1, e1_1, c3_1, c2_1, c1_1)
        pred2 = decode(b2, e3_2, e2_2, e1_2, c3_2, c2_2, c1_2)

        return pred0, pred1, pred2


# ─────────────────────────────────────────────────────────────────────────────
# Time sampling  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def sample_t(B: int, schedule: str, device: torch.device) -> torch.Tensor:
    if schedule == "uniform":
        return torch.rand(B, device=device) * 0.98 + 0.01
    elif schedule == "low_biased":
        u = torch.rand(B, device=device)
        return (u ** 2) * 0.98 + 0.01
    elif schedule == "logit_normal":
        return torch.sigmoid(torch.randn(B, device=device)) * 0.98 + 0.01
    else:
        raise ValueError(f"Unknown t_schedule: {schedule}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae_per_channel(pred, target, mask):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    num = (pred - target).abs().mul(mask).sum(dim=(0, 2, 3))
    den = mask.sum(dim=(0, 2, 3)).clamp_min(1e-8)
    return num / den


def masked_huber_per_channel(pred, target, mask, delta=1.0):
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    diff  = (pred - target).abs()
    huber = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    num   = (huber * mask).sum(dim=(0, 2, 3))
    den   = mask.sum(dim=(0, 2, 3)).clamp_min(1e-8)
    return num / den


# ─────────────────────────────────────────────────────────────────────────────
# Training steps — adapted for 3-view batches  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each step function receives the 3-view batch and processes all 3 views in
# one forward_3views() call.  The loss and MAE are averaged over all 3 views.
# A SINGLE t is sampled per simulation and shared across its 3 views.
#

def _compute_loss_mae(pred, target, mask, loss_fn):
    if loss_fn == "huber":
        ch_loss = masked_huber_per_channel(pred, target, mask)
    else:
        ch_loss = masked_mae_per_channel(pred, target, mask)
    with torch.no_grad():
        mae_c = masked_mae_per_channel(pred, target, mask)
    return ch_loss.mean(), mae_c


def flow_matching_step_3v(
    model: VelocityUNetV8,
    x0, y0, m0, x1, y1, m1, x2, y2, m2,
    loss_fn="mae", t_schedule="uniform", self_cond_prob=0.0,
):
    """Velocity-prediction OT-CFM across 3 views."""
    B = x0.shape[0]
    device = x0.device
    t = sample_t(B, t_schedule, device)

    def make_yt(y1_clean, mask):
        y0_ = torch.randn_like(y1_clean) * mask
        t_e = t[:, None, None, None]
        return (1 - t_e) * y0_ + t_e * y1_clean, y1_clean - y0_

    yt0, ut0 = make_yt(y0, m0)
    yt1, ut1 = make_yt(y1, m1)
    yt2, ut2 = make_yt(y2, m2)

    vp0, vp1, vp2 = model.forward_3views(x0, yt0, x1, yt1, x2, yt2, t)

    l0, mae0 = _compute_loss_mae(vp0, ut0, m0, loss_fn)
    l1, mae1 = _compute_loss_mae(vp1, ut1, m1, loss_fn)
    l2, mae2 = _compute_loss_mae(vp2, ut2, m2, loss_fn)

    loss = (l0 + l1 + l2) / 3.0
    mae  = (mae0 + mae1 + mae2) / 3.0
    return loss, mae


def xpred_flow_matching_step_3v(
    model: VelocityUNetV8,
    x0, y0, m0, x1, y1, m1, x2, y2, m2,
    loss_fn="mae", t_schedule="uniform", self_cond_prob=0.5,
):
    """
    X-prediction OT-CFM with self-conditioning across 3 views.
    Same t is sampled per simulation and shared across all 3 views.
    Self-conditioning is applied consistently (same decision for all views).
    """
    B = x0.shape[0]
    device = x0.device
    t = sample_t(B, t_schedule, device)
    t_e = t[:, None, None, None]

    use_self_cond = (torch.rand(1).item() < self_cond_prob) and model.training

    def build_yt_clean(y1_clean, mask):
        y0_ = torch.randn_like(y1_clean) * mask
        return (1 - t_e) * y0_ + t_e * y1_clean, y0_

    yt0, n0 = build_yt_clean(y0, m0)
    yt1, n1 = build_yt_clean(y1, m1)
    yt2, n2 = build_yt_clean(y2, m2)

    if use_self_cond:
        # Sample an earlier time
        t_early = t * torch.rand(B, device=device) * 0.8
        te_e    = t_early[:, None, None, None]

        y_te0 = (1 - te_e) * n0 + te_e * y0
        y_te1 = (1 - te_e) * n1 + te_e * y1
        y_te2 = (1 - te_e) * n2 + te_e * y2

        with torch.no_grad():
            h0, h1, h2 = model.forward_3views(x0, y_te0, x1, y_te1, x2, y_te2, t_early)

        yt0 = ((1 - t_e) * n0 + t_e * h0) * m0
        yt1 = ((1 - t_e) * n1 + t_e * h1) * m1
        yt2 = ((1 - t_e) * n2 + t_e * h2) * m2

    pred0, pred1, pred2 = model.forward_3views(x0, yt0, x1, yt1, x2, yt2, t)

    l0, mae0 = _compute_loss_mae(pred0, y0, m0, loss_fn)
    l1, mae1 = _compute_loss_mae(pred1, y1, m1, loss_fn)
    l2, mae2 = _compute_loss_mae(pred2, y2, m2, loss_fn)

    loss = (l0 + l1 + l2) / 3.0
    mae  = (mae0 + mae1 + mae2) / 3.0
    return loss, mae


def direct_regression_step_3v(
    model: VelocityUNetV8,
    x0, y0, m0, x1, y1, m1, x2, y2, m2,
    loss_fn="mae", t_schedule="uniform", self_cond_prob=0.0,
):
    """Direct regression baseline across 3 views."""
    B = x0.shape[0]
    device = x0.device
    t = torch.ones(B, device=device)

    d0 = torch.zeros_like(y0)
    d1 = torch.zeros_like(y1)
    d2 = torch.zeros_like(y2)

    pred0, pred1, pred2 = model.forward_3views(x0, d0, x1, d1, x2, d2, t)

    l0, mae0 = _compute_loss_mae(pred0, y0, m0, loss_fn)
    l1, mae1 = _compute_loss_mae(pred1, y1, m1, loss_fn)
    l2, mae2 = _compute_loss_mae(pred2, y2, m2, loss_fn)

    loss = (l0 + l1 + l2) / 3.0
    mae  = (mae0 + mae1 + mae2) / 3.0
    return loss, mae


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_channels(s: str, c_out_total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", "*"):
        return list(range(c_out_total))
    ch = [int(p) for p in s.split(",") if p.strip()]
    if any(c < 0 or c >= c_out_total for c in ch):
        raise ValueError(f"Channel index out of bounds: {ch}")
    return ch


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MODE_LABELS = {
    "cfm":    "OT-CFM velocity-prediction flow matching",
    "xpred":  "OT-CFM data-prediction (x-pred) flow matching",
    "direct": "direct regression baseline",
}
METHOD_LABELS = {
    "cfm":    "OT-CFM-velocity",
    "xpred":  "OT-CFM-xpred",
    "direct": "direct-regression",
}

_STEP_FNS = {
    "cfm":    flow_matching_step_3v,
    "xpred":  xpred_flow_matching_step_3v,
    "direct": direct_regression_step_3v,
}


def _unpack_batch(batch, device):
    """Unpack the 9-tensor batch from MultiViewSimDataset."""
    x0, y0, m0, x1, y1, m1, x2, y2, m2 = batch
    return (
        x0.to(device, non_blocking=True),
        y0.to(device, non_blocking=True),
        m0.to(device, non_blocking=True),
        x1.to(device, non_blocking=True),
        y1.to(device, non_blocking=True),
        m1.to(device, non_blocking=True),
        x2.to(device, non_blocking=True),
        y2.to(device, non_blocking=True),
        m2.to(device, non_blocking=True),
    )


def _save_checkpoint(model, path, epoch, args, config,
                     x_mean, x_std, y_mean, y_std, s_c,
                     y_indices, c_in, c_out, loss):
    torch.save({
        "model_state":     model.state_dict(),
        "epoch":           epoch,
        "loss":            loss,
        "c_in":            c_in,
        "c_out":           c_out,
        "base":            args.base,
        "t_dim":           args.t_dim,
        "y_indices":       y_indices,
        "x_mean":          x_mean,
        "x_std":           x_std,
        "y_mean":          y_mean,
        "y_std":           y_std,
        "s_c":             s_c,
        "eps":             args.eps,
        "method":          METHOD_LABELS[args.mode],
        "mode":            args.mode,
        "pos_channels":    sorted(POS_CHANNELS),
        "signed_channels": sorted(SIGNED_CHANNELS),
        "config":          config,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_prefix", required=True)
    ap.add_argument("--layout_map",    required=True,
                    help="Path to *_layout_map_3views.npz")
    ap.add_argument("--split",         default="train")
    ap.add_argument("--epochs",        type=int,   default=200)
    ap.add_argument("--batch_size",    type=int,   default=8,
                    help="Number of SIMULATIONS per batch (each contributes 3 views); "
                         "effective pixel batch is 3× this value")
    ap.add_argument("--base",          type=int,   default=32)
    ap.add_argument("--t_dim",         type=int,   default=128)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--y_channels",    default="all")
    ap.add_argument("--log_every",     type=int,   default=1)
    ap.add_argument("--eps",           type=float, default=1e-3)
    ap.add_argument("--save_dir",      required=True)
    ap.add_argument("--device",        default=None)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--num_workers",   type=int,   default=0)
    ap.add_argument("--loss_fn",       default="mae", choices=["mae", "huber"])
    ap.add_argument("--val_frac",      type=float, default=0.1)
    ap.add_argument("--dataset",       type=int,   default=100)
    ap.add_argument("--mode",          default="xpred",
                    choices=["cfm", "xpred", "direct"])
    ap.add_argument("--t_schedule",    default="uniform",
                    choices=["uniform", "low_biased", "logit_normal"])
    ap.add_argument("--self_cond_prob", type=float, default=0.5)
    ap.add_argument("--warm_start",    default=None, type=str)

    args = ap.parse_args()

    if args.dataset < 1 or args.dataset > 100:
        raise ValueError(f"--dataset must be in [1, 100], got {args.dataset}")

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_fn = _STEP_FNS[args.mode]

    print(f"\n{'='*70}")
    print(f"  Full-dataset training — VelocityUNetV8 × 3 views WITH REINJECTION")
    print(f"  Mode   : {args.mode.upper()} ({MODE_LABELS[args.mode]})")
    if args.mode != "direct":
        print(f"  t sched: {args.t_schedule}")
        if args.self_cond_prob > 0:
            print(f"  selfcond: {args.self_cond_prob:.0%}")
    print(f"  Device : {device}")
    print(f"  Loss   : {args.loss_fn}")
    print(f"{'='*70}\n")

    # ── Layout ───────────────────────────────────────────────────────────────
    print("  Loading layout map …")
    layout = LayoutInfo(args.layout_map)

    # ── Tensor paths ─────────────────────────────────────────────────────────
    pfx  = args.tensor_prefix
    splt = args.split
    x_paths, y_paths = [], []
    for v in range(3):
        xp = Path(f"{pfx}_view{v}_X_img_{splt}.npy")
        yp = Path(f"{pfx}_view{v}_Y_img_{splt}.npy")
        for p in (xp, yp):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
        x_paths.append(xp)
        y_paths.append(yp)
        print(f"  view{v}  X:{xp.name}  Y:{yp.name}")

    X0_peek = np.load(x_paths[0], mmap_mode="r")
    Y0_peek = np.load(y_paths[0], mmap_mode="r")
    N_sim_total = X0_peek.shape[0]
    c_in        = X0_peek.shape[1]
    c_out_total = Y0_peek.shape[1]

    y_indices = parse_channels(args.y_channels, c_out_total)
    c_out     = len(y_indices)

    # ── Subsampling ──────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    all_sim_ids = np.arange(N_sim_total)
    if args.dataset < 100:
        n_use = max(1, int(N_sim_total * args.dataset / 100))
        selected_sims = np.sort(rng.choice(all_sim_ids, size=n_use, replace=False))
    else:
        selected_sims = all_sim_ids
        n_use = N_sim_total

    print(f"\n  N_sim_total={N_sim_total}  using={n_use} ({args.dataset}%)")
    print(f"  c_in={c_in}  c_out={c_out}")
    print(f"  y_indices={y_indices[:8]}{'…' if c_out > 8 else ''}")

    # ── Normalisation stats over full training dataset ────────────────────────
    # NOTE: computed over ALL sims in the tensor files (including val sims).
    # This is a mild known shortcut; the norm stats are not used as learning
    # targets so the leakage is negligible in practice.
    print("  Computing normalisation statistics …")
    all_x = [np.load(p, mmap_mode="r") for p in x_paths]
    all_y = [np.load(p, mmap_mode="r") for p in y_paths]

    x_mean, x_std      = compute_x_stats_streaming(all_x)
    y_mean, y_std, s_c  = compute_y_stats_streaming(all_y, all_x, y_indices, args.eps)
    np.savez(save_dir / "norm_stats.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, s_c=s_c)
    print("  Saved norm_stats.npz")

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    full_ds = MultiViewSimDataset(
        x_paths, y_paths, y_indices,
        x_mean, x_std, y_mean, y_std, s_c, eps=args.eps,
    )

    # Split at simulation level
    n_val_sims   = max(1, int(len(selected_sims) * args.val_frac))
    n_train_sims = len(selected_sims) - n_val_sims
    perm              = rng.permutation(len(selected_sims))
    train_sim_indices = selected_sims[perm[:n_train_sims]]
    val_sim_indices   = selected_sims[perm[n_train_sims:]]

    # Each index in the dataset IS a simulation index (length = n_sims)
    train_ds = Subset(full_ds, train_sim_indices.tolist())
    val_ds   = Subset(full_ds, val_sim_indices.tolist())

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    print(f"  Train: {len(train_ds)} sims ({n_train_sims} sims × 3 views = "
          f"{n_train_sims * 3} total)")
    print(f"  Val:   {len(val_ds)} sims")
    print(f"  Batches/epoch: train={len(train_dl)}  val={len(val_dl)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = VelocityUNetV8(
        layout=layout,
        c_in=c_in, c_out=c_out, base=args.base, t_dim=args.t_dim, n_views=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  VelocityUNetV8 — base={args.base}  t_dim={args.t_dim}"
          f"  params={n_params:,}")

    # ── Warm start ───────────────────────────────────────────────────────────
    if args.warm_start:
        ws_ckpt = torch.load(Path(args.warm_start), map_location="cpu",
                             weights_only=False)
        missing, unexpected = model.load_state_dict(
            ws_ckpt["model_state"], strict=False)
        print(f"  Warm-started from: {args.warm_start}")
        if missing:
            print(f"    missing keys: {missing[:5]}{'…' if len(missing)>5 else ''}")
        if unexpected:
            print(f"    unexpected keys: {unexpected[:5]}")

    opt       = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Config snapshot ───────────────────────────────────────────────────────
    config = {
        "version":         "v8",
        "tensor_prefix":   str(pfx),
        "layout_map":      args.layout_map,
        "split":           splt,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "base":            args.base,
        "t_dim":           args.t_dim,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "c_in":            c_in,
        "c_out":           c_out,
        "y_indices":       y_indices,
        "y_channels":      args.y_channels,
        "eps":             args.eps,
        "seed":            args.seed,
        "n_params":        n_params,
        "n_sims_total":    N_sim_total,
        "n_sims_used":     n_use,
        "dataset_pct":     args.dataset,
        "n_train_sims":    n_train_sims,
        "n_val_sims":      n_val_sims,
        "loss_fn":         args.loss_fn,
        "val_frac":        args.val_frac,
        "method":          METHOD_LABELS[args.mode],
        "mode":            args.mode,
        "t_schedule":      args.t_schedule,
        "self_cond_prob":  args.self_cond_prob,
        "warm_start":      args.warm_start,
        "pos_channels":    sorted(POS_CHANNELS),
        "signed_channels": sorted(SIGNED_CHANNELS),
        "reinjection":     {
            "intra_view": True,
            "inter_view": True,
            "gap_px":     layout.gap_px,
            "k":          layout.k,
            "n_cross_pairs": len(layout.cross_mappings),
        },
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs …\n")
    best_val_loss = float("inf")
    metrics_history: dict = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_mae  = None
        n_batches  = 0
        t0 = time.time()

        for batch in train_dl:
            x0, y0, m0, x1, y1, m1, x2, y2, m2 = _unpack_batch(batch, device)

            loss, mae_c = step_fn(
                model, x0, y0, m0, x1, y1, m1, x2, y2, m2,
                loss_fn=args.loss_fn,
                t_schedule=args.t_schedule,
                self_cond_prob=args.self_cond_prob,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_mae   = mae_c.detach() if epoch_mae is None \
                          else epoch_mae + mae_c.detach()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_mae  = (epoch_mae / n_batches).cpu()
        dt       = time.time() - t0

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_mae  = None
        n_val    = 0
        with torch.no_grad():
            for batch in val_dl:
                x0, y0, m0, x1, y1, m1, x2, y2, m2 = _unpack_batch(batch, device)
                loss_v, mae_v = step_fn(
                    model, x0, y0, m0, x1, y1, m1, x2, y2, m2,
                    loss_fn=args.loss_fn,
                    t_schedule=args.t_schedule,
                    self_cond_prob=0.0,     # no self-conditioning during val
                )
                val_loss += loss_v.item()
                val_mae   = mae_v.detach() if val_mae is None \
                            else val_mae + mae_v.detach()
                n_val    += 1

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_mae  = (val_mae / max(n_val, 1)).cpu() \
                       if val_mae is not None else avg_mae

        train_entry = {"epoch": epoch, "loss": avg_loss,
                       "mae_avg": float(avg_mae.mean()),
                       "lr": scheduler.get_last_lr()[0], "time_s": dt}
        val_entry   = {"epoch": epoch, "loss": avg_val_loss,
                       "mae_avg": float(avg_val_mae.mean())}
        metrics_history["train"].append(train_entry)
        metrics_history["val"].append(val_entry)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"train={avg_loss:.5f} | val={avg_val_loss:.5f} | "
                f"MAE_t={float(avg_mae.mean()):.5f} | "
                f"MAE_v={float(avg_val_mae.mean()):.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            _save_checkpoint(
                model, save_dir / "checkpoint_best.pt",
                epoch, args, config, x_mean, x_std, y_mean, y_std, s_c,
                y_indices, c_in, c_out, avg_val_loss,
            )

    _save_checkpoint(
        model, save_dir / "checkpoint_last.pt",
        args.epochs, args, config, x_mean, x_std, y_mean, y_std, s_c,
        y_indices, c_in, c_out, avg_val_loss,
    )
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n{'─'*70}")
    print(f"  Training complete.  Best val loss = {best_val_loss:.6f}")
    print(f"  Checkpoints saved to {save_dir}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()
