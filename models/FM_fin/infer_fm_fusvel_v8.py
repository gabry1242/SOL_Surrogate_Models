#!/usr/bin/env python3
"""
infer_fm_fusvel_v9.py
─────────────────────────────────────────────────────────────────────────────
Inference for VelocityUNetV8 trained by train_fm_fusvel_full_v9.py.

Key difference from infer_fm_fusvel_v7.py
──────────────────────────────────────────
v7 runs each view independently through VelocityUNet.forward().
v9 trained with VelocityUNetV8.forward_3views(), which applies intra-view
horizontal AND inter-view vertical boundary reinjection at every encoder
ConvBlock.  Running v9 weights through the plain forward() path creates a
train/inference distribution mismatch — the model expected reinjected gap
features, not whatever the CNN produces at boundaries without them.

This script fixes that by:
  1. Instantiating VelocityUNetV8 (loaded from --layout_map) instead of
     the base VelocityUNet.
  2. Loading all 3 view tensors for each simulation together via
     ThreeViewXDataset, so each batch is (x0,m0, x1,m1, x2,m2).
  3. Calling model.forward_3views() inside every integrator step, exactly
     replicating the training forward pass.

The output layout is identical to v7: one pred_Y_img_* .npy per view
(plus pred_Y_std_* when n_samples > 1), and a single test_metrics.json.

Backward compatibility
──────────────────────
A v7 checkpoint (no "reinjection" key in ckpt["config"]) will fall back
to the original per-view VelocityUNet path automatically.  --layout_map
is only required when the checkpoint is from v9.

Usage (v9 checkpoint)
──────────────────────
python infer_fm_fusvel_v9.py \\
    --checkpoint  scripts/runs/fusvel_v9_xpred/checkpoint_best.pt \\
    --test_prefix scripts/tensor/train/global3 \\
    --layout_map  scripts/tensor/train/global3_layout_map_3views.npz \\
    --test_split  test \\
    --out_dir     scripts/runs/fusvel_v9_xpred/infer_test \\
    --batch_size  16 \\
    --integrator  stochastic \\
    --ode_steps   50 \\
    --n_samples   1000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Channel metadata  (must match training script)
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
# LayoutInfo  (copied verbatim from train_fm_fusvel_full_v9.py)
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

        self.horiz_gaps: List[List[Tuple[int, int]]] = [
            self._find_horiz_gaps(v) for v in range(3)
        ]
        self.cross_mappings: List[dict] = self._build_cross_mappings()

        print(f"  [LayoutInfo] gap_px={self.gap_px}  k={self.k}")
        for v in range(3):
            print(f"    view{v}: H={self.H_v[v]}  W={self.W_v[v]}"
                  f"  horiz_gaps={self.horiz_gaps[v]}")
        print(f"    cross_mappings: {len(self.cross_mappings)} pairs")

    def _find_horiz_gaps(self, v: int) -> List[Tuple[int, int]]:
        g  = self.gap_px
        W  = self.W_v[v]
        H  = self.H_v[v]
        m  = self.masks[v]

        r0 = g if (H > 2 * g) else 0
        r1 = H - g if (H > 2 * g) else H
        content = m[r0:r1, :W].astype(np.int32)

        col_sum = content.sum(axis=0)
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

        gaps = [(s, e) for (s, e) in gaps if s > 0 and e < W]
        return gaps

    def _strip_bounds(self, v: int):
        W    = self.W_v[v]
        gaps = self.horiz_gaps[v]

        boundaries = [0] + [c for (s, e) in gaps for c in (s, e)] + [W]
        strips = []
        for i in range(0, len(boundaries) - 1, 2):
            c0, c1 = boundaries[i], boundaries[i + 1]
            if c1 > c0:
                strips.append((c0, c1))
        return strips

    def _build_cross_mappings(self) -> List[dict]:
        g      = self.gap_px
        strips = [self._strip_bounds(v) for v in range(3)]

        mappings: List[dict] = []

        def add_pair(sv, dv, sc0, sc1, dc0, dc1,
                     src_is_top_for_sv, dst_is_top_for_dv):
            w = min(sc1 - sc0, dc1 - dc0)
            if w <= 0:
                return
            mappings.append(dict(
                src_view=sv, dst_view=dv,
                src_c0=sc0, src_c1=sc0 + w,
                dst_c0=dc0, dst_c1=dc0 + w,
                src_is_top=src_is_top_for_sv,
                dst_is_top=dst_is_top_for_dv,
            ))
            mappings.append(dict(
                src_view=dv, dst_view=sv,
                src_c0=dc0, src_c1=dc0 + w,
                dst_c0=sc0, dst_c1=sc0 + w,
                src_is_top=(not src_is_top_for_sv),
                dst_is_top=(not dst_is_top_for_dv),
            ))

        strips0 = strips[0]
        strips1 = strips[1]

        if len(strips0) >= 1 and len(strips1) >= 1:
            s0A = strips0[0];  s1C = strips1[0]
            add_pair(sv=0, dv=1,
                     sc0=s0A[0], sc1=s0A[1],
                     dc0=s1C[0], dc1=s1C[1],
                     src_is_top_for_sv=False,
                     dst_is_top_for_dv=True)

        if len(strips0) >= 2 and len(strips1) >= 3:
            s0B = strips0[-1];  s1E = strips1[-1]
            add_pair(sv=0, dv=1,
                     sc0=s0B[0], sc1=s0B[1],
                     dc0=s1E[0], dc1=s1E[1],
                     src_is_top_for_sv=False,
                     dst_is_top_for_dv=True)

        if len(strips1) >= 3:
            s1D = strips1[1]
            bc0 = g
            bc1 = self.W_v[2] - g
            add_pair(sv=2, dv=1,
                     sc0=bc0, sc1=bc1,
                     dc0=s1D[0], dc1=s1D[1],
                     src_is_top_for_sv=False,
                     dst_is_top_for_dv=True)

        return mappings


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: single-view X  (unchanged from v7, kept for v7 fallback path)
# ─────────────────────────────────────────────────────────────────────────────

class ViewXDataset(Dataset):
    def __init__(self, x_path: Path, x_mean: np.ndarray, x_std: np.ndarray):
        self.X      = np.load(x_path, mmap_mode="r")
        self.x_mean = x_mean
        self.x_std  = x_std

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x_raw  = np.array(self.X[idx], dtype=np.float32)
        mask   = x_raw[0:1].copy()
        x_norm = (x_raw - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        return torch.from_numpy(x_norm), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: all 3 views per simulation  (new for v9)
# ─────────────────────────────────────────────────────────────────────────────

class ThreeViewXDataset(Dataset):
    """
    Returns all 3 views for the same simulation index:
      (x0_norm, mask0, x1_norm, mask1, x2_norm, mask2)

    This mirrors MultiViewSimDataset from v9 training (X side only, no Y),
    and is what enables forward_3views() at inference with the same
    simulation's views processed together — exactly as during training.
    """

    def __init__(self, prefix: str, split: str,
                 x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.X: List = []
        for v in range(3):
            p = Path(f"{prefix}_view{v}_X_img_{split}.npy")
            if not p.exists():
                raise FileNotFoundError(f"Missing view{v} X tensor: {p}")
            self.X.append(np.load(p, mmap_mode="r"))

        # All views must have the same number of simulations
        n_sims = [arr.shape[0] for arr in self.X]
        if len(set(n_sims)) != 1:
            raise ValueError(f"View simulation counts differ: {n_sims}")

        self.x_mean = x_mean
        self.x_std  = x_std

    def __len__(self) -> int:
        return int(self.X[0].shape[0])

    def __getitem__(self, idx: int):
        out = []
        for v in range(3):
            x_raw  = np.array(self.X[v][idx], dtype=np.float32)
            mask   = x_raw[0:1].copy()
            x_norm = (x_raw - self.x_mean[:, None, None]) / self.x_std[:, None, None]
            out.append(torch.from_numpy(x_norm))
            out.append(torch.from_numpy(mask))
        return tuple(out)   # (x0, m0, x1, m1, x2, m2)


# ─────────────────────────────────────────────────────────────────────────────
# Model components  (must match training script exactly)
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
    """Original v7 model — kept completely unchanged for backward compat."""

    def __init__(self, c_in: int, c_out: int, base: int = 32,
                 t_dim: int = 128, n_views: int = 3):
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
    def _match(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
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
# VelocityUNetV8  (copied verbatim from train_fm_fusvel_full_v9.py)
# ─────────────────────────────────────────────────────────────────────────────

class VelocityUNetV8(VelocityUNet):
    """
    VelocityUNet + intra-view horizontal and inter-view vertical boundary
    reinjection between every encoder ConvBlock.

    forward_3views() is the only correct inference path for v9 checkpoints.
    The inherited single-view forward() is preserved as a fallback only.
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
        gaps  = self.layout.horiz_gaps[view_id]
        if not gaps:
            return feat

        k_raw = self.layout.k
        k_eff = max(1, k_raw // scale)

        feat   = feat.clone()
        W_feat = feat.shape[3]

        for (gs, ge) in gaps:
            gs_s = gs // scale
            ge_s = ge // scale
            span = ge_s - gs_s

            if span < 2 or k_eff > span:
                continue

            k_use = min(k_eff, span // 2)

            if (ge_s + k_use > W_feat) or (gs_s < k_use):
                continue

            # Left part of gap ← right strip's left edge
            feat[:, :, :, gs_s : gs_s + k_use] = \
                feat[:, :, :, ge_s : ge_s + k_use]
            # Right part of gap ← left strip's right edge
            feat[:, :, :, ge_s - k_use : ge_s] = \
                feat[:, :, :, gs_s - k_use : gs_s]

        return feat

    # ------------------------------------------------------------------
    # Inter-view vertical reinjection
    # ------------------------------------------------------------------

    def _ri_v(self,
              f0: torch.Tensor,
              f1: torch.Tensor,
              f2: torch.Tensor,
              scale: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            k_use  = min(k_s, w)
            Hv_src = H_vs[sv]
            Hv_dst = H_vs[dv]

            if m["src_is_top"]:
                sr0 = g_s
            else:
                sr0 = Hv_src - g_s - k_use

            if m["dst_is_top"]:
                dr0 = g_s - k_use
            else:
                dr0 = Hv_dst - g_s

            sr0 = max(0, sr0)
            dr0 = max(0, dr0)

            src_feat = feats[sv]
            dst_feat = feats[dv]

            if (sr0 + k_use > src_feat.shape[2] or
                dr0 + k_use > dst_feat.shape[2] or
                sc1 > src_feat.shape[3] or
                dc1 > dst_feat.shape[3]):
                continue

            feats[dv][:, :, dr0 : dr0 + k_use, dc0 : dc0 + w] = \
                src_feat[:, :, sr0 : sr0 + k_use, sc0 : sc0 + w]

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
        t:  torch.Tensor,   # (B,) — shared across all 3 views of the same sim
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x0.device
        B, _, H, W = x0.shape

        te = self.time_emb(t)

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

        # ── Encoder scale 1  +  reinjection ──────────────────────────
        e1_0 = self.enc1(torch.cat([xv0, y0], 1))
        e1_1 = self.enc1(torch.cat([xv1, y1], 1))
        e1_2 = self.enc1(torch.cat([xv2, y2], 1))
        e1_0 = self._ri_h(e1_0, 0, 1)
        e1_1 = self._ri_h(e1_1, 1, 1)
        e1_2 = self._ri_h(e1_2, 2, 1)
        e1_0, e1_1, e1_2 = self._ri_v(e1_0, e1_1, e1_2, 1)

        # ── Encoder scale 2  +  reinjection ──────────────────────────
        e2_0 = self.enc2(self.pool(e1_0))
        e2_1 = self.enc2(self.pool(e1_1))
        e2_2 = self.enc2(self.pool(e1_2))
        e2_0 = self._ri_h(e2_0, 0, 2)
        e2_1 = self._ri_h(e2_1, 1, 2)
        e2_2 = self._ri_h(e2_2, 2, 2)
        e2_0, e2_1, e2_2 = self._ri_v(e2_0, e2_1, e2_2, 2)

        # ── Encoder scale 4  +  reinjection ──────────────────────────
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

        # ── Decoder  (per-view, shared weights) ──────────────────────
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
# Integrators: v7 single-view  (unchanged, used in fallback path)
# ─────────────────────────────────────────────────────────────────────────────

def euler_integrate_velocity(model, x_cond, mask, view_id, c_out, n_steps=50, device=None):
    B, _, H, W = x_cond.shape
    dt = 1.0 / n_steps
    y_t = torch.randn(B, c_out, H, W, device=device) * mask
    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=device)
        v = model(x_cond, y_t, t, view_id)
        y_t = (y_t + v * dt) * mask
    return y_t


def euler_integrate_xpred(model, x_cond, mask, view_id, c_out, n_steps=50, device=None):
    B, _, H, W = x_cond.shape
    dt = 1.0 / n_steps
    y_t = torch.randn(B, c_out, H, W, device=device) * mask
    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)
        y1_pred = model(x_cond, y_t, t, view_id)
        denom = max(1.0 - t_val, 1e-3)
        v = (y1_pred - y_t) / denom
        y_t = (y_t + v * dt) * mask
    return y_t


def stochastic_integrate_xpred(model, x_cond, mask, view_id, c_out, n_steps=50, device=None):
    B, _, H, W = x_cond.shape
    dt = 1.0 / n_steps
    y_t = torch.randn(B, c_out, H, W, device=device) * mask
    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)
        y1_pred = model(x_cond, y_t, t, view_id)
        t_next = (step + 1) * dt
        if t_next < 1.0:
            z_fresh = torch.randn(B, c_out, H, W, device=device) * mask
            y_t = ((1.0 - t_next) * z_fresh + t_next * y1_pred) * mask
        else:
            y_t = y1_pred * mask
    return y_t


def direct_forward(model, x_cond, mask, view_id, c_out, device=None):
    B, _, H, W = x_cond.shape
    y_dummy = torch.zeros(B, c_out, H, W, device=device)
    t_ones  = torch.ones(B, device=device)
    pred    = model(x_cond, y_dummy, t_ones, view_id)
    return pred * mask


# ─────────────────────────────────────────────────────────────────────────────
# Integrators: v9 three-view  (use forward_3views — the correct path)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each integrator takes all 3 views of the same simulation together and calls
# model.forward_3views() at every step.  This is identical to the training
# forward pass, so the model always operates in distribution.
#
# All three integrators return (pred0, pred1, pred2), one tensor per view.
# The caller is responsible for applying the per-view masks afterwards.
#

def stochastic_integrate_xpred_3v(
    model: VelocityUNetV8,
    x0, m0, x1, m1, x2, m2,
    c_out: int,
    n_steps: int = 50,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stochastic x-prediction sampler for v9.

    At each step the model predicts all 3 clean fields simultaneously via
    forward_3views(), then re-noises to stay exactly on the training
    interpolation path q_t = (1-t)·N(0,I) + t·data.  This is the same
    strategy as the v7 stochastic sampler but extended to 3 views.

    A single t is shared across all 3 views of the same simulation, matching
    the convention used during training (one t per sim, not per view).
    """
    B, _, H0, W0 = x0.shape
    _, _, H1, W1 = x1.shape
    _, _, H2, W2 = x2.shape
    dt = 1.0 / n_steps

    y0 = torch.randn(B, c_out, H0, W0, device=device) * m0
    y1 = torch.randn(B, c_out, H1, W1, device=device) * m1
    y2 = torch.randn(B, c_out, H2, W2, device=device) * m2

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)

        # All 3 views forward together — reinjection happens inside
        p0, p1, p2 = model.forward_3views(x0, y0, x1, y1, x2, y2, t)

        t_next = (step + 1) * dt
        if t_next < 1.0:
            z0 = torch.randn(B, c_out, H0, W0, device=device) * m0
            z1 = torch.randn(B, c_out, H1, W1, device=device) * m1
            z2 = torch.randn(B, c_out, H2, W2, device=device) * m2
            y0 = ((1.0 - t_next) * z0 + t_next * p0) * m0
            y1 = ((1.0 - t_next) * z1 + t_next * p1) * m1
            y2 = ((1.0 - t_next) * z2 + t_next * p2) * m2
        else:
            y0 = p0 * m0
            y1 = p1 * m1
            y2 = p2 * m2

    return y0, y1, y2


def euler_integrate_xpred_3v(
    model: VelocityUNetV8,
    x0, m0, x1, m1, x2, m2,
    c_out: int,
    n_steps: int = 50,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic Euler x-prediction integrator for v9.
    Less robust than stochastic near the detachment transition — use
    stochastic unless you specifically need a deterministic trajectory.
    """
    B, _, H0, W0 = x0.shape
    _, _, H1, W1 = x1.shape
    _, _, H2, W2 = x2.shape
    dt = 1.0 / n_steps

    y0 = torch.randn(B, c_out, H0, W0, device=device) * m0
    y1 = torch.randn(B, c_out, H1, W1, device=device) * m1
    y2 = torch.randn(B, c_out, H2, W2, device=device) * m2

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)

        p0, p1, p2 = model.forward_3views(x0, y0, x1, y1, x2, y2, t)

        denom = max(1.0 - t_val, 1e-3)
        y0 = (y0 + ((p0 - y0) / denom) * dt) * m0
        y1 = (y1 + ((p1 - y1) / denom) * dt) * m1
        y2 = (y2 + ((p2 - y2) / denom) * dt) * m2

    return y0, y1, y2


def euler_integrate_velocity_3v(
    model: VelocityUNetV8,
    x0, m0, x1, m1, x2, m2,
    c_out: int,
    n_steps: int = 50,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Velocity-prediction Euler ODE for v9 (cfm mode)."""
    B, _, H0, W0 = x0.shape
    _, _, H1, W1 = x1.shape
    _, _, H2, W2 = x2.shape
    dt = 1.0 / n_steps

    y0 = torch.randn(B, c_out, H0, W0, device=device) * m0
    y1 = torch.randn(B, c_out, H1, W1, device=device) * m1
    y2 = torch.randn(B, c_out, H2, W2, device=device) * m2

    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=device)
        v0, v1, v2 = model.forward_3views(x0, y0, x1, y1, x2, y2, t)
        y0 = (y0 + v0 * dt) * m0
        y1 = (y1 + v1 * dt) * m1
        y2 = (y2 + v2 * dt) * m2

    return y0, y1, y2


def direct_forward_3v(
    model: VelocityUNetV8,
    x0, m0, x1, m1, x2, m2,
    c_out: int,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass for direct regression (direct mode)."""
    B, _, H0, W0 = x0.shape
    _, _, H1, W1 = x1.shape
    _, _, H2, W2 = x2.shape

    d0 = torch.zeros(B, c_out, H0, W0, device=device)
    d1 = torch.zeros(B, c_out, H1, W1, device=device)
    d2 = torch.zeros(B, c_out, H2, W2, device=device)
    t  = torch.ones(B, device=device)

    p0, p1, p2 = model.forward_3views(x0, d0, x1, d1, x2, d2, t)
    return p0 * m0, p1 * m1, p2 * m2


# ─────────────────────────────────────────────────────────────────────────────
# Inverse transform  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LOG10  = 37.0
MAX_SYMLOG = 37.0

def inverse_transform_y(y_t, y_indices, pos_channels, signed_channels, eps, s_c):
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)

    C_sel  = len(y_indices)
    is_pos = np.array([int(c) in pos_set    for c in y_indices], dtype=bool)
    is_sig = np.array([int(c) in signed_set for c in y_indices], dtype=bool)

    if is_pos.sum() + is_sig.sum() != C_sel:
        bad = [y_indices[j] for j in range(C_sel) if not is_pos[j] and not is_sig[j]]
        raise ValueError(f"Channels not categorised: {bad}")

    t64 = y_t.astype(np.float64)
    out = np.empty_like(t64)

    if is_pos.any():
        clamped        = np.clip(t64[:, is_pos], -MAX_LOG10, MAX_LOG10)
        out[:, is_pos] = 10.0 ** clamped - eps

    if is_sig.any():
        sc_vec         = np.asarray(s_c, dtype=np.float64)[is_sig]
        clamped        = np.clip(t64[:, is_sig], -MAX_SYMLOG, MAX_SYMLOG)
        out[:, is_sig] = (np.sign(clamped)
                          * sc_vec[None, :, None, None]
                          * (10.0 ** np.abs(clamped) - 1.0))

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Masked metrics  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae_per_channel_np(pred, y_true, mask, eps=1e-8):
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (np.abs(pred.astype(np.float64) - y_true.astype(np.float64)) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)

def masked_rmse_per_channel_np(pred, y_true, mask, eps=1e-8):
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps)
    num = (((pred.astype(np.float64) - y_true.astype(np.float64)) ** 2) * m).sum(axis=(0, 2, 3))
    return np.sqrt(num / den).astype(np.float64)

def masked_log_mae_per_channel_np(pred, y_true, mask, y_indices,
                                   pos_channels, signed_channels,
                                   eps_log=1e-3, eps_den=1e-8):
    pos_set    = set(int(c) for c in pos_channels)
    signed_set = set(int(c) for c in signed_channels)
    C_sel      = len(y_indices)
    result     = np.zeros(C_sel, dtype=np.float64)
    m          = np.broadcast_to(mask.astype(np.float64), pred.shape)
    for j, orig_c in enumerate(y_indices):
        den = max(float(m[:, j:j+1].sum()), eps_den)
        p64 = pred[:, j].astype(np.float64)
        t64 = y_true[:, j].astype(np.float64)
        if int(orig_c) in pos_set:
            log_p = np.log10(np.maximum(p64, 0.0) + eps_log)
            log_t = np.log10(np.maximum(t64, 0.0) + eps_log)
            ae    = np.abs(log_p - log_t) * m[:, j]
        elif int(orig_c) in signed_set:
            symlog_p = np.sign(p64) * np.log10(np.abs(p64) + 1.0)
            symlog_t = np.sign(t64) * np.log10(np.abs(t64) + 1.0)
            ae = np.abs(symlog_p - symlog_t) * m[:, j]
        else:
            ae = np.zeros_like(p64)
        result[j] = float(ae.sum() / den)
    return result

def masked_relative_error_per_channel_np(pred, y_true, mask, rel_floor=1e-8, eps_den=1e-8):
    m   = np.broadcast_to(mask.astype(np.float64), pred.shape)
    den = np.maximum(m.sum(axis=(0, 2, 3)), eps_den)
    p64 = pred.astype(np.float64)
    t64 = y_true.astype(np.float64)
    ae  = np.abs(p64 - t64)
    ref = np.maximum(np.abs(t64), rel_floor)
    num = ((ae / ref) * m).sum(axis=(0, 2, 3))
    return (num / den).astype(np.float64)

def _safe_float(v) -> float:
    f = float(v)
    if np.isfinite(f):
        return f
    if np.isnan(f):
        return None
    return 1e38 if f > 0 else -1e38


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for denorm + inverse transform  (factored out to avoid repetition)
# ─────────────────────────────────────────────────────────────────────────────

def _denorm_and_invert(preds_norm, masks_np, y_mean, y_std,
                       y_indices, pos_channels, signed_channels, eps, s_c):
    """Shared post-processing: denormalise → inverse transform → mask."""
    ym = y_mean.reshape(1, -1, 1, 1)
    ys = y_std.reshape( 1, -1, 1, 1)
    preds_t    = preds_norm * ys + ym
    preds_phys = inverse_transform_y(
        preds_t, y_indices=y_indices,
        pos_channels=pos_channels, signed_channels=signed_channels,
        eps=eps, s_c=s_c,
    )
    preds_phys *= masks_np
    return preds_phys


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inference for VelocityUNet(V8) — v7 and v9 checkpoints."
    )
    ap.add_argument("--checkpoint",   required=True,
                    help="Path to checkpoint_best.pt (v7 or v9)")
    ap.add_argument("--test_prefix",  required=True,
                    help="Tensor prefix, e.g. scripts/tensor/test/global3 "
                         "(files are {prefix}_view{0,1,2}_X_img_{split}.npy)")
    ap.add_argument("--layout_map",   default=None,
                    help="Path to *_layout_map_3views.npz "
                         "(required for v9 checkpoints, ignored for v7)")
    ap.add_argument("--test_split",   default="test")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--batch_size",   type=int, default=16)
    ap.add_argument("--device",       default=None)
    ap.add_argument("--ode_steps",    type=int, default=50)
    ap.add_argument("--n_samples",    type=int, default=1)
    ap.add_argument("--mode",         default=None, choices=["cfm", "xpred", "direct"])
    ap.add_argument("--integrator",   default="stochastic",
                    choices=["euler", "stochastic"])
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)

    c_in            = int(ckpt["c_in"])
    c_out           = int(ckpt["c_out"])
    base            = int(ckpt.get("base", 32))
    t_dim           = int(ckpt.get("t_dim", 128))
    y_indices       = [int(c) for c in ckpt["y_indices"]]
    pos_channels    = ckpt.get("pos_channels",    sorted(POS_CHANNELS))
    signed_channels = ckpt.get("signed_channels", sorted(SIGNED_CHANNELS))
    eps             = float(ckpt.get("eps", 1e-3))

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)
    s_c    = np.asarray(ckpt["s_c"],    dtype=np.float32)

    method = ckpt.get("method", "OT-CFM-xpred")
    mode   = ckpt.get("mode",   "cfm")
    if args.mode is not None:
        mode = args.mode

    # ── Detect whether this is a v9 (reinjection) checkpoint ────────────────
    saved_config = ckpt.get("config", {})
    is_v9 = saved_config.get("reinjection") is not None

    is_direct = (mode == "direct")
    n_samples = 1 if is_direct else args.n_samples

    print(f"\n{'='*70}")
    print(f"  VelocityUNet{'V8' if is_v9 else ''} Inference — {method}")
    print(f"  Checkpoint arch : {'v9 (reinjection)' if is_v9 else 'v7 (no reinjection)'}")
    if mode == "xpred":
        print(f"  Mode       : XPRED ({args.integrator} integrator)")
    elif mode == "cfm":
        print(f"  Mode       : CFM (velocity-prediction ODE)")
    else:
        print(f"  Mode       : DIRECT (single forward pass)")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  c_in={c_in}  c_out={c_out}  base={base}  t_dim={t_dim}")
    if not is_direct:
        print(f"  Steps      : {args.ode_steps}")
        print(f"  N samples  : {n_samples}")
    print(f"  Device     : {device}")
    print(f"{'='*70}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ──────────────────────────────────────────────────────────
    if is_v9:
        if args.layout_map is None:
            raise ValueError(
                "This is a v9 checkpoint (reinjection was used during training). "
                "You must supply --layout_map pointing to the "
                "*_layout_map_3views.npz file used at training time."
            )
        print("  Loading layout map …")
        layout = LayoutInfo(args.layout_map)
        model  = VelocityUNetV8(
            layout=layout, c_in=c_in, c_out=c_out, base=base, t_dim=t_dim, n_views=3,
        ).to(device)
        print(f"  Instantiated VelocityUNetV8")
    else:
        model = VelocityUNet(
            c_in=c_in, c_out=c_out, base=base, t_dim=t_dim, n_views=3,
        ).to(device)
        print(f"  Instantiated VelocityUNet (v7)")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ════════════════════════════════════════════════════════════════════════
    # V9 INFERENCE PATH — 3-view forward_3views() at every integration step
    # ════════════════════════════════════════════════════════════════════════
    if is_v9:
        _run_v9_inference(
            model=model,
            prefix=args.test_prefix,
            split=args.test_split,
            out_dir=out_dir,
            x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std, s_c=s_c,
            c_out=c_out, y_indices=y_indices,
            pos_channels=pos_channels, signed_channels=signed_channels, eps=eps,
            mode=mode, integrator=args.integrator,
            n_steps=args.ode_steps, n_samples=n_samples,
            batch_size=args.batch_size, device=device,
            method=method,
        )
        return

    # ════════════════════════════════════════════════════════════════════════
    # V7 FALLBACK PATH — unchanged from infer_fm_fusvel_v7.py
    # ════════════════════════════════════════════════════════════════════════
    _run_v7_inference(
        model=model,
        prefix=args.test_prefix,
        split=args.test_split,
        out_dir=out_dir,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std, s_c=s_c,
        c_out=c_out, y_indices=y_indices,
        pos_channels=pos_channels, signed_channels=signed_channels, eps=eps,
        mode=mode, integrator=args.integrator,
        n_steps=args.ode_steps, n_samples=n_samples,
        batch_size=args.batch_size, device=device,
        method=method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# V9 inference runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_v9_inference(
    model, prefix, split, out_dir,
    x_mean, x_std, y_mean, y_std, s_c,
    c_out, y_indices, pos_channels, signed_channels, eps,
    mode, integrator, n_steps, n_samples, batch_size, device, method,
):
    """
    Run inference with forward_3views() at every step.

    One DataLoader yields (x0,m0, x1,m1, x2,m2) per batch — all 3 views of
    the same simulation together.  Per-view Welford accumulators collect
    mean and variance across n_samples draws.
    """
    # ── Load 3-view dataset ──────────────────────────────────────────────────
    ds3 = ThreeViewXDataset(prefix, split, x_mean, x_std)
    dl3 = DataLoader(ds3, batch_size=batch_size, shuffle=False,
                     num_workers=0, pin_memory=(device.type == "cuda"))
    N = len(ds3)

    # ── Spatial dimensions per view ──────────────────────────────────────────
    # All views are padded to Hmax × Wmax so shapes should be identical, but
    # we derive them independently for robustness.
    shapes: List[Tuple[int, int]] = []
    for v in range(3):
        arr = np.load(Path(f"{prefix}_view{v}_X_img_{split}.npy"), mmap_mode="r")
        shapes.append((arr.shape[2], arr.shape[3]))  # (H, W)
    print(f"  View shapes (H×W): {shapes}")

    # ── Extract masks for all 3 views in one DataLoader pass ─────────────────
    # masks_v[v] shape: (N, 1, H, W)
    masks_v = [np.zeros((N, 1, shapes[v][0], shapes[v][1]), dtype=np.float32)
               for v in range(3)]
    _idx = 0
    for batch in DataLoader(ds3, batch_size=batch_size, shuffle=False, num_workers=0):
        x0, m0, x1, m1, x2, m2 = batch
        b = x0.shape[0]
        masks_v[0][_idx:_idx+b] = m0.numpy()
        masks_v[1][_idx:_idx+b] = m1.numpy()
        masks_v[2][_idx:_idx+b] = m2.numpy()
        _idx += b

    # ── Welford accumulators — one pair per view ──────────────────────────────
    wf_mean = [np.zeros((N, c_out, shapes[v][0], shapes[v][1]), dtype=np.float64)
               for v in range(3)]
    wf_M2   = [np.zeros((N, c_out, shapes[v][0], shapes[v][1]), dtype=np.float64)
               for v in range(3)]

    # ── Sample loop ──────────────────────────────────────────────────────────
    for sample_idx in range(n_samples):
        print(f"    sample {sample_idx+1}/{n_samples}", end="\r", flush=True)

        # Accumulate per-view predictions on GPU before one bulk CPU transfer
        preds_gpu = [
            torch.zeros(N, c_out, shapes[v][0], shapes[v][1],
                        dtype=torch.float32, device=device)
            for v in range(3)
        ]
        idx0 = 0

        with torch.no_grad():
            for batch in dl3:
                x0, m0, x1, m1, x2, m2 = batch
                b = x0.shape[0]

                x0g = x0.to(device);  m0g = m0.to(device)
                x1g = x1.to(device);  m1g = m1.to(device)
                x2g = x2.to(device);  m2g = m2.to(device)

                if mode == "direct":
                    p0, p1, p2 = direct_forward_3v(
                        model, x0g, m0g, x1g, m1g, x2g, m2g,
                        c_out=c_out, device=device)
                elif mode == "xpred" and integrator == "stochastic":
                    p0, p1, p2 = stochastic_integrate_xpred_3v(
                        model, x0g, m0g, x1g, m1g, x2g, m2g,
                        c_out=c_out, n_steps=n_steps, device=device)
                elif mode == "xpred":
                    p0, p1, p2 = euler_integrate_xpred_3v(
                        model, x0g, m0g, x1g, m1g, x2g, m2g,
                        c_out=c_out, n_steps=n_steps, device=device)
                else:   # cfm
                    p0, p1, p2 = euler_integrate_velocity_3v(
                        model, x0g, m0g, x1g, m1g, x2g, m2g,
                        c_out=c_out, n_steps=n_steps, device=device)

                preds_gpu[0][idx0:idx0+b] = p0
                preds_gpu[1][idx0:idx0+b] = p1
                preds_gpu[2][idx0:idx0+b] = p2
                idx0 += b

        # ── Per-view: transfer GPU→CPU, denorm, Welford update ───────────────
        for v in range(3):
            preds_norm = preds_gpu[v].cpu().numpy()
            preds_phys = _denorm_and_invert(
                preds_norm, masks_v[v], y_mean, y_std,
                y_indices, pos_channels, signed_channels, eps, s_c,
            )
            delta          = preds_phys.astype(np.float64) - wf_mean[v]
            wf_mean[v]    += delta / (sample_idx + 1)
            wf_M2[v]      += delta * (preds_phys.astype(np.float64) - wf_mean[v])
            del preds_norm, preds_phys

        del preds_gpu

    print()  # newline after \r

    # ── Save outputs and compute metrics ─────────────────────────────────────
    all_view_metrics = {}

    for v in range(3):
        view_tag = f"view{v}"

        preds_mean = wf_mean[v].astype(np.float32)
        preds_std  = np.sqrt(wf_M2[v] / max(n_samples - 1, 1)).astype(np.float32)

        pred_path = out_dir / f"pred_Y_img_{split}_{view_tag}.npy"
        np.save(pred_path, preds_mean)
        print(f"  Saved: {pred_path}  shape={preds_mean.shape}")

        if n_samples > 1:
            std_path = out_dir / f"pred_Y_std_{split}_{view_tag}.npy"
            np.save(std_path, preds_std)
            print(f"  Saved: {std_path}  shape={preds_std.shape}")

        # Metrics against ground truth
        y_path = Path(f"{prefix}_view{v}_Y_img_{split}.npy")
        Y_full = np.load(y_path, mmap_mode="r")
        Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

        mae_c     = masked_mae_per_channel_np(preds_mean, Y_sel, masks_v[v])
        rmse_c    = masked_rmse_per_channel_np(preds_mean, Y_sel, masks_v[v])
        log_mae_c = masked_log_mae_per_channel_np(
            preds_mean, Y_sel, masks_v[v],
            y_indices=y_indices, pos_channels=pos_channels,
            signed_channels=signed_channels, eps_log=eps,
        )
        mre_c = masked_relative_error_per_channel_np(preds_mean, Y_sel, masks_v[v])

        print(f"  [{view_tag}] Physical MAE_avg  = {float(np.mean(mae_c)):.6g}")
        print(f"  [{view_tag}] Physical RMSE_avg = {float(np.mean(rmse_c)):.6g}")
        print(f"  [{view_tag}] Log-space MAE_avg = {float(np.mean(log_mae_c)):.4f}")
        print(f"  [{view_tag}] Relative err avg  = {float(np.mean(mre_c)):.4f}")
        print(f"  [{view_tag}] {'channel':<12s} {'phys_MAE':>12s} {'log_MAE':>10s} {'MRE':>10s}")
        for j, c in enumerate(y_indices):
            nm = CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"
            print(f"  [{view_tag}] {nm:<12s} {mae_c[j]:>12.4g} {log_mae_c[j]:>10.4f} {mre_c[j]:>10.4f}")

        all_view_metrics[view_tag] = {
            "mae_avg":              _safe_float(np.mean(mae_c)),
            "rmse_avg":             _safe_float(np.mean(rmse_c)),
            "log_mae_avg":          _safe_float(np.mean(log_mae_c)),
            "mre_avg":              _safe_float(np.mean(mre_c)),
            "mae_per_channel":      [_safe_float(val) for val in mae_c],
            "rmse_per_channel":     [_safe_float(val) for val in rmse_c],
            "log_mae_per_channel":  [_safe_float(val) for val in log_mae_c],
            "mre_per_channel":      [_safe_float(val) for val in mre_c],
        }

    _write_metrics_json(out_dir, all_view_metrics, args_dict={
        "checkpoint":    "<from caller>",
        "arch":          "velocity_unet_v8",
        "method":        method,
        "mode":          mode,
        "integrator":    integrator if mode == "xpred" else None,
        "ode_steps":     n_steps if mode != "direct" else None,
        "n_samples":     n_samples,
        "c_in":          "<from caller>",
        "c_out":         c_out,
        "y_indices":     y_indices,
    })


# ─────────────────────────────────────────────────────────────────────────────
# V7 fallback runner  (original logic, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _run_v7_inference(
    model, prefix, split, out_dir,
    x_mean, x_std, y_mean, y_std, s_c,
    c_out, y_indices, pos_channels, signed_channels, eps,
    mode, integrator, n_steps, n_samples, batch_size, device, method,
):
    all_view_metrics = {}
    is_direct = (mode == "direct")

    for v in range(3):
        view_tag     = f"view{v}"
        test_x_path  = Path(f"{prefix}_view{v}_X_img_{split}.npy")
        test_y_path  = Path(f"{prefix}_view{v}_Y_img_{split}.npy")

        for p in (test_x_path, test_y_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        print(f"  ── {view_tag} ──")

        ds = ViewXDataset(test_x_path, x_mean=x_mean, x_std=x_std)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

        N = len(ds)
        X_peek = np.load(test_x_path, mmap_mode="r")
        _, _, H, W = X_peek.shape

        masks_np = np.zeros((N, 1, H, W), dtype=np.float32)
        _idx = 0
        for x, m in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            b = m.shape[0]
            masks_np[_idx:_idx+b] = m.numpy()
            _idx += b
        masks_gpu = torch.from_numpy(masks_np).to(device)

        wf_mean = np.zeros((N, c_out, H, W), dtype=np.float64)
        wf_M2   = np.zeros((N, c_out, H, W), dtype=np.float64)

        for sample_idx in range(n_samples):
            print(f"    sample {sample_idx+1}/{n_samples}", end="\r", flush=True)
            preds_gpu = torch.zeros(N, c_out, H, W, dtype=torch.float32, device=device)
            idx0 = 0

            with torch.no_grad():
                for x, _ in dl:
                    b        = x.shape[0]
                    x_gpu    = x.to(device)
                    view_ids = torch.full((b,), v, dtype=torch.long, device=device)

                    if mode == "direct":
                        pred = direct_forward(
                            model, x_gpu, masks_gpu[idx0:idx0+b],
                            view_ids, c_out=c_out, device=device)
                    elif mode == "xpred" and integrator == "stochastic":
                        pred = stochastic_integrate_xpred(
                            model, x_gpu, masks_gpu[idx0:idx0+b],
                            view_ids, c_out=c_out,
                            n_steps=n_steps, device=device)
                    elif mode == "xpred":
                        pred = euler_integrate_xpred(
                            model, x_gpu, masks_gpu[idx0:idx0+b],
                            view_ids, c_out=c_out,
                            n_steps=n_steps, device=device)
                    else:
                        pred = euler_integrate_velocity(
                            model, x_gpu, masks_gpu[idx0:idx0+b],
                            view_ids, c_out=c_out,
                            n_steps=n_steps, device=device)

                    preds_gpu[idx0:idx0+b] = pred
                    idx0 += b

            preds_norm = preds_gpu.cpu().numpy()
            del preds_gpu

            preds_phys = _denorm_and_invert(
                preds_norm, masks_np, y_mean, y_std,
                y_indices, pos_channels, signed_channels, eps, s_c,
            )
            del preds_norm

            delta    = preds_phys.astype(np.float64) - wf_mean
            wf_mean += delta / (sample_idx + 1)
            wf_M2   += delta * (preds_phys.astype(np.float64) - wf_mean)

        print()

        preds_phys     = wf_mean.astype(np.float32)
        preds_phys_std = np.sqrt(wf_M2 / max(n_samples - 1, 1)).astype(np.float32)

        pred_path = out_dir / f"pred_Y_img_{split}_{view_tag}.npy"
        np.save(pred_path, preds_phys)
        print(f"    Saved: {pred_path}  shape={preds_phys.shape}")

        if n_samples > 1:
            std_path = out_dir / f"pred_Y_std_{split}_{view_tag}.npy"
            np.save(std_path, preds_phys_std)
            print(f"    Saved: {std_path}  shape={preds_phys_std.shape}")

        Y_full = np.load(test_y_path, mmap_mode="r")
        Y_sel  = np.array(Y_full[:, y_indices, :, :], dtype=np.float32)

        mae_c     = masked_mae_per_channel_np(preds_phys, Y_sel, masks_np)
        rmse_c    = masked_rmse_per_channel_np(preds_phys, Y_sel, masks_np)
        log_mae_c = masked_log_mae_per_channel_np(
            preds_phys, Y_sel, masks_np,
            y_indices=y_indices, pos_channels=pos_channels,
            signed_channels=signed_channels, eps_log=eps,
        )
        mre_c = masked_relative_error_per_channel_np(preds_phys, Y_sel, masks_np)

        print(f"    Physical MAE_avg  = {float(np.mean(mae_c)):.6g}")
        print(f"    Physical RMSE_avg = {float(np.mean(rmse_c)):.6g}")
        print(f"    Log-space MAE_avg = {float(np.mean(log_mae_c)):.4f}")
        print(f"    Relative err avg  = {float(np.mean(mre_c)):.4f}")
        print(f"    {'channel':<12s} {'phys_MAE':>12s} {'log_MAE':>10s} {'MRE':>10s}")
        for j, c in enumerate(y_indices):
            nm = CH_NAMES[c] if c < len(CH_NAMES) else f"ch{c}"
            print(f"    {nm:<12s} {mae_c[j]:>12.4g} {log_mae_c[j]:>10.4f} {mre_c[j]:>10.4f}")

        all_view_metrics[view_tag] = {
            "mae_avg":              _safe_float(np.mean(mae_c)),
            "rmse_avg":             _safe_float(np.mean(rmse_c)),
            "log_mae_avg":          _safe_float(np.mean(log_mae_c)),
            "mre_avg":              _safe_float(np.mean(mre_c)),
            "mae_per_channel":      [_safe_float(val) for val in mae_c],
            "rmse_per_channel":     [_safe_float(val) for val in rmse_c],
            "log_mae_per_channel":  [_safe_float(val) for val in log_mae_c],
            "mre_per_channel":      [_safe_float(val) for val in mre_c],
        }

    _write_metrics_json(out_dir, all_view_metrics, args_dict={
        "arch":      "velocity_unet",
        "method":    method,
        "mode":      mode,
        "integrator": integrator if mode == "xpred" else None,
        "ode_steps": n_steps if mode != "direct" else None,
        "n_samples": n_samples,
        "c_out":     c_out,
        "y_indices": y_indices,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Shared metrics JSON writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_metrics_json(out_dir: Path, all_view_metrics: dict, args_dict: dict):
    all_mae     = np.mean([m["mae_avg"]     for m in all_view_metrics.values()])
    all_rmse    = np.mean([m["rmse_avg"]    for m in all_view_metrics.values()])
    all_log_mae = np.mean([m["log_mae_avg"] for m in all_view_metrics.values()])
    all_mre     = np.mean([m["mre_avg"]     for m in all_view_metrics.values()])

    metrics = {
        **args_dict,
        "mae_avg_global":     _safe_float(all_mae),
        "rmse_avg_global":    _safe_float(all_rmse),
        "log_mae_avg_global": _safe_float(all_log_mae),
        "mre_avg_global":     _safe_float(all_mre),
        "per_view":           all_view_metrics,
    }
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'─'*70}")
    print(f"  Global Physical MAE  = {all_mae:.6g}")
    print(f"  Global Physical RMSE = {all_rmse:.6g}")
    print(f"  Global Log-space MAE = {all_log_mae:.4f}")
    print(f"  Global Relative err  = {all_mre:.4f}")
    print(f"  Metrics saved: {metrics_path}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()
