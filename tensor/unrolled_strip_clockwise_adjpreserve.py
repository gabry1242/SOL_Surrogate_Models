# ============================================================
# Unroll SOLPS(-like) mesh into multiple rectangular strips of square cells
#
# What this fixes vs your previous attempts
# - Treats side labels (A..N) as *chain side* labels (left/right of a chain),
#   not just "the chain", by classifying which of the two adjacent cell regions
#   is on the requested side using edge orientation + centroid test.
# - Selects the target connected component by requiring it to touch ALL
#   requested side labels (top/bottom/left/right).
# - Builds unrolled (u,v) coordinates from graph distances:
#       u = distance from LEFT boundary cells
#       v = distance from BOTTOM boundary cells
#
# Outputs (OUT_DIR):
# - 0_mesh_chains_side_labels.png
# - debug_label_A.png ... debug_label_N.png   (verifies which cells belong to each side label)
# - 01_*.png, 02_*.png, ...                   (requested unrolled rectangles)
# - diagnostics.txt                            (chain order + inferred rectangle sides)
#
# Notes
# - This is TOPOLOGICAL unroll (graph-distance based), not metric-preserving.
# - If a rectangle spec is given with only 2 sides, missing opposite sides are inferred
#   via the per-chain opposite mapping (e.g., A<->N, B<->H, C<->I, D<->J, E<->K, F<->L, G<->M).
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection

from collections import defaultdict, deque

# ----------------------------
# Adjacency-preserving unroll
# ----------------------------
def unroll_component_adjacency_preserving(target_set, bottom_cells=None):
    """Embed a cut component into an integer (u,v) rectangle while preserving logical adjacency.

    This does NOT use geometric nearest-neighbor. It uses only (ix,iy) index adjacency.

    Conditions for success (otherwise ValueError):
      - The component is "strip-embeddable": columns connect by left/right neighbors.
      - The embedding does not create overlaps in (u,v).
    """
    target_set = set(target_set)

    by_ix = defaultdict(list)
    for (ix, iy) in target_set:
        by_ix[ix].append(iy)

    ix_list = sorted(by_ix.keys())
    if not ix_list:
        raise ValueError('Empty target_set.')

    u_of_ix = {ix: u for u, ix in enumerate(ix_list)}

    bottom_cells = set(bottom_cells) if bottom_cells is not None else set()

    base_bottom = {}
    for ix in ix_list:
        iys = by_ix[ix]
        if bottom_cells:
            cand = [iy for iy in iys if (ix, iy) in bottom_cells]
            base_bottom[ix] = min(cand) if cand else min(iys)
        else:
            base_bottom[ix] = min(iys)

    def has_cell(ix, iy):
        return (ix, iy) in target_set

    # Solve per-column vertical offsets so left/right neighbors align in v.
    offset = {ix_list[0]: 0}
    q = deque([ix_list[0]])
    ix_set = set(ix_list)

    while q:
        ix = q.popleft()
        for ix2 in (ix - 1, ix + 1):
            if ix2 not in ix_set or ix2 in offset:
                continue

            shared = []
            probe = by_ix[ix] if len(by_ix[ix]) <= len(by_ix[ix2]) else by_ix[ix2]
            for iy in probe:
                if has_cell(ix, iy) and has_cell(ix2, iy):
                    shared.append(iy)
            if not shared:
                raise ValueError(
                    f'No shared left/right neighbors between columns ix={ix} and ix={ix2} inside this component.'
                )

            offset[ix2] = offset[ix] + (base_bottom[ix2] - base_bottom[ix])
            q.append(ix2)

    if len(offset) != len(ix_list):
        missing = [ix for ix in ix_list if ix not in offset]
        raise ValueError(f'Component not strip-embeddable (unreached columns): {missing}')

    coords = {}
    v_vals = []
    for (ix, iy) in target_set:
        u = u_of_ix[ix]
        v = (iy - base_bottom[ix]) + offset[ix]
        coords[(ix, iy)] = (u, v)
        v_vals.append(v)

    v_min = min(v_vals)
    v_max = max(v_vals) - v_min
    for c in list(coords.keys()):
        u, v = coords[c]
        coords[c] = (u, v - v_min)

    # overlap check
    seen = {}
    for c, uv in coords.items():
        if uv in seen:
            raise ValueError(f'Overlap at {uv}: cells {seen[uv]} and {c}. Component not strip-embeddable.')
        seen[uv] = c

    W = len(ix_list)
    H = v_max + 1
    return coords, (W, H)

def cell_side_edgekeys(ix, iy, crx, cry):
    """
    Return edge keys for the 4 sides of cell (ix,iy) using the SAME discretization
    as the rest of the code (vkey_xy with global TOL, and edge_key).
    """
    x = crx[ix, iy, :]
    y = cry[ix, iy, :]

    # same corner order as CELL_VERTS
    p0 = (x[0], y[0])
    p1 = (x[1], y[1])
    p2 = (x[3], y[3])
    p3 = (x[2], y[2])

    k0 = vkey_xy(p0[0], p0[1])  # uses global TOL
    k1 = vkey_xy(p1[0], p1[1])
    k2 = vkey_xy(p2[0], p2[1])
    k3 = vkey_xy(p3[0], p3[1])

    return {
        "left":   edge_key(k0, k3),
        "right":  edge_key(k1, k2),
        "bottom": edge_key(k0, k1),
        "top":    edge_key(k3, k2),
    }

def build_chain_color_map(m):
    # One color per chain, reused for both side labels of that chain.
    cmap = plt.cm.get_cmap('tab20', max(1, m))
    return {i: cmap(i) for i in range(m)}


def build_cut_side_colors(target_set, return_chain_ids=False):
    """Return per-cell-per-side colors for CUT_EDGES, using geometric edge keys (fast + correct).

    Returns:
      side_colors: dict[(cell, side)] -> RGBA
      if return_chain_ids=True also returns:
      side_chain: dict[(cell, side)] -> chain_id
    """
    target_set = set(target_set)
    chain_colors = build_chain_color_map(m)

    # edge_key -> chain_id
    edge_to_chain = {}
    for ci, es in enumerate(CHAIN_EDGESETS):
        for ek in es:
            edge_to_chain[ek] = ci

    # For cells in this component, map each geometric edge_key to (cell, side).
    # Use the SAME hashing (vkey_xy + edge_key) as EDGE_TO_CELLS / INTERNAL_EDGES / CUT_EDGES.
    edge_to_sides = defaultdict(list)  # ek -> [(cell, side), ...]

    for cell in target_set:
        ix, iy = cell
        verts = CELL_VERTS[cell]

        # verts order: [p0, p1, p2, p3] where
        # p0=(x0,y0), p1=(x1,y1), p2=(x3,y3), p3=(x2,y2)
        p0 = verts[0]
        p1 = verts[1]
        p2 = verts[2]
        p3 = verts[3]

        k0 = vkey_xy(p0[0], p0[1])
        k1 = vkey_xy(p1[0], p1[1])
        k2 = vkey_xy(p2[0], p2[1])
        k3 = vkey_xy(p3[0], p3[1])

        edge_to_sides[edge_key(k0, k1)].append((cell, "bottom"))
        edge_to_sides[edge_key(k3, k2)].append((cell, "top"))
        edge_to_sides[edge_key(k0, k3)].append((cell, "left"))
        edge_to_sides[edge_key(k1, k2)].append((cell, "right"))

    side_colors = {}
    side_chain = {}

    # Only color edges that are actual cut edges of some chain.
    for ek, ci in edge_to_chain.items():
        if ek not in edge_to_sides:
            continue
        col = chain_colors[ci]
        for cell_side in edge_to_sides[ek]:
            side_colors[cell_side] = col
            side_chain[cell_side] = ci

    if return_chain_ids:
        return side_colors, side_chain
    return side_colors

# ----------------------------
# User knobs
# ----------------------------
SQUARE_GAP = 0.08

TOL = 1e-6
W_COUNT = 1.0
W_LOWY = 0.25

MAX_STEPS = 20000
CHAIN_LW = 4.0
MESH_LW = 0.12

GEOM_DIR = "geometry"
OUT_DIR = "figs_unrolled_strip"


# ----------------------------
# Requested rectangles
# ----------------------------
# Your assumed clockwise chain-side assignment:
# chain0: left=N right=A
# chain1: left=H right=B
# chain2: left=I right=C
# chain3: left=J right=D
# chain4: left=K right=E
# chain5: left=L right=F
# chain6: left=M right=G
#
# Requests you gave:
# - top A, bottom J, side H (other side inferred as opposite(H)=B)
# - top C, bottom I, side H (other side inferred as B)
# - top D, left K (bottom inferred as opposite(D)=J, right inferred as opposite(K)=E)
# - right E, top L (bottom inferred as opposite(L)=F, left inferred as opposite(E)=K)
# - top F, left M (bottom=L, right=G)
# - right G, top N (bottom=A, left=M)
RECT_SPECS = [
    # Keep the original that worked for you (top/bottom only)

    dict(name="A_top_J_bottom_H_left", top="A", bottom="J", left="H"),
    dict(name="C_top_I_bottom_B_left", top="C", bottom="I", right="B"),
    dict(name="D_top_K_left", top="K", left="D"),
    dict(name="L_top_E_right", top="E", right="L"),
    dict(name="F_top_M_left", top="M", left="F"),
    dict(name="N_top_G_right", top="G", right="N"),
    dict(name="D_top_K_left_fullrep", top="D", left="K"),
    dict(name="L_top_E_right_fullrep", top="L", right="E"),
    dict(name="F_top_M_left_fullrep", top="F", left="M"),
    dict(name="N_top_G_right_fullrep", top="N", right="G"),
]


# ----------------------------
# Geometry load
# ----------------------------
crx = np.load(os.path.join(GEOM_DIR, "crx.npy"))
cry = np.load(os.path.join(GEOM_DIR, "cry.npy"))

nx, ny, ns = crx.shape[0], crx.shape[1], crx.shape[2]
XLIM = [float(np.min(crx)), float(np.max(crx))]
YLIM = [float(np.min(cry)), float(np.max(cry))]


# ----------------------------
# Utilities
# ----------------------------
def vkey_xy(x, y, tol=TOL):
    return (int(np.round(float(x) / tol)), int(np.round(float(y) / tol)))


def key_to_xy(k, tol=TOL):
    return np.array([k[0] * tol, k[1] * tol], dtype=float)


def norm(v):
    return float(np.linalg.norm(v))


def turn_cost(prev_dir, cand_dir):
    a = norm(prev_dir)
    b = norm(cand_dir)
    if a < 1e-15 or b < 1e-15:
        return 1e9
    c = float(np.dot(prev_dir, cand_dir) / (a * b))
    c = max(-1.0, min(1.0, c))
    return 1.0 - c


def edge_key(a, b):
    return tuple(sorted((a, b)))


# ----------------------------
# Build polygons + per-cell vertices + centroids
# ----------------------------
def build_cells_and_vertices(crx_, cry_):
    cells = []
    cell_verts = {}
    cell_cent = {}
    for ix in range(nx):
        for iy in range(ny):
            x = crx_[ix, iy, :]
            y = cry_[ix, iy, :]
            verts = np.array(
                [[x[0], y[0]],
                 [x[1], y[1]],
                 [x[3], y[3]],
                 [x[2], y[2]]],
                dtype=float
            )
            cells.append(patches.Polygon(verts, closed=True))
            cell_verts[(ix, iy)] = verts
            cell_cent[(ix, iy)] = np.mean(verts, axis=0)
    return cells, cell_verts, cell_cent


CELLS, CELL_VERTS, CELL_CENT = build_cells_and_vertices(crx, cry)



import math

def order_cells_bfs(target_set, cell_adj, sources=None):
    """
    Deterministic BFS order over target_set.
    sources: iterable of starting cells; if None, pick lexicographically smallest cell.
    """
    target_set = set(target_set)

    if sources is None:
        start = min(target_set)
        sources = [start]
    else:
        sources = [s for s in sources if s in target_set]
        if len(sources) == 0:
            sources = [min(target_set)]

    seen = set()
    q = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            q.append(s)

    order = []
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        order.append(u)
        # sorted neighbors makes it reproducible
        for v in sorted(cell_adj[u]):
            if v in target_set and v not in seen:
                seen.add(v)
                q.append(v)

    # target might not be connected under your current adjacency; enforce full coverage
    if len(order) < len(target_set):
        for s in sorted(target_set):
            if s in seen:
                continue
            seen.add(s)
            q = [s]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                order.append(u)
                for v in sorted(cell_adj[u]):
                    if v in target_set and v not in seen:
                        seen.add(v)
                        q.append(v)

    return order


def pack_order_to_rectangle(order, width=None):
    """
    Map a linear order to integer lattice coordinates (u,v).
    If width None, use round(sqrt(N)).
    Returns coords dict and (W,H,N).
    """
    N = len(order)
    if N == 0:
        return {}, (0, 0, 0)

    if width is None:
        width = int(round(math.sqrt(N)))
        width = max(1, min(width, N))

    W = width
    H = int(math.ceil(N / W))

    coords = {}
    for k, c in enumerate(order):
        u = k % W
        v = k // W
        coords[c] = (u, v)

    return coords, (W, H, N)
# ----------------------------
# X-point selection (vertex multiplicity + low-y bias)
# ----------------------------
def find_xpoint(cell_verts):
    counts = {}
    sums = {}
    for verts in cell_verts.values():
        for p in verts:
            k = vkey_xy(p[0], p[1])
            counts[k] = counts.get(k, 0) + 1
            sums[k] = sums.get(k, np.zeros(2)) + p

    keys = list(counts.keys())
    xs = np.array([sums[k][0] / counts[k] for k in keys], dtype=float)
    ys = np.array([sums[k][1] / counts[k] for k in keys], dtype=float)
    cs = np.array([counts[k] for k in keys], dtype=float)

    c_norm = (cs - cs.min()) / (cs.max() - cs.min() + 1e-12)
    lowy = -ys
    lowy_norm = (lowy - lowy.min()) / (lowy.max() - lowy.min() + 1e-12)

    score = W_COUNT * c_norm + W_LOWY * lowy_norm
    idx = int(np.argmax(score))
    return np.array([xs[idx], ys[idx]], dtype=float), int(cs[idx])


XPOINT, XCOUNT = find_xpoint(CELL_VERTS)
XK = vkey_xy(XPOINT[0], XPOINT[1])


# ----------------------------
# Edge -> cells map + internal edges
# ----------------------------
def build_edge_to_cells(cell_verts):
    edge_map = {}
    for (ix, iy), verts in cell_verts.items():
        for i in range(4):
            p0 = verts[i]
            p1 = verts[(i + 1) % 4]
            k0 = vkey_xy(p0[0], p0[1])
            k1 = vkey_xy(p1[0], p1[1])
            ek = edge_key(k0, k1)
            edge_map.setdefault(ek, []).append((ix, iy))
    return edge_map


EDGE_TO_CELLS = build_edge_to_cells(CELL_VERTS)
INTERNAL_EDGES = {ek for ek, cells in EDGE_TO_CELLS.items() if len(cells) == 2}


# ----------------------------
# Vertex adjacency (internal edges only)
# ----------------------------
def build_vertex_adjacency(internal_edges):
    adj = {}
    for (a, b) in internal_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


VADJ = build_vertex_adjacency(INTERNAL_EDGES)


# ----------------------------
# Incident internal edges to X-point
# ----------------------------
def incident_internal_edges(xk, vadj):
    nbrs = sorted(list(vadj.get(xk, [])))
    return [(xk, n) for n in nbrs]


START_EDGES = incident_internal_edges(XK, VADJ)


# ----------------------------
# Trace a chain from one incident edge (straightest continuation)
# ----------------------------
def trace_chain_from_edge(xk, start_edge, vadj, max_steps=MAX_STEPS):
    u, v = start_edge
    path = [u, v]
    prev = u
    curr = v
    prev_dir = key_to_xy(curr) - key_to_xy(prev)

    for _ in range(max_steps):
        if curr == xk:
            break

        nbrs = [w for w in vadj.get(curr, []) if w != prev]
        if len(nbrs) == 0:
            break

        best_w = None
        best_cost = 1e18
        curr_xy = key_to_xy(curr)

        for w in nbrs:
            cand_dir = key_to_xy(w) - curr_xy
            c = turn_cost(prev_dir, cand_dir)
            if c < best_cost:
                best_cost = c
                best_w = w

        nxt = best_w
        path.append(nxt)
        prev_dir = key_to_xy(nxt) - key_to_xy(curr)
        prev, curr = curr, nxt

        if curr == xk:
            break

    return path


RAW_CHAINS = [trace_chain_from_edge(XK, (XK, n), VADJ) for (_, n) in START_EDGES]


# ----------------------------
# De-duplicate chains by edge-set
# ----------------------------
def chain_to_edge_set(chain):
    es = set()
    for i in range(len(chain) - 1):
        es.add(edge_key(chain[i], chain[i + 1]))
    return es


def dedup_chains(chains):
    kept = []
    kept_sets = []
    for ch in chains:
        es = chain_to_edge_set(ch)
        if any(es == ks for ks in kept_sets):
            continue
        kept.append(ch)
        kept_sets.append(es)
    return kept, kept_sets


CHAINS, CHAIN_EDGESETS = dedup_chains(RAW_CHAINS)




# ----------------------------
# CLOCKWISE order around X-point
# ----------------------------
def chain_angle_from_xpoint(chain):
    if len(chain) < 2:
        return 0.0
    p0 = key_to_xy(chain[0])
    p1 = key_to_xy(chain[1])
    d = p1 - p0
    return float(np.arctan2(d[1], d[0]))


m = len(CHAINS)
if m != 7:
    raise RuntimeError(f"Expected 7 unique chains. Got m={m}.")

angles = np.array([chain_angle_from_xpoint(ch) for ch in CHAINS], dtype=float)
order_cw = np.argsort(-angles)
CHAINS = [CHAINS[i] for i in order_cw]
CHAIN_EDGESETS = [CHAIN_EDGESETS[i] for i in order_cw]
angles = angles[order_cw]

chain_colors = build_chain_color_map(m)

edge_to_chain = {}
for ci, es in enumerate(CHAIN_EDGESETS):
    for ek in es:
        edge_to_chain[ek] = ci
# ----------------------------
# Assign 14 side-labels A..N CLOCKWISE
# ----------------------------
SIDE_LABELS = list("ABCDEFGHIJKLMN")

WEDGE_LABEL_1 = {i: SIDE_LABELS[i] for i in range(m)}
WEDGE_LABEL_2 = {i: SIDE_LABELS[i + m] for i in range(m)}

CHAIN_RIGHT_LABEL = {i: WEDGE_LABEL_1[i] for i in range(m)}
CHAIN_LEFT_LABEL = {i: WEDGE_LABEL_2[(i - 1) % m] for i in range(m)}

SIDE_TO_CHAIN = {}
LABEL_SIDE = {}
for i in range(m):
    SIDE_TO_CHAIN[CHAIN_LEFT_LABEL[i]] = i
    SIDE_TO_CHAIN[CHAIN_RIGHT_LABEL[i]] = i
    LABEL_SIDE[CHAIN_LEFT_LABEL[i]] = "left"
    LABEL_SIDE[CHAIN_RIGHT_LABEL[i]] = "right"

OPPOSITE = {}
for i in range(m):
    L = CHAIN_LEFT_LABEL[i]
    R = CHAIN_RIGHT_LABEL[i]
    OPPOSITE[L] = R
    OPPOSITE[R] = L


def chain_oriented_edges(chain):
    return [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]


CHAIN_OEDGES = [chain_oriented_edges(ch) for ch in CHAINS]


# ----------------------------
# Cell adjacency graph, then CUT adjacency along ALL chain edges
# ----------------------------
def build_cell_adjacency(edge_to_cells, internal_edges):
    adj = {(ix, iy): set() for ix in range(nx) for iy in range(ny)}
    for ek in internal_edges:
        c2 = edge_to_cells.get(ek, [])
        if len(c2) != 2:
            continue
        a, b = c2[0], c2[1]
        adj[a].add(b)
        adj[b].add(a)
    return adj


CELL_ADJ = build_cell_adjacency(EDGE_TO_CELLS, INTERNAL_EDGES)

CUT_EDGES = set()
for es in CHAIN_EDGESETS:
    CUT_EDGES |= es

for ek in CUT_EDGES:
    c2 = EDGE_TO_CELLS.get(ek, [])
    if len(c2) != 2:
        continue
    a, b = c2[0], c2[1]
    CELL_ADJ[a].discard(b)
    CELL_ADJ[b].discard(a)


# ----------------------------
# Connected components after cuts
# ----------------------------
def connected_components_cells(cell_adj):
    comp_id = {}
    comps = []
    cid = 0
    for ix in range(nx):
        for iy in range(ny):
            node = (ix, iy)
            if node in comp_id:
                continue
            q = [node]
            comp_id[node] = cid
            nodes = [node]
            while q:
                u = q.pop()
                for v in cell_adj[u]:
                    if v not in comp_id:
                        comp_id[v] = cid
                        q.append(v)
                        nodes.append(v)
            comps.append(nodes)
            cid += 1
    return comp_id, comps


CELL_COMP_ID, CELL_COMPS = connected_components_cells(CELL_ADJ)


# ----------------------------
# Side-aware: cells for a side label
# ----------------------------
def cell_side_of_oriented_edge(k0, k1, cell):
    p0 = key_to_xy(k0)
    p1 = key_to_xy(k1)
    t = p1 - p0
    if norm(t) < 1e-15:
        return None
    nL = np.array([-t[1], t[0]], dtype=float)
    mid = 0.5 * (p0 + p1)
    c = CELL_CENT[cell]
    s = float(np.dot(c - mid, nL))
    if abs(s) < 1e-14:
        return None
    return "left" if s > 0.0 else "right"


def cells_for_side_label(label):
    if label not in SIDE_TO_CHAIN:
        raise RuntimeError(f"Unknown side label {label}. Known: {sorted(SIDE_TO_CHAIN.keys())}")

    ci = SIDE_TO_CHAIN[label]
    want = LABEL_SIDE[label]
    out = set()

    for (k0, k1) in CHAIN_OEDGES[ci]:
        ek = edge_key(k0, k1)
        c2 = EDGE_TO_CELLS.get(ek, [])
        if len(c2) != 2:
            continue

        cA, cB = c2[0], c2[1]
        sA = cell_side_of_oriented_edge(k0, k1, cA)
        sB = cell_side_of_oriented_edge(k0, k1, cB)

        pick = None
        if sA == want:
            pick = cA
        elif sB == want:
            pick = cB
        else:
            # fallback: global normal from first chain segment at X-point
            p0 = key_to_xy(CHAINS[ci][0])
            p1 = key_to_xy(CHAINS[ci][1])
            d = p1 - p0
            if norm(d) < 1e-15:
                continue
            d = d / norm(d)
            nL0 = np.array([-d[1], d[0]], dtype=float)
            mid = 0.5 * (key_to_xy(k0) + key_to_xy(k1))
            sA0 = float(np.dot(CELL_CENT[cA] - mid, nL0))
            sB0 = float(np.dot(CELL_CENT[cB] - mid, nL0))
            if want == "left":
                pick = cA if sA0 > sB0 else cB
            else:
                pick = cA if sA0 < sB0 else cB

        out.add(pick)

    return out


LABEL_TO_CELLS = {lab: cells_for_side_label(lab) for lab in SIDE_TO_CHAIN.keys()}


def component_touches_label(comp_set, label):
    return len(comp_set.intersection(LABEL_TO_CELLS[label])) > 0


def pick_component_by_labels(required_labels):
    req = list(required_labels)
    candidates = []

    for comp_nodes in CELL_COMPS:
        s = set(comp_nodes)
        if all(component_touches_label(s, lab) for lab in req):
            candidates.append(comp_nodes)

    if len(candidates) == 0:
        all_labs = sorted(
            {lab for lab, cells in LABEL_TO_CELLS.items()
             if any(len(set(comp).intersection(cells)) > 0 for comp in CELL_COMPS)}
        )
        raise RuntimeError(
            f"No component touches all required side labels {req}. "
            f"Side labels seen on any component: {all_labs}"
        )

    candidates.sort(key=lambda L: -len(L))
    return set(candidates[0])


# ----------------------------
# BFS distances on CUT cell graph
# ----------------------------
def multi_source_bfs_dist(target_set, cell_adj, sources):
    dist = {c: np.inf for c in target_set}
    q = []
    for s in sources:
        if s in dist:
            dist[s] = 0
            q.append(s)

    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        du = dist[u]
        for v in cell_adj[u]:
            if v not in target_set:
                continue
            if dist[v] == np.inf:
                dist[v] = du + 1
                q.append(v)
    return dist


# ----------------------------
# Plots
# ----------------------------
def plot_mesh_with_chains_and_labels(path_png):
    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_subplot(1, 1, 1)

    pc = PatchCollection(CELLS, facecolor="none", edgecolor="black", linewidth=MESH_LW)
    ax.add_collection(pc)

    chain_colors = build_chain_color_map(m)

    for i, ch in enumerate(CHAINS):
        pts = np.array([key_to_xy(k) for k in ch], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=CHAIN_LW, zorder=6, color=chain_colors[i])

        p0 = pts[0]
        p1 = pts[1]
        d = p1 - p0
        if norm(d) < 1e-15:
            continue
        d = d / norm(d)
        nL = np.array([-d[1], d[0]], dtype=float)

        label_offset = 0.3
        posL = p0 + nL * label_offset
        posR = p0 - nL * label_offset
#Needed if I want to add the letters 
        # ax.text(posL[0], posL[1], CHAIN_LEFT_LABEL[i],
        #         fontsize=18, fontweight="bold", ha="center", va="center")
        # ax.text(posR[0], posR[1], CHAIN_RIGHT_LABEL[i],
        #         fontsize=18, fontweight="bold", ha="center", va="center")

    ax.scatter([XPOINT[0]], [XPOINT[1]], s=150, color="red", zorder=7)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Chains + side labels (CLOCKWISE) | unique chains={m} | Xcount={XCOUNT}")

    fig.tight_layout()
    ax.set_axis_off()
    fig.savefig(path_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_debug_label(label, path_png):
    cells = LABEL_TO_CELLS[label]
    fig = plt.figure(figsize=(7, 14))
    ax = fig.add_subplot(1, 1, 1)

    pc = PatchCollection(CELLS, facecolor="none", edgecolor="black", linewidth=0.2)
    ax.add_collection(pc)

    for c in cells:
        verts = CELL_VERTS[c]
        poly = patches.Polygon(verts, closed=True, facecolor="none", edgecolor="tab:orange", linewidth=1.2)
        ax.add_patch(poly)

    ax.scatter([XPOINT[0]], [XPOINT[1]], s=90, color="red", zorder=6)
    ci = SIDE_TO_CHAIN[label]
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Label {label} -> chain {ci} side={LABEL_SIDE[label]} | incident cells={len(cells)}")

    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def plot_unrolled_grid(coords, title, path_png, edge_side_colors=None):
    """Draw an unrolled strip with optional per-cell per-side edge coloring.

    coords: dict[(ix,iy)] -> (u,v)
    edge_side_colors: dict[((ix,iy), side)] -> RGBA, side in {left,right,top,bottom}
                     Only edges present in this dict are colored; others are light gray.
    """
    edge_side_colors = edge_side_colors or {}

    side = 1.0
    step = side * (1.0 + SQUARE_GAP)

    u_max = max(u for (u, v) in coords.values())
    v_max = max(v for (u, v) in coords.values())

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)

    default_ec = (0.7, 0.7, 0.7, 1.0)
    lw_default = 1.2
    lw_colored = 3.5

    for cell, (u, v) in coords.items():
        x0 = u * step
        y0 = v * step

        # draw 4 sides explicitly so they can have independent colors
        # bottom
        key = (cell, 'bottom')
        ax.plot([x0, x0 + side], [y0, y0],
                color=edge_side_colors.get((cell, 'bottom'), default_ec), linewidth=lw_colored if key in edge_side_colors else lw_default)
        # top
        key = (cell, 'top')
        ax.plot([x0, x0 + side], [y0 + side, y0 + side],
                color=edge_side_colors.get((cell, 'top'), default_ec), linewidth=lw_colored if key in edge_side_colors else lw_default)
        # left
        key = (cell, 'left')
        ax.plot([x0, x0], [y0, y0 + side],
                color=edge_side_colors.get((cell, 'left'), default_ec), linewidth=lw_colored if key in edge_side_colors else lw_default)
        # right
        key = (cell, 'right')
        ax.plot([x0 + side, x0 + side], [y0, y0 + side],
                color=edge_side_colors.get((cell, 'right'), default_ec), linewidth=lw_colored if key in edge_side_colors else lw_default)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    ax.set_xlim(-step, (u_max + 2) * step)
    ax.set_ylim(-step, (v_max + 2) * step)

    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    ax.set_axis_off()
    fig.savefig(path_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ----------------------------
# Rectangle build
# ----------------------------
def component_boundary_cells(target_set):
    """
    Cells on the boundary of the selected component in the CUT adjacency graph:
    a cell is boundary if it has at least one neighbor (in the full grid) that is not in target_set.
    """
    out = set()
    for c in target_set:
        for n in CELL_ADJ[c]:
            if n not in target_set:
                out.add(c)
                break
    return out

def normalize_rect_spec(spec):
    """
    Do NOT infer opposite sides. Missing sides are OPEN (unlabeled physical boundaries).
    Only validate labels that are provided.
    """
    out = dict(spec)

    for k in ["top", "bottom", "left", "right"]:
        if k in out:
            if out[k] not in SIDE_TO_CHAIN:
                raise RuntimeError(f"Unknown label '{out[k]}' in spec {spec.get('name','(unnamed)')}.")

    if not any(k in out for k in ["top", "bottom", "left", "right"]):
        raise RuntimeError(f"Spec {spec.get('name','(unnamed)')} has no sides.")

    return out

def _coords_bounds(coords):
    us = [uv[0] for uv in coords.values()]
    vs = [uv[1] for uv in coords.values()]
    return min(us), max(us), min(vs), max(vs)


def _boundary_counts_for_chain(coords, side_chain, chain_id):
    """Count how many cut-edges of chain_id lie on each OUTER boundary in the current embedding."""
    umin, umax, vmin, vmax = _coords_bounds(coords)
    counts = {"left": 0, "right": 0, "bottom": 0, "top": 0}

    for (cell, side), ci in side_chain.items():
        if ci != chain_id:
            continue
        u, v = coords.get(cell, (None, None))
        if u is None:
            continue

        if side == "left" and u == umin:
            counts["left"] += 1
        elif side == "right" and u == umax:
            counts["right"] += 1
        elif side == "bottom" and v == vmin:
            counts["bottom"] += 1
        elif side == "top" and v == vmax:
            counts["top"] += 1

    return counts


def _dominant_boundary_for_chain(coords, side_chain, chain_id):
    counts = _boundary_counts_for_chain(coords, side_chain, chain_id)
    best = max(counts.values())
    if best <= 0:
        return None
    # deterministic tie-break
    for k in ("left", "right", "bottom", "top"):
        if counts[k] == best:
            return k
    return None


def _flip_x(coords, side_dicts):
    """Mirror horizontally: u -> umax-u ; left<->right for any (cell,side) keyed dict."""
    umin, umax, vmin, vmax = _coords_bounds(coords)
    new_coords = {c: (umax - u, v) for c, (u, v) in coords.items()}

    def flip_side(s):
        if s == "left":
            return "right"
        if s == "right":
            return "left"
        return s

    new_side_dicts = []
    for d in side_dicts:
        nd = {}
        for (cell, side), val in d.items():
            nd[(cell, flip_side(side))] = val
        new_side_dicts.append(nd)

    return new_coords, new_side_dicts


def _flip_y(coords, side_dicts):
    """Mirror vertically: v -> vmax-v ; top<->bottom for any (cell,side) keyed dict."""
    umin, umax, vmin, vmax = _coords_bounds(coords)
    new_coords = {c: (u, vmax - v) for c, (u, v) in coords.items()}

    def flip_side(s):
        if s == "top":
            return "bottom"
        if s == "bottom":
            return "top"
        return s

    new_side_dicts = []
    for d in side_dicts:
        nd = {}
        for (cell, side), val in d.items():
            nd[(cell, flip_side(side))] = val
        new_side_dicts.append(nd)

    return new_coords, new_side_dicts


def build_unrolled_strip_rect(spec, out_png):
    spec = normalize_rect_spec(spec)
    top = spec.get('top', None)
    bottom = spec.get('bottom', None)
    left = spec.get('left', None)
    right = spec.get('right', None)

    req = []
    if top is not None:
        req.append(top)
    if bottom is not None:
        req.append(bottom)
    if left is not None:
        req.append(left)
    if right is not None:
        req.append(right)

    target_set = pick_component_by_labels(req)

    bottom_cells = None
    if bottom is not None:
        bottom_cells = [c for c in LABEL_TO_CELLS[bottom] if c in target_set]

    coords, (W, H) = unroll_component_adjacency_preserving(target_set, bottom_cells=bottom_cells)

    cut_side_colors, cut_side_chain = build_cut_side_colors(target_set, return_chain_ids=True)

    # Enforce orientation so requested side labels actually end up on the requested rectangle sides.
    # Only flips are used (no rotation): preserves “top/bottom is top/bottom”.
    orient = "id"

    def _chain_id(label):
        return SIDE_TO_CHAIN[label] if label is not None else None

    want_left_ci = _chain_id(left) if left is not None else None
    want_right_ci = _chain_id(right) if right is not None else None
    want_top_ci = _chain_id(top) if top is not None else None
    want_bottom_ci = _chain_id(bottom) if bottom is not None else None

    # Horizontal flip decision (left/right)
    if want_left_ci is not None:
        dom = _dominant_boundary_for_chain(coords, cut_side_chain, want_left_ci)
        if dom == "right":
            coords, (cut_side_colors, cut_side_chain) = _flip_x(coords, [cut_side_colors, cut_side_chain])
            orient = "fx" if orient == "id" else orient + "+fx"

    if want_right_ci is not None:
        dom = _dominant_boundary_for_chain(coords, cut_side_chain, want_right_ci)
        if dom == "left":
            coords, (cut_side_colors, cut_side_chain) = _flip_x(coords, [cut_side_colors, cut_side_chain])
            orient = "fx" if orient == "id" else orient + "+fx"

    # Vertical flip decision (top/bottom)
    if want_top_ci is not None:
        dom = _dominant_boundary_for_chain(coords, cut_side_chain, want_top_ci)
        if dom == "bottom":
            coords, (cut_side_colors, cut_side_chain) = _flip_y(coords, [cut_side_colors, cut_side_chain])
            orient = "fy" if orient == "id" else orient + "+fy"

    if want_bottom_ci is not None:
        dom = _dominant_boundary_for_chain(coords, cut_side_chain, want_bottom_ci)
        if dom == "top":
            coords, (cut_side_colors, cut_side_chain) = _flip_y(coords, [cut_side_colors, cut_side_chain])
            orient = "fy" if orient == "id" else orient + "+fy"

    print("colored sides:", len(cut_side_colors))


    # title = (
    #     f"Adjacency-preserving unroll | W={W} H={H} | orient={orient} "
    #     f"| top={top or 'OPEN'} bottom={bottom or 'OPEN'} left={left or 'OPEN'} right={right or 'OPEN'} "
    #     f"| cells={len(coords)}"
    # )
    title = ""
    plot_unrolled_grid(coords, title, out_png, edge_side_colors=cut_side_colors)
    print(f"{spec.get('name','(unnamed)')}: adjacency-preserving W={W} H={H} cells={len(coords)}")


def unroll_all_cut_components(out_dir):
    """Attempt adjacency-preserving unroll for every connected component after chain cuts.

    Saves only components that touch at least one side-label set (A..N) to avoid dumping irrelevant regions.
    """
    os.makedirs(out_dir, exist_ok=True)

    label_union = set()
    for cells in LABEL_TO_CELLS.values():
        label_union |= set(cells)

    saved = 0
    for cid, nodes in enumerate(CELL_COMPS):
        target_set = set(nodes)
        if not (target_set & label_union):
            continue

        try:
            coords, (W, H) = unroll_component_adjacency_preserving(target_set)
        except Exception:
            continue

        cut_side_colors = build_cut_side_colors(target_set)
        out_png = os.path.join(out_dir, f"component_{cid:03d}_W{W}_H{H}_N{len(coords)}.png")
        # title = f"Component {cid} adjacency-preserving | W={W} H={H} | cells={len(coords)}"
        title = ""
        plot_unrolled_grid(coords, title, out_png, edge_side_colors=cut_side_colors)
        saved += 1

    print(f"Saved {saved} adjacency-preserving component unrolls to {out_dir}")


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_mesh_with_chains_and_labels(os.path.join(OUT_DIR, "0_mesh_chains_side_labels.png"))

    for lab in sorted(SIDE_TO_CHAIN.keys()):
        plot_debug_label(lab, os.path.join(OUT_DIR, f"debug_label_{lab}.png"))

    unroll_all_cut_components(os.path.join(OUT_DIR, 'all_components'))

    for i, spec in enumerate(RECT_SPECS, start=1):
        out_png = os.path.join(OUT_DIR, f"{i:02d}_{spec['name']}.png")
        build_unrolled_strip_rect(spec, out_png)
        

    diag_path = os.path.join(OUT_DIR, "diagnostics.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(f"XPOINT: x={XPOINT[0]:.8f} y={XPOINT[1]:.8f} multiplicity={XCOUNT}\n")
        f.write(f"Unique chains after de-dup: {m}\n\n")
        f.write("Chains in CLOCKWISE order (by first segment angle):\n")
        for i in range(m):
            f.write(f"  chain{i}: angle={angles[i]:+.6f} rad | left={CHAIN_LEFT_LABEL[i]} right={CHAIN_RIGHT_LABEL[i]}\n")

        f.write("\nOpposite label mapping (same chain):\n")
        for lab in sorted(OPPOSITE.keys()):
            f.write(f"  {lab} <-> {OPPOSITE[lab]}\n")

        f.write("\nRequested rectangles (after inference):\n")
        for spec in RECT_SPECS:
            nspec = normalize_rect_spec(spec)
            f.write(
                f" {spec.get('name','(unnamed)')}: "
                f"top={nspec.get('top','OPEN')} "
                f"bottom={nspec.get('bottom','OPEN')} "
                f"left={nspec.get('left','OPEN')} "
                f"right={nspec.get('right','OPEN')}\n"
                )

    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()