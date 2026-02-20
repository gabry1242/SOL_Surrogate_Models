# scripts/verify_mapping_indices.py
# Usage examples:
#   python scripts/verify_mapping_indices.py --data_root . --layout_npz train/global_layout_map.npz --out_dir debug_indices
#   python scripts/verify_mapping_indices.py --data_root . --layout_npz scripts/filled_gap_3/train/global_layout_map.npz --out_dir scripts/filled_gap_3/debug
#
# Outputs:
#   <out_dir>/mesh_indices.png
#   <out_dir>/unrolled_indices.png
#
# Notes:
# - Cell indexing matches your tensor convention field_2d[ix, iy] with shape (104, 50).
# - Annotating all 5200 cells is technically supported but unreadable; set --annotate_stride 1 if you insist.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_layout_npz(path_npz: Path):
    z = np.load(path_npz, allow_pickle=True)

    Wg = int(z["Wg"])
    Hg = int(z["Hg"])
    gap_px = int(z["gap_px"])
    maskg = z["maskg"].astype(np.uint8)

    # discover slots
    slots = sorted({k.split("__")[0] for k in z.files if k.endswith("__mask")})

    slot_dicts = {}
    origins = {}
    for slot in slots:
        u0 = int(z[f"{slot}__u0"])
        v0 = int(z[f"{slot}__v0"])
        origins[slot] = (u0, v0)
        slot_dicts[slot] = {
            "mask": z[f"{slot}__mask"].astype(np.uint8),
            "pix_cell_ix": z[f"{slot}__pix_ix"].astype(np.int32),
            "pix_cell_iy": z[f"{slot}__pix_iy"].astype(np.int32),
            "W": int(z[f"{slot}__W"]),
            "H": int(z[f"{slot}__H"]),
        }

    return {"Wg": Wg, "Hg": Hg, "gap_px": gap_px, "maskg": maskg, "slots": slots, "slot_dicts": slot_dicts, "origins": origins}


def build_cell_id(mode: str, nx: int = 104, ny: int = 50):
    if mode == "linear":
        # id = ix*ny + iy, matches field_2d[ix, iy]
        return (np.arange(nx * ny, dtype=np.int32).reshape(nx, ny))
    if mode == "pair":
        # store packed (ix,iy) in one int for visualization: ix*1000 + iy
        ix = np.arange(nx, dtype=np.int32)[:, None]
        iy = np.arange(ny, dtype=np.int32)[None, :]
        return (ix * 1000 + iy).astype(np.int32)
    raise ValueError(f"Unknown id_mode: {mode}")


def make_unrolled_scalar_map(layout, src_2d):
    Hg = layout["Hg"]; Wg = layout["Wg"]
    out = np.full((Hg, Wg), fill_value=-1, dtype=np.int32)

    for slot in layout["slots"]:
        sd = layout["slot_dicts"][slot]
        u0, v0 = layout["origins"][slot]
        H, W = sd["H"], sd["W"]

        m = sd["mask"].astype(bool)
        pix_ix = sd["pix_cell_ix"]
        pix_iy = sd["pix_cell_iy"]

        sub = out[v0:v0 + H, u0:u0 + W]
        sub[m] = src_2d[pix_ix[m], pix_iy[m]]
        out[v0:v0 + H, u0:u0 + W] = sub

    return out


def plot_unrolled_map(arr, out_path, title, origin="lower", annotate_stride=0):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = plt.gca()

    im = ax.imshow(arr, interpolation="nearest", origin=origin)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("global x (pixels)")
    ax.set_ylabel("global y (pixels)")

    if annotate_stride and annotate_stride > 0:
        H, W = arr.shape
        for y in range(0, H, annotate_stride):
            for x in range(0, W, annotate_stride):
                v = int(arr[y, x])
                if v >= 0:
                    ax.text(x, y, str(v), fontsize=4, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mesh_with_indices(crx, cry, cell_id, out_path: Path, annotate_stride: int, label_mode: str):
    nx, ny, _ = crx.shape

    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = plt.gca()
    ax.set_aspect("equal")

    # draw cell edges
    for ix in range(nx):
        for iy in range(ny):
            xs = crx[ix, iy, :]
            ys = cry[ix, iy, :]
            # close polygon
            xs2 = np.r_[xs, xs[0]]
            ys2 = np.r_[ys, ys[0]]
            ax.plot(xs2, ys2, linewidth=0.25)

    # annotate
    if annotate_stride >= 1:
        for ix in range(0, nx, annotate_stride):
            for iy in range(0, ny, annotate_stride):
                xs = crx[ix, iy, :]
                ys = cry[ix, iy, :]
                cx = float(xs.mean())
                cy = float(ys.mean())
                if label_mode == "id":
                    s = str(int(cell_id[ix, iy]))
                elif label_mode == "ixiy":
                    s = f"{ix},{iy}"
                else:
                    raise ValueError(f"Unknown label_mode: {label_mode}")
                ax.text(cx, cy, s, fontsize=4, ha="center", va="center")

    ax.set_title(f"Physical mesh with indices (annotate_stride={annotate_stride}, label_mode={label_mode})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_unrolled_indices(unrolled_id, out_path: Path, show_grid: bool):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = plt.gca()

    im = ax.imshow(unrolled_id, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Unrolled representation with indices (-1 = gap/unmapped)")
    ax.set_xlabel("global x (pixels)")
    ax.set_ylabel("global y (pixels)")

    if show_grid:
        ax.set_xticks(np.arange(-0.5, unrolled_id.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, unrolled_id.shape[0], 1), minor=True)
        ax.grid(which="minor", linewidth=0.1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".", help="Dataset root containing geometry/")
    ap.add_argument("--layout_npz", type=str, required=True, help="Path to <out_prefix>_layout_map.npz")
    ap.add_argument("--out_dir", type=str, default="debug_indices", help="Output directory for figures")
    ap.add_argument("--id_mode", type=str, default="linear", choices=["linear", "pair"],
                    help="linear: id=ix*50+iy, pair: packed ix*1000+iy")
    ap.add_argument("--label_mode", type=str, default="id", choices=["id", "ixiy"],
                    help="What to draw on the physical mesh figure")
    ap.add_argument("--annotate_stride", type=int, default=4,
                    help="1 annotates every cell (unreadable). 2/4/5 are practical.")
    ap.add_argument("--show_grid", action="store_true", help="Draw pixel grid on unrolled figure (slow, cluttered).")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    layout_path = Path(args.layout_npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    crx = np.load(data_root / "geometry" / "crx.npy")
    cry = np.load(data_root / "geometry" / "cry.npy")

    layout = load_layout_npz(layout_path)

    cell_id = build_cell_id(args.id_mode, nx=crx.shape[0], ny=crx.shape[1])
    nx, ny, _ = crx.shape

    ix_src = np.repeat(np.arange(nx, dtype=np.int32)[:, None], ny, axis=1)  # (104,50)
    iy_src = np.repeat(np.arange(ny, dtype=np.int32)[None, :], nx, axis=0)  # (104,50)

    unrolled_ix = make_unrolled_scalar_map(layout, ix_src)
    unrolled_iy = make_unrolled_scalar_map(layout, iy_src)

    plot_unrolled_map(
        unrolled_ix,
        out_dir / "unrolled_ix.png",
        title="Unrolled ix map (-1 = gap/unmapped)",
        origin="lower",
        annotate_stride=10
    )
    plot_unrolled_map(
        unrolled_iy,
        out_dir / "unrolled_iy.png",
        title="Unrolled iy map (-1 = gap/unmapped)",
        origin="lower",
        annotate_stride=10
    )

    mesh_png = out_dir / "mesh_indices.png"
    unrolled_png = out_dir / "unrolled_indices.png"

    print(f"Saved: {mesh_png}")
    print(f"Saved: {unrolled_png}")


if __name__ == "__main__":
    main()