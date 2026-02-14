# scripts/check_mapping_swap.py
import numpy as np

LAYOUT = "global_layout_map.npz"
SPLIT_DIR = "test"
TENSOR_Y = "test_tensor/global_Y_img_test.npy"

def build_global_ixiy(lay):
    maskg = lay["maskg"].astype(bool)
    Hg, Wg = maskg.shape
    pix_ix_g = np.full((Hg, Wg), -1, dtype=np.int32)
    pix_iy_g = np.full((Hg, Wg), -1, dtype=np.int32)

    slots = ["TOP_left", "TOP_right", "MID_left", "MID_center", "MID_right", "Bottom_center"]
    for slot in slots:
        u0 = int(lay[f"{slot}__u0"])
        v0 = int(lay[f"{slot}__v0"])
        m  = lay[f"{slot}__mask"].astype(bool)
        ix = lay[f"{slot}__pix_ix"]
        iy = lay[f"{slot}__pix_iy"]
        H, W = m.shape
        sub_ix = pix_ix_g[v0:v0+H, u0:u0+W]
        sub_iy = pix_iy_g[v0:v0+H, u0:u0+W]
        sub_ix[m] = ix[m]
        sub_iy[m] = iy[m]
        pix_ix_g[v0:v0+H, u0:u0+W] = sub_ix
        pix_iy_g[v0:v0+H, u0:u0+W] = sub_iy
    return pix_ix_g, pix_iy_g

def main():
    lay = np.load(LAYOUT, allow_pickle=False)
    te = np.load(f"{SPLIT_DIR}/te_tmp.npy", mmap_mode="r")
    ti = np.load(f"{SPLIT_DIR}/ti_tmp.npy", mmap_mode="r")
    Y  = np.load(TENSOR_Y, mmap_mode="r")

    maskg = lay["maskg"].astype(bool)
    pix_ix_g, pix_iy_g = build_global_ixiy(lay)

    i = 0
    y_te = Y[i,0]
    y_ti = Y[i,1]

    vs, us = np.where(maskg)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(vs), size=min(500, len(vs)), replace=False)

    max_a_te = 0.0; max_a_ti = 0.0
    max_b_te = 0.0; max_b_ti = 0.0

    for k in idx:
        v = int(vs[k]); u = int(us[k])
        ix = int(pix_ix_g[v,u]); iy = int(pix_iy_g[v,u])
        a_te = float(te[i, ix, iy]); a_ti = float(ti[i, ix, iy])     # (ix,iy)
        b_te = float(te[i, iy, ix]); b_ti = float(ti[i, iy, ix])     # (iy,ix)
        max_a_te = max(max_a_te, abs(float(y_te[v,u]) - a_te))
        max_a_ti = max(max_a_ti, abs(float(y_ti[v,u]) - a_ti))
        max_b_te = max(max_b_te, abs(float(y_te[v,u]) - b_te))
        max_b_ti = max(max_b_ti, abs(float(y_ti[v,u]) - b_ti))

    print("Assuming te[i,ix,iy]:", max_a_te, max_a_ti)
    print("Assuming te[i,iy,ix]:", max_b_te, max_b_ti)

if __name__ == "__main__":
    main()
