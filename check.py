import numpy as np
import os

import numpy as np

J_PER_EV = 1.602176634e-19

teJ = np.load(r"train/te_tmp.npy", mmap_mode="r")  # (N,104,50)
tiJ = np.load(r"train/ti_tmp.npy", mmap_mode="r")

Xg = np.load(r"train_tensor/global_X_img_train.npy", mmap_mode="r")  # (N,C,Hg,Wg)
Yg = np.load(r"train_tensor/global_Y_img_train.npy", mmap_mode="r")  # (N,Cout,Hg,Wg)

m = (Xg[:,0] > 0)  # (N,Hg,Wg) boolean

# global tensors are already in eV if you used --te_ti_units joule in builder
te = Yg[:,0]
ti = Yg[:,1]

def summarize(name, arr, mask):
    v = arr[mask]
    p = np.percentile(v, [50, 90, 95, 99, 99.5, 99.9, 99.99])
    print(name, "min", float(v.min()), "max", float(v.max()))
    print(name, "percentiles 50 90 95 99 99.5 99.9 99.99:", p)
    print(name, "count >1e5 eV:", int((v > 1e5).sum()), " / ", v.size)

summarize("te_eV (masked)", te, m)
summarize("ti_eV (masked)", ti, m)

# find the single worst pixel location
idx = np.argmax(te[m])
flat = np.flatnonzero(m)[idx]
i, v, u = np.unravel_index(flat, m.shape)
print("te max at sample i=", int(i), "pixel (v,u)=", int(v), int(u), "value_eV=", float(te[i,v,u]))
