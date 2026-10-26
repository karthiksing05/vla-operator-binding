"""PLAN_SOLVE S1: a visual check of the rendered masks (one train and one dev frame) -> results/bind/s1/check/*.png."""
import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
od = os.path.join(ROOT, "results", "bind", "s1", "check")
os.makedirs(od, exist_ok=True)
for f in sys.argv[1:]:
    z = np.load(f, allow_pickle=True)
    names = [str(x) for x in z["body_names"]] + [str(x) for x in z["site_names"]]
    cov = np.concatenate([z["body_cov"], z["site_cov"]]).astype(np.float32)
    keep = [i for i, n in enumerate(names) if cov[i].sum() > 0 and "init_region" not in n][:23]
    fig, ax = plt.subplots(4, 6, figsize=(18, 12))
    ax = ax.ravel()
    ax[0].imshow(z["img"]); ax[0].set_title(str(z["tid"]), fontsize=8)
    for a, i in zip(ax[1:], keep):
        a.imshow(z["img"]); a.imshow(np.kron(cov[i].reshape(16, 16), np.ones((14, 14))), alpha=0.5, cmap="jet", vmin=0, vmax=1)
        a.set_title(names[i], fontsize=7)
    for a in ax:
        a.axis("off")
    out = os.path.join(od, os.path.basename(f).replace(".npz", ".png"))
    fig.savefig(out, dpi=60, bbox_inches="tight")
    print(out, [n for n in names])
