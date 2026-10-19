"""U1 read at matched cut counts (exploratory): segmenter vs random cuts at the same number of cuts per demo, over the penalty sweep.
Pick-and-place demos only (one sustained grasp + release).  Uses experiments/u1_read.py's loader."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import u1_read as U                                                        # noqa: E402
from lifelong_ood.policy import segment as S                               # noqa: E402


def run():
    demos = [d for d in U.load_all() if d["kind"] == "pick-place"]
    rng = np.random.default_rng(0)
    print(f"{len(demos)} pick-place demos")
    for method in ("label", "level"):
        for lam in (U.LAMS + [128, 256, 512]):
            gs, rs, ps, nc, rg, rr = [], [], [], [], [], []
            for d in demos:
                segs = U.segs_of(d, method, lam)
                b = [int(d["frames"][lo]) for lo, _, _ in segs[1:]]
                tr = d["grasp"][:1] + d["release"][:1]
                if len(tr) < 2:
                    continue
                m = S.match(b, tr, U.TOL)
                gs.append(m["err"][0] <= U.TOL)
                rs.append(m["err"][1] <= U.TOL)
                if m["precision"] is not None:
                    ps.append(m["precision"])
                nc.append(len(b))
                T = int(d["frames"][-1])
                for _ in range(20):
                    rb = sorted(rng.integers(1, T, size=len(b)).tolist())
                    mr = S.match(rb, tr, U.TOL)
                    rg.append(mr["err"][0] <= U.TOL)
                    rr.append(mr["err"][1] <= U.TOL)
            print(f"{method:5} lam {lam:5}: cuts/demo {np.mean(nc):5.2f} | grasp {np.mean(gs):.2f} (random {np.mean(rg):.2f}) | "
                  f"release {np.mean(rs):.2f} (random {np.mean(rr):.2f}) | precision {np.mean(ps) if ps else float('nan'):.2f}", flush=True)


if __name__ == "__main__":
    run()
