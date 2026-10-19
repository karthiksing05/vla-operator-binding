"""U1 (exploratory): the best K-cut piecewise-constant partition (exact DP, squared error) of a multivariate per-frame signal, scored
against the demo's grasp / release at K = 2 (pick-and-place) vs random 2 cuts.  Signals:
    sig   the centred log flow-loss signature over the 40 training instructions (the policy's description of each frame)
    lat   the empty-prompt prefix latent (image tokens) -- what the policy sees
    act   the demo's own actions, dims 0-5 (no gripper): a non-policy baseline
    sig6  as sig, from the arm dims 0-5 only (no gripper anywhere)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u1_read as U                                                        # noqa: E402
from lifelong_ood.policy import segment as S                               # noqa: E402
import glob                                                                # noqa: E402


def kcut(X, K, min_len=3):
    X = np.asarray(X, np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    F = len(X)
    c1 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    c2 = np.concatenate([[0], np.cumsum((X * X).sum(1))])

    def sse(a, b):
        s = c1[b] - c1[a]
        return c2[b] - c2[a] - (s * s).sum() / (b - a)
    D = np.full((K + 2, F + 1), np.inf)
    arg = np.zeros((K + 2, F + 1), int)
    D[0, 0] = 0
    for k in range(1, K + 2):
        for b in range(k * min_len, F + 1):
            for a in range((k - 1) * min_len, b - min_len + 1):
                if np.isfinite(D[k - 1, a]):
                    c = D[k - 1, a] + sse(a, b)
                    if c < D[k, b]:
                        D[k, b], arg[k, b] = c, a
    cuts, b = [], F
    for k in range(K + 1, 0, -1):
        b = arg[k, b]
        cuts.append(b)
    return sorted(cuts)[1:]


def main():
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(U.R, "u1", "*.npz"))}
    rng = np.random.default_rng(0)
    res = {}
    for d in U.load_all():
        if d["kind"] != "pick-place" or not d["grasp"] or not d["release"]:
            continue
        z = np.load(files[d["name"]], allow_pickle=True)
        full = ((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2)[:, :40]
        arm = z["err"][..., :6].mean((2, 3))[:, :40]
        fr = z["frames"]
        sigs = dict(sig=np.log(full) - np.log(full).mean(1, keepdims=True),
                    sig6=np.log(arm) - np.log(arm).mean(1, keepdims=True),
                    lat=z["lat"][:, 0], act=z["actions"][fr, :6])
        tr = [d["grasp"][0], d["release"][0]]
        for n, X in sigs.items():
            b = [int(fr[i]) for i in kcut(X, 2)]
            m = S.match(b, tr, U.TOL)
            res.setdefault(n, []).append((m["err"][0], m["err"][1]))
        T = int(fr[-1])
        for _ in range(50):
            m = S.match(sorted(rng.integers(1, T, 2).tolist()), tr, U.TOL)
            res.setdefault("random", []).append((m["err"][0], m["err"][1]))
    for n, v in res.items():
        v = np.array(v)
        print(f"{n:7} n={len(v):4}  grasp within 10: {np.mean(v[:, 0] <= 10):.2f} (median err {np.median(v[:, 0]):5.1f})   "
              f"release within 10: {np.mean(v[:, 1] <= 10):.2f} (median err {np.median(v[:, 1]):5.1f})")


if __name__ == "__main__":
    main()
