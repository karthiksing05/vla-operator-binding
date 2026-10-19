"""Plan Z, track U (docs/UNSUPERVISED.md): does the policy's likelihood NAME a contact segment, and does it FLIP across the grasp?
Offline, CPU; reads results/robot/u1/*.npz (experiments/zu_score.py).  Ledger zu_read tag identity.

IDENTITY.  Single-step tasks with a sustained grasp and release.  Per segment (approach = before the grasp, carry = grasp to
release), the training instruction (own excluded for training demos) with the lowest mean log flow loss over the segment: does it
hold the segment's concept (approach: the task's object key; carry: its goal key)?  Chance = the share of candidate instructions
that hold it.
FLIP.  Tasks whose object and goal are each held by some other training instruction: the best object source o and the best goal
source g (lowest mean loss over the demo); d = log C[:, o] - log C[:, g]; does d rise across the grasp (mean after > mean before)?
And does a single best cut of d land within 10 frames of the grasp?

    $PY experiments/zu_identity.py [--record]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy import primitives as P                        # noqa: E402
from lifelong_ood.policy import segment as S                           # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def keys(tx):
    K, ho, hg = {}, {}, {}
    for i, t in tx.items():
        K[i] = []
        for c in P.steps_of(t):
            try:
                o, g = P.object_key(c), P.goal_key(c)
            except Exception:
                continue
            K[i].append((o, g))
            ho.setdefault(o, []).append(i)
            hg.setdefault(g, []).append(i)
    return K, ho, hg


def main(record):
    tx = D.Demos().tasks
    K, ho, hg = keys(tx)
    ident, flip = {}, {}
    for st in ("l40", "l10", "l90"):
        for p in sorted(glob.glob(os.path.join(R, "u1", f"{st}_*.npz"))):
            z = np.load(p, allow_pickle=True)
            t, text = int(z["task"]), str(z["text"])
            if len(P.steps_of(text)) > 1:
                continue
            try:
                o, g = P.object_key(text), P.goal_key(text)
            except Exception:
                continue
            gr, rl = S.gripper_truth(z["actions"])
            if not gr or not rl:
                continue
            L = np.log(((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2)[:, :40])
            fr = z["frames"]
            lab = [l for l in range(40) if not (st != "l90" and l == t)]
            so = [l for l in lab if any(k[0] == o for k in K[l])]
            sg = [l for l in lab if any(k[1] == g for k in K[l])]
            for seg, (lo, hi), good in (("approach", (0, gr[0]), so), ("carry", (gr[0], rl[0]), sg)):
                m = (fr >= lo) & (fr < hi)
                if not good or m.sum() < 2:
                    continue
                mean = L[m][:, lab].mean(0)
                order = [lab[i] for i in np.argsort(mean)]
                ident.setdefault(f"{'familiar' if st != 'l90' else 'l90'}/{seg}" if st != "l10" else f"l10/{seg}", []).append(
                    dict(top1=order[0] in good, chance=len(good) / len(lab), rank=min(order.index(x) for x in good)))
            co, cg = [i for i in ho.get(o, []) if i in lab], [i for i in hg.get(g, []) if i in lab]
            if co and cg:
                a, b = min(co, key=lambda i: L[:, i].mean()), min(cg, key=lambda i: L[:, i].mean())
                if a != b:
                    d = L[:, a] - L[:, b]
                    pre, post = d[fr < gr[0]], d[(fr >= gr[0]) & (fr < rl[0])]
                    if len(pre) and len(post):
                        best = min(range(3, len(d) - 3), key=lambda c: ((d[:c] - d[:c].mean()) ** 2).sum() + ((d[c:] - d[c:].mean()) ** 2).sum())
                        flip.setdefault("familiar" if st != "l90" else "l90", []).append(
                            dict(rise=bool(post.mean() > pre.mean()), cut_near=bool(abs(int(fr[best]) - gr[0]) <= 10),
                                 post_minus_pre=float(post.mean() - pre.mean()), post=float(post.mean())))
    res = dict(identity={k: dict(n=len(v), top1=float(np.mean([x["top1"] for x in v])), chance=float(np.mean([x["chance"] for x in v])),
                                 median_rank=float(np.median([x["rank"] for x in v]))) for k, v in sorted(ident.items())},
               flip={k: dict(n=len(v), rise=float(np.mean([x["rise"] for x in v])), cut_near=float(np.mean([x["cut_near"] for x in v])),
                             rise_min=float(min(x["post_minus_pre"] for x in v)), rise_max=float(max(x["post_minus_pre"] for x in v)),
                             post_min=float(min(x["post"] for x in v)), post_max=float(max(x["post"] for x in v)))
                     for k, v in flip.items()})
    for k, v in res["identity"].items():
        print(f"[zu] identity {k:18} n {v['n']:3}  top-1 holds the concept {v['top1']:.2f}  chance {v['chance']:.2f}  median rank {v['median_rank']}")
    for k, v in res["flip"].items():
        print(f"[zu] flip {k:9} n {v['n']:3}  rises across the grasp {v['rise']:.2f}  single cut within 10 frames {v['cut_near']:.2f}  "
              f"rise {v['rise_min']:+.2f} .. {v['rise_max']:+.2f}; after the grasp {v['post_min']:+.2f} .. {v['post_max']:+.2f}")
    if record:
        from lifelong_ood import ledger
        print(ledger.record("zu_read", bench="libero", tag="identity", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/zu_identity.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main("--record" in sys.argv)
