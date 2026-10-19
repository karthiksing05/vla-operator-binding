"""Plan Z, track I (docs/PLAN_Z.md, docs/DONATION.md): a SOFT blend of held fields fitted to one demonstration, and what each
expert DONATES to it.  Offline, CPU; reads results/robot/i1/*.npz (experiments/i1_velocities.py).

THE BLEND.  Per contact segment s (approach / carry / after the release; program.segments) and per action dim d (x y z, rotation
x y z, gripper), a convex mix of the experts' velocity fields:  v_mix = sum_e beta[e, s, d] v_e,  beta on the simplex.  Fitted by
flow matching to the demo (min mean ||v_mix - u||^2 over the even frames; exact: a quadratic program on the simplex per (s, d),
projected gradient), scored on the odd frames.  Nothing in the network is fitted: a blend is E x 3 x 7 numbers.
Experts: the policy under the training instructions U1's segment likelihood ranks top-2 per segment, under the empty prompt, and
(LIBERO-90 demos only) under the task's new instruction.  For a training demo (l40) its own instruction is the held-out answer: it is
scored as a reference ceiling and never enters the blend.
Comparisons (same held-out frames): each expert alone; the hard switch (per segment, the single best expert); the uniform average
(GPC-like superposition); the blend.

DONATION.  The value of a set S of experts is the held-out loss reduction of the best blend over S + {empty prompt} against the
empty prompt alone (the unconditioned policy).  Exact Shapley values (every subset refitted) per segment and per dim group
(position, rotation, gripper), and the Shapley interaction of the best object-concept and the best goal-concept expert
(> 0 synergy, < 0 redundancy).  Each expert's RELATION to the task: its instruction shares the task's object concept, its goal
concept, both, or neither (lifelong_ood.policy.primitives' keys); donations are aggregated by relation x segment x dim group.

    $PY experiments/i1_blend.py [--record]
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P                       # noqa: E402

R = os.path.join(ROOT, "results", "robot")
GROUPS = {"position": [0, 1, 2], "rotation": [3, 4, 5], "gripper": [6]}
SEGN = ["approach", "carry", "after"]


def simplex_proj(v):
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1
    k = np.nonzero(u - c / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - c[k] / (k + 1), 0)


def qp(G, b, iters=400):
    """min beta' G beta - 2 b' beta on the simplex."""
    n = len(b)
    beta = np.full(n, 1.0 / n)
    L = 2 * np.linalg.eigvalsh(G).max() + 1e-12
    for _ in range(iters):
        beta = simplex_proj(beta - (2 * G @ beta - 2 * b) / L)
    return beta


def stats(V, U):
    """V [N, E, ...] expert velocities, U [N, ...] target -> (G [E, E], b [E], c) for the mean squared error."""
    Vf = V.reshape(V.shape[0], V.shape[1], -1)
    Uf = U.reshape(U.shape[0], -1)
    n = Vf.shape[0] * Vf.shape[2]
    G = np.einsum("nei,nfi->ef", Vf, Vf) / n
    b = np.einsum("nei,ni->e", Vf, Uf) / n
    return G, b, float((Uf * Uf).sum() / n)


def loss(beta, G, b, c):
    return float(beta @ G @ beta - 2 * b @ beta + c)


def relation(text, src):
    def keys(s):
        out = set()
        for cl in P.steps_of(s):
            try:
                out.add(("o", P.object_key(cl)))
                out.add(("g", P.goal_key(cl)))
            except Exception:
                pass
        return out
    k, ks = keys(text), keys(src)
    o, g = bool({x for x in k if x[0] == "o"} & ks), bool({x for x in k if x[0] == "g"} & ks)
    return "both" if o and g else "object" if o else "goal" if g else "neither"


def analyse(p):
    z = np.load(p, allow_pickle=True)
    name = os.path.basename(p)[:-4]
    st = name.split("_")[0]
    V, U, fr = z["v"], z["u"], z["frames"]                          # V [F, E, K, H, 7]
    eids, prompts, text = list(z["expert_ids"]), [str(x) for x in z["experts"]], str(z["text"])
    segs = [tuple(s) for s in z["segments"]]
    blend_idx = [i for i, e in enumerate(eids) if not (st == "l40" and e == -1)]
    empty = eids.index(-2)
    rel = {i: ("empty" if eids[i] == -2 else "own" if eids[i] == -1 else relation(text, prompts[i])) for i in range(len(eids))}
    out = dict(name=name, text=text, experts=prompts, expert_ids=[int(e) for e in eids], relation=rel, segments={})
    for si, (lo, hi) in enumerate(segs[:3]):
        m = (fr >= lo) & (fr < hi)
        idx = np.nonzero(m)[0]
        tr, te = idx[0::2], idx[1::2]
        if len(tr) < 2 or len(te) < 1:
            continue
        sn = SEGN[si] if len(segs) > 1 else "whole"
        seg = dict(frames=int(m.sum()), beta={}, held={}, shapley={}, interaction={})
        for gname, dims in GROUPS.items():
            Gtr = [stats(V[tr][:, :, :, :, d], U[tr][:, :, :, d]) for d in dims]
            Gte = [stats(V[te][:, :, :, :, d], U[te][:, :, :, d]) for d in dims]

            def fit_eval(S):
                S = list(S)
                tot = 0.0
                betas = []
                for (G, b, c), (Ge, be, ce) in zip(Gtr, Gte):
                    beta = qp(G[np.ix_(S, S)], b[S])
                    tot += loss(beta, Ge[np.ix_(S, S)], be[S], ce)
                    betas.append(beta)
                return tot / len(dims), betas
            single = {i: fit_eval([i])[0] for i in range(len(eids))}
            lb, betas = fit_eval(blend_idx)
            hard = min(single[i] for i in blend_idx)
            unif = np.mean([loss(np.full(len(blend_idx), 1 / len(blend_idx)), Ge[np.ix_(blend_idx, blend_idx)], be[blend_idx], ce)
                            for Ge, be, ce in Gte])
            seg["held"][gname] = dict(blend=lb, hard=hard, uniform=float(unif), empty=single[empty],
                                      own=single[eids.index(-1)], single={int(eids[i]): single[i] for i in range(len(eids))})
            seg["beta"][gname] = {int(eids[i]): float(np.mean([bt[j] for bt in betas])) for j, i in enumerate(blend_idx)}
            players = [i for i in blend_idx if i != empty]
            val = {}
            for r in range(len(players) + 1):
                for S in itertools.combinations(players, r):
                    val[S] = single[empty] - fit_eval(sorted(set(S) | {empty}))[0]
            n = len(players)
            from math import factorial
            sh = {}
            for i in players:
                acc = 0.0
                rest = [j for j in players if j != i]
                for r in range(n):
                    for S in itertools.combinations(rest, r):
                        w = factorial(r) * factorial(n - r - 1) / factorial(n)
                        acc += w * (val[tuple(sorted(S + (i,)))] - val[tuple(sorted(S))])
                sh[int(eids[i])] = acc
            seg["shapley"][gname] = sh
            ob = [i for i in players if rel[i] in ("object", "both")]
            go = [i for i in players if rel[i] in ("goal", "both")]
            if ob and go:
                i = max(ob, key=lambda j: sh[int(eids[j])])
                j = max([g for g in go if g != i] or [None], key=lambda g: -1e9 if g is None else sh[int(eids[g])])
                if j is not None:
                    k_ = lambda S: tuple(sorted(S))                                    # noqa: E731
                    seg["interaction"][gname] = dict(object=int(eids[i]), goal=int(eids[j]),
                                                     value=val[k_((i, j))] - val[k_((i,))] - val[k_((j,))] + val[()])
        out["segments"][sn] = seg
    return out


def summarise(rows):
    agg = {}
    for st in ("l40", "l90"):
        R_ = [r for r in rows if r["name"].startswith(st)]
        if not R_:
            continue
        a = dict(demos=len(R_), held={}, donation={}, interaction={})
        for sn in ("approach", "carry", "after", "whole"):
            for g in GROUPS:
                hs = [r["segments"][sn]["held"][g] for r in R_ if sn in r["segments"]]
                if not hs:
                    continue
                key = f"{sn}/{g}"
                a["held"][key] = {m: float(np.mean([h[m] for h in hs])) for m in ("blend", "hard", "uniform", "empty", "own")}
                a["held"][key]["n"] = len(hs)
                a["held"][key]["blend_beats_hard"] = float(np.mean([h["blend"] < h["hard"] for h in hs]))
                d = {}
                for r in R_:
                    if sn not in r["segments"]:
                        continue
                    sh = r["segments"][sn]["shapley"][g]
                    tot = sum(max(v, 0) for v in sh.values()) or 1e-12
                    for e, v in sh.items():
                        rr = r["relation"][r["expert_ids"].index(e)]
                        d.setdefault(rr, []).append(max(v, 0) / tot)
                a["donation"][key] = {k: dict(share=float(np.mean(v)), n=len(v)) for k, v in d.items()}
                it = [r["segments"][sn]["interaction"][g]["value"] for r in R_ if sn in r["segments"] and g in r["segments"][sn]["interaction"]]
                if it:
                    a["interaction"][key] = dict(mean=float(np.mean(it)), synergy=float(np.mean(np.array(it) > 0)), n=len(it))
        agg[st] = a
    return agg


def main(record):
    rows = [analyse(p) for p in sorted(glob.glob(os.path.join(R, "i1", "*.npz")))]
    agg = summarise(rows)
    for st, a in agg.items():
        print(f"== {st}: {a['demos']} demos")
        for key, h in a["held"].items():
            print(f"  {key:18} held-out loss: blend {h['blend']:.4f} | hard {h['hard']:.4f} | uniform {h['uniform']:.4f} | "
                  f"empty {h['empty']:.4f} | own {h['own']:.4f} | blend < hard in {h['blend_beats_hard']:.2f} of {h['n']}")
        for key, d in a["donation"].items():
            print(f"  {key:18} donation share by relation: " + ", ".join(f"{k} {v['share']:.2f} (n {v['n']})" for k, v in sorted(d.items())))
        for key, d in a["interaction"].items():
            print(f"  {key:18} object x goal interaction: mean {d['mean']:+.5f}, synergy in {d['synergy']:.2f} of {d['n']}")
    json.dump(dict(rows=rows, summary=agg), open(os.path.join(R, "i1_blend.json"), "w"), indent=1, default=str)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("i1_blend", bench="libero", tag="read", seed=0, metrics=agg,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i1_blend.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main("--record" in sys.argv)
