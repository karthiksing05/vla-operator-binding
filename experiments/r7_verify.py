"""PROBE R7 stage B gate (docs/PLAN_R7.md section 8; pre-registration experiments/prereg/pr7_verify.py, ledger m3_prereg tag
pr7_verify): is there ANY selectable signal in the frozen policy's own first chunk at the pre-grasp query?

Offline, CPU only, on the saved pre-grasp rollouts (results/r7/sb/pregrasp_*.npz).  At the pre-grasp query every one of a state's 32
runs samples its first chunk from its own noise; afterwards it continues with fresh independent noise (seeds of later queries), so
a run's success is an unbiased sample of P(success | that first chunk is executed and the policy then continues on its own).
Selecting the first chunk with the top score and reading that run's success therefore measures one-step verified selection.

Verifiers:
  V_ceil (PRIMARY gate; an INFORMATION CEILING, not deployable: trained on dev success labels): L2 logistic regression (C 1.0,
          balanced classes) on within-state-centred, standardised first-chunk features (the 10 x 7 chunk; the integrated xyz of
          the 5 executed actions; their summed rotation; the minimum gripper command), leave-one-STATE-out over the 42 wrong-goal
          states.  Leave-one-TASK-out is descriptive.
  V_mode / V_anti (descriptive, deployable heuristics): the chunk nearest to / farthest from the state's mean chunk.
Metric: top-1 gain = success of the picked run - c / 32 (random-seed selection), per state; pooled two-level task-cluster
bootstrap (d4_read.rate); the pick-covering rate on states with c >= 1.
Gate: V_ceil leave-state-out pooled gain >= +0.05 and its CI's lower bound > 0.

    python experiments/r7_verify.py [--record]        (chunks env, via slurm/jobs/cpu.sbatch)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import r7_cover as R7                                             # noqa: E402

C_REG, GAIN_BAR = 1.0, 0.05


def feats(ch):
    """ch [32, H, 7] -> [32, F]."""
    ch = np.asarray(ch, np.float64)
    ex = ch[:, :R7.REPLAN]
    return np.concatenate([ch.reshape(len(ch), -1), ex[:, :, :3].sum(1), ex[:, :, 3:6].sum(1), ch[:, :, 6].min(1, keepdims=True)], 1)


def load(grp="wg"):
    units = [u for u in R7.jload(os.path.join(R7.OUT, "plan_pg.json"))["units"] if u["grp"] == grp]
    out = []
    for u in units:
        z = np.load(os.path.join(R7.OUT, "sb", u["unit"] + ".npz"), allow_pickle=True)
        s = np.asarray(z["success"]).astype(float)[:R7.K_MAX]
        ch = np.asarray(z["chunks"])[:R7.K_MAX, 0]
        X = feats(ch)
        out.append(dict(tid=u["tid"], unit=u["unit"], y=s, X=X - X.mean(0, keepdims=True), raw=ch.reshape(R7.K_MAX, -1).astype(float)))
    return out


def pick_gain(states, scores):
    per, cov = {}, {}
    for st, sc in zip(states, scores):
        k = int(np.argmax(sc))
        g = st["y"][k] - st["y"].mean()
        per.setdefault(st["tid"], []).append(float(g))
        if st["y"].sum() > 0:
            cov.setdefault(st["tid"], []).append(float(st["y"][k]))
    return per, cov


def fit_pred(train, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([s["X"] for s in train])
    y = np.concatenate([s["y"] for s in train])
    sc = StandardScaler().fit(X)
    if y.min() == y.max():
        return [np.zeros(len(s["y"])) for s in test]
    m = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    return [m.decision_function(sc.transform(s["X"])) for s in test]


def main():
    import d4_read
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    S = load("wg")
    res = dict(n_states=len(S), base_rate=float(np.mean([s["y"].mean() for s in S])), arms={})
    sc_loso = [fit_pred(S[:i] + S[i + 1:], [S[i]])[0] for i in range(len(S))]
    tids = sorted({s["tid"] for s in S})
    sc_loto = [None] * len(S)
    for t in tids:
        te = [i for i, s in enumerate(S) if s["tid"] == t]
        pr = fit_pred([s for s in S if s["tid"] != t], [S[i] for i in te])
        for i, p in zip(te, pr):
            sc_loto[i] = p
    dist = [np.linalg.norm(s["raw"] - s["raw"].mean(0), axis=1) for s in S]
    arms = {"V_ceil_loso": sc_loso, "V_ceil_loto": sc_loto, "V_mode": [-d for d in dist], "V_anti": dist}
    lines = [f"wrong-goal pre-grasp states {len(S)}, per-seed success (random selection) {res['base_rate']:.3f}"]
    for name, scs in arms.items():
        per, cov = pick_gain(S, scs)
        g, c = d4_read.rate(per), d4_read.rate(cov)
        res["arms"][name] = dict(gain=g, pick_covering=c, per_task={t: float(np.mean(v)) for t, v in per.items()})
        lines.append(f"{name}: top-1 gain {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]; pick success on covering states "
                     f"{c['mean']:.3f} (n {c['n']}); per task " + ", ".join(f"{t} {np.mean(v):+.3f}" for t, v in per.items()))
    g = res["arms"]["V_ceil_loso"]["gain"]
    res["gate"] = dict(bar=GAIN_BAR, verdict="PASS" if (g["mean"] >= GAIN_BAR and g["lo"] > 0) else "FAIL")
    lines.append(f"GATE V_ceil leave-state-out gain {g['mean']:+.3f} (lo {g['lo']:+.3f}) vs +{GAIN_BAR} and lo > 0 -> {res['gate']['verdict']}")
    print("\n".join("[r7 verify] " + x for x in lines), flush=True)
    R7.dump(os.path.join(R7.OUT, "verify.json"), res)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("r7_read", bench="libero_90_c", tag="verify", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/r7_verify.py", command="python experiments/r7_verify.py --record"))


if __name__ == "__main__":
    main()
