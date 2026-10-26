"""PLAN_SOLVE S2 smoke read (docs/PLAN_S2.md; pre-registration experiments/prereg/ps2.py, ledger m3_prereg tag ps2).  EXPLORATORY.
CPU (chunks env).  Arms on the same 24 dev tasks x fresh_m1 states 0-4, seeds 0-3 (paired: same states, same seeds, same noise):
  zs        S0's dev_zs runs, first 4 seeds          (results/bind/s0/roll/dev_zs)
  oracle    S0's dev_oracle runs, first 4 seeds      (results/bind/s0/roll/dev_oracle)
  deploy    adapter + S1b binder masks (live, per replan)  (results/bind/s2/roll/dev_binder)
Damage: deploy vs S0's train_zs (first 4 seeds) on the 50 training-cell units.
Outcome classes = S0's classify (success / binding / motion).  Closure c = (deploy - zs) / (oracle - zs), for binding error and
for success; CI by task bootstrap of the ratio of task-pooled means (B 2000, seed 0; reported only when |oracle - zs| CI excludes 0).

    python experiments/s2_read.py [--record]  -> results/bind/s2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s0_read as S0R  # noqa: E402

K = 4
B = 2000
SRC = dict(zs=("s0", "zs"), oracle=("s0", "oracle"), deploy=("s2", "binder"))
LIST_KEYS = ("bound", "released", "rel_step", "labels", "success", "steps")


def load(set_name, arm):
    exp, a = SRC[arm]
    per = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", exp, "roll", f"{set_name}_{a}", "*.npz"))):
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        for k in LIST_KEYS:
            m[k] = m[k][:K]
        per.setdefault(m["tid"], []).append(m)
    return per


def common(arms):
    """restrict every arm to the units all arms have (so the comparison stays paired)."""
    units = None
    for per in arms.values():
        u = {m["unit"] for ms in per.values() for m in ms}
        units = u if units is None else units & u
    return {k: {t: [m for m in ms if m["unit"] in units] for t, ms in per.items() if any(m["unit"] in units for m in ms)}
            for k, per in arms.items()}, len(units or ())


def closure(r, seed=0):
    T = sorted(set(r["zs"]) & set(r["oracle"]) & set(r["deploy"]))
    w = np.array([len(r["zs"][t]) for t in T], float)
    M = {k: np.array([np.mean(r[k][t]) for t in T]) for k in r}

    def c(i):
        z, o, d = (np.average(M[k][i], weights=w[i]) for k in ("zs", "oracle", "deploy"))
        return (d - z) / (o - z) if abs(o - z) > 1e-9 else float("nan")
    rng = np.random.default_rng(seed)
    s = [c(rng.integers(0, len(T), len(T))) for _ in range(B)]
    s = [x for x in s if np.isfinite(x)]
    return dict(c=float(c(np.arange(len(T)))), lo=float(np.percentile(s, 2.5)) if s else None,
                hi=float(np.percentile(s, 97.5)) if s else None, tasks=len(T))


def main():
    import d4_read as DR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    dev, n_dev = common({k: load("dev", k) for k in ("zs", "oracle", "deploy")})
    res = dict(k_seeds=K, dev_units=n_dev, dev={}, train={})
    for k, per in dev.items():
        res["dev"][k] = {c: DR.rate(S0R.rates(per, c)) for c in ("success", "binding", "motion")}
    diffs, clos = {}, {}
    for c in ("success", "binding", "motion"):
        r = {k: S0R.rates(dev[k], c) for k in dev}
        diffs[c] = dict(deploy_minus_zs=S0R.paired(r["deploy"], r["zs"]), oracle_minus_zs=S0R.paired(r["oracle"], r["zs"]),
                        deploy_minus_oracle=S0R.paired(r["deploy"], r["oracle"]))
        clos[c] = closure(r)
    tr, n_tr = common({k: load("train", k) for k in ("zs", "deploy")})
    for k, per in tr.items():
        res["train"][k] = {c: DR.rate(S0R.rates(per, c)) for c in ("success", "binding", "motion")}
    dmg = S0R.paired(S0R.rates(tr["deploy"], "success"), S0R.rates(tr["zs"], "success")) if tr["zs"] and tr["deploy"] else None
    dg = [m["binder_diag"] for ms in dev["deploy"].values() for m in ms if m.get("binder_diag")]
    diag = {key: dict(units=len(dg), empty=float(np.mean([d[key]["n"] == 0 for d in dg])),
                      inside=float(np.mean([d[key]["inside"] for d in dg])), iou=float(np.mean([d[key]["iou"] for d in dg])),
                      majority_inside=float(np.mean([d[key]["inside"] > 0.5 for d in dg]))) for key in ("obj", "goal")} if dg else None
    res.update(train_units=n_tr, diffs=diffs, closure=clos, damage=dmg, binder_diag=diag,
               direction=dict(binding_deploy_below_zs=bool(diffs["binding"]["deploy_minus_zs"]["diff"] < 0),
                              binding_closure_in_0_1=bool(0 <= clos["binding"]["c"] <= 1)),
               per_task={t: {k: {c: float(np.mean(S0R.rates(dev[k], c)[t])) for c in ("success", "binding", "motion")}
                             for k in dev if t in dev[k]} for t in sorted(dev["zs"])})
    json.dump(res, open(os.path.join(ROOT, "results", "bind", "s2.json"), "w"), indent=1)
    print(f"[s2 read] dev units {n_dev}, train units {n_tr}, K {K}", flush=True)
    for s in ("dev", "train"):
        for k, v in res[s].items():
            print(f"[s2 read] {s} {k}: " + " | ".join(f"{c} {v[c]['mean']:.3f} [{v[c]['lo']:.3f}, {v[c]['hi']:.3f}]" for c in v), flush=True)
    for t, v in res["per_task"].items():
        print(f"[s2 read] {t}: " + " | ".join(f"{k} s {x['success']:.2f} b {x['binding']:.2f} m {x['motion']:.2f}" for k, x in v.items()),
              flush=True)
    print(f"[s2 read] DIFFS {json.dumps(diffs)}", flush=True)
    print(f"[s2 read] CLOSURE {json.dumps(clos)}", flush=True)
    print(f"[s2 read] DAMAGE {json.dumps(dmg)}; BINDER DIAG {json.dumps(diag)}; DIRECTION {json.dumps(res['direction'])}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s2_smoke", bench="libero_90_c", tag="read", seed=0, command="python experiments/s2_read.py --record",
                            metrics=dict(dev=res["dev"], train=res["train"], diffs=diffs, closure=clos, damage=dmg, binder_diag=diag,
                                         direction=res["direction"], dev_units=n_dev, train_units=n_tr, k_seeds=K),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S2 smoke: S0 adapter + S1b deployable binder, exploratory"), script="experiments/s2_read.py"))


if __name__ == "__main__":
    main()
