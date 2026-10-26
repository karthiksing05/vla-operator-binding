"""PLAN_SOLVE S0 read (docs/PLAN_S0.md; pre-registration experiments/prereg/ps0.py, ledger m3_prereg tag ps0): the dissociation test
of the binding adapter with ORACLE masks.  CPU (chunks env).  Inputs: results/bind/s0/roll/<set>_<arm>/*.npz.

Per run (E2b's labels: bound = first movable moved > 3 cm while closed; first-release In / On predicates after a 20-step settle):
  success          LIBERO success within the step budget
  binding error    a failure in which the bound object is a movable OTHER than the named one, OR the named object's first release
                   satisfies In / On for a scene candidate other than an own target
  motion failure   every other failure (nothing grasped, never released, released on no candidate, correct release then failed)
Rates are per task (two-level task-cluster bootstrap, d4_read.rate); paired arm differences by task bootstrap (B 2000, seed 0).

BARS (pre-declared; PASS iff all three):
  B1 binding:     binding-error rate (oracle) <= 0.5 x binding-error rate (zs)  [point estimate], and the paired difference's 95% CI
                  upper bound < 0
  B2 dissociation: the motion-failure rate is NOT reduced by a comparable amount: its relative reduction (zs - oracle) / zs is at
                  most half of B1's relative reduction
  B3 damage:      on the training cells, success (oracle) >= success (zs) - 0.05 [point estimate]
Controls (reported): wrong masks -> share of runs binding the wrong-mask object / releasing at the wrong-mask goal, vs zs; zero masks
-> parity with zs (expected identical).

    python experiments/s0_read.py [--record]  -> results/bind/s0.json
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
RD = os.path.join(ROOT, "results", "bind", "s0", "roll")
B = 2000


def classify(m):
    out = []
    for b, lab, rel, ok in zip(m["bound"], m["labels"], m["released"], m["success"]):
        if ok:
            out.append("success")
        elif b and b != m["named"]:
            out.append("binding")
        elif b == m["named"] and rel and lab is not None and not any(lab.get(o, False) for o in m["own"]) and \
                any(v for k, v in lab.items() if k not in m["own"]):
            out.append("binding")
        else:
            out.append("motion")
    return out


def load(set_name, arm):
    per = {}
    for f in sorted(glob.glob(os.path.join(RD, f"{set_name}_{arm}", "*.npz"))):
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        per.setdefault(m["tid"], []).append(m)
    return per


def rates(per, key):
    return {t: [float(c == key) for m in ms for c in classify(m)] for t, ms in per.items()}


def paired(a, b, seed=0):
    """task-bootstrap of mean(a) - mean(b) over the tasks both arms have (runs pooled within task)."""
    T = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    d = np.array([np.mean(a[t]) - np.mean(b[t]) for t in T])
    w = np.array([len(a[t]) for t in T], float)
    s = []
    for _ in range(B):
        i = rng.integers(0, len(T), len(T))
        s.append(float(np.average(d[i], weights=w[i])))
    return dict(diff=float(np.average(d, weights=w)), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), tasks=len(T))


def main():
    import d4_read as DR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    arms = {k: load("dev", k) for k in ("zs", "oracle", "wrong", "zero")}
    res = dict(dev={}, train={})
    for k, per in arms.items():
        if per:
            res["dev"][k] = {c: DR.rate(rates(per, c)) for c in ("success", "binding", "motion")}
    zs, orc = arms["zs"], arms["oracle"]
    rb = dict(zs=rates(zs, "binding"), oracle=rates(orc, "binding"))
    rm = dict(zs=rates(zs, "motion"), oracle=rates(orc, "motion"))
    d_bind, d_mot = paired(rb["oracle"], rb["zs"]), paired(rm["oracle"], rm["zs"])
    d_succ = paired(rates(orc, "success"), rates(zs, "success"))
    zb, ob = res["dev"]["zs"]["binding"]["mean"], res["dev"]["oracle"]["binding"]["mean"]
    zm, om = res["dev"]["zs"]["motion"]["mean"], res["dev"]["oracle"]["motion"]["mean"]
    rel_b = (zb - ob) / zb if zb else 0.0
    rel_m = (zm - om) / zm if zm else 0.0
    tr = {k: load("train", k) for k in ("zs", "oracle")}
    for k, per in tr.items():
        if per:
            res["train"][k] = {c: DR.rate(rates(per, c)) for c in ("success", "binding", "motion")}
    dmg = paired(rates(tr["oracle"], "success"), rates(tr["zs"], "success")) if tr["zs"] and tr["oracle"] else None
    # controls
    follow = {}
    if arms["wrong"]:
        for k in ("wrong", "zs"):
            fo, fg, n = 0, 0, 0
            for t, ms in arms[k].items():
                for m in ms:
                    w = next(x for x in arms["wrong"][t] if x["unit"] == m["unit"])["mask_names"] if k == "zs" else m["mask_names"]
                    for b, lab in zip(m["bound"], m["labels"]):
                        n += 1
                        fo += int(bool(b) and b == w[0])
                        fg += int(lab is not None and w[1] is not None and bool(lab.get(w[1], False)))
            follow[k] = dict(bind_wrong_mask_object=fo / max(n, 1), release_at_wrong_mask_goal=fg / max(n, 1), n=n)
    parity = None
    if arms["zero"]:
        same, tot = 0, 0
        for t, ms in arms["zero"].items():
            for m in ms:
                z = next((x for x in zs.get(t, []) if x["unit"] == m["unit"]), None)
                if z:
                    tot += 1
                    same += int(z["success"] == m["success"] and z["bound"] == m["bound"])
        parity = dict(units_identical=same, units=tot)
    bar = dict(B1_rel_binding_reduction=rel_b, B1_diff=d_bind, B1_pass=bool(rel_b >= 0.5 and d_bind["hi"] < 0),
               B2_rel_motion_reduction=rel_m, B2_pass=bool(rel_m <= 0.5 * rel_b),
               B3_damage=dmg, B3_pass=bool(dmg is not None and dmg["diff"] >= -0.05))
    bar["verdict"] = "PASS" if bar["B1_pass"] and bar["B2_pass"] and bar["B3_pass"] else "FAIL"
    res.update(bar=bar, success_diff=d_succ, motion_diff=d_mot, follow=follow, zero_parity=parity,
               per_task={t: {k: {c: float(np.mean(rates(arms[k], c)[t])) for c in ("success", "binding", "motion")}
                             for k in ("zs", "oracle", "wrong") if t in arms[k]} for t in sorted(zs)})
    json.dump(res, open(os.path.join(ROOT, "results", "bind", "s0.json"), "w"), indent=1)
    for k, v in res["dev"].items():
        print(f"[s0 read] dev {k}: " + " | ".join(f"{c} {v[c]['mean']:.3f} [{v[c]['lo']:.3f}, {v[c]['hi']:.3f}]" for c in v), flush=True)
    for k, v in res["train"].items():
        print(f"[s0 read] train {k}: " + " | ".join(f"{c} {v[c]['mean']:.3f} [{v[c]['lo']:.3f}, {v[c]['hi']:.3f}]" for c in v), flush=True)
    for t, v in res["per_task"].items():
        print(f"[s0 read] {t}: " + " | ".join(f"{k} s {x['success']:.2f} b {x['binding']:.2f} m {x['motion']:.2f}" for k, x in v.items()),
              flush=True)
    print(f"[s0 read] success oracle - zs {d_succ}; motion diff {d_mot}; follow {follow}; zero parity {parity}", flush=True)
    print(f"[s0 read] BAR {json.dumps(bar)}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s0_adapter", bench="libero_90_c", tag="read", seed=0, command="python experiments/s0_read.py --record",
                            metrics=dict(bar=bar, dev=res["dev"], train=res["train"], success_diff=d_succ, follow=follow,
                                         zero_parity=parity),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S0 binding adapter with oracle masks"), script="experiments/s0_read.py"))


if __name__ == "__main__":
    main()
