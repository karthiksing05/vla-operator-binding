"""PLAN_SOLVE S1b read (docs/PLAN_S1.md section 6; pre-registration experiments/prereg/ps1b.py, ledger m3_prereg tag ps1b): the
factorised binder (GroundingDINO body x trained relation / instance head) on S1's 240 dev units, scored exactly as S1
(s1_read.hit: peak in the target and covered >= every competitor; s1_read.iou).  CPU (chunks env).

body_seen: the target's body type (the named type; a region's body; a table region's reference body) is the body type of some
TRAINING example's target.  The caddy is the only unseen body type among the depth-2 units; it is reported separately.
BAR (pre-declared): PASS iff (i) S1b's hit on depth-2 held-out units with a SEEN body type >= 0.70 (point estimate) AND (ii) S1b's
depth-0 hit >= B1a's (GroundingDINO alone, S1) on the same units (point estimates).  Two-level task-cluster bootstrap CIs.

    python experiments/s1b_read.py [--record]  -> results/bind/s1b.json
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
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(ROOT, "results", "bind", "s1b")


def main():
    import d4_read as DR
    import s1_read as SR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    pz = np.load(os.path.join(OD, "pred_dev.npz"), allow_pickle=True)
    pred = {str(k): (int(p), m, json.loads(str(mt))) for k, p, m, mt in zip(pz["keys"], pz["peak"], pz["mask"], pz["meta"])}
    s1 = {f"{u['role']}|{u['frame']}": u for u in json.load(open(os.path.join(ROOT, "results", "bind", "s1.json")))["units"]}
    cache = {}

    def frame(f):
        if f not in cache:
            z = np.load(os.path.join(ROOT, f), allow_pickle=True)
            cache[f] = (z, *SR.masks(z), [str(b) for b in z["body_names"]])
        return cache[f]
    seen_types = set()
    for e in lab["train"]:
        _, names, _, bodies = frame(e["frame"])
        for t in e["targets"]:
            seen_types.add(SR.itype(t) if t in bodies else SR.competitors(t, names, bodies)[1])
    units = []
    for e in lab["dev"]:
        z, names, cov, bodies = frame(e["frame"])
        tg = e["targets"][0]
        t = cov[names.index(tg)]
        comp, btype = SR.competitors(tg, names, bodies)
        if tg in bodies:
            btype = SR.itype(tg)
        cc = [cov[names.index(c)] for c in comp]
        key = f"{e['role']}|{e['frame']}"
        pk, msk, meta = pred[key]
        o = s1[key]
        units.append(dict(tid=e["tid"], role=e["role"], phrase=e["phrase"], target=tg, depth=e["depth"], seen=e["seen"],
                          body_type=btype, body_seen=btype in seen_types, s1b_hit=SR.hit(pk, t, cc), s1b_iou=SR.iou(msk, t),
                          s1b_body=meta.get("body"), s1b_relation=meta.get("relation"), detected=pk >= 0,
                          b1b_hit=o["b1b_hit"], b1a_hit=o["b1a_hit"], anchor_hit=o["anchor_hit"]))

    def per(us, k):
        d = {}
        for u in us:
            if u.get(k) is not None:
                d.setdefault(u["tid"], []).append(float(u[k]))
        return DR.rate(d)
    cells = {}
    for depth in (0, 1, 2):
        for split in ("all", "seen", "heldout", "heldout_body_seen", "heldout_body_unseen"):
            us = [u for u in units if u["depth"] == depth and (
                split == "all" or (split == "seen" and u["seen"]) or (split == "heldout" and not u["seen"])
                or (split == "heldout_body_seen" and not u["seen"] and u["body_seen"])
                or (split == "heldout_body_unseen" and not u["seen"] and not u["body_seen"]))]
            cells[f"d{depth}_{split}"] = dict(n=len(us), tasks=sorted({u["tid"] for u in us}), s1b_hit=per(us, "s1b_hit"),
                                              s1b_iou=per(us, "s1b_iou"), b1b_hit=per(us, "b1b_hit"), b1a_hit=per(us, "b1a_hit"),
                                              anchor_hit=per(us, "anchor_hit"), detected=per(us, "detected"))
    by_role = {r: dict(s1b_hit=per([u for u in units if u["role"] == r], "s1b_hit"),
                       b1a_hit=per([u for u in units if u["role"] == r], "b1a_hit"),
                       b1b_hit=per([u for u in units if u["role"] == r], "b1b_hit")) for r in ("object", "goal")}
    pt = {}
    for u in units:
        p = pt.setdefault(f"{u['tid']}|{u['role']}", dict(phrase=u["phrase"], body=u["s1b_body"], relation=u["s1b_relation"],
                                                         target=u["target"], depth=u["depth"], seen=u["seen"],
                                                         body_seen=u["body_seen"], s1b=[], b1a=[], b1b=[]))
        for k in ("s1b", "b1a", "b1b"):
            p[k].append(u[f"{k}_hit"])
    pt = {k: dict(v, **{m: float(np.mean(v[m])) for m in ("s1b", "b1a", "b1b")}) for k, v in pt.items()}
    c_i = cells["d2_heldout_body_seen"]["s1b_hit"]["mean"]
    s0, a0 = cells["d0_all"]["s1b_hit"]["mean"], cells["d0_all"]["b1a_hit"]["mean"]
    bar = dict(i_depth2_heldout_body_seen=c_i, i_n=cells["d2_heldout_body_seen"]["n"], i_pass=bool(c_i is not None and c_i >= 0.70),
               ii_depth0_s1b=s0, ii_depth0_b1a=a0, ii_pass=bool(s0 is not None and s0 >= a0))
    bar["verdict"] = "PASS" if bar["i_pass"] and bar["ii_pass"] else "FAIL"
    json.dump(dict(bar=bar, cells=cells, by_role=by_role, per_task=pt, units=units, seen_body_types=sorted(seen_types)),
              open(os.path.join(ROOT, "results", "bind", "s1b.json"), "w"), indent=1)
    for k, v in cells.items():
        if v["n"]:
            print(f"[s1b read] {k}: n {v['n']} ({len(v['tasks'])} tasks) | S1b {v['s1b_hit']} | B1a {v['b1a_hit']['mean']} | "
                  f"B1b {v['b1b_hit']['mean']} | anchor {v['anchor_hit']['mean']} | S1b iou {v['s1b_iou']['mean']} | "
                  f"detected {v['detected']['mean']}", flush=True)
    for k, v in sorted(pt.items()):
        print(f"[s1b read] {k}: {v['phrase']!r} = body {v['body']!r} + rel {v['relation']!r} -> {v['target']} d{v['depth']} "
              f"seen {v['seen']} body_seen {v['body_seen']}: S1b {v['s1b']:.2f} B1a {v['b1a']:.2f} B1b {v['b1b']:.2f}", flush=True)
    print(f"[s1b read] BAR {json.dumps(bar)}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s1b_bind", bench="libero_90_c", tag="read", seed=0, command="python experiments/s1b_read.py --record",
                            metrics=dict(bar=bar, cells=cells, by_role=by_role),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S1b factorised binder accuracy"), script="experiments/s1b_read.py"))


if __name__ == "__main__":
    main()
