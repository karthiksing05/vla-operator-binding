"""PLAN_SOLVE S1 read (docs/PLAN_S1.md; pre-registration experiments/prereg/ps1.py, ledger m3_prereg tag ps1): binder accuracy on the
dev tasks' first frames (fresh_m1 states 0-4), per role, by referring-expression depth (E4's code) and by whether the role's
phrase x scene occurred among the training labels.  CPU (chunks env).

Methods: B1b (ours, results/bind/s1/b1b/pred_dev.npz), B1a (GroundingDINO, results/bind/s1/b1a/pred_dev.npz), and the SigLIP
prototype anchor (D3 / D5: the argmax agentview token of the target type's training prototype, results/rb4/protos.npz, on the same
image tokens as B1b).
hit   the method's peak token lies in the target's mask (coverage > 0) AND is covered by the target at least as much as by any
      COMPETITOR: the other regions of the same body (cabinet drawers / top side, shelf levels, caddy compartments), the other
      table regions relative to the same body, or the other instances of the same type.  (Region boxes of one body overlap in
      the image; without this a peak on the cabinet would count for every drawer.)
iou   predicted mask (B1b sigmoid >= 0.5; B1a box coverage >= 0.5) vs the target (coverage >= 0.5, or > 0 if that is empty).
The anchor's type: the named object's type (object role); the target body's type (goal role; a table region -> its reference body).
Units without a prototype are reported separately and excluded from the paired comparison.

BAR (pre-declared): PASS iff (i) B1b's hit on depth-2, held-out-combination units >= 0.70 (point estimate) AND (ii) at every depth
(0, 1, 2) B1b's hit >= the anchor's on the units where the anchor has a prototype (point estimates).  Two-level task-cluster
bootstrap CIs (d4_read.rate) are reported throughout.

    python experiments/s1_read.py [--record]  -> results/bind/s1.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
TABLES = ("kitchen_table", "study_table", "living_room_table", "main_table", "table")
DIRS = ("left", "right", "front", "back")


def itype(n):
    return re.sub(r"_\d+$", "", n)


def masks(z):
    names = [str(x) for x in z["body_names"]] + [str(x) for x in z["site_names"]]
    cov = np.concatenate([z["body_cov"], z["site_cov"]]).astype(np.float32)
    return names, cov


def competitors(target, names, bodies):
    if target in bodies:
        return [b for b in bodies if b != target and itype(b) == itype(target)], itype(target)
    tab = next((t for t in TABLES if target.startswith(t + "_")), None)
    if tab:
        core = target[len(tab) + 1:-len("_region")]
        parts = core.split("_")
        ref = "_".join(parts[:-1]) if parts[-1] in DIRS else core
        pre = f"{tab}_{ref}_"
        comp = [n for n in names if n != target and n.startswith(pre) and "init_region" not in n]
        reft = next((itype(b) for b in bodies if itype(b) == ref or itype(b).endswith("_" + ref)), ref)
        return comp, reft
    body = max([b for b in bodies if target.startswith(b + "_")], key=len, default=None)
    comp = [n for n in names if n != target and body and n.startswith(body + "_") and "init_region" not in n]
    return comp, (itype(body) if body else None)


def hit(pk, t_cov, comp_covs):
    if pk is None or pk < 0 or t_cov[pk] <= 0:
        return 0.0
    return float(all(t_cov[pk] >= c[pk] for c in comp_covs))


def iou(pred_bin, t_cov):
    tb = t_cov >= 0.5
    if not tb.any():
        tb = t_cov > 0
    u = (pred_bin | tb).sum()
    return float((pred_bin & tb).sum() / u) if u else 0.0


def main():
    import d4_read as DR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    b1b = np.load(os.path.join(S1, "b1b", "pred_dev.npz"), allow_pickle=True)
    b1b = dict(zip([str(k) for k in b1b["keys"]], b1b["logits"]))
    b1a_z = np.load(os.path.join(S1, "b1a", "pred_dev.npz"), allow_pickle=True)
    b1a = {str(k): (c, int(p)) for k, c, p in zip(b1a_z["keys"], b1a_z["cov"], b1a_z["peak"])}
    pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
    cats = [str(c) for c in pz["cats"]]
    PR = pz["sig"].astype(np.float32)
    feats = np.load(os.path.join(S1, "feats", "dev_img.npy"), mmap_mode="r")
    fidx = {f: i for i, f in enumerate(json.load(open(os.path.join(S1, "feats", "dev_frames.json"))))}
    tasks = json.load(open(os.path.join(S1, "tasks.json"), encoding="utf-8"))
    units = []
    for e in lab["dev"]:
        z = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)
        names, cov = masks(z)
        bodies = [str(b) for b in z["body_names"]]
        tg = e["targets"][0]
        if tg not in names:
            units.append(dict(e, missing=True))
            continue
        t = cov[names.index(tg)]
        comp, atype = competitors(tg, names, bodies)
        if e["role"] == "object":
            atype = itype(tg)
        cc = [cov[names.index(c)] for c in comp]
        key = f"{e['role']}|{e['frame']}"
        lo = b1b[key]
        r = dict(e, missing=False, competitors=comp, anchor_type=atype, target_tokens=int((t > 0).sum()))
        r["b1b_hit"] = hit(int(np.argmax(lo)), t, cc)
        r["b1b_iou"] = iou(1 / (1 + np.exp(-lo)) >= 0.5, t)
        c1a, p1a = b1a[key]
        r["b1a_hit"] = hit(p1a, t, cc)
        r["b1a_iou"] = iou(c1a >= 0.5, t)
        if atype in cats:
            f = np.asarray(feats[fidx[e["frame"]]], np.float32)
            f /= np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9
            r["anchor_hit"] = hit(int(np.argmax(f @ PR[cats.index(atype)])), t, cc)
        else:
            r["anchor_hit"] = None
        r["loose_b1b_hit"] = float(t[int(np.argmax(lo))] > 0)          # without the competitor rule (descriptive)
        units.append(r)
    ok = [u for u in units if not u["missing"]]

    def per(us, k):
        d = {}
        for u in us:
            if u.get(k) is not None:
                d.setdefault(u["tid"], []).append(float(u[k]))
        return DR.rate(d)
    res = dict(n_units=len(units), missing=[(u["tid"], u["role"], u["targets"]) for u in units if u["missing"]], cells={})
    for depth in (0, 1, 2):
        for seen in (None, False, True):
            us = [u for u in ok if u["depth"] == depth and (seen is None or u["seen"] == seen)]
            pu = [u for u in us if u["anchor_hit"] is not None]
            key = f"d{depth}_{'all' if seen is None else ('seen' if seen else 'heldout')}"
            res["cells"][key] = dict(n=len(us), tasks=sorted({u["tid"] for u in us}),
                                     b1b_hit=per(us, "b1b_hit"), b1a_hit=per(us, "b1a_hit"), b1b_iou=per(us, "b1b_iou"),
                                     b1a_iou=per(us, "b1a_iou"), loose_b1b_hit=per(us, "loose_b1b_hit"),
                                     proto_n=len(pu), anchor_hit_proto=per(pu, "anchor_hit"), b1b_hit_proto=per(pu, "b1b_hit"),
                                     b1a_hit_proto=per(pu, "b1a_hit"))
    by_role = {}
    for role in ("object", "goal"):
        us = [u for u in ok if u["role"] == role]
        by_role[role] = dict(b1b_hit=per(us, "b1b_hit"), b1a_hit=per(us, "b1a_hit"),
                             anchor_hit_proto=per([u for u in us if u["anchor_hit"] is not None], "anchor_hit"))
    res["by_role"] = by_role
    per_task = {}
    for u in ok:
        p = per_task.setdefault(f"{u['tid']}|{u['role']}", dict(phrase=u["phrase"], target=u["targets"][0], depth=u["depth"],
                                                                 seen=u["seen"], b1b=[], b1a=[], anchor=[]))
        p["b1b"].append(u["b1b_hit"])
        p["b1a"].append(u["b1a_hit"])
        if u["anchor_hit"] is not None:
            p["anchor"].append(u["anchor_hit"])
    res["per_task"] = {k: dict(v, b1b=float(np.mean(v["b1b"])), b1a=float(np.mean(v["b1a"])),
                               anchor=(float(np.mean(v["anchor"])) if v["anchor"] else None)) for k, v in per_task.items()}
    c2 = res["cells"]["d2_heldout"]["b1b_hit"]["mean"]
    bar_i = c2 is not None and c2 >= 0.70
    bar_ii = {}
    for d in (0, 1, 2):
        c = res["cells"][f"d{d}_all"]
        bb, aa = c["b1b_hit_proto"]["mean"], c["anchor_hit_proto"]["mean"]
        bar_ii[d] = None if bb is None or aa is None else bool(bb >= aa)
    res["bar"] = dict(i_depth2_heldout_b1b_hit=c2, i_pass=bool(bar_i), ii_b1b_ge_anchor=bar_ii,
                      ii_pass=all(v is not False for v in bar_ii.values()) and any(v is not None for v in bar_ii.values()),
                      verdict="PASS" if bar_i and all(v is not False for v in bar_ii.values()) else "FAIL")
    json.dump(dict(res, units=units), open(os.path.join(ROOT, "results", "bind", "s1.json"), "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for k, v in res["cells"].items():
        print(f"[s1 read] {k}: n {v['n']} | B1b {v['b1b_hit']} | B1a {v['b1a_hit']} | anchor(proto n {v['proto_n']}) "
              f"{v['anchor_hit_proto']} vs B1b {v['b1b_hit_proto']} | B1b iou {v['b1b_iou']['mean']} B1a iou {v['b1a_iou']['mean']}",
              flush=True)
    print(f"[s1 read] by role: {json.dumps(by_role)}", flush=True)
    for k, v in sorted(res["per_task"].items()):
        print(f"[s1 read] {k}: {v['phrase']!r} -> {v['target']} depth {v['depth']} seen {v['seen']}: B1b {v['b1b']:.2f} "
              f"B1a {v['b1a']:.2f} anchor {v['anchor']}", flush=True)
    print(f"[s1 read] BAR: {json.dumps(res['bar'])}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s1_bind", bench="libero_90_c", tag="read", seed=0, command="python experiments/s1_read.py --record",
                            metrics=dict(bar=res["bar"], cells=res["cells"], by_role=by_role),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S1 binder accuracy"), script="experiments/s1_read.py"))


if __name__ == "__main__":
    main()
