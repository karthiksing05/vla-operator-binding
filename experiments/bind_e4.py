"""PLAN_BIND E4 (docs/PLAN_E34.md section 1; pre-registration experiments/prereg/pbind_e34.py, ledger m3_prereg tag pbind_e4):
binding resolution.  Does binding error rise with the referring expression's depth (type < instance < relational sub-region) at
matched scene familiarity?  CPU only: E2b's per-task binding measures (results/bind/e2b.json) + the parse / scene.

Depth coding (mechanical, fixed before the read; CAT is a hand-written category table, disclosed):
  object role  1 (instance) if the named object's type has >= 2 instances among the movable objects (a qualifier picks one) OR
               another movable object is in the same CAT category (an attribute picks one, e.g. ketchup vs tomato sauce); else 0
  goal role    2 (relational sub-region) if the target is a table region defined relative to another body ("right of the
               caddy"), or a region of a body with >= 2 goal regions in the scene's candidates
               (fm_types.Scene.alternatives + own targets; results/bind/e1_scene.json); 1 (instance) if the target is an object
               whose type has >= 2 instances in the scene; else 0
Errors (E2b, 40 runs per task): object role = 1 - P_obj (trials 40); goal role = 1 - P_goal (trials = runs binding the named
object; the goal role is excluded where that is < 5).  Scene familiarity = fm axes scene_familiar (a training cell in the scene).
Model (PRIMARY): binomial GLM logit(err) = b0 + b_depth * depth + b_fam * familiar + b_role * [goal], fitted by IRLS; task-cluster
bootstrap (B 2000, seed 0; both roles of a task resampled together).  E4 FAILS if b_depth <= 0 or its 95% CI's lower bound <= 0.
Descriptive: mean error per depth level (monotone?), per role, within familiar / new scenes.

    python experiments/bind_e4.py [--record] [--dry]      (chunks env, slurm/jobs/cpu.sbatch) -> results/bind/e4.json
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
OUT = os.path.join(ROOT, "results", "bind")
CAT = {"sauce": {"ketchup", "tomato_sauce", "bbq_sauce", "salad_dressing", "new_salad_dressing"},
       "grocery": {"alphabet_soup", "cream_cheese", "butter", "chocolate_pudding", "cookies", "milk", "orange_juice"},
       "mug": {"porcelain_mug", "red_coffee_mug", "white_yellow_mug"},
       "bowl": {"akita_black_bowl", "white_bowl"}, "book": {"black_book", "yellow_book"},
       "cookware": {"moka_pot", "chefmate_8_frypan"}}
MIN_GOAL_TRIALS, B = 5, 2000


def itype(n):
    return re.sub(r"_\d+$", "", n)


def cat(t):
    return next((k for k, v in CAT.items() if t in v), None)


def code(t, scn):
    movable = list(t.objects)
    named = scn["named"]
    nt = itype(named)
    same = [m for m in movable if m != named and itype(m) == nt]
    attr = [m for m in movable if m != named and cat(itype(m)) is not None and cat(itype(m)) == cat(nt)]
    d_obj = 1 if (same or attr) else 0
    tgt = scn["own"][0]
    names = set(scn["alternatives"]) | set(scn["own"])
    bodies = dict(t.objects, **t.fixtures)
    body = max([b for b in bodies if tgt == b or tgt.startswith(b + "_")], key=len, default=None)
    tables = ("kitchen_table", "study_table", "living_room_table", "main_table", "table")
    if body in tables and any(b in tgt for b in bodies if b != body and b not in tables):
        d_goal = 2                                               # a table region defined relative to another body ("right of the caddy")
    elif body is not None and tgt != body:
        regs = [n for n in names if n.startswith(body + "_") and "init_region" not in n]
        d_goal = 2 if len(regs) >= 2 else (1 if body in t.objects and sum(itype(m) == itype(body) for m in movable) >= 2 else 0)
    elif body in t.objects:
        d_goal = 1 if sum(itype(m) == itype(body) for m in movable) >= 2 else 0
    else:
        d_goal = 0
    return d_obj, d_goal, dict(same=same, attr=attr, body=body)


def irls(X, k, n, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p) + 1e-9
        z = eta + (k - n * p) / W
        A = X.T @ (W[:, None] * X) + 1e-6 * np.eye(X.shape[1])
        b_new = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    import m1_compose_create as M1
    tt = M1.task_table()
    scn = json.load(open(os.path.join(OUT, "e1_scene.json"), encoding="utf-8"))
    e2b = json.load(open(os.path.join(OUT, "e2b.json"), encoding="utf-8"))["per_task"]
    rows = []
    for tid, s in scn.items():
        t = tt[tid]
        ax = json.load(open(os.path.join(ROOT, "results", "fm", "types", tid.replace(":", "_") + ".json"), encoding="utf-8"))["axes"]
        d_obj, d_goal, info = code(t, s)
        fam = int(bool(ax["scene_familiar"]))
        q = e2b[tid]
        n = int(q["n"])
        nobj = int(round(q["P_obj"] * n))
        rows.append(dict(tid=tid, role="object", depth=d_obj, fam=fam, k=n - nobj, n=n, info=info))
        if nobj >= MIN_GOAL_TRIALS:
            ncorr = int(round(q["P_hat"] * n))
            rows.append(dict(tid=tid, role="goal", depth=d_goal, fam=fam, k=nobj - ncorr, n=nobj, info=info))
    if a.dry:
        for r in rows:
            print(f"[bind e4 dry] {r['tid']} {r['role']}: depth {r['depth']} familiar {r['fam']} trials {r['n']} ({r['info']})", flush=True)
        return
    X = np.array([[1.0, r["depth"], r["fam"], 1.0 if r["role"] == "goal" else 0.0] for r in rows])
    k = np.array([r["k"] for r in rows], float)
    n = np.array([r["n"] for r in rows], float)
    b = irls(X, k, n)
    tids = sorted({r["tid"] for r in rows})
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(tids), len(tids))
        idx = [i for p in pick for i, r in enumerate(rows) if r["tid"] == tids[p]]
        Xb, kb, nb = X[idx], k[idx], n[idx]
        if len(set(Xb[:, 1])) < 2:
            continue
        bs.append(irls(Xb, kb, nb))
    bs = np.array(bs)
    lo, hi = np.percentile(bs[:, 1], [2.5, 97.5])
    res = dict(n_rows=len(rows), n_tasks=len(tids), coef=dict(zip(("b0", "b_depth", "b_fam", "b_role_goal"), map(float, b))),
               b_depth_ci=[float(lo), float(hi)], b_fam_ci=[float(x) for x in np.percentile(bs[:, 2], [2.5, 97.5])], n_boot=len(bs))
    lev = {}
    for r in rows:
        lev.setdefault(r["depth"], []).append(r["k"] / r["n"])
    res["mean_err_by_depth"] = {int(d): dict(mean=float(np.mean(v)), n=len(v)) for d, v in sorted(lev.items())}
    for fam in (0, 1):
        lv = {}
        for r in rows:
            if r["fam"] == fam:
                lv.setdefault(r["depth"], []).append(r["k"] / r["n"])
        res[f"mean_err_by_depth_fam{fam}"] = {int(d): dict(mean=float(np.mean(v)), n=len(v)) for d, v in sorted(lv.items())}
    res["counts"] = {f"{r_}|d{d}|fam{f}": sum(1 for r in rows if r["role"] == r_ and r["depth"] == d and r["fam"] == f)
                     for r_ in ("object", "goal") for d in (0, 1, 2) for f in (0, 1)}
    res["verdict"] = "PASS (not falsified)" if b[1] > 0 and lo > 0 else "FAIL (falsified)"
    res["rows"] = rows
    json.dump(res, open(os.path.join(OUT, "e4.json"), "w", encoding="utf-8"), indent=1, default=float)
    for r in rows:
        print(f"[bind e4] {r['tid']} {r['role']}: depth {r['depth']} fam {r['fam']} err {r['k']}/{r['n']}", flush=True)
    for key in ("coef", "b_depth_ci", "b_fam_ci", "n_boot", "mean_err_by_depth", "mean_err_by_depth_fam0", "mean_err_by_depth_fam1",
                "counts", "verdict"):
        print(f"[bind e4] {key}: {json.dumps(res[key])}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("bind_e4", bench="libero_90_c", tag="read", seed=0, metrics={k_: v for k_, v in res.items() if k_ != "rows"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/bind_e4.py", command="python experiments/bind_e4.py --record"))


if __name__ == "__main__":
    main()
