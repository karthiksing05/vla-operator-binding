"""PLAN_BIND E4 POST-HOC (descriptive; written AFTER the pre-registered E4 read, which stands):
(a) :76 recoded as goal depth 2 (the relational rule missed it: its target names `desk_caddy`, not `desk_caddy_1`);
(b) goal error = WRONG region only: among first releases that land in some candidate region / object (E2b's first-release labels),
    the share not in the own target ('none', 'no_release' and pre-release successes excluded from the denominator).
Same GLM and bootstrap as bind_e4.  -> results/bind/e4_posthoc.json"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import bind_e4 as E4                                              # noqa: E402

res = json.load(open(os.path.join(E4.OUT, "e4.json"), encoding="utf-8"))
e2b = json.load(open(os.path.join(E4.OUT, "e2b.json"), encoding="utf-8"))["per_task"]
out = {}


def fit(rows):
    X = np.array([[1.0, r["depth"], r["fam"], 1.0 if r["role"] == "goal" else 0.0] for r in rows])
    k = np.array([r["k"] for r in rows], float)
    n = np.array([r["n"] for r in rows], float)
    b = E4.irls(X, k, n)
    tids = sorted({r["tid"] for r in rows})
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(E4.B):
        pick = rng.integers(0, len(tids), len(tids))
        idx = [i for p in pick for i, r in enumerate(rows) if r["tid"] == tids[p]]
        if len(set(X[idx, 1])) < 2:
            continue
        bs.append(E4.irls(X[idx], k[idx], n[idx]))
    bs = np.array(bs)
    lev = {}
    for r in rows:
        lev.setdefault(r["depth"], []).append(r["k"] / r["n"])
    return dict(b_depth=float(b[1]), ci=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])], n_rows=len(rows),
                mean_err_by_depth={int(d): dict(mean=float(np.mean(v)), n=len(v)) for d, v in sorted(lev.items())})


rows_a = [dict(r, depth=(2 if r["tid"] == "libero_90:76" and r["role"] == "goal" else r["depth"])) for r in res["rows"]]
out["a_recode_76"] = fit(rows_a)
rows_b = []
for r in rows_a:
    if r["role"] == "object":
        rows_b.append(r)
        continue
    fr = e2b[r["tid"]]["first_release"]
    lab = {k: v for k, v in fr.items() if k not in ("none", "no_release", "success_before_release")}
    n = sum(lab.values())
    if n >= 5:
        rows_b.append(dict(r, k=n - lab.get("own", 0), n=n))
out["b_wrong_region_only"] = fit(rows_b)
out["b_goal_rows"] = [(r["tid"], r["depth"], r["k"], r["n"]) for r in rows_b if r["role"] == "goal"]
json.dump(out, open(os.path.join(E4.OUT, "e4_posthoc.json"), "w", encoding="utf-8"), indent=1)
for k, v in out.items():
    print(f"[bind e4 posthoc] {k}: {json.dumps(v)}", flush=True)
