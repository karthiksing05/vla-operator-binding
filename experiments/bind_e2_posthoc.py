"""PLAN_BIND E2 POST-HOC (descriptive; written AFTER the pre-registered E2 read, which stands): the release-site label counts the own
target's PARENT BODY as correct (e.g. wooden_tray_1 for wooden_tray_1_contain_region), to check whether the nearest-site rule
mislabels releases into a container region as its body.  Also the release-site label histogram per task.  -> results/bind/e2_posthoc.json"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import bind_e2 as E                                               # noqa: E402

P = E.jload(os.path.join(E.OUT, "e2_plan.json"))
zs = {}
per, hist = {}, {}
for u in P["units"]:
    m = json.loads(str(np.load(os.path.join(E.E2D, u["unit"] + ".npz"), allow_pickle=True)["meta"]))
    ok_sites = set(u["own"]) | {s for s in u["sites"] if any(o.startswith(s + "_") for o in u["own"])}
    d = per.setdefault(u["tid"], [0, 0])
    for b, s in zip(m["bound"], m["rel_site"]):
        d[1] += 1
        d[0] += int(b == u["named"] and s in ok_sites)
        if b == u["named"]:
            hist.setdefault(u["tid"], Counter())[s or "-"] += 1
T = sorted(per)
for t in T:
    eps = E.jload(os.path.join(ROOT, "results", "fm", "types", t.replace(":", "_") + ".json"))["arms"]["zs"]["episodes"]
    zs[t] = float(np.mean([e["success"] for e in eps]))
ph = np.array([per[t][0] / per[t][1] for t in T])
y = np.array([zs[t] for t in T])
rng = np.random.default_rng(0)
rs = []
for _ in range(10_000):
    i = rng.integers(0, len(T), len(T))
    rs.append(E.spearman(ph[i], y[i]))
res = dict(note="POST-HOC descriptive; the pre-registered E2 verdict stands", rho=E.spearman(ph, y),
           lo=float(np.nanpercentile(rs, 2.5)), hi=float(np.nanpercentile(rs, 97.5)), slope=float(np.polyfit(ph, y, 1)[0]),
           per_task={t: dict(P_hat_body=float(per[t][0] / per[t][1]), zs=zs[t], sites=dict(hist.get(t, {}))) for t in T})
E.dump(os.path.join(E.OUT, "e2_posthoc.json"), res)
for t in T:
    print(f"[bind e2 posthoc] {t}: P_hat(body ok) {per[t][0] / per[t][1]:.3f} zs {zs[t]:.2f}; release sites {dict(hist.get(t, {}))}", flush=True)
print(f"[bind e2 posthoc] Spearman {res['rho']:+.3f} [{res['lo']:+.3f}, {res['hi']:+.3f}]; slope {res['slope']:.3f}", flush=True)
