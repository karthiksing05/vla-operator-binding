"""PLAN_SOLVE S0: inventory of the training cells' demos with matched simulator states (no demo is executed).  CPU, chunks env.
Prints per training cell: its protos-index demo episodes, the raw match file (states per frame), the demo's length and keys, and
the same-scene training cells with a different target (the counterfactual partners)."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import l90c
from lifelong_ood.policy import data as D
tasks = json.load(open(os.path.join(ROOT, "results/bind/s1/tasks.json")))
train = l90c.load_split()["train"]
pp = json.load(open(os.path.join(ROOT, "results/protos/index_l90c.json")))
print("pp keys", list(pp.keys()), "obs keys", list(pp["obs"][0].keys()), flush=True)
eps = {}
for o in pp["obs"]:
    if o["set"] == "train":
        eps.setdefault(f"{o['suite']}:{o['index']}", set()).add((o.get("src"), o.get("hf_task"), o.get("src_hf"), o.get("l90_task"), o["episode"]))
demos, d90 = D.Demos(), D.Demos90()
for tid in train:
    x = tasks[tid]
    e = sorted(eps.get(tid, []), key=str)
    info = []
    for src, hf, shf, l90t, ep in e[:3]:
        mp = os.path.join(ROOT, "results/robot/p0_demo", f"{l90t}_match.npz") if src == "ipec" else os.path.join(ROOT, "results/protos/match", f"tr{(shf if shf is not None else hf):02d}.npz")
        ok = os.path.exists(mp)
        ks = ""
        if ok:
            z = np.load(mp, allow_pickle=True)
            ks = {k: getattr(z[k], "shape", None) for k in z.files}
        info.append((src, hf, ep, os.path.basename(mp), ok, ks))
    partners = [t for t, y in tasks.items() if y["set"] == "train" and t != tid and y["scene"] == x["scene"] and y["goal"] != x["goal"]]
    print(f"[s0 inv] {tid} {x['scene']} n_eps {len(e)} | {info} | partners {partners}", flush=True)
ep = demos.load(0)
print("demo keys", {k: getattr(v, "shape", type(v)) for k, v in ep.items()}, flush=True)
