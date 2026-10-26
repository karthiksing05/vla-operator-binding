"""PLAN_SOLVE S2 smoke: per-unit role phrases for the deployable binder (the parse's split, s1_labels.goal_steps) -> results/bind/s2/plan.json.
Dev units = E2b's (results/bind/e2_plan.json); damage units = s0_sandbox.train_units().  Chunks env, CPU."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import s1_labels as L
import s0_sandbox as S0
tasks = json.load(open(os.path.join(ROOT, "results/bind/s1/tasks.json")))
units = {}
for set_name, us in (("dev", json.load(open(os.path.join(ROOT, "results/bind/e2_plan.json")))["units"]), ("train", S0.train_units())):
    for u in us:
        gs = L.goal_steps(tasks[u["tid"]])
        (o, g), a = gs[0]
        assert a[1] == u["named"] and a[2] == u["own"][0], (u["unit"], a, u["named"], u["own"])
        units[u["unit"]] = dict(set=set_name, tid=u["tid"], obj_phrase=L.norm(o), goal_phrase=L.norm(g))
os.makedirs(os.path.join(ROOT, "results/bind/s2"), exist_ok=True)
json.dump(units, open(os.path.join(ROOT, "results/bind/s2/plan.json"), "w"), indent=1)
for t in sorted({(v["tid"], v["obj_phrase"], v["goal_phrase"]) for v in units.values()}):
    print("[s2 plan]", t, flush=True)
