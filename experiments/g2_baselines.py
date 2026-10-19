"""Plan G, read G2 (docs/history/PLAN_GENERALIZE.md; ledger m3_prereg tag g2_baselines): composition baselines from the literature on G0's
held-out tasks.  For each eligible task of G0, an OBJECT donor (a task outside its fold with the same object concept, preferring a
different goal) and a GOAL donor (outside the fold, same goal, preferring a different object), both with a per-task LoRA on their own
scene's demonstration.  Arms: lorasum (task arithmetic: both LoRAs on at full weight) and loraavg (both at weight 1/2), always on --
composition of adapters without our phase gating or consolidation.  Same states, noise seed and fresh start states as G0, so every
arm pairs with G0's zero-shot, composed-concepts and monolithic runs.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/g2_baselines.py --plan
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g2_baselines.py --accept --test
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g2_baselines.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P             # noqa: E402

spec_ = importlib.util.spec_from_file_location("g0", os.path.join(ROOT, "experiments", "g0_recombine.py"))
G0 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(G0)
R, PY, SEED = G0.R, G0.PY, G0.SEED
PLAN = os.path.join(R, "g2_plan.json")


def plan():
    from lifelong_ood import ledger
    texts = G0.E1.texts90()
    g0 = json.load(open(G0.PLAN))
    fold = {int(t): f for t, f in g0["folds"].items()}
    wrong = set(ledger.load("l90_scenes", tag="map")[-1]["metrics"]["wrong_scene_tasks"])
    lora = {t for t in fold if os.path.exists(os.path.join(R, f"r3_lora_{t}_sc.pt" if t in wrong else f"r3_lora_{t}.pt"))}
    keys = {}
    for t in fold:
        c = P.steps_of(texts[t])[0]
        ik = P.instance_key(c)
        keys[t] = dict(O="O:" + P.object_key(c), I="I:" + ik if ik else None, G="G:" + P.goal_key(c))
    out = {}
    for t, v in g0["tasks"].items():
        if not v["eligible"]:
            continue
        t_, f = int(t), v["fold"]
        cand = [u for u in sorted(lora) if fold[u] != f and u != t_]
        oc, gc = v["object_concept"], v["goal_concept"]
        obj = [u for u in cand if oc in (keys[u]["O"], keys[u]["I"])]
        goal = [u for u in cand if keys[u]["G"] == gc]
        od = next((u for u in obj if keys[u]["G"] != gc), obj[0] if obj else None)
        gd = next((u for u in goal if keys[u]["O"] != keys[t_]["O"]), goal[0] if goal else None)
        out[t] = dict(object_donor=od, goal_donor=gd, fold=f, ok=bool(od is not None and gd is not None and od != gd))
    json.dump(out, open(PLAN, "w"), indent=1)
    print(f"[g2] {sum(v['ok'] for v in out.values())} of {len(out)} eligible tasks have an object donor and a goal donor", flush=True)


def arms(v):
    return {"lorasum": f"lorasum@{v['object_donor']}+{v['goal_donor']}", "loraavg": f"loraavg@{v['object_donor']}+{v['goal_donor']}"}


def launch(exp, trials, fresh):
    for t, v in json.load(open(PLAN)).items():
        if not v["ok"]:
            continue
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def read(record):
    C = G0.E1.compare
    g0 = json.load(open(G0.PLAN))["tasks"]
    test, acc, missing = {}, {}, []
    for t, v in json.load(open(PLAN)).items():
        if not v["ok"]:
            continue
        a = dict(G0.arms(g0[t]), **arms(v))
        x = {k: G0.load("g2_main" if k in ("lorasum", "loraavg") else "g0_main", t, z) for k, z in a.items()}
        s = {k: G0.load("g2_accept" if k in ("lorasum", "loraavg") else "g0_accept", t, z) for k, z in a.items()}
        if None in x.values() or None in s.values():
            missing.append(t)
            continue
        test[t], acc[t] = x, {k: int(sum(z.values())) for k, z in s.items()}
    keep = lambda t, k: k if acc[t][k] > acc[t]["zs"] else "zs"                                   # noqa: E731
    res = {"gfac_vs_sum": C({t: {"x": x["gfac"], "y": x["lorasum"]} for t, x in test.items()}),
           "gfac_vs_avg": C({t: {"x": x["gfac"], "y": x["loraavg"]} for t, x in test.items()}),
           "sum_vs_zs": C({t: {"x": x["lorasum"], "y": x["zs"]} for t, x in test.items()}),
           "avg_vs_zs": C({t: {"x": x["loraavg"], "y": x["zs"]} for t, x in test.items()}),
           "kept_gfac_vs_kept_best_baseline": C({t: {"x": x[keep(t, "gfac")],
                                                    "y": x[max(("lorasum", "loraavg"), key=lambda k: acc[t][k]) if max(acc[t]["lorasum"], acc[t]["loraavg"]) > acc[t]["zs"] else "zs"]}
                                                for t, x in test.items()})}
    for k, v in res.items():
        print(f"[g2] {k}: {v['x']} vs {v['y']} of {v['n']} ({v['tasks']} tasks), diff {v['diff']['diff']:+.3f} "
              f"[{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}], sign p {v['sign_p']:.3g}", flush=True)
    out = dict(comparisons=res, missing=missing, per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()})
    json.dump(out, open(os.path.join(R, "g2_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g2_baselines", bench="libero_90", tag="read", seed=SEED, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g2_baselines.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--accept" in sys.argv:
        launch("g2_accept", 10, False)
    if "--test" in sys.argv:
        launch("g2_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
