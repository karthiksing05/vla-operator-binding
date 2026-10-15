"""Plan G, reads G0 + G1 (docs/PLAN_GENERALIZE.md; ledger m3_prereg tag g0_recombine): does COMPOSITION generalise?

The 67 single-step pick-and-place LIBERO-90 tasks (the stream's 30 and E5's 37) are split into 5 folds.  For each fold, every
concept its tasks need is refit WITHOUT any demonstration of the fold's tasks; a held-out task is eligible when its object (its
instance child, else its type) and its goal both keep demonstrations outside the fold.  So each eligible task is solved, if at
all, by recombining an object and a goal learned only from OTHER tasks (G0).  G1 asks whether the FACTORISATION matters: per fold,
a monolithic multitask LoRA (rank 16, every frame prompted with its own instruction) is trained on the same pool of demonstrations
and run on the same held-out tasks.  The per-task LoRA (which sees the held-out task's own demonstration) is the upper reference.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/g0_recombine.py --plan --fit
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g0_recombine.py --accept --test
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g0_recombine.py --read [--record]
"""
import collections
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as P             # noqa: E402

spec_ = importlib.util.spec_from_file_location("e1", os.path.join(ROOT, "experiments", "e1_read.py"))
E1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(E1)

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "g0_plan.json")
SEED, RANK, FOLDS, MONO_STEPS = 15, 4, 5, 4000
ORDER = ("zs", "gfac")


def plan():
    from lifelong_ood.policy import data as D
    texts = E1.texts90()
    d90 = D.Demos90()
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    e5 = json.load(open(os.path.join(R, "e5_plan.json")))["tasks"]
    tasks = sorted(int(t) for t in list(e3) + list(e5))
    has = {t for t in texts if d90.episodes_of_task(t)}
    mem = collections.defaultdict(set)
    for t, x in texts.items():
        if t not in has:
            continue
        for c in P.steps_of(x):
            ik = P.instance_key(c)
            for k in ["O:" + P.object_key(c), "G:" + P.goal_key(c)] + (["I:" + ik] if ik else []):
                mem[k].add(t)
    fold = {int(t): i % FOLDS for i, t in enumerate(np.random.default_rng(0).permutation(tasks))}
    concepts, out = {}, {}
    for t in tasks:
        f = fold[t]
        c = P.steps_of(texts[t])[0]
        ik = P.instance_key(c)
        outside = lambda k: sorted(u for u in mem[k] if fold.get(u) != f)               # noqa: E731
        obj_k = "I:" + ik if ik and outside("I:" + ik) else "O:" + P.object_key(c)
        goal_k = "G:" + P.goal_key(c)
        row = dict(fold=f, object_concept=obj_k, goal_concept=goal_k, eligible=bool(outside(obj_k) and outside(goal_k)))
        if row["eligible"]:
            for k in (obj_k, goal_k):
                ck = (f, k)
                if ck not in concepts:
                    concepts[ck] = dict(name=f"G{f}{k[0]}{sum(1 for x in concepts if x[0] == f)}", tasks=outside(k))
            row["object_proto"], row["goal_proto"] = concepts[(f, obj_k)]["name"], concepts[(f, goal_k)]["name"]
        out[str(t)] = row
    pool = sorted({u for k in {k for v in out.values() for k in (v["object_concept"], v["goal_concept"])} for u in mem[k]})
    mono = {str(f): sorted(u for u in pool if fold.get(u) != f) for f in range(FOLDS)}
    json.dump(dict(tasks=out, concepts={f"{f}|{k}": v for (f, k), v in concepts.items()}, mono=mono, folds=fold),
              open(PLAN, "w"), indent=1)
    el = [t for t, v in out.items() if v["eligible"]]
    print(f"[g0] {len(tasks)} tasks, {len(el)} eligible; {len(concepts)} fold concepts; monolithic pools "
          f"{[len(v) for v in mono.values()]} tasks", flush=True)


def fit():
    from lifelong_ood import ledger
    pl = json.load(open(PLAN))
    for fk, e in pl["concepts"].items():
        if os.path.exists(os.path.join(R, f"prim_{e['name']}_0.pt")):
            continue
        k = fk.split("|", 1)[1]
        key = k[2:] if k[0] != "I" else k[2:].rsplit("|", 1)[0]
        flag = "--goal" if k[0] == "G" else "--approach"
        print(e["name"], k, E1.sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, e['tasks']))} --key "
                                  f"{key.replace(' ', '_')} --name {e['name']} --rank {RANK} {flag}"), flush=True)
    for f, pool in pl["mono"].items():
        if not os.path.exists(os.path.join(R, f"bcl_joint_g0f{f}.pt")):
            print("mono", f, E1.sb(f"experiments/b_continual.py --mode joint --tasks {'+'.join(map(str, pool))} --joint-steps "
                                   f"{MONO_STEPS} --tag _g0f{f}"), flush=True)
    wrong = set(ledger.load("l90_scenes", tag="map")[-1]["metrics"]["wrong_scene_tasks"])
    for t in pl["tasks"]:
        f_ = f"r3_lora_{t}_sc.pt" if int(t) in wrong else f"r3_lora_{t}.pt"
        if int(t) != 51 and not os.path.exists(os.path.join(R, f_)):
            print("lora", t, E1.sb(f"experiments/r3_fewshot.py --task {t}" + (" --tag _sc" if int(t) in wrong else "")), flush=True)


def arms(v):
    a = {"zs": "zs", "gfac": f"fac@{v['object_proto']}.{v['goal_proto']}", "mono": f"bcl@joint_g0f{v['fold']}"}
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def eligible():
    return {t: v for t, v in json.load(open(PLAN))["tasks"].items() if v["eligible"]}


def launch(exp, trials, fresh):
    for t, v in eligible().items():
        a = list(arms(v).values()) + (["lorasc"] if int(t) != 51 else [])
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(a), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    C = E1.compare
    acc, test, missing = {}, {}, []
    for t, v in eligible().items():
        a = dict(arms(v), **({"lora": "lorasc"} if int(t) != 51 else {}))
        s = {k: load("g0_accept", t, x) for k, x in a.items()}
        x_ = {k: load("g0_main", t, x) for k, x in a.items()}
        if None in s.values() or None in x_.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(z.values())) for k, z in s.items()}
        test[t] = x_
    keep = lambda t, k: k if acc[t][k] > acc[t]["zs"] else "zs"                                    # noqa: E731
    hyp = {"H1": C({t: {"x": x["gfac"], "y": x["zs"]} for t, x in test.items()}),
           "H1b": C({t: {"x": x[keep(t, "gfac")], "y": x["zs"]} for t, x in test.items()}),
           "H2": C({t: {"x": x["gfac"], "y": x["mono"]} for t, x in test.items()}),
           "H2b": C({t: {"x": x[keep(t, "gfac")], "y": x[keep(t, "mono")]} for t, x in test.items()})}
    for h in hyp.values():
        h["passes"] = bool(h["x"] > h["y"] and h["sign_p"] < 0.05)
    desc = {"mono_vs_zs": C({t: {"x": x["mono"], "y": x["zs"]} for t, x in test.items()}),
            "gfac_vs_lora": C({t: {"x": x["gfac"], "y": x["lora"]} for t, x in test.items() if "lora" in x}),
            "lora_vs_zs": C({t: {"x": x["lora"], "y": x["zs"]} for t, x in test.items() if "lora" in x})}
    groups = {}                                         # EXPLORATORY (not pre-registered): split by whether the policy alone
    for t, x in test.items():                           # succeeds on at least half of the task's test episodes
        z = int(sum(x["zs"].values()))
        g = groups.setdefault("policy_fails" if z < len(x["zs"]) / 2 else "policy_succeeds",
                              dict(tasks=0, zs=0, gfac=0, mono=0, gfac_better=0, gfac_worse=0))
        f_ = int(sum(x["gfac"].values()))
        g["tasks"] += 1; g["zs"] += z; g["gfac"] += f_; g["mono"] += int(sum(x["mono"].values()))   # noqa: E702
        g["gfac_better"] += f_ > z; g["gfac_worse"] += f_ < z                                        # noqa: E702
    res = dict(missing=missing, acceptance=acc, hypotheses=hyp, descriptive=desc, exploratory_by_policy=groups,
               per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()})
    for k, v in list(hyp.items()) + list(desc.items()):
        print(f"[g0] {k}: {v['x']} vs {v['y']} of {v['n']} ({v['tasks']} tasks), diff {v['diff']['diff']:+.3f} "
              f"[{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}], sign p {v['sign_p']:.3g}", flush=True)
    json.dump(res, open(os.path.join(R, "g0_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g0_recombine", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g0_recombine.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("g0_accept", 10, False)
    if "--test" in sys.argv:
        launch("g0_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
