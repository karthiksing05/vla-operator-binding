"""Plan G, read G4: COMPOSITIONAL OOD on a published benchmark -- Li's LIBERO-Goal-OOD and LIBERO-Spatial-OOD (arXiv 2505.03500,
IJCAI 2026; reused by arXiv 2607.00351), 20 tasks that pair the object of one of pi0.5's training tasks with the target of
another.  No demonstration of any test task: every concept is fitted on pi0.5's own training demonstrations (LIBERO-Spatial,
-Object, -Goal, -10), the object prototype on the frames before the grasp of every training task whose object concept it is, the
goal prototype on the frames from the grasp to the release of every training task whose goal concept it is (ndemo
demonstrations per task).  A test task's concepts are read from its instruction (lifelong_ood/policy/primitives.py) and matched
to the library's: exactly, else the concept whose content words overlap most (Jaccard >= 0.5, prepositions kept); a factor with
no match is served by the policy alone.  Acceptance: 10 practice episodes on start states set a (LIBERO's sampler, seeds 300000+)
keep the composition only if it beats the policy; test on set b (seeds 400000+, 20 per task), paired; Li's modified LIBERO
(its looser On, its flat stove; experiments/libero_client.py), max 300 steps.

    $PY experiments/g4_ood.py --plan [--fit] | --accept --test | --read [--record]
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "g4_plan.json")
SEED, RANK, NDEMO = 13, 4, 5
SUITES = {"goal": "libero_goal_ood", "spatial": "libero_spatial_ood"}
STOP = {"the", "a", "of", "it", "and"}


def words(key):
    return {w for w in re.split(r"[ |]+", key) if w and w not in STOP}


def match(key, lib):
    """The library concept for `key`: exact, else the best content-word Jaccard >= 0.5 (unique best), else None."""
    if key in lib:
        return key
    sc = sorted(((len(words(key) & words(k)) / len(words(key) | words(k)), k) for k in lib), reverse=True)
    if sc and sc[0][0] >= 0.5 and (len(sc) == 1 or sc[1][0] < sc[0][0]):
        return sc[0][1]
    return None


def plan():
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy import primitives as P
    dm = D.Demos()
    obj, goal = {}, {}
    for ti, text in sorted(dm.tasks.items()):
        if len(P.steps_of(text)) != 1:
            continue
        obj.setdefault(P.object_key(text), []).append(ti)
        if P.goal_key(text).split("|")[0]:                          # a goal needs a preposition (not "turn on the stove")
            goal.setdefault(P.goal_key(text), []).append(ti)
    names, tasks = {}, {}
    for t in F.load(F.OOD_SUITES):
        short = next(k for k, v in SUITES.items() if v == t.suite)
        ok, gk = P.object_key(t.language), P.goal_key(t.language)
        mo, mg = match(ok, obj), match(gk, goal)
        ent = dict(suite=t.suite, index=t.index, language=t.language, object_key=ok, goal_key=gk, object_concept=mo,
                   goal_concept=mg, object_proto=None, goal_proto=None)
        for kind, m_, lib, flag in (("object", mo, obj, "--approach"), ("goal", mg, goal, "--goal")):
            if m_ is None:
                continue
            n = names.setdefault((kind, m_), dict(name=f"Q{kind[0].upper()}{len(names):02d}", kind=kind, key=m_, tasks=lib[m_],
                                                   flag=flag))["name"]
            ent[f"{kind}_proto"] = n
        tasks[f"{short}:{t.index}"] = ent
    out = dict(tasks=tasks, protos={v["name"]: v for v in names.values()}, ndemo=NDEMO, rank=RANK)
    json.dump(out, open(PLAN, "w"), indent=1)
    for k, v in tasks.items():
        print(f"[g4] {k:11} {v['language'][:46]:46} O {v['object_concept']!s:44} G {v['goal_concept']}", flush=True)
    print(f"[g4] {len(tasks)} tasks, {len(names)} prototypes", flush=True)


def sb(script):
    out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                          "--account=overcap", "--qos=short", f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def fit():
    for p in json.load(open(PLAN))["protos"].values():
        if not os.path.exists(os.path.join(R, f"prim_{p['name']}_0.pt")):
            print(p["name"], sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, p['tasks']))} --key {p['key'].replace(' ', '_')} "
                                f"--name {p['name']} --rank {RANK} {p['flag']} --source d40 --ndemo {NDEMO}"), flush=True)


def arms(v):
    a = {"zs": "zs"}
    if v["object_proto"] or v["goal_proto"]:
        a["fac"] = f"fac@{v['object_proto'] or 'x'}.{v['goal_proto'] or 'x'}"
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(stage, trials, fresh_dir):
    for k, v in json.load(open(PLAN))["tasks"].items():
        short = k.split(":")[0]
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(arms(v).values()),
               "--experiment", f"g4_{stage}_{short}", "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track",
               "--partition", "overcap", "--per", "2", "--suite", v["suite"], "--fresh", "--fresh-dir", fresh_dir]
        print(k, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def load(exp, suite, t, a):
    f = os.path.join(R, f"{exp}_{suite}_{t}_{tag(a)}.json")                # launch_pack names a non-default suite in the file
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    import e1_read as E1
    pl = json.load(open(PLAN))
    acc, test, kept, missing = {}, {}, {}, []
    for k, v in pl["tasks"].items():
        short = k.split(":")[0]
        A = arms(v)
        s = {a: load(f"g4_accept_{short}", v["suite"], v["index"], x) for a, x in A.items()}
        x_ = {a: load(f"g4_main_{short}", v["suite"], v["index"], x) for a, x in A.items()}
        if None in s.values() or None in x_.values():
            missing.append(k)
            continue
        acc[k] = {a: int(sum(z.values())) for a, z in s.items()}
        kept[k] = "fac" if "fac" in acc[k] and acc[k]["fac"] > acc[k]["zs"] else "zs"
        test[k] = x_
    C = E1.compare
    res = dict(missing=missing, acceptance=acc, kept=kept)
    for name, sel in (("all", lambda k: True), ("goal", lambda k: k.startswith("goal")), ("spatial", lambda k: k.startswith("spatial"))):
        ts = {k: x for k, x in test.items() if sel(k)}
        res[name] = dict(H1=C({k: {"x": x[kept[k]], "y": x["zs"]} for k, x in ts.items()}),
                         fac_vs_zs=C({k: {"x": x["fac"], "y": x["zs"]} for k, x in ts.items() if "fac" in x}),
                         rate_kept=sum(sum(x[kept[k]].values()) for k, x in ts.items()) / max(1, 20 * len(ts)),
                         rate_zs=sum(sum(x["zs"].values()) for k, x in ts.items()) / max(1, 20 * len(ts)))
    res["H1"] = res["all"]["H1"]
    res["H1"]["passes"] = bool(res["H1"]["x"] > res["H1"]["y"] and res["H1"]["sign_p"] < 0.05)
    res["per_task"] = {k: {a: int(sum(z.values())) for a, z in x.items()} | {"kept": kept[k]} for k, x in test.items()}
    for n in ("all", "goal", "spatial"):
        h = res[n]["H1"]
        print(f"[g4] {n:8} kept {h['x']} vs zs {h['y']} of {h['n']} (sign p {h['sign_p']:.3g}); rates {res[n]['rate_kept']:.3f} vs "
              f"{res[n]['rate_zs']:.3f}; fac alone {res[n]['fac_vs_zs']['x']}", flush=True)
    if missing:
        print("[g4] missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g4_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(ndemo=NDEMO),
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


def source(tasks, suite):
    """Plan G4b: the training task whose instruction stands for a concept in this scene -- the concept's source tasks from the test
    task's own base suite (libero_goal_ood -> libero_goal), else all of them; the first in benchmark order."""
    from lifelong_ood.policy import data as D
    base = D.SUITES[suite.replace("_ood", "")]
    own = [t for t in tasks if t in base]
    return (own or tasks)[0]


def pfield(v, pl):
    o = source(pl["protos"][v["object_proto"]]["tasks"], v["suite"]) if v["object_proto"] else "x"
    g = source(pl["protos"][v["goal_proto"]]["tasks"], v["suite"]) if v["goal_proto"] else "x"
    return f"pfield@{o}.{g}"


def screen(launch_it):
    """Plan G4b screen (exploratory): the policy's own fields under the concepts' source instructions, switched at the grasp, on
    start states set a (10 per task, noise seed 13), against the policy's recorded episodes there."""
    pl = json.load(open(PLAN))
    if launch_it:
        for k, v in pl["tasks"].items():
            cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", pfield(v, pl), "--experiment",
                   f"g4_accept_{k.split(':')[0]}", "--first", "0", "--trials", "10", "--seed", str(SEED), "--bind", "track",
                   "--partition", "overcap", "--per", "1", "--suite", v["suite"], "--fresh", "--fresh-dir", "fresh_ood_a"]
            print(k, pfield(v, pl), subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-60:], flush=True)
        return
    tot, rows = dict(zs=0, fac=0, pf=0), {}
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        z, f, p_ = (load(f"g4_accept_{sh}", v["suite"], v["index"], a) for a in ("zs", arms(v)["fac"], pfield(v, pl)))
        if p_ is None:
            continue
        rows[k] = dict(zs=sum(z.values()), fac=sum(f.values()), pf=sum(p_.values()), arm=pfield(v, pl))
        for a in tot:
            tot[a] += rows[k][a]
        print(f"[g4b] {k:11} {v['language'][:44]:44} zs {rows[k]['zs']:2} fac {rows[k]['fac']:2} pfield {rows[k]['pf']:2} "
              f"({rows[k]['arm']})", flush=True)
    print(f"[g4b] {len(rows)} tasks: zs {tot['zs']} fac {tot['fac']} pfield {tot['pf']}", flush=True)
    return rows


def diagnose(record):
    """Why the library adds nothing here: each prototype's flow loss on its first logged batch and its last three (ledger c0_fit), G4's
    prototypes (pi0.5's own training demonstrations) against E3's (LIBERO-90 demonstrations the policy never saw)."""
    import glob
    import numpy as np

    def curve(n):
        fs = sorted(glob.glob(os.path.join(ROOT, "ledger", "c0_fit", f"*joint_{n}__*.json")))
        c = json.load(open(fs[-1]))["metrics"]["curve"]
        loss = c["loss"] if isinstance(c, dict) else c
        return float(loss[0]), float(np.mean(loss[-3:]))                    # 11 logged batches: the first, the mean of the last 3
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    names = {"g4": sorted(json.load(open(PLAN))["protos"]),
             "e3": sorted({v["object_proto"] for v in e3.values()} | {v["goal_proto"] for v in e3.values()})}
    res = {}
    for k, ns in names.items():
        c = [curve(n) for n in ns]
        res[k] = dict(n=len(c), first=float(np.mean([a for a, _ in c])), last=float(np.mean([b for _, b in c])))
    print(json.dumps(res, indent=1), flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag="diagnosis", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--screen" in sys.argv:
        screen("--launch" in sys.argv)
    if "--diagnose" in sys.argv:
        diagnose("--record" in sys.argv)
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("accept", 10, "fresh_ood_a")
    if "--test" in sys.argv:
        launch("main", 20, "fresh_ood_b")
    if "--read" in sys.argv:
        read("--record" in sys.argv)
