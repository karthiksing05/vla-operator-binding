"""INTERPRETATION 6: does the frozen policy's own likelihood detect a NEW concept?  (Wang et al.'s stated gap: a mechanism to
detect new primitives.)  Exploratory, offline.

Each concept is scored by the frozen pi0.5's flow-matching loss on its phase frames (c0_fit.py --score: the frames before the
grasp for an object concept, from the grasp to the release for a goal concept; one demonstration per task; 8 fixed draws of 64
frames, time and noise).  HELD concepts: every object and goal concept of pi0.5's own single-step training tasks (LIBERO-Spatial,
-Object, -Goal, -10).  NEW concepts: the object types and goals of the LIBERO-90 library (E3's and E5's), demonstrations the
policy never saw.  Reports the separation (AUC), the ranges, and a threshold set from the held concepts alone (their maximum).
W0 found that likelihood does not SELECT among new concepts; this asks whether it DETECTS that a concept is new.

    $PY experiments/i6_novelty.py --launch     # one scoring job per concept
    $PY experiments/i6_novelty.py --read [--record]
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")


def concepts():
    """-> [(name, kind, key, tasks, source, held)]."""
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    out, obj, goal = [], {}, {}
    for ti, text in sorted(D.Demos().tasks.items()):
        if len(P.steps_of(text)) != 1:
            continue
        if P.object_key(text).startswith("put|"):                  # a grasped object (not a drawer, a knob or a push)
            obj.setdefault(P.object_key(text), []).append(ti)
        if P.goal_key(text).split("|")[0]:
            goal.setdefault(P.goal_key(text), []).append(ti)
    for i, (k, ts) in enumerate(sorted(obj.items())):
        out.append((f"H6o{i:02d}", "object", k, ts, "d40", True))
    pushed = {P.goal_key(t) for t in D.Demos().tasks.values() if len(P.steps_of(t)) == 1 and not P.object_key(t).startswith("put|")}
    for i, (k, ts) in enumerate(sorted(goal.items())):
        if k not in pushed:                                         # a carried goal (the plate pushed to the stove carries nothing)
            out.append((f"H6g{i:02d}", "goal", k, ts, "d40", True))
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    seen = {}
    for v in e3.values():
        seen.setdefault(("object", v["object_key"]), v["obj_tasks"])
        seen.setdefault(("goal", v["goal_key"]), v["goal_tasks"])
    for k, e in json.load(open(os.path.join(R, "e5_plan.json")))["new"].items():
        if k[0] in "OG":
            seen.setdefault(("object" if k[0] == "O" else "goal", k[2:]), e["tasks"])
    for i, ((kind, k), ts) in enumerate(sorted(seen.items())):
        out.append((f"N6{kind[0]}{i:02d}", kind, k, ts, "l90", False))
    return out


def launch():
    for name, kind, key, ts, src, _ in concepts():
        flag = "--approach" if kind == "object" else "--goal"
        script = (f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, ts))} --key {key.replace(' ', '_')} --name {name} "
                  f"--rank 4 {flag} --source {src} --ndemo 1 --score")
        out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                              "--account=overcap", "--qos=short", f"--export=ALL,ENV=openpi,SCRIPT={script}",
                              "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
        print(name, kind, key, out.stdout.strip() or out.stderr.strip(), flush=True)


def read(record):
    from lifelong_ood import ledger
    rows = {r["keys"]["name"]: r["metrics"] for r in ledger.load("i6_novelty", tag="score")}
    held, new, missing = {}, {}, []
    for name, kind, key, ts, src, h in concepts():
        if name not in rows:
            missing.append(name)
            continue
        (held if h else new)[name] = dict(kind=kind, key=key, loss=rows[name]["loss"], frames=rows[name]["frames"])
    a = np.array([v["loss"] for v in held.values()])
    b = np.array([v["loss"] for v in new.values()])
    thr = float(a.max()) if len(a) else None
    res = dict(n_held=len(a), n_new=len(b), missing=missing,
               auc=float(np.mean(a[:, None] < b[None, :])) if len(a) and len(b) else None,
               held_median=float(np.median(a)) if len(a) else None, held_max=thr,
               new_median=float(np.median(b)) if len(b) else None, new_min=float(b.min()) if len(b) else None,
               new_below_threshold=int((b <= thr).sum()) if thr is not None else None,
               by_kind={k: dict(held=[round(v["loss"], 5) for v in held.values() if v["kind"] == k],
                                new=[round(v["loss"], 5) for v in new.values() if v["kind"] == k]) for k in ("object", "goal")},
               held=held, new=new)
    print(json.dumps({k: v for k, v in res.items() if k not in ("held", "new", "by_kind")}, indent=1), flush=True)
    for name, v in sorted({**held, **new}.items(), key=lambda x: x[1]["loss"]):
        print(f"[i6] {v['loss']:.4f}  {'HELD' if name in held else 'new '}  {v['kind']:6} {v['key']}", flush=True)
    json.dump(res, open(os.path.join(R, "i6_novelty.json"), "w"), indent=1)
    if record and not missing:
        print(ledger.record("i6_novelty", bench="libero", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i6_novelty.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
