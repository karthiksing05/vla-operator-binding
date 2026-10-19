"""Plan Z, read Z3 (ledger m3_prereg tag z3_l90_pointers): the ZERO-FINE-TUNE row on LIBERO-90.  On the E3 / E5 tasks whose object
and goal concepts are both held as words in pi0.5's 40 training instructions, the policy's own fields under two training
instructions, switched at the grasp (pfield@<object source>.<goal source>); nothing is fitted.  The demonstration is used only as
a query: each source is the candidate instruction (holding the concept) with the lowest mean flow loss on the demo
(experiments/zu_score.py's scores).  Test on set d, 20 per task, noise seed 13 -- the start states on which G8 tested the policy,
the fitted library and the per-task LoRA (ledger g8_main).

    $PY experiments/z3_l90_pointers.py --launch | --read [--record]
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED = 13


def held():
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    ho, hg = {}, {}
    for i, t in D.Demos().tasks.items():
        for c in P.steps_of(t):
            try:
                ho.setdefault(P.object_key(c), []).append(i)
                hg.setdefault(P.goal_key(c), []).append(i)
            except Exception:
                pass
    return ho, hg


def plan():
    """-> {task: (object source, goal source)} for the both-held tasks."""
    ho, hg = held()
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    e5 = json.load(open(os.path.join(R, "e5_plan.json")))["tasks"]
    out = {}
    for t, v in list((t, v) for t, v in e3.items() if v["lora"]) + list(e5.items()):
        o, g = v["object_key"], v["goal_key"]
        if o not in ho or g not in hg:
            continue
        z = np.load(os.path.join(R, "u1", f"l90_{t}_d0.npz"))
        L = np.log(((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2)).mean(0)        # [42] mean log loss per prompt
        out[t] = (min(ho[o], key=lambda i: (L[i], i)), min(hg[g], key=lambda i: (L[i], i)))
    return out


def arm(s):
    return f"pfield@{s[0]}.{s[1]}"


def launch():
    for t, s in plan().items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", arm(s), "--experiment", "z3_main", "--first", "0",
               "--trials", "20", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "3", "--fresh",
               "--fresh-dir", "fresh_states_d"]
        print(t, s, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-70:], flush=True)


def _ld(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{a.replace('@', '_')}_seed{SEED}.json")
    return ({int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}
            if os.path.exists(f) else None)


def read(record):
    import e1_read as E1
    import g8_leaves as G8
    C = E1.compare
    g8 = G8.tasks()
    x, missing = {}, []
    for t, s in plan().items():
        arms = g8[t][1]
        z = dict(pointers=_ld("z3_main", t, arm(s)), zs=_ld("g8_main", t, arms["zs"]), library=_ld("g8_main", t, arms["hfac"]),
                 lora=_ld("g8_main", t, arms["lora"]))
        if None in z.values():
            missing.append(t)
            continue
        x[t] = z
    res = dict(tasks=len(x), missing=missing, sources=plan(),
               pointers_vs_zs=C({t: {"x": z["pointers"], "y": z["zs"]} for t, z in x.items()}),
               pointers_vs_library=C({t: {"x": z["pointers"], "y": z["library"]} for t, z in x.items()}),
               pointers_vs_lora=C({t: {"x": z["pointers"], "y": z["lora"]} for t, z in x.items()}),
               per_task={t: {k: int(sum(v.values())) for k, v in z.items()} for t, z in x.items()})
    for n in ("pointers_vs_zs", "pointers_vs_library", "pointers_vs_lora"):
        h = res[n]
        print(f"[z3] {n}: {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]")
    print(json.dumps(res["per_task"]), "missing", missing)
    json.dump(res, open(os.path.join(R, "z3_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("z3_l90_pointers", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/z3_l90_pointers.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
