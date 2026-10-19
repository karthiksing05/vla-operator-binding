"""Plan Z, track U, pilot U3 (exploratory, LIBERO's standard init states 0-9: no fresh set exists for these tasks): pointer programs (lifelong_ood/policy/
program.py) on the LIBERO-90 tasks OUTSIDE E3 / E5's template (drawers, knobs, the microwave, stacking, two-step tasks).  The demo is
cut at its contact events and each segment named by pi0.5's likelihood over its 40 training instructions; nothing is fitted.

    $PY experiments/z4_programs.py --launch | --read
"""
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
SEED, EXP, FRESH, N = 13, "z4_pilot", "fresh_states", 10


def programs():
    from lifelong_ood.policy import program as PG
    cov = set(json.load(open(os.path.join(R, "e3_plan.json")))) | set(json.load(open(os.path.join(R, "e5_plan.json")))["tasks"])
    out = {}
    for t in range(90):
        if str(t) in cov:
            continue
        z = np.load(os.path.join(R, "u1", f"l90_{t}_d0.npz"))
        L = np.log(((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2))
        out[str(t)] = PG.label(L, z["frames"], PG.segments(z["actions"]), list(range(40)))
    return out


def arms(p):
    return dict(zs="zs", prog="prog@" + ".".join(map(str, p)))


def launch():
    for t, p in programs().items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(p).values()), "--experiment", EXP, "--first",
               "0", "--trials", str(N), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "3"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-60:], flush=True)


def read():
    tot, rows = {}, {}
    for t, p in programs().items():
        r = {}
        for n, a in arms(p).items():
            f = os.path.join(R, f"{EXP}_{t}_{a.replace('@', '_')}_seed{SEED}.json")
            r[n] = (sum(bool(e["success"]) for v in json.load(open(f))["tasks"].values() for e in v["episodes"])
                    if os.path.exists(f) else None)
            if r[n] is not None:
                tot[n] = tot.get(n, 0) + r[n]
        rows[t] = r
        print(t, p, r)
    print("totals", tot)
    json.dump(dict(rows=rows, totals=tot, programs=programs()), open(os.path.join(R, "z4_pilot.json"), "w"), indent=1)


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read()
