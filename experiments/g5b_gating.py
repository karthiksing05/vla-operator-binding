"""Plan G, read G5b (ledger m3_prereg tag g5b_gating): does the SWITCH AT THE POLICY'S GRASP matter?  E3's library on the stream's 30
tasks, served with the object (or instance) prototype and the goal prototype BOTH ON for the whole episode, against E3's gated runs
(object until the grasp, goal after it) at the same fresh start states and noise seed (13).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/g5b_gating.py --test
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g5b_gating.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec_ = importlib.util.spec_from_file_location("e3", os.path.join(ROOT, "experiments", "e3_read.py"))
E3 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(E3)
R, PY, SEED = E3.R, E3.PY, E3.SEED


def arm(v):
    return f"ungated@{v['object_proto']}.{v['goal_proto']}"


def test():
    for t, v in json.load(open(E3.PLAN)).items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", arm(v), "--experiment", "g5b_main", "--first", "0", "--trials", "20",
               "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2", "--fresh", "--fresh-dir", "fresh_states_b"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-100:], flush=True)


def read(record):
    tab, missing = {}, []
    for t, v in json.load(open(E3.PLAN)).items():
        x, y = E3.load("e3_main", t, f"fac@{v['object_proto']}.{v['goal_proto']}"), E3.load("g5b_main", t, arm(v))
        z = E3.load("e3_main", t, "zs")
        if x is None or y is None:
            missing.append(t)
            continue
        tab[t] = {"x": x, "y": y, "z": z}
    h1 = E3.E1.compare({t: {"x": v["x"], "y": v["y"]} for t, v in tab.items()})
    h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
    zs = E3.E1.compare({t: {"x": v["y"], "y": v["z"]} for t, v in tab.items()})
    res = dict(H1=h1, ungated_vs_zs=zs, missing=missing,
               per_task={t: dict(gated=sum(v["x"].values()), ungated=sum(v["y"].values()), zs=sum(v["z"].values())) for t, v in tab.items()})
    print(f"[g5b] gated {h1['x']} vs ungated {h1['y']} of {h1['n']} ({h1['tasks']} tasks), diff {h1['diff']['diff']:+.3f} "
          f"[{h1['diff']['lo']:+.3f}, {h1['diff']['hi']:+.3f}], sign p {h1['sign_p']:.3g}; ungated vs zs {zs['x']} vs {zs['y']}", flush=True)
    json.dump(res, open(os.path.join(R, "g5b_read.json"), "w"), indent=1)
    if record and not missing:
        sys.path.insert(0, ROOT)
        from lifelong_ood import ledger
        print(ledger.record("g5b_gating", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g5b_gating.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--test" in sys.argv:
        test()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
