"""Plan G, read G5a (docs/PLAN_GENERALIZE.md; ledger m3_prereg tag g5_factors): FACTOR-LEVEL selection.  G0 showed recombined
concepts supply a missing skill and interfere with a present one.  Here practice chooses per factor: on G0's held-out tasks, with
G0's fold concepts (no demonstration of the task anywhere), the object concept alone (fac@<object>.x: the policy carries) and the
goal concept alone (fac@x.<goal>: the policy approaches), at G0's states and seed.  Compared: acceptance over {policy, object,
goal, both} against acceptance over {policy, both} (G0's H1b arm).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/g5_factors.py --accept --test
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g5_factors.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec_ = importlib.util.spec_from_file_location("g0", os.path.join(ROOT, "experiments", "g0_recombine.py"))
G0 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(G0)
R, PY, SEED = G0.R, G0.PY, G0.SEED
ORDER = ("zs", "gfac", "obj", "goal")                       # ties: the policy, then both factors, then one


def arms(v):
    return {"obj": f"fac@{v['object_proto']}.x", "goal": f"fac@x.{v['goal_proto']}"}


def launch(exp, trials, fresh):
    for t, v in G0.eligible().items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def read(record):
    C = G0.E1.compare
    test, acc, missing = {}, {}, []
    for t, v in G0.eligible().items():
        a = {"zs": "zs", "gfac": G0.arms(v)["gfac"], **arms(v)}
        x = {k: G0.load("g5_main" if k in ("obj", "goal") else "g0_main", t, z) for k, z in a.items()}
        s = {k: G0.load("g5_accept" if k in ("obj", "goal") else "g0_accept", t, z) for k, z in a.items()}
        if None in x.values() or None in s.values():
            missing.append(t)
            continue
        test[t], acc[t] = x, {k: int(sum(z.values())) for k, z in s.items()}
    pick = lambda t, opts: max(opts, key=lambda k: (acc[t][k], -ORDER.index(k)))              # noqa: E731
    h1 = C({t: {"x": x[pick(t, ORDER)], "y": x[pick(t, ("zs", "gfac"))]} for t, x in test.items()})
    h0 = C({t: {"x": x[pick(t, ORDER)], "y": x["zs"]} for t, x in test.items()})
    desc = {k: C({t: {"x": x[k], "y": x["zs"]} for t, x in test.items()}) for k in ("obj", "goal", "gfac")}
    kept = {k: sum(pick(t, ORDER) == k for t in test) for k in ORDER}
    res = dict(missing=missing, H1=h1, factor_kept_vs_zs=h0, alone_vs_zs=desc, kept=kept,
               per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()})
    for k, v in [("H1 factor-level vs both-only", h1), ("factor-level vs zs", h0)] + list(desc.items()):
        print(f"[g5] {k}: {v['x']} vs {v['y']} of {v['n']} ({v['tasks']} tasks), diff {v['diff']['diff']:+.3f} "
              f"[{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}], sign p {v['sign_p']:.3g}", flush=True)
    print("[g5] kept", kept, flush=True)
    json.dump(res, open(os.path.join(R, "g5_read.json"), "w"), indent=1)
    if record and not missing:
        sys.path.insert(0, ROOT)
        from lifelong_ood import ledger
        print(ledger.record("g5_factors", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g5_factors.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--accept" in sys.argv:
        launch("g5_accept", 10, False)
    if "--test" in sys.argv:
        launch("g5_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
