"""Plan Z, read Z5 (ledger m3_prereg tag z5_blend): the soft blend of held fields served online (lifelong_ood/policy/blend.py), on
Z3's 25 LIBERO-90 tasks, set d, 20 per task, noise seed 13 -- paired with pi0.5 alone (G8), Z3's pointers and G8's fitted library.
Nothing is fitted in the network: per task, E x 3 segments x 2 dim groups weights fitted to the demo (experiments/zi_blend.py).
The behavioural donation: per task, the 3 experts with the largest offline Shapley donation, each ablated, on the first 10 states.

    $PY experiments/z5_blend.py --launch | --read [--record]
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
SEED, EXP, EXPA = 13, "z5_main", "z5_ablate"


def rows():
    return {r["name"][4:]: r for r in json.load(open(os.path.join(R, "zi_blend.json")))["rows"] if r["name"].startswith("l90_")}


def eid_s(e):
    return f"m{-int(e)}" if int(e) < 0 else str(int(e))


def donations(r):
    """-> {expert id: offline Shapley donation summed over segments and dim groups}."""
    d = {}
    for seg in r["segments"].values():
        for g, sh in seg["shapley"].items():
            for e, v in sh.items():
                d[int(e)] = d.get(int(e), 0.0) + float(v)
    return d


def top(r, k=3):
    d = donations(r)
    return sorted(d, key=lambda e: -d[e])[:k]


def launch():
    for t, r in rows().items():
        a = [f"blend@l90_{t}"]
        b = [f"blend@l90_{t}.x{eid_s(e)}" for e in top(r)]
        for exp, arms, n in ((EXP, a, 20), (EXPA, b, 10)):
            cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms), "--experiment", exp, "--first", "0",
                   "--trials", str(n), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2", "--fresh",
                   "--fresh-dir", "fresh_states_d"]
            print(t, exp, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-60:], flush=True)


def _ld(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{a.replace('@', '_')}_seed{SEED}.json")
    return ({int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}
            if os.path.exists(f) else None)


def read(record):
    import e1_read as E1
    import g8_leaves as G8
    import z3_l90_pointers as Z3
    C = E1.compare
    g8, z3 = G8.tasks(), Z3.plan()
    x, abl, missing = {}, [], []
    for t, r in rows().items():
        arms = g8[t][1]
        z = dict(blend=_ld(EXP, t, f"blend@l90_{t}"), zs=_ld("g8_main", t, arms["zs"]), library=_ld("g8_main", t, arms["hfac"]),
                 pointers=_ld("z3_main", t, Z3.arm(z3[t])))
        if None in z.values():
            missing.append(t)
            continue
        x[t] = z
        b10 = sum(v for k, v in z["blend"].items() if k < 10)
        d = donations(r)
        for e in top(r):
            y = _ld(EXPA, t, f"blend@l90_{t}.x{eid_s(e)}")
            if y is not None:
                abl.append(dict(task=t, expert=e, offline=d[e], online=(b10 - sum(y.values())) / 10.0))
    res = dict(tasks=len(x), missing=missing,
               blend_vs_zs=C({t: {"x": z["blend"], "y": z["zs"]} for t, z in x.items()}),
               blend_vs_pointers=C({t: {"x": z["blend"], "y": z["pointers"]} for t, z in x.items()}),
               blend_vs_library=C({t: {"x": z["blend"], "y": z["library"]} for t, z in x.items()}),
               per_task={t: {k: int(sum(v.values())) for k, v in z.items()} for t, z in x.items()}, ablations=abl)
    if len(abl) > 2:
        o, n = np.array([a["offline"] for a in abl]), np.array([a["online"] for a in abl])
        from scipy.stats import spearmanr
        res["donation_agreement"] = dict(n=len(abl), spearman=float(spearmanr(o, n).correlation),
                                         online_mean=float(n.mean()), share_positive=float(np.mean(n > 0)))
    for k in ("blend_vs_zs", "blend_vs_pointers", "blend_vs_library"):
        h = res[k]
        print(f"[z5] {k}: {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]")
    print("[z5] donation agreement", res.get("donation_agreement"), "missing", missing)
    json.dump(res, open(os.path.join(R, "z5_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("z5_blend", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/z5_blend.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
