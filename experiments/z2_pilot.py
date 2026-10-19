"""Plan Z, track B, pilot B2 (exploratory, practice set a; docs/PLAN_Z.md): the word splice (lifelong_ood/policy/bind.py) on the 20
OOD tasks, pi0.5.  A phrase's source is a training instruction that CONTAINS the phrase (the concept's tasks in g4_plan's
prototypes, the "put the ..." template first, then the lowest index); the goal phrase falls back to its head noun.

    sbatch ... SCRIPT="experiments/z2_pilot.py --prep"       cache the bindings (GPU)
    $PY experiments/z2_pilot.py --launch                     zs, splice delta, splice replace; 10 episodes on set a
    $PY experiments/z2_pilot.py --read
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
BB = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi05_libero"


def sources():
    """-> {task: (object source, goal source)} by word containment."""
    import g4_ood as G
    from lifelong_ood.policy import bind as BD
    from lifelong_ood.policy import data as D
    tx = D.Demos().tasks
    pl = json.load(open(G.PLAN))
    out = {}
    for k, v in pl["tasks"].items():
        po, pg = BD.phrases(v["language"])
        oa, og = G.ours_sources(v, pl)

        def pick(ph, cands, fallback):
            for p in (ph, ph.split()[-1]):
                hit = [t for t in cands if re.search(rf"\b{re.escape(p)}\b", tx[t])]
                if hit:
                    return sorted(hit, key=lambda t: (not tx[t].startswith("put the"), t))[0]
            return fallback
        out[k] = (pick(po, pl["protos"][v["object_proto"]]["tasks"], oa), pick(pg, pl["protos"][v["goal_proto"]]["tasks"], og))
    return out


def arms(k, s):
    o, g = s[k]
    return dict(zs="zs", delta=f"splice@{o}.{g}.delta", replace=f"splice@{o}.{g}.replace", phase=f"splice@{o}.{g}.delta.ph")


def prep():
    import g4_ood as G
    from lifelong_ood.policy import bind as BD
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(BB, device="cuda")
    dm = D.Demos()
    pl = json.load(open(G.PLAN))
    for k, (o, g) in sources().items():
        new = pl["tasks"][k]["language"]
        po, pg = BD.phrases(new)
        for w, t, ph in (("o", o, po), ("g", g, pg)):
            m = BD.match(bk, new, dm.tasks[t], ph)
            if m:
                BD.load_or_identify(bk, BB, dm, t)
                BD.load_or_identify(bk, BB, dm, t, new)
            print(k, new, "|", w, t, dm.tasks[t], "->", m and m[2], flush=True)
    print("[z2] prep done", flush=True)


def launch():
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    s = sources()
    for k, v in pl["tasks"].items():
        print(k, G._launch(f"z2_pilot_{k.split(':')[0]}", k, v, list(arms(k, s).values()), 10, "fresh_ood_a", BB), flush=True)


def read():
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    s = sources()
    tot, rows = {}, {}
    for k, v in pl["tasks"].items():
        z = {n: G._ld(f"z2_pilot_{k.split(':')[0]}", v, a, BB) for n, a in arms(k, s).items()}
        rows[k] = {n: (None if r is None else sum(r.values())) for n, r in z.items()}
        for n, c in rows[k].items():
            if c is not None:
                tot[n] = tot.get(n, 0) + c
        print(k, v["language"], s[k], rows[k])
    print("totals", tot)
    json.dump(dict(rows=rows, totals=tot, sources=s), open(os.path.join(R, f"z2_pilot_{BB}.json"), "w"), indent=1)


if __name__ == "__main__":
    if "--prep" in sys.argv:
        prep()
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read()
