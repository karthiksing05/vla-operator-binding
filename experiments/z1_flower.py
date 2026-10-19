"""Plan Z, read Z1 (ledger m3_prereg tag z1_flower): FLOWER-goal with no fine-tuning on the 6 OOD tasks its library covers.

    $PY experiments/z1_flower.py --run
    $PY experiments/z1_flower.py --read [--record]
    $PY experiments/z1_flower.py --run-b          Z1b (ledger m3_prereg tag z1b_flower): practice chooses the sources inside FLOWER's
                                                  suite (G10's two stages, 5 episodes each on set a), test on set f
    $PY experiments/z1_flower.py --read-b [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
SEED, BB, EXP = 13, "flower", "z1_main_goal"
TASKS = [f"goal:{i}" for i in range(4, 10)]


def arms(v, pl):
    import g4_ood as G
    return dict(zs="zs", ours=G.pfield(v, pl), li_grasp=G.li_arms(v, pl)["li_grasp"])


def run():
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    for k in TASKS:
        v = pl["tasks"][k]
        print(k, G._launch(EXP, k, v, list(dict.fromkeys(arms(v, pl).values())), 20, "fresh_ood_e", BB), flush=True)


def read(record):
    import e1_read as E1
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    x, missing = {}, []
    for k in TASKS:
        v = pl["tasks"][k]
        z = {n: G._ld(EXP, v, a, BB) for n, a in arms(v, pl).items()}
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    res = dict(backbone=BB, tasks=len(x), missing=missing,
               ours_vs_zs=C({k: {"x": z["ours"], "y": z["zs"]} for k, z in x.items()}) if x else None,
               ours_vs_li=C({k: {"x": z["ours"], "y": z["li_grasp"]} for k, z in x.items()}) if x else None,
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    if x:
        res["ours_vs_li"]["noninferior"] = bool(res["ours_vs_li"]["diff"]["lo"] > -0.05)
        for n in ("ours_vs_zs", "ours_vs_li"):
            h = res[n]
            print(f"[z1] {n}: {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                  f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]")
    print(json.dumps(res["per_task"]), "missing", missing)
    json.dump(res, open(os.path.join(R, "z1_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("z1_flower", bench="libero_ood", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=BB),
                            script="experiments/z1_flower.py", command=" ".join(sys.argv)))


POOL = [10, 11, 13, 14, 17, 18]                                         # source_pool() inside FLOWER-goal's 10 tasks


def choose(stage):
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    out = {}
    for k in TASKS:
        v = pl["tasks"][k]
        oa, og = G.ours_sources(v, pl)
        if stage == 1:
            arms_, pref = {a: f"pfield@{a}.{og}" for a in POOL}, oa
        else:
            a1 = choose(1)[k]
            arms_, pref = {g: f"pfield@{a1}.{g}" for g in POOL}, og
        sc = {c: G._ld(f"z1_s{stage}_goal", v, arm, BB) for c, arm in arms_.items()}
        out[k] = None if None in sc.values() else max(POOL, key=lambda c: (sum(sc[c].values()), c == pref, -c))
    return out


def arms_b(v, k, s1, s2):
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    return dict(zs="zs", chosen=f"pfield@{s1[k]}.{s2[k]}", li_grasp=G.li_arms(v, pl)["li_grasp"])


def run_b():
    import time
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    for stage in (1, 2):
        s1 = choose(1) if stage == 2 else None
        for k in TASKS:
            v = pl["tasks"][k]
            oa, og = G.ours_sources(v, pl)
            arms_ = [f"pfield@{a}.{og}" for a in POOL] if stage == 1 else [f"pfield@{s1[k]}.{g}" for g in POOL]
            print(k, G._launch(f"z1_s{stage}_goal", k, v, arms_, 5, "fresh_ood_a", BB), flush=True)
        while None in choose(stage).values():
            time.sleep(120)
        print(f"[z1b] stage {stage}: {choose(stage)}", flush=True)
    s1, s2 = choose(1), choose(2)
    for k in TASKS:
        v = pl["tasks"][k]
        print(k, G._launch("z1b_main_goal", k, v, list(dict.fromkeys(arms_b(v, k, s1, s2).values())), 20, "fresh_ood_f", BB), flush=True)
    print("[z1b] Z1B_TEST_LAUNCHED", flush=True)


def read_b(record):
    import e1_read as E1
    import g4_ood as G
    pl = json.load(open(G.PLAN))
    s1, s2 = choose(1), choose(2)
    x, missing = {}, []
    for k in TASKS:
        v = pl["tasks"][k]
        z = {n: G._ld("z1b_main_goal", v, a, BB) for n, a in arms_b(v, k, s1, s2).items()}
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    res = dict(backbone=BB, tasks=len(x), missing=missing, sources={k: [s1[k], s2[k]] for k in TASKS},
               chosen_vs_zs=C({k: {"x": z["chosen"], "y": z["zs"]} for k, z in x.items()}) if x else None,
               chosen_vs_li=C({k: {"x": z["chosen"], "y": z["li_grasp"]} for k, z in x.items()}) if x else None,
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    if x:
        res["chosen_vs_li"]["noninferior"] = bool(res["chosen_vs_li"]["diff"]["lo"] > -0.05)
        for n in ("chosen_vs_zs", "chosen_vs_li"):
            h = res[n]
            print(f"[z1b] {n}: {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                  f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]")
    print(json.dumps(res["per_task"]), res["sources"], "missing", missing)
    json.dump(res, open(os.path.join(R, "z1b_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("z1_flower", bench="libero_ood", tag="read_b", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=BB),
                            script="experiments/z1_flower.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--run-b" in sys.argv:
        run_b()
    if "--read-b" in sys.argv:
        read_b("--record" in sys.argv)
    if "--run" in sys.argv:
        run()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
