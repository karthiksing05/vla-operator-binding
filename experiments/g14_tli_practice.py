"""Plan G, read G14 (iteration 11; ledger m3_prereg tag g14_tli_practice): practice chooses the held concepts' sources under TLI
(lambda 24), by G10's two-stage procedure on start states set a; the chosen pair is tested on set d, paired with G13.

    $PY experiments/g14_tli_practice.py --run [--backbone pi0_libero|pi05_libero]   # stages 1, 2 and the test, waiting on files
    $PY experiments/g14_tli_practice.py --read [--record]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
SEED, LAM = 13, 24
PLAN = os.path.join(R, "g4_plan.json")


def _arm(a, g):
    return f"tli@{a}.{g}.{LAM}"


def choose(stage, backbone):
    import g4_ood as G
    pl, pool, out = json.load(open(PLAN)), G.source_pool(), {}
    for k, v in pl["tasks"].items():
        oa, og = G.ours_sources(v, pl)
        if stage == 1:
            arms, pref = {a: _arm(a, og) for a in pool}, oa
        else:
            a1 = choose(1, backbone)[k]
            arms, pref = {g: _arm(a1, g) for g in pool}, og
        sc = {c: G._ld(f"g14_s{stage}_{k.split(':')[0]}", v, arm, backbone) for c, arm in arms.items()}
        out[k] = None if None in sc.values() else max(pool, key=lambda c: (sum(sc[c].values()), c == pref, -c))
    return out


def _wait(stage, backbone):
    while None in choose(stage, backbone).values():
        time.sleep(300)


def run(backbone):
    import g4_ood as G
    pl, pool = json.load(open(PLAN)), G.source_pool()
    for k, v in pl["tasks"].items():
        oa, og = G.ours_sources(v, pl)
        print(k, G._launch(f"g14_s1_{k.split(':')[0]}", k, v, [_arm(a, og) for a in pool], 5, "fresh_ood_a", backbone), flush=True)
    _wait(1, backbone)
    s1 = choose(1, backbone)
    for k, v in pl["tasks"].items():
        print(k, G._launch(f"g14_s2_{k.split(':')[0]}", k, v, [_arm(s1[k], g) for g in pool], 5, "fresh_ood_a", backbone), flush=True)
    _wait(2, backbone)
    s2 = choose(2, backbone)
    for k, v in pl["tasks"].items():
        print(k, G._launch(f"g14_main_{k.split(':')[0]}", k, v, [_arm(s1[k], s2[k])], 20, "fresh_ood_d", backbone), flush=True)
    print("G14_TEST_LAUNCHED", flush=True)


def read(backbone, record):
    import e1_read as E1
    import g4_ood as G
    import g13_tli_sources as G13
    pl = json.load(open(PLAN))
    short = backbone.split("_")[0]
    s1, s2 = choose(1, backbone), choose(2, backbone)
    x, missing, chosen = {}, [], {}
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        a13 = G13.arms(k, v, backbone)
        chosen[k] = _arm(s1[k], s2[k])
        z = dict(sel=G._ld(f"g14_main_{sh}", v, chosen[k], backbone),
                 **{a: G._ld(f"g13_main_{sh}", v, arm, backbone) for a, arm in a13.items()})
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    li = {k: tuple(G13.arms(k, v, backbone)["tli_li"].split("@")[1].split(".")[:2]) for k, v in pl["tasks"].items()}
    res = dict(backbone=backbone, missing=missing, chosen=chosen,
               same_as_li=sum((str(s1[k]), str(s2[k])) == li[k] for k in chosen),
               sel_vs_tli_li=C({k: {"x": z["sel"], "y": z["tli_li"]} for k, z in x.items()}),
               sel_vs_tli_g10=C({k: {"x": z["sel"], "y": z["tli_sel"]} for k, z in x.items()}),
               sel_vs_psel=C({k: {"x": z["sel"], "y": z["psel"]} for k, z in x.items()}),
               sel_vs_zs=C({k: {"x": z["sel"], "y": z["zs"]} for k, z in x.items()}),
               practice_episodes_per_task=2 * len(G.source_pool()) * 5,
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    res["sel_vs_tli_li"]["noninferior"] = bool(res["sel_vs_tli_li"]["diff"]["lo"] > -0.05)
    for n_ in ("sel_vs_tli_li", "sel_vs_tli_g10", "sel_vs_psel", "sel_vs_zs"):
        h = res[n_]
        print(f"[g14 {short}] {n_:14} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g14 {short}] same pair as Li {res['same_as_li']} | missing {missing}", flush=True)
    json.dump(res, open(os.path.join(R, f"g14_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g14_tli_practice", bench="libero_ood", tag=f"read_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g14_tli_practice.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi0_libero"
    if "--run" in sys.argv:
        run(bb)
    if "--read" in sys.argv:
        read(bb, "--record" in sys.argv)
