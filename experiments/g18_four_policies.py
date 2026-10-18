"""Plan G, read G18 (ledger m3_prereg tag g18_four_policies): the finished OOD system (iteration 12) on SmolVLA and X-VLA.
Practice chooses sources under the grasp switch (G10's stages) and under TLI (G14's stages, where TLI is ported), then the
operator per task (10 practice episodes each on set a, ties to the switch); test on set e with the policy alone, Li's pairs at
the grasp and Li's TLI.

    $PY experiments/g18_four_policies.py --run --backbone smolvla|xvla [--no-tli]
    $PY experiments/g18_four_policies.py --read [--record]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
SEED = 13
PLAN = os.path.join(R, "g4_plan.json")


def _wait(fn):
    while None in fn().values():
        time.sleep(300)


def tli_ready(backbone):
    import g4_ood as G
    pl = json.load(open(PLAN))
    need = set(G.source_pool()) | {t for v in pl["tasks"].values() for t in G.ours_sources(v, pl)}
    return all(os.path.exists(os.path.join(R, "tli", f"{backbone}_{t}.npy")) for t in need)


def pairs(backbone, use_tli):
    import g4_ood as G
    import g14_tli_practice as G14
    pl = json.load(open(PLAN))
    s1, s2 = G.g10_choose(1, backbone), G.g10_choose(2, backbone)
    out = {k: dict(psel=f"pfield@{s1[k]}.{s2[k]}") for k in pl["tasks"]}
    if use_tli:
        t1, t2 = G14.choose(1, backbone), G14.choose(2, backbone)
        for k in pl["tasks"]:
            out[k]["tsel"] = G14._arm(t1[k], t2[k])
    return out


def arms(k, v, backbone, use_tli, pr):
    import g4_ood as G
    import g12_tli as T
    pl = json.load(open(PLAN))
    a = dict(zs="zs", li_grasp=G.li_arms(v, pl)["li_grasp"], **pr[k])
    if use_tli:
        a["tli_li"] = T.arm(v)
    return a


def run(backbone, use_tli):
    import g4_ood as G
    import g14_tli_practice as G14
    pl, pool = json.load(open(PLAN)), G.source_pool()
    G.g10_launch(1, backbone)
    _wait(lambda: G.g10_choose(1, backbone))
    G.g10_launch(2, backbone)
    _wait(lambda: G.g10_choose(2, backbone))
    print("[g18] switch stages done", flush=True)
    if use_tli:
        while not tli_ready(backbone):
            time.sleep(300)
        for k, v in pl["tasks"].items():
            oa, og = G.ours_sources(v, pl)
            G._launch(f"g14_s1_{k.split(':')[0]}", k, v, [G14._arm(a, og) for a in pool], 5, "fresh_ood_a", backbone)
        _wait(lambda: G14.choose(1, backbone))
        s1 = G14.choose(1, backbone)
        for k, v in pl["tasks"].items():
            G._launch(f"g14_s2_{k.split(':')[0]}", k, v, [G14._arm(s1[k], g) for g in pool], 5, "fresh_ood_a", backbone)
        _wait(lambda: G14.choose(2, backbone))
        print("[g18] TLI stages done", flush=True)
    pr = pairs(backbone, use_tli)
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        G._launch(f"g10_acc_{sh}", k, v, [pr[k]["psel"]], 10, "fresh_ood_a", backbone)
        if use_tli:
            G._launch(f"g15_acc_{sh}", k, v, [pr[k]["tsel"]], 10, "fresh_ood_a", backbone)
        G._launch(f"g18_main_{sh}", k, v, list(dict.fromkeys(arms(k, v, backbone, use_tli, pr).values())), 20, "fresh_ood_e", backbone)
    json.dump(dict(backbone=backbone, tli=use_tli, pairs=pr), open(os.path.join(R, f"g18_state_{backbone}.json"), "w"), indent=1)
    print("[g18] G18_TEST_LAUNCHED", flush=True)


def read(record):
    import e1_read as E1
    import g4_ood as G
    pl = json.load(open(PLAN))
    C = E1.compare
    for backbone in ("smolvla", "xvla"):
        f = os.path.join(R, f"g18_state_{backbone}.json")
        if not os.path.exists(f):
            continue
        st = json.load(open(f))
        use_tli, pr = st["tli"], st["pairs"]
        x, missing, choice = {}, [], {}
        for k, v in pl["tasks"].items():
            sh = k.split(":")[0]
            a = arms(k, v, backbone, use_tli, pr)
            z = {n: G._ld(f"g18_main_{sh}", v, arm, backbone) for n, arm in a.items()}
            pa = G._ld(f"g10_acc_{sh}", v, a["psel"], backbone)
            ta = G._ld(f"g15_acc_{sh}", v, a["tsel"], backbone) if use_tli else None
            if None in z.values() or pa is None or (use_tli and ta is None):
                missing.append(k)
                continue
            choice[k] = "tsel" if use_tli and sum(ta.values()) > sum(pa.values()) else "psel"
            z["chosen"] = z[choice[k]]
            x[k] = z
        ref = "tli_li" if use_tli else "li_grasp"
        res = dict(backbone=backbone, tli=use_tli, missing=missing, tasks=len(x), choice=choice, reference=ref,
                   n_tsel=sum(c == "tsel" for c in choice.values()),
                   chosen_vs_zs=C({k: {"x": z["chosen"], "y": z["zs"]} for k, z in x.items()}),
                   chosen_vs_ref=C({k: {"x": z["chosen"], "y": z[ref]} for k, z in x.items()}),
                   chosen_vs_li_grasp=C({k: {"x": z["chosen"], "y": z["li_grasp"]} for k, z in x.items()}),
                   per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
        if x:
            res["chosen_vs_ref"]["noninferior"] = bool(res["chosen_vs_ref"]["diff"]["lo"] > -0.05)
            for n_ in ("chosen_vs_zs", "chosen_vs_ref", "chosen_vs_li_grasp"):
                h = res[n_]
                print(f"[g18 {backbone}] {n_:18} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                      f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
        print(f"[g18 {backbone}] TLI {use_tli}, chosen TLI on {res['n_tsel']} | missing {missing}", flush=True)
        json.dump(res, open(os.path.join(R, f"g18_read_{backbone}.json"), "w"), indent=1)
        if record and not missing:
            from lifelong_ood import ledger
            print(ledger.record("g18_four_policies", bench="libero_ood", tag=f"read_{backbone}", seed=SEED, metrics=res,
                                bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                                script="experiments/g18_four_policies.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "smolvla"
    if "--run" in sys.argv:
        run(bb, "--no-tli" not in sys.argv)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
