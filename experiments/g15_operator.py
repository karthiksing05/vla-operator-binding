"""Plan G, read G15 (iteration 12; ledger m3_prereg tag g15_operator): practice chooses the operator for held concepts per task.
Candidates: the grasp switch with G10's practice-chosen pair (psel) and TLI with G14's practice-chosen pair (tsel), each scored by
10 practice episodes on start states set a (psel: G10's g10_acc; tsel: g15_acc, run here); the higher count wins, ties to psel.
Tested on set e with zs, Li's TLI (tli_li), psel and tsel, paired.

    $PY experiments/g15_operator.py --accept [--backbone ...]   # tsel's 10 practice episodes on set a
    $PY experiments/g15_operator.py --test [--backbone ...]     # set e: the four arms
    $PY experiments/g15_operator.py --read [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
SEED = 13
PLAN = os.path.join(R, "g4_plan.json")


def arms(k, v, backbone):
    import g12_tli as T
    import g14_tli_practice as G14
    g10 = json.load(open(os.path.join(R, f"g10_read_{backbone.split('_')[0]}.json")))
    s1, s2 = G14.choose(1, backbone)[k], G14.choose(2, backbone)[k]
    return dict(zs="zs", tli_li=T.arm(v), psel=g10["chosen"][k], tsel=G14._arm(s1, s2))


def accept(backbone):
    import g4_ood as G
    pl = json.load(open(PLAN))
    for k, v in pl["tasks"].items():
        print(k, G._launch(f"g15_acc_{k.split(':')[0]}", k, v, [arms(k, v, backbone)["tsel"]], 10, "fresh_ood_a", backbone), flush=True)


def test(backbone):
    import g4_ood as G
    pl = json.load(open(PLAN))
    for k, v in pl["tasks"].items():
        print(k, G._launch(f"g15_main_{k.split(':')[0]}", k, v, list(dict.fromkeys(arms(k, v, backbone).values())), 20, "fresh_ood_e",
                           backbone), flush=True)


def choice(backbone):
    """-> {task: 'psel' | 'tsel' | None (practice incomplete)}, and the practice counts."""
    import g4_ood as G
    pl = json.load(open(PLAN))
    out, cnt = {}, {}
    for k, v in pl["tasks"].items():
        a, sh = arms(k, v, backbone), k.split(":")[0]
        p, t = G._ld(f"g10_acc_{sh}", v, a["psel"], backbone), G._ld(f"g15_acc_{sh}", v, a["tsel"], backbone)
        if p is None or t is None:
            out[k] = None
            continue
        cnt[k] = (int(sum(p.values())), int(sum(t.values())))
        out[k] = "tsel" if cnt[k][1] > cnt[k][0] else "psel"
    return out, cnt


def read(backbone, record):
    import e1_read as E1
    import g4_ood as G
    pl = json.load(open(PLAN))
    short = backbone.split("_")[0]
    ch, cnt = choice(backbone)
    x, missing = {}, []
    for k, v in pl["tasks"].items():
        z = {a: G._ld(f"g15_main_{k.split(':')[0]}", v, arm, backbone) for a, arm in arms(k, v, backbone).items()}
        if None in z.values() or ch[k] is None:
            missing.append(k)
            continue
        z["chosen"] = z[ch[k]]
        x[k] = z
    C = E1.compare
    res = dict(backbone=backbone, missing=missing, choice=ch, practice=cnt,
               n_tsel=sum(c == "tsel" for c in ch.values()), n_psel=sum(c == "psel" for c in ch.values()),
               **{f"chosen_vs_{a}": C({k: {"x": z["chosen"], "y": z[a]} for k, z in x.items()}) for a in ("tli_li", "zs", "psel", "tsel")},
               tli_li_vs_zs=C({k: {"x": z["tli_li"], "y": z["zs"]} for k, z in x.items()}),
               by_suite={s: {a: int(sum(sum(z[a].values()) for k, z in x.items() if pl["tasks"][k]["suite"] == s)) for a in
                             ("zs", "tli_li", "psel", "tsel", "chosen")} for s in ("libero_goal_ood", "libero_spatial_ood")},
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    res["chosen_vs_tli_li"]["noninferior"] = bool(res["chosen_vs_tli_li"]["diff"]["lo"] > -0.05)
    for n_ in ("chosen_vs_tli_li", "chosen_vs_zs", "chosen_vs_psel", "chosen_vs_tsel"):
        h = res[n_]
        print(f"[g15 {short}] {n_:16} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g15 {short}] TLI chosen on {res['n_tsel']}, switch on {res['n_psel']} | by suite {res['by_suite']} | missing {missing}",
          flush=True)
    json.dump(res, open(os.path.join(R, f"g15_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g15_operator", bench="libero_ood", tag=f"read_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g15_operator.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi0_libero"
    if "--accept" in sys.argv:
        accept(bb)
    if "--test" in sys.argv:
        test(bb)
    if "--read" in sys.argv:
        read(bb, "--record" in sys.argv)
