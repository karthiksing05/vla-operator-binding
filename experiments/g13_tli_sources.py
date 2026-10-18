"""Plan G, read G13 (iteration 10; ledger m3_prereg tag g13_tli_sources): TLI as the operator for held concepts with
practice-chosen sources and no hand labels.  Arms per policy on start states set d: the policy alone (zs), TLI with Li's pairs
and lambda (tli_li), TLI with G10's practice-chosen pair and lambda 24 (tli_sel), and that pair switched at the grasp (psel).
Descriptive: tli_li on pi0.5 on set c (paired with G10).

    $PY experiments/g13_tli_sources.py --launch [--backbone pi0_libero|pi05_libero] | --launch-c | --read [--record]
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED, LAM = 13, 24


def arms(k, v, backbone):
    import g12_tli as T
    g10 = json.load(open(os.path.join(R, f"g10_read_{backbone.split('_')[0]}.json")))
    psel = g10["chosen"][k]
    s1, s2 = psel.split("@")[1].split(".")
    return dict(zs="zs", tli_li=T.arm(v), tli_sel=f"tli@{s1}.{s2}.{LAM}", psel=psel)


def _launch(exp, k, v, arm_list, backbone, fresh):
    cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(dict.fromkeys(arm_list)), "--experiment",
           f"{exp}_{k.split(':')[0]}", "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind", "track", "--partition",
           "overcap", "--per", "2", "--suite", v["suite"], "--fresh", "--fresh-dir", fresh]
    cmd += ["--backbone", backbone] if backbone != "pi05_libero" else []
    print(k, arm_list, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def launch(backbone):
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    for k, v in pl["tasks"].items():
        _launch("g13_main", k, v, list(arms(k, v, backbone).values()), backbone, "fresh_ood_d")


def launch_c():
    import g12_tli as T
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    for k, v in pl["tasks"].items():
        _launch("g13_c", k, v, [T.arm(v)], "pi05_libero", "fresh_ood_c")


def read(backbone, record):
    import e1_read as E1
    import g4_ood as G
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    short = backbone.split("_")[0]
    x, missing = {}, []
    for k, v in pl["tasks"].items():
        z = {a: G._ld(f"g13_main_{k.split(':')[0]}", v, arm, backbone) for a, arm in arms(k, v, backbone).items()}
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    res = dict(backbone=backbone, missing=missing, tasks=len(x), arms={k: arms(k, v, backbone) for k, v in pl["tasks"].items()},
               same_pair_as_li=sum(a["tli_sel"].split(".")[:2] == a["tli_li"].split(".")[:2] for a in
                                   (arms(k, v, backbone) for k, v in pl["tasks"].items())),
               sel_vs_li=C({k: {"x": z["tli_sel"], "y": z["tli_li"]} for k, z in x.items()}),
               sel_vs_psel=C({k: {"x": z["tli_sel"], "y": z["psel"]} for k, z in x.items()}),
               psel_vs_li=C({k: {"x": z["psel"], "y": z["tli_li"]} for k, z in x.items()}),
               **{f"{a}_vs_zs": C({k: {"x": z[a], "y": z["zs"]} for k, z in x.items()}) for a in ("tli_li", "tli_sel", "psel")},
               by_suite={s: {a: int(sum(sum(z[a].values()) for k, z in x.items() if pl["tasks"][k]["suite"] == s)) for a in
                             ("zs", "tli_li", "tli_sel", "psel")} for s in ("libero_goal_ood", "libero_spatial_ood")},
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    res["sel_vs_li"]["noninferior"] = bool(res["sel_vs_li"]["diff"]["lo"] > -0.05)
    if backbone == "pi05_libero":                                       # descriptive: Li's TLI on pi0.5, set c, against G10
        import g12_tli as T
        g10 = json.load(open(os.path.join(R, "g10_read_pi05.json")))
        xc = {}
        for k, v in pl["tasks"].items():
            z = dict(tli=G._ld(f"g13_c_{k.split(':')[0]}", v, T.arm(v), backbone),
                     sel=G._ld(f"g10_main_{k.split(':')[0]}", v, g10["chosen"][k], backbone),
                     zs=G._ld(f"g4b_main_{k.split(':')[0]}", v, "zs", backbone))
            if None not in z.values():
                xc[k] = z
        res["set_c"] = dict(tasks=len(xc), sel_vs_tli=C({k: {"x": z["sel"], "y": z["tli"]} for k, z in xc.items()}),
                            tli_vs_zs=C({k: {"x": z["tli"], "y": z["zs"]} for k, z in xc.items()}))
    for n_ in ("sel_vs_li", "sel_vs_psel", "psel_vs_li", "tli_li_vs_zs", "tli_sel_vs_zs", "psel_vs_zs"):
        h = res[n_]
        print(f"[g13 {short}] {n_:14} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    if "set_c" in res:
        for n_ in ("sel_vs_tli", "tli_vs_zs"):
            h = res["set_c"][n_]
            print(f"[g13 {short} set c] {n_} {h['x']} vs {h['y']} of {h['n']}, sign p {h['sign_p']:.3g}", flush=True)
    print(f"[g13 {short}] by suite {res['by_suite']} | same pair as Li {res['same_pair_as_li']} | missing {missing}", flush=True)
    json.dump(res, open(os.path.join(R, f"g13_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g13_tli_sources", bench="libero_ood", tag=f"read_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g13_tli_sources.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi0_libero"
    if "--launch" in sys.argv:
        launch(bb)
    if "--launch-c" in sys.argv:
        launch_c()
    if "--read" in sys.argv:
        read(bb, "--record" in sys.argv)
