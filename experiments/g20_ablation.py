"""Plan G, read G20 (ledger m3_prereg tag g20_ablation): the ablation on pi0.5 (experiments/prereg/g20_ablation.py).
LIBERO-90 on set d, paired with G8; Li's OOD suites on set e, paired with G15.

    $PY experiments/g20_ablation.py --launch | --read [--record]
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
SEED = 13
PLAN = os.path.join(R, "g4_plan.json")


def l90_arms(t):
    """-> {name: arm} of the new composition ablations for a G8 task."""
    e3, e5 = json.load(open(os.path.join(R, "e3_plan.json"))), json.load(open(os.path.join(R, "e5_plan.json")))["tasks"]
    if t in e3:
        v = e3[t]
        o, g, ty = v["object_proto"], v["goal_proto"], (v["type_proto"] if v["instance_key"] else None)
    else:
        v = e5[t]
        o, g, ty = v["object_proto"], v["goal"], None
    a = dict(obj_only=f"fac@{o}.x", goal_only=f"fac@x.{g}", ungated=f"ungated@{o}.{g}")
    if ty and ty != o:
        a["type"] = f"fac@{ty}.{g}"
    return a


def ood_arms(k, v):
    import g4_ood as G
    pl = json.load(open(PLAN))
    psel = json.load(open(os.path.join(R, "g10_read_pi05.json")))["chosen"][k]
    a, g = psel.split("@")[1].split(".")
    q = G.li_arms(v, pl)["li_fixed"].rsplit(".q", 1)[1]
    return dict(word=G.pfield(v, pl), fixed=f"{psel}.q{q}", obj_src_only=f"pfield@{a}.x", goal_src_only=f"pfield@x.{g}")


def launch():
    import g4_ood as G
    import g8_leaves as G8
    for t in G8.tasks():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(l90_arms(t).values()), "--experiment", "g20_main",
               "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2",
               "--fresh", "--fresh-dir", "fresh_states_d"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-80:], flush=True)
    for k, v in json.load(open(PLAN))["tasks"].items():
        print(k, G._launch(f"g20_ood_{k.split(':')[0]}", k, v, list(dict.fromkeys(ood_arms(k, v).values())), 20, "fresh_ood_e",
                           "pi05_libero"), flush=True)


def _l90(t, arm):
    f = os.path.join(R, f"g20_main_{t}_{arm.replace('@', '_')}_seed{SEED}.json")
    return ({int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}
            if os.path.exists(f) else None)


def read(record):
    import e1_read as E1
    import g4_ood as G
    import g8_leaves as G8
    import g15_operator as G15
    C = E1.compare
    # LIBERO-90
    sysrows, comp, missing = {}, {}, []
    for t, (mod, arms, acc_exp) in G8.tasks().items():
        s = {k: mod.load(acc_exp, t, arms[k]) for k in G8.ORDER}
        x = {k: G8.load_test(t, arms[k]) for k in G8.ORDER}
        new = {n: _l90(t, a) for n, a in l90_arms(t).items()}
        if None in s.values() or None in x.values() or None in new.values():
            missing.append(t)
            continue
        acc = {k: sum(z.values()) for k, z in s.items()}
        kept = max(G8.ORDER, key=lambda k: (acc[k], -G8.ORDER.index(k)))
        lib = max(("zs", "hfac"), key=lambda k: (acc[k], -G8.ORDER.index(k)))
        kl = "lora" if acc["lora"] > acc["zs"] else "zs"
        sysrows[t] = dict(full=x[kept], no_leaves=x[lib], no_practice=x["hfac"], no_composition=x[kl], policy=x["zs"])
        comp[t] = dict(composition=x["hfac"], **new)
    l90 = dict(tasks=len(sysrows), missing=missing,
               system={n: C({t: {"x": z["full"], "y": z[n]} for t, z in sysrows.items()}) for n in
                       ("no_leaves", "no_practice", "no_composition", "policy")},
               composition={n: C({t: {"x": z["composition"], "y": z[n]} for t, z in comp.items() if n in z}) for n in
                            ("obj_only", "goal_only", "ungated", "type")})
    # OOD
    pl = json.load(open(PLAN))
    g15 = json.load(open(os.path.join(R, "g15_read_pi05.json")))
    rows, omissing = {}, []
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        a15 = G15.arms(k, v, "pi05_libero")
        z = {n: G._ld(f"g15_main_{sh}", v, a15[n], "pi05_libero") for n in ("zs", "psel", "tsel")}
        z["full"] = G._ld(f"g15_main_{sh}", v, a15[g15["choice"][k]], "pi05_libero")
        z.update({n: G._ld(f"g20_ood_{sh}", v, a, "pi05_libero") for n, a in ood_arms(k, v).items()})
        if None in z.values():
            omissing.append(k)
            continue
        rows[k] = z
    ood = dict(tasks=len(rows), missing=omissing,
               system={n: C({k: {"x": z["full"], "y": z[n]} for k, z in rows.items()}) for n in
                       ("psel", "tsel", "word", "fixed", "obj_src_only", "goal_src_only", "zs")})
    res = dict(libero_90=l90, ood=ood)
    for side in ("libero_90", "ood"):
        for grp, d in res[side].items():
            if not isinstance(d, dict) or grp == "missing":
                continue
            for n, h in d.items():
                print(f"[g20 {side} {grp}] {n:15} {h['x']} vs {h['y']} of {h['n']}, sign p {h['sign_p']:.3g}, diff "
                      f"{h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g20] missing libero_90 {len(missing)}, ood {len(omissing)}", flush=True)
    json.dump(res, open(os.path.join(R, "g20_read.json"), "w"), indent=1)
    if record and not missing and not omissing:
        from lifelong_ood import ledger
        print(ledger.record("g20_ablation", bench="libero", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g20_ablation.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
