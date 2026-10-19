"""Plan G, read G19 (ledger m3_prereg tag g19_leaves_policies): G8's task leaves on pi0, SmolVLA and X-VLA.  G3's refit library,
per-task LoRA and acceptance episodes (g3_accept) per policy; the leaves rule picks per task among the policy, the composition and
the LoRA; test on start states set d.

    $PY experiments/g19_leaves_policies.py --test [--backbone pi0_libero|smolvla|xvla]
    $PY experiments/g19_leaves_policies.py --read [--record]
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
ORDER = ("zs", "hfac", "lora")
LTAG = {"pi0": "_p0", "smolvla": "_sv", "xvla": "_xv"}


def short(bk):
    return bk.split("_")[0]


def plan(bk):
    s = short(bk)
    return json.load(open(os.path.join(R, "g3_plan.json" if s == "smolvla" else f"g3_plan_{s}.json")))["tasks"]


def arms(v, bk):
    a = {"zs": "zs", "hfac": f"fac@{v['object_proto']}.{v['goal_proto']}"}
    if v["lora"]:
        a["lora"] = f"lorat@{LTAG[short(bk)]}"
    return a


def load(exp, t, a, bk):
    f = os.path.join(R, f"{exp}_{t}_{a.replace('@', '_')}_seed{SEED}_{short(bk)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def launch(bk):
    for t, v in plan(bk).items():
        if not v["lora"]:
            continue
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v, bk).values()), "--experiment", "g19_main",
               "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2",
               "--backbone", bk, "--fresh", "--fresh-dir", "fresh_states_d"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-100:], flush=True)


def read(record):
    import e1_read as E1
    C = E1.compare
    for bk in ("pi0_libero", "smolvla", "xvla"):
        test, kept, missing = {}, {}, []
        for t, v in plan(bk).items():
            if not v["lora"]:
                continue
            a = arms(v, bk)
            s = {k: load("g3_accept", t, a[k], bk) for k in ORDER}
            x = {k: load("g19_main", t, a[k], bk) for k in ORDER}
            if None in s.values() or None in x.values():
                missing.append(t)
                continue
            acc = {k: sum(z.values()) for k, z in s.items()}
            kept[t] = max(ORDER, key=lambda k: (acc[k], -ORDER.index(k)))
            kl = "lora" if acc["lora"] > acc["zs"] else "zs"
            lib = max(("zs", "hfac"), key=lambda k: (acc[k], -ORDER.index(k)))
            test[t] = dict(leaves=x[kept[t]], lora_acc=x[kl], library=x[lib], zs=x["zs"])
        if not test:
            print(f"[g19 {short(bk)}] nothing yet; missing {len(missing)}", flush=True)
            continue
        h1 = C({t: {"x": z["leaves"], "y": z["lora_acc"]} for t, z in test.items()})
        h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
        res = dict(backbone=bk, missing=missing, tasks=len(test), H1=h1,
                   H2=C({t: {"x": z["leaves"], "y": z["zs"]} for t, z in test.items()}),
                   library_vs_lora_acc=C({t: {"x": z["library"], "y": z["lora_acc"]} for t, z in test.items()}),
                   kept={k: sum(c == k for c in kept.values()) for k in ORDER},
                   per_task={t: {k: int(sum(v.values())) for k, v in z.items()} | {"kept": kept[t]} for t, z in test.items()})
        for k in ("H1", "H2", "library_vs_lora_acc"):
            h = res[k]
            print(f"[g19 {short(bk)}] {k:20} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                  f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
        print(f"[g19 {short(bk)}] kept {res['kept']} | missing {missing}", flush=True)
        json.dump(res, open(os.path.join(R, f"g19_read_{short(bk)}.json"), "w"), indent=1)
        if record and not missing:
            from lifelong_ood import ledger
            print(ledger.record("g19_leaves_policies", bench="libero_90", tag=f"read_{short(bk)}", seed=SEED, metrics=res,
                                bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=bk),
                                script="experiments/g19_leaves_policies.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--test" in sys.argv:
        launch(sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "smolvla")
    if "--read" in sys.argv:
        read("--record" in sys.argv)
