"""Plan G, read G3 (docs/PLAN_GENERALIZE.md; ledger m3_prereg tag g3_crossmodel): the hierarchical factored library on ANOTHER
policy.  E3's library (the same concepts, the same demonstrations of each concept) refit on SmolVLA (LeRobot's smolvla_libero,
trained on the same 40 LIBERO tasks as pi0.5), against SmolVLA's own per-task LoRA, on the stream's 30 tasks, E3's stored practice
states and fresh start states (set b) at E3's noise seed -- so each episode pairs with pi0.5's on the same start state.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/g3_crossmodel.py --plan --fit
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g3_crossmodel.py --accept --test
    ~/flash/miniconda3/envs/chunks/bin/python experiments/g3_crossmodel.py --read [--record]
"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec_ = importlib.util.spec_from_file_location("e1", os.path.join(ROOT, "experiments", "e1_read.py"))
E1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(E1)

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "g3_plan.json")
SEED, RANK, BK, LTAG = 13, 4, "smolvla", "_sv"
ORDER = ("zs", "hfac")


def plan():
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    names, out = {}, {}
    for t, v in e3.items():
        for proto, tasks, key, flag in ((v["object_proto"], v["object_tasks"], v["object_key"], "--approach"),
                                        (v["goal_proto"], v["goal_tasks"], v["goal_key"], "--goal")):
            names.setdefault(proto, dict(name="V" + proto, tasks=tasks, key=key, flag=flag))
        out[t] = dict(object_proto="V" + v["object_proto"], goal_proto="V" + v["goal_proto"], lora=bool(v["lora"]))
    json.dump(dict(tasks=out, protos=names), open(PLAN, "w"), indent=1)
    print(f"[g3] {len(out)} tasks, {len(names)} prototypes on {BK}", flush=True)


def sb(script):
    out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                          "--account=overcap", "--qos=short", f"--export=ALL,ENV=smolvla,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def fit():
    pl = json.load(open(PLAN))
    for p in pl["protos"].values():
        if not os.path.exists(os.path.join(R, f"prim_{p['name']}_0.pt")):
            print(p["name"], sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, p['tasks']))} --key {p['key'].replace(' ', '_')} "
                                f"--name {p['name']} --rank {RANK} {p['flag']} --backbone {BK}"), flush=True)
    for t, v in pl["tasks"].items():
        if v["lora"] and not os.path.exists(os.path.join(R, f"r3_lora_{t}{LTAG}.pt")):
            print("lora", t, sb(f"experiments/r3_fewshot.py --task {t} --tag {LTAG} --backbone {BK}"), flush=True)


def arms(v):
    a = {"zs": "zs", "hfac": f"fac@{v['object_proto']}.{v['goal_proto']}"}
    if v["lora"]:
        a["lora"] = f"lorat@{LTAG}"
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}_{BK}"


def launch(exp, trials, fresh):
    for t, v in json.load(open(PLAN))["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v).values()), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2", "--backbone", BK]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def load(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    C = E1.compare
    acc, test, kept, kept_lora, missing = {}, {}, {}, {}, []
    for t, v in json.load(open(PLAN))["tasks"].items():
        a = arms(v)
        s = {k: load("g3_accept", t, x) for k, x in a.items()}
        x_ = {k: load("g3_main", t, x) for k, x in a.items()}
        if None in s.values() or None in x_.values():
            missing.append(t)
            continue
        acc[t] = {k: int(sum(z.values())) for k, z in s.items()}
        kept[t] = max(ORDER, key=lambda k: (acc[t][k], -ORDER.index(k)))
        if "lora" in a:
            kept_lora[t] = "lora" if acc[t]["lora"] > acc[t]["zs"] else "zs"
        test[t] = x_
    h1 = C({t: {"x": x[kept[t]], "y": x["zs"]} for t, x in test.items()})
    h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05)
    h2 = C({t: {"x": x[kept[t]], "y": x[kept_lora[t]]} for t, x in test.items() if t in kept_lora})
    h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
    desc = {"hfac_vs_zs": C({t: {"x": x["hfac"], "y": x["zs"]} for t, x in test.items()}),
            "hfac_vs_lora": C({t: {"x": x["hfac"], "y": x["lora"]} for t, x in test.items() if "lora" in x})}
    res = dict(missing=missing, acceptance=acc, kept=kept, kept_lora=kept_lora, hypotheses={"H1": h1, "H2": h2}, descriptive=desc,
               per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()})
    for k, v in list(res["hypotheses"].items()) + list(desc.items()):
        print(f"[g3] {k}: {v['x']} vs {v['y']} of {v['n']} ({v['tasks']} tasks), diff {v['diff']['diff']:+.3f} "
              f"[{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}], sign p {v['sign_p']:.3g}", flush=True)
    json.dump(res, open(os.path.join(R, "g3_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g3_crossmodel", bench="libero_90", tag="read_smolvla", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=BK),
                            script="experiments/g3_crossmodel.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("g3_accept", 10, False)
    if "--test" in sys.argv:
        launch("g3_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
