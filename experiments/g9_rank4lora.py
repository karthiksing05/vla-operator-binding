"""Plan G, read G9: a RANK-MATCHED per-task adapter (novelty audit, docs/NOVELTY.md section 3.3 item 2).  E3's storage comparison
set rank-4 shared prototypes against rank-16 per-task LoRAs (TAIL-style), so part of the 1/5 is rank.  Here each of E3's 29 tasks
with a demonstration gets a rank-4 one-demo LoRA (r3_fewshot.py --rank 4, its own scene's demonstration), given the same
acceptance test (stored states 0-9, seed 13), tested on E3's fresh set b at seed 13, paired with E3's library.

    $PY experiments/g9_rank4lora.py --fit | --accept --test | --read [--record]
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED, TAG, ARM = 13, "_r4", "lorat@_r4"
E3PLAN = json.load(open(os.path.join(R, "e3_plan.json")))
TASKS = [t for t, v in E3PLAN.items() if v["lora"]]


def fit():
    for t in TASKS:
        if os.path.exists(os.path.join(R, f"r3_lora_{t}{TAG}.pt")):
            continue
        out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                              "--account=overcap", "--qos=short",
                              f"--export=ALL,ENV=openpi,SCRIPT=experiments/r3_fewshot.py --task {t} --rank 4 --tag {TAG}",
                              "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
        print(t, out.stdout.strip() or out.stderr.strip(), flush=True)


def launch(exp, trials, fresh):
    for t in TASKS:
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ARM, "--experiment", exp, "--first", "0", "--trials",
               str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "1"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-80:], flush=True)


def lora_bytes(tag):
    from lifelong_ood import ledger
    rows = [r for r in ledger.load("r3_fewshot") if str(r.get("command", "")).endswith(f"--tag {tag}") or
            f"--tag {tag} " in str(r.get("command", ""))]
    return rows[-1]["bytes"]["experts"] if rows else None


def read(record):
    import e1_read as E1
    import e3_read as E3
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    x, missing, kept = {}, [], {}
    for t in TASKS:
        v = E3PLAN[t]
        za, zx = E3.load("e3_accept", t, "zs"), E3.load("e3_main", t, "zs")
        la, lx = E3.load("g9_accept", t, ARM), E3.load("g9_main", t, ARM)
        if None in (za, zx, la, lx):
            missing.append(t)
            continue
        kept[t] = "lora" if sum(la.values()) > sum(za.values()) else "zs"
        x[t] = dict(lib=E3.load("e3_main", t, E3.arms(v, True)[e3["kept"][t]]), r4_acc=lx if kept[t] == "lora" else zx, r4=lx,
                    r16_acc=E3.load("e3_main", t, E3.arms(v, True)[e3["kept_lora"][t]]))
    C = E1.compare
    h1 = C({t: {"x": z["lib"], "y": z["r4_acc"]} for t, z in x.items()})
    h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
    res = dict(missing=missing, tasks=len(x), H1=h1, kept_lora=sum(v == "lora" for v in kept.values()),
               r4_acc_vs_r16_acc=C({t: {"x": z["r4_acc"], "y": z["r16_acc"]} for t, z in x.items()}),
               per_task={t: {k: int(sum(v.values())) for k, v in z.items()} for t, z in x.items()},
               bytes=dict(library=e3["bytes"]["library"], prototypes=e3["bytes"]["prototypes"], r4_each=lora_bytes(TAG),
                          r4_total=(lora_bytes(TAG) or 0) * len(TASKS), loras=len(TASKS)))
    for k in ("H1", "r4_acc_vs_r16_acc"):
        h = res[k]
        print(f"[g9] {k:18} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, diff "
              f"{h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print("[g9] bytes", res["bytes"], "| missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g9_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g9_rank4lora", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g9_rank4lora.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("g9_accept", 10, False)
    if "--test" in sys.argv:
        launch("g9_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
