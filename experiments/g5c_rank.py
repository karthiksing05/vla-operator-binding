"""Plan G, read G5c: does prototype CAPACITY close the gap to a per-task adapter?  E3's hierarchical library (the same 22 concepts,
the same demonstrations, the same fit settings, read from each prototype's c0_fit ledger row) refit at rank 8 and rank 16 (E3's is
rank 4; the per-task LoRA is rank 16).  Per rank: acceptance on stored states 0-9 (noise seed 13, paired with E3's zero-shot),
test on E3's fresh set b (20 per task), the kept arm compared with E3's per-task LoRA with acceptance and with E3's rank-4 kept
arm, paired episode by episode.

    $PY experiments/g5c_rank.py --fit | --accept --test | --read [--record]
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
SEED, RANKS = 13, (8, 16)
E3PLAN = json.load(open(os.path.join(R, "e3_plan.json")))


def names():
    return sorted({v["object_proto"] for v in E3PLAN.values()} | {v["goal_proto"] for v in E3PLAN.values()})


def fit():
    for n in names():
        cmd = json.load(open(sorted(glob.glob(os.path.join(ROOT, "ledger", "c0_fit", f"*joint_{n}__*.json")))[-1]))["command"]
        assert f"--name {n} --rank 4" in cmd, cmd
        for r in RANKS:
            if os.path.exists(os.path.join(R, f"prim_{n}R{r}_0.pt")):
                continue
            script = cmd.replace(f"--name {n} --rank 4", f"--name {n}R{r} --rank {r}")
            out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                                  "--account=overcap", "--qos=short", f"--export=ALL,ENV=openpi,SCRIPT={script}",
                                  "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
            print(n, r, out.stdout.strip() or out.stderr.strip(), flush=True)


def arm(v, r):
    return f"fac@{v['object_proto']}R{r}.{v['goal_proto']}R{r}"


def launch(exp, trials, fresh):
    for t, v in E3PLAN.items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arm(v, r) for r in RANKS), "--experiment", exp,
               "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def read(record):
    import e1_read as E1
    import e3_read as E3
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    C = E1.compare
    res, missing = {}, []
    for r in RANKS:
        test, kept = {}, {}
        for t, v in E3PLAN.items():
            za, zx = E3.load("e3_accept", t, "zs"), E3.load("e3_main", t, "zs")
            fa, fx = E3.load("g5c_accept", t, arm(v, r)), E3.load("g5c_main", t, arm(v, r))
            if None in (za, zx, fa, fx):
                missing.append(f"{t}:r{r}")
                continue
            kept[t] = "fac" if sum(fa.values()) > sum(za.values()) else "zs"
            z = dict(zs=zx, fac=fx, kept=fx if kept[t] == "fac" else zx,
                     r4=E3.load("e3_main", t, E3.arms(v, True)[e3["kept"][t]]))
            if t in e3["kept_lora"]:
                z["lora_acc"] = E3.load("e3_main", t, E3.arms(v, True)[e3["kept_lora"][t]])
            test[t] = z
        h2 = C({t: {"x": z["kept"], "y": z["lora_acc"]} for t, z in test.items() if "lora_acc" in z})
        h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
        res[f"r{r}"] = dict(kept_vs_zs=C({t: {"x": z["kept"], "y": z["zs"]} for t, z in test.items()}), vs_lora_acc=h2,
                            vs_rank4=C({t: {"x": z["kept"], "y": z["r4"]} for t, z in test.items()}),
                            fac_vs_zs=C({t: {"x": z["fac"], "y": z["zs"]} for t, z in test.items()}),
                            kept={k: sum(x == k for x in kept.values()) for k in ("zs", "fac")},
                            per_task={t: {k: int(sum(x.values())) for k, x in z.items()} for t, z in test.items()})
        for k in ("kept_vs_zs", "vs_lora_acc", "vs_rank4", "fac_vs_zs"):
            h = res[f"r{r}"][k]
            print(f"[g5c] rank {r:2} {k:12} {h['x']} vs {h['y']} of {h['n']}, diff {h['diff']['diff']:+.3f} "
                  f"[{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}], sign p {h['sign_p']:.3g}", flush=True)
    res["missing"] = missing
    json.dump(res, open(os.path.join(R, "g5c_read.json"), "w"), indent=1)
    if missing:
        print("[g5c] missing", missing, flush=True)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g5c_rank", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(ranks=RANKS),
                            script="experiments/g5c_rank.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("g5c_accept", 10, False)
    if "--test" in sys.argv:
        launch("g5c_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
