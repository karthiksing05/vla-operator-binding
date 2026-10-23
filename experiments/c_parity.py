"""P1 parity read for pi05_l90c (docs/L90C.md): per-task success on its TRAINING cells against pi05_libero's level on its own 40
fine-tuning tasks, 20 episodes per task (LIBERO init states 0-19), through the repo's rollout harness (experiments/roll_pack.sbatch
-> roll.sh -> serve_guided.py MODE=ours -> libero_client.py -> roll_record.py), two instances per A40.

    python experiments/c_parity.py --model pi05_libero --launch [--dry]     # the reference: its 40 tasks
    python experiments/c_parity.py --model pi05_l90c --launch [--after JOB]  # the split's training tasks (LIBERO-90 + suite)
    python experiments/c_parity.py --read [--record]

Rows: ledger c_parity, tag = the model.  The served backbone is PolicyBackbone(BACKBONE_CONFIG) with the plain Euler sampler
(MODE=ours, which matched openpi's own Policy on LIBERO-Goal: ledger r1_rollout).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "l90c")
TRIALS = 20
CHUNK = 5
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"   # sky1 = the login node (also a compute node): never land there
SUITES40 = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def instances(model):
    """-> [(suite, [task indices])] in chunks of CHUNK tasks."""
    from lifelong_ood import l90c
    if model == "pi05_libero":
        groups = {s: list(range(10)) for s in SUITES40}
    else:
        s = l90c.load_split()
        groups = {}
        for tid in s["train"]:
            suite, i = tid.split(":")
            groups.setdefault(suite, []).append(int(i))
    out = []
    for suite, ids in groups.items():
        ids = sorted(ids)
        out += [(suite, ids[k:k + CHUNK]) for k in range(0, len(ids), CHUNK)]
    return out


def launch(a):
    os.makedirs(os.path.join(R, "packs"), exist_ok=True)
    specs = []
    for suite, ids in instances(a.model):
        tag = a.tag or a.model
        name = f"parity_{tag}_{suite}_{ids[0]}-{ids[-1]}"
        out = os.path.join(R, name + ".json")
        if os.path.exists(out):
            continue
        sp = dict(MODE="ours", SUITE=suite, TASKS="+".join(map(str, ids)), TRIALS=TRIALS, FIRST=0, EXPERIMENT="c_parity",
                  TAG=tag, OUT=out, BACKBONE_CONFIG=a.model, **({"BACKBONE_CKPT": a.ckpt} if a.ckpt else {}))
        specs.append(sp)
    jobs = []
    for k in range(0, len(specs), 2):
        path = os.path.join(R, "packs", f"{a.tag or a.model}_{k // 2:03d}.tsv")
        with open(path, "w", encoding="utf-8") as fh:
            for sp in specs[k:k + 2]:
                fh.write("\t".join(f"{kk}={v}" for kk, v in sp.items()) + "\n")
        args = ["--partition=overcap", "--account=overcap", "--qos=short", "--time=04:00:00", "--gres=gpu:a40:1",
                f"--exclude={EXCLUDE}"] + ([f"--dependency=afterok:{a.after}"] if a.after else []) + \
               [f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        if a.dry:
            print("[dry] sbatch", " ".join(args), [sp["TASKS"] for sp in specs[k:k + 2]])
            continue
        j = subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        jobs.append(j)
        print(f"pack {os.path.basename(path)} {[sp['SUITE'] + ':' + sp['TASKS'] for sp in specs[k:k + 2]]} -> job {j}", flush=True)
    print("jobs:", ":".join(jobs))


def per_task(model):
    from lifelong_ood import ledger
    out = {}
    for r in ledger.load("c_parity", tag=model):
        for k, v in r["metrics"]["tasks"].items():
            out[f"{r['bench']}:{k}"] = dict(sr=v["success_rate"], n=v["n"], language=v["language"])
    return out


def read(record, l90c_tag="pi05_l90c"):
    ref, l90c_ = per_task("pi05_libero"), per_task(l90c_tag)
    res = {}
    for name, d in (("pi05_libero", ref), ("pi05_l90c", l90c_)):
        if d:
            v = np.array([x["sr"] for x in d.values()])
            res[name] = dict(tasks=len(d), episodes=int(sum(x["n"] for x in d.values())), mean=float(v.mean()),
                             median=float(np.median(v)), min=float(v.min()),
                             by_suite={s: float(np.mean([x["sr"] for k, x in d.items() if k.startswith(s + ":")]))
                                       for s in sorted({k.split(":")[0] for k in d})})
    if ref and l90c_:
        level = res["pi05_libero"]["mean"]
        v = np.array([x["sr"] for x in l90c_.values()])
        rng = np.random.default_rng(0)
        boot = [rng.choice(v, len(v)).mean() for _ in range(5000)]
        res["parity"] = dict(level_40=level, l90c_mean=float(v.mean()), l90c_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                             reached=bool(v.mean() >= level), tasks_below_level_minus_0p2=sorted(k for k, x in l90c_.items() if x["sr"] < level - 0.2))
        shared = sorted(set(ref) & set(l90c_))
        if shared:
            res["parity"]["same_tasks"] = dict(n=len(shared), libero=float(np.mean([ref[k]["sr"] for k in shared])),
                                               l90c=float(np.mean([l90c_[k]["sr"] for k in shared])))
    res["per_task"] = dict(pi05_libero=ref, pi05_l90c=l90c_)
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))
    with open(os.path.join(R, "c_parity.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if record and "parity" in res:
        from lifelong_ood import ledger
        print(ledger.record("c_parity", bench="libero_90_c", tag=f"read_{l90c_tag}", seed=7, metrics={k: v for k, v in res.items() if k != "per_task"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(trials=TRIALS),
                            script="experiments/c_parity.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi05_l90c")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--after", default="")
    ap.add_argument("--ckpt", default="", help="BACKBONE_CKPT: a converted checkpoint dir (e.g. pytorch/pi05_l90c_5000)")
    ap.add_argument("--tag", default="", help="ledger tag / file prefix, e.g. pi05_l90c@5000 (default: the model)")
    ap.add_argument("--l90c-tag", default="pi05_l90c", help="--read: the tag of the pi05_l90c rows to compare")
    a = ap.parse_args()
    if a.launch:
        launch(a)
    if a.read:
        read(a.record, a.l90c_tag)
