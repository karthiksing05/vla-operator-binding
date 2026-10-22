"""Plan D2 (docs/PLAN_D.md, docs/BASELINES.md): the SFT and continual baselines that are small extensions of existing code, on E3's
30 LIBERO-90 tasks (29 with a demonstration; task 51 has none) and W1's order a, as G11 (experiments/g11_cl_baselines.py).

    full     -- full-parameter SFT of pi0.5's action expert on the task's one own-scene demonstration (r3_fewshot.py --full,
                adapt.train_full: 1,000 steps, lr 2.5e-5, batch 32, bf16 weights with fp32 AdamW masters); r3_full_<t>_g30.pt,
                arm fullft@_g30.  The whole model (3.3B with the VLM) does not fit an A40 with AdamW (about 16 bytes per
                parameter, 53 GB), so the action expert is the full-parameter arm.
    multi5   -- the per-task rank-16 LoRA recipe on 5 own-scene demonstrations (demos 0-4), 2,000 steps; r3_lora_<t>_g30n5.pt,
                arm r3lora@_g30n5.
    multiall -- the same on ALL of the task's own-scene demonstrations (28 to 50; LIBERO ships 50 per task, some in other
                scenes' blocks), 3,000 steps, prefixes computed per batch; r3_lora_<t>_g30nall.pt, arm r3lora@_g30nall.
    ewc, packnet, packnet75 -- experiments/g30_cl.py on W1's order a (bcl_ewc_g30.pt; bcl_packnet_g30_<t>_lora.pt with an
                equal share per task; bcl_packnet75_g30_<t>_lora.pt with LIBERO's prune_perc 0.75).

    $PY experiments/g30_baselines.py --submit full|multi5|multiall|ewc|packnet|packnet75 [--tasks a+b] [--per-job k]
    $PY experiments/g30_baselines.py --job full|multi5|multiall --tasks a+b+...     # inside a GPU job: fits each missing task
    $PY experiments/g30_baselines.py --parity [--dry]                                # stage-1 smoke rollouts (states 30-49)
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
EXCL = "--exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw"
JOBS = {"full": ("--full --half", "r3_full_{t}_g30.pt", "_g30"),
        "multi5": ("--ndemo 5 --steps 2000", "r3_lora_{t}_g30n5.pt", "_g30n5"),
        "multiall": ("--ndemo 0 --steps 3000 --ckpt-every 200", "r3_lora_{t}_g30nall.pt", "_g30nall")}
CL = {"ewc": "--mode ewc --tag _g30", "packnet": "--mode packnet --tag _g30", "packnet75": "--mode packnet --prune 0.75 --tag 75_g30"}
PARITY_TASKS = (13, 36, 52)          # E3 tasks the policy fails zero-shot (0 of 20 on set b); see docs/BASELINES.md


def e3_tasks():
    return sorted(int(t) for t in json.load(open(os.path.join(R, "e3_plan.json"))) if int(t) != 51)


def w1_order():
    return json.load(open(os.path.join(R, "w1_state.json")))["a"]["order"]


def job(kind, tasks):
    """Inside one GPU job: fit every task whose file is missing, one process each (the policy reloaded per task)."""
    extra, pat, tag = JOBS[kind]
    for t in tasks:
        if os.path.exists(os.path.join(R, pat.format(t=t))):
            print(f"[g30] {kind} {t}: exists", flush=True)
            continue
        cmd = [sys.executable, "experiments/r3_fewshot.py", "--task", str(t), "--tag", tag] + extra.split()
        print("[g30] " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"[g30] {kind} {t}: exit {rc}", flush=True)


def prep_demodiff(tasks):
    """DemoDiffusion's demo programs from each task's CURRENT own-scene demonstration (r3_prepare.py, tag _lib40g30; the files of
    the M3-era read, before the own-scene fix of 2026-10-04, are left untouched).  CPU."""
    for t in tasks:
        if os.path.exists(os.path.join(R, f"r3_prog_{t}_raw_lib40g30.npz")):
            continue
        cmd = [sys.executable, "experiments/r3_prepare.py", "--tree", "results/robot/r2b_libero_40_tree.npz", "--library-npz",
               "results/robot/r2b_libero_40.npz", "--tasks", str(t), "--levels", "2,3,4", "--tag", "_lib40g30"]
        print(f"[g30] demodiff prep {t}: exit {subprocess.run(cmd, cwd=ROOT).returncode}", flush=True)


def sbatch(script, dry=False):
    args = ["sbatch", "--parsable", EXCL, "--partition=overcap", "--account=overcap", "--qos=short", "--time=04:00:00",
            "--job-name=g30", f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"]
    if dry:
        print("[dry]", " ".join(args))
        return "DRY"
    out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def submit(kind, tasks, per_job, dry):
    if kind in CL:
        print(kind, sbatch(f"experiments/g30_cl.py {CL[kind]} --tasks {'+'.join(map(str, w1_order()))}", dry), flush=True)
        return
    _, pat, _ = JOBS[kind]
    todo = [t for t in tasks if not os.path.exists(os.path.join(R, pat.format(t=t)))]
    for i in range(0, len(todo), per_job):
        part = todo[i:i + per_job]
        print(kind, part, sbatch(f"experiments/g30_baselines.py --job {kind} --tasks {'+'.join(map(str, part))}", dry), flush=True)


def parity(dry):
    """Stage-1 smoke rollouts on the stored init states 30-49 (no read uses them; E3's reads use fresh set b): the all-demo LoRA
    (parity with the published LIBERO-90 multi-task level, docs/BASELINES.md), 5-demo LoRA and the full-SFT expert, seed 13."""
    for t in PARITY_TASKS:
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", "r3lora@_g30nall,r3lora@_g30n5,fullft@_g30",
               "--experiment", "g30_parity", "--first", "30", "--trials", "20", "--seed", "13", "--partition", "overcap", "--per", "2"]
        if dry:
            cmd.append("--dry")
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-300:], flush=True)


def parity_read():
    out = {}
    for t in PARITY_TASKS:
        for arm in ("r3lora__g30nall", "r3lora__g30n5", "fullft__g30"):
            f = os.path.join(R, f"g30_parity_{t}_{arm}_seed13.json")
            if os.path.exists(f):
                z = json.load(open(f))
                out[f"{t}/{arm}"] = dict(successes=round(z["success_rate"] * z["n_episodes"]), n=z["n_episodes"])
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--per-job", type=int, default=10)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--prep-demodiff", action="store_true", help="CPU: DemoDiffusion's demo programs (sbatch slurm/jobs/cpu.sbatch)")
    ap.add_argument("--parity-read", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    ts = [int(x) for x in a.tasks.split("+")] if a.tasks else e3_tasks()
    if a.job:
        job(a.job, ts)
    if a.submit:
        submit(a.submit, ts, a.per_job, a.dry)
    if a.prep_demodiff:
        prep_demodiff(ts)
    if a.parity:
        parity(a.dry)
    if a.parity_read:
        parity_read()
