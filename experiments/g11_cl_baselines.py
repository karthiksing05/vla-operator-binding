"""Plan G, read G11: standard continual-learning baselines on the W1 stream (novelty audit, docs/history/NOVELTY.md section 3.3 item 4).
One rank-16 LoRA on pi0.5's action expert trained through W1's arrival order a, one demonstration per task (each task's own
scene): sequential fine-tuning (each task on its own demonstration, in turn) and experience replay (each batch half the current
task's demonstration, half all earlier ones); experiments/b_continual.py, 500 steps per task.  The final adapter serves EVERY
task, as these methods do; tested on E3's fresh set b at noise seed 13, paired with the W1 learner's served arms (order a).

    $PY experiments/g11_cl_baselines.py --fit | --test | --read [--record]
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
SEED, TAG = 13, "_g11"
MODES = ("seq", "er")


def order():
    return json.load(open(os.path.join(R, "w1_state.json")))["a"]["order"]


def fit():
    for m in MODES:
        if os.path.exists(os.path.join(R, f"bcl_{m}{TAG}.pt")):
            continue
        script = f"experiments/b_continual.py --mode {m} --tasks {'+'.join(map(str, order()))} --tag {TAG}"
        out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                              "--account=overcap", "--qos=short", f"--export=ALL,ENV=openpi,SCRIPT={script}",
                              "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
        print(m, out.stdout.strip() or out.stderr.strip(), flush=True)


def launch():
    for t in order():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(f"bcl@{m}{TAG}" for m in MODES), "--experiment",
               "g11_main", "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind", "track", "--partition", "overcap",
               "--per", "2", "--fresh", "--fresh-dir", "fresh_states_b"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-80:], flush=True)


def read(record):
    import e1_read as E1
    import e3_read as E3
    import e4_stream as E4
    w1 = json.load(open(os.path.join(R, "w1_state.json")))
    sv = E4.served(w1)["a"]
    x, missing = {}, []
    for t, (fin, _) in sv.items():
        zs = E3.load("e3_main", t, "zs")
        served = zs if fin == "zs" else next((z for z in (E4.load(m, t, fin) for m in ("w1_main", "e4_main")) if z is not None), None)
        z = dict(w1=served, zs=zs, **{m: E3.load("g11_main", t, f"bcl@{m}{TAG}") for m in MODES})
        if None in z.values():
            missing.append(t)
            continue
        x[t] = z
    C = E1.compare
    res = dict(missing=missing, tasks=len(x))
    for m in MODES:
        res[f"w1_vs_{m}"] = C({t: {"x": z["w1"], "y": z[m]} for t, z in x.items()})
        res[f"{m}_vs_zs"] = C({t: {"x": z[m], "y": z["zs"]} for t, z in x.items()})
    first, last = order()[:10], order()[-10:]
    res["by_position"] = {m: dict(first10=int(sum(sum(x[str(t)][m].values()) for t in first if str(t) in x)),
                                  last10=int(sum(sum(x[str(t)][m].values()) for t in last if str(t) in x))) for m in MODES + ("w1",)}
    res["per_task"] = {t: {k: int(sum(v.values())) for k, v in z.items()} for t, z in x.items()}
    for k in [f"w1_vs_{m}" for m in MODES] + [f"{m}_vs_zs" for m in MODES]:
        h = res[k]
        print(f"[g11] {k:10} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}", flush=True)
    print("[g11] by arrival position (first 10 / last 10):", res["by_position"], "| missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g11_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g11_cl_baselines", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(tag=TAG),
                            script="experiments/g11_cl_baselines.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--fit" in sys.argv:
        fit()
    if "--test" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
