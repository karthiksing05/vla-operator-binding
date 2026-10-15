"""PLAN G, READ G7: DSRL (Wagenmaker et al., CoRL 2025, arXiv 2506.15799) against the library, on E3's tasks, start states and seed.

DSRL keeps the policy's weights frozen and trains a SAC agent that chooses each action chunk's initial noise, from sparse task
reward over practice episodes (lifelong_ood/policy/dsrl.py, the authors' pi0.5 LIBERO configuration: official=1).  Two arms:
  dsrlz -- steering the frozen policy itself: no demonstration, 100 practice episodes (the library uses 10, plus at most one
           demonstration);
  dsrls -- steering the task's scene-correct one-demo LoRA (the per-task baseline's adapter): one demonstration, 100 episodes.
Training runs on stored start states 0-29 (STATE_MOD 30); each trained actor then gets the same acceptance test as every other
arm (stored states 0-9, kept only if it beats the policy) and E3's fresh test states (set b, 20 per task, noise seed 13), paired
with E3's recorded arms.

    --train | --accept | --test | --read [--record]
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
SEED, EPISODES = 13, 100
E3PLAN = json.load(open(os.path.join(R, "e3_plan.json")))
TASKS = [int(t) for t in E3PLAN]
ARMS = ("dsrlz", "dsrls")


def arms_of(t):
    return [a for a in ARMS if a == "dsrlz" or E3PLAN[str(t)]["lora"]]          # task 51 has no demonstration, so no LoRA


def launch(exp, trials, fresh, train=False):
    for t in TASKS:
        todo = [a for a in arms_of(t) if not train or not os.path.exists(os.path.join(R, f"{a}_{t}_actor.pt"))]
        if not todo:
            continue
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(a + "_train" if train else a for a in todo),
               "--experiment", exp, "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track",
               "--partition", "overcap", "--per", "1"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def read(record):
    import e1_read as E1
    import e3_read as E3
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    acc, test, kept, practice, missing = {}, {}, {}, {}, []
    for t in map(str, TASKS):
        v = E3PLAN[t]
        za, zx = E3.load("e3_accept", t, "zs"), E3.load("e3_main", t, "zs")
        lib = E3.load("e3_main", t, E3.arms(v, True)[e3["kept"][t]])
        lora = E3.load("e3_main", t, E3.arms(v, True)[e3["kept_lora"][t]]) if t in e3["kept_lora"] else None
        cur = dict(zs=zx, lib=lib)
        if lora is not None:
            cur["lora_acc"] = lora
        acc[t], kept[t] = {"zs": int(sum(za.values())) if za else None}, {}
        for a in arms_of(int(t)):
            sa, sx, tr = E3.load("g7_accept", t, a), E3.load("g7_main", t, a), E3.load("g7_train", t, a + "_train")
            if None in (sa, sx, za, zx, lib):
                missing.append(f"{t}:{a}")
                continue
            acc[t][a] = int(sum(sa.values()))
            kept[t][a] = a if acc[t][a] > acc[t]["zs"] else "zs"
            cur[a], cur[a + "_acc"] = sx, sx if kept[t][a] == a else zx
            if tr:
                practice.setdefault(a, [0, 0, 0, 0])
                n = len(tr)
                practice[a][0] += sum(tr[i] for i in tr if i < 10)
                practice[a][1] += min(n, 10)
                practice[a][2] += sum(tr[i] for i in tr if i >= n - 20)
                practice[a][3] += min(n, 20)
        test[t] = cur
    C = E1.compare
    hyp = {f"lib_vs_{a}_acc": C({t: {"x": z["lib"], "y": z[a + "_acc"]} for t, z in test.items() if a + "_acc" in z}) for a in ARMS}
    for h in hyp.values():
        h["noninferior"] = bool(h["diff"]["lo"] > -0.05)
        h["superior"] = bool(h["diff"]["lo"] > 0)
    desc = {f"{a}_vs_zs": C({t: {"x": z[a], "y": z["zs"]} for t, z in test.items() if a in z}) for a in ARMS}
    desc["dsrls_acc_vs_lora_acc"] = C({t: {"x": z["dsrls_acc"], "y": z["lora_acc"]} for t, z in test.items()
                                       if "dsrls_acc" in z and "lora_acc" in z})
    res = dict(missing=missing, acceptance=acc, kept=kept, hypotheses=hyp, descriptive=desc,
               practice={a: dict(first10=p[0], first10_n=p[1], last20=p[2], last20_n=p[3]) for a, p in practice.items()},
               per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()})
    for k, h in list(hyp.items()) + list(desc.items()):
        print(f"[g7] {k}: {h['x']} vs {h['y']} of {h['n']}, diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, "
              f"{h['diff']['hi']:+.3f}]", flush=True)
    print("[g7] practice", res["practice"], "| missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g7_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g7_dsrl", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(episodes=EPISODES), script="experiments/g7_dsrl.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--train" in sys.argv:
        launch("g7_train", EPISODES, False, train=True)
    if "--accept" in sys.argv:
        launch("g7_accept", 10, False)
    if "--test" in sys.argv:
        launch("g7_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
