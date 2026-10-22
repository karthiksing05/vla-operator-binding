"""Plan D2, read G30e (ledger m3_prereg tag g30_rl; docs/BASELINES.md): pi_RL's Flow-Noise PPO with a LoRA on pi0.5's action expert
(lifelong_ood/policy/pirl.py), initialised from each task's one-demonstration LoRA, on the 20 G3/G4 tasks of E3.  One training run
per task (100 practice episodes on stored states 0-29, one A40 to itself) saves two adapters: lora@pirlm at the matched budget
(577 s of wall-clock, the library's practice) and lora@pirl at 100 episodes (DSRL's budget, descriptive).  Each gets E3's
acceptance test (states 0-9) and E3's set b (20 per task, seed 13), paired with E3's and G7's arms.

    --train | --accept | --test | --read [--record]
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED, EPISODES = 13, 100
TASKS = [9, 10, 12, 13, 14, 15, 30, 25, 31, 36, 41, 43, 55, 56, 57, 58, 59, 61, 62, 71]
ARMS = ("lora@pirlm", "lora@pirl")
EXCL = "hal,tars,t1000,vicki,johnny5,c3po,olivaw"
ENV = dict(os.environ)


def launch(exp, arms, trials, fresh, per, tasks=None):
    for t in tasks or TASKS:
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(arms), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", str(per)]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=ENV).stdout.strip()
        for j in re.findall(r"-> job (\d+)", out):                    # launch_pack has no --exclude (SBATCH_EXCLUDE is not read)
            subprocess.run(["scontrol", "update", f"JobId={j}", f"ExcNodeList={EXCL}"])
        print(t, out[-140:], flush=True)


def trained(t):
    return all(os.path.exists(os.path.join(R, f"{a[5:]}_{t}_lora.pt")) for a in ARMS)


def read(record):
    import e1_read as E1
    import e3_read as E3
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    plan = json.load(open(os.path.join(R, "e3_plan.json")))
    acc, kept, test, practice, missing = {}, {}, {}, {}, []
    for t in map(str, TASKS):
        v = plan[t]
        za, zx = E3.load("e3_accept", t, "zs"), E3.load("e3_main", t, "zs")
        lib = E3.load("e3_main", t, E3.arms(v, True)[e3["kept"][t]])
        lora = E3.load("e3_main", t, E3.arms(v, True)[e3["kept_lora"][t]])
        lora_raw = E3.load("e3_main", t, "lorasc")
        ga, gx = E3.load("g7_accept", t, "dsrls"), E3.load("g7_main", t, "dsrls")
        cur = dict(zs=zx, lib=lib, lora_acc=lora, lora=lora_raw)
        acc[t], kept[t] = {"zs": int(sum(za.values()))}, {}
        if ga is not None and gx is not None:
            acc[t]["dsrls"] = int(sum(ga.values()))
            cur["dsrls_acc"] = gx if acc[t]["dsrls"] > acc[t]["zs"] else zx
        for a in ARMS:
            sa, sx = E3.load("g30_rlaccept", t, a), E3.load("g30_rlmain", t, a)
            if sa is None or sx is None:
                missing.append(f"{t}:{a}")
                continue
            k = a[5:]
            acc[t][k] = int(sum(sa.values()))
            kept[t][k] = k if acc[t][k] > acc[t]["zs"] else "zs"
            cur[k], cur[k + "_acc"] = sx, (sx if kept[t][k] == k else zx)
        tr = E3.load("g30_rltrain", t, "pirl_train")
        if tr:
            n = len(tr)
            practice[t] = dict(first10=sum(tr[i] for i in tr if i < 10), last20=sum(tr[i] for i in tr if i >= n - 20), n=n)
        test[t] = cur
    C = E1.compare
    hyp = {}
    for k in ("pirlm", "pirl"):
        h = C({t: {"x": z["lib"], "y": z[k + "_acc"]} for t, z in test.items() if k + "_acc" in z})
        h["noninferior"] = bool(h["diff"]["lo"] > -0.05)
        h["superior"] = bool(h["diff"]["lo"] > 0)
        hyp[f"lib_vs_{k}_acc"] = h
    desc = {}
    for k in ("pirlm", "pirl"):
        sub = {t: z for t, z in test.items() if k in z}
        desc[f"{k}_vs_zs"] = C({t: {"x": z[k], "y": z["zs"]} for t, z in sub.items()})
        desc[f"{k}_vs_lora"] = C({t: {"x": z[k], "y": z["lora"]} for t, z in sub.items()})
        desc[f"{k}_acc_vs_lora_acc"] = C({t: {"x": z[k + "_acc"], "y": z["lora_acc"]} for t, z in sub.items()})
        desc[f"{k}_acc_vs_dsrls_acc"] = C({t: {"x": z[k + "_acc"], "y": z["dsrls_acc"]} for t, z in sub.items() if "dsrls_acc" in z})
    desc["lib_vs_lora_acc"] = C({t: {"x": z["lib"], "y": z["lora_acc"]} for t, z in test.items()})
    desc["lib_vs_dsrls_acc"] = C({t: {"x": z["lib"], "y": z["dsrls_acc"]} for t, z in test.items() if "dsrls_acc" in z})
    runs = {}
    for t in TASKS:
        f = os.path.join(R, f"pirl_{t}_lora.pt")
        if os.path.exists(f):
            import torch
            m = torch.load(f, map_location="cpu", weights_only=False)["meta"]
            mm = torch.load(os.path.join(R, f"pirlm_{t}_lora.pt"), map_location="cpu", weights_only=False)["meta"]
            tj = os.path.join(R, f"g30_rltrain_{t}_pirl_train_seed13.json")
            eps = sorted((e for v in json.load(open(tj))["tasks"].values() for e in v["episodes"]), key=lambda e: e["trial"]) if os.path.exists(tj) else []
            steps = lambda n: int(sum(e["steps"] for e in eps[:n]))                       # noqa: E731
            runs[str(t)] = dict(final=dict(episodes=m["episodes"], updates=m["updates"], secs=m["elapsed_s"], env_steps=steps(m["episodes"]),
                                           successes=int(sum(m["outcomes"]))),
                                matched=dict(episodes=mm["episodes"], updates=mm["updates"], secs=mm["elapsed_s"], env_steps=steps(mm["episodes"]),
                                             successes=int(sum(mm["outcomes"]))))
    res = dict(missing=missing, acceptance=acc, kept=kept, hypotheses=hyp, descriptive=desc, practice=practice, runs=runs,
               per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()},
               bytes=dict(per_task_lora=18_808_995))
    for k, h in list(hyp.items()) + list(desc.items()):
        print(f"[g30rl] {k}: {h['x']} vs {h['y']} of {h['n']}, discordant {h.get('discordant')}, sign p {h.get('sign_p', float('nan')):.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print("[g30rl] missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g30_rl_read.json"), "w", encoding="utf-8"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g30_rl", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=18_808_995 * len(TASKS), tree=0, codes=0, samples=0, teacher_copies=0, total=18_808_995 * len(TASKS)),
                            config=dict(episodes=EPISODES, budget_s=577), script="experiments/g30_rl.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    only = [int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].split("+")] if "--tasks" in sys.argv else None
    if "--train" in sys.argv:
        launch("g30_rltrain", ["pirl_train"], EPISODES + 1, False, 1, [t for t in (only or TASKS) if not trained(t)])
    if "--accept" in sys.argv:
        launch("g30_rlaccept", list(ARMS), 10, False, 2, [t for t in (only or TASKS) if trained(t)])
    if "--test" in sys.argv:
        launch("g30_rlmain", list(ARMS), 20, True, 2, [t for t in (only or TASKS) if trained(t)])
    if "--read" in sys.argv:
        read("--record" in sys.argv)
