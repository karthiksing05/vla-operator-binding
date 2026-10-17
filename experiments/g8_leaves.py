"""Plan G, read G8: the concept hierarchy with TASK LEAVES.  Below the type and instance concepts sits the task itself: its leaf is
the one-demonstration adapter of the task (rank 16, its own scene: the per-task LoRA), stored only where practice says the shared
concepts do not serve the task.  The acceptance test (stored states 0-9, as recorded for E3 and E5) keeps the best of the
policy, the composition of shared concepts and the leaf (ties: the policy, then the composition); the comparison is the per-task
LoRA with its own acceptance test (the policy or the LoRA).  Test on a NEW start-state set d (LIBERO's sampler, seeds 600000+,
20 per task), noise seed 13, over the 66 single-step tasks of E3 and E5 with a demonstration.

    $PY experiments/g8_leaves.py --test | --read [--record]
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


def tasks():
    """-> {task: (module, arms, acceptance experiment)} for E3's and E5's tasks with a demonstration."""
    import e3_read as E3
    import e5_read as E5
    out = {}
    for t, v in json.load(open(os.path.join(R, "e3_plan.json"))).items():
        if v["lora"]:
            a = E3.arms(v)
            out[t] = (E3, {k: a[k] for k in ORDER}, "e3_accept")
    for t, v in json.load(open(os.path.join(R, "e5_plan.json")))["tasks"].items():
        out[t] = (E5, E5.arms(v), "e5_accept")
    return out


def launch():
    for t, (_, arms, _) in tasks().items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms[k] for k in ORDER), "--experiment", "g8_main",
               "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "3",
               "--fresh", "--fresh-dir", "fresh_states_d"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def load_test(t, a):
    f = os.path.join(R, f"g8_main_{t}_{a.replace('@', '_')}_seed{SEED}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    import e1_read as E1
    C = E1.compare
    kept, kept_lora, test, missing = {}, {}, {}, []
    for t, (mod, arms, acc_exp) in tasks().items():
        s = {k: mod.load(acc_exp, t, arms[k]) for k in ORDER}
        x = {k: load_test(t, arms[k]) for k in ORDER}
        if None in s.values() or None in x.values():
            missing.append(t)
            continue
        acc = {k: sum(z.values()) for k, z in s.items()}
        kept[t] = max(ORDER, key=lambda k: (acc[k], -ORDER.index(k)))
        kept_lora[t] = "lora" if acc["lora"] > acc["zs"] else "zs"
        lib = max(("zs", "hfac"), key=lambda k: (acc[k], -ORDER.index(k)))
        test[t] = dict(leaves=x[kept[t]], lora_acc=x[kept_lora[t]], library=x[lib], zs=x["zs"])
    h1 = C({t: {"x": z["leaves"], "y": z["lora_acc"]} for t, z in test.items()})
    h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
    h1["superior"] = bool(h1["diff"]["lo"] > 0)
    res = dict(missing=missing, tasks=len(test), H1=h1, kept={k: sum(v == k for v in kept.values()) for k in ORDER},
               library_vs_lora_acc=C({t: {"x": z["library"], "y": z["lora_acc"]} for t, z in test.items()}),
               leaves_vs_zs=C({t: {"x": z["leaves"], "y": z["zs"]} for t, z in test.items()}),
               leaves_vs_library=C({t: {"x": z["leaves"], "y": z["library"]} for t, z in test.items()}),
               per_task={t: {k: int(sum(v.values())) for k, v in z.items()} | {"kept": kept[t]} for t, z in test.items()})
    res["bytes"] = dict(leaves=res["kept"]["lora"], per_task_loras=sum(v == "lora" for v in kept_lora.values()))
    for k in ("H1", "library_vs_lora_acc", "leaves_vs_zs", "leaves_vs_library"):
        h = res[k]
        print(f"[g8] {k:20} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, diff "
              f"{h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print("[g8] kept", res["kept"], "| leaves", res["bytes"], "| missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g8_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g8_leaves", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g8_leaves.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--test" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
