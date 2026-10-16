"""PLAN G, READ G7b: DSRL steering the LIBRARY's composition (lifelong_ood/policy/dsrl.py StackHooks): E3's kept concepts decide the
field in each phase (object prototype until the grasp, goal prototype after), a SAC agent trained from 100 practice episodes per
task picks each chunk's initial noise (the authors' pi0.5 LIBERO configuration, as in G7).  Complementary, not competing: does
noise-space refinement on top of shared prototypes reach DSRL on a per-task adapter (G7's dsrls, the strongest arm run)?
Training on stored states 0-29, acceptance on states 0-9 (kept only if it beats the policy), test on E3's fresh set b, noise
seed 13, paired with E3's and G7's episodes.

    $PY experiments/g7b_dsrl_library.py --train | --accept --test | --read [--record]
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


def arm(v):
    return f"dsrlf@{v['object_proto']}.{v['goal_proto']}"


def launch(exp, trials, fresh, train=False):
    for t, v in E3PLAN.items():
        if train and os.path.exists(os.path.join(R, f"dsrlf_{t}_actor.pt")):
            continue
        a = arm(v).replace("dsrlf@", "dsrlf_train@") if train else arm(v)
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", a, "--experiment", exp, "--first", "0", "--trials",
               str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "1"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def read(record):
    import e1_read as E1
    import e3_read as E3
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    g7 = json.load(open(os.path.join(R, "g7_read.json")))
    test, kept, missing = {}, {}, []
    for t, v in E3PLAN.items():
        za, zx = E3.load("e3_accept", t, "zs"), E3.load("e3_main", t, "zs")
        fa, fx = E3.load("g7b_accept", t, arm(v)), E3.load("g7b_main", t, arm(v))
        if None in (za, zx, fa, fx):
            missing.append(t)
            continue
        kept[t] = "dsrlf" if sum(fa.values()) > sum(za.values()) else "zs"
        z = dict(zs=zx, dsrlf=fx, dsrlf_acc=fx if kept[t] == "dsrlf" else zx,
                 lib=E3.load("e3_main", t, E3.arms(v, True)[e3["kept"][t]]))
        if t in g7["kept"] and "dsrls" in g7["kept"][t]:
            z["dsrls_acc"] = E3.load("g7_main", t, "dsrls") if g7["kept"][t]["dsrls"] == "dsrls" else zx
        test[t] = z
    C = E1.compare
    h1 = C({t: {"x": z["dsrlf_acc"], "y": z["dsrls_acc"]} for t, z in test.items() if "dsrls_acc" in z})
    h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
    h2 = C({t: {"x": z["dsrlf_acc"], "y": z["lib"]} for t, z in test.items()})
    h2["passes"] = bool(h2["x"] > h2["y"] and h2["sign_p"] < 0.05)
    res = dict(missing=missing, kept={k: sum(x == k for x in kept.values()) for k in ("zs", "dsrlf")}, H1=h1, H2=h2,
               dsrlf_vs_zs=C({t: {"x": z["dsrlf"], "y": z["zs"]} for t, z in test.items()}),
               per_task={t: {k: int(sum(x.values())) for k, x in z.items()} for t, z in test.items()})
    for k in ("H1", "H2", "dsrlf_vs_zs"):
        h = res[k]
        print(f"[g7b] {k:12} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, diff "
              f"{h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print("[g7b] kept", res["kept"], "| missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g7b_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g7_dsrl", bench="libero_90", tag="read_library", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(episodes=EPISODES), script="experiments/g7b_dsrl_library.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--train" in sys.argv:
        launch("g7b_train", EPISODES, False, train=True)
    if "--accept" in sys.argv:
        launch("g7b_accept", 10, False)
    if "--test" in sys.argv:
        launch("g7b_main", 20, True)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
