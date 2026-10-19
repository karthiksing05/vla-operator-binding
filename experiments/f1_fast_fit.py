"""Plan F1, step 3: the confirmatory test of the fast fit recipe (pre-registered: experiments/prereg/f1_fast_fit.py, ledger m3_prereg
tag f1_fast_fit).  docs/FIT_TIME.md.

H1 (library): E3's 22 prototypes refit with the fast recipe (RECIPE; chosen by experiments/f1_levers.py on E5's units only) as
prim_<name>F1_0.pt, ledger rows f1_fit.  The fast library's composition (arm fac@<object>F1.<goal>F1) on E3's 29 tasks with a
per-task LoRA, fresh start states set b (results/robot/fresh_states_b), 20 per task, noise seed 13, paired by state with E3's own
library's composition episodes (e3_main, arm fac@<object>.<goal>).  Non-inferiority: task-level bootstrap 95% lower bound of
(fast - E3) above -0.05.  Set b is reused: no design decision of F1 read any episode (the recipe was chosen on held-out flow loss of
E5's units); it is the set on which the existing library's episodes exist.  Descriptive: acceptance (stored states 0-9, seed 13,
f1_accept) and the kept arm against E3's kept arm; the 30th task (no LoRA); fit minutes per unit (job wall).

H2 (leaves, rank 4): G8's rule (keep the best of the policy, the composition and the leaf on the stored acceptance states; ties to
the policy, then the composition) with RANK-4 leaves: E3's tasks reuse G9's rank-4 LoRAs (r3_lora_<t>_r4.pt) and acceptance
(g9_accept, seed 13); E5's 37 tasks get rank-4 leaves (experiments/f1_leaf_fit.py: r3_fewshot.py --rank 4, r3_lora_<t>_f1r4.pt)
and acceptance on E5's acceptance states and seed (f1_leaf_accept, seed 14, as e5_accept).  Test on G8's set d (fresh_states_d, 20
per task, seed 13) only where the rank-4 rule keeps the leaf (f1_leaf_main); elsewhere the kept arm's G8 episodes (g8_main) are the
test.  Against G8's rule with rank-16 leaves (g8_main), paired by state on the 66 tasks: non-inferiority at -0.05.

    $PY experiments/f1_fast_fit.py --run >> results/robot/f1/f1_fast_fit.log 2>&1 &   # detached: fits, waits, launches, reads
    $PY experiments/f1_fast_fit.py --read [--record]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1_common as F                                               # noqa: E402

ROOT, R = F.ROOT, F.R
SEED, SUFFIX = 13, "F1"
RECIPE = None                       # filled from the lever read before the registration: dict(steps=, init=, flags=)
LEAF_ARM_E3, LEAF_ARM_E5 = "lorat@_r4", "lorat@_f1r4"
LOG = os.path.join(F.F1, "f1_fast_fit.log")


def say(*x):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[f1f]", *x, flush=True)


def lib_tasks():
    return {t: v for t, v in F.E3PLAN.items() if v["lora"]}


def arm(v):
    return f"fac@{v['object_proto']}{SUFFIX}.{v['goal_proto']}{SUFFIX}"


def fit_cmd(n):
    """The prototype's own c0_fit command with the recipe's flags; instance children start from their (fast) parent."""
    cmd = F.fit_row(n)["command"].split("c0_fit.py", 1)[1].strip()
    assert f"--name {n} " in cmd + " ", cmd
    cmd = cmd.replace(f"--name {n}", f"--name {n}{SUFFIX}")
    cmd += f" --cache --steps {RECIPE['steps']} --record-as f1_fit"
    init = RECIPE["init"]
    if init == "parent_or_best" and n in F.PARENT:
        cmd += f" --init prim_{F.PARENT[n]}{SUFFIX}_0.pt"
    elif init in ("parent_or_best", "best"):
        cmd += " --init-best " + "+".join(F.pool(F.kind(n), exclude=[n] + [c for c, p in F.PARENT.items() if p == n]))
    elif init == "parent" and n in F.PARENT:
        cmd += f" --init prim_{F.PARENT[n]}{SUFFIX}_0.pt"
    return "experiments/c0_fit.py " + cmd


def fit():
    jobs = {}
    names = F.E3_NAMES
    for n in sorted(names, key=lambda n: n in F.PARENT):            # parents first: children wait for them
        if os.path.exists(os.path.join(R, f"prim_{n}{SUFFIX}_0.pt")):
            continue
        dep = jobs.get(F.PARENT.get(n), "") if "parent" in RECIPE["init"] else ""
        jobs[n] = F.sbatch(fit_cmd(n), dep=dep)
        say("fit", n, jobs[n], fit_cmd(n)[:200])
    return jobs


def leaf_fit():
    jobs = {}
    for t in sorted(F.E5PLAN, key=int):
        if not os.path.exists(os.path.join(R, f"r3_lora_{t}_f1r4.pt")):
            jobs[t] = F.sbatch(f"experiments/f1_leaf_fit.py --task {t}")
            say("leaf fit", t, jobs[t])
    return jobs


def pack(tasks_arms, exp, trials, seed, fresh_dir=None, per=2):
    for t, arms in tasks_arms.items():
        cmd = [F.PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", ",".join(arms), "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(seed), "--bind", "track", "--partition", "overcap", "--per", str(per)]
        cmd += ["--fresh", "--fresh-dir", fresh_dir] if fresh_dir else []
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        say("launch", exp, t, arms, (out.stdout.strip() or out.stderr.strip())[-120:])


def load(exp, t, a, seed):
    f = os.path.join(R, f"{exp}_{t}_{a.replace('@', '_')}_seed{seed}.json")
    if not os.path.exists(f):
        return None
    eps = {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}
    return eps or None


def wait_files(paths, what, poll=120):
    while True:
        miss = [p for p in paths if not os.path.exists(p)]
        if not miss:
            say(f"{what}: all {len(paths)} present")
            return
        say(f"{what}: {len(paths) - len(miss)} of {len(paths)} present")
        time.sleep(poll)


def wait_jobs(jobs, what, poll=120):
    ids = [j for j in jobs.values() if j and j.split(";")[0].isdigit()]
    while True:
        out = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(ids)], capture_output=True, text=True).stdout.split() if ids else []
        if not out:
            say(f"{what}: {len(ids)} jobs finished")
            return
        say(f"{what}: {len(out)} of {len(ids)} jobs queued or running")
        time.sleep(poll)


def leaf_tasks():
    """-> {task: (acceptance module tag, acceptance seed, the rank-4 leaf arm, G8's arms)}."""
    import g8_leaves as G8
    out = {}
    for t, (mod, arms, acc_exp) in G8.tasks().items():
        e3 = acc_exp == "e3_accept"
        out[t] = dict(acc_exp=acc_exp, acc_seed=13 if e3 else 14, leaf=LEAF_ARM_E3 if e3 else LEAF_ARM_E5,
                      leaf_acc_exp="g9_accept" if e3 else "f1_leaf_accept", arms=arms)
    return out


def leaf_rule(t, L):
    """The rank-4 rule's kept arm on task t from acceptance, or None when an acceptance file is missing."""
    acc = {k: load(L["acc_exp"], t, L["arms"][k], L["acc_seed"]) for k in ("zs", "hfac")}
    acc["lora"] = load(L["leaf_acc_exp"], t, L["leaf"], L["acc_seed"])
    if None in acc.values():
        return None
    s = {k: sum(v.values()) for k, v in acc.items()}
    order = ("zs", "hfac", "lora")
    return max(order, key=lambda k: (s[k], -order.index(k)))


def run():
    """Detached: fits -> acceptance and test -> leaves -> read; one line per stage in LOG."""
    os.makedirs(F.F1, exist_ok=True)
    say("start; recipe", RECIPE)
    fj = fit()
    lj = leaf_fit()
    protos = [os.path.join(R, f"prim_{n}{SUFFIX}_0.pt") for n in F.E3_NAMES]
    wait_jobs(fj, "prototype fits")
    wait_files(protos, "fast prototypes")
    tasks = {t: [arm(v)] for t, v in F.E3PLAN.items()}
    pack(tasks, "f1_main", 20, SEED, "fresh_states_b")
    pack(tasks, "f1_accept", 10, SEED)
    wait_jobs(lj, "leaf fits")
    LT = leaf_tasks()
    e5 = {t: [L["leaf"]] for t, L in LT.items() if L["leaf_acc_exp"] == "f1_leaf_accept"}
    wait_files([os.path.join(R, f"r3_lora_{t}_f1r4.pt") for t in e5], "rank-4 leaves")
    pack(e5, "f1_leaf_accept", 10, 14)
    wait_files([os.path.join(R, f"f1_leaf_accept_{t}_{LT[t]['leaf'].replace('@', '_')}_seed14.json") for t in e5], "leaf acceptance")
    kept = {t: leaf_rule(t, L) for t, L in LT.items()}
    say("rank-4 rule kept", {k: sum(v == k for v in kept.values()) for k in ("zs", "hfac", "lora", None)})
    test = {t: [LT[t]["leaf"]] for t, k in kept.items() if k == "lora"}
    pack(test, "f1_leaf_main", 20, SEED, "fresh_states_d")
    wait_files([os.path.join(R, f"f1_main_{t}_{arm(v).replace('@', '_')}_seed{SEED}.json") for t, v in F.E3PLAN.items()], "H1 test")
    wait_files([os.path.join(R, f"f1_accept_{t}_{arm(v).replace('@', '_')}_seed{SEED}.json") for t, v in F.E3PLAN.items()], "acceptance")
    wait_files([os.path.join(R, f"f1_leaf_main_{t}_{LT[t]['leaf'].replace('@', '_')}_seed{SEED}.json") for t in test], "H2 test")
    read(True)
    say("DONE")


def fit_minutes():
    from lifelong_ood import ledger
    rows = {r["tag"][6:]: r for r in ledger.load("f1_fit")}
    out = {}
    for n in F.E3_NAMES:
        r = rows.get(f"{n}{SUFFIX}")
        if r:
            out[n] = dict(job_min=r["metrics"]["f1"]["secs"]["total"] / 60, train_min=r["metrics"]["f1"]["secs"]["train"] / 60,
                          frames=r["metrics"]["frames"], init=r["metrics"]["init"])
    return out


def read(record):
    import e1_read as E1
    C = E1.compare
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    import e3_read as E3
    h1t, desc_all, kept_t, missing = {}, {}, {}, []
    for t, v in F.E3PLAN.items():
        x = load("f1_main", t, arm(v), SEED)
        y = load("e3_main", t, E3.arms(v, True)["hfac"], SEED)
        if x is None or y is None:
            missing.append(t)
            continue
        desc_all[t] = {"x": x, "y": y}
        if v["lora"]:
            h1t[t] = {"x": x, "y": y}
        ax = load("f1_accept", t, arm(v), SEED)
        az = load("e3_accept", t, "zs", SEED)
        if ax is not None and az is not None:
            k = "hfac" if sum(ax.values()) > sum(az.values()) else "zs"
            kept_t[t] = {"x": x if k == "hfac" else load("e3_main", t, "zs", SEED),
                         "y": load("e3_main", t, E3.arms(v, True)[e3["kept"][t]], SEED), "kept": k}
    res = dict(recipe=RECIPE, missing=missing)
    if h1t:
        h1 = C(h1t)
        h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
        res["H1"] = h1
        res["all30"] = C(desc_all)
        if kept_t:
            res["kept_vs_e3_kept"] = C({t: {"x": z["x"], "y": z["y"]} for t, z in kept_t.items() if t in h1t})
            res["kept_fast"] = {k: sum(z["kept"] == k for z in kept_t.values()) for k in ("zs", "hfac")}
    res["fit"] = fit_minutes()
    if res["fit"]:
        res["fit_job_min_mean"] = sum(v["job_min"] for v in res["fit"].values()) / len(res["fit"])
        res["fit_train_min_mean"] = sum(v["train_min"] for v in res["fit"].values()) / len(res["fit"])
    # H2: the rank-4 rule against G8's rank-16 rule
    import g8_leaves as G8
    g8 = json.load(open(os.path.join(R, "g8_read.json")))
    LT = leaf_tasks()
    h2t, miss2, kept4 = {}, [], {}
    for t, L in LT.items():
        k = leaf_rule(t, L)
        kept4[t] = k
        y = G8.load_test(t, L["arms"][g8["per_task"][t]["kept"]]) if t in g8["per_task"] else None
        x = None if k is None else (load("f1_leaf_main", t, L["leaf"], SEED) if k == "lora" else G8.load_test(t, L["arms"][k]))
        if x is None or y is None:
            miss2.append(t)
            continue
        h2t[t] = {"x": x, "y": y}
    if h2t:
        h2 = C(h2t)
        h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
        res["H2"] = h2
    res["H2_missing"] = miss2
    res["kept_r4_rule"] = {k: sum(v == k for v in kept4.values()) for k in ("zs", "hfac", "lora")}
    res["kept_r16_rule"] = g8["kept"]
    for k in ("H1", "all30", "kept_vs_e3_kept", "H2"):
        if k in res:
            h = res[k]
            say(f"{k:16} {h['x']} vs {h['y']} of {h['n']} ({h['tasks']} tasks), diff {h['diff']['diff']:+.3f} "
                f"[{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]" + (f" -> non-inferior {h.get('noninferior')}" if "noninferior" in h else ""))
    say("fit min/unit", res.get("fit_job_min_mean"), "| kept r4 rule", res["kept_r4_rule"], "| missing", missing, miss2)
    json.dump(res, open(os.path.join(F.F1, "f1_fast_fit_read.json"), "w", encoding="utf-8"), indent=1)
    if record and not missing and not miss2:
        from lifelong_ood import ledger
        say(ledger.record("f1_fast_fit", bench="libero_90", tag="read", seed=SEED, metrics=res,
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(recipe=RECIPE),
                          script="experiments/f1_fast_fit.py", command=" ".join(sys.argv)))
    return res


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
