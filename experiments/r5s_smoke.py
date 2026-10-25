"""PROBE R5s (EXPLORATORY closed-loop smoke; docs/PLAN_R5S.md; ledger m3_prereg tag pr5s_smoke, read row r5s_read tag smoke).

Does the frozen pi05_l90c@5000, guided early in its own flow sampler toward a vision-anchored target via D4's extracted approach
operator (results/d4: the bank, J per type > category > role), grasp the NAMED object on the part dev tasks where zs grasps the
wrong one, without hurting tasks zs solves?  Serving: experiments/r5s_guide.py (serve_guided.py mode r5s:, opt-in) and
libero_client.py (opt-in R5S_GEOM=1: the live agentview depth's per-token 3D points).

Stages (each a SLURM job; nothing runs on the login node):
  --prep                 (chunks env)   task selection (pre-declared rules), D4 bank rebuilt -> bank.json, per-task specs, D4 pairing
  --fresh-check          (chunks env)   fresh_r5s (training cells) vs every earlier fresh set (> 1e-6)
  --launch-smoke         (chunks env)   the 2-episode serving smoke pack -> results/r5s/packs/submit_smoke.sh
  --smoke-check          (chunks env)   unguided serving (MODE ours and r5s arm=off with R5S_GEOM=1) == D4's zs actions
  --launch               (chunks env)   the rollout packs -> results/r5s/packs/submit.sh
  --type [--shard i/n]   (LIBERO env)   fm_types' typer on every episode (zs re-typed from D4's recordings too)
  --read [--record]      (chunks env)   success, named grasp, failure types, harm, damage, localisation, cost -> read.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

OUT = os.path.join(ROOT, "results", "r5s")
D4D = os.path.join(ROOT, "results", "d4")
RROBOT = os.path.join(ROOT, "results", "robot")
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
# ── settings (fixed before any R5s rollout was read; docs/PLAN_R5S.md) ─────────────────────────────────────────────
WO_TASKS = ("libero_90:38", "libero_90:58", "libero_90:73", "libero_90:77", "libero_90:79", "libero_90:80", "libero_90:83")
N_HARM, N_CELLS = 4, 2
FRESH_D4 = "fresh_d4"
FRESH_DIR, BASE, N_STATES = "fresh_r5s", 8600000, 20           # training cells only (no D4 zs rows there)
SERVE_SEED = 13
ARMS = {                                                       # arm -> (r5s arm, sched, w)
    "vis_e1": ("vis", "early", 1.0),                           # primary: vision anchor, first 5 Euler steps, w 1
    "vis_a025": ("vis", "all", 0.25),                          # secondary: vision anchor, every step, w 0.25
    "orc_e1": ("orc", "early", 1.0),                           # privileged upper bound: simulator pose, primary config
}
SMOKE_TASK, SMOKE_TRIALS = "libero_90:73", 2
B_BOOT = 10000
# exploratory decision rule (docs/PLAN_R5S.md section 5): a pre-registered confirmatory read is worth running iff, on the 7
# wrong-object tasks, the primary arm (vis_e1) raises success by >= +0.10 (point) AND raises the named-grasp rate (point > 0),
# AND the harm (4 zs-success recombination tasks) and damage (2 training cells) differences are both >= -0.05 (point)
GO_GAIN, GO_HARM = 0.10, -0.05


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def dump(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, p)


def slug(tid):
    return tid.replace(":", "_")


def h40(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % (1 << 40)


def plan():
    return jload(os.path.join(OUT, "plan.json"))


def spec_path(tid):
    return os.path.join(OUT, "specs", f"{slug(tid)}.json")


def roll_path(arm, tid):
    if arm == "zs" and tid in plan()["d4_tasks"]:
        return os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}.json")
    return os.path.join(OUT, "roll", f"r5s_{arm}_{slug(tid)}.json")


def fresh_of(tid):
    return FRESH_D4 if tid in plan()["d4_tasks"] else FRESH_DIR


# ── prep (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────────
def prep():
    import d4_bind as D4
    import d4_read as DR
    import m1_compose_create as M1
    import of_frame as OFm
    import r2a_null_modes as R2A
    tt = M1.task_table()
    d4_dev = D4.dev_tasks()
    d4_tasks = [t for t, _ in d4_dev]
    part = [t for t, b in d4_dev if b == "part"]
    assert sorted(part) == sorted(WO_TASKS), (part, WO_TASKS)
    recomb = [t for t, b in d4_dev if b != "part"]
    # harm check: the 4 recombination D4 tasks with the highest R2 zs success on fresh_rb2 (ties: sha1), i.e. M1/R2's zs rates,
    # not D4's paired rows
    rb2 = jload(os.path.join(ROOT, "results", "rb2", "read.json"))
    r2zs = {t: int(rb2["per_task"][t]["zs"]) for t in recomb}
    harm = sorted(recomb, key=lambda t: (-r2zs[t], h40("r5s_harm:" + t)))[:N_HARM]
    # damage: R2's damage-cell order (rb2 plan.json, sha1-ranked), the first 2 training cells with an In / On goal on a movable object
    p2 = jload(os.path.join(ROOT, "results", "rb2", "plan.json"))
    cells = []
    for c in p2["damage"]:
        tid = c if ":" in c else f"libero_90:{c}"
        t = tt[tid]
        if any(a[0] in ("In", "On") and len(a) == 3 and a[1] in t.objects for a in t.goal):
            cells.append(tid)
        if len(cells) == N_CELLS:
            break
    tasks = list(WO_TASKS) + harm + cells
    # D4's bank, rebuilt with d4_read's own code and checked against read.json
    tix = {x["tid"]: x for x in D4.jload(os.path.join(D4D, "index_train_l90c.json"))["tasks"]}
    recs, scale = DR.load("l90c", "train_l90c")
    summ, ops = DR.bank(recs, lambda r, n: D4.type_of(n, dict(tix[r["tid"]]["objects"], **tix[r["tid"]]["fixtures"])),
                        lambda r, n: f"{tix[r['tid']]['scene']}/{n}")
    ref = jload(os.path.join(D4D, "read.json"))["models"]["l90c"]["bank"]
    assert summ["n_ops"] == ref["n_ops"] and summ["bytes_total"] == ref["bytes_total"], (summ, ref)
    bank = [dict(level=lev, key=list(k), J=np.asarray(op[:6]).reshape(3, 2)[:2].tolist(), b=np.asarray(op[6:8]).tolist())
            for (lev, k), op in ops.items() if k[0] == "approach"]
    dump(os.path.join(OUT, "bank.json"), dict(source="d4_read.bank on results/d4/gpu/l90c_train_l90c", summary=summ, approach=bank,
                                              scale=list(map(float, scale))))
    pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
    cats = [str(c) for c in pz["cats"]]
    specs = {}
    for tid in tasks:
        t = tt[tid]
        named = sorted({a[1] for a in t.goal if a[0] in ("In", "On") and len(a) == 3})
        ent = []
        for n in named:
            ty = D4.type_of(n, dict(t.objects, **t.fixtures))
            cat = D4.CATEGORY.get(ty, ty)
            op, lev = None, None
            for lv, k in (("type", ("approach", cat, ty)), ("category", ("approach", cat)), ("role", ("approach",))):
                if (lv, k) in ops:
                    op, lev = ops[(lv, k)], lv
                    break
            ent.append(dict(name=n, type=ty, category=cat, level=lev, J=np.asarray(op[:6]).reshape(3, 2)[:2].tolist(),
                            b=np.asarray(op[6:8]).tolist(), proto=(cats.index(ty) if ty in cats else None)))
        if tid in d4_tasks:                                      # the parse's named objects = D4 / R2a's pre-grasp `named`
            ix = jload(os.path.join(D4D, "dev", "index", f"{slug(tid)}.json"))
            seen = {tuple(m["named"]) for e in ix["episodes"] for m in e["obs"] if m["role"] == "pre"}
            assert not seen or seen == {tuple(named)}, (tid, seen, named)
        sp = dict(task=tid, text=t.language, named=ent, scale=list(map(float, scale)))
        dump(spec_path(tid), sp)
        specs[tid] = sp
    # D4 pairing: the zs rows' settings
    pair = {}
    tsv = {}
    for f in sorted(glob.glob(os.path.join(D4D, "packs", "d4_zs_*.tsv"))):
        for line in open(f, encoding="utf-8"):
            kv = dict(x.split("=", 1) for x in line.rstrip("\n").split("\t") if "=" in x)
            if kv:
                tsv[f"{kv['SUITE']}:{kv['TASKS']}"] = kv
    for tid in tasks:
        if tid not in d4_tasks:
            continue
        z = jload(os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}.json"))
        a = z["args"]
        eps = [e for x in z["tasks"].values() for e in x["episodes"]]
        rd = os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}_rec")
        kv = tsv[tid]
        pair[tid] = dict(seed=a["seed"], replan=a["replan"], wait=a["wait"], trials=a["trials"], first=a["first_trial"],
                         fresh=os.path.basename(a["fresh"].rstrip("/")), objects=a["objects"], all_objects=a["all_objects"],
                         bind=a["bind"], mode=kv["MODE"], serve_seed=kv["SERVE_SEED"], ckpt=kv["BACKBONE_CKPT"],
                         n_eps=len(eps), n_rec=len(glob.glob(os.path.join(rd, "*.npz"))))
        assert (a["seed"], a["replan"], a["wait"], a["trials"], a["first_trial"]) == (7, 5, 10, 20, 0), (tid, a)
        assert pair[tid]["fresh"] == FRESH_D4 and kv["MODE"] == "ours" and kv["SERVE_SEED"] == str(SERVE_SEED), pair[tid]
        assert kv["BACKBONE_CKPT"] == M1.CKPT and kv["BACKBONE_CONFIG"] == M1.CFG and a["bind"] == "track", kv
        assert pair[tid]["n_eps"] == N_STATES and pair[tid]["n_rec"] == N_STATES, pair[tid]
    res = dict(tasks=tasks, groups=dict(wo=list(WO_TASKS), harm=harm, cell=cells), d4_tasks=d4_tasks,
               r2_zs_recombination=r2zs, r2_damage_order=p2["damage"], cells_rule="first 2 of R2's damage order with an In/On "
               "goal on a movable object", harm_rule="top 4 R2 zs (fresh_rb2) among D4's 11 recombination tasks, ties sha1",
               pairing=pair, scale_d4=list(map(float, scale)), scale_r2a=list(map(float, R2A.SCALE)),
               scale_of=list(map(float, OFm.scale_for("l90c"))), arms=ARMS, fresh=dict(dir=FRESH_DIR, base=BASE, n=N_STATES),
               levels={t: [(q["name"], q["type"], q["level"], q["proto"]) for q in specs[t]["named"]] for t in tasks})
    dump(os.path.join(OUT, "plan.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairing",)}, indent=1, default=str))
    print("pairing:", json.dumps(pair))
    print("bank:", json.dumps(summ))
    for b in bank:
        if b["level"] in ("role", "category"):
            print(f"  {b['level']:9} {'/'.join(b['key']):30} J {np.round(b['J'], 3).tolist()} b {np.round(b['b'], 4).tolist()}")


def fresh_check():
    import r2a_null_modes as R2A
    used = tuple(R2A.used_bases()) + (5300000, 6700000, 7100000, 7500000, 8500000)
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in used), BASE
    p = plan()
    out = {}
    for tid in p["groups"]["cell"]:
        suite, i = tid.split(":")
        x = np.load(os.path.join(RROBOT, FRESH_DIR, f"{suite}_{i}.npy"))
        assert len(x) == N_STATES, (tid, x.shape)
        dmin, sets = np.inf, []
        for f in sorted(glob.glob(os.path.join(RROBOT, "fresh_*", f"{suite}_{i}.npy"))):
            if os.path.basename(os.path.dirname(f)) == FRESH_DIR:
                continue
            y = np.load(f)
            if y.shape[1] != x.shape[1]:
                continue
            dmin = min(dmin, float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))))
            sets.append(os.path.basename(os.path.dirname(f)))
        within = float(np.min([np.linalg.norm(x[a] - x[b]) for a in range(len(x)) for b in range(a + 1, len(x))]))
        assert dmin > 1e-6 and within > 1e-6, (tid, dmin, within)
        out[tid] = dict(min_dist=dmin, compared=sets, min_within=within)
        print(tid, f"min distance to earlier sets {dmin:.4f} over {len(sets)} sets; within-set {within:.4f}", flush=True)
    dump(os.path.join(OUT, "fresh_check.json"), out)


# ── launch (chunks env) ─────────────────────────────────────────────────────────────────────────────────────────────
def line(tid, arm, out, trials=N_STATES, qlog=None):
    import m1_compose_create as M1
    suite, i = tid.split(":")
    sp = dict(SUITE=suite, TASKS=i, TRIALS=trials, FIRST=0, EXPERIMENT=f"r5s_{arm}", TAG=arm, PROGRAM_BYTES=0, OUT=out,
              BACKBONE_CONFIG=M1.CFG, BACKBONE_CKPT=M1.CKPT, BIND="track", RECORD="1", LIGHT="1", OBJECTS="1", ALL_OBJECTS="1",
              SERVE_SEED=str(SERVE_SEED), FRESH=os.path.join(RROBOT, fresh_of(tid)))
    if arm in ("zs", "smoke_ours"):
        sp["MODE"] = "ours"
    else:
        a, sc, w = ARMS[arm] if arm in ARMS else {"smoke_off": ("off", "early", 1.0), "smoke_vis": ("vis", "early", 1.0),
                                                  "smoke_orc": ("orc", "early", 1.0)}[arm]
        sp["MODE"] = f"r5s:{spec_path(tid)}?arm={a}&sched={sc}&w={w:g}&log={qlog}"
        sp["R5S_GEOM"] = "1"
    return sp


def write_packs(lines, name, per=4, hours="04:00:00"):
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    cmds = []
    for j, k in enumerate(range(0, len(lines), per)):
        p = os.path.join(OUT, "packs", f"{name}_{j:03d}.tsv")
        with open(p, "w", encoding="utf-8") as f:
            for sp in lines[k:k + per]:
                f.write("\t".join(f"{a}={b}" for a, b in sp.items()) + "\n")
        cmds.append(f"sbatch --parsable -p overcap -A overcap --qos=short --requeue --time={hours} --gres=gpu:a40:1 "
                    f"--exclude={EXCLUDE} --job-name={name} -o slurm/{name}_{j:03d}_%j.out --export=ALL,SPECS={p} "
                    f"experiments/roll_pack.sbatch")
        print(f"pack {j}: " + ", ".join(f"{s['TAG']}@{s['TASKS']}" for s in lines[k:k + per]))
    sh = os.path.join(OUT, "packs", f"submit_{name}.sh")
    with open(sh, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\ncd " + ROOT + "\n" + "\n".join(cmds) + "\n")
    print(f"{len(lines)} instances, {len(cmds)} packs -> bash {sh}")


def launch_smoke():
    lines = []
    for arm in ("smoke_ours", "smoke_off", "smoke_vis", "smoke_orc"):
        lines.append(line(SMOKE_TASK, arm, os.path.join(OUT, "smoke", f"{arm}_{slug(SMOKE_TASK)}.json"), trials=SMOKE_TRIALS,
                          qlog=os.path.join(OUT, "smoke", f"{arm}_{slug(SMOKE_TASK)}.jsonl")))
    write_packs(lines, "r5s_smoke", hours="01:00:00")


def launch():
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    assert os.path.exists(os.path.join(OUT, "smoke_check.json")), "run the serving smoke first"
    p = plan()
    lines = []
    for tid in p["tasks"]:
        for arm in (["zs"] if tid not in p["d4_tasks"] else []) + list(ARMS):
            out = roll_path(arm, tid)
            if os.path.exists(out):
                continue
            lines.append(line(tid, arm, out, qlog=os.path.join(OUT, "qlog", f"{arm}_{slug(tid)}.jsonl")))
    write_packs(lines, "r5s_roll")


def smoke_check():
    tid = SMOKE_TASK
    i = tid.split(":")[1]
    res = {}
    for arm in ("smoke_ours", "smoke_off", "smoke_vis", "smoke_orc"):
        f = os.path.join(OUT, "smoke", f"{arm}_{slug(tid)}.json")
        z = jload(f)
        eps = [e for x in z["tasks"].values() for e in x["episodes"]]
        rd = f[:-5] + "_rec"
        per = []
        for e in eps:
            k = int(e["trial"])
            a = np.load(os.path.join(rd, f"task{i}_trial{k}.npz"))["action"]
            b = np.load(os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}_rec", f"task{i}_trial{k}.npz"))["action"]
            n = min(len(a), len(b))
            per.append(dict(trial=k, len=len(a), len_d4=len(b), exact=bool(len(a) == len(b) and np.array_equal(a, b)),
                            max_abs=float(np.abs(a[:n] - b[:n]).max()) if n else None,
                            first_diff=int(np.argmax(np.any(a[:n] != b[:n], 1))) if n and not np.array_equal(a[:n], b[:n]) else None,
                            success=bool(e["success"]), bound=e.get("bound")))
        q = os.path.join(OUT, "smoke", f"{arm}_{slug(tid)}.jsonl")
        qs = [json.loads(x) for x in open(q, encoding="utf-8")] if os.path.exists(q) else []
        res[arm] = dict(episodes=per, queries=len(qs), guided=sum(x.get("guided", False) for x in qs),
                        ms_mean=float(np.mean([x["ms"] for x in qs])) if qs else None,
                        hits=[x.get("hit") for x in qs if x.get("guided")][:20])
        print(arm, json.dumps(res[arm]), flush=True)
    d4 = [e for x in jload(os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}.json"))["tasks"].values() for e in x["episodes"]][:SMOKE_TRIALS]
    res["d4"] = [dict(trial=e["trial"], success=e["success"], bound=e.get("bound")) for e in d4]
    res["reproduces"] = {a: all(x["exact"] for x in res[a]["episodes"]) for a in ("smoke_ours", "smoke_off")}
    print("d4", json.dumps(res["d4"]), "reproduces", res["reproduces"], flush=True)
    dump(os.path.join(OUT, "smoke_check.json"), res)


# ── typing (LIBERO env) ─────────────────────────────────────────────────────────────────────────────────────────────
def scene(FT, tid):
    """fm_types.Scene with the episode start states of this task's own fresh set (fresh_d4, or fresh_r5s for training cells)."""
    import pathlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    sc = FT.Scene.__new__(FT.Scene)
    suite, i = tid.split(":")
    task = benchmark.get_benchmark_dict()[suite]().get_task(int(i))
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    sc.env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    sc.env.seed(0)
    sc.env.reset()
    sc.e = sc.env.env
    sc.fresh = np.load(os.path.join(RROBOT, fresh_of(tid), f"{suite}_{i}.npy"))
    sc.goal = [list(g) for g in sc.e.parsed_problem["goal_state"]]
    return sc


def type_all(shard):
    import fm_types as FT
    p = plan()
    i, n = map(int, shard.split("/"))
    od = os.path.join(OUT, "types")
    os.makedirs(od, exist_ok=True)
    for j, tid in enumerate(p["tasks"]):
        if j % n != i:
            continue
        sc = None
        for arm in ["zs"] + list(ARMS):
            f = roll_path(arm, tid)
            outp = os.path.join(od, f"{arm}_{slug(tid)}.json")
            if not os.path.exists(f) or os.path.exists(outp):
                continue
            z = jload(f)
            eps = [e for x in z["tasks"].values() for e in x["episodes"]]
            if len(eps) < N_STATES:
                print(f"[r5s type] {tid} {arm}: only {len(eps)} episodes, skipped", flush=True)
                continue
            if sc is None:
                sc = scene(FT, tid)
            rd = f[:-5] + "_rec"
            typed = [FT.type_episode(sc, None, e, FT.load_rec(rd, tid, int(e["trial"]))) for e in eps]
            dump(outp, dict(task=tid, arm=arm, roll=os.path.relpath(f, ROOT), episodes=typed))
            print(f"[r5s type] {tid} {arm}: " + " ".join(f"{x['trial']}:{x['type']}" for x in typed), flush=True)
        if sc is not None:
            sc.env.close()


# ── read (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────────
def boot_paired(per_task, B=B_BOOT, seed=0):
    """Two-level task-cluster bootstrap of a pooled mean of per-episode values (paired differences): resample tasks, then episodes
    within each task.  -> dict(mean, lo, hi, n, tasks)."""
    xs = [np.asarray(v, float) for v in per_task.values() if len(v)]
    if not xs:
        return dict(mean=None, lo=None, hi=None, n=0, tasks=0)
    rng = np.random.default_rng(seed)
    s = np.empty(B)
    for b in range(B):
        num, den = 0.0, 0
        for j in rng.integers(0, len(xs), len(xs)):
            m = len(xs[j])
            num += xs[j][rng.integers(0, m, m)].sum()
            den += m
        s[b] = num / den
    allx = np.concatenate(xs)
    return dict(mean=float(allx.mean()), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), n=int(len(allx)),
                tasks=len(xs))


def pack_hours():
    """A40-hours of every R5s pack from its log (start line to exit line)."""
    import datetime as dtm
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "slurm", "r5s_*_*.out"))):
        txt = open(f, encoding="utf-8", errors="replace").read()
        pat = r"\w{3} (\w{3}) +(\d+) ([\d:]+ [AP]M) \w+ (\d{4})"
        st = [" ".join(m) for m in re.findall(r"\[pack\] " + pat + " on", txt)]
        en = [" ".join(m) for m in re.findall(r"exit=\d+ \| " + pat, txt)]
        if not st or not en:
            continue
        fmt = "%b %d %I:%M:%S %p %Y"
        try:
            secs = sum((dtm.datetime.strptime(e, fmt) - dtm.datetime.strptime(s, fmt)).total_seconds() for s, e in zip(st, en))
        except ValueError:
            continue
        out[os.path.basename(f)] = secs / 3600.0
    return out


def read(record):
    p = plan()
    T = p["tasks"]
    named = {t: {q["name"] for q in jload(spec_path(t))["named"]} for t in T}
    ty = {}
    for t in T:
        for arm in ["zs"] + list(ARMS):
            f = os.path.join(OUT, "types", f"{arm}_{slug(t)}.json")
            if os.path.exists(f):
                ty[(t, arm)] = {int(e["trial"]): e for e in jload(f)["episodes"]}
    # zs re-typed from D4's recordings must equal D4's own types
    mism = []
    for t in T:
        if t in p["d4_tasks"] and (t, "zs") in ty:
            ix = jload(os.path.join(D4D, "dev", "index", f"{slug(t)}.json"))
            for e in ix["episodes"]:
                mine = ty[(t, "zs")][int(e["trial"])]
                if (mine["type"], mine["sub"]) != (e["type"], e["sub"]):
                    mism.append((t, e["trial"], e["type"], mine["type"]))
    arms = ["zs"] + list(ARMS)
    res = dict(plan={k: p[k] for k in ("tasks", "groups", "r2_zs_recombination", "arms")}, zs_retype_mismatch=mism)
    per_task = {}
    for t in T:
        per_task[t] = {}
        for arm in arms:
            if (t, arm) not in ty:
                continue
            E = [ty[(t, arm)][k] for k in sorted(ty[(t, arm)])]
            cnt = {}
            for e in E:
                if e["type"] != "success":
                    cnt[e["type"] + ("/" + e["sub"] if e.get("sub") else "")] = cnt.get(e["type"] + ("/" + e["sub"] if e.get("sub") else ""), 0) + 1
            per_task[t][arm] = dict(n=len(E), success=sum(e["success"] for e in E),
                                    named_grasp=sum(e.get("bound") in named[t] for e in E),
                                    any_grasp=sum(e.get("bound") is not None for e in E),
                                    wrong_object=sum(e["type"] == "wrong_object" for e in E), types=cnt)
    res["per_task"] = per_task

    def ep(t, arm, f):
        E = ty[(t, arm)]
        return {k: f(E[k], t) for k in E}
    feats = dict(success=lambda e, t: float(e["success"]), named_grasp=lambda e, t: float(e.get("bound") in named[t]),
                 wrong_object=lambda e, t: float(e["type"] == "wrong_object"))
    groups = {}
    for g, tl in p["groups"].items():
        G = {}
        for arm in arms:
            have = [t for t in tl if (t, arm) in ty]
            if not have:
                continue
            G[arm] = {fn: boot_paired({t: list(ep(t, arm, f).values()) for t in have}) for fn, f in feats.items()}
            G[arm]["types"] = {}
            for t in have:
                for k_, v in per_task[t][arm]["types"].items():
                    G[arm]["types"][k_] = G[arm]["types"].get(k_, 0) + v
            if arm != "zs":
                pt = [t for t in have if (t, "zs") in ty]
                G[arm]["diff_vs_zs"] = {}
                for fn, f in feats.items():
                    d = {}
                    for t in pt:
                        a_, z_ = ep(t, arm, f), ep(t, "zs", f)
                        d[t] = [a_[k] - z_[k] for k in sorted(set(a_) & set(z_))]
                    G[arm]["diff_vs_zs"][fn] = boot_paired(d)
                # harm on the episodes zs solves: the share of zs successes this arm fails (and the reverse)
                lost = {t: [float(not ty[(t, arm)][k]["success"]) for k in ty[(t, "zs")] if ty[(t, "zs")][k]["success"] and k in ty[(t, arm)]]
                        for t in pt}
                won = {t: [float(ty[(t, arm)][k]["success"]) for k in ty[(t, "zs")] if not ty[(t, "zs")][k]["success"] and k in ty[(t, arm)]]
                       for t in pt}
                G[arm]["lost_of_zs_successes"] = boot_paired(lost)
                G[arm]["won_of_zs_failures"] = boot_paired(won)
                G[arm]["discordant"] = [int(sum(sum(1 for x in v if x > 0) for v in won.values())),
                                        int(sum(sum(1 for x in v if x > 0) for v in lost.values()))]
        groups[g] = G
    res["groups"] = groups
    # guidance logs: the share of queries guided, the anchor's localisation (vis arms)
    qs = {}
    for t in T:
        for arm in ARMS:
            f = os.path.join(OUT, "qlog", f"{arm}_{slug(t)}.jsonl")
            if not os.path.exists(f):
                continue
            Q = [json.loads(x) for x in open(f, encoding="utf-8")]
            g = [q for q in Q if q.get("guided")]
            ph = [q for q in Q if q.get("phase_open")]
            byep = {}
            for q in g:
                if "hit" in q:
                    byep.setdefault(q["ep"], []).append(float(q["hit"]))
            qs.setdefault(arm, {})[t] = dict(queries=len(Q), open_pre_grasp=len(ph), guided=len(g),
                                             fallback=sum(1 for q in ph if not q.get("guided")),
                                             guided_per_ep=len(g) / max(1, len({q["ep"] for q in Q})),
                                             levels={lv: sum(q.get("level") == lv for q in g) for lv in ("type", "category", "role")},
                                             hit_ep=[float(np.mean(v)) for v in byep.values()],
                                             err_median=float(np.median([q["err"] for q in g if "err" in q])) if any("err" in q for q in g) else None,
                                             resid_median=float(np.median([np.linalg.norm(np.subtract(q["D"], q["dstar"])) for q in g])) if g else None,
                                             ms_guided=float(np.mean([q["ms"] for q in g])) if g else None)
    res["guidance"] = qs
    loc = {}
    for arm in ("vis_e1", "vis_a025"):
        if arm not in qs:
            continue
        loc[arm] = {g: boot_paired({t: qs[arm][t]["hit_ep"] for t in tl if t in qs[arm]}) for g, tl in p["groups"].items()}
        loc[arm]["all"] = boot_paired({t: v["hit_ep"] for t, v in qs[arm].items()})
    res["localisation"] = loc
    ph = pack_hours()
    res["cost"] = dict(packs=ph, a40_h=float(sum(ph.values())))
    # the exploratory decision rule (pre-declared)
    wo = groups.get("wo", {}).get("vis_e1", {}).get("diff_vs_zs", {})
    hm = groups.get("harm", {}).get("vis_e1", {}).get("diff_vs_zs", {})
    dm = groups.get("cell", {}).get("vis_e1", {}).get("diff_vs_zs", {})
    if wo and hm and dm:
        crit = dict(gain=wo["success"]["mean"] >= GO_GAIN, named=wo["named_grasp"]["mean"] > 0,
                    harm=hm["success"]["mean"] >= GO_HARM, damage=dm["success"]["mean"] >= GO_HARM)
        res["decision"] = dict(criteria=crit, confirmatory_warranted=bool(all(crit.values())))
    dump(os.path.join(OUT, "read.json"), res)
    print(json.dumps(dict(zs_retype_mismatch=mism, per_task=per_task), indent=1))
    for g, G in groups.items():
        for arm, v in G.items():
            print(f"[{g}] {arm}: success {v['success']['mean']:.3f} [{v['success']['lo']:.3f}, {v['success']['hi']:.3f}] named "
                  f"{v['named_grasp']['mean']:.3f} wrong_obj {v['wrong_object']['mean']:.3f} types {v['types']}"
                  + ("" if arm == "zs" else " | diff " + " ".join(f"{k} {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
                                                                  for k, d in v["diff_vs_zs"].items())
                     + f" | lost/zs-succ {v['lost_of_zs_successes']['mean']} won/zs-fail {v['won_of_zs_failures']['mean']} disc {v['discordant']}"),
                  flush=True)
    print("guidance:", json.dumps({a: {t: {k: x for k, x in v.items() if k != "hit_ep"} for t, v in d.items()} for a, d in qs.items()}, indent=1))
    print("localisation:", json.dumps(loc, indent=1))
    print("cost:", json.dumps(res["cost"]), "decision:", json.dumps(res.get("decision")), flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r5s_read", bench="libero_90_c", tag="smoke", seed=0,
                            metrics=dict(exploratory=True, groups=groups, per_task=per_task, localisation=loc, guidance=qs,
                                         cost=res["cost"], decision=res.get("decision"), zs_retype_mismatch=mism),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(arms=ARMS),
                            script="experiments/r5s_smoke.py", command="python experiments/r5s_smoke.py --read --record",
                            notes="EXPLORATORY closed-loop smoke (docs/PLAN_R5S.md); no confirmatory claim"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("prep", "fresh-check", "launch-smoke", "smoke-check", "launch", "type", "read", "record"):
        ap.add_argument("--" + s, action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.prep:
        prep()
    if a.fresh_check:
        fresh_check()
    if a.launch_smoke:
        launch_smoke()
    if a.smoke_check:
        smoke_check()
    if a.launch:
        launch()
    if a.type:
        type_all(a.shard)
    if a.read:
        read(a.record)


if __name__ == "__main__":
    main()
