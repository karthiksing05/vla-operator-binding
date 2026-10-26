"""PROBE R6 (EXPLORATORY closed-loop smoke; docs/PLAN_R6.md; ledger m3_prereg tag pr6_smoke, read row r6_read tag smoke).

Does the frozen pi05_l90c@5000, guided in its own flow sampler during the CARRY phase toward a vision-anchored goal through D5's
extracted release operator (and, in the `both` arm, during the approach as in R5s), place the object where the instruction says on
the recombination dev tasks where zs makes wrong-goal errors, without hurting tasks zs solves?  Serving: experiments/r6_guide.py
(serve_guided.py mode r6:, opt-in) and libero_client.py (opt-in R5S_GEOM=1 live-depth token points; R6_TGT=<region> the goal's
simulator position, read by the oracle arm and for logging the anchor's error).

Stages (each a SLURM job; nothing runs on the login node):
  --prep          (chunks env)  task selection (pre-declared rules), the D5 release bank -> results/r6/plan.json, bank.json, specs/
  --launch-smoke  (chunks env)  the serving smoke pack -> results/r6/packs/submit_r6_smoke.sh
  --smoke-check   (chunks env)  unguided serving (MODE ours, r6 arm=off with both flags) == D4's zs actions
  --launch        (chunks env)  the rollout packs -> results/r6/packs/submit_r6_roll.sh
  --type          (LIBERO env)  fm_types' typer on every episode
  --read [--record] (chunks env) success, named-goal placement, failure types, harm, damage, anchor hit, cost -> read.json
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

OUT = os.path.join(ROOT, "results", "r6")
D4D = os.path.join(ROOT, "results", "d4")
D5D = os.path.join(ROOT, "results", "d5")
R5S = os.path.join(ROOT, "results", "r5s")
RROBOT = os.path.join(ROOT, "results", "robot")
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
# ── settings (fixed before any R6 rollout; docs/PLAN_R6.md) ─────────────────────────────────────────────────────────
N_WG, N_HARM = 6, 3
CELLS = ("libero_spatial:6", "libero_90:70")     # R5s's damage cells (fresh_r5s; R5s's zs rows)
SERVE_SEED = 13
ARMS = {"both": "both", "goal": "goal", "orc": "orc"}   # w 1, early schedule (R5s's primary configuration) for every arm
W = 1.0
SMOKE_TRIALS = 2
B_BOOT = 10000
GO_GAIN = 0.10                                   # stage C if the pooled wrong-goal-set success gain of `both` is >= +0.10 or its CI excludes 0


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
    if arm == "zs":
        if tid in CELLS:
            return os.path.join(R5S, "roll", f"r5s_zs_{slug(tid)}.json")
        return os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}.json")
    return os.path.join(OUT, "roll", f"r6_{arm}_{slug(tid)}.json")


def fresh_of(tid):
    return "fresh_r5s" if tid in CELLS else "fresh_d4"


def region_suffix(target, body):
    return target[len(body):].lstrip("_") if target and body and target.startswith(body) else ""


# ── the D5 release bank (chunks env) ────────────────────────────────────────────────────────────────────────────────
def release_bank():
    """Per D5 training carry observation: the target's release operator J (2x2 xy, primary readout) and two offsets, one per target
    frame: b_orc = (e0 - h0) - J (p_region - h0) with p_region the named goal region's site (the body itself for an object target),
    and b_vis = (e0 - h0) - J (c_anchor - h0) with c_anchor the SigLIP target-type anchor's live-depth point.  Nodes: role > category
    > type > type+region; each node stores the mean J and mean offsets of its observations."""
    import d4_bind as D4
    import d5_read as DR5
    ix = {o["key"]: o for o in jload(os.path.join(D5D, "index.json"))["train"]}
    sc = DR5.scale()
    pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
    cats = [str(c) for c in pz["cats"]]
    nodes = {}
    for f in sorted(glob.glob(os.path.join(D5D, "sb", "train", "*.npz"))):
        r = DR5.obs_record(f, "release", sc)
        o = ix[r["key"]]
        tb = o["filler"]
        if tb not in r["J"] or r["e0"] is None:
            continue
        lab = {b["label"]: i for i, b in enumerate(r["meta"]["boxes"])}
        h0 = np.asarray(r["z"]["h0"][lab["base_A1"]], np.float64)
        J = r["J"][tb][:2]
        reg = r["meta"]["regions"].get(o["target"]) or r["meta"]["pos"][tb]
        p = np.asarray(reg, np.float64)
        b_orc = (r["e0"] - h0)[:2] - J @ (p - h0)[:2]
        b_vis = None
        t = o["filler_type"]
        if t in cats and len(r["z"]["sig_tok"]):
            c = np.asarray(r["z"]["tok_xyz"], np.float64).reshape(256, 3)[int(r["z"]["sig_tok"][cats.index(t)])]
            if np.all(np.isfinite(c)):
                b_vis = (r["e0"] - h0)[:2] - J @ (c - h0)[:2]
        cat = D4.CATEGORY.get(t, t)
        rs = region_suffix(o["target"], tb)
        for lev, k in (("role", ("place",)), ("category", ("place", cat)), ("type", ("place", cat, t)),
                       ("region", ("place", cat, t, rs))):
            nodes.setdefault((lev, k), []).append((J, b_orc, b_vis))
    out = []
    for (lev, k), v in nodes.items():
        bv = [x[2] for x in v if x[2] is not None]
        out.append(dict(level=lev, key=list(k), n=len(v), J=np.mean([x[0] for x in v], 0).tolist(),
                        b_orc=np.mean([x[1] for x in v], 0).tolist(), b_vis=(np.mean(bv, 0).tolist() if bv else None), n_vis=len(bv)))
    return out


def lookup(bank, cat, t, rs, frame):
    by = {(b["level"], tuple(b["key"])): b for b in bank}
    for lev, k in (("region", ("place", cat, t, rs)), ("type", ("place", cat, t)), ("category", ("place", cat)), ("role", ("place",))):
        b = by.get((lev, k))
        if b is not None and b[f"b_{frame}"] is not None:
            return b, lev
    return None, None


# ── prep (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────────
def prep():
    import d4_bind as D4
    import m1_compose_create as M1
    tt = M1.task_table()
    d4_dev = D4.dev_tasks()
    recomb = [t for t, b in d4_dev if b != "part"]
    # wrong-goal set: the 6 of D4's 11 recombination In / On tasks with the most R2 zs wrong-goal episodes on fresh_rb2 (ties sha1);
    # harm: the 3 of the rest with the highest R2 zs success (ties sha1).  R2's rows, not D4's, so selection does not condition on
    # the paired baseline
    wg, succ = {}, {}
    for t in recomb:
        eps = jload(os.path.join(ROOT, "results", "rb2", "types", f"{slug(t)}.json"))["arms"]["zs"]["episodes"]
        wg[t] = sum(e["type"] == "wrong_goal" for e in eps)
        succ[t] = sum(bool(e["success"]) for e in eps)
    wg_set = sorted(recomb, key=lambda t: (-wg[t], h40("r6_wg:" + t)))[:N_WG]
    rest = [t for t in recomb if t not in wg_set]
    harm = sorted(rest, key=lambda t: (-succ[t], h40("r6_harm:" + t)))[:N_HARM]
    tasks = wg_set + harm + list(CELLS)
    bank = release_bank()
    dump(os.path.join(OUT, "bank.json"), dict(source="d5_read.obs_record on results/d5/sb/train (release, +-5 cm)", place=bank))
    abank = jload(os.path.join(R5S, "bank.json"))
    aops = {(b["level"], tuple(b["key"])): b for b in abank["approach"]}
    pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
    cats = [str(c) for c in pz["cats"]]
    specs = {}
    for tid in tasks:
        t = tt[tid]
        insts = dict(t.objects, **t.fixtures)
        bodies = [q for q in insts if not q.startswith(D4.TABLES)]
        atoms = [a for a in t.goal if a[0] in ("In", "On") and len(a) == 3]
        named, goal = [], []
        for a in atoms:
            n = a[1]
            ty = D4.type_of(n, insts)
            cat = D4.CATEGORY.get(ty, ty)
            op, lev = None, None
            for lv, k in (("type", ("approach", cat, ty)), ("category", ("approach", cat)), ("role", ("approach",))):
                if (lv, k) in aops:
                    op, lev = aops[(lv, k)], lv
                    break
            named.append(dict(name=n, type=ty, category=cat, level=lev, J=op["J"], b=op["b"], proto=(cats.index(ty) if ty in cats else None)))
            tb = D4.body_of(a[2], bodies)
            gty = insts.get(tb) if tb else None
            gcat = D4.CATEGORY.get(gty, gty)
            rs = region_suffix(a[2], tb)
            for frame in ("vis", "orc"):
                b, lv = lookup(bank, gcat, gty, rs, frame)
                goal.append(dict(frame=frame, region=a[2], body=tb, type=gty, category=gcat, region_suffix=rs, level=lv,
                                 J=(b["J"] if b else None), b=(b[f"b_{frame}"] if b else None),
                                 proto=(cats.index(gty) if gty in cats else None)))
        sp = dict(task=tid, text=t.language, named=named, goal_vis=[g for g in goal if g["frame"] == "vis" and g["J"] is not None],
                  goal_orc=[g for g in goal if g["frame"] == "orc" and g["J"] is not None], tgt=(atoms[0][2] if atoms else None),
                  scale=abank["scale"])
        dump(spec_path(tid), sp)
        specs[tid] = sp
    # pairing checks: D4 / R5s zs rows, settings as served
    pair = {}
    for tid in tasks:
        z = jload(roll_path("zs", tid))
        a = z["args"]
        eps = [e for x in z["tasks"].values() for e in x["episodes"]]
        assert (a["seed"], a["replan"], a["wait"], a["trials"], a["first_trial"]) == (7, 5, 10, 20, 0), (tid, a)
        assert os.path.basename(a["fresh"].rstrip("/")) == fresh_of(tid) and a["bind"] == "track" and len(eps) == 20, (tid, a)
        pair[tid] = dict(fresh=fresh_of(tid), n=len(eps), zs_success=sum(bool(e["success"]) for e in eps))
    res = dict(tasks=tasks, groups=dict(wg=wg_set, harm=harm, cell=list(CELLS)), r2_zs_wrong_goal=wg, r2_zs_success=succ,
               rules=dict(wg="top 6 R2 zs wrong-goal count (fresh_rb2) among D4's 11 recombination In/On tasks, ties sha1",
                          harm="top 3 R2 zs success among the other 5, ties sha1", cells="R5s's 2 damage cells"),
               pairing=pair, arms=ARMS, w=W,
               levels={t: dict(named=[(q["name"], q["level"]) for q in specs[t]["named"]],
                               goal_vis=[(g["region"], g["type"], g["level"]) for g in specs[t]["goal_vis"]],
                               goal_orc=[(g["region"], g["type"], g["level"]) for g in specs[t]["goal_orc"]]) for t in tasks})
    dump(os.path.join(OUT, "plan.json"), res)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str))
    for b in sorted(bank, key=lambda b: (b["level"], b["key"])):
        print(f"  {b['level']:8} {'/'.join(map(str, b['key'])):55} n {b['n']:3} J {np.round(b['J'], 2).tolist()} b_orc "
              f"{np.round(b['b_orc'], 3).tolist()} b_vis {None if b['b_vis'] is None else np.round(b['b_vis'], 3).tolist()}")


# ── launch (chunks env) ─────────────────────────────────────────────────────────────────────────────────────────────
def line(tid, arm, out, trials=20, qlog=None):
    import m1_compose_create as M1
    suite, i = tid.split(":")
    sp = dict(SUITE=suite, TASKS=i, TRIALS=trials, FIRST=0, EXPERIMENT=f"r6_{arm}", TAG=arm, PROGRAM_BYTES=0, OUT=out,
              BACKBONE_CONFIG=M1.CFG, BACKBONE_CKPT=M1.CKPT, BIND="track", RECORD="1", LIGHT="1", OBJECTS="1", ALL_OBJECTS="1",
              SERVE_SEED=str(SERVE_SEED), FRESH=os.path.join(RROBOT, fresh_of(tid)))
    if arm == "smoke_ours":
        sp["MODE"] = "ours"
    else:
        a = {"smoke_off": "off", "smoke_both": "both", "smoke_orc": "orc"}.get(arm, ARMS.get(arm))
        sp["MODE"] = f"r6:{spec_path(tid)}?arm={a}&w={W:g}&log={qlog}"
        sp["R5S_GEOM"] = "1"
        sp["R6_TGT"] = jload(spec_path(tid))["tgt"]
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
                    f"--exclude={EXCLUDE} --job-name={name} -o slurm/{name}_{j:03d}_%j.out -e slurm/{name}_{j:03d}_%j.out "
                    f"--export=ALL,SPECS={p} experiments/roll_pack.sbatch")
        print(f"pack {j}: " + ", ".join(f"{s['TAG']}@{s['TASKS']}" for s in lines[k:k + per]))
    sh = os.path.join(OUT, "packs", f"submit_{name}.sh")
    with open(sh, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\ncd " + ROOT + "\n" + "\n".join(cmds) + "\n")
    print(f"{len(lines)} instances, {len(cmds)} packs -> bash {sh}")


def smoke_task():
    return plan()["groups"]["wg"][0]


def launch_smoke():
    tid = smoke_task()
    lines = [line(tid, arm, os.path.join(OUT, "smoke", f"{arm}_{slug(tid)}.json"), trials=SMOKE_TRIALS,
                  qlog=os.path.join(OUT, "smoke", f"{arm}_{slug(tid)}.jsonl")) for arm in ("smoke_ours", "smoke_off", "smoke_both", "smoke_orc")]
    write_packs(lines, "r6_smoke", hours="01:00:00")


def smoke_check():
    tid = smoke_task()
    i = tid.split(":")[1]
    res = {}
    for arm in ("smoke_ours", "smoke_off", "smoke_both", "smoke_orc"):
        f = os.path.join(OUT, "smoke", f"{arm}_{slug(tid)}.json")
        z = jload(f)
        eps = [e for x in z["tasks"].values() for e in x["episodes"]]
        per = []
        for e in eps:
            k = int(e["trial"])
            a = np.load(os.path.join(f[:-5] + "_rec", f"task{i}_trial{k}.npz"))["action"]
            b = np.load(os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}_rec", f"task{i}_trial{k}.npz"))["action"]
            n = min(len(a), len(b))
            per.append(dict(trial=k, exact=bool(len(a) == len(b) and np.array_equal(a, b)), max_abs=float(np.abs(a[:n] - b[:n]).max()),
                            success=bool(e["success"])))
        q = os.path.join(OUT, "smoke", f"{arm}_{slug(tid)}.jsonl")
        qs = [json.loads(x) for x in open(q, encoding="utf-8")] if os.path.exists(q) else []
        res[arm] = dict(episodes=per, queries=len(qs), guided=sum(x.get("guided", False) for x in qs),
                        guided_by_phase={p: sum(1 for x in qs if x.get("guided") and x.get("phase") == p) for p in ("approach", "carry")},
                        reasons=sorted({x.get("reason") for x in qs if x.get("reason")}))
        print(arm, json.dumps(res[arm]), flush=True)
    res["reproduces"] = {a: all(x["exact"] for x in res[a]["episodes"]) for a in ("smoke_ours", "smoke_off")}
    print("reproduces", res["reproduces"], flush=True)
    dump(os.path.join(OUT, "smoke_check.json"), res)


def launch():
    assert jload(os.path.join(OUT, "smoke_check.json"))["reproduces"] == {"smoke_ours": True, "smoke_off": True}
    lines = []
    for tid in plan()["tasks"]:
        for arm in ARMS:
            out = roll_path(arm, tid)
            if not os.path.exists(out):
                lines.append(line(tid, arm, out, qlog=os.path.join(OUT, "qlog", f"{arm}_{slug(tid)}.jsonl")))
    write_packs(lines, "r6_roll")


# ── typing (LIBERO env) ─────────────────────────────────────────────────────────────────────────────────────────────
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
            if len(eps) < 20:
                print(f"[r6 type] {tid} {arm}: only {len(eps)} episodes, skipped", flush=True)
                continue
            if sc is None:
                sc = _scene(FT, tid)
            typed = [FT.type_episode(sc, None, e, FT.load_rec(f[:-5] + "_rec", tid, int(e["trial"]))) for e in eps]
            dump(outp, dict(task=tid, arm=arm, roll=os.path.relpath(f, ROOT), episodes=typed))
            print(f"[r6 type] {tid} {arm}: " + " ".join(f"{x['trial']}:{x['type']}" for x in typed), flush=True)
        if sc is not None:
            sc.env.close()


def _scene(FT, tid):
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


# ── read (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────────
def boot_paired(per_task, B=B_BOOT, seed=0):
    import r5s_smoke as R5
    return R5.boot_paired(per_task, B=B, seed=seed)


def pack_hours():
    import datetime as dtm
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "slurm", "r6_*_*.out"))):
        txt = open(f, encoding="utf-8", errors="replace").read()
        pat = r"\w{3} (\w{3}) +(\d+) ([\d:]+ [AP]M) \w+ (\d{4})"
        st = [" ".join(m) for m in re.findall(r"\[pack\] " + pat + " on", txt)]
        en = [" ".join(m) for m in re.findall(r"exit=\d+ \| " + pat, txt)]
        if not st or not en:
            continue
        fmt = "%b %d %I:%M:%S %p %Y"
        try:
            out[os.path.basename(f)] = sum((dtm.datetime.strptime(e, fmt) - dtm.datetime.strptime(s, fmt)).total_seconds()
                                           for s, e in zip(st, en)) / 3600.0
        except ValueError:
            continue
    return out


def read(record):
    p = plan()
    T = p["tasks"]
    arms = ["zs"] + list(ARMS)
    ty = {}
    for t in T:
        for arm in arms:
            f = os.path.join(OUT, "types", f"{arm}_{slug(t)}.json")
            if os.path.exists(f):
                ty[(t, arm)] = {int(e["trial"]): e for e in jload(f)["episodes"]}
    # zs re-typed from D4's recordings must equal D4's own types (the 9 dev tasks)
    mism = []
    for t in T:
        if t in CELLS or (t, "zs") not in ty:
            continue
        for e in jload(os.path.join(D4D, "dev", "index", f"{slug(t)}.json"))["episodes"]:
            m = ty[(t, "zs")][int(e["trial"])]
            if (m["type"], m["sub"]) != (e["type"], e["sub"]):
                mism.append((t, e["trial"], e["type"], m["type"]))

    def own_goal(e):                 # the named object's own goal atom holds at the final pose (success's placement part)
        v = e.get("own_atoms_final") or {}
        return float(bool(v) and all(bool(x) for x in v.values()))
    feats = dict(success=lambda e: float(e["success"]), goal_placed=own_goal, wrong_goal=lambda e: float(e["type"] == "wrong_goal"),
                 binding_error=lambda e: float(e["type"] in ("wrong_goal", "wrong_object")))
    per_task = {}
    for t in T:
        per_task[t] = {}
        for arm in arms:
            if (t, arm) not in ty:
                continue
            E = list(ty[(t, arm)].values())
            cnt = {}
            for e in E:
                if e["type"] != "success":
                    k = e["type"] + ("/" + e["sub"] if e.get("sub") else "")
                    cnt[k] = cnt.get(k, 0) + 1
            per_task[t][arm] = dict(n=len(E), **{f: int(sum(fn(e) for e in E)) for f, fn in feats.items()}, types=cnt)
    groups = {}
    for g, tl in p["groups"].items():
        G = {}
        for arm in arms:
            have = [t for t in tl if (t, arm) in ty]
            if not have:
                continue
            G[arm] = {fn: boot_paired({t: [f(e) for e in ty[(t, arm)].values()] for t in have}) for fn, f in feats.items()}
            G[arm]["types"] = {}
            for t in have:
                for k_, v in per_task[t][arm]["types"].items():
                    G[arm]["types"][k_] = G[arm]["types"].get(k_, 0) + v
            if arm != "zs":
                pt = [t for t in have if (t, "zs") in ty]
                G[arm]["diff_vs_zs"] = {fn: boot_paired({t: [f(ty[(t, arm)][k]) - f(ty[(t, "zs")][k]) for k in sorted(set(ty[(t, arm)]) & set(ty[(t, "zs")]))]
                                                         for t in pt}) for fn, f in feats.items()}
                won = sum(1 for t in pt for k in ty[(t, "zs")] if not ty[(t, "zs")][k]["success"] and ty[(t, arm)][k]["success"])
                lost = sum(1 for t in pt for k in ty[(t, "zs")] if ty[(t, "zs")][k]["success"] and not ty[(t, arm)][k]["success"])
                G[arm]["discordant"] = [won, lost]
        groups[g] = G
    qs = {}
    for t in T:
        for arm in ARMS:
            f = os.path.join(OUT, "qlog", f"{arm}_{slug(t)}.jsonl")
            if not os.path.exists(f):
                continue
            Q = [json.loads(x) for x in open(f, encoding="utf-8")]
            g = [q for q in Q if q.get("guided")]
            byep = {}
            for q in g:
                if q.get("phase") == "carry" and "hit" in q:
                    byep.setdefault(q["ep"], []).append(float(q["hit"]))
            qs.setdefault(arm, {})[t] = dict(queries=len(Q), guided_approach=sum(q.get("phase") == "approach" for q in g),
                                             guided_carry=sum(q.get("phase") == "carry" for q in g),
                                             carry_queries=sum(q.get("phase") == "carry" for q in Q),
                                             fallback_carry=sum(1 for q in Q if q.get("phase") == "carry" and not q.get("guided")
                                                                and arm in ("both", "goal", "orc")),
                                             levels={lv: sum(q.get("level") == lv for q in g if q.get("phase") == "carry")
                                                     for lv in ("region", "type", "category", "role")},
                                             goal_hit_ep=[float(np.mean(v)) for v in byep.values()],
                                             goal_err_median=(float(np.median([q["err"] for q in g if q.get("phase") == "carry" and "err" in q]))
                                                              if any(q.get("phase") == "carry" and "err" in q for q in g) else None),
                                             ms_guided=float(np.mean([q["ms"] for q in g])) if g else None)
    loc = {arm: {g: boot_paired({t: qs[arm][t]["goal_hit_ep"] for t in tl if t in qs.get(arm, {})}) for g, tl in p["groups"].items()}
           for arm in ("both", "goal") if arm in qs}
    ph = pack_hours()
    res = dict(plan={k: p[k] for k in ("tasks", "groups", "arms")}, zs_retype_mismatch=mism, per_task=per_task, groups=groups,
               guidance=qs, goal_anchor_hit_5cm=loc, cost=dict(packs=ph, a40_h=float(sum(ph.values()))))
    d = groups.get("wg", {}).get("both", {}).get("diff_vs_zs", {}).get("success")
    if d:
        res["decision"] = dict(gain=d["mean"], lo=d["lo"], hi=d["hi"], stage_c=bool(d["mean"] >= GO_GAIN or d["lo"] > 0))
    dump(os.path.join(OUT, "read.json"), res)
    print(json.dumps(dict(zs_retype_mismatch=mism, per_task=per_task), indent=1))
    for g, G in groups.items():
        for arm, v in G.items():
            print(f"[{g}] {arm}: " + " ".join(f"{k} {v[k]['mean']:.3f} [{v[k]['lo']:.3f}, {v[k]['hi']:.3f}]" for k in feats)
                  + f" types {v['types']}" + ("" if arm == "zs" else " | diff " + " ".join(
                      f"{k} {x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for k, x in v["diff_vs_zs"].items()) + f" disc {v['discordant']}"),
                  flush=True)
    print("guidance:", json.dumps({a: {t: {k: x for k, x in v.items() if k != "goal_hit_ep"} for t, v in d_.items()} for a, d_ in qs.items()}, indent=1))
    print("goal anchor hit:", json.dumps(loc, indent=1), "cost:", json.dumps(res["cost"]), "decision:", json.dumps(res.get("decision")), flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r6_read", bench="libero_90_c", tag="smoke", seed=0,
                            metrics=dict(exploratory=True, groups=groups, per_task=per_task, guidance=qs, goal_anchor_hit_5cm=loc,
                                         cost=res["cost"], decision=res.get("decision"), zs_retype_mismatch=mism),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(arms=ARMS, w=W),
                            script="experiments/r6_smoke.py", command="python experiments/r6_smoke.py --read --record",
                            notes="EXPLORATORY closed-loop smoke (docs/PLAN_R6.md); no confirmatory claim"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("prep", "launch-smoke", "smoke-check", "launch", "type", "read", "record"):
        ap.add_argument("--" + s, action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.prep:
        prep()
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
