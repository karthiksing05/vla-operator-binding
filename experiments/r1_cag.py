"""PROBE R1 of docs/PLAN_R.md (design + result: docs/PLAN_R1.md): training-free CAG-VA and the empty-instruction probe on
pi05_l90c@5000, LIBERO-90-C's 29 dev tasks (results/m1/plan.json), frozen policy, no demo.

CAG-VA (Fang et al. 2602.17659, eq. 4, training-free variant): at every flow-integration step
    v = v(o, empty) + w (v(o, instr) - v(o, empty)),
the same VLA with the language dropped as the unconditional.  Served by the EXISTING cfg control of serve_guided.py's factor mode
(factor:<task>?ctrl=cfg&w=<w>: lifelong_ood/policy/factor_hooks.py, Composition([instr, ""], [w, 1 - w]) through
conditionals.sample_composed, i.e. the affine velocity mix at every Euler step).  w = 1 is zs (served as MODE ours).
The empty instruction alone is promptx: (baselines.PromptHooks("")).

Sets (every rollout records as M1's did: BIND=track RECORD=1 LIGHT=1 OBJECTS=1 ALL_OBJECTS=1):
  dev    w-SELECTION states: NEW set results/robot/fresh_r1 (LIBERO's sampler, seed BASE 6700000 + k, 20 per task), noise seed 13;
         arms zs (w 1), cag1.5 / cag2 / cag3 / cag5.  w* = argmax successes on the 15 recombination tasks, ties to the smaller w.
  test   M1's fresh_m1 states (BASE 5700000), noise seed 13, paired with M1's zs rollouts (results/m1/roll, not re-rolled);
         arms cag at every w (the headline is w*, chosen on dev; the other w are descriptive) and empty.
  cells  M1's 10 damage cells (training cells), stored init states 0-19, noise seed 0, paired with the parity read of
         pi05_l90c@5000 (results/l90c/parity_*); arms cag at every w (damage; headline w*) and empty (the vision-shortcut rate).

"Which demonstrated task did it do instead" (--instead, LIBERO's venv): fm_types' reconstruction (start state, every movable object
at its last recorded pose, 10 control steps of physics) and, at that settled final state, the In / On goal atoms of every TRAINING
task of the same scene (results/l90c/split.json "train"; their BDDL goals via factors.py).  Articulated atoms (Open / Close /
Turnon / Turnoff) are not recorded, so a training task's PLACEMENT goal (its In / On atoms) is what is read; articulation-only
training tasks are not evaluable.  An episode completes a demonstrated task c when every placement atom of c holds at the settled
final state, does not all hold at the settled first record, and c's placement atoms are not a subset of the episode's own task's
goal atoms.  Shares the object / target: some atom of c has the instructed task's named object as its first / its target as its
second argument.

    sbatch ... fm_cpu.sbatch SCRIPT="experiments/r1_cag.py --plan"          design table -> results/r1/plan.json
    sbatch -p overcap ... --export=ALL,TASKS=..,N=20,BASE=6700000,FRESH_DIR=fresh_r1 experiments/fresh_states.sbatch
    sbatch ... fm_cpu.sbatch SCRIPT="experiments/r1_cag.py --fresh-check"
    (login shell) ... python experiments/r1_cag.py --launch [--dry]          is run inside a CPU job; it only calls sbatch
    sbatch ... fm_cpu.sbatch SCRIPT="experiments/r1_cag.py --instead --shard 0/4"
    sbatch ... --wrap "$PY experiments/r1_cag.py --read --record"
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1                                          # noqa: E402

R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "r1")
SMOKE = os.environ.get("R1_SMOKE") == "1"                              # serving + reconstruction smoke: results/r1/smoke, no ledger
DEV_DIR, DEV_BASE, N_STATES = "fresh_r1", 6700000, 20
SEED = 13
OMEGAS = (1.5, 2.0, 3.0, 5.0)                                           # w = 1 is zs
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
SETTLE = 10
H2_SHARE = 0.25                                                         # pre-registered threshold (experiments/prereg/pr1_cag.py)
DAMAGE_GATE = -0.01


def slug(tid):
    return tid.replace(":", "_")


def wtag(w):
    return f"cag{w:g}"


def mode_of(arm, tid):
    if arm == "zs":
        return "ours"
    if arm == "empty":
        return "promptx:"
    if arm.startswith("cag"):
        return f"factor:{tid.split(':')[1]}?ctrl=cfg&w={float(arm[3:]):g}"
    raise KeyError(arm)


def roll_out(exp, tid, mode):
    if exp == "m1_main":
        return M1.roll_out(exp, tid, mode)
    h = hashlib.sha1(mode.encode()).hexdigest()[:10]
    return os.path.join(OUT, "smoke" if SMOKE else "roll", f"{exp}_{slug(tid)}_{h}.json")


def arms_of(s):
    cag = [wtag(w) for w in OMEGAS]
    return {"dev": ["zs"] + cag, "test": cag + ["empty"], "cells": cag + ["empty"]}[s]


def roll_of(s, tid, arm):
    """The rollout file of (set, task, arm); the test set's zs is M1's own rollout."""
    if s == "test" and arm == "zs":
        return M1.roll_out("m1_main", tid, "ours")
    return roll_out(f"r1cag_{s}", tid, mode_of(arm, tid))


# ── the design table ─────────────────────────────────────────────────────────────────────────────────────────────
def make_plan():
    m1 = M1.load_plan()
    tasks = {tid: dict(text=v["text"], scene=v["scene"], bin=v["bin"]) for tid, v in m1["tasks"].items()}
    cells = [c["cell"] for c in m1["damage"]]
    res = dict(tasks=tasks, cells=cells, omegas=list(OMEGAS), seed=SEED, ckpt=M1.CKPT,
               dev=dict(dir=DEV_DIR, base=DEV_BASE, n=N_STATES, seed=SEED, arms=arms_of("dev")),
               test=dict(dir=M1.FRESH_DIR, base=M1.BASE, n=N_STATES, seed=SEED, arms=arms_of("test"), zs="M1 m1_main zs rollouts"),
               cells_set=dict(states="stored init states 0-19", seed=0, arms=arms_of("cells"), base="parity_pi05_l90c_5000"),
               candidates=candidates_table(list(tasks) + cells))
    # the served cfg mode builds a factor code first (serve_guided.py factor:): check it builds for every task we serve
    import json as _json
    from lifelong_ood.policy import factor_hooks as FH
    from lifelong_ood.policy import factors as F
    ft = F.load()
    zs0 = {int(k): v["zeroshot"] for k, v in _json.load(open(os.path.join(R, "f0_coverage.json")))["tasks"].items()}
    for tid in list(tasks) + cells:
        t = next(x for x in ft if x.suite == "libero_90" and x.index == int(tid.split(":")[1]))
        FH.code_for(t, ft, zs0)
        assert FH.FactorHooks({}, mode="cfg", w=2.0).compose({"prompt": t.language}).prompts == [t.language, ""]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for tid, c in res["candidates"].items():
        print(tid, c["scene"], "evaluable", [x["tid"] for x in c["cands"] if x["placement"]],
              "articulation-only", [x["tid"] for x in c["cands"] if not x["placement"]])


def load_plan():
    return json.load(open(os.path.join(OUT, "plan.json"), encoding="utf-8"))


def candidates_table(tids):
    """Per task: the training tasks of its scene (split.json "train") with their placement atoms (In / On, lower-cased)."""
    from lifelong_ood import l90c
    tt = M1.task_table()
    train = l90c.load_split()["train"]
    out = {}
    for tid in tids:
        t = tt[tid]
        cs = []
        for c in train:
            if tt[c].scene != t.scene:
                continue
            pl = [[a[0].lower(), a[1], a[2]] for a in tt[c].goal if a[0] in ("In", "On") and len(a) == 3]
            cs.append(dict(tid=c, text=tt[c].language, placement=pl, articulated=[list(a) for a in tt[c].goal
                                                                                  if a[0] not in ("In", "On")]))
        own = [[a[0].lower(), *a[1:]] for a in t.goal]
        out[tid] = dict(scene=t.scene, own=own, named=sorted({a[1] for a in t.goal if a[0] in ("In", "On") and len(a) == 3}),
                        targets=sorted({a[2] for a in t.goal if a[0] in ("In", "On") and len(a) == 3}), cands=cs)
    return out


# ── start states ─────────────────────────────────────────────────────────────────────────────────────────────────
def fresh_check():
    """fresh_r1: 20 states per dev task, each > 1e-6 from every state of every earlier libero_90 fresh set and the stored ones."""
    used = M1.USED_BASES + (5700000, 6100000, 6300000)                 # + M1's and L4's bases (slurm/*.out, 2026-10-24)
    assert not any(b <= DEV_BASE < b + 1000 or DEV_BASE <= b < DEV_BASE + N_STATES for b in used), DEV_BASE
    out = {}
    for tid in load_plan()["tasks"]:
        i = tid.split(":")[1]
        x = np.load(os.path.join(R, DEV_DIR, f"libero_90_{i}.npy"))
        assert len(x) == N_STATES, (tid, x.shape)
        dmin, sets = np.inf, []
        for f in sorted(glob.glob(os.path.join(R, "fresh_*", f"libero_90_{i}.npy"))):
            if os.path.basename(os.path.dirname(f)) == DEV_DIR:
                continue
            y = np.load(f)
            if y.shape[1] != x.shape[1]:
                continue
            dmin = min(dmin, float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))))
            sets.append(os.path.basename(os.path.dirname(f)))
        assert dmin > 1e-6 and "fresh_m1" in sets, (tid, dmin, sets)
        out[tid] = dict(min_dist=dmin, compared=sets)
        print(tid, f"min distance to earlier sets {dmin:.4f} over {len(sets)} sets", flush=True)
    with open(os.path.join(OUT, "fresh_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


# ── closed loop ──────────────────────────────────────────────────────────────────────────────────────────────────
def _line(s, tid, arm):
    suite, i = tid.split(":")
    mode = mode_of(arm, tid)
    sp = dict(MODE=mode, SUITE=suite, TASKS=i, TRIALS=N_STATES, FIRST=0, EXPERIMENT=f"r1cag_{s}", TAG=arm, PROGRAM_BYTES=0,
              OUT=roll_of(s, tid, arm), BACKBONE_CONFIG=M1.CFG, BACKBONE_CKPT=M1.CKPT,
              BIND="track", RECORD="1", LIGHT="1", OBJECTS="1", ALL_OBJECTS="1")
    if s in ("dev", "test"):
        sp["SERVE_SEED"] = str(SEED)
        sp["FRESH"] = os.path.join(R, DEV_DIR if s == "dev" else M1.FRESH_DIR)
    return sp


def submit(lines, lanes=12, per=4, dry=False):
    """Write the pack files and a bash script that submits them in `lanes` afterany chains.  The script is run from the LOGIN
    shell (GPU chains are never submitted from inside a job): results/r1/packs/submit_<ts>.sh."""
    todo, seen = [], set()
    for sp in lines:
        if not os.path.exists(sp["OUT"]) and sp["OUT"] not in seen:
            seen.add(sp["OUT"])
            todo.append(sp)
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    if todo:
        os.makedirs(os.path.dirname(todo[0]["OUT"]), exist_ok=True)
    ts = int(time.time())
    sh = ["#!/bin/bash", "# generated by experiments/r1_cag.py; run from the login shell", "set -e",
          f"cd {ROOT}"] + [f"L{k}=" for k in range(lanes)]
    for j, i in enumerate(range(0, len(todo), per)):
        grp = todo[i:i + per]
        path = os.path.join(OUT, "packs", f"{grp[0]['EXPERIMENT']}_{j:03d}_{ts}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        lane = j % lanes
        sh.append(f"L{lane}=$(sbatch --parsable ${{L{lane}:+--dependency=afterany:$L{lane}}} -p overcap -A overcap --requeue "
                  f"--time=04:00:00 --gres=gpu:a40:1 --exclude={EXCLUDE} --export=ALL,SPECS={path} experiments/roll_pack.sbatch)")
        sh.append(f"echo \"pack {os.path.basename(path)} [{', '.join(sp['TASKS'] + ':' + sp['TAG'] for sp in grp)}] lane {lane} -> $L{lane}\"")
    script = os.path.join(OUT, "packs", f"submit_{ts}.sh")
    with open(script, "w", encoding="utf-8") as f:
        f.write("\n".join(sh) + "\n")
    print(f"{len(todo)} instances, {(len(todo) + per - 1) // per} packs -> {script}")
    return script


def launch(dry):
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    assert glob.glob(os.path.join(ROOT, "ledger", "m3_prereg", "*pr1_cag*")), "pre-register first (experiments/prereg/pr1_cag.py)"
    plan = load_plan()
    lines = []
    for s in ("dev", "test"):
        for tid in plan["tasks"]:
            for arm in arms_of(s):
                lines.append(_line(s, tid, arm))
    for c in plan["cells"]:
        for arm in arms_of("cells"):
            lines.append(_line("cells", c, arm))
    submit(lines, dry=dry)


def smoke_serve(dry):
    """2 episodes per arm (zs, cag2, empty) on libero_90:14's STORED init states 0-1 (never fresh_r1 / fresh_m1)."""
    assert SMOKE, "set R1_SMOKE=1"
    lines = []
    for arm in ("zs", "cag2", "empty"):
        sp = _line("cells", "libero_90:14", arm)
        sp.update(TRIALS=2, EXPERIMENT="r1cag_smoke")
        lines.append(sp)
    submit(lines, lanes=1, per=3, dry=dry)


# ── which demonstrated task did it do instead (LIBERO's venv) ───────────────────────────────────────────────────────
class Scene:
    """fm_types.Scene with the start states of the episode's set (fresh dir or the stored init states)."""

    def __init__(self, tid, states):
        import pathlib
        import fm_types as FT
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv
        suite, i = tid.split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(i))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.inner = FT.Scene.__new__(FT.Scene)
        sc = self.inner
        sc.env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        sc.env.seed(0)
        sc.env.reset()
        sc.e = sc.env.env
        sc.fresh = (benchmark.get_benchmark_dict()[suite]().get_task_init_states(int(i)) if states == "stored"
                    else np.load(os.path.join(R, states, f"{suite}_{i}.npy")))
        sc.goal = [list(g) for g in sc.e.parsed_problem["goal_state"]]


def eval_cands(sc, cands):
    """-> {training task: True / False / None (a name absent from this scene)} at the env's current state."""
    names = set(sc.e.object_states_dict)
    out = {}
    for c in cands:
        if not c["placement"]:
            continue
        if any(a[1] not in names or a[2] not in names for a in c["placement"]):
            out[c["tid"]] = None
            continue
        out[c["tid"]] = bool(all(sc.pred(a[0], a[1], a[2]) for a in c["placement"]))
    return out


def instead_episode(sc, cand, e, rec):
    import fm_types as FT
    ty = FT.type_episode(sc, None, e, rec)                               # leaves the env at the settled final state
    if rec is None:
        return dict(ty, final=None, start=None, instead=[])
    final = eval_cands(sc, cand["cands"])
    names = [str(x) for x in rec["obj_names"]]
    sc.set(int(e["trial"]), names, rec["obj_pos"][0], rec["obj_quat"][0], settle=SETTLE)
    start = eval_cands(sc, cand["cands"])
    own = {tuple(a) for a in cand["own"]}
    inst = []
    for c in cand["cands"]:
        if final.get(c["tid"]) and not start.get(c["tid"]) and not {tuple(a) for a in c["placement"]} <= own:
            inst.append(dict(tid=c["tid"], shares_object=any(a[1] in cand["named"] for a in c["placement"]),
                             shares_target=any(a[2] in cand["targets"] for a in c["placement"])))
    return dict(ty, final=final, start=start, instead=inst)


def instead_task(s, tid, arms, cand):
    states = "stored" if s == "cells" else (DEV_DIR if s == "dev" else M1.FRESH_DIR)
    if SMOKE:
        states = "stored"
    S = Scene(tid, states)
    sc = S.inner
    f = os.path.join(OUT, "smoke_instead" if SMOKE else "instead", f"{s}_{slug(tid)}.json")
    res = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else dict(set=s, task=tid, arms={})
    import fm_types as FT
    for arm in arms:
        rf = roll_of(s, tid, arm) if not SMOKE else roll_out("r1cag_cells", tid, mode_of(arm, tid))
        if not os.path.exists(rf) or arm in res["arms"]:
            continue
        z = json.load(open(rf, encoding="utf-8"))
        eps = [ep for t in z["tasks"].values() for ep in t["episodes"]]
        rd = rf[:-5] + "_rec"
        out = [instead_episode(sc, cand, ep, FT.load_rec(rd, tid, int(ep["trial"]))) for ep in eps]
        res["arms"][arm] = dict(roll=os.path.relpath(rf, ROOT), episodes=out)
        print(f"[r1 instead] {s} {tid} {arm}: " + " ".join(f"{x['trial']}:{x['type']}{'+' + ','.join(i['tid'].split(':')[1] for i in x['instead']) if x['instead'] else ''}"
                                                       for x in out), flush=True)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(f + ".tmp", f)
    sc.env.close()


def instead_all(shard, w_arms):
    plan = load_plan()
    jobs = []
    if SMOKE:
        jobs = [("cells", "libero_90:14", ["zs", "cag2", "empty"])]
    else:
        for tid in plan["tasks"]:
            jobs.append(("test", tid, ["zs", "empty"] + w_arms))
        for c in plan["cells"]:
            jobs.append(("cells", c, ["empty"] + w_arms))
    i, n = map(int, shard.split("/"))
    for j, (s, tid, arms) in enumerate(jobs):
        if j % n == i:
            instead_task(s, tid, arms, plan["candidates"][tid])


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _eps(f):
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for t in json.load(open(f, encoding="utf-8"))["tasks"].values()
            for e in t["episodes"]}


def holm(ps, alpha=0.05):
    return M1.holm(ps, alpha)


def table(s, tids, arms):
    tab, missing = {}, []
    for tid in tids:
        z = {a: _eps(roll_of(s, tid, a)) for a in arms}
        if any(e is None or len(e) != N_STATES for e in z.values()):
            missing.append(tid)
            continue
        tab[tid] = z
    return tab, missing


def select(plan):
    """w* on the dev set: argmax successes on the 15 recombination tasks over {1 (zs), 1.5, 2, 3, 5}; ties to the smaller w."""
    tids = list(plan["tasks"])
    tab, missing = table("dev", tids, arms_of("dev"))
    if missing:
        return None, dict(missing=missing)
    rec = [t for t in tids if plan["tasks"][t]["bin"] != "part"]
    tot = {a: dict(recombination=int(sum(sum(tab[t][a].values()) for t in rec)),
                   part=int(sum(sum(tab[t][a].values()) for t in tids if t not in rec)),
                   all=int(sum(sum(tab[t][a].values()) for t in tids))) for a in arms_of("dev")}
    ws = [(1.0, "zs")] + [(w, wtag(w)) for w in OMEGAS]
    best = max(ws, key=lambda x: (tot[x[1]]["recombination"], -x[0]))
    return best, dict(totals=tot, per_task={t: {a: int(sum(tab[t][a].values())) for a in arms_of("dev")} for t in tids},
                      w_star=best[0], arm_star=best[1])


def read(record):
    from lifelong_ood import stats as ST
    plan = load_plan()
    tids = list(plan["tasks"])
    rec = [t for t in tids if plan["tasks"][t]["bin"] != "part"]
    part = [t for t in tids if t not in rec]
    best, sel = select(plan)
    res = dict(select=sel)
    if best is None:
        print("[r1] dev set incomplete:", sel)
        return
    w_star, a_star = best
    print(f"[r1] dev selection: w* = {w_star:g} ({a_star}); totals {json.dumps(sel['totals'])}")
    # test set
    arms_t = ["zs"] + arms_of("test")
    tab, missing = table("test", tids, arms_t)
    res["test_missing"] = missing
    if missing:
        print("[r1] test set incomplete:", missing)
        return
    if a_star == "zs":
        tab = {t: dict(v, star=v["zs"]) for t, v in tab.items()}
    else:
        tab = {t: dict(v, star=v[a_star]) for t, v in tab.items()}
    arms_r = arms_t + ["star"]
    groups = dict(recombination=rec, part=part, all=tids)
    res["test_rate"] = {g: {a: {k: x for k, x in ST.bootstrap_rate(tab, a, ts).items() if k != "samples"} for a in arms_r}
                        for g, ts in groups.items()}
    by_bin = {}
    for t in tids:
        g = by_bin.setdefault(plan["tasks"][t]["bin"], {a: 0 for a in arms_r} | dict(tasks=0))
        g["tasks"] += 1
        for a in arms_r:
            g[a] += int(sum(tab[t][a].values()))
    res["test_by_bin"] = by_bin

    def cmp(a, b, ts):
        pairs = ST.paired(tab, a, b, ts)
        d = ST.bootstrap_diff(pairs, B=10000)
        bb, cc = ST.discordant(pairs)
        x = int(sum(p[0].sum() for p in pairs.values()))
        y = int(sum(p[1].sum() for p in pairs.values()))
        return dict(x=x, y=y, diff=d, discordant=[bb, cc], sign_p=float(ST.sign_test(bb, cc)))
    res["vs_zs"] = {g: {a: cmp(a, "zs", ts) for a in arms_r if a != "zs"} for g, ts in groups.items()}
    h1 = res["vs_zs"]["recombination"]["star"]
    p1 = h1["sign_p"] if (h1["x"] > h1["y"] and a_star != "zs") else 1.0
    # instead / types
    inst = load_instead(plan)
    res["instead"] = inst["summary"]
    res["zs_type_agreement"] = inst["zs_type_agreement"]
    h2 = inst["H2"]
    hm = holm(dict(H1=p1, H2=h2["p"]))
    res["H1"] = dict(h1, arm=a_star, w=w_star, p_used=p1, passes=hm["H1"]["reject"])
    res["H2"] = dict(h2, passes=hm["H2"]["reject"])
    res["holm"] = hm
    # cells: damage and the vision-shortcut rate
    base = M1.parity_base()
    ctab, cmiss = {}, []
    for c in plan["cells"]:
        z = {a: _eps(roll_of("cells", c, a)) for a in arms_of("cells")}
        if base.get(c) is None or any(e is None or len(e) != N_STATES for e in z.values()):
            cmiss.append(c)
            continue
        ctab[c] = dict(base=base[c], **z)
    res["cells_missing"] = cmiss
    if not cmiss:
        dmg = {}
        for a in arms_of("cells"):
            ch = {c: float(np.mean(list(z[a].values())) - np.mean(list(z["base"].values()))) for c, z in ctab.items()}
            pairs = ST.paired(ctab, a, "base", list(ctab))
            bb, cc = ST.discordant(pairs)
            dmg[a] = dict(mean_change=float(np.mean(list(ch.values()))), per_cell=ch, successes=int(sum(sum(z[a].values()) for z in ctab.values())),
                          discordant=[bb, cc], sign_p=float(ST.sign_test(bb, cc)),
                          rate=({k: x for k, x in ST.bootstrap_rate(ctab, a, list(ctab)).items() if k != "samples"}))
        dmg["base"] = dict(successes=int(sum(sum(z["base"].values()) for z in ctab.values())),
                           rate=({k: x for k, x in ST.bootstrap_rate(ctab, "base", list(ctab)).items() if k != "samples"}))
        res["cells"] = dmg
        star_c = dmg[a_star]["mean_change"] if a_star != "zs" else 0.0
        res["H3"] = dict(arm=a_star, mean_change=star_c, passes=bool(star_c >= DAMAGE_GATE))
    print(json.dumps({k: res[k] for k in ("select", "test_by_bin", "H1", "H2", "H3", "holm") if k in res}, indent=1, default=str)[:6000])
    for g in groups:
        print(f"[r1] {g}: " + "  ".join(f"{a} {res['test_rate'][g][a]['k']}/{res['test_rate'][g][a]['n']}" for a in arms_r))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "read.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r1cag_read", bench="libero_90_c", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(dev=plan["dev"], test=plan["test"], cells=plan["cells_set"], omegas=list(OMEGAS),
                                        ckpt=M1.CKPT, prereg="m3_prereg tag pr1_cag"),
                            script="experiments/r1_cag.py", command=" ".join(sys.argv)))


def load_instead(plan):
    """Aggregate results/r1/instead/*.json: failure types per arm and set, the 'did a demonstrated task instead' rates, H2."""
    from lifelong_ood import stats as ST
    fam = [t for t, c in plan["candidates"].items() if t in plan["tasks"] and any(x["placement"] for x in c["cands"])]
    summ, tabs = {}, {}
    for f in sorted(glob.glob(os.path.join(OUT, "instead", "*.json"))):
        z = json.load(open(f, encoding="utf-8"))
        for arm, v in z["arms"].items():
            key = f"{z['set']}:{arm}"
            g = summ.setdefault(key, dict(episodes=0, success=0, types={}, failures=0, fail_instead=0, fail_instead_obj=0,
                                          fail_instead_tgt=0, fail_instead_either=0, all_instead=0, all_instead_either=0,
                                          success_settled_of_success=0, evaluable_tasks=0, by_task={}))
            bt = g["by_task"].setdefault(z["task"], dict(success=0, fail=0, fail_instead=0, all_instead=0, matched={}))
            for e in v["episodes"]:
                g["episodes"] += 1
                inst = e.get("instead") or []
                either = any(i["shares_object"] or i["shares_target"] for i in inst)
                g["all_instead"] += bool(inst)
                g["all_instead_either"] += bool(inst and either)
                bt["all_instead"] += bool(inst)
                for i in inst:
                    bt["matched"][i["tid"]] = bt["matched"].get(i["tid"], 0) + 1
                if e["success"]:
                    g["success"] += 1
                    bt["success"] += 1
                    g["success_settled_of_success"] += bool(e.get("success_settled"))
                    continue
                g["failures"] += 1
                bt["fail"] += 1
                g["types"][e["type"]] = g["types"].get(e["type"], 0) + 1
                g["fail_instead"] += bool(inst)
                bt["fail_instead"] += bool(inst)
                g["fail_instead_obj"] += any(i["shares_object"] for i in inst)
                g["fail_instead_tgt"] += any(i["shares_target"] for i in inst)
                g["fail_instead_either"] += either
            tabs.setdefault(key, {})[z["task"]] = {key: {int(e["trial"]): int(bool(e.get("instead"))) for e in v["episodes"]}}
    # restricted to the tasks with >= 1 evaluable same-scene training placement goal
    for key, g in summ.items():
        g["evaluable_tasks"] = sum(1 for t in g["by_task"] if t in fam)
        sub = {t: v for t, v in g["by_task"].items() if t in fam}
        g["fam"] = dict(tasks=len(sub), episodes=sum(v["success"] + v["fail"] for v in sub.values()),
                        all_instead=sum(v["all_instead"] for v in sub.values()),
                        failures=sum(v["fail"] for v in sub.values()), fail_instead=sum(v["fail_instead"] for v in sub.values()))
    h2 = dict(test="share of empty-instruction episodes on the dev tasks with >= 1 evaluable same-scene training placement goal "
                   f"that complete a demonstrated task > {H2_SHARE} (one-sided two-level cluster bootstrap, B 10000)")
    key = "test:empty"
    if key in tabs:
        tb = {t: v for t, v in tabs[key].items() if t in fam}
        r = ST.bootstrap_rate(tb, key, list(tb))
        h2.update(rate=r["rate"], lo=r["lo"], hi=r["hi"], k=r["k"], n=r["n"], tasks=r["tasks"],
                  p=float(np.mean(np.asarray(r["samples"]) <= H2_SHARE)) if "samples" in r else None)
        if h2["p"] is None:                                              # stats.bootstrap_rate keeps no samples: recompute
            h2["p"] = _boot_p(tb, key, H2_SHARE)
    else:
        h2.update(p=1.0, note="no empty-instruction instead file")
    # our retyping of M1's zs against the failure map's own types (results/fm/types)
    agree, total = 0, 0
    for f in sorted(glob.glob(os.path.join(OUT, "instead", "test_*.json"))):
        z = json.load(open(f, encoding="utf-8"))
        g = os.path.join(ROOT, "results", "fm", "types", f"{slug(z['task'])}.json")
        if "zs" not in z["arms"] or not os.path.exists(g):
            continue
        fm = {int(e["trial"]): e["type"] for e in json.load(open(g, encoding="utf-8"))["arms"].get("zs", {}).get("episodes", [])}
        for e in z["arms"]["zs"]["episodes"]:
            if int(e["trial"]) in fm:
                total += 1
                agree += fm[int(e["trial"])] == e["type"]
    return dict(summary=summ, H2=h2, fam_tasks=fam, zs_type_agreement=dict(agree=agree, total=total))


def _boot_p(tb, arm, thr, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    xs = [np.array(list(tb[t][arm].values()), float) for t in tb]
    s = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(xs), len(xs))
        num, den = 0.0, 0
        for j in pick:
            m = len(xs[j])
            num += xs[j][rng.integers(0, m, m)].sum()
            den += m
        s[i] = num / den
    return float(np.mean(s <= thr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("plan", "fresh-check", "launch", "smoke-serve", "instead", "read", "select", "dry", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--w-arms", default="+".join(wtag(w) for w in OMEGAS))
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.plan:
        make_plan()
    if a.fresh_check:
        fresh_check()
    if a.launch:
        launch(a.dry)
    if a.smoke_serve:
        smoke_serve(a.dry)
    if a.instead:
        instead_all(a.shard, a.w_arms.split("+"))
    if a.select:
        print(json.dumps(select(load_plan())[1], indent=1))
    if a.read:
        read(a.record)
