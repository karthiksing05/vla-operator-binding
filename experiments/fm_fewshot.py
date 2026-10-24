"""FAILURE MAP (docs/PLAN_FM.md; docs/PLAN_STORY.md section 2; pre-registration experiments/prereg/pfm_failure_map.py, ledger
m3_prereg tag pfm_failure_map): where few-shot composition breaks in pi05_l90c@5000 on LIBERO-90-C's 29 dev tasks, typed per episode
(experiments/fm_types.py), and the few-shot curve: stage B adds per-task LoRA from k = 5 and 10 demos (budgets 200 / 1000) and the KV
token from k = 5, at M1's settings, on M1's tasks, start states (fresh_m1) and noise seed (13), so every arm pairs with M1's.

The k demos are the task's own-scene IPEC episodes 0..k-1 (l90c.task_episodes; episode 0 is M1's one demo, so the sets nest).
LIBERO's teleop demos never start from the evaluation states (fresh_m1 is LIBERO's sampler at BASE 5700000, checked new by M1).
Every frame of every demo trains (PrefixBank on the CPU), batch 32, the same steps, learning rates and seeds as M1.

    sbatch ... robot_gpu.sbatch SCRIPT="experiments/fm_fewshot.py --fit --task libero_90:9 --k 5"   one task, one k (GPU)
    $PY experiments/fm_fewshot.py --demo-grasp                 the demos' grasp points (layout distance, F2) -> results/fm/
    $PY experiments/fm_fewshot.py --launch-fits [--dry] | --launch [--dry] | --read [--record]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "fm")
FITS = os.path.join(OUT, "fits")
KS = (5, 10)
TOKEN_KS = (5,)
SNAPS = (200, 1000)
FM_ARMS = ("create_k5", "lora_k5_200", "lora_k5_1000", "lora_k10_200", "lora_k10_1000")
ALL_ARMS = M1.ARMS + FM_ARMS
TYPED = ("no_commit", "wrong_object", "wrong_goal", "motion")       # fixture failures are success-only (not typed)
EXCLUDE = M1.EXCLUDE


def slug(tid):
    return tid.replace(":", "_")


# ── fits (GPU) ───────────────────────────────────────────────────────────────────────────────────────────────────
def fit_task(bk, tid, v, k):
    import importlib.util
    import torch
    import lifelong_ood.lang.transplant as LT
    import pl2_compose as PC
    from lifelong_ood import l90c, ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import token_fold as TF
    sp = importlib.util.spec_from_file_location("c0", os.path.join(ROOT, "experiments", "c0_fit.py"))
    c0 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c0)
    d = os.path.join(FITS, slug(tid), f"k{k}")
    done = os.path.join(d, "fit.json")
    if os.path.exists(done):
        print(f"[fm] {tid} k{k}: exists; skipped", flush=True)
        return
    os.makedirs(d, exist_ok=True)
    eps = [int(e) for e in l90c.task_episodes(tid)]
    assert len(eps) >= k and eps[0] == v["demo_episode"], (tid, len(eps), eps[:1], v["demo_episode"])
    demo_eps = eps[:k]
    text = v["text"]
    norm = D.Norm.for_backbone(bk)
    tm = {}
    t0 = M1._sync()
    d90 = D.Demos90()
    obs, rows, Ts = [], [], []
    for e in demo_eps:
        dm = PC.load_demo((d90, e), bk.H)
        T = dm["T"]
        obs += [dict(dm["_get"](i), prompt=text) for i in range(T)]
        idx, _ = D.chunk_indices(T, bk.H)
        rows.append(np.asarray(D.pad_rows(c0.chunk_rows(dict(actions=dm["actions"], state=dm["state"]), idx, range(T), norm),
                                          bk.H, bk.a, bk.A)))
        Ts.append(int(T))
        del dm
    full = np.concatenate(rows, 0)
    assert len(full) == len(obs)
    tm["demo"] = M1._sync() - t0
    t0 = M1._sync()
    bank = adapt.PrefixBank(bk, obs, store="cpu")
    tm["bank"] = M1._sync() - t0
    del obs
    specs, hist, byt = {}, {}, {}
    if k in TOKEN_KS:
        t1 = M1._sync()
        tok = M1.init_token(bank, 0, M1.TOK_N, M1.SEED).to(bk.device)
        hist["create"] = TF.fit(bk, tok, bank, full, steps=M1.TOK_STEPS, lr=M1.TOK_LR, batch=M1.BATCH, seed=M1.SEED, log_every=200,
                                log=lambda x: print(f"[fm] {tid} k{k} create {x}", flush=True))
        tp = os.path.join(d, "tok_create.pt")
        torch.save(dict(token=tok.state_dict_fp16(), task=tid, text=text, n=M1.TOK_N, on="instruction", k=k, demos=demo_eps), tp)
        byt["token"] = tok.nbytes()
        tm[f"create_k{k}"] = tm["demo"] + tm["bank"] + M1._sync() - t1
        p = os.path.join(d, "create.json")
        LT.save_spec(p, [], prompt=text, meta=dict(task=tid, k=k), token=tp)
        specs[f"create_k{k}"] = p
        del tok
        torch.cuda.empty_cache()
    state, banks = adapt.inject(bk)
    key = f"fm_{slug(tid)}_k{k}"
    params = add_node(banks, key, M1.LORA_RANK)
    snaps, cnt, shift = {}, [0], [0.0]
    t1 = M1._sync()

    def snap():
        cnt[0] += 1
        s = 50 * cnt[0]
        if s in SNAPS and s != SNAPS[-1]:
            ts = M1._sync()
            snaps[s] = dict(path=os.path.join(d, f"lora{s}.pt"), secs=ts - t1 - shift[0])
            adapt.save(banks, key, snaps[s]["path"], meta=dict(task=tid, text=text, rank=M1.LORA_RANK, steps=s, scope="expert", k=k))
            shift[0] += M1._sync() - ts
        return 0.0
    hl = adapt.train(bk, state, key, params, bank, full, steps=SNAPS[-1], lr=M1.LORA_LR, batch=M1.BATCH, seed=0, eval_fn=snap,
                     eval_every=50, log_every=100, log=lambda x: print(f"[fm] {tid} k{k} {x}", flush=True))
    te = M1._sync() - t1
    p = os.path.join(d, f"lora{SNAPS[-1]}.pt")
    byt["lora"] = adapt.save(banks, key, p, meta=dict(task=tid, text=text, rank=M1.LORA_RANK, steps=SNAPS[-1], scope="expert", k=k))
    snaps[SNAPS[-1]] = dict(path=p, secs=te - shift[0])
    for s, x in snaps.items():
        specs[f"lora_k{k}_{s}"] = x["path"]
        tm[f"lora_k{k}_{s}"] = tm["demo"] + tm["bank"] + x["secs"]
    hist["lora"] = {kk: vv for kk, vv in hl.items() if kk in ("step", "loss")}
    res = dict(task=tid, text=text, bin=v["bin"], k=k, demos=demo_eps, T=Ts, frames=int(len(full)), specs=specs, seconds=tm,
               curves=hist, bytes=byt, gpu=torch.cuda.get_device_name(0))
    tmp = done + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, done)
    tot = int(sum(byt.values()))
    print(ledger.record("fm_fit", bench="libero_90_c", tag=f"k{k}", seed=M1.SEED, keys=dict(task=tid, k=k),
                        metrics=dict(seconds=tm, T=Ts, frames=int(len(full)), bin=v["bin"],
                                     final_loss={kk: (h["loss"][-1] if h.get("loss") else None) for kk, h in hist.items()}),
                        bytes=dict(experts=byt.get("lora", 0), tree=0, codes=byt.get("token", 0), samples=0, teacher_copies=0,
                                   total=tot),
                        config=dict(k=k, demos=demo_eps, tok=dict(n=M1.TOK_N, steps=M1.TOK_STEPS, lr=M1.TOK_LR, batch=M1.BATCH),
                                    lora=dict(rank=M1.LORA_RANK, lr=M1.LORA_LR, steps=SNAPS[-1], snaps=SNAPS, batch=M1.BATCH),
                                    ckpt=M1.CKPT),
                        script="experiments/fm_fewshot.py", command=" ".join(sys.argv), artifacts=[done]))
    print(f"[fm] {tid} k{k} done: seconds {json.dumps({kk: round(x, 1) for kk, x in tm.items()})}", flush=True)


def fit(task, k):
    plan = M1.load_plan()
    bk = M1.backbone()
    for tid in ([task] if task != "all" else list(plan["tasks"])):
        fit_task(bk, tid, plan["tasks"][tid], k)


def fit_of(tid, k):
    f = os.path.join(FITS, slug(tid), f"k{k}", "fit.json")
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def arm_modes(tid):
    out = {}
    for k in KS:
        ft = fit_of(tid, k)
        if ft is None:
            continue
        for a, p in ft["specs"].items():
            out[a] = f"lora:{p}" if a.startswith("lora") else f"lt:{p}"
    return out


def launch_fits(dry):
    import launch_pack as LP
    plan = M1.load_plan()
    jobs = []
    for k in KS:
        for tid in plan["tasks"]:
            if fit_of(tid, k) is not None:
                continue
            args = ["--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=01:30:00", "--gres=gpu:a40:1",
                    f"--mem={64 if k == 5 else 128}G", f"--exclude={EXCLUDE}", f"--job-name=fm{k}_{tid.split(':')[1]}",
                    f"--export=ALL,SCRIPT=experiments/fm_fewshot.py --fit --task {tid} --k {k}", "slurm/jobs/robot_gpu.sbatch"]
            jobs.append(LP.sbatch(args, dry))
            print(tid, k, jobs[-1], flush=True)
    os.makedirs(OUT, exist_ok=True)
    if not dry:
        with open(os.path.join(OUT, "fit_jobs.txt"), "a", encoding="utf-8") as f:
            f.write(",".join(jobs) + "\n")


# ── the demos' grasp points (F2's layout distance) ───────────────────────────────────────────────────────────────
def demo_grasp():
    """Per dev task, M1's one demo (episode 0): the EEF position at its first sustained gripper close (pl2_compose.carry_of) and
    at frame 0.  The layout distance of an episode = xy distance from its named object's start to this grasp point."""
    import pl2_compose as PC
    from lifelong_ood.policy import data as D
    plan = M1.load_plan()
    d90 = D.Demos90()
    out = {}
    for tid, v in plan["tasks"].items():
        ep = d90.load(v["demo_episode"])
        _, c0, rel = PC.carry_of(ep["actions"])
        st = ep["state"]
        out[tid] = dict(episode=v["demo_episode"], c0=c0, grasp_xyz=None if c0 < 0 else [float(x) for x in st[c0, :3]],
                        start_eef=[float(x) for x in st[0, :3]], T=int(len(st)),
                        closes=[[int(f)] + [float(x) for x in st[f, :3]] for f in range(len(st)) if (ep["actions"][f:f + 10, 6] > 0).all()
                                and len(ep["actions"][f:f + 10]) == 10 and (f == 0 or ep["actions"][f - 1, 6] <= 0)])
        print(tid, json.dumps(out[tid]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "demo_grasp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


# ── rollouts ─────────────────────────────────────────────────────────────────────────────────────────────────────
def roll_out(tid, mode):
    h = hashlib.sha1(mode.encode()).hexdigest()[:10]
    return os.path.join(OUT, "roll", f"fm_main_{slug(tid)}_{h}.json")


def submit(lines, lanes=10, per=4, dry=False):
    import launch_pack as LP
    todo, seen = [], set()
    for s in lines:
        if not os.path.exists(s["OUT"]) and s["OUT"] not in seen:
            seen.add(s["OUT"])
            todo.append(s)
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "roll"), exist_ok=True)
    last, jobs = [None] * lanes, []
    for j, i in enumerate(range(0, len(todo), per)):
        grp = todo[i:i + per]
        path = os.path.join(OUT, "packs", f"fm_main_{j:03d}_{int(time.time())}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        lane = j % lanes
        args = ([f"--dependency=afterany:{last[lane]}"] if last[lane] else []) + [
            "--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=04:00:00", "--gres=gpu:a40:1",
            f"--exclude={EXCLUDE}", f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        jid = LP.sbatch(args, dry)
        last[lane] = jid
        jobs.append(jid)
        print(f"pack {os.path.basename(path)} [{', '.join(s['TASKS'] + ':' + s['TAG'] for s in grp)}] lane {lane} -> {jid}", flush=True)
    print(f"{len(todo)} instances, {len(jobs)} packs: {':'.join(jobs)}")
    return jobs


def launch(dry):
    plan = M1.load_plan()
    lines, missing = [], []
    for tid in plan["tasks"]:
        am = arm_modes(tid)
        for arm in FM_ARMS:
            if arm not in am:
                missing.append(f"{tid}:{arm}")
                continue
            sp = M1._line(tid, am[arm], "fm_main", arm, True, M1.SEED)
            sp["OUT"] = roll_out(tid, am[arm])
            lines.append(sp)
    print("not fitted:", missing)
    submit(lines, dry=dry)


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────
def load_types():
    out = {}
    for tid in M1.load_plan()["tasks"]:
        f = os.path.join(OUT, "types", f"{slug(tid)}.json")
        if os.path.exists(f):
            out[tid] = json.load(open(f, encoding="utf-8"))
    return out


def ep_table(types, arm):
    """{task: {trial: episode dict}} of one arm."""
    return {t: {int(e["trial"]): e for e in z["arms"][arm]["episodes"]} for t, z in types.items() if arm in z["arms"]}


def is_pickplace(ax):
    return bool(ax["named"])


def inst_error(e):
    """An instance-role binding error: the grasp lands on another instance of the named object's type, or the named object ends in
    / on another instance of its target's type."""
    ty = lambda n: re.sub(r"_\d+$", "", n)                              # noqa: E731
    if e.get("type") == "wrong_object" and e.get("sub") == "wrong_instance":
        return True
    if e.get("type") == "wrong_goal" and e.get("target"):
        return any(ty(a) == ty(e["target"]) and a != e["target"] for a in e.get("alt", []))
    return False


def boot_share(groups, thr=0.5, B=10000, seed=0):
    """Pooled share of 1s over task groups (lists of 0/1), two-level cluster bootstrap (tasks, then episodes);
    p = the share of bootstrap shares <= thr (one-sided test that the share exceeds thr)."""
    gs = [np.asarray(g, float) for g in groups if len(g)]
    if not gs:
        return dict(k=0, n=0, share=None, lo=None, hi=None, p=1.0)
    rng = np.random.default_rng(seed)
    k, n = sum(g.sum() for g in gs), sum(len(g) for g in gs)
    s = np.empty(B)
    for b in range(B):
        num = den = 0.0
        for i in rng.integers(0, len(gs), len(gs)):
            g = gs[i]
            x = g[rng.integers(0, len(g), len(g))]
            num += x.sum()
            den += len(x)
        s[b] = num / den
    return dict(k=int(k), n=int(n), share=float(k / n), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)),
                p=float(np.mean(s <= thr)))


def rankdata(x):
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_rho(x, y, n=100000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = spearman(x, y)
    y = np.asarray(y, float)
    c = sum(spearman(x, rng.permutation(y)) >= r0 - 1e-12 for _ in range(n))
    return r0, float((c + 1) / (n + 1))


def sign_one_sided(b, c):
    """P(X >= b) for X ~ Bin(b + c, 1/2): the one-sided exact sign test that b > c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n)


def logit_slope(xs, ys):
    """Logistic regression y ~ 1 + x by Newton (IRLS), with a tiny ridge for separable data -> slope."""
    X = np.c_[np.ones(len(xs)), xs]
    y = np.asarray(ys, float)
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - 1e-3 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-3 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[1])


def grasp_point(dg, eps, named):
    """The demo's grasp of the named object: among its sustained-close onsets, the one whose EEF xy is nearest the named object's
    mean start xy over the 20 fresh states (two-step tasks also close on a knob or a handle).  None if the demo never closes."""
    if not dg.get("closes"):
        return None
    m = np.mean([e["start_xy"][named] for e in eps.values()], 0)
    c = min(dg["closes"], key=lambda r: np.hypot(r[1] - m[0], r[2] - m[1]))
    return c[1:4]


def read(record):
    from lifelong_ood import stats as ST
    plan = M1.load_plan()
    types = load_types()
    assert len(types) == 29, f"typed tasks {len(types)}"
    ax = {t: z["axes"] for t, z in types.items()}
    arms = [a for a in ALL_ARMS if all(a in z["arms"] for z in types.values())]
    missing_arms = [a for a in ALL_ARMS if a not in arms]
    tab = {a: ep_table(types, a) for a in arms}
    res = dict(arms=arms, missing_arms=missing_arms, tasks=len(types))
    # success (cross-checked against M1's read for its arms)
    succ = {t: {a: {tr: int(e["success"]) for tr, e in tab[a][t].items()} for a in arms} for t in types}
    m1r = json.load(open(os.path.join(M1.OUT, "read.json"), encoding="utf-8"))
    for a in M1.ARMS:
        if a in arms:
            assert sum(sum(succ[t][a].values()) for t in types) == m1r["totals"][a], a
    res["totals"] = {a: int(sum(sum(succ[t][a].values()) for t in types)) for a in arms}
    # typer validation: on successes of pick-and-place tasks, the named object's own atom holds at the settled final pose
    val = [all(v for v in e["own_atoms_final"].values()) for a in arms for t in types if is_pickplace(ax[t])
           for e in tab[a][t].values() if e["success"] and e.get("own_atoms_final")]
    res["validation"] = dict(successes=len(val), own_atom_holds=int(sum(val)), share=float(np.mean(val)) if val else None)
    # failure-type shares by arm x axis
    levels = dict(bin=("0.1", "0.3", "0.5", "part"), kind=("combination", "part"), scene_familiar=(True, False), instance=(True, False))

    def count(a, tasks):
        c = dict(success=0, fixture=0, untyped=0, **{k: 0 for k in TYPED})
        for t in tasks:
            for e in tab[a][t].values():
                c[e["type"] if e["type"] in c else "untyped"] += 1
        c["episodes"] = int(sum(c[k] for k in ("success", "fixture", "untyped") + TYPED))
        f = c["episodes"] - c["success"]
        c["shares_of_failures"] = {k: (c[k] / f if f else None) for k in TYPED + ("fixture",)}
        return c
    by = {}
    for axis, lv in levels.items():
        by[axis] = {str(l): {a: count(a, [t for t in types if ax[t][axis] == l]) for a in arms} for l in lv}
    by["all"] = {"all": {a: count(a, list(types)) for a in arms}}
    res["types"] = by
    pp = [t for t in types if is_pickplace(ax[t])]
    comb = [t for t in pp if ax[t]["kind"] == "combination"]
    part = [t for t in pp if ax[t]["kind"] == "part"]
    inst = [t for t in pp if ax[t]["instance"]]
    res["sets"] = dict(pickplace=pp, combination_pp=comb, part_pp=part, instance=inst)
    bind = lambda e: e["type"] in ("wrong_object", "wrong_goal")       # noqa: E731
    # F1: zs binding errors rise with CD (combination pick-and-place tasks)
    cd = [ax[t]["cd"] for t in comb]
    rate = {a: [np.mean([bind(e) for e in tab[a][t].values()]) for t in comb] for a in arms}
    r0, p1 = perm_rho(cd, rate["zs"])
    F = {}
    F["F1"] = dict(test="Spearman rho(task CD, zs binding-error rate) > 0 on the combination pick-and-place tasks; one-sided "
                        "permutation p (100,000)", rho=r0, p=p1, per_task={t: dict(cd=c, rate=float(r)) for t, c, r in zip(comb, cd, rate["zs"])})
    F["F1"]["repair"] = {a: dict(rho=spearman(cd, rate[a]), by_bin={b: float(np.mean([r for t, r in zip(comb, rate[a]) if ax[t]["bin"] == b]))
                                                                    for b in ("0.1", "0.3", "0.5")})
                         for a in ("zs", "compose_sel", "compose_cov", "create", "cc", "lora1000", "create_k5", "lora_k10_1000") if a in arms}
    # F2: LoRA from one demo binds the demo's layout; more demos reduce it
    dg = json.load(open(os.path.join(OUT, "demo_grasp.json"), encoding="utf-8"))
    rng = np.random.default_rng(0)
    rows = []
    for t in pp:
        g = grasp_point(dg[t], tab["lora1000"][t], ax[t]["named"][0])
        if g is None:
            continue
        nm = ax[t]["named"][0]
        for tr, e in tab["lora1000"][t].items():
            s = e["start_xy"][nm]
            rows.append((t, float(np.hypot(s[0] - g[0], s[1] - g[1])), int(e["type"] == "wrong_object"), e))
    S_obs, perms = 0.0, np.zeros(20000)
    groups = {}
    for t, dl, w, _ in rows:
        groups.setdefault(t, []).append((dl, w))
    for t, xs in groups.items():
        dl = np.array([x[0] for x in xs])
        w = np.array([x[1] for x in xs])
        c = dl - dl.mean()
        S_obs += float((c * w).sum())
        for b in range(len(perms)):
            perms[b] += float((c * rng.permutation(w)).sum())
    p2a = float((np.sum(perms >= S_obs - 1e-12) + 1) / (len(perms) + 1))
    wo = [r for r in rows if r[2]]
    near = []
    for t, dl, w, e in wo:                                              # is the wrongly bound object nearer the demo's grasp point?
        g = grasp_point(dg[t], tab["lora1000"][t], ax[t]["named"][0])
        sb, sn = e["start_xy"][e["bound"]], e["start_xy"][ax[t]["named"][0]]
        near.append(int(np.hypot(sb[0] - g[0], sb[1] - g[1]) < np.hypot(sn[0] - g[0], sn[1] - g[1])))
    F2 = dict(a=dict(test="lora1000 (k = 1): wrong-object episodes start farther from the demo's grasp point (named object start "
                          "xy); within-task permutation of the wrong-object labels (20,000), one-sided", S=S_obs, p=p2a,
                     wrong_object=len(wo), episodes=len(rows),
                     mean_dlay_wrong=float(np.mean([r[1] for r in wo])) if wo else None,
                     mean_dlay_other=float(np.mean([r[1] for r in rows if not r[2]])) if rows else None),
              nearer_demo_grasp=dict(k=int(sum(near)), n=len(near)))
    if "lora_k10_1000" in arms:
        b = c = 0
        for t in pp:
            for tr, e in tab["lora1000"][t].items():
                x, y = e["type"] == "wrong_object", tab["lora_k10_1000"][t][tr]["type"] == "wrong_object"
                b += x and not y
                c += y and not x
        F2["b"] = dict(test="lora1000 wrong-object episodes at k = 1 > at k = 10 (paired over task x start state, pick-and-place "
                            "tasks): one-sided exact sign test", k1_only=int(b), k10_only=int(c), p=sign_one_sided(int(b), int(c)))
    else:
        F2["b"] = dict(p=1.0, note="k = 10 arm not rolled")
    F2["p"] = max(F2["a"]["p"], F2["b"]["p"])
    F2["repair"] = {a: int(sum(e["type"] == "wrong_object" for t in pp for e in tab[a][t].values())) for a in
                    ("zs", "compose_sel", "create", "create_k5", "lora200", "lora1000", "lora_k5_200", "lora_k5_1000", "lora_k10_200",
                     "lora_k10_1000") if a in arms}
    F["F2"] = F2
    # F3: instance tasks fail by instance-role binding errors (zs)
    gz = [[int(inst_error(e)) for e in tab["zs"][t].values() if e["type"] in TYPED] for t in inst]
    f3 = boot_share(gz)
    if f3["n"] < 10:
        f3["p"], f3["note"] = 1.0, "fewer than 10 typed zs failures on the instance tasks: not testable (pre-registered)"
    f3["test"] = "zs typed failures on the instance tasks: share of instance-role binding errors > 0.5 (cluster bootstrap)"
    f3["per_task"] = {t: dict(fail=int(sum(e["type"] in TYPED for e in tab["zs"][t].values())),
                              inst=int(sum(inst_error(e) for e in tab["zs"][t].values()))) for t in inst}
    rep = {}
    for a in ("compose_sel", "create", "cc", "lora1000", "create_k5", "lora_k10_1000"):
        if a not in arms:
            continue
        x = sum(inst_error(tab["zs"][t][tr]) and not inst_error(tab[a][t][tr]) for t in inst for tr in tab["zs"][t])
        y = sum(inst_error(tab[a][t][tr]) and not inst_error(tab["zs"][t][tr]) for t in inst for tr in tab["zs"][t])
        rep[a] = dict(inst_errors=int(sum(inst_error(e) for t in inst for e in tab[a][t].values())), zs_only=int(x), arm_only=int(y),
                      sign_p_two_sided=float(ST.sign_test(int(x), int(y))),
                      success=int(sum(e["success"] for t in inst for e in tab[a][t].values())))
    rep["zs"] = dict(inst_errors=int(sum(inst_error(e) for t in inst for e in tab["zs"][t].values())),
                     success=int(sum(e["success"] for t in inst for e in tab["zs"][t].values())))
    f3["repair"] = rep
    F["F3"] = f3
    # F4: on new parts, the repaired arms' failures are motion failures
    f4 = {}
    for a in ("create", "lora1000"):
        f4[a] = boot_share([[int(e["type"] == "motion") for e in tab[a][t].values() if e["type"] in TYPED] for t in part])
    F["F4"] = dict(test="part pick-and-place tasks: motion share of typed failures > 0.5 for create AND for lora1000 (cluster "
                        "bootstrap; p = max)", **f4, p=max(f4["create"]["p"], f4["lora1000"]["p"]))
    F["F4"]["zs_types"] = count("zs", part)
    F["F4"]["repair"] = {a: dict(success=int(sum(e["success"] for t in [t for t in types if ax[t]["kind"] == "part"]
                                                 for e in tab[a][t].values()))) for a in arms}
    # F5: compose_sel's induced failures (zs succeeds, compose_sel fails, on tasks where it composed) are binding errors
    fits = {t: M1.fit_of(t) for t in types}
    composed = [t for t in pp if fits[t]["selection"]["best"] != "zs"]
    g5, induced = [], []
    for t in composed:
        g = []
        for tr, e in tab["compose_sel"][t].items():
            if tab["zs"][t][tr]["success"] and not e["success"] and e["type"] in TYPED:
                g.append(int(bind(e)))
                induced.append((t, e))
        g5.append(g)
    f5 = boot_share(g5)
    if f5["n"] < 10:
        f5["p"], f5["note"] = 1.0, "fewer than 10 induced typed failures: not testable (pre-registered)"
    f5["test"] = "compose_sel's induced failures (zs success -> compose_sel failure, composed pick-and-place tasks): share of " \
                 "binding errors (wrong object / wrong goal) > 0.5 (cluster bootstrap)"
    f5["composed_tasks"] = composed
    f5["induced_types"] = {k: int(sum(e["type"] == k for _, e in induced)) for k in TYPED}
    from lifelong_ood.policy import factors as FA
    phrases = FA.object_phrases(list(M1.task_table().values()))
    src = []
    for t, e in induced:
        if e["type"] == "wrong_object":
            sel = fits[t]["selection"]["cands"][fits[t]["selection"]["best"]]
            texts = []
            for c in (plan["tasks"][t]["blind"], plan["tasks"][t]["scene_cover"]):
                if c is not None and (c["t1"], c["t2"]) == (sel["t1"], sel["t2"]):
                    texts = [c["t1_text"], c["t2_text"]]
            ph = FA.phrase_of(re.sub(r"_\d+$", "", e["bound"]), phrases)
            src.append(int(any(ph in s for s in texts)))
    f5["wrong_object_named_by_a_source"] = dict(k=int(sum(src)), n=len(src))
    f5["guard"] = dict(zs=res["totals"]["zs"], compose_sel=res["totals"]["compose_sel"], note="M1: the zs guard did not remove it")
    F["F5"] = f5
    hm = M1.holm({k: F[k]["p"] for k in ("F1", "F2", "F3", "F4", "F5")})
    for k in hm:
        F[k]["holm"] = hm[k]
        F[k]["passes"] = hm[k]["reject"]
    res["F"] = F
    # the few-shot curve
    curve = {}
    k_arms = dict(lora200={1: "lora200", 5: "lora_k5_200", 10: "lora_k10_200"}, lora1000={1: "lora1000", 5: "lora_k5_1000", 10: "lora_k10_1000"},
                  create={1: "create", 5: "create_k5"}, zs={1: "zs"}, compose_sel={1: "compose_sel"})
    allc = [t for t in types if ax[t]["kind"] == "combination"]
    allp = [t for t in types if ax[t]["kind"] == "part"]
    for fam, ks in k_arms.items():
        curve[fam] = {}
        for k, a in ks.items():
            if a not in arms:
                continue
            row = dict(arm=a, all=res["totals"][a], combination=int(sum(sum(succ[t][a].values()) for t in allc)),
                       part=int(sum(sum(succ[t][a].values()) for t in allp)),
                       wrong_object=by["all"]["all"][a]["wrong_object"], wrong_goal=by["all"]["all"][a]["wrong_goal"],
                       motion=by["all"]["all"][a]["motion"], no_commit=by["all"]["all"][a]["no_commit"])
            if k != 1:
                pairs = ST.paired({t: succ[t] for t in types}, a, ks[1], list(types))
                row["vs_k1"] = ST.bootstrap_diff(pairs, B=10000)
                row["vs_k1"].pop("samples", None)
                row["vs_k1"]["discordant"] = list(ST.discordant(pairs))
            curve[fam][str(k)] = row
    res["curve"] = curve
    if all(a in arms for a in ("create_k5", "lora_k5_1000")):
        pairs = ST.paired({t: succ[t] for t in types}, "create_k5", "lora_k5_1000", list(types))
        d = ST.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        res["create_k5_vs_lora_k5_1000"] = dict(diff=d, discordant=list(ST.discordant(pairs)))
    # systematicity: logistic slope of success on CD per arm, combination tasks, cluster bootstrap over tasks
    sys_ = {}
    rngb = np.random.default_rng(0)
    for a in ("zs", "compose_sel", "create", "lora1000", "create_k5", "lora_k10_1000"):
        if a not in arms:
            continue
        X = [(ax[t]["cd"], s) for t in allc for s in succ[t][a].values()]
        sl = logit_slope([x for x, _ in X], [y for _, y in X])
        bs = []
        for _ in range(2000):
            tt = [allc[i] for i in rngb.integers(0, len(allc), len(allc))]
            Xb = [(ax[t]["cd"], s) for t in tt for s in succ[t][a].values()]
            bs.append(logit_slope([x for x, _ in Xb], [y for _, y in Xb]))
        sys_[a] = dict(slope=sl, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    res["systematicity"] = sys_
    # composition cards: tasks the created token fixed (create - zs >= 5 of 20)
    cards = {}
    for t in types:
        dz, dc = sum(succ[t]["zs"].values()), sum(succ[t]["create"].values())
        if dc - dz >= 5:
            v = plan["tasks"][t]
            bz = {}
            bc = {}
            for e in tab["zs"][t].values():
                bz[str(e["bound"])] = bz.get(str(e["bound"]), 0) + 1
            for e in tab["create"][t].values():
                bc[str(e["bound"])] = bc.get(str(e["bound"]), 0) + 1
            cards[t] = dict(text=v["text"], bin=v["bin"], sources=[v["blind"]["t1_text"], v["blind"]["t2_text"]],
                            uncovered=v["blind"]["uncovered"], selection=fits[t]["selection"]["best"], token="74 KB KV token",
                            zs=int(dz), create=int(dc), bound_zs=bz, bound_create=bc, named=ax[t]["named"],
                            zs_types=count("zs", [t])["shares_of_failures"])
    res["cards"] = cards
    with open(os.path.join(OUT, "read.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    figures(res, by)
    print(json.dumps(dict(totals=res["totals"], validation=res["validation"], missing_arms=missing_arms), indent=1))
    for k in ("F1", "F2", "F3", "F4", "F5"):
        print(f"[fm] {k} p {F[k]['p']:.4g} holm {F[k]['holm']} passes {F[k]['passes']}")
    print(json.dumps(dict(curve=curve, systematicity=sys_), indent=1, default=str)[:6000])
    if record:
        from lifelong_ood import ledger
        print(ledger.record("fm_read", bench="libero_90_c", tag="read", seed=M1.SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(arms=arms, prereg="m3_prereg tag pfm_failure_map", typer="experiments/fm_types.py",
                                        settle=10, carry_min=0.10, half=0.5),
                            script="experiments/fm_fewshot.py", command=" ".join(sys.argv)))


def figures(res, by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(no_commit="#2a78d6", wrong_object="#eb6834", wrong_goal="#1baf7a", motion="#eda100", fixture="#e87ba4", success="#d6d5d0")
    order = ("success", "no_commit", "wrong_object", "wrong_goal", "motion", "fixture")
    arms = [a for a in ("zs", "compose_sel", "create", "lora1000", "create_k5", "lora_k5_1000", "lora_k10_1000") if a in res["arms"]]
    lab = dict(zs="zs", compose_sel="compose", create="token k1", lora1000="LoRA k1", create_k5="token k5", lora_k5_1000="LoRA k5",
               lora_k10_1000="LoRA k10")
    panels = [("bin", "0.1", "CD 0.1"), ("bin", "0.3", "CD 0.3"), ("bin", "0.5", "CD 0.5"), ("bin", "part", "new part"),
              ("scene_familiar", "True", "familiar scene"), ("scene_familiar", "False", "new scene"),
              ("instance", "True", "instance qualifier"), ("instance", "False", "no qualifier")]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.5), sharey=True)
    for p, (axis, lv, title) in zip(axs.ravel(), panels):
        bottom = np.zeros(len(arms))
        for ty in order:
            v = np.array([by[axis][lv][a][ty] / max(1, by[axis][lv][a]["episodes"]) for a in arms])
            p.bar(range(len(arms)), v, bottom=bottom, color=col[ty], width=0.75, edgecolor="#fcfcfb", linewidth=1.5,
                  label=ty.replace("_", " "))
            bottom += v
        n = by[axis][lv][arms[0]]["episodes"] // 20
        p.set_title(f"{title} ({n} tasks)", fontsize=10, color="#0b0b0b")
        p.set_xticks(range(len(arms)))
        p.set_xticklabels([lab[a] for a in arms], rotation=45, ha="right", fontsize=8, color="#52514e")
        for s in ("top", "right"):
            p.spines[s].set_visible(False)
        p.tick_params(colors="#52514e")
    axs[0, 0].set_ylabel("share of episodes")
    axs[1, 0].set_ylabel("share of episodes")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Failure types by arm and axis (pi05_l90c@5000, 29 LIBERO-90-C dev tasks x 20 fresh states)", y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.join(ROOT, "figures", "robot"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "robot", "fm_types.png"), dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    # success vs k
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    cs = dict(lora200="#2a78d6", lora1000="#eb6834", create="#1baf7a")
    lb = dict(lora200="LoRA 200 steps", lora1000="LoRA 1000 steps", create="KV token")
    for p, (col_, n, title) in zip(axs, (("all", 580, "all 29 tasks"), ("combination", 300, "15 combination tasks"),
                                         ("part", 280, "14 new-part tasks"))):
        for fam in ("lora200", "lora1000", "create"):
            ks = sorted(int(k) for k in res["curve"][fam])
            ys = [res["curve"][fam][str(k)][col_] / n for k in ks]
            p.plot(ks, ys, marker="o", lw=2, ms=7, color=cs[fam], label=lb[fam])
            p.annotate(lb[fam], (ks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color="#52514e", va="center")
        for fam, ls in (("zs", ":"), ("compose_sel", "--")):
            y = res["curve"][fam]["1"][col_] / n
            p.axhline(y, color="#52514e", ls=ls, lw=1)
            p.annotate("zero-shot" if fam == "zs" else "compose", (10.3, y), fontsize=8, color="#52514e", va="bottom", ha="right")
        p.set_xticks([1, 5, 10])
        p.set_xlim(0.5, 12.5)
        p.set_ylim(0, 1)
        p.set_xlabel("demos k")
        p.set_title(title, fontsize=10)
        for s in ("top", "right"):
            p.spines[s].set_visible(False)
    axs[0].set_ylabel("success rate")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "robot", "fm_k_curve.png"), dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("fit", "launch-fits", "launch", "read", "record", "dry", "demo-grasp"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--task", default="all")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.fit:
        fit(a.task, a.k)
    if a.demo_grasp:
        demo_grasp()
    if a.launch_fits:
        launch_fits(a.dry)
    if a.launch:
        launch(a.dry)
    if a.read:
        read(a.record)
