"""PLAN L, L2b (exploratory, dev tasks only; docs/PLAN_LANG.md "L2 results"): which OFFLINE selection criterion, computed from ONE
demo with no practice and no reward, best predicts the closed-loop success of a language-repair operator?

Ground truth is the L2 dev closed loop (results/lang/pl2/dev_read.json, ledger pl2_dev: 8 dev tasks x 20 episodes x every arm x
{pi0.5, pi0}).  Nothing here rolls out or edits L2's files: it imports experiments/pl2_compose.py and lifelong_ood/lang/transplant.py
read-only, reads L2's offline selections, served specs and caches, and writes only results/lang/pl2b/ and ledger/pl2b_criteria/.

Offline quantities (GPU, --compute):
  fields   for every rolled arm (zs, null, tli_li, word, tmpl, tli_sel, role, rand, pos), rebuilt from L2's offline file and served
           spec, the predicted velocity at demo 0's frames (the SAME draws as L2: seed 1, K 4) and demo 1's (seed 2), plus the
           targets.  -> per-frame flow losses (reproduces L2's l0 / l1) and the field displacement
           D(arm) = mean ||v_arm - v_zs|| / mean ||v_zs - v_null||  over demo-0 frames and draws (norm over the chunk x 7 dims).
  splits   L2's whole demo selection (pl2_compose.select_task: sources, spans, template, pair, lambda) re-run on HALF of demo 0's
           frames and scored on the other half: early -> late, late -> early (by time), even -> odd, odd -> even (interleaved).
           Missing binding caches go to results/lang/pl2b/bind (L2's results/lang/pl2/bind is only read).

Criteria (--read, CPU), each one rule over the candidate set M = {zs, word, tmpl, tli_sel, role} (OPS = M without zs is L2's set):
  a   demo-0 loss (L2's l2sel, over OPS; "a_zs" also lets zs win)
  b   split-demo validation: per operator the held-out half's log loss ratio to zs with its choices fitted on the other half,
      averaged over the two folds (b_time: early/late, b_inter: even/odd, b_cv4: all four); the deployed arm is the operator's
      rolled configuration (fitted on the whole demo 0), so the rule maps onto existing rollouts.  "strict" counts how often the
      half-demo configuration equals the rolled one.
  c   MDL / minimal edit: log(l0 / l0_zs) + w * D, one w chosen leave-one-task-out (pooled over both policies) by closed-loop success
  d   zero-shot guard: rule a, but keep zs unless log(l0 / l0_zs) < -m, one m chosen leave-one-task-out (pooled)
  e   demo-1 loss (TWO demos: a reference ceiling, not allowed in the method)

    sbatch ... experiments/pl2b_criteria.sbatch   (ARGS="--compute --backbone pi0_libero --shard 0/4"; resumable per file)
    $PY experiments/pl2b_criteria.py --read [--record]       table, results/lang/pl2b/criteria_read.json, candidates.csv, ledger
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import pl2_compose as PC  # noqa: E402  (read-only import; it sets the openpi path)

OUTB = os.path.join(ROOT, "results", "lang", "pl2b")
ARMS = ("zs", "null", "tli_li", "word", "tmpl", "tli_sel", "role", "rand", "pos")
ROLLED = ("zs", "tli_li", "word", "tmpl", "tli_sel", "role", "rand", "pos")
M = ("zs", "word", "tmpl", "tli_sel", "role")                      # the method's candidates (tie order: zs first, then L2's)
OPS = PC.OPS
SPLITS = ("early", "late", "even", "odd")


def halves(F, sp):
    i = np.arange(F)
    a = dict(early=i[:F // 2], late=i[F // 2:], even=i[0::2], odd=i[1::2])[sp]
    return a, np.setdiff1d(i, a)


def sub(d, ix, obj=True):
    return dict(d, frames=[d["frames"][int(i)] for i in ix], chunks=d["chunks"][ix], states=d["states"][ix], carry=d["carry"][ix],
                q=d["q"][ix], idx=d["idx"][ix], **({} if obj else {"obj": None}))


# ── GPU ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def velocities(bk, sc, prompt, comp):
    """DemoScorer.loss's loop, keeping the predicted velocity: -> [F, K, H, 7] float32."""
    import torch
    out = []
    if comp is not None:
        comp.attach()
    try:
        with torch.no_grad():
            for s in range(0, len(sc.frames), sc.batch):
                e = min(s + sc.batch, len(sc.frames))
                if comp is not None:
                    comp.carry = torch.as_tensor(sc.carry[s:e])
                    comp.q = sc.q[s:e]
                pre = bk.prefix(sc._observations(prompt, s, e))
                n = e - s
                t = sc.t[s:e].reshape(-1)
                eps = sc.eps[s:e].reshape(n * sc.K, bk.H, bk.A)
                x = sc.x[s:e].repeat_interleave(sc.K, 0)
                xt = t[:, None, None] * eps + (1 - t[:, None, None]) * x
                v = bk.v_full(xt, t, bk.repeat(pre, sc.K)).reshape(n * sc.K, bk.H, bk.A)[:, :, :7]
                out.append(v.reshape(n, sc.K, bk.H, 7).float().cpu().numpy())
    finally:
        if comp is not None:
            comp.detach()
            comp.carry = comp.q = None
    return np.concatenate(out)


def compute(bb, only, shard, which="dev", fields=True, splits=SPLITS):
    import torch  # noqa: F401
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    import lifelong_ood.lang.transplant as LT
    import g12_tli as T
    os.makedirs(os.path.join(OUTB, "bind"), exist_ok=True)
    os.makedirs(os.path.join(OUTB, "fields"), exist_ok=True)
    os.makedirs(os.path.join(OUTB, "split"), exist_ok=True)
    orig = LT._bpath

    def bpath(config, task, prompt):                                    # read L2's caches, write new ones under pl2b/bind
        f = orig(config, task, prompt)
        return f if os.path.exists(f) else os.path.join(OUTB, "bind", os.path.basename(f))
    LT._bpath = bpath
    bk = PolicyBackbone(bb, device="cuda")
    demos = D.Demos()
    idx = json.load(open(os.path.join(PC.OUT, "demos.json")))[bb]
    pl = PC.plan()
    dev = PC.split()[which]
    for j, k in enumerate(dev):
        if (only and k not in only) or j % shard[1] != shard[0]:
            continue
        v = pl["tasks"][k]
        off = json.load(open(os.path.join(PC.OUT, "offline", f"{PC.short(bb)}_{PC.sh(k)}.json")))
        a_, b_, lam = T.MAP[v["language"].replace(" ", "_")]
        li = (T.KEY[a_], T.KEY[b_], lam)
        d0, d1 = PC.load_demo(idx[k]["d0"], bk.H), PC.load_demo(idx[k]["d1"], bk.H)
        assert d0["T"] == off["d0"]["T"], (k, d0["T"], off["d0"]["T"])
        tag = f"{PC.short(bb)}_{PC.sh(k)}"
        f = os.path.join(OUTB, "fields", f"{tag}.npz")
        if fields and not os.path.exists(f):
            s0 = LT.DemoScorer(bk, d0["frames"], d0["chunks"], d0["states"], d0["carry"], d0["q"], K=PC.K_DRAWS, seed=1)
            s1 = LT.DemoScorer(bk, d1["frames"], d1["chunks"], d1["states"], d1["carry"], d1["q"], K=PC.K_DRAWS, seed=2)
            lat, arr = {}, {}
            for sc, nm in ((s0, "0"), (s1, "1")):
                arr[f"u{nm}"] = (sc.eps - sc.x[:, None])[..., :7].float().cpu().numpy()
            for name in ARMS:
                a = off["arms"][name]
                comp = None
                if a["tli"] is not None:
                    for t in (a["tli"]["t1"], a["tli"]["t2"]):
                        if t not in lat:
                            lat[t] = PC._tli_lat(bk, bb, demos, t)
                    comp = LT.Composer(bk, (), tli=dict(d=lat[a["tli"]["t1"]] - lat[a["tli"]["t2"]], lam=a["tli"]["lam"]))
                elif a["edits"]:
                    es, kw = LT.load_spec(PC.spec_path(bb, v["suite"], v["index"], name))       # the served form (float16 shifts)
                    assert kw["prompt"] == a["prompt"] and len(es) == len(a["edits"]), (k, name)
                    comp = LT.Composer(bk, es, lo=kw["lo"], hi=kw["hi"], schedule=kw["schedule"])
                for sc, nm in ((s0, "0"), (s1, "1")):
                    arr[f"v{nm}_{name}"] = velocities(bk, sc, a["prompt"], comp)
                l0 = float(((arr[f"v0_{name}"] - arr["u0"]) ** 2).mean((2, 3)).mean())
                print(f"[pl2b] {tag} {name}: l0 {l0:.5f} (L2 {off['l0'][name]:.5f})", flush=True)
            tmp = f"{f[:-4]}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp, **{x: y.astype(np.float16) for x, y in arr.items()}, frames=d0["idx"], frames1=d1["idx"])
            os.replace(tmp, f)
            PC.torch_empty()
        for sp in splits:
            g = os.path.join(OUTB, "split", f"{tag}_{sp}.json")
            if os.path.exists(g):
                continue
            A, B = halves(len(d0["frames"]), sp)
            res, _, _ = PC.select_task(bk, bb, demos, k, v["language"], sub(d0, A), sub(d0, B, obj=False), li=li, named=None)
            res.update(split=sp, sel_frames=A.tolist(), val_frames=B.tolist())
            tmp = g + f".{os.getpid()}.tmp"
            json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=float)
            os.replace(tmp, g)
            print(f"[pl2b] {tag} split {sp}: pick {res['l2sel']}; held-out " +
                  " ".join(f"{n} {res['l1'][n]:.4f}" for n in M), flush=True)
            PC.torch_empty()
    print("[pl2b] compute done", flush=True)


# ── read ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    a, b = rank(x), rank(y)
    return float(np.corrcoef(a, b)[0, 1])


def cfg_of(a):
    """An arm's configuration (what a rollout would serve), for comparing a half-demo selection with the rolled one."""
    if a["tli"] is not None:
        return ("tli", a["tli"]["t1"], a["tli"]["t2"], a["tli"]["lam"])
    return ("lt", a["prompt"], tuple(sorted((e["role"], e["src"]) for e in a["edits"])), a["lo"], a["hi"], a["sched"])


def load_units():
    """-> list of per (policy, task) dicts: success per rolled arm, and every criterion's value per arm."""
    dev = PC.split()["dev"]
    dr = json.load(open(os.path.join(PC.OUT, "dev_read.json")))
    units = []
    for bb in PC.BBS:
        b = PC.short(bb)
        for k in dev:
            off = json.load(open(os.path.join(PC.OUT, "offline", f"{b}_{PC.sh(k)}.json")))
            s = {n: dr[b]["per_task"][k][n]["s"] for n in ROLLED}
            assert dr[b]["per_task"][k]["l2sel"]["s"] == s[off["l2sel"]], (b, k)
            u = dict(bb=b, task=k, succ=s, l2sel=off["l2sel"], l0=off["l0"], l1=off["l1"], crit={}, strict={})
            f = os.path.join(OUTB, "fields", f"{b}_{PC.sh(k)}.npz")
            if os.path.exists(f):
                z = np.load(f)
                V = {n: z[f"v0_{n}"].astype(np.float32) for n in ARMS}
                U = z["u0"].astype(np.float32)
                l0r = {n: float(((V[n] - U) ** 2).mean((2, 3)).mean()) for n in ARMS}
                u["l0_recomputed"] = l0r
                u["l0_maxrel_err"] = max(abs(l0r[n] / off["l0"][n] - 1) for n in ARMS)
                den = float(np.linalg.norm((V["zs"] - V["null"]).reshape(*V["zs"].shape[:2], -1), axis=-1).mean())
                u["D"] = {n: float(np.linalg.norm((V[n] - V["zs"]).reshape(*V[n].shape[:2], -1), axis=-1).mean()) / den for n in ARMS}
                u["field_null_norm"] = den
            sp = {}
            for x in SPLITS:
                g = os.path.join(OUTB, "split", f"{b}_{PC.sh(k)}_{x}.json")
                if os.path.exists(g):
                    sp[x] = json.load(open(g))
            u["splits_have"] = sorted(sp)
            lr = lambda d_, n: float(np.log(d_[n] / d_["zs"]))           # noqa: E731
            u["crit"]["a"] = {n: lr(off["l0"], n) for n in ROLLED}
            u["crit"]["e"] = {n: lr(off["l1"], n) for n in ROLLED}
            for nm, folds in (("b_time", ("early", "late")), ("b_inter", ("even", "odd")), ("b_cv4", SPLITS)):
                if all(x in sp for x in folds):
                    u["crit"][nm] = {n: float(np.mean([lr(sp[x]["l1"], n) for x in folds])) for n in M + ("tli_li",)}
                    u["strict"][nm] = {n: float(np.mean([cfg_of(sp[x]["arms"][n]) == cfg_of(off["arms"][n]) for x in folds]))
                                       for n in OPS}
            if "D" in u:
                u["crit"]["D"] = {n: u["D"][n] for n in ROLLED}
            units.append(u)
    return units


def pick(vals, cands):
    """argmin over cands in the given order (first wins ties)."""
    return min(cands, key=lambda n: (vals[n], cands.index(n)))


def rule_c(u, w):
    v = {n: u["crit"]["a"][n] + w * u["D"][n] for n in M}
    return pick(v, list(M))


def rule_d(u, m):
    b = pick(u["crit"]["a"], list(OPS))
    return b if u["crit"]["a"][b] < -m else "zs"


W_GRID = [0.0] + list(np.geomspace(1e-3, 10, 41))
M_GRID = list(np.linspace(0, 0.5, 51))


def loto(units, rule, grid, pooled=True):
    """Leave-one-task-out choice of the rule's one constant by closed-loop success of the other units (pooled: all other
    (policy, task) units; else the same policy's other tasks); ties -> the median of the maximising grid points."""
    picks, chosen = {}, {}
    for i, u in enumerate(units):
        tr = [x for j, x in enumerate(units) if j != i and (pooled or x["bb"] == u["bb"])]
        sc = np.array([sum(x["succ"][rule(x, g)] for x in tr) for g in grid])
        best = np.flatnonzero(sc == sc.max())
        g = grid[int(best[len(best) // 2])]
        chosen[(u["bb"], u["task"])] = float(g)
        picks[(u["bb"], u["task"])] = rule(u, g)
    sc = np.array([sum(x["succ"][rule(x, g)] for x in units) for g in grid])
    best = np.flatnonzero(sc == sc.max())
    full = dict(value=float(grid[int(best[len(best) // 2])]), range=[float(grid[best[0]]), float(grid[best[-1]])],
                success=int(sc.max()), curve={f"{grid[i]:.4g}": int(sc[i]) for i in range(len(grid))})
    return picks, chosen, full


def read(record):
    units = load_units()
    miss = [(u["bb"], u["task"]) for u in units if "D" not in u or len(u["splits_have"]) < 4]
    if miss:
        print("[pl2b] incomplete units (fields / splits):", miss)
    full = [u for u in units if "D" in u and len(u["splits_have"]) == 4]
    # rules -> pick per unit
    rules = {}
    for u in units:
        key = (u["bb"], u["task"])
        c = u["crit"]
        r = {"zs (always)": "zs", "tli_sel (always)": "tli_sel", "a: demo-0 loss (l2sel)": pick(c["a"], list(OPS)),
             "a_zs: demo-0 loss, zs allowed": pick(c["a"], list(M)),
             "e: demo-1 loss (2 demos, ref)": pick(c["e"], list(M)), "e_ops: demo-1 loss over OPS (ref)": pick(c["e"], list(OPS)),
             "oracle (best rolled in M)": max(M, key=lambda n: (u["succ"][n], -M.index(n)))}
        assert r["a: demo-0 loss (l2sel)"] == u["l2sel"], key
        for nm in ("b_time", "b_inter", "b_cv4"):
            if nm in c:
                r[f"{nm}: split-demo validation"] = pick(c[nm], list(M))
                r[f"{nm}_ops: split-demo validation over OPS"] = pick(c[nm], list(OPS))
        rules[key] = r
    lt = {}
    if full:
        for nm, fn, grid in (("c: MDL l0 + w*D (LOTO w)", rule_c, W_GRID), ("d: zs guard, margin m (LOTO m)", rule_d, M_GRID)):
            for pooled in (True, False):
                p, ch, fl = loto(full, fn, grid, pooled)
                lab = nm + ("" if pooled else " [per-policy LOTO]")
                lt[lab] = dict(chosen={f"{a}|{b}": v for (a, b), v in ch.items()}, full_data=fl)
                for key, x in p.items():
                    rules[key][lab] = x
    # per criterion x policy metrics
    names = list(dict.fromkeys(n for r in rules.values() for n in r))
    out = {}
    for bb in ("pi05", "pi0", "both"):
        us = [u for u in units if bb == "both" or u["bb"] == bb]
        tab = {}
        for n in names:
            ks = [u for u in us if n in rules[(u["bb"], u["task"])]]
            if not ks:
                continue
            s = [u["succ"][rules[(u["bb"], u["task"])][n]] for u in ks]
            best = [max(u["succ"][m] for m in M) for u in ks]
            tab[n] = dict(success=int(sum(s)), of=20 * len(ks), tasks=len(ks), missing=len(us) - len(ks),
                          hit_exact=int(sum(a == b for a, b in zip(s, best))), hit_within2=int(sum(b - a <= 2 for a, b in zip(s, best))),
                          picks={f"{u['bb']}|{u['task']}": rules[(u["bb"], u["task"])][n] for u in ks})
        cor = {}
        for cn in ("a", "b_time", "b_inter", "b_cv4", "D", "e", "c"):
            for cset, cs in (("M", M), ("M+tli_li", M + ("tli_li",)), ("all8", ROLLED)):
                w_, p_ = [], []
                xs, ys = [], []
                for u in us:
                    if cn == "c":
                        if "D" not in u or not lt:
                            continue
                        w = lt["c: MDL l0 + w*D (LOTO w)"]["chosen"].get(f"{u['bb']}|{u['task']}")
                        if w is None:
                            continue
                        val = {n: u["crit"]["a"][n] + w * u["D"][n] for n in ROLLED}
                    else:
                        val = u["crit"].get(cn)
                    if val is None or any(n not in val for n in cs):
                        continue
                    r = spearman([-val[n] for n in cs], [u["succ"][n] for n in cs])
                    (w_ if r is not None else p_).append(r)
                    xs += [-(val[n] - val["zs"]) for n in cs if n != "zs"]
                    ys += [u["succ"][n] - u["succ"]["zs"] for n in cs if n != "zs"]
                if w_ or p_:
                    cor[f"{cn}|{cset}"] = dict(within_mean=None if not w_ else float(np.mean(w_)), within_n=len(w_),
                                               within_undefined=len(p_), within_pos=int(sum(r > 0 for r in w_)),
                                               pooled=spearman(xs, ys), pooled_n=len(xs))
        out[bb] = dict(rules=tab, spearman=cor)
    strict = {}
    for nm in ("b_time", "b_inter", "b_cv4"):
        v = [u["strict"][nm][n] for u in units if nm in u["strict"] for n in OPS]
        if v:
            strict[nm] = float(np.mean(v))
    res = dict(units=len(units), complete=len(full), incomplete=miss, rules_note="success of the rolled arm the rule deploys",
               table=out, loto=lt, strict_config_match=strict,
               l0_reproduction_max_rel_err=max([u.get("l0_maxrel_err", 0) for u in units]),
               D={f"{u['bb']}|{u['task']}": u.get("D") for u in units},
               crit={f"{u['bb']}|{u['task']}": u["crit"] for u in units})
    os.makedirs(OUTB, exist_ok=True)
    json.dump(res, open(os.path.join(OUTB, "criteria_read.json"), "w", encoding="utf-8"), indent=1, default=float)
    with open(os.path.join(OUTB, "candidates.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cn = ("a", "e", "b_time", "b_inter", "b_cv4", "D")
        w.writerow(["policy", "task", "arm", "success_of_20", "l0", "l1", "l0_over_zs", "l1_over_zs"] +
                   [f"crit_{c}" for c in cn] + ["l2sel"])
        for u in units:
            for n in ROLLED:
                w.writerow([u["bb"], u["task"], n, u["succ"][n], f"{u['l0'][n]:.5f}", f"{u['l1'][n]:.5f}",
                            f"{u['l0'][n] / u['l0']['zs']:.4f}", f"{u['l1'][n] / u['l1']['zs']:.4f}"] +
                           ["" if c not in u["crit"] or n not in u["crit"][c] else f"{u['crit'][c][n]:.4f}" for c in cn] +
                           [int(u["l2sel"] == n)])
    # print
    print(f"[pl2b] units {len(units)}, complete {len(full)}; l0 reproduction max rel err {res['l0_reproduction_max_rel_err']:.4f}")
    print("strict (half-demo config == rolled config, mean over ops/folds):", strict)
    for lab, x in lt.items():
        print(f"LOTO {lab}: full-data value {x['full_data']['value']:.4g} range {x['full_data']['range']} "
              f"(in-sample {x['full_data']['success']}); per-fold " + " ".join(f"{v:.3g}" for v in x["chosen"].values()))
    for bb in ("pi05", "pi0", "both"):
        print(f"\n== {bb}: rule -> closed-loop success (of), hit exact / within 2, missing")
        for n, t in out[bb]["rules"].items():
            print(f"  {n:52} {t['success']:4d}/{t['of']:<4d} hit {t['hit_exact']}/{t['tasks']} w2 {t['hit_within2']}/{t['tasks']}"
                  f" miss {t['missing']}")
        print(f"  Spearman(-criterion, success): within-task mean [n def / undef, #>0] | pooled (delta vs zs)")
        for n, c in out[bb]["spearman"].items():
            wm = "  n/a" if c["within_mean"] is None else f"{c['within_mean']:+.3f}"
            pp = "  n/a" if c["pooled"] is None else f"{c['pooled']:+.3f}"
            print(f"  {n:18} {wm} [{c['within_n']}/{c['within_undefined']}, {c['within_pos']}] | {pp} (n {c['pooled_n']})")
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl2b_criteria", bench="libero_ood", tag="read", seed=PC.SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(K=PC.K_DRAWS, frames=PC.F_TARGET, splits=SPLITS, candidates=M, w_grid=W_GRID, m_grid=M_GRID,
                                        ground_truth="ledger pl2_dev / results/lang/pl2/dev_read.json", exploratory=True,
                                        dev_tasks=PC.split()["dev"]),
                            script="experiments/pl2b_criteria.py", command=" ".join(sys.argv),
                            notes="L2b exploratory: offline selection criteria vs L2 dev closed loop; no new rollouts"))


# ── the chosen rule (pre-registered: experiments/prereg/pl2b_rule.py) and its confirm read ───────────────────────────
RULE_FOLDS = ("even", "odd")


def rule_b_inter(bb, k):
    """b_inter: per operator in M, the mean over the two interleaved folds (choices fitted on even frames scored on odd, and the
    reverse) of log(held-out loss / zs's held-out loss); zs = 0; argmin over M in the order (zs, word, tmpl, tli_sel, role);
    the deployed arm is that operator's L2 configuration (fitted on all of demo 0).  -> (pick, criterion values)."""
    sp = [json.load(open(os.path.join(OUTB, "split", f"{PC.short(bb)}_{PC.sh(k)}_{x}.json"))) for x in RULE_FOLDS]
    val = {n: float(np.mean([np.log(s["l1"][n] / s["l1"]["zs"]) for s in sp])) for n in M}
    return pick(val, list(M)), val


def read_confirm(record):
    """The frozen rule on the pi0.5 confirm tasks, from the existing pl2_conf rollouts (results/lang/pl2/confirm_read.json)."""
    conf = PC.split()["confirm"]
    cr = json.load(open(os.path.join(PC.OUT, "confirm_read.json")))["pi05"]["per_task"]
    per, tot = {}, {n: 0 for n in ROLLED + ("l2sel",)}
    for k in conf:
        p, val = rule_b_inter("pi05_libero", k)
        s = {n: cr[k][n]["s"] for n in ROLLED + ("l2sel",)}
        per[k] = dict(pick=p, success=s[p], crit=val, **{f"s_{n}": s[n] for n in s})
        for n in tot:
            tot[n] += s[n]
    res = dict(rule="b_inter", tasks=len(conf), success=int(sum(r["success"] for r in per.values())), of=20 * len(conf),
               references=tot, picks={k: r["pick"] for k, r in per.items()},
               oracle_M=int(sum(max(cr[k][n]["s"] for n in M) for k in conf)), per_task=per)
    json.dump(res, open(os.path.join(OUTB, "confirm_rule_read.json"), "w", encoding="utf-8"), indent=1)
    print(f"[pl2b confirm] rule b_inter: {res['success']}/{res['of']}; refs {tot}; oracle in M {res['oracle_M']}")
    for k, r in per.items():
        print(f"  {k:10} pick {r['pick']:8} {r['success']:2d}/20   zs {r['s_zs']:2d} l2sel {r['s_l2sel']:2d} tli_sel {r['s_tli_sel']:2d}"
              f" tli_li {r['s_tli_li']:2d}")
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl2b_criteria", bench="libero_ood", tag="confirm_rule", seed=PC.SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(rule="b_inter", folds=RULE_FOLDS, candidates=M, backbone="pi05_libero",
                                        prereg="m3_prereg tag pl2b_rule (experiments/prereg/pl2b_rule.py)",
                                        ground_truth="ledger pl2_confirm / results/lang/pl2/confirm_read.json"),
                            script="experiments/pl2b_criteria.py", command=" ".join(sys.argv),
                            notes="frozen L2b rule evaluated offline on the existing pi0.5 confirm rollouts; no new rollouts"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--which", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--splits", default="+".join(SPLITS))
    ap.add_argument("--read-confirm", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--backbone", default="pi05_libero")
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.compute:
        compute(a.backbone, [x for x in a.only.split("+") if x], tuple(int(x) for x in a.shard.split("/")), a.which,
                not a.no_fields, tuple(a.splits.split("+")))
    if a.read:
        read(a.record)
    if a.read_confirm:
        read_confirm(a.record)
