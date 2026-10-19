"""H1 (C): does composed success FACTOR into the component competences?  (exploratory, from recorded episodes; docs/history/PLAN_H.md
section H1.3)

Question.  A composed step runs one unit until the policy's own grasp and another after it.  If the grasp is the only interface
between them, the composed success is
    P(success | A, T) = g(A) p(T) + (1 - g(A)) w(T),
g(A) = P(the first grasp takes the named object | approach unit A), p(T) = P(success | named object grasped, carry unit T),
w(T) = P(success | another object or nothing grasped, T), each estimated from the arms that use that unit alone.  Where does the
product hold, where does it fail, and does the frozen policy's likelihood of the concept (ledger i6_novelty) predict the
failures?

Method.  Every episode JSON records `bound`, the object the policy's first sustained grasp took (launch_pack --bind track); the
named object is the first argument of the task's BDDL In / On goal.
  LIBERO-90, G5 / G0 (53 held-out tasks, fold concepts, set b at seed 15, 20 episodes; practice states 0-9, 10 episodes): the
    four arms policy alone (zs), object unit only (fac@O.x), goal unit only (fac@x.G) and both (fac@O.G).  g from the object-only
    arm, p and w from the goal-only arm; predicted against the both-arm, in sample (set b -> set b) and out of sample (practice
    states -> set b).  The interface test: P(success | named grasp) under both units against under the goal unit alone (the same
    carry unit after a grasp made by a different approach unit).
  LIBERO-90, I7 (10 E3 tasks, the full library, 5 episodes; experiments/i7_anatomy.py's recordings): the same four arms.
  Li's OOD suites, four policies (pi0.5, pi0, SmolVLA, X-VLA): every grasp-switch arm pfield@a.g (object source a until the grasp,
    goal source g after it).  G10's practice (start states set a: stage 1 crosses 16 object sources with one goal source, stage 2
    16 goal sources with the chosen object source) gives g(a) per source and p(g), w(g) per goal source; (i) leave-one-cell-out
    within set a (is the place competence of a goal source independent of which object source brought the object?), and (ii)
    out of sample: every grasp-switch arm on sets c, d, e, f (G10's chosen pairs, Li's pairs at the grasp, the word rule, the
    G13 / G15 / G16 / G17 / G18 arms) predicted from set a alone.
  Fit: Pearson and Spearman correlation, calibration slope and intercept (observed on predicted), mean absolute error, total
  predicted against observed successes, and the cells whose observed count is unlikely under the prediction (two-sided exact
  binomial p < 0.05).  Likelihood: the residual (observed - predicted) and the failures against the i6 flow loss of the task's
  object and goal concepts.
G20's object-only and goal-only arms (pending at the time of writing) enter the LIBERO-90 read when their files exist.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/h1_factor.py [--record]
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
FIG = os.path.join(ROOT, "figures", "robot")
SFX = {"": "pi05", "_pi0": "pi0", "_smolvla": "smolvla", "_xvla": "xvla"}
SHRINK = 2.0                                            # Beta pseudo-count toward the pooled rate when a cell has few named grasps


# ── episodes ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def episodes(path):
    d = json.load(open(path))
    return [dict(trial=int(e["trial"]), success=bool(e["success"]), bound=e.get("bound")) for v in d["tasks"].values()
            for e in v["episodes"]], d["args"]


def named_objects():
    """-> {(suite, index): the named object's instance name}, from the BDDL goal's first In / On atom."""
    from lifelong_ood.policy import factors as F
    out = {}
    for t in F.load(("libero_90",) + F.OOD_SUITES):
        at = [a for a in t.goal if a[0] in ("In", "On") and len(a) == 3]
        if at:
            out[(t.suite, t.index)] = at[0][1]
    return out


def counts(eps, named):
    """-> dict(n, s, nn = named first grasps, sn = successes among them, so = successes among the others)."""
    nn = [e for e in eps if e["bound"] == named]
    oth = [e for e in eps if e["bound"] != named]
    return dict(n=len(eps), s=sum(e["success"] for e in eps), nn=len(nn), sn=sum(e["success"] for e in nn),
                no=len(oth), so=sum(e["success"] for e in oth))


def add(a, b):
    return {k: a.get(k, 0) + b[k] for k in b}


# ── fit statistics ────────────────────────────────────────────────────────────────────────────────────────────────────
def fit(pred, obs_s, obs_n):
    """pred: predicted success rates; obs_s / obs_n: observed successes and episodes per cell."""
    from scipy import stats as S
    p, s, n = np.asarray(pred, float), np.asarray(obs_s, float), np.asarray(obs_n, float)
    o = s / n
    out = dict(cells=int(len(p)), episodes=int(n.sum()), predicted=float((p * n).sum()), observed=int(s.sum()),
               mae=float(np.abs(p - o).mean()))
    if len(p) > 2 and np.std(p) > 0 and np.std(o) > 0:
        out["pearson"] = float(S.pearsonr(p, o)[0])
        out["spearman"] = float(S.spearmanr(p, o).correlation)
        sl = S.linregress(p, o)
        out["slope"], out["intercept"] = float(sl.slope), float(sl.intercept)
    pc = np.clip(p, 1e-3, 1 - 1e-3)
    bp = np.array([S.binomtest(int(si), int(ni), float(pi)).pvalue for si, ni, pi in zip(s, n, pc)])
    out["misfit_cells"] = int((bp < 0.05).sum())
    out["misfit_over"] = int(((bp < 0.05) & (o > p)).sum())
    out["misfit_under"] = int(((bp < 0.05) & (o < p)).sum())
    return out


def rate(s, n, prior, k=SHRINK):
    return (s + k * prior) / (n + k)


def boot_diff(pairs, B=10000, seed=0):
    """Task-level bootstrap of a pooled difference of rates: pairs = [(s1, n1, s2, n2)] per task."""
    rng = np.random.default_rng(seed)
    P = np.asarray(pairs, float)
    est = P[:, 0].sum() / P[:, 1].sum() - P[:, 2].sum() / P[:, 3].sum()
    bs = []
    for _ in range(B):
        q = P[rng.integers(0, len(P), len(P))]
        if q[:, 1].sum() and q[:, 3].sum():
            bs.append(q[:, 0].sum() / q[:, 1].sum() - q[:, 2].sum() / q[:, 3].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(diff=float(est), lo=float(lo), hi=float(hi), tasks=len(P))


# ── LIBERO-90 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def l90_sets():
    """-> {dataset: {task: {split: {arm: path}}}} for G5 / G0, I7 and (when present) G20."""
    import g0_recombine as G0
    out = {"g5": {}, "i7": {}, "g20": {}}
    for t, v in G0.eligible().items():
        a = {"zs": "zs", "both": f"fac@{v['object_proto']}.{v['goal_proto']}", "obj": f"fac@{v['object_proto']}.x",
             "goal": f"fac@x.{v['goal_proto']}"}
        sp = {}
        for split, (e0, e5) in dict(main=("g0_main", "g5_main"), accept=("g0_accept", "g5_accept")).items():
            sp[split] = {k: os.path.join(R, f"{e5 if k in ('obj', 'goal') else e0}_{t}_{x.replace('@', '_')}_seed15.json")
                         for k, x in a.items()}
        out["g5"][t] = sp
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    for t in ("53", "52", "48", "25", "12", "13", "14", "71", "58", "43"):
        o, g = e3[t]["object_proto"], e3[t]["goal_proto"]
        a = {"zs": "zs", "both": f"fac@{o}.{g}", "obj": f"fac@{o}.x", "goal": f"fac@x.{g}"}
        out["i7"][t] = dict(main={k: os.path.join(R, f"i7_anat_{t}_{x.replace('@', '_')}_seed13.json") for k, x in a.items()})
    try:                                    # G20 on set d: object-only / goal-only arms, paired with G8's policy and library arms
        import g20_ablation as G20
        import g8_leaves as G8
        for t, (_m, a8, _e) in G8.tasks().items():
            a = G20.l90_arms(t)
            f = lambda e, x: os.path.join(R, f"{e}_{t}_{x.replace('@', '_')}_seed13.json")   # noqa: E731
            out["g20"][t] = dict(main=dict(zs=f("g8_main", "zs"), both=f("g8_main", a8["hfac"]), obj=f("g20_main", a["obj_only"]),
                                           goal=f("g20_main", a["goal_only"])))
    except Exception as ex:                                                                         # noqa: BLE001
        print(f"[h1f] G20 arms unavailable: {ex}", flush=True)
    out["g20_ood"] = {}                     # G20 on Li's suites, set e (pi0.5): the chosen pair's sources alone, paired with G15
    try:
        chosen = json.load(open(os.path.join(R, "g10_read_pi05.json")))["chosen"]
        pl = json.load(open(os.path.join(R, "g4_plan.json")))["tasks"]
        for k, v in pl.items():
            a, g = chosen[k].split("@")[1].split(".")
            sh = k.split(":")[0]
            f = lambda e, x: os.path.join(R, f"{e}_{sh}_{v['suite']}_{v['index']}_{x.replace('@', '_')}_seed13.json")   # noqa: E731
            out["g20_ood"][f"{v['suite']}:{v['index']}"] = dict(main=dict(
                zs=f("g15_main", "zs"), both=f("g15_main", chosen[k]), obj=f("g20_ood", f"pfield@{a}.x"), goal=f("g20_ood", f"pfield@x.{g}")))
    except Exception as ex:                                                                         # noqa: BLE001
        print(f"[h1f] G20 OOD arms unavailable: {ex}", flush=True)
    return out


def l90_read(named):
    sets = l90_sets()
    res, tab = {}, {}
    for ds, tasks in sets.items():
        C = {}
        for t, sp in tasks.items():
            nm = named.get((t.split(":")[0], int(t.split(":")[1])) if ":" in t else ("libero_90", int(t)))
            if nm is None:
                continue
            row = {}
            for split, arms in sp.items():
                if not all(os.path.exists(p) for p in arms.values()):
                    continue
                row[split] = {k: counts(episodes(p)[0], nm) for k, p in arms.items()}
            if "main" in row:
                C[t] = row
        if not C:
            res[ds] = dict(tasks=0)
            continue
        tab[ds] = C
        pooled = {k: collections.Counter() for k in ("zs", "obj", "goal", "both")}
        for t, row in C.items():
            for k, c in row["main"].items():
                pooled[k].update(c)
        prior_p = pooled["goal"]["sn"] / max(1, pooled["goal"]["nn"])
        prior_w = pooled["goal"]["so"] / max(1, pooled["goal"]["no"])

        def predict(comp, zero_w=False):
            g = comp["obj"]["nn"] / comp["obj"]["n"]
            p = rate(comp["goal"]["sn"], comp["goal"]["nn"], prior_p)
            w = 0.0 if zero_w else rate(comp["goal"]["so"], comp["goal"]["no"], prior_w)
            return g * p + (1 - g) * w, g, p, w
        r = dict(tasks=len(C), arms={k: dict(episodes=v["n"], successes=v["s"], named_first=v["nn"],
                                            success_given_named=v["sn"], success_given_other=v["so"],
                                            p_named=v["nn"] / max(1, v["n"]), p_succ_named=v["sn"] / max(1, v["nn"]),
                                            p_succ_other=v["so"] / max(1, v["no"])) for k, v in pooled.items()})
        pr = {t: predict(row["main"]) for t, row in C.items()}
        ts = list(C)
        r["in_sample"] = fit([pr[t][0] for t in ts], [C[t]["main"]["both"]["s"] for t in ts], [C[t]["main"]["both"]["n"] for t in ts])
        r["goal_arm_without_named_grasp"] = int(sum(C[t]["main"]["goal"]["nn"] == 0 for t in ts))
        # where the prediction's excess sits: the named-grasp term against the other-grasp term
        r["decomposition"] = dict(pred_named=float(sum(pr[t][1] * pr[t][2] * C[t]["main"]["both"]["n"] for t in ts)),
                                  obs_named=int(sum(C[t]["main"]["both"]["sn"] for t in ts)),
                                  pred_other=float(sum((1 - pr[t][1]) * pr[t][3] * C[t]["main"]["both"]["n"] for t in ts)),
                                  obs_other=int(sum(C[t]["main"]["both"]["so"] for t in ts)))
        r["in_sample_zero_w"] = fit([predict(C[t]["main"], True)[0] for t in ts], [C[t]["main"]["both"]["s"] for t in ts],
                                    [C[t]["main"]["both"]["n"] for t in ts])
        # the grasp tracker's label (`bound`: the object nearest the gripper at the grasp) confuses adjacent objects in some
        # scenes; flagged from the policy-alone arm, which no prediction uses: >= 3 successes after a non-named first grasp
        conf = [t for t in ts if C[t]["main"]["zs"]["so"] >= 3]
        r["tracker_confused_tasks"] = conf
        est = [t for t in ts if C[t]["main"]["goal"]["nn"] >= 3 and t not in conf]     # p measured, not only the prior
        if len(est) > 2:
            r["in_sample_estimable_zero_w"] = fit([predict(C[t]["main"], True)[0] for t in est], [C[t]["main"]["both"]["s"] for t in est],
                                                  [C[t]["main"]["both"]["n"] for t in est])
            r["estimable_tasks"] = len(est)
        keep = [t for t in ts if t not in conf]
        if len(keep) > 2:
            r["in_sample_clean"] = fit([pr[t][0] for t in keep], [C[t]["main"]["both"]["s"] for t in keep],
                                       [C[t]["main"]["both"]["n"] for t in keep])
            r["in_sample_clean_zero_w"] = fit([predict(C[t]["main"], True)[0] for t in keep], [C[t]["main"]["both"]["s"] for t in keep],
                                              [C[t]["main"]["both"]["n"] for t in keep])
        # baselines: the better single component arm; the object arm alone
        r["baseline_max_component"] = fit([max(C[t]["main"]["obj"]["s"], C[t]["main"]["goal"]["s"]) / C[t]["main"]["obj"]["n"]
                                           for t in ts], [C[t]["main"]["both"]["s"] for t in ts], [C[t]["main"]["both"]["n"] for t in ts])
        oos = [t for t in ts if "accept" in C[t]]
        if oos:
            po = {t: predict(C[t]["accept"]) for t in oos}
            r["out_of_sample"] = fit([po[t][0] for t in oos], [C[t]["main"]["both"]["s"] for t in oos],
                                     [C[t]["main"]["both"]["n"] for t in oos])
        # the interface: does the carry unit's success after a named grasp depend on which approach unit made the grasp?
        pairs = [(C[t]["main"]["both"]["sn"], C[t]["main"]["both"]["nn"], C[t]["main"]["goal"]["sn"], C[t]["main"]["goal"]["nn"])
                 for t in ts if C[t]["main"]["both"]["nn"] and C[t]["main"]["goal"]["nn"]]
        if len(pairs) > 2:
            r["interface_both_vs_goal"] = boot_diff(pairs)
        pairs = [(C[t]["main"]["obj"]["sn"], C[t]["main"]["obj"]["nn"], C[t]["main"]["zs"]["sn"], C[t]["main"]["zs"]["nn"])
                 for t in ts if C[t]["main"]["obj"]["nn"] and C[t]["main"]["zs"]["nn"]]
        if len(pairs) > 2:
            r["interface_obj_vs_zs"] = boot_diff(pairs)
        # grasp transfer: does the object unit's grasp in the both arm equal the object-only arm's (same approach)?
        r["grasp_both_vs_obj"] = dict(both=int(sum(C[t]["main"]["both"]["nn"] for t in ts)),
                                      obj=int(sum(C[t]["main"]["obj"]["nn"] for t in ts)), n=int(sum(C[t]["main"]["obj"]["n"] for t in ts)))
        r["per_task"] = {t: dict(pred=round(pr[t][0], 3), pred0=round(predict(C[t]["main"], True)[0], 3), g=round(pr[t][1], 3), p=round(pr[t][2], 3), w=round(pr[t][3], 3),
                                 obs=C[t]["main"]["both"]["s"] / C[t]["main"]["both"]["n"],
                                 p_both_named=(C[t]["main"]["both"]["sn"] / C[t]["main"]["both"]["nn"]) if C[t]["main"]["both"]["nn"]
                                 else None, n=C[t]["main"]["both"]["n"]) for t in ts}
        res[ds] = r
    return res, tab


# ── Li's OOD suites ───────────────────────────────────────────────────────────────────────────────────────────────────
PAT = re.compile(r"^(?P<exp>.+?)_(?P<suite>libero_(?:goal|spatial)_ood)_(?P<idx>\d+)_pfield_(?P<a>\d+)\.(?P<g>\d+)_seed13"
                 r"(?P<sfx>_pi0|_smolvla|_xvla)?\.json$")


def ood_cells(named):
    cells = []
    for f in glob.glob(os.path.join(R, "*_libero_*_ood_*_pfield_*_seed13*.json")):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        eps, args = episodes(f)
        key = (m["suite"], int(m["idx"]))
        if key not in named:
            continue
        cells.append(dict(bk=SFX[m["sfx"] or ""], suite=m["suite"], idx=int(m["idx"]), a=int(m["a"]), g=int(m["g"]), exp=m["exp"],
                          set=os.path.basename(args["fresh"]).replace("fresh_ood_", ""), eps=eps, named=named[key]))
    return cells


def ood_read(named):
    from scipy import stats as S
    cells = ood_cells(named)
    res = {}
    by_bk = collections.defaultdict(list)
    for c in cells:
        by_bk[c["bk"]].append(c)
    figdata = {}
    for bk, cs in sorted(by_bk.items()):
        prac = [c for c in cs if c["set"] == "a"]
        test = [c for c in cs if c["set"] != "a"]
        # g(a): per task and source, majority bound per trial (cells sharing a source share their approach)
        trial_bound = collections.defaultdict(lambda: collections.defaultdict(list))
        for c in prac:
            for e in c["eps"]:
                trial_bound[(c["suite"], c["idx"], c["a"])][e["trial"]].append(e["bound"] == c["named"])
        gA = {k: (sum(collections.Counter(v).most_common(1)[0][0] for v in tb.values()), len(tb)) for k, tb in trial_bound.items()}
        same = sum(len(set(v)) == 1 for tb in trial_bound.values() for v in tb.values())
        tot = sum(1 for tb in trial_bound.values() for v in tb.values())
        # p(g), w(g): per task and goal source, pooled over every practice cell with that goal source
        pg = collections.defaultdict(lambda: collections.Counter())
        for c in prac:
            pg[(c["suite"], c["idx"], c["g"])].update(counts(c["eps"], c["named"]))
        allc = collections.Counter()
        task_c = collections.defaultdict(collections.Counter)
        for k, v in pg.items():
            allc.update(v)
            task_c[k[:2]].update(v)
        prior_p, prior_w = allc["sn"] / max(1, allc["nn"]), allc["so"] / max(1, allc["no"])
        tp_ = {k: (rate(v["sn"], v["nn"], prior_p), rate(v["so"], v["no"], prior_w)) for k, v in task_c.items()}   # per-task priors

        def pred(suite, idx, a, g, minus=None):
            ga = gA.get((suite, idx, a))
            c = pg.get((suite, idx, g))
            if ga is None or c is None:
                return None
            c = collections.Counter(c)
            if minus is not None:
                c.subtract(minus)
            if c["n"] <= 0:
                return None
            gr = ga[0] / ga[1]
            pp, pw = tp_[(suite, idx)]
            return gr * rate(c["sn"], c["nn"], pp) + (1 - gr) * rate(c["so"], c["no"], pw), gr, c["nn"], rate(c["sn"], c["nn"], pp)
        r = dict(practice_cells=len(prac), test_cells=len(test), grasp_same_object_across_goal_sources=[same, tot],
                 practice_pooled=dict(p_succ_named=prior_p, p_succ_other=prior_w, named=allc["nn"], other=allc["no"]))
        # (i) leave-one-cell-out within set a: only cells whose goal source has other practice cells with named grasps
        loo, lo_s, lo_n, lo_key = [], [], [], []
        for c in prac:
            cc = counts(c["eps"], c["named"])
            x = pred(c["suite"], c["idx"], c["a"], c["g"], minus=cc)
            if x is None or x[2] <= 0:
                continue
            loo.append(x[0]); lo_s.append(cc["s"]); lo_n.append(cc["n"]); lo_key.append(c["exp"])
        if loo:
            r["loo_practice"] = fit(loo, lo_s, lo_n)
        # homogeneity of P(success | named) across object sources at a fixed goal source (stage 1), pooled G-test
        G, df = 0.0, 0
        for (suite, idx, g), _ in pg.items():
            cs1 = [counts(c["eps"], c["named"]) for c in prac if (c["suite"], c["idx"], c["g"]) == (suite, idx, g) and
                   c["exp"].startswith("g10_s1")]
            cs1 = [x for x in cs1 if x["nn"] > 0]
            if len(cs1) < 2:
                continue
            tbl = np.array([[x["sn"], x["nn"] - x["sn"]] for x in cs1], float)
            if tbl[:, 0].sum() == 0 or tbl[:, 1].sum() == 0:
                df += len(cs1) - 1
                continue
            exp_ = tbl.sum(1, keepdims=True) * tbl.sum(0, keepdims=True) / tbl.sum()
            G += 2 * float(np.sum(np.where(tbl > 0, tbl * np.log(np.maximum(tbl, 1e-12) / np.maximum(exp_, 1e-12)), 0)))
            df += len(cs1) - 1
        if df:
            r["stage1_homogeneity"] = dict(G=G, df=df, p=float(S.chi2.sf(G, df)))
        # (ii) out of sample: every test cell predicted from set a
        tp, ts_, tn, tk, grasp_obs, grasp_pred, pn_obs, pn_pred = [], [], [], [], [], [], [], []
        for c in test:
            x = pred(c["suite"], c["idx"], c["a"], c["g"])
            if x is None:
                continue
            cc = counts(c["eps"], c["named"])
            tp.append(x[0]); ts_.append(cc["s"]); tn.append(cc["n"]); tk.append((c["suite"], c["idx"], c["a"], c["g"], c["exp"], c["set"]))
            grasp_obs.append(cc["nn"] / cc["n"]); grasp_pred.append(x[1])
            if cc["nn"] >= 3:
                pn_obs.append(cc["sn"] / cc["nn"]); pn_pred.append(x[3])
        if tp:
            r["out_of_sample"] = fit(tp, ts_, tn)
            r["out_of_sample"]["by_set"] = {s: fit([p for p, k in zip(tp, tk) if k[5] == s], [x for x, k in zip(ts_, tk) if k[5] == s],
                                                   [x for x, k in zip(tn, tk) if k[5] == s]) for s in sorted({k[5] for k in tk})}
            r["grasp_transfer"] = dict(pearson=float(S.pearsonr(grasp_pred, grasp_obs)[0]) if np.std(grasp_obs) > 0 else None,
                                       mae=float(np.mean(np.abs(np.array(grasp_pred) - grasp_obs))), cells=len(grasp_obs),
                                       bias=float(np.mean(np.array(grasp_obs) - grasp_pred)))
            if len(pn_obs) > 2:
                r["place_transfer"] = dict(pearson=float(S.pearsonr(pn_pred, pn_obs)[0]) if np.std(pn_obs) > 0 else None,
                                           mae=float(np.mean(np.abs(np.array(pn_pred) - pn_obs))), cells=len(pn_obs),
                                           bias=float(np.mean(np.array(pn_obs) - pn_pred)))
            res_t = collections.defaultdict(list)
            for p, s, n, k in zip(tp, ts_, tn, tk):
                res_t[(k[0], k[1])].append(s / n - p)
            r["residual_by_task"] = {f"{k[0]}:{k[1]}": float(np.mean(v)) for k, v in res_t.items()}
            figdata[bk] = dict(pred=tp, obs=[s / n for s, n in zip(ts_, tn)], n=tn, set=[k[5] for k in tk], loo=loo,
                               loo_obs=[s / n for s, n in zip(lo_s, lo_n)])
        res[bk] = r
    return res, figdata


# ── likelihood ────────────────────────────────────────────────────────────────────────────────────────────────────────
def concept_losses():
    """-> {concept key: frozen pi0.5 flow loss} from ledger i6_novelty (the held concepts on their training demonstrations, the
    new ones on LIBERO-90's)."""
    from lifelong_ood import ledger
    rows = ledger.load("i6_novelty", tag="read")
    m = max(rows, key=lambda x: x["started"])["metrics"]
    held = {(v["kind"], v["key"]): v["loss"] for v in m["held"].values()}
    new = {(v["kind"], v["key"]): v["loss"] for v in m["new"].values()}
    return held, new


def likelihood(l90, ood):
    from scipy import stats as S
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy import primitives as P
    held, new = concept_losses()          # LIBERO-90 concepts: their loss on LIBERO-90 frames; Li's OOD concepts: held ones
    tasks = {(t.suite, t.index): t for t in F.load(("libero_90",) + F.OOD_SUITES)}
    out, pts = {}, {}
    if l90.get("g5", {}).get("per_task"):
        xs = []
        for t, v in l90["g5"]["per_task"].items():
            c = P.steps_of(tasks[("libero_90", int(t))].language)[0]
            lo, lg = new.get(("object", P.object_key(c))), new.get(("goal", P.goal_key(c)))
            if lo is None or lg is None:
                continue
            xs.append((lo, lg, v["obs"] - v["pred0"], v["obs"], v["g"], v["p"]))
        X = np.array(xs)
        out["l90_g5"] = dict(tasks=len(X), **{f"{a}_vs_{b}": dict(rho=float(S.spearmanr(X[:, i], X[:, j]).correlation),
                                                                  p=float(S.spearmanr(X[:, i], X[:, j]).pvalue))
                                              for i, a in ((0, "object_loss"), (1, "goal_loss"))
                                              for j, b in ((2, "residual"), (3, "composed_success"), (4, "g_object_unit"), (5, "p_goal_unit"))})
        out["l90_g5"]["abs_residual_vs_goal_loss"] = float(S.spearmanr(X[:, 1], np.abs(X[:, 2])).correlation)
        out["l90_g5"]["abs_residual_vs_object_loss"] = float(S.spearmanr(X[:, 0], np.abs(X[:, 2])).correlation)
        pts["l90"] = X
    for bk, r in ood.items():
        if "residual_by_task" not in r:
            continue
        xs = []
        for k, v in r["residual_by_task"].items():
            s, i = k.split(":")
            c = tasks[(s, int(i))].language
            lo, lg = held.get(("object", P.object_key(c))), held.get(("goal", P.goal_key(c)))
            xs.append((lo, lg, v))
        X = np.array([x for x in xs if x[0] is not None and x[1] is not None], float)
        out[f"ood_{bk}"] = dict(tasks_with_both_losses=int(len(X)), tasks=len(xs))
        if len(X) > 4:
            out[f"ood_{bk}"]["object_loss_vs_residual"] = float(S.spearmanr(X[:, 0], X[:, 2]).correlation)
            out[f"ood_{bk}"]["goal_loss_vs_residual"] = float(S.spearmanr(X[:, 1], X[:, 2]).correlation)
    return out, pts


def figures(l90, tab, ood, figdata, pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for ds, col in (("g5", "#1d5e86"), ("i7", "#a35d2b"), ("g20", "#2f7d4f"), ("g20_ood", "#8b5cf6")):
        r = l90.get(ds, {})
        if not r.get("per_task"):
            continue
        pt = r["per_task"]
        f0 = r["in_sample_zero_w"]
        ax[0].scatter([v["pred0"] for v in pt.values()], [v["obs"] for v in pt.values()], color=col, s=20,
                      label=f"{ds}: r {f0.get('pearson', float('nan')):.2f}, slope {f0.get('slope', float('nan')):.2f}"
                            f", {f0['observed']} obs / {f0['predicted']:.0f} pred")
    ax[0].plot([0, 1], [0, 1], color="#9ca3af", lw=1)
    ax[0].set_xlabel("predicted: g(object unit) x p(goal unit)")
    ax[0].set_ylabel("observed success, both units")
    ax[0].set_title("LIBERO-90: composed success against the product of components", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    r = l90["g5"]
    arms = ("zs", "obj", "goal", "both")
    lab = dict(zs="policy alone", obj="object unit only", goal="goal unit only", both="both units")
    x = np.arange(len(arms))
    ax[1].bar(x - 0.2, [r["arms"][a]["p_named"] for a in arms], 0.2, color="#1d5e86", label="P(first grasp = named object)")
    ax[1].bar(x, [r["arms"][a]["p_succ_named"] for a in arms], 0.2, color="#2f7d4f", label="P(success | named grasp)")
    ax[1].bar(x + 0.2, [r["arms"][a]["p_succ_other"] for a in arms], 0.2, color="#a35d2b", label="P(success | other grasp)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([lab[a] for a in arms], fontsize=8)
    ax[1].set_ylabel("rate (G5, 53 held-out tasks, set b)")
    ax[1].set_title("each arm's grasp and place competence", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    if "l90" in pts:
        X = pts["l90"]
        ax[2].scatter(X[:, 1], X[:, 2], color="#a35d2b", s=18, label="goal concept")
        ax[2].scatter(X[:, 0], X[:, 2], color="#1d5e86", s=18, marker="^", label="object concept")
        ax[2].axhline(0, color="#9ca3af", lw=1)
        ax[2].set_xlabel("frozen policy's flow loss on the concept's frames (i6)")
        ax[2].set_ylabel("observed - g x p (both units)")
        ax[2].set_title("does the policy's likelihood predict where the product fails?", fontsize=9)
        ax[2].legend(fontsize=7, frameon=False)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h1_factor_l90.png"), dpi=150)
    plt.close(fig)
    bks = [b for b in ("pi05", "pi0", "smolvla", "xvla") if b in figdata]
    if bks:
        fig, ax = plt.subplots(1, len(bks), figsize=(4.2 * len(bks), 4.2), sharey=True)
        SC = dict(c="#1d5e86", d="#2f7d4f", e="#a35d2b", f="#8b5cf6")
        for a_, b in zip(np.atleast_1d(ax), bks):
            d = figdata[b]
            a_.scatter(d["loo"], d["loo_obs"], color="#d1d5db", s=8, label="set a, leave one cell out")
            for s in sorted(set(d["set"])):
                ii = [i for i, z in enumerate(d["set"]) if z == s]
                a_.scatter([d["pred"][i] for i in ii], [d["obs"][i] for i in ii], color=SC.get(s, "#111827"), s=18, label=f"set {s}")
            o = ood[b]["out_of_sample"]
            a_.plot([0, 1], [0, 1], color="#9ca3af", lw=1)
            a_.set_title(f"{b}: r {o.get('pearson', float('nan')):.2f}, slope {o.get('slope', float('nan')):.2f}, "
                         f"{o['observed']} obs / {o['predicted']:.0f} pred", fontsize=8)
            a_.set_xlabel("predicted from practice (set a): g(a) p(g) + (1 - g) w(g)", fontsize=7)
            a_.spines[["top", "right"]].set_visible(False)
        np.atleast_1d(ax)[0].set_ylabel("observed success of the grasp switch pfield@a.g")
        np.atleast_1d(ax)[0].legend(fontsize=6, frameon=False)
        fig.suptitle("Li's OOD suites: composed success predicted from the sources' competences, on start states practice never saw",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "h1_factor_ood.png"), dpi=150)
        plt.close(fig)


def main(record):
    named = named_objects()
    l90, tab = l90_read(named)
    ood, figdata = ood_read(named)
    lik, pts = likelihood(l90, ood)
    figures(l90, tab, ood, figdata, pts)
    res = dict(l90=l90, ood=ood, likelihood=lik)
    json.dump(res, open(os.path.join(R, "h1_factor.json"), "w"), indent=1)
    for ds, r in l90.items():
        if not r.get("tasks"):
            print(f"[h1f] {ds}: no tasks with all four arms", flush=True)
            continue
        print(f"[h1f] L90 {ds} ({r['tasks']} tasks): arms {json.dumps({k: (v['successes'], v['named_first'], v['episodes'], round(v['p_succ_named'], 3), round(v['p_succ_other'], 3)) for k, v in r['arms'].items()})}",
              flush=True)
        print(f"[h1f]   in sample {r['in_sample']}", flush=True)
        print(f"[h1f]   out of sample {r.get('out_of_sample')}", flush=True)
        print(f"[h1f]   baseline max component {r['baseline_max_component']}", flush=True)
        print(f"[h1f]   interface both-vs-goal {r.get('interface_both_vs_goal')} | obj-vs-zs {r.get('interface_obj_vs_zs')} | "
              f"grasp both/obj {r['grasp_both_vs_obj']}", flush=True)
    for bk, r in ood.items():
        print(f"[h1f] OOD {bk}: {json.dumps({k: v for k, v in r.items() if k != 'residual_by_task'})}", flush=True)
    print(f"[h1f] likelihood {json.dumps(lik)}", flush=True)
    if record:
        from lifelong_ood import ledger
        z0 = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
        cfg = dict(shrink=SHRINK)
        for ds, r in l90.items():
            if r.get("tasks"):
                print(ledger.record("h1_factor", bench="libero_ood" if ds.endswith("ood") else "libero_90", tag=f"l90_{ds}", seed=0, metrics=r, bytes=z0, config=cfg,
                                    script="experiments/h1_factor.py", command=" ".join(sys.argv)))
        for bk, r in ood.items():
            print(ledger.record("h1_factor", bench="libero_ood", tag=f"ood_{bk}", seed=0, metrics=r, bytes=z0, config=cfg,
                                script="experiments/h1_factor.py", command=" ".join(sys.argv)))
        print(ledger.record("h1_factor", bench="libero", tag="likelihood", seed=0, metrics=lik, bytes=z0, config=cfg,
                            script="experiments/h1_factor.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main("--record" in sys.argv)
