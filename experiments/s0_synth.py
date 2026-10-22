"""S0: the composition machinery (lifelong_ood/compose) on the synthetic manipulation domain.

Measures (docs/S0_SYNTH.md):
  (a) boundary F1 of parses vs the true unit changes (tolerance +-2 frames), and a contact-event baseline;
  (b) taxonomy recovery of units (object / goal as diagnostics only): purity per depth, basic level,
      unit-concept recall, constituent precision of committed composites;
  (c) G-rec: one demo of a held-out object x goal: does the parse use only stored elements, and does
      generation reproduce the new unit sequence (zero-demo / after one demo / generic seed after one demo);
  (d) G-new: a new object: NEW remainder, creation, reuse by a second task;
  (e) continual: 30-task stream, store size vs tasks and parts, preservation of earlier recognition;
  (f) ablations: flat stores, no content/context split, no "with".

Usage:  python experiments/s0_synth.py --configs full,flat,nosplit,nowith --seeds 0,1,2 --procs 4
        python experiments/s0_synth.py --quick      (small sizes, for the test suite)
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.setrecursionlimit(100000)

from lifelong_ood.compose.generate import context_fn_obs, generate          # noqa: E402
from lifelong_ood.compose.learn import calibrate_content, calibrate_windows, discover, learn   # noqa: E402
from lifelong_ood.compose.parse import parse                                # noqa: E402
from lifelong_ood.compose.store import Config, Store                        # noqa: E402
from lifelong_ood.compose.synth import D, KOBS, Domain, Env                 # noqa: E402

CONFIGS = {
    "full": {},
    "flat": {"flat": True},
    "nosplit": {"split": False},
    "nowith": {"with_on": False},
}


# ---- metrics -----------------------------------------------------------------------------------------
def boundary_prf(pred, true, tol=2):
    pred, true = sorted(pred), sorted(true)
    used, tp = set(), 0
    for b in pred:
        best = None
        for j, t in enumerate(true):
            if j not in used and abs(b - t) <= tol and (best is None or abs(b - t) < abs(b - true[best])):
                best = j
        if best is not None:
            used.add(best)
            tp += 1
    P = tp / len(pred) if pred else (1.0 if not true else 0.0)
    R = tp / len(true) if true else 1.0
    F = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return P, R, F


def collapse(seq):
    out = []
    for u in seq:
        if not out or out[-1] != u:
            out.append(u)
    return out


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def seq_metrics(gen, true):
    g = collapse(gen)
    ed = edit_distance(g, true)
    return {"exact": float(g == true), "unit_acc": max(0.0, 1 - ed / len(true)), "len": len(g)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ---- build -------------------------------------------------------------------------------------------
def build(dom, cfg, n_disc, n_cal, objects=None, seed=0):
    combos = [c for c in dom.train_combos if objects is None or c[0] in objects]
    st = Store(D, KOBS, cfg)
    train = [dom.demo(dom.sample_task(combos)) for _ in range(n_disc)]
    for i, d in enumerate(train):
        discover(st, d.Z, d.O, tag=("disc", i))
    cal = [dom.demo(dom.sample_task(combos)) for _ in range(n_cal)]
    st.calibrate_cover(np.concatenate([c.Z for c in cal]))
    calibrate_windows(st, [(c.Z, c.O) for c in cal])
    n_created = 0
    learned = []
    for i, d in enumerate(train):
        r = learn(st, d.Z, d.O, label_fn=d.label, tag=("train", i))
        n_created += len(r["created"])
        learned.append(r["parse"])
    calibrate_content(st, [(c.Z, c.O) for c in cal])
    return st, train, learned, cal, n_created


def parse_stats(st, dom, demos):
    rows = []
    for d in demos:
        p = parse(st, d.Z, d.O, gated=True)
        P, R, F = boundary_prf(p.boundaries(), d.boundaries())
        cP, cR, cF = boundary_prf(dom.contact_boundaries(d), d.boundaries())
        rows.append({"P": P, "R": R, "F1": F, "contact_F1": cF, "new_frac": p.new_frac(),
                     "complete": float(p.complete()), "n_windows": len(p.windows), "n_units": len(d.units),
                     "frontier": len(p.frontier),
                     "with_frac": mean([float(w.kind == "with") for w in p.windows])})
    return {k: mean([r[k] for r in rows]) for k in rows[0]}


# ---- (b) taxonomy recovery ---------------------------------------------------------------------------
def unit_label(lab):
    return lab[0] if len(lab) == 1 else None


def taxonomy_stats(st):
    out = {}
    win = {k for k, e in st.elements.items() if e.kind in ("prim", "with")}
    lab = {k: unit_label(st.labels[k]) for k in win if k in st.labels}
    lab = {k: v for k, v in lab.items() if v is not None}
    # members of each ctx node restricted to window instances
    node_members = {}
    for n in st.ctx.nodes():
        ms = [lab[m] for m in n.all_members() if m in lab]
        if ms:
            node_members[n.id] = (n, ms)
    by_depth = defaultdict(lambda: [0, 0, 0])
    for nid, (n, ms) in node_members.items():
        dpt = n.depth()
        if dpt == 0 or dpt > 8:
            continue
        ty = Counter(m[0] for m in ms).most_common(1)[0][1]
        un = Counter(ms).most_common(1)[0][1]
        b = by_depth[dpt]
        b[0] += len(ms); b[1] += ty; b[2] += un
    out["ctx_purity_by_depth"] = {d: {"type": b[1] / b[0], "unit": b[2] / b[0], "n": b[0]}
                                  for d, b in sorted(by_depth.items())}
    # basic level of each window's leaf
    idx = st.ctx.node_index()
    bl_depth, bl_type, bl_unit, bl_size = [], [], [], []
    for k, v in lab.items():
        leaf = st.elements[k].ctx_node
        if leaf is None or leaf.id not in idx:
            continue
        b = leaf.basic()
        ms = node_members.get(b.id, (None, [v]))[1]
        bl_depth.append(b.depth())
        bl_type.append(np.mean([m[0] == v[0] for m in ms]))
        bl_unit.append(np.mean([m == v for m in ms]))
        bl_size.append(len(ms))
    out["basic_level"] = {"depth": mean(bl_depth), "type_purity": mean(bl_type), "unit_purity": mean(bl_unit),
                          "size": mean(bl_size)}
    # unit-concept recall: a node whose window members are >=90% one unit and hold >=80% of that unit
    tot = Counter(lab.values())
    rec = {}
    for u, cnt in tot.items():
        ok = False
        for nid, (n, ms) in node_members.items():
            if n is st.ctx.root:
                continue
            c = Counter(ms)
            if c[u] >= 0.8 * cnt and c[u] >= 0.9 * len(ms):
                ok = True
                break
        rec[u] = ok
    out["unit_concept_recall"] = mean([float(v) for v in rec.values()])
    type_tot = Counter(u[0] for u in lab.values())
    trec = []
    for ty, cnt in type_tot.items():
        ok = False
        for nid, (n, ms) in node_members.items():
            if n is st.ctx.root:
                continue
            c = Counter(m[0] for m in ms)
            if c[ty] >= 0.8 * cnt and c[ty] >= 0.9 * len(ms):
                ok = True
                break
        trec.append(float(ok))
    out["type_concept_recall"] = mean(trec)
    # content taxonomy: purity of composites w.r.t. their unit-type sequence
    if st.cfg.split:
        comp = {k: tuple(u[0] for u in st.labels[k]) for k, e in st.elements.items()
                if e.kind == "then" and k in st.labels}
        cd = defaultdict(lambda: [0, 0])
        for n in st.cnt.nodes():
            ms = [comp[m] for m in n.all_members() if m in comp]
            dpt = n.depth()
            if not ms or dpt == 0 or dpt > 8:
                continue
            cd[dpt][0] += len(ms)
            cd[dpt][1] += Counter(ms).most_common(1)[0][1]
        out["cnt_purity_by_depth"] = {d: b[1] / b[0] for d, b in sorted(cd.items())}
    return out


def constituent_precision(learned, demos):
    hit, tot, kinds = 0, 0, Counter()
    for p, d in zip(learned, demos):
        gold = {(s, e): lab for lab, s, e in d.constituents}
        for c in p.commits:
            tot += 1
            # a commit matches a constituent if its span is within 2 frames at both ends
            m = [lab for (s, e), lab in gold.items() if abs(c.span[0] - s) <= 2 and abs(c.span[1] - e) <= 2]
            if m:
                hit += 1
                kinds[m[0]] += 1
    return {"precision": hit / tot if tot else None, "matched": dict(kinds), "n": tot}


# ---- (c) G-rec ---------------------------------------------------------------------------------------
def gen_seq(st, dom, seed_el, combos):
    env = Env(dom, combos)
    fn = context_fn_obs(env.obs)
    out, trace = generate(st, seed_el, fn, emit_fn=lambda el: env.step(el.code))
    return env.executed, trace


def generic_seed(st, learned, train):
    for p, d in zip(learned, train):
        if len(d.combos) == 1 and p.root is not None:
            return p.root
    return None


def grec(st, dom, learned, train):
    rows = []
    gseed = generic_seed(st, learned, train)
    for h in dom.heldout:
        s2 = copy.deepcopy(st)
        d = dom.demo([h])
        true = d.unit_seq()
        p = parse(s2, d.Z, d.O, gated=True)
        carry = [w for w in p.windows if any(u[0] == "carry" for u in d.label(w.span))]
        row = {"combo": list(h), "new_frac": p.new_frac(), "complete": float(p.complete()),
               "frontier": len(p.frontier), "F1": boundary_prf(p.boundaries(), d.boundaries())[2],
               "carry_with": float(bool(carry) and all(w.kind == "with" for w in carry))}
        if gseed is not None:
            seed_copy = s2.elements.get(gseed.key, gseed)
            g0, _ = gen_seq(s2, dom, seed_copy, [h])
            row["zero_demo"] = seq_metrics(g0, true)
        r = learn(s2, d.Z, d.O, label_fn=d.label, tag=("grec", h))
        pr = r["parse"].root
        if pr is not None:
            g1, tr = gen_seq(s2, dom, pr, [h])
            row["one_demo"] = seq_metrics(g1, true)
            row["one_demo"]["root_self"] = float(tr[0]["self"]) if tr else None
            row["one_demo"]["pool"] = tr[0]["pool"] if tr else None
            row["one_demo"]["seq"] = [list(map(str, u)) for u in collapse(g1)]
        else:
            row["one_demo"] = {"exact": 0.0, "unit_acc": 0.0, "len": 0, "note": "no root"}
        if gseed is not None:
            g2, _ = gen_seq(s2, dom, s2.elements.get(gseed.key, gseed), [h])
            row["generic_after_demo"] = seq_metrics(g2, true)
        rows.append(row)
    summ = {"new_frac": mean([r["new_frac"] for r in rows]), "complete": mean([r["complete"] for r in rows]),
            "carry_with": mean([r["carry_with"] for r in rows]), "F1": mean([r["F1"] for r in rows])}
    for k in ("zero_demo", "one_demo", "generic_after_demo"):
        summ[k + "_exact"] = mean([r[k]["exact"] for r in rows if k in r])
        summ[k + "_unit_acc"] = mean([r[k]["unit_acc"] for r in rows if k in r])
    return summ, rows


# ---- (d) G-new ---------------------------------------------------------------------------------------
def frac_new_in(p, d, pred):
    """Fraction of frames of units satisfying pred(unit) that lie in NEW windows."""
    newf = np.zeros(len(d.Z), bool)
    for s, e in p.new_spans():
        newf[s:e] = True
    sel = np.zeros(len(d.Z), bool)
    for (ty, arg, s, e) in d.units:
        if pred(ty, arg):
            sel[s:e] = True
    return float(newf[sel].mean()) if sel.any() else None


def gnew(st, dom):
    o = dom.new_obj
    has_new = lambda ty, arg: (ty in ("approach", "grasp") and arg == o) or (ty == "carry" and arg[0] == o)
    no_new = lambda ty, arg: not has_new(ty, arg)
    out = {}
    for create_on in (True, False):
        s2 = copy.deepcopy(st)
        d1 = dom.demo([(o, 0)])
        p1 = parse(s2, d1.Z, d1.O, gated=True)
        row = {"d1_new_frac_newobj_units": frac_new_in(p1, d1, has_new),
               "d1_new_frac_other_units": frac_new_in(p1, d1, no_new),
               "d1_F1": boundary_prf(p1.boundaries(), d1.boundaries())[2]}
        n0 = len(s2.pool()[0])
        r = learn(s2, d1.Z, d1.O, label_fn=d1.label, allow_create=create_on, tag=("gnew", 1))
        row["created_prototypes"] = len(r["created"])
        row["pool_growth"] = len(s2.pool()[0]) - n0
        p1b = parse(s2, d1.Z, d1.O, gated=True)
        row["d1_reparse_new_frac"] = p1b.new_frac()
        row["d1_reparse_complete"] = float(p1b.complete())
        d2 = dom.demo([(o, 2)])
        p2 = parse(s2, d2.Z, d2.O, gated=True)
        row["d2_new_frac_newobj_units"] = frac_new_in(p2, d2, has_new)
        row["d2_new_frac"] = p2.new_frac()
        row["d2_complete"] = float(p2.complete())
        row["d2_F1"] = boundary_prf(p2.boundaries(), d2.boundaries())[2]
        created = s2.created
        uses = []
        for w in p2.windows:
            lab = d2.label(w.span)
            if len(lab) == 1 and has_new(*lab[0]) and not w.new:
                uses.append(float(any(c in created for c in w.cover)))
        row["d2_newobj_windows_using_created"] = mean(uses)
        row["d2_newobj_carry_with"] = mean([float(w.kind == "with") for w in p2.windows
                                            if len(d2.label(w.span)) == 1 and d2.label(w.span)[0][0] == "carry"])
        r2 = learn(s2, d2.Z, d2.O, label_fn=d2.label, allow_create=create_on, tag=("gnew", 2))
        if r2["parse"].root is not None:
            g, _ = gen_seq(s2, dom, r2["parse"].root, [(o, 2)])
            row["d2_one_demo_gen"] = seq_metrics(g, d2.unit_seq())
        out["create" if create_on else "no_create"] = row
    return out


# ---- (e) continual -----------------------------------------------------------------------------------
def continual(dom, cfg, n_tasks=30, n_disc=10, n_cal=5, schedule=((0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4))):
    st = Store(D, KOBS, cfg)
    base = [c for c in dom.train_combos if c[0] in (0, 1)]
    for i in range(n_disc):
        d = dom.demo(dom.sample_task(base))
        discover(st, d.Z, d.O, tag=("cdisc", i))
    cal = [dom.demo(dom.sample_task(base)) for _ in range(n_cal)]
    st.calibrate_cover(np.concatenate([c.Z for c in cal]))
    calibrate_windows(st, [(c.Z, c.O) for c in cal])
    for c in cal:
        learn(st, c.Z, c.O, label_fn=c.label, tag=("ccal",))
    calibrate_content(st, [(c.Z, c.O) for c in cal])
    tasks, curve, rec = [], [], {}
    parts_seen = set()
    per = max(1, n_tasks // len(schedule))
    for t in range(n_tasks):
        objs = schedule[min(t // per, len(schedule) - 1)]
        combos = [c for c in dom.train_combos if c[0] in objs]
        d = dom.demo(dom.sample_task(combos))
        tasks.append(d)
        parts_seen.update(d.unit_seq())
        r = learn(st, d.Z, d.O, label_fn=d.label, tag=("stream", t))
        sz = st.sizes()
        sz.update({"task": t + 1, "parts_seen": len(parts_seen), "created": len(r["created"]),
                   "new_frac_at_learn": r["parse"].new_frac()})
        rs = []
        for i, di in enumerate(tasks):
            p = parse(st, di.Z, di.O, gated=True)
            rec.setdefault(i, []).append((t, 1 - p.new_frac(), float(p.complete())))
            rs.append((1 - p.new_frac(), float(p.complete())))
        sz["recog_cover_mean"] = mean([a for a, _ in rs])
        sz["recog_complete_mean"] = mean([b for _, b in rs])
        curve.append(sz)
    drops_cover, drops_complete, worst = 0, 0, 0.0
    for i, hist in rec.items():
        c0, k0 = hist[0][1], hist[0][2]
        for (_, c, k) in hist[1:]:
            if c < c0 - 0.02:
                drops_cover += 1
                worst = max(worst, c0 - c)
            if k < k0:
                drops_complete += 1
    # growth: fit store size ~ a * tasks + b * parts
    T = np.array([c["task"] for c in curve], float)
    Pp = np.array([c["parts_seen"] for c in curve], float)
    out = {"curve": curve, "drops_cover": drops_cover, "drops_complete": drops_complete,
           "worst_cover_drop": worst, "n_pairs": sum(len(h) - 1 for h in rec.values())}
    for key in ("prototypes", "ctx_nodes", "cnt_nodes", "rules", "elements"):
        if key in curve[0]:
            y = np.array([c[key] for c in curve], float)
            out[f"{key}_corr_tasks"] = float(np.corrcoef(T, y)[0, 1]) if y.std() > 0 else None
            out[f"{key}_corr_parts"] = float(np.corrcoef(Pp, y)[0, 1]) if y.std() > 0 and Pp.std() > 0 else None
            # increments within an object phase (parts fixed) vs at phase changes
            dy = np.diff(y)
            dp = np.diff(Pp)
            out[f"{key}_inc_when_new_parts"] = mean(list(dy[dp > 0])) if (dp > 0).any() else None
            out[f"{key}_inc_when_no_new_parts"] = mean(list(dy[dp == 0])) if (dp == 0).any() else None
    return out


# ---- one run -----------------------------------------------------------------------------------------
def run(config, seed, quick=False, overrides=None):
    t0 = time.time()
    cfg = Config(**{**CONFIGS[config], **(overrides or {})}, seed=seed)
    dom = Domain(seed)
    n_disc, n_cal, n_test = (12, 4, 6) if quick else (40, 10, 20)
    st, train, learned, cal, n_created_train = build(dom, cfg, n_disc, n_cal)
    res = {"config": config, "seed": seed, "theta_cover": st.theta_cover, "theta_new": st.theta_new, "theta_content": st.theta_content,
           "train_spurious_created": n_created_train, "sizes": st.sizes()}
    test = [dom.demo(dom.sample_task(dom.train_combos)) for _ in range(n_test)]
    res["a_test"] = parse_stats(st, dom, test)
    res["a_train_learned_F1"] = mean([boundary_prf(p.boundaries(), d.boundaries())[2]
                                      for p, d in zip(learned, train)])
    res["b"] = taxonomy_stats(st)
    res["b"]["constituents"] = constituent_precision(learned, train)
    res["c"], res["c_rows"] = grec(st, dom, learned, train)
    res["d"] = gnew(st, dom)
    res["e"] = continual(Domain(seed + 1000), cfg, n_tasks=9 if quick else 30, n_disc=6 if quick else 10,
                         n_cal=3 if quick else 5)
    res["seconds"] = time.time() - t0
    return jsonable(res)


def _run(args):
    config, seed, quick, outdir, ov = args
    r = run(config, seed, quick, ov)
    r["overrides"] = ov
    with open(os.path.join(outdir, f"{config}_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    return r


def summarize(results):
    def g(r, path):
        for k in path:
            if r is None or k not in r:
                return None
            r = r[k]
        return r
    keys = {
        "a_F1": ("a_test", "F1"), "a_contact_F1": ("a_test", "contact_F1"), "a_new_frac": ("a_test", "new_frac"),
        "a_complete": ("a_test", "complete"), "a_with_frac": ("a_test", "with_frac"),
        "b_unit_recall": ("b", "unit_concept_recall"), "b_type_recall": ("b", "type_concept_recall"),
        "b_basic_depth": ("b", "basic_level", "depth"), "b_basic_type_purity": ("b", "basic_level", "type_purity"),
        "b_basic_unit_purity": ("b", "basic_level", "unit_purity"),
        "b_constituent_precision": ("b", "constituents", "precision"),
        "c_new_frac": ("c", "new_frac"), "c_complete": ("c", "complete"), "c_carry_with": ("c", "carry_with"),
        "c_zero_demo_exact": ("c", "zero_demo_exact"), "c_one_demo_exact": ("c", "one_demo_exact"),
        "c_one_demo_unit_acc": ("c", "one_demo_unit_acc"),
        "c_generic_after_demo_exact": ("c", "generic_after_demo_exact"),
        "d_d1_new_newobj": ("d", "create", "d1_new_frac_newobj_units"),
        "d_d1_new_other": ("d", "create", "d1_new_frac_other_units"),
        "d_created": ("d", "create", "created_prototypes"),
        "d_d2_new_newobj": ("d", "create", "d2_new_frac_newobj_units"),
        "d_d2_using_created": ("d", "create", "d2_newobj_windows_using_created"),
        "d_d2_complete": ("d", "create", "d2_complete"),
        "d_d2_gen_exact": ("d", "create", "d2_one_demo_gen", "exact"),
        "d_nocreate_d2_new_newobj": ("d", "no_create", "d2_new_frac_newobj_units"),
        "e_drops_cover": ("e", "drops_cover"), "e_drops_complete": ("e", "drops_complete"),
        "e_n_pairs": ("e", "n_pairs"),
        "e_proto_inc_new_parts": ("e", "prototypes_inc_when_new_parts"),
        "e_proto_inc_no_new_parts": ("e", "prototypes_inc_when_no_new_parts"),
        "e_rules_inc_new_parts": ("e", "rules_inc_when_new_parts"),
        "e_rules_inc_no_new_parts": ("e", "rules_inc_when_no_new_parts"),
        "e_final_recog_complete": ("e", "curve", -1, "recog_complete_mean"),
        "e_final_recog_cover": ("e", "curve", -1, "recog_cover_mean"),
    }
    out = {}
    by_cfg = defaultdict(list)
    for r in results:
        by_cfg[r["config"]].append(r)
    for cfgname, rs in by_cfg.items():
        row = {}
        for name, path in keys.items():
            vals = []
            for r in rs:
                v = r
                for k in path:
                    if isinstance(v, list) and isinstance(k, int):
                        v = v[k] if v else None
                    elif isinstance(v, dict):
                        v = v.get(k)
                    else:
                        v = None
                    if v is None:
                        break
                if v is not None:
                    vals.append(float(v))
            row[name] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n": len(vals)} if vals else None
        out[cfgname] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="full,flat,nosplit,nowith")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "s0"))
    ap.add_argument("--set", default="", help="Config overrides key=val;key=val")
    a = ap.parse_args()
    ov = {}
    for kv in filter(None, a.set.split(";")):
        k, v = kv.split("=")
        cur = getattr(Config(), k)
        ov[k] = (v.lower() == "true") if isinstance(cur, bool) else type(cur)(v)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(c, int(s), a.quick, a.out, ov) for s in a.seeds.split(",") for c in a.configs.split(",")]
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as pool:
            results = pool.map(_run, jobs, chunksize=1)
    else:
        results = [_run(j) for j in jobs]
    summ = summarize(results)
    name = "summary_quick.json" if a.quick else "summary.json"
    with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    for c, row in summ.items():
        print(c)
        for k, v in row.items():
            print(f"  {k:32s} {('%.3f +- %.3f (n=%d)' % (v['mean'], v['sd'], v['n'])) if v else 'n/a'}")


if __name__ == "__main__":
    main()
