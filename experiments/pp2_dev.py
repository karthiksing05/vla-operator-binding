"""P2 DEVELOPMENT PRE-CHECK (docs/PLAN_V4.md section 8, P2 row), on the existing pi05_libero checkpoint: are the action modes
under part prompts distinct, interpretable and transferable?  Reads results/protos/obs/*.npz (experiments/pp_fields.py),
results/protos/poses/*.npz and the f0b first frames (experiments/pp_prep.py).  numpy only.

Per level u (flow time of the ascent):
  separation    between-condition mode distance / within-condition spread over the 4 shared starts (rms over the 70 normalised
                action dims), over the 9 core conditions (null, instruction, 3 object parts, 2 goal parts, 2 verb parts);
                go: > 2 at some u <= 0.35 on >= 60% of observations
  distinct      clusters among the 9 core condition modes (greedy, rms tolerance 0.05 and 0.15)
  attribution   the mean mode chunk (unnormalised; end-effector path integrated with the demo-calibrated displacement scale) points
                (cosine of horizontal displacement with hand -> body) at a body of the part's TYPE: object parts on frames before the
                grasp, over movable objects; goal parts on frames after it, over every body.  chance = the share of candidates of
                that type.  go: accuracy >= 2 x chance
  precision     log10 Laplace precision by kind x (pos / rot / grip) x chunk half
  transfer      across the 10 init states of a task (first frames), Pearson r of the named-object part's mode endpoint with the
                named object's position (x and y, averaged), and of the displacement with hand -> object; go: >= 0.5
  hierarchy     median symmetric KL of each key to its parents (along u, along condition granularity)

    python experiments/pp2_dev.py [--record]      (via SLURM: slurm/jobs/cpu.sbatch)
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "protos")
F0B = os.path.join(ROOT, "results", "robot", "f0b_firstframes")
CORE = ("null", "instr", "obj:named", "obj:d1", "obj:d2", "goal:own", "goal:d1", "verb:own", "verb:d1")
PARTS_OBJ = ("obj:named", "obj:d1", "obj:d2", "obj:named:plain", "obj:named:tmpl")
PARTS_GOAL = ("goal:own", "goal:d1", "goal:own:plain", "goal:own:tmpl")


def rms(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def clusters(M, tol):
    cs = []
    for x in M:
        if not any(rms(c, x) < tol for c in cs):
            cs.append(x)
    return len(cs)


def scale_from_demos(idx):
    from lifelong_ood.policy import data as D
    from lifelong_ood.protos import store as S
    demos = D.Demos()
    eps = sorted({o["episode"] for o in idx["obs"] if o["set"] == "train"})
    st, ac = [], []
    for e in eps:
        ep = demos.load(e)
        st.append(ep["state"])
        ac.append(ep["actions"])
    return S.displacement_scale(st, ac)


def task_types():
    from lifelong_ood.policy import factors as F
    out = {}
    for t in F.load():
        out[(t.suite, t.index)] = (dict(t.objects, **t.fixtures), t)
    return out


def main(record=False, tag=""):
    from lifelong_ood.protos import store as S
    idx = json.load(open(os.path.join(OUT, "index.json")))
    od = os.path.join(OUT, "obs" + (f"_{tag}" if tag else ""))
    scale = scale_from_demos(idx)
    TT = task_types()
    L = len(S.LEVELS)
    rec = dict(sep={l: [] for l in range(L)}, sep_all={l: [] for l in range(L)}, n05={l: [] for l in range(L)},
               n15={l: [] for l in range(L)}, att={}, prec={}, kl={"u": [], "c": []})
    per_set = {}
    n_obs = 0
    for o in idx["obs"]:
        if o["set"] == "transfer":
            continue
        p = os.path.join(od, o["key"] + ".npz")
        pp = os.path.join(OUT, "poses", o["key"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        n_obs += 1
        names = [str(x) for x in z["names"]]
        kinds = [str(x) for x in z["kinds"]]
        targets = [str(x) for x in z["targets"]]
        x0, ext, raw = z["x0"], z["ext"], z["x0_raw"]                 # [L, C, S, dim], [L, C, S, dim], [L, C, S, H, 7]
        core = [names.index(n) for n in CORE if n in names]
        for l in range(L):
            for which, sel in (("sep", core), ("sep_all", list(range(len(names))))):
                M = x0[l, sel].mean(1)
                within = np.mean([np.mean([rms(x0[l, c, s], M[i]) for s in range(x0.shape[2])]) for i, c in enumerate(sel)])
                between = np.mean([rms(M[i], M[j]) for i in range(len(sel)) for j in range(i + 1, len(sel))])
                rec[which][l].append(float(between / max(within, 1e-9)))
                if which == "sep":
                    per_set.setdefault(o["set"], {}).setdefault(l, []).append(float(between / max(within, 1e-9)))
            M = x0[l, core].mean(1)
            rec["n05"][l].append(clusters(M, 0.05))
            rec["n15"][l].append(clusters(M, 0.15))
            for c, k in enumerate(kinds):
                prof = S.precision_profile(ext[l, c].mean(0))
                rec["prec"].setdefault(k, {}).setdefault(l, []).append(prof)
        # hierarchy
        st = S.Store([S.Condition(n, k, "", variant=str(v)) for n, k, v in zip(names, kinds, z["variants"])], S.LEVELS)
        for l, u in enumerate(S.LEVELS):
            for c, n in enumerate(names):
                st.add(n, u, x0[l, c], ext[l, c])
        for a, b, kind in st.edges():
            rec["kl"][kind].append(st.sym_kl(a, b))
        # attribution
        if not os.path.exists(pp):
            continue
        q = np.load(pp, allow_pickle=True)
        bodies = [str(x) for x in q["names"]]
        bkind = [str(x) for x in q["kinds"]]
        types, _ = TT[(o["suite"], o["index"])]
        btype = [types.get(b, b) for b in bodies]
        eef = q["eef"]
        named_t = targets[names.index("obj:named")] if "obj:named" in names else ""
        goal_t = targets[names.index("goal:own")] if "goal:own" in names else ""
        jobs = []
        for c, n in enumerate(names):
            if n in PARTS_OBJ and o["phase"] == 0:
                jobs.append((n, c, targets[c], "object"))
            elif n in PARTS_GOAL and o["phase"] == 1:
                jobs.append((n, c, targets[c], "any"))
            elif n in ("null", "instr"):                     # controls: the same targets without a part prompt
                if o["phase"] == 0 and named_t:
                    jobs.append((f"{n}->obj", c, named_t, "object"))
                if o["phase"] == 1 and goal_t:
                    jobs.append((f"{n}->goal", c, goal_t, "any"))
        for n, c, T, which in jobs:
            cand = [i for i, k in enumerate(bkind) if k == "object"] if which == "object" else list(range(len(bodies)))
            if not T or not any(btype[i] == T for i in cand):
                continue
            chance = float(np.mean([btype[i] == T for i in cand]))
            for l in range(L):
                it = S.interpret(raw[l, c].mean(0), eef, q["pos"][cand], [bodies[i] for i in cand], scale)
                hit = btype[cand[it["pointed"]]] == T
                near = btype[cand[it["nearest"]]] == T
                rec["att"].setdefault(n, {}).setdefault(l, []).append((hit, near, chance, o["set"]))
    # transfer
    tr = {}
    for o in idx["obs"]:
        if o["set"] != "transfer":
            continue
        p = os.path.join(od, o["key"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        f = np.load(os.path.join(F0B, f"{o['suite']}_{o['index']}.npz"), allow_pickle=True)
        types, task = TT[(o["suite"], o["index"])]
        from lifelong_ood.policy import factors as F
        pt = S.parts_of(task, F.scene_phrases(task, [v[1] for v in TT.values()]))
        bodies = [str(x) for x in f["names"]]
        if pt["obj_inst"] not in bodies:
            continue
        j = bodies.index(pt["obj_inst"])
        s = o["state"]
        eef = np.asarray(f["state8"][s][:3], np.float64)
        objp = np.asarray(f["pos"][s][j], np.float64)
        names = [str(x) for x in z["names"]]
        for n in ("obj:named", "obj:named:tmpl", "instr", "null"):
            if n not in names:
                continue
            c = names.index(n)
            for l in range(L):
                it = S.interpret(z["x0_raw"][l, c].mean(0), eef, objp[None], [pt["obj_inst"]], scale)
                tr.setdefault((o["suite"], o["index"]), {}).setdefault(n, {}).setdefault(l, []).append(
                    (it["endpoint"][:2], objp[:2], (it["endpoint"] - eef)[:2], (objp - eef)[:2]))

    def corr2(A, B):
        A, B = np.asarray(A), np.asarray(B)
        rs = [np.corrcoef(A[:, k], B[:, k])[0, 1] for k in range(2) if A[:, k].std() > 1e-9 and B[:, k].std() > 1e-9]
        return float(np.mean(rs)) if rs else float("nan")

    transfer = {}
    for n in ("obj:named", "obj:named:tmpl", "instr", "null"):
        transfer[n] = {}
        for l in range(L):
            r_end, r_disp = [], []
            for task, d in tr.items():
                v = d.get(n, {}).get(l, [])
                if len(v) < 5:
                    continue
                r_end.append(corr2([x[0] for x in v], [x[1] for x in v]))
                r_disp.append(corr2([x[2] for x in v], [x[3] for x in v]))
            transfer[n][str(S.LEVELS[l])] = dict(endpoint_r_median=float(np.nanmedian(r_end)) if r_end else None,
                                                 displacement_r_median=float(np.nanmedian(r_disp)) if r_disp else None,
                                                 endpoint_r_per_task=[round(x, 3) for x in r_end], tasks=len(r_end))
    # summaries
    lv = {}
    for l, u in enumerate(S.LEVELS):
        sep = np.asarray(rec["sep"][l])
        d = dict(n_obs=int(len(sep)), sep_median=float(np.median(sep)) if len(sep) else None,
                 frac_sep_gt2=float((sep > 2).mean()) if len(sep) else None,
                 sep_all_median=float(np.median(rec["sep_all"][l])) if rec["sep_all"][l] else None,
                 frac_sep_gt2_by_set={k: float((np.asarray(v[l]) > 2).mean()) for k, v in per_set.items()},
                 distinct_tol05=float(np.mean(rec["n05"][l])) if rec["n05"][l] else None,
                 distinct_tol15=float(np.mean(rec["n15"][l])) if rec["n15"][l] else None, attribution={})
        for n, byl in rec["att"].items():
            v = byl.get(l, [])
            if not v:
                continue
            hit = np.mean([x[0] for x in v])
            near = np.mean([x[1] for x in v])
            ch = np.mean([x[2] for x in v])
            d["attribution"][n] = dict(n=len(v), pointed=float(hit), nearest=float(near), chance=float(ch),
                                       pointed_over_chance=float(hit / ch), by_set={s: float(np.mean([x[0] for x in v if x[3] == s]))
                                                                                   for s in {x[3] for x in v}})
        lv[str(u)] = d
    prec = {k: {str(S.LEVELS[l]): {g: float(np.mean([p[g] for p in ps])) for g in ps[0]} for l, ps in byl.items()}
            for k, byl in rec["prec"].items()}
    best_u = [u for u in S.LEVELS if u <= 0.35]
    go_sep = max(lv[str(u)]["frac_sep_gt2"] or 0 for u in best_u)
    att_named = max((lv[str(u)]["attribution"].get("obj:named", {}).get("pointed_over_chance", 0) for u in S.LEVELS), default=0)
    att_goal = max((lv[str(u)]["attribution"].get("goal:own", {}).get("pointed_over_chance", 0) for u in S.LEVELS), default=0)
    att_null_obj = max((lv[str(u)]["attribution"].get("null->obj", {}).get("pointed_over_chance", 0) for u in S.LEVELS), default=0)
    att_null_goal = max((lv[str(u)]["attribution"].get("null->goal", {}).get("pointed_over_chance", 0) for u in S.LEVELS), default=0)
    att_bare = {u: np.mean([lv[str(u)]["attribution"][n]["pointed_over_chance"] for n in ("obj:named", "obj:d1", "obj:d2")
                            if n in lv[str(u)]["attribution"]]) for u in S.LEVELS}
    tr_best = max((transfer["obj:named"][str(u)]["endpoint_r_median"] or -1) for u in S.LEVELS)
    go = dict(separation_frac_best_u_le_0_35=go_sep, separation_go=bool(go_sep >= 0.6),
              attribution_named_best=float(att_named), attribution_goal_own_best=float(att_goal),
              attribution_null_to_named_obj=float(att_null_obj), attribution_null_to_own_goal=float(att_null_goal), attribution_obj_parts_mean_best=float(max(att_bare.values())),
              attribution_go=bool(max(att_bare.values()) >= 2.0),
              transfer_endpoint_r_best=float(tr_best), transfer_go=bool(tr_best >= 0.5))
    res = dict(n_obs=n_obs, scale=scale.tolist(), levels=list(S.LEVELS), by_level=lv, precision=prec, transfer=transfer,
               hierarchy=dict(sym_kl_u_median=float(np.median(rec["kl"]["u"])) if rec["kl"]["u"] else None,
                              sym_kl_c_median=float(np.median(rec["kl"]["c"])) if rec["kl"]["c"] else None), go=go)
    path = os.path.join(OUT, f"pp2_dev{('_' + tag) if tag else ''}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(dict(go=go, by_level={u: {k: v for k, v in d.items() if k != "attribution"} for u, d in lv.items()}), indent=1))
    for u, d in lv.items():
        print(u, {n: (round(a["pointed"], 2), round(a["chance"], 2), a["n"]) for n, a in d["attribution"].items()})
    print("transfer", {n: {u: v["endpoint_r_median"] for u, v in t.items()} for n, t in transfer.items()})
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pp2_dev", bench="libero", tag=tag or "pi05_libero", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(levels=list(S.LEVELS), core=list(CORE)), script="experiments/pp2_dev.py",
                            command=" ".join(sys.argv), backbone=dict(config="pi05_libero"), artifacts=[path]))


if __name__ == "__main__":
    tag = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--tag=")), "")
    main("--record" in sys.argv, tag)
