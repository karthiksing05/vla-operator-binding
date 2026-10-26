"""PROBE D5 read (docs/PLAN_D5.md; pre-registration experiments/prereg/pd5_goal.py): the goal-role binding operator from the sandboxed
release points (experiments/d5_goal.py --sandbox outputs).  Run through `experiments/d5_goal.py --read [--record]` (chunks env).

Per observation and readout: the readout point of every sandbox (PRIMARY: the hand's xy at the release, +-5 cm, mean over the noise
sets A1 and A2); J_Z (3 x 2, central differences; the 2 x 2 xy block primary) for every body whose 4 perturbations are valid; the bound
body = argmax ||J_Z,xy||_F; the noise floor from the unperturbed B-set pair.
  G1  training observations: the target body is the bound body in >= 0.8 and the CI's lower bound exceeds chance.
  G2  training observations: leave-type-out R^2 of the target's endpoint responses >= 0.5 and > the filler-shuffled control (d4_read.h2).
  G3  zs dev wrong-goal observations with an actual body other than the own target: the bound body is the actual one in >= 0.7.
Holm over G1, G2, G3 at 0.05.  Everything else is descriptive.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import d4_bind as D4                                                    # noqa: E402
import d4_read as DR                                                    # noqa: E402
import d5_goal as G                                                     # noqa: E402

OUT = G.OUT
READOUTS = ("release", "release_10", "release_xyz", "chunk_mean", "chunk_end", "obj_release")
PRIMARY = "release"


def scale():
    return DR.scale_for("l90c")


def points(z, meta, readout, sc):
    """-> per-box readout point [n_box, 3]."""
    if readout in ("release", "release_10", "release_xyz"):
        return np.asarray(z["rel_hand"], np.float64)
    if readout == "obj_release":
        return np.asarray(z["rel_obj"], np.float64)
    ch = np.asarray(z["chunk0"], np.float64)[..., :3] * np.asarray(sc, np.float64)
    h0 = np.asarray(z["h0"], np.float64)
    if readout == "chunk_end":
        return h0 + ch.sum(1)
    return h0 + np.cumsum(ch, 1).mean(1)                                 # chunk_mean: the mean of the chunk's integrated path


def obs_record(path, readout, sc):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    E = points(z, meta, readout, sc)
    lab = {b["label"]: i for i, b in enumerate(meta["boxes"])}
    mag = 0.10 if readout == "release_10" else G.PRIMARY_MAG
    nz = G.NOISES_BY_MAG[mag]
    rows = 3 if readout == "release_xyz" else 2

    def mean_of(labels):
        v = [E[lab[q]] for q in labels if q in lab]
        v = [x for x in v if np.all(np.isfinite(x))]
        return np.mean(v, 0) if len(v) == len(labels) else None
    e0 = mean_of([f"base_{q}" for q in nz])
    eA = mean_of(["base_A1", "base_A2"])
    eB = mean_of(["base_B1", "base_B2"])
    floor = (float(np.linalg.norm((eB - eA)[:2]) / (np.sqrt(2.0) * mag)) if eA is not None and eB is not None else None)
    J, resp, nrm = {}, {}, {}
    for n in meta["cands"]:
        e = {}
        for ax, sg in G.PERTS:
            e[(ax, sg)] = mean_of([f"{n}|{ax}{'+' if sg > 0 else '-'}|{mag:g}|{q}" for q in nz])
        if e0 is None or any(v is None for v in e.values()):
            continue
        J3 = np.stack([(e[("x", 1)] - e[("x", -1)]) / (2 * mag), (e[("y", 1)] - e[("y", -1)]) / (2 * mag)], 1)
        J[n] = J3
        nrm[n] = float(np.linalg.norm(J3[:rows]))
        resp[n] = [((e[(ax, sg)] - e0)[:2], np.array([sg * mag if ax == "x" else 0.0, sg * mag if ax == "y" else 0.0])) for ax, sg in G.PERTS]
    bound = max(nrm, key=nrm.get) if nrm else None
    rel = [b["released"] for b in meta["boxes"]]
    return dict(meta=meta, key=meta["key"], tid=meta["tid"], role="place", J=J, resp=resp, nrm=nrm, bound=bound, floor=floor, e0=e0,
                yaw={n: 0.0 for n in J}, pos={n: np.asarray(p, np.float64) for n, p in meta["pos"].items()},
                released=float(np.mean(rel)) if rel else None, z=z)


def load(set_name, readout, sc):
    return [obs_record(f, readout, sc) for f in sorted(glob.glob(os.path.join(OUT, "sb", set_name, "*.npz")))]


def g1(recs):
    per_hit, per_ch, per_diff, per_above = {}, {}, {}, {}
    uns = 0
    for r in recs:
        f = r["meta"]["filler"]
        if f not in r["J"]:
            uns += 1
            continue
        hit, ch = float(r["bound"] == f), 1.0 / len(r["J"])
        per_hit.setdefault(r["tid"], []).append(hit)
        per_ch.setdefault(r["tid"], []).append(ch)
        per_diff.setdefault(r["tid"], []).append(hit - ch)
        if r["floor"] is not None:
            per_above.setdefault(r["tid"], []).append(float(r["nrm"][r["bound"]] > r["floor"]))
    if not per_hit:
        return dict(rate=DR.rate({}), chance=DR.rate({}), p=1.0, point=False, n_unscored=uns)
    s = DR.boot2(per_diff)
    out = dict(rate=DR.rate(per_hit), chance=DR.rate(per_ch), p=float(np.mean(s <= 0)), n_unscored=uns, above_floor=DR.rate(per_above))
    out["point"] = bool(out["rate"]["mean"] >= 0.8 and out["rate"]["lo"] > out["chance"]["mean"])
    return out


def actual_body(o):
    """G3's actual body: the typer's alternative bodies minus the own target (and the held object); exactly one -> it; none and the
    episode was carried elsewhere -> the body nearest the zs release (index rule, within 10 cm); otherwise excluded."""
    alt = [a for a in o["alt_bodies"] if a != o["filler"] and a != o["held"]]
    if len(alt) == 1:
        return alt[0], "alt"
    if alt:
        return None, "ambiguous"
    if o["alt_bodies"]:
        return None, "same_body_region"
    if o["near_body"]:
        return o["near_body"], "near"
    return None, "no_body"


def g3(recs, dix):
    per_g, per_n = {}, {}
    excl = {}
    rows = []
    for r in recs:
        o = dix.get(r["key"])
        if o is None or o["type"] != "wrong_goal":
            continue
        a, how = actual_body(o)
        if a is None:
            excl[how] = excl.get(how, 0) + 1
            continue
        if a not in r["J"]:
            excl["unscored"] = excl.get("unscored", 0) + 1
            continue
        g, nn = float(r["bound"] == a), float(r["bound"] == o["filler"])
        per_g.setdefault(o["tid"], []).append(g)
        per_n.setdefault(o["tid"], []).append(nn)
        rows.append(dict(key=r["key"], tid=o["tid"], actual=a, how=how, target=o["filler"], bound=r["bound"]))
    if not per_g:
        return dict(rate_actual=DR.rate({}), rate_named=DR.rate({}), p=1.0, point=False, excluded=excl, rows=rows)
    s = DR.boot2({t: [a - b for a, b in zip(per_g[t], per_n[t])] for t in per_g})
    out = dict(rate_actual=DR.rate(per_g), rate_named=DR.rate(per_n), p=float(np.mean(s <= 0)), excluded=excl, rows=rows)
    out["point"] = bool(out["rate_actual"]["mean"] >= 0.7)
    return out


def anchor(recs, ix_by_key, group):
    """The goal-role vision anchor (D3 / R4): the argmax agentview token of the target type's SigLIP prototype (rb4 protos, training
    frames only).  hit = the token lies in the target body's segmentation region (D3's criterion); err = xy distance of its live-depth
    point to the body's position; chance = the region's token share."""
    pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
    cats = [str(c) for c in pz["cats"]]
    per = {k: {} for k in ("hit", "chance", "err5")}
    errs, n_noproto = [], 0
    for r in recs:
        o = ix_by_key[r["key"]]
        t = o.get("filler_type")
        z = r["z"]
        if t not in cats or len(z["sig_tok"]) == 0:
            n_noproto += 1
            continue
        tok = int(z["sig_tok"][cats.index(t)])
        segn = r["meta"]["bodies_seg"]
        if o["filler"] not in segn:
            n_noproto += 1
            continue
        cov = np.asarray(z["cov"], np.float32)[segn.index(o["filler"])]
        xyz = np.asarray(z["tok_xyz"], np.float64).reshape(256, 3)[tok]
        err = float(np.linalg.norm(xyz[:2] - r["pos"][o["filler"]][:2]))
        errs.append(err)
        per["hit"].setdefault(o["tid"], []).append(float(cov[tok] > 0))
        per["chance"].setdefault(o["tid"], []).append(float((cov > 0).mean()))
        per["err5"].setdefault(o["tid"], []).append(float(err < 0.05))
    return dict(group=group, hit=DR.rate(per["hit"]), chance=DR.rate(per["chance"]), within_5cm=DR.rate(per["err5"]),
                err_median_m=(float(np.median(errs)) if errs else None), n=len(errs), no_proto=n_noproto)


def fidelity(recs, dix):
    """Sandbox fidelity (dev): the base sandbox under the recorded zs noise seeds vs the recorded zs episode."""
    d, same_rel = [], []
    for r in recs:
        o = dix[r["key"]]
        lab = {b["label"]: i for i, b in enumerate(r["meta"]["boxes"])}
        if "base_zs" not in lab:
            continue
        b = r["meta"]["boxes"][lab["base_zs"]]
        p = np.asarray(r["z"]["rel_hand"][lab["base_zs"]], np.float64)
        d.append(float(np.linalg.norm(p[:2] - np.asarray(o["zs_release_hand"])[:2])))
        same_rel.append(float(b["released"] == (o["zs_release_step"] is not None)))
    return dict(n=len(d), median_xy_m=(float(np.median(d)) if d else None), within_3cm=(float(np.mean(np.array(d) < 0.03)) if d else None),
                release_agree=(float(np.mean(same_rel)) if same_rel else None))


def region_readout(recs, dix):
    """Descriptive: the target body's BDDL region site nearest (xyz) the base release (A1 / A2 mean) vs the own target region, and vs
    the region nearest the RECORDED zs release; on zs successes and on same-body wrong-goal episodes (:29, :40)."""
    out = {}
    for grp, sel in (("success", lambda o: o["success"]), ("same_body_wrong_goal",
                                                           lambda o: o["type"] == "wrong_goal" and actual_body(o)[1] == "same_body_region")):
        own, zsr, n = [], [], 0
        for r in recs:
            o = dix[r["key"]]
            if not sel(o) or r["e0"] is None or o["filler"] is None:
                continue
            reg = {k: np.asarray(v, np.float64) for k, v in r["meta"].get("regions", {}).items()
                   if k == o["filler"] or k.startswith(o["filler"] + "_")}
            reg = {k: v for k, v in reg.items() if k != o["filler"]}
            if len(reg) < 2:
                continue
            lab = {b["label"]: i for i, b in enumerate(r["meta"]["boxes"])}
            e = np.mean([np.asarray(r["z"]["rel_hand"][lab[q]], np.float64) for q in ("base_A1", "base_A2")], 0)
            near = min(reg, key=lambda k: float(np.linalg.norm(reg[k] - e)))
            zn = min(reg, key=lambda k: float(np.linalg.norm(reg[k] - np.asarray(o["zs_release_hand"]))))
            own.append(float(near == o["target"]))
            zsr.append(float(near == zn))
            n += 1
        out[grp] = dict(n=n, sandbox_region_is_target=(float(np.mean(own)) if own else None),
                        sandbox_region_is_zs_region=(float(np.mean(zsr)) if zsr else None))
    return out


def validity(set_name):
    tot = {f"{m:g}": [0, 0] for m in G.MAGS}
    reasons = {}
    for f in sorted(glob.glob(os.path.join(OUT, "sb", set_name, "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        for m, v in meta["validity"].items():
            for _, (ok, why) in v.items():
                tot[m][0] += 1
                tot[m][1] += bool(ok)
                if not ok:
                    k = f"{m}:{why.split(':')[0]}"
                    reasons[k] = reasons.get(k, 0) + 1
    return dict(perturbations={m: v[0] for m, v in tot.items()}, valid_rate={m: (v[1] / v[0] if v[0] else None) for m, v in tot.items()},
                rejected_by=reasons)


def op_table(recs, types):
    by = {}
    for r in recs:
        f = r["meta"]["filler"]
        if f in r["J"]:
            by.setdefault(types(r, f), []).append(r)
    out = {}
    for t, lst in sorted(by.items()):
        Js = np.stack([r["J"][r["meta"]["filler"]] for r in lst])
        out[t] = dict(n=len(lst), tasks=len({r["tid"] for r in lst}), J_mean=np.round(Js.mean(0), 3).tolist(),
                      lam=float(np.mean([np.trace(J[:2]) / 2 for J in Js])), norm=float(np.mean([np.linalg.norm(J[:2]) for J in Js])),
                      hit=float(np.mean([r["bound"] == r["meta"]["filler"] for r in lst])))
    return out


def read(record):
    sc = scale()
    ix = G.load_index()
    tix = {o["key"]: o for o in ix["train"]}
    dix = {o["key"]: o for o in ix["dev"]}
    allix = dict(tix, **dix)
    types = lambda r, n: (allix[r["key"]]["filler_type"] if n == allix[r["key"]]["filler"] else n.rsplit("_", 1)[0])   # noqa: E731
    res = dict(settings=dict(mags=G.MAGS, primary_mag=G.PRIMARY_MAG, noises=G.NOISES_BY_MAG, base=G.BASE_NOISES, q_max=G.Q_MAX,
                             replan=G.REPLAN, primary=PRIMARY), validity={s: validity(s) for s in ("train", "dev")})
    res["readouts"] = {}
    prim = None
    for ro in READOUTS:
        tr = load("train", ro, sc)
        dv = load("dev", ro, sc)
        R = dict(n_train=len(tr), n_dev=len(dv), G1=g1(tr), G2=DR.h2(tr, "place", types), G3=g3(dv, dix),
                 released_train=float(np.mean([r["released"] for r in tr])) if tr else None,
                 released_dev=float(np.mean([r["released"] for r in dv])) if dv else None,
                 floor_median=(float(np.median([r["floor"] for r in tr if r["floor"] is not None])) if tr else None),
                 dev_success_bound_target=g1([r for r in dv if dix[r["key"]]["success"]]),
                 dev_recomb_wrong_goal_bound_target=g1([r for r in dv if dix[r["key"]]["type"] == "wrong_goal" and dix[r["key"]]["bin"] != "part"]),
                 ops=op_table(tr, types))
        res["readouts"][ro] = R
        if ro == PRIMARY:
            prim = (tr, dv)
    tr, dv = prim
    res["fidelity"] = fidelity(dv, dix)
    res["region"] = region_readout(dv, dix)
    res["anchor"] = dict(train=anchor(tr, allix, "train"),
                         dev_success=anchor([r for r in dv if dix[r["key"]]["success"]], allix, "dev_success"),
                         dev_wrong_goal=anchor([r for r in dv if dix[r["key"]]["type"] == "wrong_goal"], allix, "dev_wrong_goal"),
                         dev_wrong_goal_recomb=anchor([r for r in dv if dix[r["key"]]["type"] == "wrong_goal" and dix[r["key"]]["bin"] != "part"],
                                                      allix, "dev_wrong_goal_recomb"))
    P = res["readouts"][PRIMARY]
    pts = {"G1": P["G1"]["point"], "G2": P["G2"]["point"], "G3": P["G3"]["point"]}
    ps = {"G1": P["G1"]["p"], "G2": P["G2"]["p"], "G3": P["G3"]["p"]}
    hp = DR.holm(ps)
    res["verdict"] = {h: dict(point=bool(pts[h]), p=ps[h], holm=hp[h], go=bool(pts[h] and hp[h])) for h in ps}
    G.dump(os.path.join(OUT, "read.json"), res)
    short = {ro: dict(G1=dict(rate=v["G1"]["rate"], chance=v["G1"]["chance"]["mean"], p=v["G1"]["p"], unscored=v["G1"]["n_unscored"]),
                      G2={k: v["G2"].get(k) for k in ("r2", "r2_ci", "r2_ctrl", "diff", "p", "r2_type_shuffled_ref", "n")},
                      G3={k: v["G3"][k] for k in ("rate_actual", "rate_named", "p", "excluded")}, released=[v["released_train"], v["released_dev"]],
                      floor=v["floor_median"], dev_success=v["dev_success_bound_target"]["rate"],
                      dev_recomb_wg_target=v["dev_recomb_wrong_goal_bound_target"]["rate"]) for ro, v in res["readouts"].items()}
    print(json.dumps(dict(validity=res["validity"], verdict=res["verdict"], readouts=short, fidelity=res["fidelity"], region=res["region"],
                          anchor=res["anchor"],
                          ops=P["ops"], g3_rows=P["G3"]["rows"]), indent=1, default=float))
    if record:
        from lifelong_ood import ledger
        print(ledger.record("d5_read", bench="libero_90_c", tag="read", seed=0,
                            metrics=dict(verdict=res["verdict"], validity=res["validity"], readouts=res["readouts"], fidelity=res["fidelity"],
                                         anchor=res["anchor"], region=res["region"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=res["settings"],
                            script="experiments/d5_read.py", command="python experiments/d5_goal.py --read --record",
                            artifacts=[os.path.join(OUT, "read.json")]))
