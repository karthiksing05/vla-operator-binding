"""PROBE D4 read (docs/PLAN_D4.md; pre-registration experiments/prereg/pd4_bind.py): binding operators from the frozen policies'
counterfactual chunks (experiments/d4_bind.py --gpu outputs).  Run through `experiments/d4_bind.py --read [--record]` (chunks env).

  per observation: endpoints e = h + scale * sum_t a_t[:3] of every chunk, mean over the K shared noises; J_Z (central differences,
  2x2 xy block primary) for every body whose 4 perturbations are valid; bound body = argmax ||J_Z||_F; noise floor from the
  unperturbed second-seed pair.
  H1  readout validity (training observations): the role filler (named object / target body) is the bound body.
  H2  systematicity: leave-type-out prediction of the filler's endpoint responses from the role's pooled operator, R^2, vs a
      filler-shuffled control (the operator pooled from a non-filler body of each observation).
  H3  diagnosis (zs dev observations of pi05_l90c): on wrong-object (wrong-goal) failures the bound body is the one grasped (placed
      on), not the named one.
  H4  descriptive: nested variance of J over category > type > instance > observation; the bank's size.
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

OUT = D4.OUT
ROLES = ("approach", "place")
D_MIN, CORRIDOR = 0.02, 0.05          # R2a's pointing readout (a heading < 2 cm points at nothing; 5 cm corridor)


def scale_for(model):
    import of_frame as OFm
    return OFm.scale_for(model)


def endpoints(raw, h, scale):
    """raw [..., H, 7] -> endpoints [..., 3] (h + scale * summed xyz deltas)."""
    return np.asarray(h, np.float64) + np.asarray(raw, np.float64)[..., :3].sum(-2) * np.asarray(scale, np.float64)


def obs_record(path, scale):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    h = np.asarray(meta["eef"], np.float64)
    E = endpoints(z["raw"], h, scale)                                   # [n_img, K, 3]
    Em = E.mean(1)
    eA = Em[0]
    eB = endpoints(z["rawB"], h, scale).mean(0)
    floor = float(np.linalg.norm((eB - eA)[:2]) / (np.sqrt(2.0) * D4.DELTA))
    cf = {(c["cand"], c["axis"], c["sign"]): c for c in meta["cf"]}
    J, resp, nrm = {}, {}, {}
    for n in meta["cands"]:
        cs = [cf.get((n, ax, sg)) for ax, sg in D4.PERTS]
        if not all(c is not None and c["valid"] for c in cs):
            continue
        e = {(ax, sg): Em[cf[(n, ax, sg)]["img"]] for ax, sg in D4.PERTS}
        J3 = np.stack([(e[("x", 1)] - e[("x", -1)]) / (2 * D4.DELTA), (e[("y", 1)] - e[("y", -1)]) / (2 * D4.DELTA)], 1)   # [3, 2]
        J[n] = J3
        nrm[n] = float(np.linalg.norm(J3[:2]))
        resp[n] = [((e[(ax, sg)] - eA)[:2], np.array([sg * D4.DELTA if ax == "x" else 0.0, sg * D4.DELTA if ax == "y" else 0.0]))
                   for ax, sg in D4.PERTS]
    bound = max(nrm, key=nrm.get) if nrm else None
    pos = {n: np.asarray(p, np.float64) for n, p in meta["pos"].items()}
    off = {n: ((eA - h)[:2] - J[n][:2] @ (pos[n] - h)[:2]) for n in J}
    return dict(meta=meta, key=meta["key"], tid=meta["tid"], role=meta["role"], h=h, eA=eA, eB=eB, floor=floor, J=J, nrm=nrm, resp=resp,
                bound=bound, pos=pos, yaw=meta["yaw"], off=off, n_valid=sum(c["valid"] for c in meta["cf"]), n_cf=len(meta["cf"]))


def load(model, set_name):
    scale = scale_for(model)
    return [obs_record(f, scale) for f in sorted(glob.glob(os.path.join(OUT, "gpu", f"{model}_{set_name}", "*.npz")))], scale


def boot2(per_task, B=D4.B_BOOT, seed=0):
    """Two-level cluster bootstrap samples of a pooled mean (R2a ci_rate's scheme): per_task {task: [unit values]}."""
    rng = np.random.default_rng(seed)
    xs = [np.asarray(v, float) for v in per_task.values() if len(v)]
    s = np.empty(B)
    for i in range(B):
        num, den = 0.0, 0
        for j in rng.integers(0, len(xs), len(xs)):
            m = len(xs[j])
            num += xs[j][rng.integers(0, m, m)].sum()
            den += m
        s[i] = num / max(den, 1)
    return s


def rate(per_task):
    xs = [v for vs in per_task.values() for v in vs]
    if not xs:
        return dict(mean=None, lo=None, hi=None, n=0, tasks=0)
    s = boot2(per_task)
    return dict(mean=float(np.mean(xs)), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), n=len(xs),
                tasks=sum(1 for v in per_task.values() if v))


def holm(ps, alpha=0.05):
    order = sorted(ps, key=lambda k: ps[k])
    passed, ok = {}, True
    for r, k in enumerate(order):
        ok = ok and ps[k] <= alpha / (len(ps) - r)
        passed[k] = bool(ok)
    return passed


# ── validity of the perturbations ──────────────────────────────────────────────────────────────────────────────────
def validity(set_name):
    tot, ok, reasons, scored, cands, fixt = 0, 0, {}, 0, 0, dict(tot=0, ok=0)
    for f in sorted(glob.glob(os.path.join(OUT, "render", set_name, "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        by = {}
        for c in meta["cf"]:
            tot += 1
            ok += c["valid"]
            if not c["valid"]:
                r = c["reason"].split(":")[0]
                reasons[r] = reasons.get(r, 0) + 1
            by.setdefault(c["cand"], []).append(c["valid"])
            if c["cand"] not in meta["pos"] or c["cand"] in meta.get("movable", []):
                pass
        for n, v in by.items():
            cands += 1
            scored += all(v)
    return dict(perturbations=tot, valid=ok, valid_rate=(ok / tot if tot else None), rejected_by=reasons, bodies=cands,
                bodies_scored=scored, scored_rate=(scored / cands if cands else None))


# ── H1: readout validity ──────────────────────────────────────────────────────────────────────────────────────────
def filler_of(r):
    return r["meta"].get("filler")


def h1(recs, role):
    per_hit, per_diff, per_ch, per_above = {}, {}, {}, {}
    n_unscored = 0
    for r in recs:
        if r["role"] != role:
            continue
        f = filler_of(r)
        if f not in r["J"]:
            n_unscored += 1
            continue
        hit = float(r["bound"] == f)
        ch = 1.0 / len(r["J"])
        per_hit.setdefault(r["tid"], []).append(hit)
        per_ch.setdefault(r["tid"], []).append(ch)
        per_diff.setdefault(r["tid"], []).append(hit - ch)
        per_above.setdefault(r["tid"], []).append(float(r["nrm"][r["bound"]] > r["floor"]))
    if not per_hit:
        return dict(rate=rate({}), chance=rate({}), p=1.0, point=False, n_unscored=n_unscored)
    s = boot2(per_diff)
    out = dict(rate=rate(per_hit), chance=rate(per_ch), p=float(np.mean(s <= 0)), n_unscored=n_unscored,
               above_floor=rate(per_above))
    out["point"] = bool(out["rate"]["mean"] >= 0.8)
    out["ci_above_chance"] = bool(out["rate"]["lo"] > out["chance"]["mean"])
    return out


# ── H2: systematicity ─────────────────────────────────────────────────────────────────────────────────────────────
def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def h2_rows(recs, role, types_of):
    rows = []
    for r in recs:
        if r["role"] != role:
            continue
        f = filler_of(r)
        if f not in r["J"]:
            continue
        others = sorted(n for n in r["J"] if n != f)
        ctrl = others[D4.h8("d4ctrl:" + r["key"]) % len(others)] if others else None
        rows.append(dict(tid=r["tid"], key=r["key"], type=types_of(r, f), J=r["J"][f][:2], resp=r["resp"][f], yaw=r["yaw"].get(f, 0.0),
                         Jc=(r["J"][ctrl][:2] if ctrl else None), yawc=(r["yaw"].get(ctrl, 0.0) if ctrl else 0.0)))
    return rows


def lto_ss(rows, labels, use_ctrl=False, frame="world"):
    """Leave-label-out predictions of every row's endpoint responses from the mean operator of the rows with other labels.
    -> per-row (ss_res, ss_tot) arrays."""
    res, tot = np.zeros(len(rows)), np.zeros(len(rows))
    labels = np.asarray(labels)
    for lab in sorted(set(labels)):
        tr = [i for i in range(len(rows)) if labels[i] != lab]
        if use_ctrl:
            mats = [(rows[i]["Jc"], rows[i]["yawc"]) for i in tr if rows[i]["Jc"] is not None]
        else:
            mats = [(rows[i]["J"], rows[i]["yaw"]) for i in tr]
        if not mats:
            Jbar = np.zeros((2, 2))
        elif frame == "object":
            Jbar = np.mean([rot(-y) @ M @ rot(y) for M, y in mats], 0)
        else:
            Jbar = np.mean([M for M, _ in mats], 0)
        for i in range(len(rows)):
            if labels[i] != lab:
                continue
            Jp = rot(rows[i]["yaw"]) @ Jbar @ rot(-rows[i]["yaw"]) if frame == "object" else Jbar
            for de, dp in rows[i]["resp"]:
                res[i] += float(((de - Jp @ dp) ** 2).sum())
                tot[i] += float((de ** 2).sum())
    return res, tot


def r2_of(res, tot, idx=None):
    idx = np.arange(len(res)) if idx is None else idx
    t = tot[idx].sum()
    return float(1 - res[idx].sum() / t) if t > 0 else float("nan")


def h2(recs, role, types_of, B=2000):
    rows = h2_rows(recs, role, types_of)
    types = [r["type"] for r in rows]
    if len(set(types)) < 2:
        return dict(n=len(rows), types=sorted(set(types)), r2=None, r2_ctrl=None, p=1.0, point=False)
    res, tot = lto_ss(rows, types)
    resc, _ = lto_ss(rows, types, use_ctrl=True)
    reso, _ = lto_ss(rows, types, frame="object")
    r2, r2c = r2_of(res, tot), r2_of(resc, tot)
    tasks = sorted({r["tid"] for r in rows})
    of = {t: np.array([i for i, r in enumerate(rows) if r["tid"] == t]) for t in tasks}
    rng = np.random.default_rng(0)
    d, s1 = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([of[tasks[q]] for q in rng.integers(0, len(tasks), len(tasks))])
        s1[b] = r2_of(res, tot, idx)
        d[b] = s1[b] - r2_of(resc, tot, idx)
    # descriptive: type labels shuffled across rows (the within-type information returns), 200 permutations
    sh = []
    for p in range(200):
        lab = np.random.default_rng(1000 + p).permutation(types)
        rs, ts = lto_ss(rows, lab)
        sh.append(r2_of(rs, ts))
    per_type = {}
    for t in sorted(set(types)):
        idx = np.array([i for i in range(len(rows)) if types[i] == t])
        per_type[t] = dict(n=int(len(idx)), r2=r2_of(res, tot, idx), r2_ctrl=r2_of(resc, tot, idx))
    out = dict(n=len(rows), tasks=len(tasks), types=sorted(set(types)), r2=r2, r2_ci=[float(np.nanpercentile(s1, 2.5)), float(np.nanpercentile(s1, 97.5))],
               r2_ctrl=r2c, diff=r2 - r2c, diff_ci=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
               p=float(np.mean(d <= 0)), r2_object_frame=r2_of(reso, tot), r2_type_shuffled_ref=float(np.mean(sh)),
               n_ctrl_rows=int(sum(r["Jc"] is not None for r in rows)), per_type=per_type)
    out["point"] = bool(r2 >= 0.5 and r2 > r2c)
    return out


# ── H3: diagnosis on zs failures ──────────────────────────────────────────────────────────────────────────────────
def dev_join(recs):
    ix = D4.load_dev_index()
    epi = {}
    for tid, x in ix.items():
        for e in x["episodes"]:
            for m in e["obs"]:
                epi[m["key"]] = (tid, e)
    return epi


def h3(recs, epi):
    out = {}
    for name, role, typ in (("wrong_object", "approach", "wrong_object"), ("wrong_goal", "place", "wrong_goal")):
        per_g, per_n, per_d = {}, {}, {}
        excl = dict(no_alt_body=0, unscored=0)
        for r in recs:
            if r["role"] != role or r["key"] not in epi:
                continue
            tid, e = epi[r["key"]]
            if e["type"] != typ:
                continue
            named = set(r["meta"]["named"])
            if typ == "wrong_object":
                act = {e["bound"]} if e["bound"] else set()
            else:
                act = {D4.body_of(a, list(r["pos"])) for a in (e.get("alt") or [])} - {None} - named - {r["meta"]["held"]}
            if not act:
                excl["no_alt_body"] += 1
                continue
            if not (act & set(r["J"])):
                excl["unscored"] += 1
                continue
            g, nn = float(r["bound"] in act), float(r["bound"] in named)
            k = (tid, e["trial"])
            per_g.setdefault(tid, {}).setdefault(k, []).append(g)
            per_n.setdefault(tid, {}).setdefault(k, []).append(nn)
        eg = {t: [float(np.mean(v)) for v in d.values()] for t, d in per_g.items()}
        en = {t: [float(np.mean(v)) for v in d.values()] for t, d in per_n.items()}
        ed = {t: [a - b for a, b in zip(eg[t], en[t])] for t in eg}
        if not eg:
            out[name] = dict(rate_actual=rate({}), rate_named=rate({}), p=1.0, point=False, excluded=excl)
            continue
        s = boot2(ed)
        out[name] = dict(rate_actual=rate(eg), rate_named=rate(en), p=float(np.mean(s <= 0)), excluded=excl)
        out[name]["point"] = bool(out[name]["rate_actual"]["mean"] >= 0.7)
    return out


def dev_success_readout(recs, epi):
    """Descriptive: on zs SUCCESS episodes, is the bound body the named one (approach) / own target (place)?"""
    out = {}
    for role in ROLES:
        per = {}
        for r in recs:
            if r["role"] != role or r["key"] not in epi:
                continue
            tid, e = epi[r["key"]]
            if not e["success"]:
                continue
            named = set(r["meta"]["named"])
            if not (named & set(r["J"])):
                continue
            per.setdefault(tid, {}).setdefault(e["trial"], []).append(float(r["bound"] in named))
        out[role] = rate({t: [float(np.mean(v)) for v in d.values()] for t, d in per.items()})
    return out


# ── H4 + the bank ─────────────────────────────────────────────────────────────────────────────────────────────────
def h4(rows_full):
    """rows_full: [dict(J (2x2), type, cat, inst)] -> shares of the total sum of squares by nested level."""
    if not rows_full:
        return None
    X = np.stack([r["J"].reshape(-1) for r in rows_full])
    g = X.mean(0)

    def means(key):
        out = {}
        for r, x in zip(rows_full, X):
            out.setdefault(key(r), []).append(x)
        return {k: np.mean(v, 0) for k, v in out.items()}
    mc, mt, mi = means(lambda r: r["cat"]), means(lambda r: (r["cat"], r["type"])), means(lambda r: (r["cat"], r["type"], r["inst"]))
    ss = dict(category=0.0, type_in_category=0.0, instance_in_type=0.0, observation_in_instance=0.0)
    for r, x in zip(rows_full, X):
        c, t, i = mc[r["cat"]], mt[(r["cat"], r["type"])], mi[(r["cat"], r["type"], r["inst"])]
        ss["category"] += float(((c - g) ** 2).sum())
        ss["type_in_category"] += float(((t - c) ** 2).sum())
        ss["instance_in_type"] += float(((i - t) ** 2).sum())
        ss["observation_in_instance"] += float(((x - i) ** 2).sum())
    tot = sum(ss.values())
    return dict(n=len(rows_full), categories=len(mc), types=len(mt), instances=len(mi), ss=ss,
                share={k: (v / tot if tot > 0 else None) for k, v in ss.items()})


def operator_table(recs, role, types_of, inst_of):
    by = {}
    for r in recs:
        if r["role"] != role:
            continue
        f = filler_of(r)
        if f not in r["J"]:
            continue
        t = types_of(r, f)
        by.setdefault(t, []).append((r, f))
    tab = {}
    for t, lst in sorted(by.items()):
        Js = np.stack([r["J"][f] for r, f in lst])
        offs = np.stack([r["off"][f] for r, f in lst])
        tab[t] = dict(category=D4.CATEGORY.get(t, t), n=len(lst), tasks=len({r["tid"] for r, _ in lst}),
                      instances=len({inst_of(r, f) for r, f in lst}), J_mean=np.round(Js.mean(0), 3).tolist(),
                      J_norm_mean=float(np.mean([np.linalg.norm(J[:2]) for J in Js])),
                      lam_mean=float(np.mean([np.trace(J[:2]) / 2 for J in Js])), offset_mean=np.round(offs.mean(0), 4).tolist(),
                      hit=float(np.mean([r["bound"] == f for r, f in lst])))
    return tab


BYTES_PER_OP = 8 * 2          # J 3x2 (incl. the z row) + offset 2, float16


def bank(recs, types_of, inst_of):
    """The hierarchical bank: role > category > type > instance; every node stores the mean operator of its observations."""
    nodes = {}
    for r in recs:
        f = filler_of(r)
        if f not in r["J"]:
            continue
        t = types_of(r, f)
        c = D4.CATEGORY.get(t, t)
        op = np.r_[r["J"][f].reshape(-1), r["off"][f]]
        for lev, k in (("role", (r["role"],)), ("category", (r["role"], c)), ("type", (r["role"], c, t)),
                       ("instance", (r["role"], c, t, inst_of(r, f)))):
            nodes.setdefault((lev, k), []).append(op)
    count = {}
    for (lev, _), _v in nodes.items():
        count[lev] = count.get(lev, 0) + 1
    keys = sum(len("/".join(map(str, k)).encode("utf-8")) for (_, k) in nodes)
    ops = {k: np.mean(v, 0) for k, v in nodes.items()}
    return dict(nodes=count, n_ops=len(nodes), bytes_per_op=BYTES_PER_OP, bytes_ops=len(nodes) * BYTES_PER_OP, bytes_keys=keys,
                bytes_total=len(nodes) * BYTES_PER_OP + keys), ops


def pointed(disp, h, cand_pos):
    """R2a's readout on an endpoint displacement: the first candidate along the heading inside the corridor, else best cosine."""
    names = list(cand_pos)
    vec = np.array([cand_pos[n][:2] - h[:2] for n in names])
    if np.linalg.norm(disp) < D_MIN or not names:
        return None
    u = disp / np.linalg.norm(disp)
    along = vec @ u
    perp = np.abs(vec[:, 0] * u[1] - vec[:, 1] * u[0])
    inside = np.flatnonzero((along > 0) & (perp < CORRIDOR))
    if len(inside):
        return names[int(inside[np.argmin(along[inside])])]
    cos = along / np.maximum(np.linalg.norm(vec, axis=1), 1e-12)
    return names[int(np.argmax(cos))]


def compose(dev, epi, ops, types_dev):
    """Descriptive: the parse names the role's filler, its pose (oracle here; the D3 vision anchor would supply it) fills the
    bank's most specific operator for the filler's type (type, else category, else role); the predicted endpoint
    e_hat = h + J (p - h) + b.  Scored against the model's own endpoint on zs successes, and by where it points on failures."""
    out = {}
    for role in ROLES:
        rows = dict(success=[], wrong=[])
        lev_used = {}
        for r in dev:
            if r["role"] != role or r["key"] not in epi:
                continue
            tid, e = epi[r["key"]]
            named = [n for n in r["meta"]["named"] if n in r["pos"]]
            if not named:
                continue
            n = named[0]
            t = types_dev(tid, n)
            c = D4.CATEGORY.get(t, t)
            op, lev = None, None
            for lv, k in (("type", (role, c, t)), ("category", (role, c)), ("role", (role,))):
                if (lv, k) in ops:
                    op, lev = ops[(lv, k)], lv
                    break
            if op is None:
                continue
            lev_used[lev] = lev_used.get(lev, 0) + 1
            J = op[:6].reshape(3, 2)[:2]
            b = op[6:8]
            h = r["h"]
            ehat = h[:2] + J @ (r["pos"][n] - h)[:2] + b
            cand = {q: r["pos"][q] for q in r["meta"]["cands"]}
            p_hat = pointed(ehat - h[:2], h, cand)
            p_own = pointed((r["eA"] - h)[:2], h, cand)
            g = "success" if e["success"] else ("wrong" if e["type"] in ("wrong_object", "wrong_goal") else None)
            if g is None:
                continue
            rows[g].append(dict(tid=tid, err=float(np.linalg.norm(ehat - r["eA"][:2])), err_stay=float(np.linalg.norm(h[:2] - r["eA"][:2])),
                                hat_named=float(p_hat == n), own_named=float(p_own == n)))
        res = {}
        for g, rs in rows.items():
            if not rs:
                continue
            per = lambda f: {t: [x[f] for x in rs if x["tid"] == t] for t in {x["tid"] for x in rs}}   # noqa: E731
            res[g] = dict(n=len(rs), err_median_m=float(np.median([x["err"] for x in rs])),
                          stay_err_median_m=float(np.median([x["err_stay"] for x in rs])),
                          bank_points_at_named=rate(per("hat_named")), model_points_at_named=rate(per("own_named")))
        out[role] = dict(levels_used=lev_used, **res)
    return out


def synth():
    """Crash test of read() on SYNTHETIC files (random chunks; D4_OUT must point at a scratch directory).  No policy output is used."""
    import shutil
    real = os.path.join(ROOT, "results", "d4")
    assert os.path.abspath(OUT) != os.path.abspath(real), "set D4_OUT to a scratch directory"
    rng = np.random.default_rng(0)
    K = D4.K_NOISE

    def fake(set_name, model, key, tid, role, cands, filler=None, named=None, held=None, trial=None):
        cf, img = [], 1
        for n in cands:
            for ax, sg in D4.PERTS:
                ok = bool(rng.random() > 0.05)
                cf.append(dict(cand=n, axis=ax, sign=sg, valid=ok, pen=0.0, pen0=0.0, settle_err=0.0, reason="" if ok else "penetration",
                               img=img if ok else -1))
                img += ok
        meta = dict(key=key, tid=tid, role=role, named=named or [filler], held=held, cands=cands, cf=cf,
                    pos={n: rng.normal(0, 0.1, 3).tolist() for n in cands + ([held] if held else [])},
                    yaw={n: float(rng.uniform(-3, 3)) for n in cands}, eef=rng.normal(0, 0.1, 3).tolist(), check={})
        if filler is not None:
            meta.update(filler=filler, frame=0)
        else:
            meta.update(trial=trial)
        D4.save_atomic(os.path.join(OUT, "render", set_name, key + ".npz"), meta=np.array(json.dumps(meta)))
        D4.save_atomic(os.path.join(OUT, "gpu", f"{model}_{set_name}", key + ".npz"), raw=rng.normal(0, 1, (img, K, 10, 7)).astype(np.float32),
                       rawB=rng.normal(0, 1, (K, 10, 7)).astype(np.float32), meta=np.array(json.dumps(meta)))
    for m in D4.MODELS:
        os.makedirs(OUT, exist_ok=True)
        shutil.copy(os.path.join(real, f"index_train_{m}.json"), os.path.join(OUT, f"index_train_{m}.json"))
        for x in D4.jload(os.path.join(OUT, f"index_train_{m}.json"))["tasks"][:12]:
            others = [q for q in list(x["objects"]) + list(x["fixtures"]) if q != x["inst"] and not q.startswith(D4.TABLES)][:4]
            for f in range(3):
                fake(f"train_{m}", m, f"{slug(x['tid'])}_pre{f}", x["tid"], "approach", [x["inst"]] + others, filler=x["inst"])
                if x["target_body"]:
                    fake(f"train_{m}", m, f"{slug(x['tid'])}_carry{f}", x["tid"], "place",
                         [x["target_body"]] + [q for q in others if q != x["target_body"]], filler=x["target_body"], held=x["inst"])
    import m1_compose_create as M1
    tt = M1.task_table()
    for tid in ("libero_90:14", "libero_90:38", "libero_90:29"):
        t = tt[tid]
        objs = list(t.objects)
        fx = [q for q in t.fixtures if not q.startswith(D4.TABLES)]
        own = [a for a in t.goal if len(a) == 3]
        nm = own[0][1]
        tg = D4.body_of(own[0][2], objs + fx)
        eps = []
        for k in range(6):
            ty = ("success", "wrong_object", "wrong_goal")[k % 3]
            other = [q for q in objs if q != nm]
            b = other[0] if (ty == "wrong_object" and other) else nm
            alt = [q for q in objs + fx if q not in (nm, tg)][:1] if ty == "wrong_goal" else None
            obs = []
            for role in ("pre", "post"):
                key = f"{slug(tid)}_t{k}_{role}5"
                cands = [q for q in objs + fx if not (role == "post" and q == nm)]
                fake("dev", "l90c", key, tid, "approach" if role == "pre" else "place", cands,
                     named=[nm] if role == "pre" else [tg], held=(nm if role == "post" else None), trial=k)
                obs.append(dict(key=key, role=role, j=5, named=[nm] if role == "pre" else [own[0][2]], held=(nm if role == "post" else None)))
            eps.append(dict(trial=k, success=ty == "success", bound=b, bound_step=10, type=ty, sub="other_region" if ty == "wrong_goal" else "",
                            alt=alt, target=own[0][2], obs=obs))
        D4.dump(os.path.join(OUT, "dev", "index", f"{slug(tid)}.json"), dict(task=tid, episodes=eps))
    print(f"[d4 synth] synthetic files under {OUT}", flush=True)


def slug(tid):
    return tid.replace(":", "_")


def read(record):
    import m1_compose_create as M1
    tt = M1.task_table()
    tix = {m: {x["tid"]: x for x in D4.jload(os.path.join(OUT, f"index_train_{m}.json"))["tasks"]} for m in D4.MODELS}

    def types_train(m):
        return lambda r, n: D4.type_of(n, dict(tix[m][r["tid"]]["objects"], **tix[m][r["tid"]]["fixtures"]))

    def inst_train(m):
        return lambda r, n: f"{tix[m][r['tid']]['scene']}/{n}"

    def types_dev(tid, n):
        return D4.type_of(n, dict(tt[tid].objects, **tt[tid].fixtures))
    res = dict(settings=dict(delta=D4.DELTA, perts=D4.PERTS, pen_tol=D4.PEN_TOL, settle=[D4.SETTLE_STEPS, D4.SETTLE_TOL],
                             k_noise=D4.K_NOISE, seeds=[D4.SEED_A, D4.SEED_B]))
    res["validity"] = {s: validity(s) for s in ("train_l90c", "train_libero", "dev")}
    res["models"] = {}
    ops_l90c = None
    for m in D4.MODELS:
        recs, scale = load(m, f"train_{m}")
        R = dict(n_obs={ro: sum(r["role"] == ro for r in recs) for ro in ROLES}, scale=list(map(float, scale)))
        R["floor"] = {ro: dict(median=float(np.median([r["floor"] for r in recs if r["role"] == ro])) if any(r["role"] == ro for r in recs) else None,
                               max_norm_median=float(np.median([max(r["nrm"].values()) for r in recs if r["role"] == ro and r["nrm"]]))
                               if any(r["role"] == ro and r["nrm"] for r in recs) else None) for ro in ROLES}
        R["H1"] = {ro: h1(recs, ro) for ro in ROLES}
        R["H2"] = {ro: h2(recs, ro, types_train(m)) for ro in ROLES}
        R["operators"] = {ro: operator_table(recs, ro, types_train(m), inst_train(m)) for ro in ROLES}
        rows4 = {ro: [dict(J=r["J"][filler_of(r)][:2], type=types_train(m)(r, filler_of(r)),
                           cat=D4.CATEGORY.get(types_train(m)(r, filler_of(r)), types_train(m)(r, filler_of(r))),
                           inst=inst_train(m)(r, filler_of(r))) for r in recs if r["role"] == ro and filler_of(r) in r["J"]] for ro in ROLES}
        R["H4"] = {ro: h4(rows4[ro]) for ro in ROLES}
        R["bank"], ops = bank(recs, types_train(m), inst_train(m))
        if m == "l90c":
            ops_l90c = ops
        res["models"][m] = R
    dev, _ = load("l90c", "dev")
    epi = dev_join(dev)
    res["dev"] = dict(n_obs={ro: sum(r["role"] == ro for r in dev) for ro in ROLES}, H3=h3(dev, epi),
                      success_readout=dev_success_readout(dev, epi), compose=compose(dev, epi, ops_l90c, types_dev),
                      render_check=dict(exact=float(np.mean([r["meta"]["check"].get("render_exact", False) for r in dev])) if dev else None,
                                        mae_max=float(max([r["meta"]["check"].get("render_mae", 0.0) for r in dev], default=0.0)),
                                        state_err_max=float(max([r["meta"]["check"].get("state_err", 0.0) for r in dev], default=0.0))))
    # hypotheses: Holm over H1 (l90c, both roles: intersection-union, p = max), H2 (likewise), H3a wrong object, H3b wrong goal
    L = res["models"]["l90c"]
    pt = {"H1": all(L["H1"][ro]["point"] and L["H1"][ro].get("ci_above_chance", False) for ro in ROLES),
          "H2": all(L["H2"][ro]["point"] for ro in ROLES),
          "H3a": res["dev"]["H3"]["wrong_object"]["point"], "H3b": res["dev"]["H3"]["wrong_goal"]["point"]}
    ps = {"H1": max(L["H1"][ro]["p"] for ro in ROLES), "H2": max(L["H2"][ro]["p"] for ro in ROLES),
          "H3a": res["dev"]["H3"]["wrong_object"]["p"], "H3b": res["dev"]["H3"]["wrong_goal"]["p"]}
    hp = holm(ps)
    res["verdict"] = {h: dict(point=bool(pt[h]), p=ps[h], holm=hp[h], go=bool(pt[h] and hp[h])) for h in ps}
    Lb = res["models"]["libero"]
    res["cross_model_libero"] = dict(H1={ro: dict(rate=Lb["H1"][ro]["rate"], chance=Lb["H1"][ro]["chance"]["mean"],
                                                  point=Lb["H1"][ro]["point"], p=Lb["H1"][ro]["p"]) for ro in ROLES},
                                     H2={ro: dict(r2=Lb["H2"][ro]["r2"], r2_ctrl=Lb["H2"][ro]["r2_ctrl"], p=Lb["H2"][ro]["p"],
                                                  point=Lb["H2"][ro]["point"]) for ro in ROLES})
    D4.dump(os.path.join(OUT, "read.json"), res)
    print(json.dumps(dict(validity=res["validity"], verdict=res["verdict"],
                          H1={m: {ro: dict(rate=res["models"][m]["H1"][ro]["rate"], chance=res["models"][m]["H1"][ro]["chance"]["mean"],
                                           p=res["models"][m]["H1"][ro]["p"], unscored=res["models"][m]["H1"][ro]["n_unscored"])
                                  for ro in ROLES} for m in D4.MODELS},
                          H2={m: {ro: {k: v for k, v in res["models"][m]["H2"][ro].items() if k != "per_type"} for ro in ROLES}
                              for m in D4.MODELS},
                          H3=res["dev"]["H3"], H4={m: res["models"][m]["H4"] for m in D4.MODELS},
                          bank={m: res["models"][m]["bank"] for m in D4.MODELS}, floor={m: res["models"][m]["floor"] for m in D4.MODELS},
                          dev=dict(n=res["dev"]["n_obs"], success_readout=res["dev"]["success_readout"], compose=res["dev"]["compose"],
                                   render_check=res["dev"]["render_check"])), indent=1, default=float))
    if record:
        from lifelong_ood import ledger
        bt = res["models"]["l90c"]["bank"]["bytes_total"]
        print(ledger.record("d4_read", bench="libero_90_c", tag="read", seed=0,
                            metrics=dict(verdict=res["verdict"], validity=res["validity"],
                                         models={m: {k: v for k, v in res["models"][m].items()} for m in D4.MODELS},
                                         dev=res["dev"], cross_model_libero=res["cross_model_libero"]),
                            bytes=dict(experts=0, tree=0, codes=int(bt), samples=0, teacher_copies=0, total=int(bt)),
                            config=res["settings"], script="experiments/d4_read.py", command=" ".join(sys.argv),
                            backbone=dict(models={m: D4.MODELS[m]["cfg"] for m in D4.MODELS}),
                            artifacts=[os.path.join(OUT, "read.json")]))
