"""PROBE R4b (docs/PLAN_R4B.md; pre-registration experiments/prereg/pr4b_gate.py, ledger m3_prereg tag pr4b_gate): R4-bind
(docs/PLAN_R4.md, experiments/rb4_bind.py) with a NON-CIRCULAR on-manifold judge.  Everything else of R4-bind is fixed: the guidance
form, the 12-config grid, the tuning pool, the noise floor, the movability bar (g_cf >= 0.130), the tie rule and stage 1's H1-H4.

Why: R4-bind judged a guided chunk by its flow-matching loss under the CURRENT conditioning (scene + instruction), which the atlas
shows is misbound.  A correct approach to an object the policy did not bind is high-loss under that misbinding by construction, so
the check penalised the intended fix.  R4b judges the chunk under a conditioning that BINDS ITS TARGET:
  primary:  the flow-matching loss under a same-scene training instruction that names the target in its role (R3c's oracle source
            rule: a train_l90 cell of the same scene whose BDDL In/On goal moves the target (object role) or places on it (goal
            role); several -> the lowest sha1 of rs4src:{task}:{O|G}:{name}:{cell}).  Reference = the policy's own UNGUIDED chunks
            under that same binding conditioning: outlier = FM > mean + 2 sd of the seed-11 reference's 12; rise = guided share minus
            the seed-21 reference's share.
  fallback: where no binding source exists, the same construction under the NULL prompt ("").
  secondary (bars as R4-bind): |normalised action| > 1 share rise <= 0.05, still (< 2 cm) rise <= 0.10.

Stages (each its own SLURM job; nothing here runs on the login node):
  --sources              (chunks env, CPU)  binding sources for every tuning / test observation + inventory -> results/rb4b/sources.json
  --judge --shard i/n    (openpi GPU)       R4-bind's STORED tuning chunks (results/rb4/tune/) re-judged: binding / null references,
                                            flow losses under the binding, null and (check) instruction prefixes -> results/rb4b/judge/
  --stage0 [--record]    (chunks env)       movability (recomputed, must equal R4-bind's) + the new judge + the choice -> stage0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import r2a_null_modes as R2A                                            # noqa: E402
import rb4_bind as RB                                                   # noqa: E402

OUT = os.path.join(ROOT, "results", "rb4b")
RB4 = RB.OUT
R2 = R2A.OUT
JUDGES = ("bind", "null")                                               # primary where a source exists, else the null fallback
CHECK_N = 2                                                             # observations per shard whose stored instruction FM is recomputed
INV_TOL = 1e-4                                                          # round-trip bound of the stored-chunk inversion (action units)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


# ── sources (chunks env) ─────────────────────────────────────────────────────────────────────────────────────────
def sources():
    import glob
    import m1_compose_create as M1
    from lifelong_ood import l90c
    tt = M1.task_table()
    train = list(l90c.load_split()["train_l90"])
    RB._load_meta()
    tune, test, sp = RB._keys_all()
    rs4 = jload(os.path.join(ROOT, "results", "rs4", "sources.json"))["obs"]
    res, agree, disagree = dict(obs={}), 0, []
    pools = {}
    for key in tune + test:
        m = RB.META[key]
        tid = m["task"]
        if tid not in pools:
            t = tt[tid]
            cells = [c for c in train if tt[c].scene == t.scene and tt[c].language != t.language]
            mv, pl = {}, {}
            for c in cells:
                for _, o, g in [(a[0].lower(), a[1], a[2]) for a in tt[c].goal if len(a) == 3 and a[0].lower() in ("in", "on")]:
                    mv.setdefault(o, set()).add(c)
                    pl.setdefault(g, set()).add(c)
            pools[tid] = (mv, pl)
        mv, pl = pools[tid]
        pool, tag = (mv, "O:") if m["role"] == "pre" else (pl, "G:")

        def pick(name):
            cs = pool.get(name, ())
            c = min(sorted(cs), key=lambda q: h(f"rs4src:{tid}:{tag}{name}:{q}")) if cs else None
            return dict(cell=c, text=tt[c].language) if c else None
        z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
        cn = [str(q) for q in z["cand_names"]]
        named = list(m["named"])
        cf = RB.cf_pick(key, cn, set(named))
        nm_o = RB.oracle_target(z, named)[0]
        bn = pick(nm_o) or next((pick(x) for x in sorted(named) if pick(x)), None)
        for x in named:                                                  # cross-check against R3c's table
            mine, theirs = pick(x), rs4.get(key, {}).get("src", {}).get(x)
            if (mine is None) == (theirs is None) and (mine is None or mine["cell"] == theirs["cell"]):
                agree += 1
            else:
                disagree.append((key, x, mine, theirs))
        res["obs"][key] = dict(task=tid, role=m["role"], named=named, oracle_named=nm_o, cf=cf,
                               bind_cf=pick(cf) if cf else None, bind_named=bn)
    O = res["obs"]
    ep = lambda ks: len({(O[k]["task"], RB.META[k]["trial"]) for k in ks})   # noqa: E731
    cnt = {"tune": dict(n=len(tune), episodes=ep(tune), cf=sum(O[k]["cf"] is not None for k in tune),
                        bind_cf=sum(O[k]["bind_cf"] is not None for k in tune),
                        bind_cf_episodes=ep([k for k in tune if O[k]["bind_cf"]]),
                        bind_cf_tasks=sorted({O[k]["task"] for k in tune if O[k]["bind_cf"]}),
                        bind_named=sum(O[k]["bind_named"] is not None for k in tune),
                        tasks=sorted({O[k]["task"] for k in tune}))}
    for s, ks in sp["test"].items():
        cnt[s] = dict(n=len(ks), bind_named=sum(O[k]["bind_named"] is not None for k in ks),
                      bind_cf=sum(O[k]["bind_cf"] is not None for k in ks))
    res["counts"] = cnt
    res["rs4_crosscheck"] = dict(agree=agree, disagree=disagree)
    have = lambda d, ks: sum(os.path.exists(os.path.join(d, k + ".npz")) for k in ks)   # noqa: E731
    res["inventory"] = dict(rb4_tune=have(os.path.join(RB4, "tune"), tune), rb4_test=len(glob.glob(os.path.join(RB4, "test", "*.npz"))),
                            rb4_geom_test=have(os.path.join(RB4, "geom"), test), r2a_obs=have(os.path.join(R2, "obs"), tune + test),
                            rb4b_judge=len(glob.glob(os.path.join(OUT, "judge", "*.npz"))))
    RB.dump(os.path.join(OUT, "sources.json"), res)
    print("[rb4b sources] counts " + json.dumps(cnt), flush=True)
    print(f"[rb4b sources] rs4 cross-check: {agree} agree, {len(disagree)} disagree {disagree[:5]}", flush=True)
    print("[rb4b sources] inventory " + json.dumps(res["inventory"]), flush=True)


# ── judge (openpi GPU): re-judge R4-bind's stored tuning chunks ──────────────────────────────────────────────────
class Inverse:
    """the policy's output transform is affine per entry on the 70 real entries: raw = beta + alpha * row.  -> its inverse."""

    def __init__(self, bk):
        u0 = bk.unnormalize(np.zeros((1, bk.dim), np.float32))[0].reshape(-1).astype(np.float64)
        al = np.zeros(bk.dim)
        for e in range(bk.dim):
            E = np.zeros((1, bk.dim), np.float32)
            E[0, e] = 1.0
            al[e] = bk.unnormalize(E)[0].reshape(-1)[e] - u0[e]
        R = np.random.default_rng(0).standard_normal((6, bk.dim)).astype(np.float32)
        self.affine_err = float(np.abs(R * al[None] + u0[None] - bk.unnormalize(R).reshape(6, -1)).max())
        assert self.affine_err < 1e-3 and np.all(np.abs(al) > 1e-6), (self.affine_err, al.min())
        self.al, self.u0 = al, u0

    def rows(self, raw):
        return ((np.asarray(raw, np.float64).reshape(len(raw), -1) - self.u0[None]) / self.al[None]).astype(np.float32)


def judge(shard):
    import torch
    RB._load_meta()
    tune, _, _ = RB._keys_all()
    src = jload(os.path.join(OUT, "sources.json"))["obs"]
    rn = RB.Runner()
    bk = rn.bk
    inv = Inverse(bk)
    od = os.path.join(OUT, "judge")
    os.makedirs(od, exist_ok=True)
    i, n = map(int, shard.split("/"))
    mine = tune[i::n]
    print(f"[rb4b judge] {bk.describe()}; inverse affine error {inv.affine_err:.2e}; {len(mine)} observations, shard {shard}", flush=True)

    def prefix(z, text):
        od_ = {"observation/image": z["image"], "observation/wrist_image": z["wrist"],
               "observation/state": np.asarray(z["state"], np.float32), "prompt": text}
        with torch.no_grad():
            return bk.prefix(bk.observation(od_))
    t_all, done = time.time(), 0
    for jj, key in enumerate(mine):
        outp = os.path.join(od, key + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        s = np.load(os.path.join(RB4, "tune", key + ".npz"), allow_pickle=True)
        names = [str(q) for q in s["arms"]]
        info = json.loads(str(s["info"]))
        so = src[key]
        assert info["cf"] == so["cf"] and info["oracle"] == so["oracle_named"], (key, info["cf"], so["cf"])
        rows = {a: inv.rows(r) for a, r in zip(names, s["raw"])}
        rt = max(float(np.abs(bk.unnormalize(rows[a]).reshape(RB.N_SAMP, bk.H, -1) - s["raw"][k]).max()) for k, a in enumerate(names))
        assert rt < INV_TOL, (key, rt)
        z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
        ex = dict(key=key, roundtrip=rt, cf=so["cf"], bind_cf=so["bind_cf"], bind_named=so["bind_named"])
        arrs = {}
        with torch.no_grad():
            if jj < CHECK_N:                                             # the stored instruction FM must be reproduced from the inversion
                pre_i = prefix(z, str(z["text"]))
                chk = [a for a in ("u11", "dir|all|w1|cf", "d1|early|w1|cf") if a in rows]
                f_i = RB.fm_loss(bk, pre_i, np.concatenate([rows[a] for a in chk]), rn.E).reshape(len(chk), RB.N_SAMP)
                ex["fm_instr_check"] = {a: float(np.abs(f_i[k] - s["fm"][names.index(a)]).max()) for k, a in enumerate(chk)}
            conds = {"null": ("", names)}
            cf_arms = [a for a in names if a.endswith("|cf")] + ["u11", "u21"]
            nm_arms = [a for a in names if a.endswith("|named")] + ["u11", "u21"]
            if so["bind_cf"]:
                conds["bcf"] = (so["bind_cf"]["text"], cf_arms)
            if so["bind_named"]:
                conds["bnm"] = (so["bind_named"]["text"], nm_arms)
            for j, (text, arms) in conds.items():
                pre_j = prefix(z, text)
                r11 = rn.gd.sample(RB.N_SAMP, pre_j, RB.SEED_A)
                r21 = rn.gd.sample(RB.N_SAMP, pre_j, RB.SEED_B)
                allr = np.concatenate([rows[a] for a in arms] + [r11, r21])
                f = RB.fm_loss(bk, pre_j, allr, rn.E).reshape(len(arms) + 2, RB.N_SAMP)
                arrs[f"names_{j}"] = np.array(list(arms) + ["ref11", "ref21"])
                arrs[f"fm_{j}"] = f.astype(np.float32)
                arrs[f"refraw_{j}"] = np.stack([bk.unnormalize(r11), bk.unnormalize(r21)]).reshape(2, RB.N_SAMP, bk.H, -1).astype(np.float32)
                arrs[f"refoob_{j}"] = np.array([float((np.abs(r11) > RB.OOB).mean()), float((np.abs(r21) > RB.OOB).mean())])
        ex["secs"] = time.time() - t0
        R2A.save_atomic(outp, info=np.array(json.dumps(ex)), **arrs)
        done += 1
        print(f"[rb4b judge] {key}: judges {sorted(conds)}; roundtrip {rt:.1e}; "
              + (f"instr-FM check {json.dumps(ex['fm_instr_check'])}; " if "fm_instr_check" in ex else "")
              + f"{ex['secs']:.1f}s ({done} done, {time.time() - t_all:.0f}s)", flush=True)
    print(f"[rb4b judge] shard {shard} finished: {done} new", flush=True)


# ── stage 0 (chunks env) ─────────────────────────────────────────────────────────────────────────────────────────
def _out_shares(j, arms):
    """{arm: outlier share} under judge j: FM > mean + 2 sd of ref11's 12 (ref21 included as an arm)."""
    nm = [str(q) for q in j["names"]]
    f = j["fm"]
    r = f[nm.index("ref11")]
    thr = float(r.mean()) + RB.FM_Z * float(r.std())
    return {a: float((f[nm.index(a)] > thr).mean()) for a in arms if a in nm}, {a: float(f[nm.index(a)].mean()) for a in nm}


def stage0(record):
    from lifelong_ood import stats
    RB._load_meta()
    tune, _, _ = RB._keys_all()
    old = jload(os.path.join(RB4, "stage0.json"))
    src = jload(os.path.join(OUT, "sources.json"))
    rows, cov, checks, rts, hitY = [], dict(bind_cf=0, null_cf=0, bind_named=0, null_named=0), [], [], {"bind": [], "null": []}
    for key in tune:
        fo = os.path.join(RB4, "tune", key + ".npz")
        fj = os.path.join(OUT, "judge", key + ".npz")
        assert os.path.exists(fj), f"judging incomplete: {key}"
        sh, info = RB.obs_rows(key, fo)
        zj = np.load(fj, allow_pickle=True)
        ex = json.loads(str(zj["info"]))
        rts.append(ex["roundtrip"])
        if "fm_instr_check" in ex:
            checks.append(max(ex["fm_instr_check"].values()))
        J = {k: dict(names=zj[f"names_{k}"], fm=zj[f"fm_{k}"]) for k in ("null", "bcf", "bnm") if f"names_{k}" in zj.files}
        z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
        cn = [str(q) for q in z["cand_names"]]
        eef, cp = z["state"][:3], z["cand_pos"]
        # the cf arms' judge: binding where a source exists, else the null prompt
        jc = "bcf" if "bcf" in J else "null"
        jn = "bnm" if "bnm" in J else "null"
        cov["bind_cf" if jc == "bcf" else "null_cf"] += int(info["cf"] is not None)
        cov["bind_named" if jn == "bnm" else "null_named"] += 1
        arms = list(sh)
        for tag, jk, sel in (("P", jc, lambda a: a.endswith("|cf") or a.startswith("posthoc")), ("N", "null", lambda a: True),
                             ("Q", jn, lambda a: a.endswith("|named"))):
            outs, means = _out_shares(J[jk], [a for a in arms if sel(a)] + ["ref21", "u11", "u21"])
            for a, v in outs.items():
                if a in ("ref21", "u11", "u21"):
                    sh.setdefault(f"{a}#{tag}", {})["out"] = v
                    sh[f"{a}#{tag}"]["fm"] = means[a]
                else:
                    sh[a][f"out{tag}"] = v
                    sh[a][f"fm{tag}"] = means[a]
            sh[f"ref11#{tag}"] = dict(out=None, fm=means["ref11"])
        for jk, lab in (("bcf", "bind"), ("null", "null")):            # descriptive: does the judging reference itself hit Y?
            if jk in J and info["cf"] is not None:
                rr = zj[f"refraw_{jk}"][0]
                pts = [R2A.pointed(r, eef, cp)[0] for r in rr]
                hitY[lab].append(float(np.mean([p >= 0 and cn[p] == info["cf"] for p in pts])))
        sh["_judge_cf"] = jc
        rows.append((key, sh))

    def fpairs(rs, a, b, fa, fb):
        ep = {}
        for key, sh in rs:
            if a in sh and b in sh and sh[a].get(fa) is not None and sh[b].get(fb) is not None:
                m = RB.META[key]
                ep.setdefault((m["task"], m["trial"]), []).append((sh[a][fa], sh[b][fb]))
        out = {}
        for (t, _), v in ep.items():
            pa, pb = out.setdefault(t, ([], []))
            pa.append(float(np.mean([q[0] for q in v])))
            pb.append(float(np.mean([q[1] for q in v])))
        return {t: (np.asarray(p[0]), np.asarray(p[1])) for t, p in out.items()}
    floor = RB._md(RB._pairs(rows, "u21@cf", "u11@cf"))
    thr = RB.MOVE_BAR + max(RB.FLOOR_MIN, abs(floor))
    bind_rows = [(k, sh) for k, sh in rows if sh["_judge_cf"] == "bcf"]
    null_rows = [(k, sh) for k, sh in rows if sh["_judge_cf"] == "null"]
    res, mism = {}, []
    for c in RB.grid():
        nm, cf = c["name"] + "|named", c["name"] + "|cf"
        g = stats.bootstrap_diff(RB._pairs(rows, cf, "u11@cf"), B=10000, seed=0)
        r = dict(g_cf=g["diff"], g_cf_ci=[g["lo"], g["hi"]], n_cf=g["n_pairs"], tasks_cf=g["n_tasks"],
                 g_named=RB._md(RB._pairs(rows, nm, "u11")), still_named=RB._md(RB._pairs(rows, nm, "u11", "still")),
                 still_cf=RB._md(RB._pairs(rows, cf, "u11@cf", "still")), oob_rise_cf=RB._md(RB._pairs(rows, cf, "u11", "oob")))
        for k_ in ("g_cf", "g_named", "still_named", "still_cf", "oob_rise_cf"):
            if abs(r[k_] - old["table"][c["name"]][k_]) > 1e-9:
                mism.append((c["name"], k_, r[k_], old["table"][c["name"]][k_]))
        bd = stats.bootstrap_diff(fpairs(rows, cf, "ref21#P", "outP", "out"), B=10000, seed=0)
        r.update(fm_out_rise_new=bd["diff"], fm_out_rise_new_ci=[bd["lo"], bd["hi"]],
                 fm_out_rise_new_bind_only=RB._md(fpairs(bind_rows, cf, "ref21#P", "outP", "out")),
                 fm_out_rise_new_null_only=RB._md(fpairs(null_rows, cf, "ref21#P", "outP", "out")),
                 fm_out_rise_null_all=RB._md(fpairs(rows, cf, "ref21#N", "outN", "out")),
                 fm_out_rise_named_new=RB._md(fpairs(rows, nm, "ref21#Q", "outQ", "out")),
                 fm_ratio_new=float(np.mean([sh[cf]["fmP"] for _, sh in rows if cf in sh]) /
                                    np.mean([sh["ref11#P"]["fm"] for _, sh in rows if cf in sh])),
                 fm_out_rise_old=old["table"][c["name"]]["fm_out_rise_cf"])
        r["movable"] = bool(r["g_cf"] >= thr)
        r["on_manifold"] = bool(r["fm_out_rise_new"] <= RB.ADMIT_FM and r["oob_rise_cf"] <= RB.ADMIT_OOB and r["still_cf"] <= RB.ADMIT_STILL)
        r["admissible"] = bool(r["movable"] and r["on_manifold"] and r["g_named"] >= RB.ADMIT_NAMED and r["still_named"] <= RB.ADMIT_STILL)
        res[c["name"]] = r
    assert not mism, f"recomputed movability differs from R4-bind's stage0.json: {mism[:4]}"
    ref = {tg: dict(fm_out_rise_new=RB._md(fpairs(rows, f"posthoc|{tg}|cf", "ref21#P", "outP", "out")),
                    fm_out_rise_null_all=RB._md(fpairs(rows, f"posthoc|{tg}|cf", "ref21#N", "outN", "out"))) for tg in RB.TARGETS}
    cg = {c["name"]: c for c in RB.grid()}
    adm = [k for k, v in res.items() if v["admissible"]]
    chosen = None
    if adm:
        best = max(res[k]["g_cf"] for k in adm)
        near = [k for k in adm if res[k]["g_cf"] >= best - RB.TIE]
        chosen = min(near, key=lambda k: (cg[k]["w"], RB.SCHEDS.index(cg[k]["sched"]), RB.TARGETS.index(cg[k]["target"])))
    out = dict(stage0_pass=bool(chosen is not None), any_movable=any(v["movable"] for v in res.values()),
               any_movable_on_manifold=any(v["movable"] and v["on_manifold"] for v in res.values()), threshold=thr,
               floor_cf_u21_minus_u11=floor, chosen=chosen, table=res, posthoc_reference=ref, n_tune=len(rows),
               coverage=dict(cov, cf_obs=sum("u11@cf" in sh for _, sh in rows),
                             bind_cf_tasks=sorted({RB.META[k]["task"] for k, sh in bind_rows}),
                             bind_cf_episodes=len({(RB.META[k]["task"], RB.META[k]["trial"]) for k, sh in bind_rows})),
               u_vs_ref_primary=dict(u11=RB._md(fpairs(rows, "u11#P", "ref21#P", "out", "out")),
                                     u21=RB._md(fpairs(rows, "u21#P", "ref21#P", "out", "out")),
                                     fm_u11_over_ref11=float(np.mean([sh["u11#P"]["fm"] for _, sh in rows]) /
                                                             np.mean([sh["ref11#P"]["fm"] for _, sh in rows]))),
               reference_hit_at_cf=dict(bind=float(np.mean(hitY["bind"])) if hitY["bind"] else None, n_bind=len(hitY["bind"]),
                                        null=float(np.mean(hitY["null"])) if hitY["null"] else None, n_null=len(hitY["null"]),
                                        u11=old["unsteered_at_cf"]),
               inversion=dict(max_roundtrip=float(max(rts)), instr_fm_check_max=float(max(checks)) if checks else None,
                              n_checked=len(checks)),
               old_judge_stage0_pass=old["stage0_pass"])
    RB.dump(os.path.join(OUT, "stage0.json"), out)
    for k, v in res.items():
        print(f"[rb4b stage0] {k:14s} g_cf {v['g_cf']:+.3f} [{v['g_cf_ci'][0]:+.3f}, {v['g_cf_ci'][1]:+.3f}] fm_out NEW {v['fm_out_rise_new']:+.3f} "
              f"[{v['fm_out_rise_new_ci'][0]:+.3f}, {v['fm_out_rise_new_ci'][1]:+.3f}] (bind {v['fm_out_rise_new_bind_only']:+.3f} / null "
              f"{v['fm_out_rise_new_null_only']:+.3f}; null-all {v['fm_out_rise_null_all']:+.3f}; named {v['fm_out_rise_named_new']:+.3f}; "
              f"ratio {v['fm_ratio_new']:.3f}) OLD {v['fm_out_rise_old']:+.3f}; still cf {v['still_cf']:+.3f} oob {v['oob_rise_cf']:+.3f} "
              f"g_named {v['g_named']:+.3f} still n {v['still_named']:+.3f} | movable {v['movable']} manifold {v['on_manifold']} adm {v['admissible']}")
    print("[rb4b stage0] posthoc " + json.dumps(ref))
    print("[rb4b stage0] coverage " + json.dumps(out["coverage"]) + "; u vs ref " + json.dumps(out["u_vs_ref_primary"]) +
          "; reference hit at cf " + json.dumps(out["reference_hit_at_cf"]) + "; inversion " + json.dumps(out["inversion"]))
    print(f"[rb4b stage0] threshold {thr:.3f}; stage 0 {'PASSES' if out['stage0_pass'] else 'FAILS'}; chosen {chosen}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rb4b_stage0", bench="libero_90_c", tag="stage0", seed=RB.SEED_A, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(RB._settings(), judge="binding-source FM (R3c oracle rule) with own unguided binding "
                                        "references (ref11 mean + 2 sd; rise vs ref21); null-prompt fallback"),
                            script="experiments/rb4b_bind.py", command=" ".join(sys.argv), backbone=dict(config=RB.CFG, checkpoint=RB.CKPT),
                            artifacts=[os.path.join(OUT, "stage0.json"), os.path.join(OUT, "sources.json")]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("sources", "judge", "stage0", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.sources:
        sources()
    if a.judge:
        judge(a.shard)
    if a.stage0:
        stage0(a.record)
