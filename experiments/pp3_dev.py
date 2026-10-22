"""P3 DEVELOPMENT PRE-CHECK (docs/PLAN_V4.md section 8, P3 row), on the existing pi05_libero checkpoint: does a closed-form
relation over part fields reproduce the instruction's field?  Leave-task-out over the 40 training tasks.  numpy only.

Target   Y = eps(x | o, instruction) - eps(x | o, null) at the cached points (4 per level on the flow path of the demo's chunk;
         experiments/pp_fields.py), for every training observation (200).
Roles    obj (the named object part), goal (the own goal part), verb (the own verb part); deltas X_r = eps(part_r) - eps(null).
         Three part variants are fitted separately: bare spliced ("the X"), plain (unspliced), template ("pick up the X" /
         "put it G").  A missing role (fixture tasks have no goal) contributes zero.
R0       per-cell ridge (protos.relation), the ridge scale chosen by an inner leave-task-out; outer leave-task-out score =
         1 - SSE / SST pooled over held-out tasks (the fraction of the instruction-vs-null gap explained)
R1       R0 + context terms (gripper opening, frames since the last gripper event; standardised over the training observations)
Baselines on the same held-out fields:
         equal-weight superposition (every w = 1: composable diffusion's sum)
         Comp-1 / Comp-3: the nearest OTHER training instructions by flow loss on the demo chunk at this observation, softmax
         weighted (temperature = std of the top-3 losses), Y_hat = sum_k w_k (eps_k - eps_null).  Comp uses the demonstration
         to rank; R0 / R1 do not.
Ceiling  the instruction's own field from the 40-instruction batch (a different batch composition: bf16 batch numerics) as the
         prediction: no relation can explain more of the gap than this duplicate does
Stability  bootstrap over tasks (B = 100): correlation of each refit's weights with the full fit
Readout    AND / OR / THEN per role and dimension group (protos.relation.readout)
go: R0 explains >= 0.70 of the gap; template stability >= 0.8.

    python experiments/pp3_dev.py [--record]      (via SLURM: slurm/jobs/cpu.sbatch)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "protos")
ROLES = ("obj", "goal", "verb")
VARIANTS = {"bare": ("obj:named", "goal:own", "verb:own"), "plain": ("obj:named:plain", "goal:own:plain", "verb:own"),
            "tmpl": ("obj:named:tmpl", "goal:own:tmpl", "verb:own")}


def frac(Y, Yh):
    return float(1.0 - ((Y - Yh) ** 2).sum() / max((Y ** 2).sum(), 1e-12))


def main(record=False, tag=""):
    from lifelong_ood.protos import relation as R
    from lifelong_ood.protos import store as S
    idx = json.load(open(os.path.join(OUT, "index.json")))
    od = os.path.join(OUT, "obs" + (f"_{tag}" if tag else ""))
    L = len(S.LEVELS)
    data = []
    for o in idx["obs"]:
        if o["set"] != "train":
            continue
        p = os.path.join(od, o["key"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        Ec, Ei = z["E_core"].astype(np.float64), z["E_instr"].astype(np.float64)
        e_null = Ec[names.index("null")]
        Y = Ec[names.index("instr")] - e_null
        own_gap = float(np.abs(Ei[o["hf_task"]] - Ec[names.index("instr")]).max())       # same prompt, no splice: ~0
        Xs = {}
        for v, rn in VARIANTS.items():
            Xs[v] = np.stack([Ec[names.index(n)] - e_null if n in names else np.zeros_like(Y) for n in rn])
        u = np.asarray(S.LEVELS)[:, None, None]
        loss = (((Ei - z["eps_true"][None]) ** 2) / (1 - u[None]) ** 2).mean(axis=(1, 2, 3))   # flow loss per instruction
        data.append(dict(o=o, Y=Y, X=Xs, e_null=e_null, Ei=Ei, loss=loss, own_gap=own_gap))
    print(f"[pp3] {len(data)} training observations over {len({d['o']['hf_task'] for d in data})} tasks; "
          f"max |own instruction field - instr condition| {max(d['own_gap'] for d in data):.2e}", flush=True)
    ctx = np.array([[d["o"]["grip"], d["o"]["since"]] for d in data], float)
    ctx = (ctx - ctx.mean(0)) / np.maximum(ctx.std(0), 1e-9)
    res = dict(n_obs=len(data), n_tasks=len({d["o"]["hf_task"] for d in data}), variants={})
    grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

    def per_level(S_, T_):
        Sx, Tx = S_.reshape(3, 2, 2, L), T_.reshape(3, 2, 2, L)
        return dict(level={str(S.LEVELS[l]): float(1 - Sx[..., l].sum() / Tx[..., l].sum()) for l in range(L)},
                    group={g: float(1 - Sx[i].sum() / Tx[i].sum()) for i, g in enumerate(R.GROUPS)},
                    phase={p: float(1 - Sx[:, :, i].sum() / Tx[:, :, i].sum()) for i, p in enumerate(R.PHASES)})

    for v in VARIANTS:
        st0, st1 = R.Stats(), R.Stats()
        for d, z in zip(data, ctx):
            t, ph = d["o"]["hf_task"], d["o"]["phase"]
            st0.add(t, R.obs_stats(R.r0_features(d["X"][v]), d["Y"], ph))
            st1.add(t, R.obs_stats(R.r1_features(d["X"][v], z), d["Y"], ph))
        ex0, per0, (S0, T0) = R.nested_lto(st0, grid)
        ex1, per1, (S1, T1) = R.nested_lto(st1, grid)
        lam = float(np.median([p["lam"] for p in per0.values()]))
        W = R.solve(st0.total(), lam)
        stab = R.bootstrap_stability(st0, lam, B=100, seed=0, roles=list(ROLES), L=L)
        rd = R.readout(W, list(ROLES), L)
        Tm = R.templates(W, list(ROLES), L)
        weights = {r: {g: {p: float(Tm[r][gi, :, pi].mean()) for pi, p in enumerate(R.PHASES)} for gi, g in enumerate(R.GROUPS)}
                   for r in ROLES}
        # baselines on the same held-out fields (no fitting: equal weights; retrieval: Comp-1 / Comp-3)
        Yall, Yeq, Yc1, Yc3, Ydup = [], [], [], [], []
        for d in data:
            k = d["o"]["hf_task"]
            Yall.append(d["Y"])
            Ydup.append(d["Ei"][k] - d["e_null"])           # the same instruction in another batch: the numerical ceiling
            Yeq.append(d["X"][v].sum(0))
            cand = np.array([j for j in range(len(d["loss"])) if j != k])
            Ls = d["loss"][cand]
            order = cand[np.argsort(Ls)]
            top = order[:3]
            lt = d["loss"][top]
            T = max(float(lt.std()), 1e-6)
            w = np.exp(-(lt - lt.min()) / T)
            w /= w.sum()
            Yc1.append(d["Ei"][order[0]] - d["e_null"])
            Yc3.append(sum(wi * (d["Ei"][j] - d["e_null"]) for wi, j in zip(w, top)))
        Yall, Yeq, Yc1, Yc3, Ydup = map(np.stack, (Yall, Yeq, Yc1, Yc3, Ydup))

        def lvl(Yh):
            return {str(S.LEVELS[l]): frac(Yall[:, l], Yh[:, l]) for l in range(L)}
        per_task = {int(t): round(p["explained"], 3) for t, p in per0.items()}
        suite_of = {d["o"]["hf_task"]: d["o"]["suite"] for d in data}
        by_suite = {}
        for su in sorted(set(suite_of.values())):
            ts = [t for t in per0 if suite_of[t] == su]
            m = np.array([suite_of[d["o"]["hf_task"]] == su for d in data])
            by_suite[su] = dict(R0=float(1 - sum(per0[t]["sse"].sum() for t in ts) / sum(per0[t]["sst"].sum() for t in ts)),
                                equal=frac(Yall[m], Yeq[m]), comp1=frac(Yall[m], Yc1[m]), comp3=frac(Yall[m], Yc3[m]), tasks=len(ts))
        ns = [t for t in per0 if suite_of[t] != "libero_spatial"]
        mns = np.array([suite_of[d["o"]["hf_task"]] != "libero_spatial" for d in data])
        no_spatial = dict(R0=float(1 - sum(per0[t]["sse"].sum() for t in ns) / sum(per0[t]["sst"].sum() for t in ns)),
                          equal=frac(Yall[mns], Yeq[mns]), comp3=frac(Yall[mns], Yc3[mns]), tasks=len(ns))
        res["variants"][v] = dict(
            R0=dict(explained=ex0, lam_median=lam, **per_level(S0, T0), per_task=per_task, by_suite=by_suite, no_spatial=no_spatial,
                    per_task_median=float(np.median(list(per_task.values())))),
            R1=dict(explained=ex1, gain_over_R0=ex1 - ex0, **per_level(S1, T1)),
            equal_weight=dict(explained=frac(Yall, Yeq), level=lvl(Yeq)),
            comp1=dict(explained=frac(Yall, Yc1), level=lvl(Yc1)),
            comp3=dict(explained=frac(Yall, Yc3), level=lvl(Yc3)),
            duplicate_ceiling=dict(explained=frac(Yall, Ydup), level=lvl(Ydup)),
            stability=stab, readout=rd, weights=weights)
        print(f"[pp3] {v}: R0 {ex0:.3f} (lam {lam:g}), R1 {ex1:.3f}, equal {frac(Yall, Yeq):.3f}, Comp-1 {frac(Yall, Yc1):.3f}, "
              f"Comp-3 {frac(Yall, Yc3):.3f}, duplicate ceiling {frac(Yall, Ydup):.3f}; stability median {stab['median']:.3f} (p05 {stab['p05']:.3f})", flush=True)
        print("    by suite", {k: {m: round(x, 2) for m, x in v.items()} for k, v in by_suite.items()}, "| without spatial",
              {m: round(x, 2) for m, x in no_spatial.items()}, flush=True)
        for r in ROLES:
            print("   ", r, {g: (round(x["pre"], 2), round(x["post"], 2), x["label"]) for g, x in rd[r].items()}, flush=True)
    best = max(res["variants"], key=lambda v: res["variants"][v]["R0"]["explained"])
    b = res["variants"][best]
    res["go"] = dict(best_variant=best, R0_explained=b["R0"]["explained"], R0_go=bool(b["R0"]["explained"] >= 0.70),
                     stability=b["stability"]["median"], stability_go=bool(b["stability"]["median"] >= 0.8),
                     R0_beats_equal=bool(b["R0"]["explained"] > b["equal_weight"]["explained"]),
                     R0_beats_comp3=bool(b["R0"]["explained"] > b["comp3"]["explained"]))
    path = os.path.join(OUT, f"pp3_dev{('_' + tag) if tag else ''}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["go"], indent=1))
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pp3_dev", bench="libero", tag=tag or "pi05_libero", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(levels=list(S.LEVELS), roles=list(ROLES), variants=VARIANTS, lam_grid=list(grid)),
                            script="experiments/pp3_dev.py", command=" ".join(sys.argv), backbone=dict(config="pi05_libero"),
                            artifacts=[path]))


if __name__ == "__main__":
    tag = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--tag=")), "")
    main("--record" in sys.argv, tag)
