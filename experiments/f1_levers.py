"""Plan F1, step 2: the fit-time levers, measured offline by HELD-OUT flow loss against fit minutes (docs/FIT_TIME.md).  GPU part
writes ledger f1_levers rows (tag concept_<name> / leaf_<task>); --read (CPU) summarises them (ledger f1_levers tag read).

Design set: E5's 26 new prototypes (object types, instance children, goals; experiments/e5_read.py), so that the recipe is chosen
on units the confirmatory test (experiments/prereg/f1_fast_fit.py: E3's library) does not refit.  For each concept:
  * training frames: its fit's own (demonstration 0 of every task of the concept, cut as c0_fit cut them; frame count checked);
  * held-out frames: demonstration 1 of every task of the concept, cut the same way (no unit has seen it; h1_donate.py uses the
    same held-out episodes), at most 256 evenly spaced, 256 fixed (frame, time, noise) draws (adapt.fixed_draws);
  * the prefixes of both cached once (lever c), so every arm runs the fit's own 1,000 steps at the cached step time;
  * arms, each evaluated on the held-out draws every 25 steps (lever a: how many steps are needed):
      scratch       the fit's recipe (rank 4, lr 1e-4, batch 32, zero-initialised LoRA = the policy);
      near          warm start from the existing unit (E3's library and G4's held concepts, pi0.5's own demonstrations, of the
                    same kind: approach or goal) with the lowest flow loss on the concept's TRAINING frames (c0_fit.init_best);
      parent        for an instance child: warm start from its object type (lever b);
      near_sub      warm start from the unit whose update subspace is closest to the finished scratch fit (mean cos^2 of the
                    principal angles of B, h1_donate.py's measure); an oracle (it needs the fit it would shorten), run only when it
                    differs from near, to see whether the loss-chosen unit is the subspace-nearest one;
      recorded      the existing prototype file (its 1,000 lazy steps; a check that the cached fit reproduces it);
  * lever (d): the training frames per demonstration against the demonstration's length (phase frames only?).
Leaf mode (--leaf <task>, lever e): the per-task one-demonstration LoRA (r3_fewshot.py: every frame of demonstration 0) at rank 4
and rank 16, held-out = demonstration 1 of the task (every frame, at most 256), the same schedule.

    sbatch ... SCRIPT="experiments/f1_levers.py --concept XG0" slurm/jobs/robot_gpu.sbatch
    $PY experiments/f1_levers.py --launch          # one job per design concept and per leaf task
    $PY experiments/f1_levers.py --read [--record]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1_common as F                                               # noqa: E402

EVAL_EVERY, STEPS = 25, 1000
LEAF_TASKS = None                                                   # set in leaf_tasks()


def leaf_tasks():
    """Every third of E5's 37 tasks by id (12 tasks): the leaf comparison's design sample."""
    return sorted(int(t) for t in F.E5PLAN)[::3]


def overlap_B(za, zb):
    """h1_donate.py's subspace measure between two saved updates {module: {A, B}}: per module the mean cos^2 of the principal
    angles between the column spaces of B (rank-normalised; 1 = the same subspace), averaged over the modules both hold."""
    import torch
    mods = sorted(set(za) & set(zb))
    out = []
    for m in mods:
        qa = torch.linalg.qr(za[m]["B"].float())[0]
        qb = torch.linalg.qr(zb[m]["B"].float())[0]
        out.append(float((qa.T @ qb).pow(2).sum() / qa.shape[1]))
    return float(np.mean(out))


def run_concept(a):
    c0 = F.c0(tmp=a.tmp)
    import torch
    from lifelong_ood import ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    t_start = time.time()
    bk, norm = c0.backbone("pi05_libero")
    fa = F.fit_args(a.concept)
    row = F.fit_row(a.concept)
    obs, full, used = c0.concept_frames(fa, bk, norm, fa.demo, fa.ndemo)
    assert len(obs) == row["metrics"]["frames"], (len(obs), row["metrics"]["frames"])
    t0 = time.time()
    bank = adapt.PrefixBank(bk, obs, store="cpu" if len(obs) > 1200 else None)
    torch.cuda.synchronize()
    t_cache = time.time() - t0
    state, banks = adapt.inject(bk, scope="expert")
    fa.heldout, fa.eval_frames = 1, 256
    ev = c0.heldout_setup(fa, bk, norm, state)
    evaluate = lambda keys: adapt.heldout_loss(bk, state, keys, ev["bank"], ev["full"], ev["draws"])   # noqa: E731
    kind_ = "goal" if fa.goal else "approach"
    cands = F.pool(kind_, exclude=[a.concept])
    choice = c0.init_best(argparse.Namespace(init_best="+".join(cands)), bk, state, bank, full)
    res = dict(concept=a.concept, kind=kind_, parent=F.PARENT.get(a.concept), frames=len(obs), demos=len(used), used=used,
               heldout_frames=ev["frames"], heldout_demos=len(ev["used"]), frozen_heldout=ev["frozen"], cache_secs=t_cache,
               startup_secs=t0 - t_start, init_choice=choice, candidates=cands, arms={})

    def copy_into(key, src_path):
        z = torch.load(os.path.join(F.R, src_path), map_location="cpu")
        by = {b.name: b for b in banks}
        for n_, w in z["banks"].items():
            by[n_].A[key].weight.data.copy_(w["A"].to(by[n_].A[key].weight.dtype))
            by[n_].B[key].weight.data.copy_(w["B"].to(by[n_].B[key].weight.dtype))

    def arm(name, init_path, rank=fa.rank, bank_=bank, full_=full, evaluate_=evaluate):
        key = f"f1_{name}"
        params = add_node(banks, key, rank)
        if init_path:
            copy_into(key, init_path)
        e0 = evaluate_({key})
        prof = {}
        t = time.time()
        hist = adapt.train(bk, state, key, params, bank_, full_, steps=a.steps, lr=fa.lr, batch=fa.batch, log_every=EVAL_EVERY,
                           log=lambda x: None, eval_fn=lambda: evaluate_({key}), eval_every=EVAL_EVERY, profile=prof)
        torch.cuda.synchronize()
        secs = time.time() - t
        ev_secs = secs - sum(v for k, v in prof.items() if not k.endswith("_n"))
        hist["eval_step"].insert(0, 0)
        hist["eval"].insert(0, e0)
        sd = {b.name: dict(A=b.A[key].weight.detach().cpu().half(), B=b.B[key].weight.detach().cpu().half()) for b in banks if key in b.A}
        out = dict(init=init_path, rank=rank, curve=hist, train_secs=secs - ev_secs, eval_secs=ev_secs,
                   step_secs=(secs - ev_secs) / a.steps, profile=prof)
        print(f"[f1l] {a.concept} {name}: held-out {e0:.4f} -> {hist['eval'][-1]:.4f} (min {min(hist['eval']):.4f} at step "
              f"{hist['eval_step'][int(np.argmin(hist['eval']))]}), {out['step_secs']:.3f} s/step", flush=True)
        return out, sd

    res["arms"]["scratch"], sd_scratch = arm("scratch", "")
    if choice["chosen_path"]:
        res["arms"]["near"], _ = arm("near", choice["chosen_path"])
    if res["parent"]:
        res["arms"]["parent"], _ = arm("parent", f"prim_{res['parent']}_0.pt")
    subs = {n: overlap_B(sd_scratch, torch.load(os.path.join(F.R, f"prim_{n}_0.pt"), map_location="cpu")["banks"]) for n in cands}
    res["subspace_overlap_with_scratch"] = subs
    sub_best = max(subs, key=subs.get)
    res["near_sub"] = sub_best
    if sub_best != choice["chosen"]:
        res["arms"]["near_sub"], _ = arm("near_sub", f"prim_{sub_best}_0.pt")
    # the recorded prototype (lazy prefixes, 1,000 steps): its held-out loss and its subspace overlap with the cached scratch fit
    _, k_rec, _ = adapt.load(bk, os.path.join(F.R, f"prim_{a.concept}_0.pt"), state=state)
    z_rec = torch.load(os.path.join(F.R, f"prim_{a.concept}_0.pt"), map_location="cpu")["banks"]
    res["recorded"] = dict(heldout=evaluate({k_rec}), overlap_with_scratch=overlap_B(sd_scratch, z_rec),
                           final_train_loss=row["metrics"]["curve"]["loss"][-1])
    print(f"[f1l] {a.concept}: recorded prototype held-out {res['recorded']['heldout']:.4f}, overlap with the cached scratch fit "
          f"{res['recorded']['overlap_with_scratch']:.3f}; near {choice['chosen']}, subspace-nearest {sub_best}", flush=True)
    res["total_secs"] = time.time() - t_start
    print(ledger.record("f1_levers", bench="libero_90", tag=f"concept_{a.concept}", seed=0, metrics=res,
                        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                        script="experiments/f1_levers.py", command=" ".join(sys.argv)), flush=True)


def run_leaf(a):
    """Lever (e): the per-task leaf at rank 4 against rank 16 (r3_fewshot.py's data and recipe)."""
    c0 = F.c0(tmp=a.tmp)
    import torch
    from lifelong_ood import ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    bk, norm = c0.backbone("pi05_libero")
    d90 = D.Demos90()
    eps = d90.episodes_of_task(a.leaf)

    def frames(e_, cap=None):
        ep = d90.load(e_)
        T = len(ep["actions"])
        keep = np.arange(T) if cap is None else np.unique(np.linspace(0, T - 1, min(cap, T)).round().astype(int))
        obs = [d90.observation(ep, int(i)) for i in keep]
        idx, _ = D.chunk_indices(T, bk.H)
        full = D.pad_rows(c0.chunk_rows(ep, idx, range(T), norm), bk.H, bk.a, bk.A)[keep]
        return obs, full, T
    obs, full, T = frames(eps[0])
    hobs, hfull, hT = frames(eps[1], cap=256)
    t0 = time.time()
    bank = adapt.PrefixBank(bk, obs)
    torch.cuda.synchronize()
    t_cache = time.time() - t0
    hbank = adapt.PrefixBank(bk, hobs)
    draws = adapt.fixed_draws(hbank.n, bk.H, bk.A)
    state, banks = adapt.inject(bk, scope="expert")
    evaluate = lambda keys: adapt.heldout_loss(bk, state, keys, hbank, hfull, draws)               # noqa: E731
    res = dict(task=a.leaf, demo_episode=int(eps[0]), heldout_episode=int(eps[1]), frames=T, heldout_frames=hbank.n,
               cache_secs=t_cache, frozen_heldout=evaluate(set()), arms={})
    for r in (4, 16):
        key = f"f1_leaf_r{r}"
        params = add_node(banks, key, r)
        e0 = evaluate({key})
        prof = {}
        t = time.time()
        hist = adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=1e-4, batch=32, log_every=EVAL_EVERY,
                           log=lambda x: None, eval_fn=lambda: evaluate({key}), eval_every=EVAL_EVERY, profile=prof)
        torch.cuda.synchronize()
        secs = time.time() - t
        tr = sum(v for k, v in prof.items() if not k.endswith("_n"))
        hist["eval_step"].insert(0, 0)
        hist["eval"].insert(0, e0)
        res["arms"][f"r{r}"] = dict(curve=hist, train_secs=tr, eval_secs=secs - tr, step_secs=tr / a.steps,
                                    params=adapt.n_params(params), profile=prof)
        print(f"[f1l] leaf {a.leaf} rank {r}: held-out {e0:.4f} -> {hist['eval'][-1]:.4f} (min {min(hist['eval']):.4f}), "
              f"{tr / a.steps:.3f} s/step", flush=True)
    print(ledger.record("f1_levers", bench="libero_90", tag=f"leaf_{a.leaf}", seed=0, metrics=res,
                        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                        script="experiments/f1_levers.py", command=" ".join(sys.argv)), flush=True)


GRID = (25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000)
TOL_MEAN, TOL_MAX = 0.02, 0.05      # the recipe's held-out loss cost, fixed before any lever row was read: at most 2% above the
#                                     scratch fit at 1,000 steps on average over the design concepts and 5% on the worst one


def at(curve, s):
    i = curve["eval_step"].index(s)
    return curve["eval"][i]


def read(record):
    """Summary of the lever rows: per arm and step budget S the held-out loss relative to the scratch fit at 1,000 steps (mean, worst),
    the smallest S meeting TOL_MEAN / TOL_MAX, the warm-start choices, the cached fit against the recorded prototype, and lever (d)."""
    from lifelong_ood import ledger
    from lifelong_ood.policy import data as D
    rows = {r["tag"][8:]: r for r in ledger.load("f1_levers") if r["tag"].startswith("concept_")}
    leaves = {r["tag"][5:]: r for r in ledger.load("f1_levers") if r["tag"].startswith("leaf_")}
    missing = [n for n in F.E5_NAMES if n not in rows]
    out = dict(missing=missing, concepts=len(rows), grid=GRID, tol=dict(mean=TOL_MEAN, max=TOL_MAX), arms={})
    ref = {n: at(r["metrics"]["arms"]["scratch"]["curve"], 1000) for n, r in rows.items()}
    frozen = {n: r["metrics"]["frozen_heldout"] for n, r in rows.items()}

    def summarise(arm, names, fallback=None):
        names = [n for n in names if arm in rows[n]["metrics"]["arms"] or fallback]
        tab = {}
        for s in GRID:
            rel = []
            for n in names:
                a_ = rows[n]["metrics"]["arms"].get(arm) or rows[n]["metrics"]["arms"][fallback]
                rel.append(at(a_["curve"], s) / ref[n])
            tab[s] = dict(mean=float(np.mean(rel) - 1), max=float(np.max(rel) - 1), n=len(rel))
        ok = [s for s in GRID if tab[s]["mean"] <= TOL_MEAN and tab[s]["max"] <= TOL_MAX]
        best_step = {n: (rows[n]["metrics"]["arms"].get(arm) or rows[n]["metrics"]["arms"][fallback])["curve"] for n in names}
        best_step = {n: c["eval_step"][int(np.argmin(c["eval"]))] for n, c in best_step.items()}
        return dict(names=names, by_steps=tab, smallest_steps=ok[0] if ok else None, argmin_step=best_step,
                    gain_at_1000=float(np.mean([(frozen[n] - ref[n]) / frozen[n] for n in names])))
    out["arms"]["scratch"] = summarise("scratch", list(rows))
    out["arms"]["near_or_scratch"] = summarise("near", list(rows), fallback="scratch")   # scratch where no unit beats the policy
    out["arms"]["near"] = summarise("near", list(rows))
    inst = [n for n in rows if rows[n]["metrics"]["parent"]]
    out["arms"]["parent"] = summarise("parent", inst)
    out["arms"]["scratch_instances"] = summarise("scratch", inst)
    out["arms"]["near_sub"] = summarise("near_sub", list(rows))
    out["choices"] = {n: dict(near=r["metrics"]["init_choice"]["chosen"], near_sub=r["metrics"]["near_sub"],
                              parent=r["metrics"]["parent"], held=str(r["metrics"]["init_choice"]["chosen"]).startswith("Q"),
                              overlap_near=r["metrics"]["subspace_overlap_with_scratch"].get(r["metrics"]["init_choice"]["chosen"]),
                              overlap_sub=r["metrics"]["subspace_overlap_with_scratch"][r["metrics"]["near_sub"]])
                      for n, r in rows.items()}
    out["agree_near_sub"] = int(sum(v["near"] == v["near_sub"] for v in out["choices"].values()))
    out["recorded_check"] = {n: dict(recorded=r["metrics"]["recorded"]["heldout"], cached_scratch=ref[n],
                                     overlap=r["metrics"]["recorded"]["overlap_with_scratch"]) for n, r in rows.items()}
    st = [r["metrics"]["arms"]["scratch"]["step_secs"] for r in rows.values()]
    out["step_secs"] = dict(mean=float(np.mean(st)), min=float(np.min(st)), max=float(np.max(st)))
    out["cache_secs_per_frame"] = float(np.mean([r["metrics"]["cache_secs"] / r["metrics"]["frames"] for r in rows.values()]))
    out["startup_secs"] = float(np.mean([r["metrics"]["startup_secs"] for r in rows.values()]))
    # lever (d): phase frames against the demonstrations' lengths, over every E3 and E5 unit
    d90 = D.Demos90()
    dl = {}
    for n in F.E3_NAMES + F.E5_NAMES:
        r = F.fit_row(n)
        eps = {u["task"]: d90.episodes_of_task(u["task"])[0] for u in r["metrics"]["joint"]}
        T = sum(len(d90.load(e)["actions"]) for e in eps.values())
        dl[n] = dict(frames=r["metrics"]["frames"], demo_frames=int(T), share=r["metrics"]["frames"] / T, kind=F.kind(n))
    out["frames"] = dict(per_unit=dl, share_mean=float(np.mean([v["share"] for v in dl.values()])),
                         share_goal=float(np.mean([v["share"] for v in dl.values() if v["kind"] == "goal"])),
                         share_approach=float(np.mean([v["share"] for v in dl.values() if v["kind"] == "approach"])))
    # lever (e): leaves
    if leaves:
        lt = {}
        for t, r in leaves.items():
            m = r["metrics"]
            c4, c16 = m["arms"]["r4"]["curve"], m["arms"]["r16"]["curve"]
            lt[t] = dict(r4=at(c4, 1000), r16=at(c16, 1000), frozen=m["frozen_heldout"], rel=at(c4, 1000) / at(c16, 1000) - 1,
                         step_r4=m["arms"]["r4"]["step_secs"], step_r16=m["arms"]["r16"]["step_secs"],
                         argmin_r4=c4["eval_step"][int(np.argmin(c4["eval"]))], argmin_r16=c16["eval_step"][int(np.argmin(c16["eval"]))],
                         by_steps_r4={s: at(c4, s) / at(c16, 1000) - 1 for s in GRID})
        out["leaves"] = dict(per_task=lt, rel_mean=float(np.mean([v["rel"] for v in lt.values()])),
                             rel_max=float(np.max([v["rel"] for v in lt.values()])),
                             step_r4=float(np.mean([v["step_r4"] for v in lt.values()])),
                             step_r16=float(np.mean([v["step_r16"] for v in lt.values()])))
    for k, v in out["arms"].items():
        print(f"[f1l] {k:18} n={len(v['names']):2} smallest S={v['smallest_steps']}  " +
              " ".join(f"{s}:{v['by_steps'][s]['mean']:+.3f}/{v['by_steps'][s]['max']:+.3f}" for s in GRID), flush=True)
    print(f"[f1l] near = subspace-nearest on {out['agree_near_sub']} of {len(rows)}; step {out['step_secs']}; missing {missing}", flush=True)
    os.makedirs(F.F1, exist_ok=True)
    json.dump(out, open(os.path.join(F.F1, "f1_levers_read.json"), "w", encoding="utf-8"), indent=1, default=str)
    if record and not missing:
        print(ledger.record("f1_levers", bench="libero_90", tag="read", seed=0, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/f1_levers.py", command=" ".join(sys.argv)), flush=True)
    return out


def launch():
    for n in F.E5_NAMES:
        print(n, F.sbatch(f"experiments/f1_levers.py --concept {n}" + (" --tmp" if TMP else "")), flush=True)
    for t in leaf_tasks():
        print("leaf", t, F.sbatch(f"experiments/f1_levers.py --leaf {t}" + (" --tmp" if TMP else "")), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="")
    ap.add_argument("--leaf", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tmp", action="store_true", help="run against the not-yet-installed edits (f1_common.c0)")
    a = ap.parse_args()
    TMP = a.tmp
    if a.launch:
        launch()
    elif a.read:
        read(a.record)
    elif a.concept:
        run_concept(a)
    elif a.leaf >= 0:
        run_leaf(a)
