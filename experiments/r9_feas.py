"""PROBE R9 feasibility (docs/PLAN_R9.md): is a failure-gated retry possible within LIBERO's budget, and does the policy already retry?
From EXISTING records only (no policy, no simulator): R7's `start` runs (results/r7/sb/start_*.npz: 32 seeds per wrong-goal episode,
each run continues after a release until success or the 400-step budget, with fresh noise at every query) and the 42 recorded zs
episodes (results/d4/roll/*_rec).  Release = the first executed open command after the first sustained grasp (>= GRASP_Q consecutive
queries whose executed gripper commands are all closed); re-grasp = another sustained grasp after that release.

    python experiments/r9_feas.py        (chunks env, via slurm/jobs/cpu.sbatch) -> results/r9/feas.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import r7_cover as R7                                             # noqa: E402

GRASP_Q = 3


def phases(g):
    """g: executed gripper commands per step (>0 closed).  -> (grasp step, first release step after it, n re-grasps after it)."""
    closed = g > 0
    n = len(g)
    run, grasp = 0, None
    for s in range(n):
        run = run + 1 if closed[s] else 0
        if run >= GRASP_Q * R7.REPLAN:
            grasp = s - run + 1
            break
    if grasp is None:
        return None, None, 0
    rel = next((s for s in range(grasp, n) if not closed[s]), None)
    if rel is None:
        return grasp, None, 0
    regr, run, s = 0, 0, rel
    while s < n:
        run = run + 1 if closed[s] else 0
        if run == GRASP_Q * R7.REPLAN:
            regr += 1
            while s < n and closed[s]:
                s += 1
            run = 0
        s += 1
    return grasp, rel, regr


def run_g(chunks, steps):
    """Executed gripper commands of one run from its stored chunks (5 per query, truncated at `steps`)."""
    g = np.asarray(chunks[:, :R7.REPLAN, 6], np.float32).reshape(-1)
    g = g[np.isfinite(g)]
    return g[:int(steps)]


def summarize(rows):
    r = [x for x in rows if x["rel"] is not None]
    rem = np.array([x["budget"] - x["rel"] for x in r]) if r else np.zeros(0)
    succ = [x for x in rows if x["success"]]
    late = [x for x in succ if x["rel"] is not None and x["succ_step"] is not None and x["succ_step"] > x["rel"] and x["regr"] > 0]
    fail = [x for x in rows if not x["success"]]
    return dict(n=len(rows), released=len(r), remaining_steps_median=float(np.median(rem)) if len(rem) else None,
                remaining_q25=float(np.percentile(rem, 25)) if len(rem) else None,
                regrasp_after_release=float(np.mean([x["regr"] > 0 for x in r])) if r else None,
                successes=len(succ), success_after_regrasp=len(late),
                failed=len(fail), failed_with_regrasp=float(np.mean([x["regr"] > 0 for x in fail if x["rel"] is not None]))
                if any(x["rel"] is not None for x in fail) else None,
                failed_no_release=sum(1 for x in fail if x["rel"] is None))


def main():
    units = [u for u in R7.jload(os.path.join(R7.OUT, "plan.json"))["units"] if u["grp"] == "wg" and u["cond"] == "start"]
    seeds, zs = {}, {}
    for u in units:
        z = np.load(os.path.join(R7.OUT, "sb", u["unit"] + ".npz"), allow_pickle=True)
        S, ST, CH = np.asarray(z["success"]).astype(bool), np.asarray(z["steps"]), np.asarray(z["chunks"], np.float32)
        for k in range(R7.K_MAX):
            g = run_g(CH[k], ST[k])
            gr, rel, regr = phases(g)
            seeds.setdefault(u["tid"], []).append(dict(success=bool(S[k]), rel=rel, regr=regr, budget=R7.MAX_STEPS,
                                                       succ_step=int(ST[k]) if S[k] else None))
        rz = np.load(os.path.join(R7.D4D, "roll", f"d4_zs_{R7.slug(u['tid'])}_rec", f"task{u['index']}_trial{u['trial']}.npz"),
                     allow_pickle=True)
        A = np.asarray(rz["action"])
        gr, rel, regr = phases(A[:, 6])
        zs.setdefault(u["tid"], []).append(dict(success=False, rel=rel, regr=regr, budget=R7.MAX_STEPS, succ_step=None, T=int(len(A))))
    res = dict(seeds={t: summarize(v) for t, v in seeds.items()}, zs={t: summarize(v) for t, v in zs.items()},
               seeds_all=summarize([x for v in seeds.values() for x in v]), zs_all=summarize([x for v in zs.values() for x in v]),
               zs_len={t: sorted(x["T"] for x in v) for t, v in zs.items()})
    os.makedirs(os.path.join(ROOT, "results", "r9"), exist_ok=True)
    R7.dump(os.path.join(ROOT, "results", "r9", "feas.json"), res)
    for k in ("seeds_all", "zs_all"):
        print(f"[r9 feas] {k}: {json.dumps(res[k])}", flush=True)
    for t in sorted(seeds):
        print(f"[r9 feas] {t} seeds: {json.dumps(res['seeds'][t])}", flush=True)
        print(f"[r9 feas] {t} zs: {json.dumps(res['zs'][t])}; zs episode lengths {res['zs_len'][t]}", flush=True)


if __name__ == "__main__":
    main()
