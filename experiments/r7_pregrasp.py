"""PROBE R7 option 1 (docs/PLAN_R7.md section 7; pre-registration experiments/prereg/pr7_pregrasp.py, ledger m3_prereg tag
pr7_pregrasp): coverage@K of the frozen pi05_l90c's own samples from the PRE-GRASP state of the same 42 wrong-goal + 37 control
episodes as R7 stage A.  The pre-grasp state = the episode's LAST D4 `pre` observation (experiments/d4_bind.py: an approach query
before the bound step, hand >= R2A.HAND_CLEAR xy from every object, open gripper).  Rollouts reuse r7_cover.run_unit unchanged.

Stages (SLURM only):
  --plan                  (chunks env)   -> results/r7/plan_pg.json
  --sandbox --shard i/n   (LIBERO env + experiments/d5_serve.py)  -> results/r7/sb/pregrasp_<obs>.npz
  --read [--record]       (chunks env)   pre-grasp coverage + carry / pre-grasp / start on the same states -> results/r7/read_pg.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import r7_cover as R7                                             # noqa: E402

PLAN_PG = os.path.join(R7.OUT, "plan_pg.json")
K_PRIMARY, BAR = 16, 0.5          # H-pregrasp: pooled coverage@16 on wrong-goal pre-grasp states >= 0.5 (point estimate)


def plan():
    units = R7.jload(os.path.join(R7.OUT, "plan.json"))["units"]
    d5 = {o["key"]: o for o in R7.jload(os.path.join(R7.D5D, "index.json"))["dev"]}
    out, missing = [], []
    for u in units:
        if u["cond"] != "carry":
            continue
        o = d5[u["obs"]]
        x = R7.jload(os.path.join(R7.D4D, "dev", "index", f"{R7.slug(u['tid'])}.json"))
        ep = [e for e in x["episodes"] if int(e["trial"]) == int(u["trial"])]
        pre = sorted(int(m["j"]) for e in ep for m in e["obs"] if m["role"] == "pre")
        if not pre:
            missing.append(u["obs"])
            continue
        j = pre[-1]
        assert j < int(o["j"]) and j % R7.REPLAN == 0, (u["obs"], j, o["j"])
        seed_start = int(o["zs_seed0"]) - int(o["j"]) // R7.REPLAN
        out.append(dict(u, unit=f"pregrasp_{u['obs'][3:]}", cond="pregrasp", j=j, carry_j=int(o["j"]),
                        grip_cmds=[-1.0] * R7.WAIT + o["grip_cmds"][R7.WAIT:R7.WAIT + j],
                        q_budget=(R7.MAX_STEPS - j) // R7.REPLAN, zs_seed0=seed_start + j // R7.REPLAN))
    by = {}
    for u in out:
        k = f"{u['tid']}|{u['grp']}"
        by[k] = by.get(k, 0) + 1
    R7.dump(PLAN_PG, dict(units=out, counts=by, missing=missing,
                          gap_steps=dict(median=float(np.median([u["carry_j"] - u["j"] for u in out])),
                                         min=int(min(u["carry_j"] - u["j"] for u in out)),
                                         max=int(max(u["carry_j"] - u["j"] for u in out)))))
    print(f"[r7pg plan] {len(out)} units; missing {missing}; counts {by}", flush=True)


def sandbox(shard, port):
    from openpi_client import websocket_client_policy as _wcp
    import d5_goal as D5
    units = R7.jload(PLAN_PG)["units"]
    i, n = map(int, shard.split("/"))
    od = os.path.join(R7.OUT, "sb")
    by = {}
    for j, u in enumerate(units):
        if j % n == i:
            by.setdefault((u["suite"], u["index"]), []).append(u)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    done, t_all = 0, time.time()
    for tk in sorted(by):
        todo = [u for u in by[tk] if not os.path.exists(os.path.join(od, u["unit"] + ".npz"))]
        if not todo:
            continue
        SB = D5.Sandbox(*tk)
        for u in todo:
            t0 = time.time()
            meta, arr = R7.run_unit(SB, u, client)
            D5.save_atomic(os.path.join(od, u["unit"] + ".npz"), meta=np.array(json.dumps(meta)), **arr)
            done += 1
            print(f"[r7pg sandbox] {u['unit']} ({u['grp']}): {len(meta['runs'])} runs, {time.time() - t0:.0f}s "
                  f"(server {meta['secs']['server']:.0f}s); {done} done, {time.time() - t_all:.0f}s", flush=True)
        SB.S.close()
    print(f"[r7pg sandbox] shard {shard} finished: {done} new", flush=True)


def load_c(unit):
    f = os.path.join(R7.OUT, "sb", unit + ".npz")
    if not os.path.exists(f):
        return None
    s = np.asarray(np.load(f, allow_pickle=True)["success"]).astype(bool)
    return int(s[:R7.K_MAX].sum()), bool(s[R7.K_MAX])


def read(record=False):
    import d4_read
    pg = R7.jload(PLAN_PG)
    res = dict(n_units=len(pg["units"]), missing_pregrasp_obs=pg["missing"], gap_steps=pg["gap_steps"], cells={}, paired={})
    lines = [f"carry - pre-grasp gap (steps): {pg['gap_steps']}"]
    unread = []
    for grp in ("wg", "succ"):
        st = [u for u in pg["units"] if u["grp"] == grp]
        rows = []
        for u in st:
            o = u["obs"][3:]
            c = {cond: load_c(f"{cond}_{o}") for cond in ("carry", "pregrasp", "start")}
            if any(v is None for v in c.values()):
                unread.append(o)
                continue
            rows.append((u["tid"], c))
        tids = sorted({t for t, _ in rows})
        for cond in ("carry", "pregrasp", "start"):
            cell = dict(n=len(rows), pooled={}, per_task={})
            for K in R7.KS:
                per = {t: [R7.cov_at(c[cond][0], R7.K_MAX, K) for tt, c in rows if tt == t] for t in tids}
                cell["pooled"][f"cov@{K}"] = d4_read.rate(per)
                for t in tids:
                    lo, hi = R7.boot_state(per[t])
                    cell["per_task"].setdefault(t, {})[f"cov@{K}"] = dict(mean=float(np.mean(per[t])), lo=lo, hi=hi, n=len(per[t]))
            perz = {t: [float(c[cond][1]) for tt, c in rows if tt == t] for t in tids}
            cell["pooled"]["zs_run_success"] = d4_read.rate(perz)
            for t in tids:
                cell["per_task"][t]["hist_c"] = sorted(c[cond][0] for tt, c in rows if tt == t)
                cell["per_task"][t]["zs_run_success"] = float(np.mean(perz[t]))
            res["cells"][f"{grp}|{cond}"] = cell
            p = cell["pooled"]
            lines.append(f"{grp} {cond} (n {len(rows)}): " + " ".join(
                f"cov@{K} {p[f'cov@{K}']['mean']:.3f} [{p[f'cov@{K}']['lo']:.2f},{p[f'cov@{K}']['hi']:.2f}]" for K in R7.KS)
                + f"; zs run {p['zs_run_success']['mean']:.3f}")
            for t in tids:
                q = cell["per_task"][t]
                lines.append(f"   {t} (n {q['cov@1']['n']}): " + " ".join(f"@{K} {q[f'cov@{K}']['mean']:.2f}" for K in R7.KS)
                             + f"; zs run {q['zs_run_success']:.2f}; successes/32 {q['hist_c']}")
        for a, b in (("pregrasp", "carry"), ("start", "pregrasp")):
            per = {t: [R7.cov_at(c[a][0], R7.K_MAX, K_PRIMARY) - R7.cov_at(c[b][0], R7.K_MAX, K_PRIMARY) for tt, c in rows if tt == t]
                   for t in tids}
            s = d4_read.boot2(per)
            xs = [v for vs in per.values() for v in vs]
            res["paired"][f"{grp}|{a}-{b}|cov@{K_PRIMARY}"] = dict(mean=float(np.mean(xs)), lo=float(np.percentile(s, 2.5)),
                                                                    hi=float(np.percentile(s, 97.5)), n=len(xs))
            r = res["paired"][f"{grp}|{a}-{b}|cov@{K_PRIMARY}"]
            lines.append(f"{grp} paired cov@{K_PRIMARY} {a} - {b}: {r['mean']:+.3f} [{r['lo']:+.2f},{r['hi']:+.2f}]")
    prim = res["cells"].get("wg|pregrasp", {}).get("pooled", {}).get(f"cov@{K_PRIMARY}")
    res["unread"] = unread
    res["primary"] = dict(stat=prim, bar=BAR, verdict=(None if prim is None else ("PASS" if prim["mean"] >= BAR else "FAIL")))
    lines.append(f"PRIMARY wg pre-grasp cov@{K_PRIMARY}: {prim and round(prim['mean'], 3)} vs bar {BAR} -> {res['primary']['verdict']}; "
                 f"unread {len(unread)}")
    print("\n".join("[r7pg read] " + x for x in lines), flush=True)
    R7.dump(os.path.join(R7.OUT, "read_pg.json"), res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r7_read", bench="libero_90_c", tag="pregrasp", seed=0,
                            metrics=dict(primary=res["primary"], cells=res["cells"], paired=res["paired"], unread=unread,
                                         missing_pregrasp_obs=pg["missing"], gap_steps=pg["gap_steps"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/r7_pregrasp.py", command="python experiments/r7_pregrasp.py --read --record"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("plan", "sandbox", "read", "record"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    if a.plan:
        plan()
    if a.sandbox:
        sandbox(a.shard, a.port)
    if a.read:
        read(a.record)


if __name__ == "__main__":
    main()
