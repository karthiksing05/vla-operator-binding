"""PROBE R7 stage A (docs/PLAN_R7.md; pre-registration experiments/prereg/pr7_cover.py, ledger m3_prereg tag pr7_cover): does the
frozen pi05_l90c@5000's OWN sample distribution contain the correct behaviour on the recombination wrong-goal states?

coverage@K = the share of states where at least one of K independent policy-noise rollouts (sandboxed, from a recorded state, no
guidance, no weight change) reaches LIBERO task success (the named object in the NAMED region) within the episode's remaining budget.
Demos are never executed; the sandbox is D5's (experiments/d5_goal.py Sandbox, unchanged).

Stages (each its own SLURM job; nothing here runs on the login node):
  --plan                  (chunks env)   units (state x condition) from results/d5/index.json -> results/r7/plan.json
  --sandbox --shard i/n   (LIBERO env + experiments/d5_serve.py)  -> results/r7/sb/<unit>.npz
  --read [--record]       (chunks env)   coverage@K, per-seed success, zs-box reproduction -> results/r7/read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")

OUT = os.environ.get("R7_OUT") or os.path.join(ROOT, "results", "r7")
D4D = os.path.join(ROOT, "results", "d4")
D5D = os.path.join(ROOT, "results", "d5")
# ── settings (fixed before any R7 rollout; docs/PLAN_R7.md section 2) ────────────────────────────────────────────────
K_MAX = 32                        # independent noise seeds per state
KS = (1, 4, 8, 16, 32)            # coverage@K read from the 32 by the exact without-replacement estimator
K_PRIMARY = 16
SEED_BASE = 41_000_000            # seed k, query q: SEED_BASE + 100_000 k + q (shared across states)
MAX_STEPS, WAIT, REPLAN = 400, 10, 5   # libero_90 episode budget (libero_client MAX_STEPS) after the 10 wait steps
BATCH = 48
BAR = 0.5                         # hypothesis: pooled coverage@16 on carry wrong-goal states >= 0.5


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def slug(t):
    return t.replace(":", "_")


# ── plan (chunks env) ────────────────────────────────────────────────────────────────────────────────────────────────
def plan():
    """Units: every D5 dev observation on a recombination task (bins 0.1 / 0.3 / 0.5) that is a zs wrong-goal episode (PRIMARY) or one
    of D5's zs-success observations (control), each under two conditions: `carry` (D5's recorded post-grasp state, step j) and
    `start` (the same episode's recorded first policy state, step 0)."""
    ix = jload(os.path.join(D5D, "index.json"))
    units = []
    for o in ix["dev"]:
        if o["bin"] == "part":
            continue
        grp = "wg" if o["type"] == "wrong_goal" else ("succ" if o["success"] else None)
        if grp is None:
            continue
        j = int(o["j"])
        seed_start = int(o["zs_seed0"]) - j // REPLAN
        for cond in ("carry", "start"):
            jj = j if cond == "carry" else 0
            units.append(dict(unit=f"{cond}_{o['key'][3:]}", cond=cond, grp=grp, obs=o["key"], tid=o["tid"], suite=o["suite"],
                              index=o["index"], trial=o["trial"], j=jj, prompt=o["prompt"], held=o["held"], sub=o.get("sub"),
                              grip_cmds=[-1.0] * WAIT + o["grip_cmds"][WAIT:WAIT + jj],
                              q_budget=(MAX_STEPS - jj) // REPLAN, zs_seed0=seed_start + jj // REPLAN))
    by = {}
    for u in units:
        k = (u["tid"], u["grp"], u["cond"])
        by[k] = by.get(k, 0) + 1
    dump(os.path.join(OUT, "plan.json"), dict(units=units, counts={"|".join(k): v for k, v in sorted(by.items())}))
    print("[r7 plan] " + "; ".join(f"{k} {v}" for k, v in sorted(by.items())), flush=True)


# ── sandbox (LIBERO env) ─────────────────────────────────────────────────────────────────────────────────────────────
class Run:
    __slots__ = ("label", "seed0", "snap", "q", "done", "success", "steps", "rel_hand", "rel_step", "rel_obj", "hand_q", "chunks")

    def __init__(self, label, seed0, snap):
        self.label, self.seed0, self.snap = label, seed0, snap
        self.q, self.done, self.success, self.steps = 0, False, False, 0
        self.rel_hand, self.rel_step, self.rel_obj = None, None, None
        self.hand_q, self.chunks = [], []


def roll(SB, runs, held, prompt, client, q_budget):
    """Lockstep: render + query every live run, execute REPLAN actions; a run ends at LIBERO success or after q_budget queries.  The
    first executed open command (the release) is recorded but does not end the run."""
    t_srv = 0.0
    while True:
        act = [r for r in runs if not r.done]
        if not act:
            break
        rows = []
        for r in act:
            SB.restore(r.snap)
            st8, hand = SB.obs8()
            r.hand_q.append(hand.copy())
            im, wr = SB.S.render()
            rows.append((im, wr, st8, r.seed0 + r.q))
        chunks = []
        for g0 in range(0, len(rows), BATCH):
            grp = rows[g0:g0 + BATCH]
            req = {"b/image": np.stack([x[0] for x in grp]), "b/wrist": np.stack([x[1] for x in grp]),
                   "b/state": np.stack([x[2] for x in grp]).astype(np.float32), "b/seed": np.array([x[3] for x in grp], np.int64),
                   "prompt": prompt}
            t0 = time.time()
            out = client.infer(req)
            t_srv += time.time() - t0
            chunks.extend(list(np.asarray(out["actions"])))
        for r, ch in zip(act, chunks):
            SB.restore(r.snap)
            r.chunks.append(np.asarray(ch, np.float16))
            for a in ch[:REPLAN]:
                a = np.asarray(a, np.float64)
                if a[6] < 0 and r.rel_step is None:
                    _, hand = SB.obs8()
                    r.rel_hand, r.rel_step = hand, r.steps
                    hp = SB.held_pos(held)
                    r.rel_obj = None if hp is None else np.asarray(hp, np.float64)
                _, _, done, _ = SB.e.step(a.tolist())
                r.steps += 1
                if done:
                    r.success, r.done = True, True
                    break
            r.q += 1
            if not r.done and r.q >= q_budget:
                r.done = True
            r.snap = SB.snapshot(r.snap)
    return t_srv


def unit_state(u):
    rz = np.load(os.path.join(D4D, "roll", f"d4_zs_{slug(u['tid'])}_rec", f"task{u['index']}_trial{u['trial']}.npz"), allow_pickle=True)
    return rz["sim"][u["j"]]


def run_unit(SB, u, client):
    state = unit_state(u)
    runs = [Run(f"k{k}", SEED_BASE + 100_000 * k, SB.start(state, u["grip_cmds"])) for k in range(K_MAX)]
    runs.append(Run("zs", int(u["zs_seed0"]), SB.start(state, u["grip_cmds"])))
    SB.S.set(state)
    regions = {}
    for rn, ost in SB.e.object_states_dict.items():
        try:
            regions[rn] = [float(v) for v in ost.get_geom_state()["pos"]]
        except Exception:                                       # noqa: BLE001
            pass
    t0 = time.time()
    t_srv = roll(SB, runs, u["held"], u["prompt"], client, int(u["q_budget"]))
    final_obj = []
    for r in runs:
        SB.restore(r.snap)
        hp = SB.held_pos(u["held"])
        final_obj.append(np.full(3, np.nan) if hp is None else np.asarray(hp, np.float64))
    qmax = max(len(r.chunks) for r in runs)
    H = runs[0].chunks[0].shape
    CH = np.full((len(runs), qmax) + H, np.nan, np.float16)
    HQ = np.full((len(runs), qmax, 3), np.nan, np.float32)
    for i, r in enumerate(runs):
        CH[i, :len(r.chunks)] = np.stack(r.chunks)
        HQ[i, :len(r.hand_q)] = np.stack(r.hand_q)
    meta = dict(unit=u["unit"], labels=[r.label for r in runs], regions=regions,
                runs=[dict(label=r.label, success=r.success, steps=r.steps, queries=r.q, rel_step=r.rel_step) for r in runs],
                secs=dict(run=time.time() - t0, server=t_srv))
    nan3 = np.full(3, np.nan)
    arr = dict(success=np.array([r.success for r in runs]), steps=np.array([r.steps for r in runs]),
               rel_hand=np.stack([r.rel_hand if r.rel_hand is not None else nan3 for r in runs]),
               rel_obj=np.stack([r.rel_obj if r.rel_obj is not None else nan3 for r in runs]),
               final_obj=np.stack(final_obj), chunks=CH, hand_q=HQ)
    return meta, arr


def sandbox(shard, port, smoke=False):
    from openpi_client import websocket_client_policy as _wcp
    import d5_goal as D5
    units = jload(os.path.join(OUT, "plan.json"))["units"]
    i, n = map(int, shard.split("/"))
    od = os.path.join(OUT, "sb" if not smoke else "sb_smoke")
    by = {}
    for j, u in enumerate(units):
        if j % n == i:
            by.setdefault((u["suite"], u["index"]), []).append(u)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    done, t_all = 0, time.time()
    for tk in (sorted(by) if not smoke else sorted(by)[:1]):
        todo = [u for u in by[tk] if not os.path.exists(os.path.join(od, u["unit"] + ".npz"))]
        if smoke:
            todo = [dict(u, q_budget=3) for u in todo[:1]]
        if not todo:
            continue
        SB = D5.Sandbox(*tk)
        for u in todo:
            t0 = time.time()
            meta, arr = run_unit(SB, u, client)
            D5.save_atomic(os.path.join(od, u["unit"] + ".npz"), meta=np.array(json.dumps(meta)), **arr)
            done += 1
            print(f"[r7 sandbox] {u['unit']} ({u['grp']}): {len(meta['runs'])} runs, {time.time() - t0:.0f}s "
                  f"(server {meta['secs']['server']:.0f}s); {done} done, {time.time() - t_all:.0f}s", flush=True)
        SB.S.close()
    print(f"[r7 sandbox] shard {shard} finished: {done} new", flush=True)


# ── read (chunks env) ────────────────────────────────────────────────────────────────────────────────────────────────
def cov_at(c, n, K):
    """P(at least one success among K drawn without replacement from n runs with c successes)."""
    if K > n:
        return None
    return 1.0 - comb(n - c, K) / comb(n, K)


def boot_state(xs, B=10_000, seed=0):
    xs = np.asarray(xs, float)
    rng = np.random.default_rng(seed)
    s = xs[rng.integers(0, len(xs), (B, len(xs)))].mean(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def read(record=False):
    import d4_read
    units = {u["unit"]: u for u in jload(os.path.join(OUT, "plan.json"))["units"]}
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "sb", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        u = units[meta["unit"]]
        succ = np.asarray(z["success"]).astype(bool)
        rows[meta["unit"]] = dict(u=u, c=int(succ[:K_MAX].sum()), zs=bool(succ[K_MAX]), secs=meta["secs"])
    missing = sorted(set(units) - set(rows))
    res = dict(n_units=len(units), n_read=len(rows), missing=missing, cells={})
    lines = []
    for grp in ("wg", "succ"):
        for cond in ("carry", "start"):
            sel = [r for r in rows.values() if r["u"]["grp"] == grp and r["u"]["cond"] == cond]
            if not sel:
                continue
            cell = dict(n=len(sel), per_task={}, pooled={})
            tids = sorted({r["u"]["tid"] for r in sel})
            for K in KS:
                per = {t: [cov_at(r["c"], K_MAX, K) for r in sel if r["u"]["tid"] == t] for t in tids}
                pr = d4_read.rate(per)
                cell["pooled"][f"cov@{K}"] = pr
                for t in tids:
                    lo, hi = boot_state(per[t])
                    cell["per_task"].setdefault(t, {})[f"cov@{K}"] = dict(mean=float(np.mean(per[t])), lo=lo, hi=hi, n=len(per[t]))
            per1 = {t: [r["c"] / K_MAX for r in sel if r["u"]["tid"] == t] for t in tids}
            cell["pooled"]["per_seed"] = d4_read.rate(per1)
            perz = {t: [float(r["zs"]) for r in sel if r["u"]["tid"] == t] for t in tids}
            cell["pooled"]["zs_box_success"] = d4_read.rate(perz)
            for t in tids:
                cell["per_task"][t]["per_seed"] = float(np.mean(per1[t]))
                cell["per_task"][t]["zs_box_success"] = float(np.mean(perz[t]))
                cell["per_task"][t]["hist_c"] = sorted(r["c"] for r in sel if r["u"]["tid"] == t)
            res["cells"][f"{grp}|{cond}"] = cell
            p = cell["pooled"]
            lines.append(f"{grp} {cond} (n {len(sel)}): " + " ".join(f"cov@{K} {p[f'cov@{K}']['mean']:.3f} "
                                                                    f"[{p[f'cov@{K}']['lo']:.2f},{p[f'cov@{K}']['hi']:.2f}]" for K in KS)
                         + f"; per-seed {p['per_seed']['mean']:.3f}; zs box {p['zs_box_success']['mean']:.3f}")
            for t in tids:
                q = cell["per_task"][t]
                lines.append(f"   {t} (n {q['cov@1']['n']}): " + " ".join(f"@{K} {q[f'cov@{K}']['mean']:.2f}" for K in KS)
                             + f"; per-seed {q['per_seed']:.3f}; zs box {q['zs_box_success']:.2f}; successes/32 {q['hist_c']}")
    prim = res["cells"].get("wg|carry", {}).get("pooled", {}).get(f"cov@{K_PRIMARY}")
    res["primary"] = dict(stat=prim, bar=BAR, verdict=(None if prim is None else ("PASS" if prim["mean"] >= BAR else "FAIL")))
    res["secs"] = dict(run=sum(r["secs"]["run"] for r in rows.values()), server=sum(r["secs"]["server"] for r in rows.values()))
    lines.append(f"PRIMARY wg carry cov@{K_PRIMARY}: {prim and round(prim['mean'], 3)} vs bar {BAR} -> {res['primary']['verdict']}; "
                 f"missing {len(missing)}")
    print("\n".join("[r7 read] " + x for x in lines), flush=True)
    dump(os.path.join(OUT, "read.json"), res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r7_read", bench="libero_90_c", tag="cover", seed=0,
                            metrics=dict(primary=res["primary"], cells=res["cells"], n_read=len(rows), missing=missing),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/r7_cover.py", command="python experiments/r7_cover.py --read --record"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("plan", "sandbox", "smoke", "read", "record"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    if a.plan:
        plan()
    if a.sandbox:
        sandbox(a.shard, a.port, smoke=a.smoke)
    if a.read:
        read(a.record)


if __name__ == "__main__":
    main()
