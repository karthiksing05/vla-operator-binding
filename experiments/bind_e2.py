"""PLAN_BIND E2 (docs/PLAN_E12.md section 2; pre-registration experiments/prereg/pbind_e2.py, ledger m3_prereg tag pbind_e2):
measuring P, the policy's mass on the correct binding, from its own early samples, and testing whether it predicts per-task zs success.

Tasks: the 23 dev tasks with an In / On goal (results/bind/e1_scene.json).  States: fresh_m1 init states k = 0..N_STATES-1 (the
first 5 of the 20 states zs was rolled from), each followed by the client's 10 wait steps (zero delta, gripper open).  K seeds per
state (SEED_BASE + 1e5 k + q), D5's sandbox and server (experiments/d5_goal.Sandbox, d5_serve.py; unchanged).  A run stops at its
RELEASE (the first executed open command after an object is bound), at LIBERO success, or at the 400-step budget.
Binding classification of a run (intent, not outcome):
  bound object  = the first movable object moved > 3 cm from its start while the gripper is commanded closed (fm_types' rule)
  release site  = among the scene's goal candidates for the named object (fm_types.Scene.alternatives) and its own targets, the site
                  nearest (xy) to the bound object's position at the release (success before a release -> the own target)
  correct       = bound object == named object AND release site == an own target
P_hat(task) = the share of correct runs (no grasp within the budget counts as not correct).  P_obj = the share binding the named
object.  zs success per task = fresh_m1's 20 zs episodes (results/fm/types, arm zs).

    --plan                    (chunks env)  -> results/bind/e2_plan.json
    --sandbox --shard i/n     (LIBERO env + d5_serve.py)  -> results/bind/e2/<task>_s<k>.npz
    --read [--record]         (chunks env)  -> results/bind/e2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")

OUT = os.path.join(ROOT, "results", "bind")
N_STATES, K = 5, 8
SEED_BASE = 51_000_000
MAX_STEPS, WAIT, REPLAN, BATCH = int(os.environ.get("BIND_MAX_STEPS", 400)), 10, 5, 48     # BIND_MAX_STEPS: smoke only
E2D = os.environ.get("BIND_E2_DIR") or os.path.join(OUT, "e2")                                        # BIND_E2_DIR: smoke only
MOVE = 0.03
HOLD = [0.0] * 6 + [-1.0]


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def plan():
    import m1_compose_create as M1
    tt = M1.task_table()
    scn = jload(os.path.join(OUT, "e1_scene.json"))
    units = []
    for tid, v in scn.items():
        suite, i = tid.split(":")
        for k in range(N_STATES):
            units.append(dict(unit=f"{tid.replace(':', '_')}_s{k}", tid=tid, suite=suite, index=int(i), state=k, prompt=tt[tid].language,
                              named=v["named"], own=v["own"], sites=sorted(set(v["alternatives"]) | set(v["own"]))))
    dump(os.path.join(OUT, "e2_plan.json"), dict(units=units, n_tasks=len(scn), N_STATES=N_STATES, K=K))
    print(f"[bind e2 plan] {len(scn)} tasks, {len(units)} units x {K} seeds", flush=True)


class Run:
    def __init__(self, k, snap):
        self.k, self.seed0, self.snap = k, SEED_BASE + 100_000 * k, snap
        self.q, self.steps, self.done = 0, 0, False
        self.bound, self.rel_pos, self.success, self.released = None, None, False, False
        self.start_pos = None


def sandbox(shard, port):
    from openpi_client import websocket_client_policy as _wcp
    import d5_goal as D5
    units = jload(os.path.join(OUT, "e2_plan.json"))["units"]
    i, n = map(int, shard.split("/"))
    by = {}
    for j, u in enumerate(units):
        if j % n == i:
            by.setdefault((u["suite"], u["index"]), []).append(u)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    import m1_compose_create as M1
    for tk in sorted(by):
        todo = [u for u in by[tk] if not os.path.exists(os.path.join(E2D, u["unit"] + ".npz"))]
        if not todo:
            continue
        SB = D5.Sandbox(*tk)
        fresh = np.load(os.path.join(M1.R, M1.FRESH_DIR, f"{tk[0]}_{tk[1]}.npy"))
        movable = list(SB.S.movable)
        for u in todo:
            t0 = time.time()
            SB.S.set(fresh[u["state"]])
            for _ in range(WAIT):
                SB.e.step(HOLD)
            st = SB.flat()
            runs = [Run(k, SB.start(st, [-1.0] * WAIT)) for k in range(K)]
            p0 = {m: np.asarray(SB.S.pos(m), float) for m in movable}
            t_srv = 0.0
            while True:
                act = [r for r in runs if not r.done]
                if not act:
                    break
                rows = []
                for r in act:
                    SB.restore(r.snap)
                    st8, _ = SB.obs8()
                    im, wr = SB.S.render()
                    rows.append((im, wr, st8, r.seed0 + r.q))
                chunks = []
                for g0 in range(0, len(rows), BATCH):
                    grp = rows[g0:g0 + BATCH]
                    req = {"b/image": np.stack([x[0] for x in grp]), "b/wrist": np.stack([x[1] for x in grp]),
                           "b/state": np.stack([x[2] for x in grp]).astype(np.float32),
                           "b/seed": np.array([x[3] for x in grp], np.int64), "prompt": u["prompt"]}
                    ts = time.time()
                    chunks.extend(list(np.asarray(client.infer(req)["actions"])))
                    t_srv += time.time() - ts
                for r, ch in zip(act, chunks):
                    SB.restore(r.snap)
                    for a in ch[:REPLAN]:
                        a = np.asarray(a, np.float64)
                        if a[6] < 0 and r.bound is not None:
                            r.released, r.done = True, True
                            r.rel_pos = np.asarray(SB.S.pos(r.bound), float)
                            break
                        _, _, done, _ = SB.e.step(a.tolist())
                        r.steps += 1
                        if a[6] > 0 and r.bound is None:
                            for m in movable:
                                if np.linalg.norm(np.asarray(SB.S.pos(m), float) - p0[m]) > MOVE:
                                    r.bound = m
                                    break
                        if done:
                            r.success, r.done = True, True
                            if r.bound is not None:
                                r.rel_pos = np.asarray(SB.S.pos(r.bound), float)
                            break
                    r.q += 1
                    if not r.done and r.steps >= MAX_STEPS:
                        r.done = True
                    r.snap = SB.snapshot(r.snap)
            SB.S.set(st)
            sites = {s: np.asarray(SB.e.object_states_dict[s].get_geom_state()["pos"], float) for s in u["sites"]}
            rel_site = []
            for r in runs:
                if r.rel_pos is None:
                    rel_site.append("")
                elif r.success and not r.released:
                    rel_site.append(u["own"][0])
                else:
                    rel_site.append(min(sites, key=lambda s: float(np.linalg.norm(sites[s][:2] - r.rel_pos[:2]))))
            meta = dict(unit=u["unit"], bound=[r.bound or "" for r in runs], rel_site=rel_site, success=[r.success for r in runs],
                        released=[r.released for r in runs], steps=[r.steps for r in runs], secs=time.time() - t0, server=t_srv)
            os.makedirs(E2D, exist_ok=True)
            D5.save_atomic(os.path.join(E2D, u["unit"] + ".npz"), meta=np.array(json.dumps(meta)))
            print(f"[bind e2 sandbox] {u['unit']}: {K} runs, {meta['secs']:.0f}s (server {t_srv:.0f}s)", flush=True)
        SB.S.close()
    print(f"[bind e2 sandbox] shard {shard} finished", flush=True)


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def read(record=False):
    P = jload(os.path.join(OUT, "e2_plan.json"))
    tasks = sorted({u["tid"] for u in P["units"]})
    per = {}
    for u in P["units"]:
        f = os.path.join(E2D, u["unit"] + ".npz")
        if not os.path.exists(f):
            continue
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        d = per.setdefault(u["tid"], dict(c=0, obj=0, n=0, succ=0, grasp=0))
        for b, s, ok in zip(m["bound"], m["rel_site"], m["success"]):
            d["n"] += 1
            d["obj"] += int(b == u["named"])
            d["grasp"] += int(b != "")
            d["c"] += int(b == u["named"] and s in u["own"])
            d["succ"] += int(ok)
    zs = {}
    for t in tasks:
        eps = jload(os.path.join(ROOT, "results", "fm", "types", t.replace(":", "_") + ".json"))["arms"]["zs"]["episodes"]
        zs[t] = float(np.mean([e["success"] for e in eps]))
    T = [t for t in tasks if t in per and per[t]["n"]]
    ph = np.array([per[t]["c"] / per[t]["n"] for t in T])
    po = np.array([per[t]["obj"] / per[t]["n"] for t in T])
    y = np.array([zs[t] for t in T])
    rng = np.random.default_rng(0)
    B = 10_000
    rho, slope = [], []
    for _ in range(B):
        idx = rng.integers(0, len(T), len(T))
        rho.append(spearman(ph[idx], y[idx]))
        xs = ph[idx]
        slope.append(np.polyfit(xs, y[idx], 1)[0] if np.std(xs) > 0 else np.nan)
    rho, slope = np.array(rho), np.array(slope)
    r0 = spearman(ph, y)
    s0 = float(np.polyfit(ph, y, 1)[0])
    i0 = float(np.polyfit(ph, y, 1)[1])
    res = dict(n_tasks=len(T), per_task={t: dict(P_hat=per[t]["c"] / per[t]["n"], P_obj=per[t]["obj"] / per[t]["n"],
                                                  grasp=per[t]["grasp"] / per[t]["n"], early_success=per[t]["succ"] / per[t]["n"],
                                                  n=per[t]["n"], zs_success=zs[t]) for t in T},
               spearman=dict(rho=r0, lo=float(np.nanpercentile(rho, 2.5)), hi=float(np.nanpercentile(rho, 97.5)),
                             p_le0=float(np.nanmean(rho <= 0))),
               calibration=dict(slope=s0, intercept=i0, lo=float(np.nanpercentile(slope, 2.5)), hi=float(np.nanpercentile(slope, 97.5))),
               spearman_P_obj=spearman(po, y), mean_abs_err=float(np.mean(np.abs(ph - y))))
    sp, ca = res["spearman"], res["calibration"]
    res["verdict_rank"] = "PASS" if sp["rho"] > 0 and sp["lo"] > 0 else "FAIL"
    res["verdict_calibration"] = "FAIL" if (ca["hi"] < 0.5 or ca["lo"] > 2.0) else "PASS"
    res["verdict"] = "PASS (not falsified)" if res["verdict_rank"] == "PASS" and res["verdict_calibration"] == "PASS" else "FAIL (falsified)"
    for t in T:
        q = res["per_task"][t]
        print(f"[bind e2] {t}: P_hat {q['P_hat']:.3f} P_obj {q['P_obj']:.3f} grasp {q['grasp']:.2f} (n {q['n']}) | zs success "
              f"{q['zs_success']:.2f}", flush=True)
    print(f"[bind e2] Spearman(P_hat, zs) {sp['rho']:+.3f} [{sp['lo']:+.3f}, {sp['hi']:+.3f}]; calibration slope {ca['slope']:.3f} "
          f"[{ca['lo']:.3f}, {ca['hi']:.3f}] intercept {ca['intercept']:.3f}; Spearman(P_obj, zs) {res['spearman_P_obj']:+.3f}; "
          f"MAE {res['mean_abs_err']:.3f}; rank {res['verdict_rank']}, calibration {res['verdict_calibration']} -> {res['verdict']}",
          flush=True)
    dump(os.path.join(OUT, "e2.json"), res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("bind_e2", bench="libero_90_c", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/bind_e2.py", command="python experiments/bind_e2.py --read --record"))


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
