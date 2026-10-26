"""PLAN_BIND E2b (docs/PLAN_E12.md section 5; pre-registration experiments/prereg/pbind_e2b.py, ledger m3_prereg tag pbind_e2b):
P re-measured with LIBERO's own predicates.  E2's tasks, states and seeds (bind_e2: 24 tasks x fresh_m1 states 0-4 x 8 seeds); D5's
sandbox and server, unchanged.  E2 saved only labels, so the runs are re-rolled.

Per run:
  bound object      fm_types' rule (first movable moved > 3 cm while the gripper is commanded closed), as E2
  first release     the first executed open command after an object is bound.  There the run is BRANCHED: a copy settles for
                    SETTLE no-op steps (zero delta, gripper open) and LIBERO's predicates In / On(bound object, c) are evaluated
                    for every scene candidate c (fm_types.Scene.alternatives) and the own targets; the copy is discarded
  continuation      the run itself continues (the policy's own closed loop, fresh noise at every query) to LIBERO success or the
                    400-step budget; a success before any release counts as a correct first release
  correct           bound object == named AND the own goal atom's predicate holds at the settled first release
P_obj = share binding the named object; P_goal = share correct among those; P_hat = P_obj x P_goal = share correct.
competence = success rate of the CORRECT runs' continuations (undefined -> 0 weight; S_hat = P_hat x competence = 0 then).

    --sandbox --shard i/n   (LIBERO env + d5_serve.py)  -> results/bind/e2b/<unit>.npz
    --read [--record]       (chunks env)  -> results/bind/e2b.json
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
import bind_e2 as E                                               # noqa: E402

SETTLE = 20
E2BD = os.environ.get("BIND_E2B_DIR") or os.path.join(E.OUT, "e2b")
MAX_STEPS = int(os.environ.get("BIND_MAX_STEPS", 400))           # smoke only


def preds(SB, obj, names):
    out = {}
    for c in names:
        v = []
        for p in ("in", "on"):
            try:
                v.append(bool(SB.e._eval_predicate([p, obj, c])))
            except Exception:                                    # noqa: BLE001
                pass
        out[c] = any(v)
    return out


def sandbox(shard, port):
    from openpi_client import websocket_client_policy as _wcp
    import d5_goal as D5
    import m1_compose_create as M1
    units = E.jload(os.path.join(E.OUT, "e2_plan.json"))["units"]
    i, n = map(int, shard.split("/"))
    by = {}
    for j, u in enumerate(units):
        if j % n == i:
            by.setdefault((u["suite"], u["index"]), []).append(u)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    for tk in sorted(by):
        todo = [u for u in by[tk] if not os.path.exists(os.path.join(E2BD, u["unit"] + ".npz"))]
        if not todo:
            continue
        SB = D5.Sandbox(*tk)
        fresh = np.load(os.path.join(M1.R, M1.FRESH_DIR, f"{tk[0]}_{tk[1]}.npy"))
        movable = list(SB.S.movable)
        for u in todo:
            t0 = time.time()
            SB.S.set(fresh[u["state"]])
            for _ in range(E.WAIT):
                SB.e.step(E.HOLD)
            st = SB.flat()
            runs = [E.Run(k, SB.start(st, [-1.0] * E.WAIT)) for k in range(E.K)]
            for r in runs:
                r.labels, r.rel_step = None, None
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
                for g0 in range(0, len(rows), E.BATCH):
                    grp = rows[g0:g0 + E.BATCH]
                    req = {"b/image": np.stack([x[0] for x in grp]), "b/wrist": np.stack([x[1] for x in grp]),
                           "b/state": np.stack([x[2] for x in grp]).astype(np.float32),
                           "b/seed": np.array([x[3] for x in grp], np.int64), "prompt": u["prompt"]}
                    ts = time.time()
                    chunks.extend(list(np.asarray(client.infer(req)["actions"])))
                    t_srv += time.time() - ts
                for r, ch in zip(act, chunks):
                    SB.restore(r.snap)
                    for a in ch[:E.REPLAN]:
                        a = np.asarray(a, np.float64)
                        if a[6] < 0 and r.bound is not None and not r.released:
                            r.released, r.rel_step = True, r.steps
                            keep = SB.snapshot()
                            for _ in range(SETTLE):
                                SB.e.step(E.HOLD)
                            r.labels = preds(SB, r.bound, u["sites"])
                            SB.restore(keep)
                        _, _, done, _ = SB.e.step(a.tolist())
                        r.steps += 1
                        if a[6] > 0 and r.bound is None:
                            for m in movable:
                                if np.linalg.norm(np.asarray(SB.S.pos(m), float) - p0[m]) > E.MOVE:
                                    r.bound = m
                                    break
                        if done:
                            r.success, r.done = True, True
                            break
                    r.q += 1
                    if not r.done and r.steps >= MAX_STEPS:
                        r.done = True
                    r.snap = SB.snapshot(r.snap)
            meta = dict(unit=u["unit"], bound=[r.bound or "" for r in runs], released=[r.released for r in runs],
                        rel_step=[r.rel_step for r in runs], labels=[r.labels for r in runs], success=[r.success for r in runs],
                        steps=[r.steps for r in runs], secs=time.time() - t0, server=t_srv)
            os.makedirs(E2BD, exist_ok=True)
            D5.save_atomic(os.path.join(E2BD, u["unit"] + ".npz"), meta=np.array(json.dumps(meta)))
            print(f"[bind e2b sandbox] {u['unit']}: {E.K} runs, {meta['secs']:.0f}s (server {t_srv:.0f}s)", flush=True)
        SB.S.close()
    print(f"[bind e2b sandbox] shard {shard} finished", flush=True)


def boot_rank(x, y, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    rs, sl = [], []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        rs.append(E.spearman(x[i], y[i]))
        sl.append(np.polyfit(x[i], y[i], 1)[0] if np.std(x[i]) > 0 else np.nan)
    return (float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5)),
            float(np.nanpercentile(sl, 2.5)), float(np.nanpercentile(sl, 97.5)))


def read(record=False):
    P = E.jload(os.path.join(E.OUT, "e2_plan.json"))
    per = {}
    for u in P["units"]:
        f = os.path.join(E2BD, u["unit"] + ".npz")
        if not os.path.exists(f):
            continue
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        d = per.setdefault(u["tid"], dict(n=0, obj=0, corr=0, succ=0, corr_succ=0, wrong_succ=0, rel=0, first=dict()))
        for b, lab, rel, ok in zip(m["bound"], m["labels"], m["released"], m["success"]):
            d["n"] += 1
            d["succ"] += int(ok)
            if b != u["named"]:
                d["wrong_succ"] += int(ok)
                continue
            d["obj"] += 1
            if rel and lab is not None:
                d["rel"] += 1
                c = any(lab.get(o, False) for o in u["own"])
                hold = [k for k, v in lab.items() if v and k not in u["own"]]
                key = "own" if c else (hold[0] if hold else "none")
            else:
                c = bool(ok)
                key = "success_before_release" if ok else "no_release"
            d["first"][key] = d["first"].get(key, 0) + 1
            d["corr"] += int(c)
            d["corr_succ"] += int(c and ok)
    T = sorted(per)
    zs = {}
    for t in T:
        eps = E.jload(os.path.join(ROOT, "results", "fm", "types", t.replace(":", "_") + ".json"))["arms"]["zs"]["episodes"]
        zs[t] = float(np.mean([e["success"] for e in eps]))
    ph = np.array([per[t]["corr"] / per[t]["n"] for t in T])
    comp = np.array([per[t]["corr_succ"] / per[t]["corr"] if per[t]["corr"] else 0.0 for t in T])
    sh = ph * comp
    sb = np.array([per[t]["succ"] / per[t]["n"] for t in T])
    y = np.array([zs[t] for t in T])
    lo, hi, slo, shi = boot_rank(ph, y)
    lo2, hi2, slo2, shi2 = boot_rank(sh, y)
    res = dict(n_tasks=len(T),
               per_task={t: dict(P_obj=per[t]["obj"] / per[t]["n"], P_goal=(per[t]["corr"] / per[t]["obj"] if per[t]["obj"] else None),
                                 P_hat=per[t]["corr"] / per[t]["n"], competence=(per[t]["corr_succ"] / per[t]["corr"] if per[t]["corr"]
                                                                                 else None),
                                 S_hat=float(ph[i] * comp[i]), sandbox_success=per[t]["succ"] / per[t]["n"], zs_success=zs[t],
                                 first_release=per[t]["first"], n=per[t]["n"]) for i, t in enumerate(T)},
               primary=dict(rho=E.spearman(ph, y), lo=lo, hi=hi), P_hat_slope=dict(slope=float(np.polyfit(ph, y, 1)[0]), lo=slo, hi=shi),
               S_hat=dict(rho=E.spearman(sh, y), lo=lo2, hi=hi2, slope=float(np.polyfit(sh, y, 1)[0]) if np.std(sh) > 0 else None,
                          slope_lo=slo2, slope_hi=shi2),
               sandbox_vs_zs=dict(rho=E.spearman(sb, y), mae=float(np.mean(np.abs(sb - y)))))
    pr = res["primary"]
    res["verdict"] = "PASS (not falsified)" if pr["rho"] > 0 and pr["lo"] > 0 else "FAIL (falsified)"
    s = res["S_hat"]
    res["S_hat_calibration"] = ("FAIL" if s["slope"] is None or s["slope_hi"] < 0.5 or s["slope_lo"] > 2.0 else "PASS")
    for t in T:
        q = res["per_task"][t]
        print(f"[bind e2b] {t}: P_obj {q['P_obj']:.2f} P_goal {q['P_goal'] if q['P_goal'] is None else round(q['P_goal'], 2)} "
              f"P_hat {q['P_hat']:.3f} competence {q['competence'] if q['competence'] is None else round(q['competence'], 2)} "
              f"S_hat {q['S_hat']:.3f} | sandbox success {q['sandbox_success']:.2f} zs {q['zs_success']:.2f} | first release {q['first_release']}",
              flush=True)
    print(f"[bind e2b] PRIMARY Spearman(P_hat, zs) {pr['rho']:+.3f} [{pr['lo']:+.3f}, {pr['hi']:+.3f}] -> {res['verdict']}; P_hat slope "
          f"{res['P_hat_slope']['slope']:.3f} [{slo:.3f}, {shi:.3f}]; S_hat rho {s['rho']:+.3f} [{lo2:+.3f}, {hi2:+.3f}], slope {s['slope']} "
          f"[{slo2:.3f}, {shi2:.3f}] -> calibration {res['S_hat_calibration']}; sandbox success vs zs rho {res['sandbox_vs_zs']['rho']:+.3f}, "
          f"MAE {res['sandbox_vs_zs']['mae']:.3f}", flush=True)
    E.dump(os.path.join(E.OUT, "e2b.json"), res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("bind_e2", bench="libero_90_c", tag="e2b", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/bind_e2b.py", command="python experiments/bind_e2b.py --read --record"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("sandbox", "read", "record"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    if a.sandbox:
        sandbox(a.shard, a.port)
    if a.read:
        read(a.record)


if __name__ == "__main__":
    main()
