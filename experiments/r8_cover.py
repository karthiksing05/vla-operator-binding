"""PROBE R8 stage A (docs/PLAN_R8.md section 6; pre-registration experiments/prereg/pr8_cover.py, ledger m3_prereg tag pr8_cover):
does a test-time re-conditioning of the REGION word raise the frozen pi05_l90c's own success from the pre-grasp states?

Units: R7's pre-grasp units (results/r7/plan_pg.json) whose task has an r8_ops.spec (a goal region with a training phrase found
verbatim in the instruction): the wrong-goal region-error states of libero_90:29 and :40 (PRIMARY) and the zs-success controls of
any task with a spec.  Every other unit gets no operator (identical to R7 by construction) and is not re-rolled.
Rollouts: R7's run loop (r7_cover.run_unit) with K = 16 (R7's seeds k = 0..15, paired) + the zs-seed run, through a client proxy that
adds the arm's r8/* fields to every request (experiments/r8_serve.py).

  --plan                       (chunks env)  -> results/r8/plan.json
  --sandbox --arm A --shard    (LIBERO env + r8_serve.py)  -> results/r8/sb/<arm>/<unit>.npz
  --read [--record]            (chunks env)  -> results/r8/read.json
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
import r7_cover as R7                                             # noqa: E402
import r8_ops                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "r8")
K = 16
ARMS = {"c3": dict(arm="c3"), "c2w1": dict(arm="c2", w=1.0), "c2w2": dict(arm="c2", w=2.0)}
METHOD_ARMS = ("c3", "c2w1", "c2w2")
GAIN_BAR, HARM_BAR = 0.10, 0.05
ACT_BOUND = 1.0                   # on-manifold guard: share of executed actions with any |a[0:6]| > 1 (descriptive; flag at +0.10)


def proto_key(sp):
    return sp["phrase"].replace(" ", "_") + "|" + ",".join(sp["sources"])


def plan():
    import m1_compose_create as M1
    tt = M1.task_table()
    units, protos, skipped = [], {}, {}
    for u in R7.jload(os.path.join(R7.OUT, "plan_pg.json"))["units"]:
        t = tt[u["tid"]]
        sp = r8_ops.spec(t.language, t.goal, dict(t.fixtures))
        if sp is None:
            skipped[u["tid"]] = skipped.get(u["tid"], 0) + 1
            continue
        pk = proto_key(sp)
        protos[pk] = dict(phrase=sp["phrase"], sources=sp["sources"])
        units.append(dict(u, spec=sp, proto=pk))
    by = {}
    for u in units:
        k = f"{u['tid']}|{u['grp']}"
        by[k] = by.get(k, 0) + 1
    specs = {u["tid"]: u["spec"] for u in units}
    R7.dump(os.path.join(OUT, "plan.json"), dict(units=units, protos=protos, counts=by, skipped_no_spec=skipped, specs=specs, K=K,
                                                 arms=ARMS))
    print(f"[r8 plan] {len(units)} units {by}; skipped (no spec, identical to R7) {skipped}", flush=True)
    for tid, sp in specs.items():
        print(f"[r8 plan] {tid}: {json.dumps(sp)}", flush=True)


class Proxy:
    def __init__(self, client, extra):
        self.c, self.extra = client, extra

    def infer(self, req):
        return self.c.infer(dict(req, **self.extra))


def sandbox(arm, shard, port, smoke=False):
    from openpi_client import websocket_client_policy as _wcp
    import d5_goal as D5
    R7.K_MAX = K                                                  # R7's seeds k = 0..15 + the zs-seed run
    P = R7.jload(os.path.join(OUT, "plan.json"))
    units = P["units"]
    i, n = map(int, shard.split("/"))
    od = os.path.join(OUT, "sb" if not smoke else "sb_smoke", arm)
    by = {}
    for j, u in enumerate(units):
        if j % n == i:
            by.setdefault((u["suite"], u["index"]), []).append(u)
    base = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    done, t_all = 0, time.time()
    for tk in (sorted(by) if not smoke else sorted(by)[:1]):
        todo = [u for u in by[tk] if not os.path.exists(os.path.join(od, u["unit"] + ".npz"))]
        if smoke:
            todo = [dict(u, q_budget=2) for u in todo[:1]]
        if not todo:
            continue
        SB = D5.Sandbox(*tk)
        for u in todo:
            sp, a = u["spec"], ARMS[arm]
            extra = {"r8/arm": a["arm"], "r8/phrase": sp["phrase"]}
            if a["arm"] == "c3":
                extra["r8/proto"] = u["proto"]
            else:
                extra["r8/alts"] = "|".join(sp["alt_prompts"])
                extra["r8/w"] = float(a["w"])
            t0 = time.time()
            meta, arr = R7.run_unit(SB, u, Proxy(base, extra))
            D5.save_atomic(os.path.join(od, u["unit"] + ".npz"), meta=np.array(json.dumps(meta)), **arr)
            done += 1
            print(f"[r8 sandbox] {arm} {u['unit']} ({u['grp']}): {len(meta['runs'])} runs, {time.time() - t0:.0f}s "
                  f"(server {meta['secs']['server']:.0f}s); {done} done, {time.time() - t_all:.0f}s", flush=True)
        SB.S.close()
    print(f"[r8 sandbox] {arm} shard {shard} finished: {done} new", flush=True)


def _succ(path, k=K):
    z = np.load(path, allow_pickle=True)
    s = np.asarray(z["success"]).astype(float)
    ch = np.asarray(z["chunks"], np.float32)[:k, :, :5, :6]               # executed actions of every query, seeds only
    ok = np.isfinite(ch).all(-1)
    out = (np.abs(np.nan_to_num(ch)) > ACT_BOUND).any(-1)
    return s[:k], float(out[ok].mean()) if ok.any() else float("nan")


def read(record=False):
    import d4_read
    P = R7.jload(os.path.join(OUT, "plan.json"))
    res = dict(arms={}, K=K, specs=P["specs"], skipped_no_spec=P["skipped_no_spec"])
    lines = []
    base_cache = {}
    for u in P["units"]:
        base_cache[u["unit"]] = _succ(os.path.join(R7.OUT, "sb", u["unit"] + ".npz"))
    for arm in ARMS:
        cells = {}
        for grp in ("wg", "succ"):
            per_d, per_b, per_a, cov_b, cov_a, oob_b, oob_a, miss = {}, {}, {}, {}, {}, [], [], []
            for u in P["units"]:
                if u["grp"] != grp:
                    continue
                f = os.path.join(OUT, "sb", arm, u["unit"] + ".npz")
                if not os.path.exists(f):
                    miss.append(u["unit"])
                    continue
                sa, oa = _succ(f)
                sb, ob = base_cache[u["unit"]]
                t = u["tid"]
                per_d.setdefault(t, []).append(float(sa.mean() - sb.mean()))
                per_b.setdefault(t, []).append(float(sb.mean()))
                per_a.setdefault(t, []).append(float(sa.mean()))
                cov_b.setdefault(t, []).append(float(sb.max()))
                cov_a.setdefault(t, []).append(float(sa.max()))
                oob_a.append(oa)
                oob_b.append(ob)
            if not per_d:
                continue
            c = dict(n=sum(len(v) for v in per_d.values()), missing=miss, base=d4_read.rate(per_b), arm=d4_read.rate(per_a),
                     diff=d4_read.rate(per_d), cov16_base=d4_read.rate(cov_b), cov16_arm=d4_read.rate(cov_a),
                     per_task={t: dict(base=float(np.mean(per_b[t])), arm=float(np.mean(per_a[t])), n=len(per_b[t]),
                                       cov16_base=float(np.mean(cov_b[t])), cov16_arm=float(np.mean(cov_a[t]))) for t in per_b},
                     oob_base=float(np.nanmean(oob_b)), oob_arm=float(np.nanmean(oob_a)))
            cells[grp] = c
            d = c["diff"]
            lines.append(f"{arm} {grp} (n {c['n']}, missing {len(miss)}): per-seed {c['base']['mean']:.3f} -> {c['arm']['mean']:.3f}, "
                         f"diff {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]; cov@16 {c['cov16_base']['mean']:.3f} -> "
                         f"{c['cov16_arm']['mean']:.3f}; out-of-bound actions {c['oob_base']:.3f} -> {c['oob_arm']:.3f}; "
                         + ", ".join(f"{t} {v['base']:.3f}->{v['arm']:.3f}" for t, v in c["per_task"].items()))
        wg, sc = cells.get("wg"), cells.get("succ")
        verdict = None
        if wg is not None:
            ok_gain = wg["diff"]["mean"] >= GAIN_BAR and wg["diff"]["lo"] > 0
            ok_harm = sc is None or sc["diff"]["mean"] >= -HARM_BAR
            verdict = "PASS" if (ok_gain and ok_harm) else "FAIL"
        cells["verdict"] = verdict
        cells["offmanifold_flag"] = bool(wg is not None and wg["oob_arm"] - wg["oob_base"] > 0.10)
        res["arms"][arm] = cells
        lines.append(f"{arm}: VERDICT {verdict} (gain >= +{GAIN_BAR} with CI lo > 0, control harm >= -{HARM_BAR}); off-manifold flag "
                     f"{cells['offmanifold_flag']}")
    res["any_method_pass"] = any(res["arms"].get(a, {}).get("verdict") == "PASS" for a in METHOD_ARMS)
    lines.append(f"ANY METHOD ARM PASSES: {res['any_method_pass']}")
    print("\n".join("[r8 read] " + x for x in lines), flush=True)
    R7.dump(os.path.join(OUT, "read.json"), res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r8_read", bench="libero_90_c", tag="cover", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/r8_cover.py", command="python experiments/r8_cover.py --read --record"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("plan", "sandbox", "smoke", "read", "record"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--arm", default="c3")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    if a.plan:
        plan()
    if a.sandbox:
        sandbox(a.arm, a.shard, a.port, smoke=a.smoke)
    if a.read:
        read(a.record)


if __name__ == "__main__":
    main()
