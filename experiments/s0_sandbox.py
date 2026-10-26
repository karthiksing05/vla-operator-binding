"""PLAN_SOLVE S0 (docs/PLAN_S0.md): closed-loop dissociation test.  E2b's sandbox (experiments/bind_e2b.py: the same units, start
states, seeds, step budget, bound-object rule and first-release predicate labels), with the role masks of an ARM sent to
s0_serve.py at every query.  LIBERO env client.

Arms (--arm):
  oracle  the named instance's segmentation and the own target's mask (S1's region box / body segmentation), live from the sim
  wrong   a DIFFERENT movable object (the first other movable, by name, that is not the target) and a different goal candidate (the
          first scene candidate of E2's site list that is not an own target): does the policy follow the mask?
  zero    zero masks: the adapter's edit is zero by construction (a parity arm; run on a subset)
  zs      no masks (plain d5_serve sampling through the same server)
--set dev    the 24 dev tasks (results/bind/e2_plan.json units; fresh_m1 states 0-4; K 8 seeds)
--set train  DAMAGE: S0_TRAIN_CELLS training cells x 5 fresh sampler states (env.seed(9400000 + k); reset) x K 8 seeds
-> results/bind/s0/roll/<set>_<arm>/<unit>.npz (meta: E2b's fields + the mask names)

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/s0_sandbox.py --set dev --arm oracle --shard i/n --port P
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

OUT = os.path.join(ROOT, "results", "bind", "s0", "roll")
SETTLE = 20
MAX_STEPS = int(os.environ.get("BIND_MAX_STEPS", 400))
TRAIN_BASE, N_TRAIN_STATES = 9_400_000, 5
# damage cells: training cells with an In / On goal in libero_90 scenes that have a dev task (fixed before any rollout)
S0_TRAIN_CELLS = ("libero_90:13", "libero_90:17", "libero_90:25", "libero_90:26", "libero_90:42", "libero_90:43", "libero_90:46",
                  "libero_90:53", "libero_90:63", "libero_90:65")


def train_units():
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    out = []
    for tid in S0_TRAIN_CELLS:
        x = tasks[tid]
        at = next(a for a in x["goal"] if a[0] in ("In", "On") and len(a) == 3)
        suite, i = tid.split(":")
        sites = sorted({a[2] for t, y in tasks.items() if y["scene"] == x["scene"] for a in y["goal"]
                        if a[0] in ("In", "On") and len(a) == 3} | {at[2]})
        for k in range(N_TRAIN_STATES):
            out.append(dict(unit=f"{tid.replace(':', '_')}_t{k}", tid=tid, suite=suite, index=int(i), state=k, prompt=x["lang"],
                            named=at[1], own=[at[2]], sites=sites))
    return out


class Masker:
    """Live oracle masks over the agentview tokens: bodies by segmentation, region sites by their projected box (S1's rules)."""

    def __init__(self, SB):
        import libero_client as LC
        import s1_render as SR
        self.LC, self.W, self.SB = LC, SR._where(), SB
        self.tables = LC.seg_tables(SB.S.env)
        self.sid = {SB.sim.model.site_id2name(i): i for i in range(SB.sim.model.nsite)}

    def __call__(self, names):
        import f0b_firstframes_seg as F0
        sim = self.SB.sim
        cov = self.LC.seg_maps(self.SB.S.env, self.tables, "agentview").astype(np.float32)
        bn = list(self.tables[0])
        out = []
        for n in names:
            if n is None:
                out.append(np.zeros(256, np.float32))
            elif n in bn:
                out.append(cov[bn.index(n)])
            elif n in self.sid:
                i = self.sid[n]
                K = F0.CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
                Ex = F0.CU.get_camera_extrinsic_matrix(sim, "agentview")
                out.append(self.W.box_map(sim.data.site_xpos[i], sim.data.site_xmat[i], sim.model.site_size[i], K, Ex).reshape(-1)
                           .astype(np.float32))
            else:
                out.append(np.zeros(256, np.float32))
        return out


def make_fresh():
    """The damage cells' start states: LIBERO's sampler, env.seed(TRAIN_BASE + k); reset -> results/bind/s0/fresh_train/<suite>_<i>.npy."""
    import pathlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    od = os.path.join(ROOT, "results", "bind", "s0", "fresh_train")
    os.makedirs(od, exist_ok=True)
    for tid in S0_TRAIN_CELLS:
        suite, i = tid.split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(i))
        env = OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        st = []
        for k in range(N_TRAIN_STATES):
            env.seed(TRAIN_BASE + k)
            env.reset()
            st.append(env.env.sim.get_state().flatten())
        env.close()
        np.save(os.path.join(od, f"{suite}_{i}.npy"), np.stack(st))
        print(f"[s0 fresh] {tid}: {len(st)} states", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("dev", "train"), default="dev")
    ap.add_argument("--arm", choices=("oracle", "wrong", "zero", "zs"), default="zs")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--make-fresh", action="store_true")
    a = ap.parse_args()
    if a.make_fresh:
        return make_fresh()
    from openpi_client import websocket_client_policy as _wcp
    import bind_e2b as E2B
    import d5_goal as D5
    import m1_compose_create as M1
    units = E.jload(os.path.join(E.OUT, "e2_plan.json"))["units"] if a.set == "dev" else train_units()
    i, n = map(int, a.shard.split("/"))
    units = [u for j, u in enumerate(units) if j % n == i]
    if a.max_units:
        units = units[:a.max_units]
    od = os.path.join(OUT, f"{a.set}_{a.arm}")
    os.makedirs(od, exist_ok=True)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", a.port)
    by = {}
    for u in units:
        by.setdefault((u["suite"], u["index"]), []).append(u)
    for tk in sorted(by):
        todo = [u for u in by[tk] if not os.path.exists(os.path.join(od, u["unit"] + ".npz"))]
        if not todo:
            continue
        SB = D5.Sandbox(*tk)
        fresh = np.load(os.path.join(M1.R, M1.FRESH_DIR, f"{tk[0]}_{tk[1]}.npy") if a.set == "dev"
                        else os.path.join(ROOT, "results", "bind", "s0", "fresh_train", f"{tk[0]}_{tk[1]}.npy"))
        movable = list(SB.S.movable)
        for u in todo:
            t0 = time.time()
            SB.S.set(fresh[u["state"]])
            for _ in range(E.WAIT):
                SB.e.step(E.HOLD)
            st = SB.flat()
            mk = Masker(SB)
            if a.arm == "wrong":
                wo = next((m for m in sorted(movable) if m != u["named"] and m not in u["own"]), None)
                wg = next((s for s in u["sites"] if s not in u["own"] and s != wo and s != u["named"]), None)
                names = (wo, wg)
            else:
                names = (u["named"], u["own"][0])
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
                    mo, mg = mk(names) if a.arm in ("oracle", "wrong") else (np.zeros(256, np.float32), np.zeros(256, np.float32))
                    rows.append((im, wr, st8, r.seed0 + r.q, mo, mg))
                chunks = []
                for g0 in range(0, len(rows), E.BATCH):
                    grp = rows[g0:g0 + E.BATCH]
                    req = {"b/image": np.stack([x[0] for x in grp]), "b/wrist": np.stack([x[1] for x in grp]),
                           "b/state": np.stack([x[2] for x in grp]).astype(np.float32),
                           "b/seed": np.array([x[3] for x in grp], np.int64), "prompt": u["prompt"]}
                    if a.arm != "zs":
                        req["s0/mobj"] = np.stack([x[4] for x in grp])
                        req["s0/mgoal"] = np.stack([x[5] for x in grp])
                    ts = time.time()
                    chunks.extend(list(np.asarray(client.infer(req)["actions"])))
                    t_srv += time.time() - ts
                for r, ch in zip(act, chunks):
                    SB.restore(r.snap)
                    for x in ch[:E.REPLAN]:
                        x = np.asarray(x, np.float64)
                        if x[6] < 0 and r.bound is not None and not r.released:
                            r.released, r.rel_step = True, r.steps
                            keep = SB.snapshot()
                            for _ in range(SETTLE):
                                SB.e.step(E.HOLD)
                            r.labels = E2B.preds(SB, r.bound, sorted(set(u["sites"]) | set(n for n in names if n)))
                            SB.restore(keep)
                        _, _, done, _ = SB.e.step(x.tolist())
                        r.steps += 1
                        if x[6] > 0 and r.bound is None:
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
            meta = dict(unit=u["unit"], tid=u["tid"], arm=a.arm, mask_names=list(names), named=u["named"], own=u["own"],
                        bound=[r.bound or "" for r in runs], released=[r.released for r in runs], rel_step=[r.rel_step for r in runs],
                        labels=[r.labels for r in runs], success=[r.success for r in runs], steps=[r.steps for r in runs],
                        secs=time.time() - t0, server=t_srv)
            D5.save_atomic(os.path.join(od, u["unit"] + ".npz"), meta=np.array(json.dumps(meta)))
            print(f"[s0 sandbox] {a.set} {a.arm} {u['unit']}: success {sum(meta['success'])}/{E.K}, bound {meta['bound']}, "
                  f"{meta['secs']:.0f}s (server {t_srv:.0f}s)", flush=True)
        SB.S.close()
    print(f"[s0 sandbox] {a.set} {a.arm} shard {a.shard} finished", flush=True)


if __name__ == "__main__":
    main()
