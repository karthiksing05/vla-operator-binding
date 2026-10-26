"""PLAN_BIND E3 (docs/PLAN_E34.md section 2; pre-registration experiments/prereg/pbind_e34.py, ledger m3_prereg tag pbind_e3):
the frame experiment on pi05_l90c@5000 (frozen).  D4's training APPROACH observations (results/d4/index_train_l90c.json `pre`
frames: open gripper, before the grasp; the demo state is set in the simulator, never executed), one query each, first-chunk
endpoint e = h + scale * sum of the chunk's xyz deltas (D4's readout), K shared noises.

Conditions per observation (validity: D4's rules -- penetration growth <= 2 mm and a pinned 50-step settle within 5 mm of the
expected positions; an invalid condition is skipped):
  base
  obj   the named object moved d = +-x / +-y at 5 cm and 10 cm (nothing else moves)
  rig   every non-table body (movable objects and fixtures) translated d = +-x / +-y, 5 cm (a rigid scene translation)
  yaw   every non-table body rotated +-10 deg about the bodies' xy centroid (positions and orientations)
  inj   action-expert decoder layers 16+17 outputs replaced by those of a SOURCE observation (another training task in another
        scene, sha1 pairing), at base and with the named object moved 5 cm (4 directions)
  inj16 the same, layer 16 only
    --plan                  (chunks env)  -> results/bind/e3_plan.json
    --sandbox --shard i/n   (LIBERO env + bind_e3_serve.py)  -> results/bind/e3/<key>.npz
    --read [--record]       (chunks env)  -> results/bind/e3.json
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
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
OUT = os.path.join(ROOT, "results", "bind")
E3D = os.environ.get("BIND_E3_DIR") or os.path.join(OUT, "e3")
K = 4
SEED_BASE = 61_000_000
MAGS = (0.05, 0.10)
RIG = 0.05
YAW = np.deg2rad(10.0)
DIRS = (("x", 1), ("x", -1), ("y", 1), ("y", -1))
PEN_TOL, SETTLE_TOL = 0.002, 0.005
MAX_OBS = int(os.environ.get("BIND_E3_MAX", 10 ** 6))      # smoke only


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def h8(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def plan():
    import m1_compose_create as M1
    tt = M1.task_table()
    ix = jload(os.path.join(ROOT, "results", "d4", "index_train_l90c.json"))["tasks"]
    obs = []
    for x in ix:
        if not x.get("pre") or not x.get("inst"):
            continue
        f = int(sorted(x["pre"])[-1])
        obs.append(dict(key=f"{x['tid'].replace(':', '_')}_e{x['episode']}_pre{f}", tid=x["tid"], suite=x["suite"], index=int(x["index"]),
                        frame=f, match=x["match"], named=x["inst"], scene=x["scene"], prompt=tt[x["tid"]].language))
    seen, uniq = set(), []
    for o in sorted(obs, key=lambda o: h8("e3:" + o["key"])):
        if o["tid"] not in seen:
            seen.add(o["tid"])
            uniq.append(o)
    for o in uniq:
        src = sorted([s for s in uniq if s["scene"] != o["scene"]], key=lambda s: h8(f"e3src:{o['key']}:{s['key']}"))[0]
        o["source"] = src["key"]
    json.dump(dict(obs=uniq), open(os.path.join(OUT, "e3_plan.json"), "w", encoding="utf-8"), indent=1)
    print(f"[bind e3 plan] {len(uniq)} approach observations (one per training task)", flush=True)


def qyaw(th):
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Scene:
    def __init__(self, suite, index):
        import d5_goal as D5
        self.SB = D5.Sandbox(suite, index)
        self.S = self.SB.S
        self.quat0 = np.array(self.S.sim.model.body_quat, np.float64).copy()

    def reset(self, state):
        self.S.sim.model.body_quat[:] = self.quat0
        self.S.set(state)
        self.S.sim.forward()

    def transform(self, d, th, c):
        """Rigid transform of every non-table body: p -> c + R(th)(p - c) + d, orientation rotated by th about z."""
        S, m, q = self.S, self.S.sim.model, self.S.sim.data.qpos
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for n in S.movable:
            a = S.qadr[n]
            q[a:a + 2] = c + R @ (q[a:a + 2] - c) + d
            q[a + 3:a + 7] = qmul(qyaw(th), q[a + 3:a + 7])
        for n in S.fixtures:
            b = S.bid[n]
            m.body_pos[b, :2] = c + R @ (m.body_pos[b, :2] - c) + d
            m.body_quat[b] = qmul(qyaw(th), m.body_quat[b])
        S.sim.forward()

    def snapshot_pos(self):
        return {n: self.S.pos(n) for n in self.S.bodies}

    def valid(self, apply, expect_fn, bodies):
        """D4's rules for the condition `apply()` from the current base state."""
        S = self.S
        pen0 = {n: S.penetration(n) for n in bodies}
        st = S.sim.get_state().flatten()
        rest0 = S.settle()
        self.reset(st)
        apply()
        exp = expect_fn(rest0)
        pen = {n: S.penetration(n) for n in bodies}
        ok_pen = all(pen[n] <= pen0[n] + PEN_TOL for n in bodies)
        st2 = S.sim.get_state().flatten()
        rest = S.settle()
        err = max((float(np.linalg.norm(rest[n] - exp[n])) for n in S.movable), default=0.0)
        self.reset(st2)
        return bool(ok_pen and err <= SETTLE_TOL), (None if ok_pen and err <= SETTLE_TOL else ("penetration" if not ok_pen else f"settle {err:.3f}"))

    def render(self):
        st8, eef = self.SB.obs8()
        im, wr = self.S.render()
        return im, wr, st8, eef


def train_state(o):
    return np.load(os.path.join(ROOT, o["match"]), allow_pickle=True)["states"][o["frame"]]


def sandbox(shard, port):
    from openpi_client import websocket_client_policy as _wcp
    PA = jload(os.path.join(OUT, "e3_plan.json"))["obs"]
    byk = {o["key"]: o for o in PA}
    P = PA[:MAX_OBS]
    i, n = map(int, shard.split("/"))
    mine = [o for j, o in enumerate(P) if j % n == i]
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    src_cache = {}

    def src_render(key):
        if key not in src_cache:
            s = byk[key]
            sc = Scene(s["suite"], s["index"])
            sc.reset(train_state(s))
            src_cache[key] = sc.render() + (s["prompt"],)
            sc.S.close()
        return src_cache[key]
    for o in mine:
        f = os.path.join(E3D, o["key"] + ".npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        sI, sW, sS, sE, sP = src_render(o["source"])
        sc = Scene(o["suite"], o["index"])
        base = train_state(o)
        sc.reset(base)
        S = sc.S
        named = o["named"]
        p_named = S.pos(named)
        c = np.mean([S.pos(b)[:2] for b in S.bodies], 0)
        conds = [("base", None)]
        for mag in MAGS:
            for ax, sg in DIRS:
                conds.append((f"obj|{ax}{'+' if sg > 0 else '-'}|{mag:g}", ("obj", ax, sg, mag)))
        for ax, sg in DIRS:
            conds.append((f"rig|{ax}{'+' if sg > 0 else '-'}|{RIG:g}", ("rig", ax, sg, RIG)))
        for sg in (1, -1):
            conds.append((f"yaw|{'+' if sg > 0 else '-'}", ("yaw", None, sg, 0.0)))
        rows, meta_c = [], []
        for name, spec in conds:
            sc.reset(base)
            if spec is None:
                ok, why, dvec, th = True, None, np.zeros(2), 0.0
            else:
                kind, ax, sg, mag = spec
                dvec = np.zeros(2)
                th = 0.0
                if kind in ("obj", "rig"):
                    dvec["xy".index(ax)] = sg * mag
                if kind == "yaw":
                    th = sg * YAW
                if kind == "obj":
                    apply = (lambda d=dvec: S.displace(named, d))
                    expect = (lambda r0, d=dvec: {q: r0[q] + (np.r_[d, 0.0] if q == named else 0.0) for q in r0})
                    bodies = [named]
                else:
                    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                    apply = (lambda d=dvec, t_=th: sc.transform(d, t_, c))
                    expect = (lambda r0, d=dvec, R=R: {q: np.r_[c + R @ (r0[q][:2] - c) + d, r0[q][2]] for q in r0})
                    bodies = list(S.bodies)
                ok, why = sc.valid(apply, expect, bodies)
                if ok:                                           # validity resets the model; re-apply before the render
                    sc.reset(base)
                    apply()
            meta_c.append(dict(name=name, valid=ok, why=why, d=dvec.tolist(), th=float(th)))
            if ok:
                rows.append((name,) + sc.render())
        out = {}
        for name, im, wr, st8, eef in rows:
            req = {"b/image": np.stack([im] * K), "b/wrist": np.stack([wr] * K), "b/state": np.stack([st8] * K).astype(np.float32),
                   "b/seed": np.array([SEED_BASE + q for q in range(K)], np.int64), "prompt": o["prompt"]}
            out[name] = np.asarray(client.infer(req)["actions"], np.float32)
            if name == "base" or name.startswith("obj|") and name.endswith("|0.05"):
                for inj in ("16-17", "16"):
                    r = client.infer(dict(req, **{"e3/inj": inj, "e3/src_image": sI, "e3/src_wrist": sW, "e3/src_state": sS,
                                                  "e3/src_prompt": sP}))
                    out[f"inj{inj}|{name}"] = np.asarray(r["actions"], np.float32)
                    if name == "base":
                        out[f"src{inj}"] = np.asarray(r["src_actions"], np.float32)
        eefs = {name: eef.tolist() for name, _, _, _, eef in rows}
        meta = dict(key=o["key"], tid=o["tid"], named=named, p_named=p_named.tolist(), centroid=c.tolist(), conds=meta_c, eef=eefs,
                    source=o["source"], src_eef=sE.tolist(), secs=time.time() - t0)
        os.makedirs(E3D, exist_ok=True)
        tmp = f + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), names=np.array(sorted(out)), **{k.replace("|", "__"): v for k, v in out.items()})
        os.replace(tmp, f)
        print(f"[bind e3 sandbox] {o['key']}: {sum(c_['valid'] for c_ in meta_c)}/{len(meta_c)} valid conditions, {meta['secs']:.0f}s", flush=True)
        sc.S.close()
    print(f"[bind e3 sandbox] shard {shard} finished", flush=True)


def read(record=False):
    import d4_read
    scale = d4_read.scale_for("pi05_l90c")
    P = jload(os.path.join(OUT, "e3_plan.json"))["obs"]
    rec = {}
    for o in P:
        f = os.path.join(E3D, o["key"] + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))

        def ep(name, eef_name=None):
            k = name.replace("|", "__")
            if k not in z.files:
                return None
            h = np.asarray(meta["eef"][eef_name or name.split("|", 1)[-1] if name.startswith("inj") else (eef_name or name)], float)
            return d4_read.endpoints(z[k], h, scale).mean(0)
        e0 = ep("base")
        cond = {c["name"]: c for c in meta["conds"]}
        r = dict(tid=o["tid"], obj={}, rig={}, yaw={}, inj={})
        pn = np.asarray(meta["p_named"], float)[:2]
        cc = np.asarray(meta["centroid"], float)
        for name, c in cond.items():
            if not c["valid"] or name == "base":
                continue
            e = ep(name)
            de = (e - e0)[:2]
            d = np.asarray(c["d"], float)
            if name.startswith("obj"):
                r["obj"][name] = dict(de=de.tolist(), d=d.tolist(), gain=float(de @ d / (d @ d)), mag=float(np.linalg.norm(d)))
            elif name.startswith("rig"):
                r["rig"][name] = dict(de=de.tolist(), d=d.tolist(), gain=float(de @ d / (d @ d)))
            else:
                th = c["th"]
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                pred_scene = cc + R @ (e0[:2] - cc) - e0[:2]
                pred_obj = cc + R @ (pn - cc) - pn
                r["yaw"][name] = dict(de=de.tolist(), pred_scene=pred_scene.tolist(), pred_obj=pred_obj.tolist(),
                                      err_world=float(np.linalg.norm(de)), err_scene=float(np.linalg.norm(de - pred_scene)),
                                      err_obj=float(np.linalg.norm(de - pred_obj)))
        hs = np.asarray(meta["src_eef"], float)
        h0 = np.asarray(meta["eef"]["base"], float)
        for inj in ("16-17", "16"):
            eb = ep(f"inj{inj}|base", "base")
            if eb is None:
                continue
            k = f"src{inj}"
            es = d4_read.endpoints(z[k], hs, scale).mean(0)
            delta_lock = h0 + (es - hs)
            g = {}
            for name in [n for n in cond if n.startswith("obj|") and n.endswith("|0.05") and cond[n]["valid"]]:
                ei = ep(f"inj{inj}|{name}", name)
                if ei is None:
                    continue
                d = np.asarray(cond[name]["d"], float)
                g[name] = float((ei - eb)[:2] @ d / (d @ d))
            r["inj"][inj] = dict(dist_target=float(np.linalg.norm((eb - e0)[:2])), dist_delta_lock=float(np.linalg.norm((eb - delta_lock)[:2])),
                                 dist_src_abs=float(np.linalg.norm((eb - es)[:2])), gains=g)
        rec[o["key"]] = r
    keys = sorted(rec)
    per = lambda f: {rec[k]["tid"]: [v for v in f(rec[k])] for k in keys}                     # noqa: E731
    res = dict(n_obs=len(keys))
    g5 = per(lambda r: [v["gain"] for n, v in r["obj"].items() if n.endswith("|0.05")])
    g10 = per(lambda r: [v["gain"] for n, v in r["obj"].items() if n.endswith("|0.1")])
    grig = per(lambda r: [v["gain"] for v in r["rig"].values()])
    res["gain_obj_5cm"] = d4_read.rate(g5)
    res["gain_obj_10cm"] = d4_read.rate(g10)
    res["gain_rig_5cm"] = d4_read.rate(grig)
    ssd = per(lambda r: [float(np.sum(np.asarray(v["de"]) ** 2) - np.sum((np.asarray(v["de"]) - np.asarray(v["d"])) ** 2))
                         for v in list(r["obj"].values()) + list(r["rig"].values())])
    res["SSworld_minus_SSobj"] = d4_read.rate(ssd)
    ys = per(lambda r: [v["err_world"] - v["err_scene"] for v in r["yaw"].values()])
    yo = per(lambda r: [v["err_obj"] - v["err_scene"] for v in r["yaw"].values()])
    res["yaw_errworld_minus_errscene"] = d4_read.rate(ys)
    res["yaw_errobj_minus_errscene"] = d4_read.rate(yo)
    for inj in ("16-17", "16"):
        gi = per(lambda r: list(r["inj"].get(inj, {}).get("gains", {}).values()))
        lock = per(lambda r: [r["inj"][inj]["dist_target"] - r["inj"][inj]["dist_delta_lock"]] if inj in r["inj"] else [])
        res[f"inj{inj}_gain_obj_5cm"] = d4_read.rate(gi)
        res[f"inj{inj}_target_minus_lock_dist"] = d4_read.rate(lock)
        res[f"inj{inj}_median_dist"] = dict(target=float(np.median([rec[k]["inj"][inj]["dist_target"] for k in keys if inj in rec[k]["inj"]])),
                                            delta_lock=float(np.median([rec[k]["inj"][inj]["dist_delta_lock"] for k in keys if inj in rec[k]["inj"]])),
                                            src_abs=float(np.median([rec[k]["inj"][inj]["dist_src_abs"] for k in keys if inj in rec[k]["inj"]])))
    v = sum(c for r in rec.values() for c in [len(r["obj"]) + len(r["rig"]) + len(r["yaw"])])
    res["valid_conditions"] = v
    s = res["SSworld_minus_SSobj"]
    res["H1_verdict"] = "PASS" if s["lo"] is not None and s["lo"] > 0 and res["gain_obj_5cm"]["lo"] > 0.5 else "FAIL"
    gi = res["inj16-17_gain_obj_5cm"]
    lk = res["inj16-17_target_minus_lock_dist"]
    res["H2_verdict"] = "PASS" if gi["hi"] is not None and gi["hi"] < 0.5 and lk["lo"] > 0 else "FAIL"
    res["H3_verdict"] = "PASS" if res["gain_obj_10cm"]["lo"] is not None and res["gain_obj_10cm"]["lo"] > 0.5 else "FAIL"
    json.dump(dict(res, rec=rec), open(os.path.join(OUT, "e3.json"), "w", encoding="utf-8"), indent=1, default=float)
    for k_, v_ in res.items():
        print(f"[bind e3] {k_}: {json.dumps(v_, default=float)}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("bind_e3", bench="libero_90_c", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/bind_e3.py", command="python experiments/bind_e3.py --read --record"))


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
