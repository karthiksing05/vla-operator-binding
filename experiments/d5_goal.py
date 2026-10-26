"""PROBE D5 (docs/PLAN_D5.md; pre-registration experiments/prereg/pd5_goal.py, ledger m3_prereg tag pd5_goal): a goal-role binding
operator extracted from the frozen pi05_l90c@5000 by counterfactual displacement, read out at the RELEASE point.

D4 (docs/PLAN_D4.md section 6) found the approach operator but no place operator: a +-2 cm shift of a container or fixture barely moves a
10-step carry chunk.  D5 changes two things, both pre-declared: larger displacements (+-5 cm primary, +-10 cm secondary) and a readout
that resolves a goal -- the hand position at the policy's own RELEASE (the first executed open command), rolled forward from the
carry state in a SANDBOXED copy of the simulator, the frozen policy queried every 5 steps on the renders of that counterfactual world
with shared noise.  No guidance, no weight change; demos are never executed or replayed (training states are set in the simulator,
as in D4, and the demo's gripper commands are read only to reconstruct the gripper's internal ramp state at that frame).

Stages (each its own SLURM job; nothing here runs on the login node):
  --inspect                    (chunks env)  data availability -> print only
  --index                      (chunks env)  training + dev carry observations, gripper histories, zs seeds -> results/d5/index.json
  --parity                     (LIBERO env)  sandbox restore parity: recorded zs actions re-executed from a restored recorded state, and
                                             interleaved vs uninterrupted sandboxes -> results/d5/parity.json
  --sandbox --set S --shard    (LIBERO env + experiments/d5_serve.py)  counterfactual sandboxes -> results/d5/sb/<S>/<key>.npz
  --read [--record]            (chunks env)  G1-G3, descriptive readouts, goal-anchor localisation -> results/d5/read.json
"""
from __future__ import annotations

import argparse
import glob
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

OUT = os.environ.get("D5_OUT") or os.path.join(ROOT, "results", "d5")
D4D = os.path.join(ROOT, "results", "d4")
RB2 = os.path.join(ROOT, "results", "rb2")
RROBOT = os.path.join(ROOT, "results", "robot")
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
# ── settings (fixed before any D5 policy output; docs/PLAN_D5.md) ─────────────────────────────────────────────────────
MAGS = (0.05, 0.10)              # displacement magnitudes (m): +-5 cm PRIMARY, +-10 cm secondary; one body at a time, x and y
PRIMARY_MAG = 0.05
PERTS = (("x", +1), ("x", -1), ("y", +1), ("y", -1))
NOISE_BASE = {"A1": 11_000_000, "A2": 12_000_000, "B1": 21_000_000, "B2": 22_000_000}   # query q of a sandbox: seed base + q
NOISES_BY_MAG = {0.05: ("A1", "A2"), 0.10: ("A1",)}     # shared noise sets per magnitude (A1 shared by both)
BASE_NOISES = ("A1", "A2", "B1", "B2")                  # the unperturbed base under every set (B: the noise floor)
Q_MAX = 24                       # queries per sandbox (120 steps = 6 s) before the release is declared missing
REPLAN = 5
BATCH = 48                       # sandbox rows per server request
PEN_TOL, SETTLE_STEPS, SETTLE_TOL = 0.002, 50, 0.005     # D4's validity rules, unchanged
N_SUCC_DEV = 4                   # dev: zs-success post-grasp observations per recombination task (sha1 order); every wrong-goal one
NEAR_REL = 0.10                  # carried_elsewhere: the actual body = the scored body nearest (xy) the zs release, if within 10 cm
SERVE_SEED, WARM_CALLS = 13, 2   # zs serving: seed 13 + the server's call count (2 warm-up calls before the first query)
TABLES = ("main_table", "kitchen_table", "study_table", "living_room_table", "table", "floor")


def slug(tid):
    return tid.replace(":", "_")


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def save_atomic(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def h8(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def body_of(name, bodies):
    if name in bodies:
        return name
    c = [b for b in bodies if name.startswith(b + "_")]
    return max(c, key=len) if c else None


def grip_ramp(cmds, nsub=25):
    """The Panda gripper's internal ramp state (robosuite PandaGripper.current_action: +-0.01 per substep toward the commanded sign,
    clipped to [-1, 1]; never reset by env.reset) after executing the given gripper commands in order.  Increments within one control
    step share a sign, so clipping once per step equals clipping per substep."""
    ca = np.zeros(2)
    for g in cmds:
        ca = np.clip(ca + np.array([-1.0, 1.0]) * 0.01 * nsub * np.sign(float(g)), -1.0, 1.0)
    return ca


# ── inspect (chunks env) ─────────────────────────────────────────────────────────────────────────────────────────────
def inspect():
    import m1_compose_create as M1
    tt = M1.task_table()
    for tid, b in M1.dev_tasks():
        t = tt[tid]
        print(f"{tid} ({b}) '{t.language}' goal {t.goal} objects {t.objects} fixtures {t.fixtures}")


# ── index (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────────
def index():
    import m1_compose_create as M1
    from lifelong_ood.policy import data as D
    tt = M1.task_table()
    dev = dict(M1.dev_tasks())
    tix = {x["tid"]: x for x in jload(os.path.join(D4D, "index_train_l90c.json"))["tasks"]}
    demos, d90 = D.Demos(), D.Demos90()
    cache = {}
    train = []
    for f in sorted(glob.glob(os.path.join(D4D, "render", "train_l90c", "*carry*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        x = tix[meta["tid"]]
        ek = (x["ipec"], x["episode"])
        if ek not in cache:
            cache[ek] = (d90 if x["ipec"] else demos).load(x["episode"])["actions"]
        A = cache[ek]
        fr = int(meta["frame"])
        train.append(dict(key="tr_" + meta["key"], set="train", tid=meta["tid"], suite=x["suite"], index=int(x["index"]),
                          prompt=tt[meta["tid"]].language, filler=meta["filler"], filler_type=x["target_type"], held=meta["held"],
                          target=x["target"], match=x["match"], frame=fr, grip_cmds=[float(v) for v in A[:fr, 6]],
                          wait=0, scene=x["scene"]))
    devo = []
    for f in sorted(glob.glob(os.path.join(D4D, "dev", "index", "*.json"))):
        x = jload(f)
        tid = x["task"]
        suite, ti = tid.split(":")
        t = tt[tid]
        roll = jload(os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}.json"))
        eps = [e for v in roll["tasks"].values() for e in v["episodes"]]
        assert [int(e["trial"]) for e in eps] == list(range(len(eps))), tid
        cum = np.concatenate([[0], np.cumsum([int(e["calls"]) for e in eps])])
        succ = sorted([e for e in x["episodes"] if e.get("success") and any(m["role"] == "post" for m in e["obs"])],
                      key=lambda e: h8(f"d5succ:{tid}:{e['trial']}"))
        succ_keep = {e["trial"] for e in succ[:N_SUCC_DEV]} if dev[tid] != "part" else set()
        for e in x["episodes"]:
            post = [m for m in e["obs"] if m["role"] == "post"]
            if not post or not (e.get("type") == "wrong_goal" or e["trial"] in succ_keep):
                continue
            m = post[0]
            j = int(m["j"])
            assert j % REPLAN == 0, (tid, e["trial"], j)
            rz = np.load(os.path.join(D4D, "roll", f"d4_zs_{slug(tid)}_rec", f"task{ti}_trial{e['trial']}.npz"), allow_pickle=True)
            A, S = rz["action"], rz["state"]
            names = [str(q) for q in rz["obj_names"]]
            rel = [s for s in range(j, len(A)) if A[s, 6] < 0]
            s_rel = rel[0] if rel else None
            rmeta = json.loads(str(np.load(os.path.join(D4D, "render", "dev", m["key"] + ".npz"), allow_pickle=True)["meta"]))
            bodies = list(rmeta["pos"])
            held = m["held"]
            tgt = m["named"][0]
            tbody = body_of(tgt, bodies)
            alt = [body_of(a, bodies) for a in (e.get("alt") or [])]
            alt = sorted({a for a in alt if a is not None and a != held})
            hand_rel = (S[s_rel, :3] if s_rel is not None else S[-1, :3]).astype(float)
            obj_end = rz["obj_pos"][-1, names.index(held)].astype(float) if held in names else None
            cands = [q for q in bodies if q != held]
            near = None
            if e.get("type") == "wrong_goal" and not alt:
                dn = {q: float(np.linalg.norm(np.asarray(rmeta["pos"][q])[:2] - hand_rel[:2])) for q in cands if q != tbody}
                if dn and min(dn.values()) <= NEAR_REL:
                    near = min(dn, key=dn.get)
            devo.append(dict(key="dv_" + m["key"], set="dev", tid=tid, suite=suite, index=int(ti), bin=dev[tid], trial=int(e["trial"]),
                             prompt=t.language, filler=tbody, target=tgt, filler_type=(dict(t.objects, **t.fixtures).get(tbody)
                                                                                       if tbody else None),
                             held=held, j=j, grip_cmds=[-1.0] * 10 + [float(v) for v in A[:j, 6]], wait=10,
                             type=e.get("type"), sub=e.get("sub"), success=bool(e.get("success")), alt_bodies=alt, near_body=near,
                             actual_body=(alt[0] if len(alt) == 1 else (near if not alt else None)), n_alt=len(alt),
                             zs_release_step=s_rel, zs_release_hand=hand_rel.tolist(), zs_obj_end=(obj_end.tolist() if obj_end is not None else None),
                             zs_seed0=int(SERVE_SEED + WARM_CALLS + cum[int(e["trial"])] + j // REPLAN), zs_len=int(len(A))))
    res = dict(train=train, dev=devo)
    dump(os.path.join(OUT, "index.json"), res)
    by = {}
    for o in devo:
        k = (o["tid"], o["type"])
        by[k] = by.get(k, 0) + 1
    print(f"[d5 index] training carry observations {len(train)} ({len({o['tid'] for o in train})} tasks); dev {len(devo)}: "
          + "; ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(by.items())), flush=True)
    print("[d5 index] dev wrong-goal actual bodies: " + "; ".join(f"{o['key']} target {o['filler']} alt {o['alt_bodies']} near "
                                                                 f"{o['near_body']}" for o in devo if o["type"] == "wrong_goal"), flush=True)


def load_index():
    return jload(os.path.join(OUT, "index.json"))


# ── the sandbox (LIBERO env) ─────────────────────────────────────────────────────────────────────────────────────────
class Box:
    """One sandboxed counterfactual world: its full simulator snapshot + the robosuite state not in the MjSimState."""
    __slots__ = ("label", "cand", "axis", "sign", "mag", "noise", "seed0", "snap", "q", "done", "released", "success", "rel_hand",
                 "rel_obj", "rel_step", "chunk0", "h0", "hand_q", "steps")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.q, self.done, self.released, self.success, self.steps, self.hand_q = 0, False, False, False, 0, []


class Sandbox:
    """D4's Sim (one task's LIBERO env) with snapshot / restore of everything an env.step depends on."""

    def __init__(self, suite, index):
        import d4_bind as D4
        self.S = D4.Sim(suite, index)
        self.e, self.sim = self.S.e, self.S.sim
        self.robot = self.e.robots[0]
        self.ctrl, self.grip = self.robot.controller, self.robot.gripper
        self.nsub = int(round(self.e.control_timestep / self.e.model_timestep))
        self.e.horizon = 10 ** 9
        for n in list(self.e._observables):                    # the policy's images are rendered at each query (D4's render)
            if n.endswith("_image") or n.endswith("_depth"):
                self.e.modify_observable(n, "enabled", False)

    def snapshot(self, into=None):
        """MuJoCo's whole integration state (mj_getState, mjSTATE_INTEGRATION: time, qpos, qvel, act, warm start, ctrl, applied forces,
        mocap, user data, plugins) + the model's body positions (fixture displacements) + the robosuite state outside MjData.  `into`:
        an earlier snapshot whose buffer is reused."""
        import mujoco
        m = self.sim.model._model
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        buf = into["data"] if into is not None else np.empty(mujoco.mj_stateSize(m, spec), np.float64)
        mujoco.mj_getState(m, self.sim.data._data, buf, spec)
        return dict(data=buf, bp=np.array(self.sim.model.body_pos, np.float64).copy(), ca=np.array(self.grip.current_action).copy(),
                    gpos=np.array(self.ctrl.goal_pos).copy(), gori=np.array(self.ctrl.goal_ori).copy(), ts=int(self.e.timestep))

    def restore(self, s):
        import mujoco
        self.sim.model.body_pos[:] = s["bp"]
        mujoco.mj_setState(self.sim.model._model, self.sim.data._data, s["data"], int(mujoco.mjtState.mjSTATE_INTEGRATION))
        self.sim.forward()
        self.grip.current_action = s["ca"].copy()
        self.ctrl.goal_pos, self.ctrl.goal_ori = s["gpos"].copy(), s["gori"].copy()
        self.ctrl.new_update = True
        self.e.timestep, self.e.done = s["ts"], False

    def flat(self):
        return self.sim.get_state().flatten()

    def start(self, state, grip_cmds, disp=None):
        """The observation's state, optionally one body displaced, the gripper ramp reconstructed, the OSC goal at the current pose,
        a zero warm start -> snapshot."""
        self.S.set(state)
        if disp is not None:
            self.S.displace(disp[0], disp[1])
        self.sim.forward()
        self.sim.data._data.qacc_warmstart[:] = 0.0
        self.grip.current_action = grip_ramp(grip_cmds, self.nsub)
        self.ctrl.update(force=True)
        self.ctrl.goal_pos, self.ctrl.goal_ori = np.array(self.ctrl.ee_pos).copy(), np.array(self.ctrl.ee_ori_mat).copy()
        self.e.timestep, self.e.done = 0, False
        return self.snapshot()

    def obs8(self):
        import d4_bind as D4
        ob = self.e._get_observations(force_update=True)
        return D4.state8(ob), np.asarray(ob["robot0_eef_pos"], np.float64)

    def held_pos(self, held):
        return self.S.pos(held) if held else None


def valid_perts(SB, state, cands, mag):
    """D4's validity rules at magnitude `mag`: penetration growth <= 2 mm and the pinned 50-step settle (the body within 5 mm of base +
    delta, every other movable body within 5 mm of its base end).  -> {(cand, axis, sign): (valid, reason)}."""
    S = SB.S
    S.set(state)
    pen0 = {n: S.penetration(n) for n in cands}
    rest0 = S.settle()
    out = {}
    for n in cands:
        for ax, sg in PERTS:
            d = np.zeros(2)
            d["xy".index(ax)] = sg * mag
            S.set(state)
            S.displace(n, d)
            pen = S.penetration(n)
            ok_pen = pen <= pen0[n] + PEN_TOL
            rest = S.settle()
            err = {q: float(np.linalg.norm(rest[q] - (rest0[q] + (np.r_[d, 0.0] if q == n else 0.0)))) for q in S.movable}
            worst = max(err.values()) if err else 0.0
            ok = bool(ok_pen and worst <= SETTLE_TOL)
            out[(n, ax, sg)] = (ok, "" if ok else ("penetration" if not ok_pen else "settle:" + max(err, key=err.get)))
    S.set(state)
    return out


def run_boxes(SB, boxes, held, prompt, client, sig_first=False, log=None):
    """Advance every box in lockstep: at each round, render + query (one batched request per BATCH rows), then execute REPLAN actions
    per box until its first open command (the release), success, or Q_MAX queries.  -> per-box results; the base box's first-query
    SigLIP readout if sig_first."""
    sig = None
    t_srv = 0.0
    while True:
        act = [b for b in boxes if not b.done]
        if not act:
            break
        rows = []
        for b in act:
            SB.restore(b.snap)
            st8, hand = SB.obs8()
            if b.q == 0:
                b.h0 = hand.copy()
            b.hand_q.append(hand.copy())
            im, wr = SB.S.render()
            rows.append((im, wr, st8, b.seed0 + b.q))
        chunks = []
        for g0 in range(0, len(rows), BATCH):
            grp = rows[g0:g0 + BATCH]
            req = {"b/image": np.stack([r[0] for r in grp]), "b/wrist": np.stack([r[1] for r in grp]),
                   "b/state": np.stack([r[2] for r in grp]).astype(np.float32), "b/seed": np.array([r[3] for r in grp], np.int64),
                   "prompt": prompt}
            if sig_first and sig is None and g0 == 0:
                req["b/sig"] = np.array([act[0].q == 0 and act[0].label == "base_A1"] + [False] * (len(grp) - 1))
            t0 = time.time()
            out = client.infer(req)
            t_srv += time.time() - t0
            chunks.extend(list(np.asarray(out["actions"])))
            if "b/sig" in req and req["b/sig"][0]:
                sig = (np.asarray(out["sig_tok"])[0], np.asarray(out["sig_val"])[0])
        for b, ch in zip(act, chunks):
            SB.restore(b.snap)
            if b.q == 0:
                b.chunk0 = np.asarray(ch, np.float32)
            for a in ch[:REPLAN]:
                a = np.asarray(a, np.float64)
                if a[6] < 0:                                    # the first executed open command: the release
                    _, hand = SB.obs8()
                    b.released, b.done, b.rel_hand, b.rel_step = True, True, hand, b.steps
                    hp = SB.held_pos(held)
                    b.rel_obj = None if hp is None else np.asarray(hp, np.float64)
                    break
                _, _, done, _ = SB.e.step(a.tolist())
                b.steps += 1
                if done:
                    _, hand = SB.obs8()
                    b.success, b.done, b.rel_hand, b.rel_step = True, True, hand, b.steps
                    hp = SB.held_pos(held)
                    b.rel_obj = None if hp is None else np.asarray(hp, np.float64)
                    break
            b.q += 1
            if not b.done and b.q >= Q_MAX:
                _, hand = SB.obs8()
                b.done, b.rel_hand, b.rel_step = True, hand, b.steps
                hp = SB.held_pos(held)
                b.rel_obj = None if hp is None else np.asarray(hp, np.float64)
            b.snap = SB.snapshot(b.snap)
    return sig, t_srv


def obs_state(o):
    if o["set"] == "train":
        mz = np.load(os.path.join(ROOT, o["match"]), allow_pickle=True)
        return mz["states"][o["frame"]]
    return np.load(os.path.join(D4D, "dev", "obs", o["key"][3:] + ".npz"), allow_pickle=True)["sim"]


def sandbox_obs(SB, o, client, extra_zs=True):
    """Every sandbox of one observation -> dict of arrays (saved)."""
    import libero_client as LC
    state = obs_state(o)
    S = SB.S
    held = o["held"]
    cands = [q for q in S.bodies if q != held]
    t0 = time.time()
    val = {mag: valid_perts(SB, state, cands, mag) for mag in MAGS}
    t_val = time.time() - t0
    # the base: geometry for the goal anchor (segmentation coverage per body, live-depth token points), body poses
    S.set(state)
    tables = LC.seg_tables(S.env)
    cov = LC.seg_maps(S.env, tables, "agentview")
    tok_xyz = LC.r5s_tok_xyz(S.sim)
    pos = {n: S.pos(n).tolist() for n in S.bodies}
    regions = {}
    for rn, ost in S.e.object_states_dict.items():               # every BDDL object / region site (descriptive region readout)
        try:
            regions[rn] = [float(v) for v in ost.get_geom_state()["pos"]]
        except Exception:                                            # noqa: BLE001
            pass
    boxes = []
    for nz in BASE_NOISES:
        boxes.append(Box(label=f"base_{nz}", cand=None, axis=None, sign=0, mag=0.0, noise=nz, seed0=NOISE_BASE[nz],
                         snap=SB.start(state, o["grip_cmds"])))
    if extra_zs and o["set"] == "dev":
        boxes.append(Box(label="base_zs", cand=None, axis=None, sign=0, mag=0.0, noise="zs", seed0=int(o["zs_seed0"]),
                         snap=SB.start(state, o["grip_cmds"])))
    for mag in MAGS:
        for n in cands:
            if not all(val[mag][(n, ax, sg)][0] for ax, sg in PERTS):
                continue                                        # a body is scored only if all 4 perturbations are valid
            for ax, sg in PERTS:
                d = np.zeros(2)
                d["xy".index(ax)] = sg * mag
                for nz in NOISES_BY_MAG[mag]:
                    boxes.append(Box(label=f"{n}|{ax}{'+' if sg > 0 else '-'}|{mag:g}|{nz}", cand=n, axis=ax, sign=sg, mag=mag, noise=nz,
                                     seed0=NOISE_BASE[nz], snap=SB.start(state, o["grip_cmds"], (n, d))))
    t1 = time.time()
    sig, t_srv = run_boxes(SB, boxes, held, o["prompt"], client, sig_first=True)
    t_run = time.time() - t1
    labels = [b.label for b in boxes]
    meta = dict(key=o["key"], set=o["set"], tid=o["tid"], filler=o["filler"], held=held, cands=cands, pos=pos, regions=regions,
                bodies_seg=list(tables[0]), labels=labels,
                validity={f"{mag:g}": {f"{n}|{ax}|{sg}": list(v) for (n, ax, sg), v in val[mag].items()} for mag in MAGS},
                boxes=[dict(label=b.label, cand=b.cand, axis=b.axis, sign=b.sign, mag=b.mag, noise=b.noise, released=b.released,
                            success=b.success, steps=b.steps, queries=b.q) for b in boxes],
                secs=dict(validity=t_val, run=t_run, server=t_srv))
    arr = dict(rel_hand=np.stack([b.rel_hand for b in boxes]).astype(np.float64),
               rel_obj=np.stack([b.rel_obj if b.rel_obj is not None else np.full(3, np.nan) for b in boxes]).astype(np.float64),
               h0=np.stack([b.h0 for b in boxes]).astype(np.float64), chunk0=np.stack([b.chunk0 for b in boxes]).astype(np.float32),
               cov=cov.astype(np.float16), tok_xyz=np.asarray(tok_xyz, np.float32),
               sig_tok=(sig[0] if sig is not None else np.zeros(0)), sig_val=(sig[1] if sig is not None else np.zeros(0)))
    return meta, arr


def sandbox(set_name, shard, port, smoke=False):
    from openpi_client import websocket_client_policy as _wcp
    ix = load_index()
    items = [o for o in ix[set_name]]
    i, n = map(int, shard.split("/"))
    od = os.path.join(OUT, "sb" if not smoke else "sb_smoke", set_name)
    by = {}
    for j, o in enumerate(items):                               # observations dealt round-robin to shards, grouped by task per shard
        if smoke or j % n == i:
            by.setdefault((o["suite"], o["index"]), []).append(o)
    tasks = sorted(by)
    mine = tasks if not smoke else tasks[:1]
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
    done, t_all = 0, time.time()
    for tk in mine:
        todo = [o for o in by[tk] if not os.path.exists(os.path.join(od, o["key"] + ".npz"))]
        if smoke:
            todo = todo[:1]
        if not todo:
            continue
        SB = Sandbox(*tk)
        for o in todo:
            t0 = time.time()
            meta, arr = sandbox_obs(SB, o, client)
            save_atomic(os.path.join(od, o["key"] + ".npz"), meta=np.array(json.dumps(meta)), **arr)
            done += 1
            nb = len(meta["boxes"])
            rel = sum(b["released"] for b in meta["boxes"])
            print(f"[d5 sandbox] {set_name} {o['key']}: {len(meta['cands'])} bodies, {nb} sandboxes ({rel} released), "
                  f"{time.time() - t0:.0f}s (validity {meta['secs']['validity']:.0f}s, server {meta['secs']['server']:.0f}s); "
                  f"{done} done, {time.time() - t_all:.0f}s", flush=True)
        SB.S.close()
    print(f"[d5 sandbox] {set_name} shard {shard} finished: {done} new", flush=True)


# ── sandbox parity (LIBERO env, no policy) ───────────────────────────────────────────────────────────────────────────
def parity(n_obs=6, horizon=40):
    """(a) restore fidelity: from a recorded zs carry state (D4's dev recording, full MjSimState + the reconstructed gripper ramp),
    re-execute the RECORDED zs actions for `horizon` steps and compare the eef and the held object with the recording; (b) snapshot
    completeness: the same re-execution run interleaved with a second sandbox (restore / step / snapshot alternately) must equal the
    uninterrupted run bit for bit."""
    ix = load_index()
    obs = sorted(ix["dev"], key=lambda o: h8("d5par:" + o["key"]))[:n_obs]
    res = []
    for o in obs:
        suite, ti = o["suite"], o["index"]
        rz = np.load(os.path.join(D4D, "roll", f"d4_zs_{slug(o['tid'])}_rec", f"task{ti}_trial{o['trial']}.npz"), allow_pickle=True)
        A, S8, simrec = rz["action"], rz["state"], rz["sim"]
        names = [str(q) for q in rz["obj_names"]]
        P = rz["obj_pos"]
        j = o["j"]
        hz = min(horizon, len(A) - j - 1)
        SB = Sandbox(suite, ti)
        st = simrec[j]

        def run(interleave):
            s = SB.start(st, o["grip_cmds"])
            other = SB.start(st, o["grip_cmds"], (SB.S.bodies[0], np.array([0.03, 0.0])))
            eef, held = [], []
            for k in range(hz):
                if interleave:
                    SB.restore(other)
                    SB.e.step(np.asarray(A[j + k], np.float64).tolist())
                    other = SB.snapshot(other)
                SB.restore(s)
                SB.e.step(np.asarray(A[j + k], np.float64).tolist())
                s = SB.snapshot(s)
                _, hand = SB.obs8()
                eef.append(hand)
                held.append(SB.S.pos(o["held"]))
            SB.restore(s)
            return np.array(eef), np.array(held), SB.flat()
        e1, h1, f1 = run(False)
        e2, h2, f2 = run(True)
        rec_eef = S8[j + 1:j + 1 + hz, :3]
        rec_held = P[j + 1:j + 1 + hz, names.index(o["held"])] if o["held"] in names else None
        r = dict(key=o["key"], steps=int(hz), interleave_exact=bool(np.array_equal(f1, f2)),
                 interleave_max=float(np.abs(f1 - f2).max()),
                 eef_err_max=float(np.linalg.norm(e1 - rec_eef, axis=1).max()), eef_err_end=float(np.linalg.norm(e1[-1] - rec_eef[-1])),
                 held_err_end=(float(np.linalg.norm(h1[-1] - rec_held[-1])) if rec_held is not None else None),
                 sim_state_err_end=float(np.abs(f1[1:] - simrec[j + hz][1:]).max()))
        res.append(r)
        print("[d5 parity] " + json.dumps(r), flush=True)
        SB.S.close()
    dump(os.path.join(OUT, "parity.json"), dict(n=len(res), horizon=horizon, rows=res))


def smoke_diag():
    """Mechanics of the smoke sandboxes only (validity reasons, release / steps / success per box, timing); no readout point, J or
    bound body is computed or printed."""
    ix = load_index()
    allix = {o["key"]: o for s in ("train", "dev") for o in ix[s]}
    for f in sorted(glob.glob(os.path.join(OUT, "sb_smoke", "*", "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        o = allix[meta["key"]]
        rs = {}
        for m, v in meta["validity"].items():
            for k, (ok, why) in v.items():
                if not ok:
                    rs[f"{m}:{why}"] = rs.get(f"{m}:{why}", 0) + 1
        print(f"[d5 smoke] {meta['key']} cands {meta['cands']} invalid {rs}; secs {meta['secs']}; zs j {o.get('j')} release step "
              f"{o.get('zs_release_step')} len {o.get('zs_len')}", flush=True)
        for b in meta["boxes"]:
            print(f"   {b['label']}: released {b['released']} success {b['success']} steps {b['steps']} queries {b['queries']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for s in ("inspect", "index", "parity", "sandbox", "smoke", "smoke-diag", "read", "record"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--set", default="train")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    if a.inspect:
        inspect()
    if a.index:
        index()
    if a.parity:
        parity()
    if a.sandbox:
        sandbox(a.set, a.shard, a.port, smoke=a.smoke)
    if a.smoke_diag:
        smoke_diag()
    if a.read:
        import d5_read
        d5_read.read(a.record)


if __name__ == "__main__":
    main()
