"""PROBE D4 (docs/PLAN_D4.md; pre-registration experiments/prereg/pd4_bind.py, ledger m3_prereg tag pd4_bind): binding-operator
extraction from a frozen post-trained flow VLA.

A model has bound object Z to a role if its action FOLLOWS Z.  At an observation's full simulator state, displace ONE body Z by
+-DELTA in x and in y (collision-free, still supported: validated, invalid perturbations rejected), re-render both cameras, query the
frozen policy on every counterfactual with the SAME K noises, integrate each chunk into the gripper endpoint (demo-calibrated scale),
and read the finite-difference response J_Z = d endpoint / d pos(Z) (central differences, 2x2 xy block primary, z row stored) and
its offset b_Z = (e0 - h) - J_Z (p_Z - h) (OF D1's aim term).  The bound body = the largest ||J_Z||_F.  An unperturbed second-seed
pair (noise set B vs A) is the noise floor.  Roles: approach (pre-grasp, the object role) and place (post-grasp / carry, the goal
role: the target body).  No guidance, no closed loop; demos never execute (their states are set in the simulator only).

Stages (each its own SLURM job; nothing here runs on the login node):
  --inspect / --inspect-sim                      layout checks (no model output)
  --index-train --model m          (chunks env)  OF's pre-grasp frames (<= PRE_PER_TASK per task) + carry candidates from the demo's
                                                 gripper commands -> results/d4/index_train_<m>.json
  --fresh-check                    (chunks env)  fresh_d4 start states vs every earlier fresh set (> 1e-6)
  --launch                         (chunks env)  zs rollout packs recording full sim states (REC_SIM=1) -> results/d4/packs/submit.sh
  --extract --shard i/n            (LIBERO env)  type every dev episode (fm_types), pick R2a's pre / post observation points, save the
                                                 sim state + recorded images -> results/d4/dev/obs/, results/d4/dev/index/
  --render --set S --shard i/n     (LIBERO env)  S in train_l90c / train_libero / dev: base + counterfactual renders, validity table
                                                 -> results/d4/render/<S>/<key>.npz
  --gpu --model m --set S --shard  (openpi GPU)  the frozen policy's chunks on every render (shared noise) -> results/d4/gpu/<m>_<S>/
  --read [--record]                (chunks env)  H1-H4, tables, bank size -> results/d4/read.json, ledger d4_read
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

OUT = os.environ.get("D4_OUT") or os.path.join(ROOT, "results", "d4")      # D4_OUT: the synthetic crash test only (d4_read.synth)
OF = os.path.join(ROOT, "results", "of")
R2A_OUT = os.path.join(ROOT, "results", "r2a")
RROBOT = os.path.join(ROOT, "results", "robot")
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
MODELS = {
    "l90c": dict(cfg="pi05_l90c", ckpt="/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000"),
    "libero": dict(cfg="pi05_libero", ckpt=None),
}
# ── settings (fixed before any model output was computed; docs/PLAN_D4.md) ─────────────────────────────────────────
DELTA = 0.02                     # the displacement: +-2 cm in x and in y (table plane), one body at a time
PERTS = (("x", +1), ("x", -1), ("y", +1), ("y", -1))
PEN_TOL = 0.002                  # validity 1: Z's deepest contact penetration may grow by at most 2 mm over the base state
SETTLE_STEPS, SETTLE_TOL = 50, 0.005   # validity 2: 0.1 s of physics (robot pinned): Z ends within 5 mm of (base + delta), every
                                       # other movable body within 5 mm of where it ends in the base run (no push, no fall)
K_NOISE, SEED_A, SEED_B = 8, 11, 21    # K shared noises (set A for every counterfactual), set B for the unperturbed noise-floor pair
PRE_PER_TASK, CARRY_PER_TASK = 6, 4    # training observations per task (evenly spaced among the eligible frames)
CARRY_STRIDE, CARRY_START = 3, 5       # carry candidates: every 3rd frame from first close + 5 while the gripper stays closed
LIFT, POST_CLEAR, HELD = 0.03, 0.06, 0.10   # carry: object lifted 3 cm, hand >= 6 cm (xy) from the target, object within 10 cm of hand
FRESH_DIR, BASE, N_STATES = "fresh_d4", 8500000, 20
SERVE_SEED = 13                  # M1 / R2a zs serving noise seed
CHUNK_GROUP = 12                 # renders per policy batch (x K noises)
B_BOOT = 10000
TABLES = ("main_table", "kitchen_table", "study_table", "living_room_table", "table", "floor")
# H4 categories: one level above the BDDL type (pre-declared map; types not listed are their own category)
CATEGORY = {
    "akita_black_bowl": "bowl", "white_bowl": "bowl", "black_bowl": "bowl",
    "porcelain_mug": "mug", "red_coffee_mug": "mug", "white_yellow_mug": "mug",
    "alphabet_soup": "grocery", "tomato_sauce": "grocery", "ketchup": "grocery", "cream_cheese": "grocery", "butter": "grocery",
    "chocolate_pudding": "grocery", "milk": "grocery", "orange_juice": "grocery", "bbq_sauce": "grocery",
    "salad_dressing": "grocery", "new_salad_dressing": "grocery",
    "plate": "plate", "basket": "basket", "wooden_tray": "tray",
    "wooden_cabinet": "cabinet", "white_cabinet": "cabinet", "wooden_two_layer_shelf": "shelf",
    "flat_stove": "stove", "microwave": "microwave", "desk_caddy": "caddy", "wine_rack": "rack",
    "moka_pot": "cookware", "chefmate_8_frypan": "cookware",
    "black_book": "book", "yellow_book": "book", "cookies": "grocery", "wine_bottle": "bottle",
    "glazed_rim_porcelain_ramekin": "ramekin",
}


def slug(tid):
    return tid.replace(":", "_")


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def save_atomic(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def h8(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def body_of(name, bodies):
    """A BDDL object / region / site name -> the perturbable body that carries it (the longest body name it equals or starts
    with), or None (table regions)."""
    if name in bodies:
        return name
    c = [b for b in bodies if name.startswith(b + "_")]
    return max(c, key=len) if c else None


def type_of(inst, types):
    return types.get(inst, inst.rsplit("_", 1)[0])


# ── layout checks ──────────────────────────────────────────────────────────────────────────────────────────────────
def inspect():
    import m1_compose_create as M1
    tt = M1.task_table()
    for tid, b in M1.dev_tasks():
        f = os.path.join(R2A_OUT, "index", f"{slug(tid)}.json")
        if not os.path.exists(f):
            continue
        cnt = {}
        for e in jload(f)["episodes"]:
            cnt[e["type"]] = cnt.get(e["type"], 0) + 1
        print(f"{tid} bin {b} '{tt[tid].language}' r2a types {cnt}")
    types = set()
    for m in MODELS:
        for x in jload(os.path.join(OF, f"index_{m}.json"))["tasks"]:
            types |= set(x["objects"].values()) | set(x["fixtures"].values())
    for tid, _ in M1.dev_tasks():
        types |= set(tt[tid].objects.values()) | set(tt[tid].fixtures.values())
    print("types (training + dev scenes):", sorted(types))
    print("types without a category entry:", sorted(t for t in types if t not in CATEGORY))


def inspect_sim():
    import pathlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    for suite, i in (("libero_90", 14),):
        task = benchmark.get_benchmark_dict()[suite]().get_task(i)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        e = env.env
        print([e.sim.model.joint_id2name(j) for j in range(e.sim.model.njnt)])
        env.close()


# ── training observations (chunks env) ───────────────────────────────────────────────────────────────────────────
def index_train(model):
    """OF's pre-grasp frames (hand >= 5 cm xy from the named object; OF sel), at most PRE_PER_TASK per task evenly spaced; carry
    candidates from the demo's own gripper commands (every CARRY_STRIDE-th frame from first close + CARRY_START while the command
    stays closed); the carry frames are finalised by poses in --render (lifted, held, hand clear of the target)."""
    from lifelong_ood.policy import data as D
    import m1_compose_create as M1
    tt = M1.task_table()
    ix = jload(os.path.join(OF, f"index_{model}.json"))["tasks"]
    sel = jload(os.path.join(OF, f"sel_{model}.json"))["obs"]
    demos, d90 = D.Demos(), D.Demos90()
    out = []
    for x in ix:
        t = tt[x["tid"]]
        bodies = list(x["objects"]) + [q for q in x["fixtures"] if not any(q.startswith(tb) for tb in TABLES)]
        atoms = [a for a in t.goal if len(a) == 3 and a[0] in ("In", "On") and a[1] == x["inst"]]
        target = atoms[0][2] if atoms else None
        tbody = body_of(target, bodies) if target else None
        pre = [s["frame"] for s in sel if s["tid"] == x["tid"]]
        if len(pre) > PRE_PER_TASK:
            pre = [pre[q] for q in np.unique(np.linspace(0, len(pre) - 1, PRE_PER_TASK).round().astype(int))]
        ep = (d90 if x["ipec"] else demos).load(x["episode"])
        A = ep["actions"]
        g = int(x["first_close"])
        carry = []
        f = g + CARRY_START
        while f < len(A) - 10 and A[f, 6] > 0 and A[max(f - 1, 0), 6] > 0:
            carry.append(int(f))
            f += CARRY_STRIDE
        out.append(dict(tid=x["tid"], suite=x["suite"], index=x["index"], scene=x["scene"], language=x["language"], inst=x["inst"],
                        obj_type=x["cat"], target=target, target_body=tbody,
                        target_type=(type_of(tbody, dict(x["objects"], **x["fixtures"])) if tbody else None),
                        episode=x["episode"], ipec=x["ipec"], match=x["match"], pre=pre, carry_cand=carry,
                        objects=x["objects"], fixtures=x["fixtures"]))
        print(f"[d4 index] {model} {x['tid']} '{x['language']}': {x['inst']} -> {target} (body {tbody}); pre {len(pre)}, "
              f"carry candidates {len(carry)}", flush=True)
    dump(os.path.join(OUT, f"index_train_{model}.json"), dict(model=model, tasks=out))
    print(f"[d4 index] {model}: {len(out)} tasks, {sum(len(x['pre']) for x in out)} pre-grasp frames, "
          f"{sum(len(x['carry_cand']) for x in out)} carry candidates", flush=True)


# ── dev rollouts: fresh states, launch, extract ─────────────────────────────────────────────────────────────────────
def dev_tasks():
    """The 11 recombination dev tasks with an In / On goal (the 4 articulation-only ones have no object or goal role) + the part
    dev tasks on which R2a's zs made wrong-object errors (no recombination task has one: R2a 0 / 300), in split.json order."""
    import m1_compose_create as M1
    out = []
    for tid, b in M1.dev_tasks():
        x = jload(os.path.join(R2A_OUT, "index", f"{slug(tid)}.json"))
        own = x.get("own") or {}
        types = [e["type"] for e in x["episodes"]]
        if b != "part" and own:
            out.append((tid, b))
        elif b == "part" and "wrong_object" in types:
            out.append((tid, b))
    return out


def fresh_check():
    import m1_compose_create as M1
    import r2a_null_modes as R2A
    used = tuple(R2A.used_bases()) + (5300000, 6700000, 7100000)
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in used), BASE
    out = {}
    for tid, _ in dev_tasks():
        i = tid.split(":")[1]
        x = np.load(os.path.join(RROBOT, FRESH_DIR, f"libero_90_{i}.npy"))
        assert len(x) == N_STATES, (tid, x.shape)
        dmin, sets = np.inf, []
        for f in sorted(glob.glob(os.path.join(RROBOT, "fresh_*", f"libero_90_{i}.npy"))):
            if os.path.basename(os.path.dirname(f)) == FRESH_DIR:
                continue
            y = np.load(f)
            if y.shape[1] != x.shape[1]:
                continue
            dmin = min(dmin, float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))))
            sets.append(os.path.basename(os.path.dirname(f)))
        assert dmin > 1e-6, (tid, dmin)
        out[tid] = dict(min_dist=dmin, compared=sets)
        print(tid, f"min distance to earlier sets {dmin:.4f} over {len(sets)} sets", flush=True)
    dump(os.path.join(OUT, "fresh_check.json"), out)


def roll_path(tid):
    return os.path.join(OUT, "roll", f"d4_zs_{slug(tid)}.json")


def launch(per=4):
    import m1_compose_create as M1
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    lines = []
    for tid, _ in dev_tasks():
        suite, i = tid.split(":")
        if os.path.exists(roll_path(tid)):
            continue
        lines.append(dict(MODE="ours", SUITE=suite, TASKS=i, TRIALS=N_STATES, FIRST=0, EXPERIMENT="d4_zs", TAG="zs",
                          PROGRAM_BYTES=0, OUT=roll_path(tid), BACKBONE_CONFIG=M1.CFG, BACKBONE_CKPT=M1.CKPT, BIND="track", RECORD="1",
                          OBJECTS="1", ALL_OBJECTS="1", SERVE_SEED=str(SERVE_SEED), FRESH=os.path.join(RROBOT, FRESH_DIR), REC_SIM="1"))
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "roll"), exist_ok=True)
    cmds = []
    for j, k in enumerate(range(0, len(lines), per)):
        grp = lines[k:k + per]
        p = os.path.join(OUT, "packs", f"d4_zs_{j:03d}.tsv")
        with open(p, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{a}={b}" for a, b in sp.items()) + "\n")
        cmds.append(f"sbatch --parsable --partition=overcap --account=overcap --qos=short --requeue --time=04:00:00 "
                    f"--gres=gpu:a40:1 --exclude={EXCLUDE} --job-name=d4_zs -o slurm/d4_zs_{j:03d}_%j.out "
                    f"--export=ALL,SPECS={p} experiments/roll_pack.sbatch")
        print(f"pack {j}: " + ", ".join(s["TASKS"] for s in grp))
    sh = os.path.join(OUT, "packs", "submit.sh")
    with open(sh, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\ncd " + ROOT + "\n" + "\n".join(cmds) + "\n")
    print(f"{len(lines)} instances, {len(cmds)} packs -> bash {sh}")


def scene(tid):
    import fm_types as FT

    class Scene(FT.Scene):
        def __init__(self, t):
            super().__init__(t)
            suite, i = t.split(":")
            self.fresh = np.load(os.path.join(RROBOT, FRESH_DIR, f"{suite}_{i}.npy"))
    return Scene(tid)


def extract_task(tid, ax):
    """R2a's observation points on the new rollouts (r2a_null_modes.extract_task's rules, unchanged), plus the full sim state."""
    import fm_types as FT
    import r2a_null_modes as R2A
    f = roll_path(tid)
    if not os.path.exists(f):
        print(f"[d4 extract] {tid}: no rollout", flush=True)
        return
    z = jload(f)
    eps = [e for t in z["tasks"].values() for e in t["episodes"]]
    rd = f[:-5] + "_rec"
    sc = scene(tid)
    own = R2A.own_atoms(sc)
    text = ax[tid]["text"]
    od = os.path.join(OUT, "dev", "obs")
    os.makedirs(od, exist_ok=True)
    res = dict(task=tid, axes=ax[tid], own={k: v for k, v in own.items()}, episodes=[])
    H, REPLAN = R2A.H, R2A.REPLAN
    for e in eps:
        k = int(e["trial"])
        rp = os.path.join(rd, f"task{tid.split(':')[1]}_trial{k}.npz")
        if not os.path.exists(rp):
            res["episodes"].append(dict(trial=k, type=None, obs=[]))
            continue
        rz = np.load(rp, allow_pickle=True)
        rec = {q: rz[q] for q in ("obj_names", "obj_pos", "obj_quat", "state", "action")}
        assert "sim" in rz.files, rp
        ty = FT.type_episode(sc, ax[tid], e, rec)
        names = [str(x) for x in rec["obj_names"]]
        A, S, P, Qt = rec["action"], rec["state"], rec["obj_pos"], rec["obj_quat"]
        T = len(A)
        bstep, b = e.get("bound_step"), e.get("bound")
        obs = []
        if own and tid not in R2A.ARTIC_FIRST:
            lim = min(T - H + 1, int(bstep) if bstep is not None else min(T, 200))
            q = [j for j in range(0, max(lim, 0), REPLAN) if (j == 0 or A[j - 1, 6] <= 0)
                 and np.min(np.linalg.norm(P[j, :, :2] - S[j, None, :2], axis=-1)) >= R2A.HAND_CLEAR]
            pick = sorted({q[0], q[len(q) // 2], q[-1]}) if len(q) >= 3 else q
            for j in pick[:R2A.PRE_MAX]:
                obs.append(("pre", j, None))
        if b is not None and b in own and bstep is not None:
            i = names.index(b)
            q = []
            for j in range(int(bstep) + REPLAN, T - H + 1):
                if A[j - 1, 6] <= 0:
                    break
                if j % REPLAN:
                    continue
                if P[j, i, 2] - P[0, i, 2] < R2A.LIFT:
                    continue
                cn, cp = R2A.goal_candidates(sc, names, P[j], Qt[j], k, b, own[b])
                if not cn or np.linalg.norm(cp[0, :2] - S[j, :2]) < R2A.POST_CLEAR:
                    continue
                q.append(j)
            for j in q[:R2A.POST_MAX]:
                obs.append(("post", j, own[b][0]))
        meta = []
        for role, j, tgt in obs:
            key = f"{slug(tid)}_t{k}_{role}{j}"
            save_atomic(os.path.join(od, key + ".npz"), sim=rz["sim"][j], image=rz["image"][j], wrist=rz["wrist"][j], state=S[j],
                        text=np.array(text), role=np.array(role))
            named = sorted(own) if role == "pre" else [tgt]
            meta.append(dict(key=key, role=role, j=int(j), named=named, held=(b if role == "post" else None)))
        res["episodes"].append(dict(trial=k, success=bool(e["success"]), bound=b, bound_step=bstep, type=ty.get("type"),
                                    sub=ty.get("sub"), alt=ty.get("alt"), target=ty.get("target"), obs=meta))
        print(f"[d4 extract] {tid} trial {k}: {ty.get('type')}/{ty.get('sub')} bound {b}@{bstep} alt {ty.get('alt')}; obs "
              + " ".join(f"{m['role']}{m['j']}" for m in meta), flush=True)
    dump(os.path.join(OUT, "dev", "index", f"{slug(tid)}.json"), res)
    sc.env.close()


def extract(shard):
    import fm_types as FT
    ax = FT.axes()
    i, n = map(int, shard.split("/"))
    for j, (tid, _) in enumerate(dev_tasks()):
        if j % n == i:
            extract_task(tid, ax)


def load_dev_index():
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "dev", "index", "*.json"))):
        x = jload(f)
        out[x["task"]] = x
    return out


# ── counterfactual renders (LIBERO env, EGL) ──────────────────────────────────────────────────────────────────────
class Sim:
    """One task's LIBERO env: set a full state, perturb one body, validate, render the policy's two cameras."""

    def __init__(self, suite, index):
        import pathlib
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(index))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        self.env.seed(0)
        self.env.reset()
        self.e = self.env.env
        self.sim = self.e.sim
        m = self.sim.model
        self.movable = list(self.e.objects_dict.keys())
        self.fixtures = [n for n in self.e.fixtures_dict if not any(n.startswith(t) for t in TABLES)]
        self.bodies = self.movable + self.fixtures
        self.bid = {n: int(self.e.obj_body_id[n]) for n in self.bodies}
        par = np.asarray(m.body_parentid)

        def subtree(b):
            out = {b}
            ch = True
            while ch:
                ch = False
                for q in range(m.nbody):
                    if q not in out and int(par[q]) in out:
                        out.add(q)
                        ch = True
            return out
        self.geoms = {n: set(int(g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in subtree(self.bid[n])) for n in self.bodies}
        jn = [m.joint_id2name(j) or "" for j in range(m.njnt)]
        rj = [j for j in range(m.njnt) if jn[j].startswith(("robot0_", "gripper0_"))]
        self.rq = np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + (7 if m.jnt_type[j] == 0 else 4 if m.jnt_type[j] == 1
                                                                                  else 1)) for j in rj]).astype(int)
        self.rv = np.concatenate([np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + (6 if m.jnt_type[j] == 0 else 3 if m.jnt_type[j] == 1
                                                                                 else 1)) for j in rj]).astype(int)
        self.qadr = {n: int(m.jnt_qposadr[m.joint_name2id(self.e.objects_dict[n].joints[0])]) for n in self.movable}
        self.body_pos0 = np.array(m.body_pos, np.float64).copy()

    def close(self):
        self.env.close()

    def set(self, state):
        self.sim.model.body_pos[:] = self.body_pos0                     # undo a fixture displacement before the state is set
        return self.env.set_init_state(state)

    def pos(self, n):
        return np.array(self.sim.data.body_xpos[self.bid[n]], np.float64)

    def yaw(self, n):
        w, x, y, z = self.sim.data.body_xquat[self.bid[n]]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def displace(self, n, d):
        """Move body n by d (xy) in the current state (free joint for movable objects, model body_pos for fixtures)."""
        if n in self.qadr:
            a = self.qadr[n]
            self.sim.data.qpos[a:a + 2] += d
        else:
            self.sim.model.body_pos[self.bid[n], :2] += d
        self.sim.forward()

    def penetration(self, n):
        d = self.sim.data
        g = self.geoms[n]
        worst = 0.0
        for c in range(d.ncon):
            ct = d.contact[c]
            if int(ct.geom1) in g or int(ct.geom2) in g:
                worst = max(worst, -float(ct.dist))
        return worst

    def settle(self):
        """SETTLE_STEPS of physics from the current state with zero velocities and the robot pinned -> movable body positions."""
        d = self.sim.data
        q0, v0 = d.qpos[self.rq].copy(), np.zeros(len(self.rv))
        d.qvel[:] = 0.0
        for _ in range(SETTLE_STEPS):
            self.sim.step()
            d.qpos[self.rq] = q0
            d.qvel[self.rv] = v0
        self.sim.forward()
        return {n: self.pos(n) for n in self.movable}

    def render(self):
        from openpi_client import image_tools
        im = np.ascontiguousarray(self.sim.render(camera_name="agentview", width=256, height=256)[::-1, ::-1])
        wr = np.ascontiguousarray(self.sim.render(camera_name="robot0_eye_in_hand", width=256, height=256)[::-1, ::-1])
        return (image_tools.convert_to_uint8(image_tools.resize_with_pad(im, 224, 224)),
                image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, 224, 224)))


def state8(ob):
    import libero_client as LC
    return np.concatenate((ob["robot0_eef_pos"], LC.quat2axisangle(ob["robot0_eef_quat"]), ob["robot0_gripper_qpos"])).astype(np.float32)


def counterfactuals(S, state, cands):
    """At `state`: the base render, then each candidate x perturbation: validity (penetration, settle) and, if valid, its render.
    -> dict of arrays."""
    ob = S.set(state)
    st8 = state8(ob)
    base_img, base_wr = S.render()
    pen0 = {n: S.penetration(n) for n in cands}
    pos0 = {n: S.pos(n) for n in S.bodies}
    yaw0 = {n: S.yaw(n) for n in S.bodies}
    rest0 = S.settle()
    imgs, wrs, meta = [base_img], [base_wr], []
    for n in cands:
        for ax, sg in PERTS:
            d = np.zeros(2)
            d["xy".index(ax)] = sg * DELTA
            S.set(state)
            S.displace(n, d)
            pen = S.penetration(n)
            img = wr = None
            ok_pen = pen <= pen0[n] + PEN_TOL
            if ok_pen:
                img, wr = S.render()
            rest = S.settle()
            exp = {q: rest0[q] + (np.r_[d, 0.0] if q == n else 0.0) for q in S.movable}
            err = {q: float(np.linalg.norm(rest[q] - exp[q])) for q in S.movable}
            worst = max(err.values()) if err else 0.0
            ok = bool(ok_pen and worst <= SETTLE_TOL)
            meta.append(dict(cand=n, axis=ax, sign=sg, valid=ok, pen=round(pen, 5), pen0=round(pen0[n], 5), settle_err=round(worst, 5),
                             reason=("" if ok else "penetration" if not ok_pen else "settle:" + max(err, key=err.get)),
                             img=(len(imgs) if ok else -1)))
            if ok:
                imgs.append(img)
                wrs.append(wr)
    S.set(state)
    return dict(state8=st8, images=np.stack(imgs), wrists=np.stack(wrs), meta=meta,
                pos={n: pos0[n].tolist() for n in S.bodies}, yaw={n: yaw0[n] for n in S.bodies}, eef=np.asarray(ob["robot0_eef_pos"]))


def render_items(set_name):
    """-> [(suite, index, key, state-loader info, role, filler, held, extra)] for a render set."""
    items = []
    if set_name.startswith("train_"):
        model = set_name.split("_", 1)[1]
        for x in jload(os.path.join(OUT, f"index_train_{model}.json"))["tasks"]:
            for f in x["pre"]:
                items.append(dict(suite=x["suite"], index=x["index"], tid=x["tid"], key=f"{slug(x['tid'])}_e{x['episode']}_pre{f}",
                                  frame=f, role="approach", filler=x["inst"], held=None, match=x["match"], x=x))
            if x["target_body"]:
                for f in x["carry_cand"]:
                    items.append(dict(suite=x["suite"], index=x["index"], tid=x["tid"],
                                      key=f"{slug(x['tid'])}_e{x['episode']}_carry{f}", frame=f, role="place", filler=x["target_body"],
                                      held=x["inst"], match=x["match"], x=x))
    else:
        for tid, x in load_dev_index().items():
            suite, ti = tid.split(":")
            for e in x["episodes"]:
                for m in e["obs"]:
                    items.append(dict(suite=suite, index=int(ti), tid=tid, key=m["key"], role="approach" if m["role"] == "pre" else "place",
                                      named=m["named"], held=m.get("held"), trial=e["trial"], ep=e))
    return items


def render(set_name, shard):
    i, n = map(int, shard.split("/"))
    smoke = n == 0
    od = os.path.join(OUT, "render", set_name if not smoke else f"smoke_{set_name}")
    items = render_items(set_name)
    by_task = {}
    for it in items:
        by_task.setdefault((it["suite"], it["index"]), []).append(it)
    tasks = sorted(by_task)
    if smoke:
        tasks = tasks[:1]
    done, t_all = 0, time.time()
    report = []
    for j, tk in enumerate(tasks):
        if not smoke and j % n != i:
            continue
        todo = [it for it in by_task[tk] if not os.path.exists(os.path.join(od, it["key"] + ".npz"))]
        if smoke:
            todo = todo[:2]
        if not todo:
            continue
        S = Sim(*tk)
        carry_kept = {}
        if set_name.startswith("train_"):
            mz = np.load(os.path.join(ROOT, todo[0]["match"]), allow_pickle=True)
            states = mz["states"]
            S.set(states[0])
            z0 = {q: S.pos(q)[2] for q in S.movable}
        for it in todo:
            t0 = time.time()
            if set_name.startswith("train_"):
                st = states[it["frame"]]
                if it["role"] == "place":
                    # carry finalisation by poses: lifted, held, the hand clear of the target; at most CARRY_PER_TASK, the earliest
                    ob = S.set(st)
                    h = np.asarray(ob["robot0_eef_pos"], np.float64)
                    p = S.pos(it["held"])
                    tp = S.pos(it["filler"])
                    ok = (p[2] - z0[it["held"]] >= LIFT and np.linalg.norm(p - h) <= HELD and np.linalg.norm(tp[:2] - h[:2]) >= POST_CLEAR)
                    cnt = carry_kept.setdefault(it["tid"], 0)
                    if not ok or cnt >= CARRY_PER_TASK:
                        continue
                    carry_kept[it["tid"]] = cnt + 1
                named = [it["filler"]]
                rec_img = None
            else:
                z = np.load(os.path.join(OUT, "dev", "obs", it["key"] + ".npz"), allow_pickle=True)
                st = z["sim"]
                rec_img = z["image"]
                named = [body_of(q, S.bodies) for q in it["named"]]
            cands = [q for q in S.bodies if q != it["held"]]
            cf = counterfactuals(S, st, cands)
            chk = {}
            if rec_img is not None:
                d = np.abs(cf["images"][0].astype(np.float32) - rec_img.astype(np.float32))
                chk = dict(render_mae=float(d.mean()), render_exact=bool(d.max() == 0),
                           state_err=float(np.abs(cf["state8"] - z["state"]).max()))
            meta = dict(key=it["key"], tid=it["tid"], role=it["role"], named=named, held=it["held"], cands=cands, cf=cf["meta"],
                        pos=cf["pos"], yaw=cf["yaw"], eef=[float(v) for v in cf["eef"]], check=chk, **({"filler": it["filler"], "frame": it["frame"]}
                                                                     if set_name.startswith("train_") else {"trial": it["trial"]}))
            save_atomic(os.path.join(od, it["key"] + ".npz"), images=cf["images"], wrists=cf["wrists"], state8=cf["state8"], eef=cf["eef"],
                        meta=np.array(json.dumps(meta)))
            nv = sum(c["valid"] for c in cf["meta"])
            report.append(dict(key=it["key"], n_cf=len(cf["meta"]), valid=nv, **chk))
            if smoke:
                print("[d4 render] smoke: pen / pen0 / settle_err per perturbation " +
                      json.dumps([(c["cand"], c["axis"], c["sign"], c["pen"], c["pen0"], c["settle_err"]) for c in cf["meta"]]), flush=True)
                mv = [q for q in S.movable if q != it["held"]]
                if len(mv) >= 2:                                         # self-test: a body dropped onto another must be rejected
                    a_, b_ = mv[0], mv[1]
                    S.set(st)
                    pen0a = S.penetration(a_)
                    rest0 = S.settle()
                    S.set(st)
                    S.displace(a_, S.pos(b_)[:2] - S.pos(a_)[:2])
                    pen = S.penetration(a_)
                    rest = S.settle()
                    print(f"[d4 render] self-test: {a_} moved onto {b_}: penetration {pen:.4f} (base {pen0a:.4f}), "
                          f"{b_} settle shift {np.linalg.norm(rest[b_] - rest0[b_]):.4f} m", flush=True)
            done += 1
            if smoke or done <= 3 or done % 25 == 0:
                print(f"[d4 render] {set_name} {it['key']}: {len(cands)} bodies, {nv}/{len(cf['meta'])} valid "
                      f"({ {c['reason'] for c in cf['meta'] if not c['valid']} }); check {chk}; {time.time() - t0:.1f}s "
                      f"({done} done, {time.time() - t_all:.0f}s)", flush=True)
        S.close()
    dump(os.path.join(od, f"_report_{i}of{n}.json"), report)
    print(f"[d4 render] {set_name} shard {shard} finished: {done} new", flush=True)


# ── the frozen policy on every render (openpi GPU) ──────────────────────────────────────────────────────────────────
def gpu(model, set_name, shard):
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    M = MODELS[model]
    i, n = map(int, shard.split("/"))
    smoke = n == 0
    src = os.path.join(OUT, "render", set_name if not smoke else f"smoke_{set_name}")
    od = os.path.join(OUT, "gpu", f"{model}_{set_name}" if not smoke else f"smoke_{model}_{set_name}")
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    mine = files if smoke else files[i::n]
    bk = PolicyBackbone(M["cfg"], checkpoint_dir=M["ckpt"], device="cuda")
    print(f"[d4 gpu] {bk.describe()}; {len(mine)} observations, shard {shard}", flush=True)

    def noise(seed):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn((K_NOISE, bk.dim_full), generator=g).numpy()
    nA, nB = noise(SEED_A), noise(SEED_B)
    done, t_all = 0, time.time()
    texts = {}
    for f in mine:
        key = os.path.basename(f)[:-4]
        outp = os.path.join(od, key + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if meta["tid"] not in texts:
            import m1_compose_create as M1
            texts[meta["tid"]] = M1.task_table()[meta["tid"]].language
        text = texts[meta["tid"]]
        imgs, wrs, st = z["images"], z["wrists"], np.asarray(z["state8"], np.float32)
        raw = []
        with torch.no_grad():
            for g0 in range(0, len(imgs), CHUNK_GROUP):
                ods = [{"observation/image": imgs[q], "observation/wrist_image": wrs[q], "observation/state": st, "prompt": text}
                       for q in range(g0, min(g0 + CHUNK_GROUP, len(imgs)))]
                pre = bk.prefix(bk.observations(ods))
                pr = bk.repeat(pre, K_NOISE)
                X = bk.sample(len(ods) * K_NOISE, pr, noise=np.tile(nA, (len(ods), 1)))
                raw.append(bk.unnormalize(X).reshape(len(ods), K_NOISE, bk.H, -1))
            pre0 = bk.prefix(bk.observation({"observation/image": imgs[0], "observation/wrist_image": wrs[0], "observation/state": st,
                                             "prompt": text}))
            XB = bk.sample(K_NOISE, pre0, noise=nB)
            rawB = bk.unnormalize(XB).reshape(K_NOISE, bk.H, -1)
            if smoke or done == 0:
                XA1 = bk.sample(K_NOISE, pre0, seed=SEED_A)              # parity: the batched path = the single-observation sampler
                par = float(np.abs(bk.unnormalize(XA1).reshape(K_NOISE, bk.H, -1) - raw[0][0]).max())
                print(f"[d4 gpu] parity batched vs single (seed {SEED_A}): max |diff| {par:.2e}", flush=True)
        raw = np.concatenate(raw)
        save_atomic(outp, raw=raw.astype(np.float32), rawB=rawB.astype(np.float32), meta=np.array(json.dumps(meta)))
        done += 1
        if smoke or done <= 3 or done % 25 == 0:
            print(f"[d4 gpu] {key}: {len(imgs)} renders x {K_NOISE} noises, {time.time() - t0:.1f}s ({done} done, "
                  f"{time.time() - t_all:.0f}s)", flush=True)
    print(f"[d4 gpu] shard {shard} finished: {done} new ({time.time() - t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for s in ("inspect", "inspect-sim", "index-train", "fresh-check", "launch", "extract", "render", "gpu", "read", "record", "synth"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--model", default="l90c")
    ap.add_argument("--set", default="train_l90c")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.inspect:
        inspect()
    if a.inspect_sim:
        inspect_sim()
    if a.index_train:
        index_train(a.model)
    if a.fresh_check:
        fresh_check()
    if a.launch:
        launch()
    if a.extract:
        extract(a.shard)
    if a.render:
        render(a.set, a.shard)
    if a.gpu:
        gpu(a.model, a.set, a.shard)
    if a.synth:
        import d4_read
        d4_read.synth()
    if a.read:
        import d4_read
        d4_read.read(a.record)


if __name__ == "__main__":
    main()
