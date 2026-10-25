"""PROBE R2a (docs/PLAN_R2A.md; pre-registration experiments/prereg/pr2a_null_modes.py, ledger m3_prereg tag pr2a_null_modes):
can the frozen VLA's own action modes recover binding without a demo?

Gradient-free, no demo, no weight change.  At an observation o, ascend the NULL-prompt action field of pi05_l90c@5000 from many
starts at several flow levels u, cluster the Tweedie modes into the scene's afforded options, and rank the options by how much more
the full instruction supports them than the null prompt: the paired-noise conditional flow-matching loss difference
    Delta(a) = L(a | o, null) - L(a | o, instr),   L(a | o, c) = mean_{t in T, k <= K} || v_c((1 - t) a + t eps_k, t) - (eps_k - a) ||^2
(the Diffusion-Classifier estimator of log p(a | o, instr) - log p(a | o, null) up to the ELBO's t-weighting; shared noise across
conditions and candidates).  Offline, on observations saved DURING new zero-shot rollouts (LIBERO episodes are not reproducible
from an init state).

Stages (each its own job; nothing here runs on the login node):
  --fresh-check                 (chunks env, CPU) the new start states are > 1e-6 from every earlier libero_90 fresh set
  --launch [--dry]              (chunks env, CPU) writes the zs rollout packs + results/r2a/packs/submit.sh (run from the login shell)
  --extract --shard i/n         (LIBERO env, CPU) type every episode (fm_types' typer), pick pre-grasp / post-grasp query observations,
                                save them (images + state + scene poses) -> results/r2a/obs/*.npz, results/r2a/index/<task>.json
  --probe TASK TRIAL            (LIBERO env) hand check of the object-assignment readout on one episode's observations (+ PNGs)
  --validate                    (chunks env) the readout's validity on the policy's EXECUTED 10-step windows (no mode, no ratio)
  --modes --shard i/n           (openpi env, GPU) per observation: null / instr drafts, mode ascent under null, instr and CAG fields
                                from shared starts, the flow-matching losses of every chunk under null and instr -> results/r2a/modes/
  --read [--record]             (chunks env) coverage and rank-1 tables, H1-H3, ledger r2a_read

Settings are fixed here, before any mode was computed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
import m1_compose_create as M1                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "r2a")
R = M1.R
CFG, CKPT = M1.CFG, M1.CKPT
FRESH_DIR, BASE, N_STATES = "fresh_r2a", 7100000, 20                    # new: no logged base within 1000 (fresh-check)
SEED = 13                                                               # M1's serving noise seed
EXCLUDE = M1.EXCLUDE
REPLAN, H = 5, 10
# observation selection (fixed before any rollout)
PRE_MAX, POST_MAX = 3, 1
HAND_CLEAR = 0.05                                                       # pre-grasp: the hand >= 5 cm (xy) from every movable object
POST_CLEAR, LIFT = 0.06, 0.03                                           # post-grasp: >= 6 cm (xy) from the own target, object lifted 3 cm
ARTIC_FIRST = ("libero_90:1", "libero_90:45")                           # first step articulates a fixture: no object-role observation
# the readout (fixed before validation)
D_MIN = 0.02                                                            # a chunk moving < 2 cm (xy) points at nothing
CORRIDOR = 0.05                                                         # the heading's lateral corridor (set after the hand check)
MERGE = 0.03                                                            # goal candidates within 3 cm (xy) of each other are one option
SCALE = (0.01052401745032942, 0.011963828742732078, 0.011237455538931119)   # pp2_dev_l90c: metres per action unit (demo-calibrated)
# modes (fixed before any mode was computed)
LEVELS = (0.1, 0.3, 0.55)                                               # null drafts noised to these levels, then ascended there
U_NOISE = 0.7                                                           # + uninformed starts 0.7 * eps ascended at u 0.7
N_DRAFT, N_NOISE = 12, 12
ITERS, LR = 80, 0.02                                                    # pp_fields' ascent (Adam, Tweedie x0)
CAG_W = 2.0                                                             # CAG-VA Eq. 4: v_null + w (v_instr - v_null)
T_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
K_NOISE, SCORE_SEED = 8, 5
TOL = 0.15                                                              # greedy rms clustering of modes (pp2's coarse tolerance)
FIELDS = ("null", "instr", "cag")


def slug(tid):
    return tid.replace(":", "_")


def save_atomic(path, **arrs):
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def roll_path(tid):
    return os.path.join(OUT, "roll", f"r2a_zs_{slug(tid)}.json")


# ── the readout: which candidate a chunk heads to ───────────────────────────────────────────────────────────────
def pointed(raw, eef, cand_pos, scale=SCALE, d_min=D_MIN):
    """raw [H, 7] unnormalised chunk, eef [3], cand_pos [n, 3] -> (index or -1, cos [n], displacement [2]).
    The integrated end-effector path (eef + cumsum(scale * dpos)); the candidate whose direction from the hand best matches the
    path's horizontal displacement (cosine), as in protos.store.interpret; -1 when the path moves less than d_min (xy)."""
    raw = np.asarray(raw, np.float64)
    disp = (np.cumsum(raw[:, :3] * np.asarray(scale)[None], 0)[-1])[:2]
    vec = np.asarray(cand_pos, np.float64)[:, :2] - np.asarray(eef, np.float64)[None, :2]
    cos = (vec @ disp) / np.maximum(np.linalg.norm(vec, axis=1) * np.linalg.norm(disp), 1e-12)
    if np.linalg.norm(disp) < d_min:
        return -1, cos, disp
    # the first candidate the heading reaches: in front of the hand and within CORRIDOR (xy) of the heading's ray, the nearest along
    # it (hand check 2026-10-24, libero_90:14 trial 13: max-cosine picked the cabinet BEHIND the plate on the same heading)
    u = disp / np.linalg.norm(disp)
    along = vec @ u
    perp = np.abs(vec[:, 0] * u[1] - vec[:, 1] * u[0])
    inside = np.flatnonzero((along > 0) & (perp < CORRIDOR))
    if len(inside):
        return int(inside[np.argmin(along[inside])]), cos, disp
    return int(np.argmax(cos)), cos, disp


def clusters(X, tol=TOL):
    """Greedy rms clustering of rows X [n, dim] in the given order -> labels [n] (cluster of the first row within tol of a centre)."""
    centres, lab = [], []
    for x in X:
        d = [float(np.sqrt(((x - c) ** 2).mean())) for c in centres]
        if d and min(d) < tol:
            lab.append(int(np.argmin(d)))
        else:
            centres.append(x)
            lab.append(len(centres) - 1)
    return np.asarray(lab)


# ── stage 0: start states, rollouts ─────────────────────────────────────────────────────────────────────────────
def used_bases():
    return tuple(M1.USED_BASES) + (M1.BASE, 6100000, 6300000)          # + L4 eval / practice (experiments/l4_store.py)


def fresh_check():
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in used_bases()), BASE
    out = {}
    for tid, _ in M1.dev_tasks():
        i = tid.split(":")[1]
        x = np.load(os.path.join(R, FRESH_DIR, f"libero_90_{i}.npy"))
        assert len(x) == N_STATES, (tid, x.shape)
        dmin, sets = np.inf, []
        for f in sorted(glob.glob(os.path.join(R, "fresh_*", f"libero_90_{i}.npy"))):
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


def launch(dry, per=4):
    """zs on the 29 dev tasks, 20 fresh_r2a states, noise seed 13, recording images + states + every movable object's pose + the
    bound object.  Writes the pack TSVs and a submit script (sbatch from the login shell: no parent SLURM_* variables)."""
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    lines = []
    for tid, _ in M1.dev_tasks():
        suite, i = tid.split(":")
        if os.path.exists(roll_path(tid)):
            continue
        lines.append(dict(MODE="ours", SUITE=suite, TASKS=i, TRIALS=N_STATES, FIRST=0, EXPERIMENT="r2a_zs", TAG="zs",
                          PROGRAM_BYTES=0, OUT=roll_path(tid), BACKBONE_CONFIG=CFG, BACKBONE_CKPT=CKPT, BIND="track", RECORD="1",
                          OBJECTS="1", ALL_OBJECTS="1", SERVE_SEED=str(SEED), FRESH=os.path.join(R, FRESH_DIR)))
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "roll"), exist_ok=True)
    cmds = []
    for j, k in enumerate(range(0, len(lines), per)):
        grp = lines[k:k + per]
        p = os.path.join(OUT, "packs", f"r2a_zs_{j:03d}.tsv")
        with open(p, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{a}={b}" for a, b in sp.items()) + "\n")
        cmds.append(f"sbatch --parsable --partition=overcap --account=overcap --qos=short --requeue --time=04:00:00 "
                    f"--gres=gpu:a40:1 --exclude={EXCLUDE} --job-name=r2a_zs --export=ALL,SPECS={p} experiments/roll_pack.sbatch")
        print(f"pack {j}: " + ", ".join(s["TASKS"] for s in grp))
    sh = os.path.join(OUT, "packs", "submit.sh")
    with open(sh, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\ncd " + ROOT + "\n" + "\n".join(cmds) + "\n")
    print(f"{len(lines)} instances, {len(cmds)} packs -> bash {sh}" + (" (dry)" if dry else ""))


# ── stage 1: type episodes, pick observations (LIBERO env) ──────────────────────────────────────────────────────
def scene(tid):
    import fm_types as FT

    class Scene(FT.Scene):
        def __init__(self, t):
            super().__init__(t)
            suite, i = t.split(":")
            self.fresh = np.load(os.path.join(R, FRESH_DIR, f"{suite}_{i}.npy"))
    return Scene(tid)


def own_atoms(sc):
    own = {}
    for a in sc.goal:
        if len(a) == 3 and a[0] in ("in", "on"):
            own.setdefault(a[1], []).append(a[2])
    return own


def goal_candidates(sc, names, P, Q, k, b, own_t):
    """Post-grasp goal options at one step: the own target(s) first, then every other BDDL region / object a placement could land in
    (fm_types.Scene.alternatives), positions with every movable object at its recorded pose; options within MERGE (xy) of an
    earlier one are merged into it."""
    sc.set(k, names, P, Q)
    cand = list(own_t) + [n for n in sc.alternatives(b, set(own_t)) if n not in own_t]
    pos, keep = [], []
    for n in cand:
        try:
            p = sc.pos_of(n)
        except Exception:                                               # noqa: BLE001  (a state without a geom position)
            continue
        if any(np.linalg.norm(p[:2] - q[:2]) < MERGE for q in pos):
            continue
        keep.append(n)
        pos.append(p)
    return keep, np.asarray(pos)


def extract_task(tid, ax):
    import fm_types as FT
    f = roll_path(tid)
    if not os.path.exists(f):
        print(f"[r2a extract] {tid}: no rollout", flush=True)
        return
    z = json.load(open(f, encoding="utf-8"))
    eps = [e for t in z["tasks"].values() for e in t["episodes"]]
    rd = f[:-5] + "_rec"
    sc = scene(tid)
    own = own_atoms(sc)
    text = ax[tid]["text"]
    od = os.path.join(OUT, "obs")
    os.makedirs(od, exist_ok=True)
    res = dict(task=tid, axes=ax[tid], own={k: v for k, v in own.items()}, episodes=[])
    for e in eps:
        k = int(e["trial"])
        rp = os.path.join(rd, f"task{tid.split(':')[1]}_trial{k}.npz")
        if not os.path.exists(rp):
            res["episodes"].append(dict(trial=k, type=None, obs=[]))
            continue
        rz = np.load(rp, allow_pickle=True)
        rec = {q: rz[q] for q in ("obj_names", "obj_pos", "obj_quat", "state", "action")}
        ty = FT.type_episode(sc, ax[tid], e, rec)
        names = [str(x) for x in rec["obj_names"]]
        A, S, P, Qt = rec["action"], rec["state"], rec["obj_pos"], rec["obj_quat"]
        T = len(A)
        bstep = e.get("bound_step")
        b = e.get("bound")
        obs = []
        # pre-grasp: queries before the first bind, gripper open, the hand clear of every movable object, a full executed window
        if own and tid not in ARTIC_FIRST:
            lim = min(T - H + 1, int(bstep) if bstep is not None else min(T, 200))
            q = [j for j in range(0, max(lim, 0), REPLAN) if (j == 0 or A[j - 1, 6] <= 0)
                 and np.min(np.linalg.norm(P[j, :, :2] - S[j, None, :2], axis=-1)) >= HAND_CLEAR]
            pick = sorted({q[0], q[len(q) // 2], q[-1]}) if len(q) >= 3 else q
            for j in pick[:PRE_MAX]:
                obs.append(("pre", j, names, P[j], None))
        # post-grasp: the named object bound, still held, lifted, the hand clear of the own target
        if b is not None and b in own and bstep is not None:
            i = names.index(b)
            q = []
            for j in range(int(bstep) + REPLAN, T - H + 1):
                if A[j - 1, 6] <= 0:
                    break                                               # released: the goal phase is over
                if j % REPLAN:
                    continue
                if P[j, i, 2] - P[0, i, 2] < LIFT:
                    continue
                cn, cp = goal_candidates(sc, names, P[j], Qt[j], k, b, own[b])
                if not cn or np.linalg.norm(cp[0, :2] - S[j, :2]) < POST_CLEAR:
                    continue
                q.append((j, cn, cp))
            pick = q[:1]                                                # the first eligible post-grasp query
            for j, cn, cp in pick[:POST_MAX]:
                obs.append(("post", j, cn, cp, own[b][0]))
        meta = []
        for role, j, cn, cp, tgt in obs:
            key = f"{slug(tid)}_t{k}_{role}{j}"
            win = A[j:j + H]
            pi, cos, disp = pointed(win, S[j, :3], cp)
            save_atomic(os.path.join(od, key + ".npz"), image=rz["image"][j], wrist=rz["wrist"][j], state=S[j], text=np.array(text),
                        role=np.array(role), cand_names=np.array(cn), cand_pos=np.asarray(cp, np.float32), exec=win)
            named = sorted(own) if role == "pre" else [tgt]
            meta.append(dict(key=key, role=role, j=int(j), cand=list(cn), named=named, exec_pointed=(cn[pi] if pi >= 0 else None),
                             exec_cos=[round(float(c), 3) for c in cos], exec_disp=[round(float(d), 4) for d in disp]))
        res["episodes"].append(dict(trial=k, success=bool(e["success"]), bound=b, bound_step=bstep, type=ty.get("type"),
                                    sub=ty.get("sub"), obs=meta))
        print(f"[r2a extract] {tid} trial {k}: {ty.get('type')} bound {b}@{bstep}; obs " +
              " ".join(f"{m['role']}{m['j']}->{m['exec_pointed']}" for m in meta), flush=True)
    dump(os.path.join(OUT, "index", f"{slug(tid)}.json"), res)
    sc.env.close()


def extract(shard, tasks="all"):
    import fm_types as FT
    ax = FT.axes()
    i, n = map(int, shard.split("/"))
    for j, (tid, _) in enumerate(M1.dev_tasks()):
        if j % n == i and (tasks == "all" or tid in tasks.split("+")):
            extract_task(tid, ax)


def probe(tid, trial):
    """Print the readout of every saved observation of one episode, and write its images as PNGs (results/r2a/scratch)."""
    import imageio
    ix = json.load(open(os.path.join(OUT, "index", f"{slug(tid)}.json"), encoding="utf-8"))
    e = next(x for x in ix["episodes"] if x["trial"] == int(trial))
    print(f"[probe] {tid} '{ix['axes']['text']}' trial {trial}: success {e['success']} type {e['type']} bound {e['bound']}@"
          f"{e['bound_step']}")
    os.makedirs(os.path.join(OUT, "scratch"), exist_ok=True)
    for m in e["obs"]:
        z = np.load(os.path.join(OUT, "obs", m["key"] + ".npz"), allow_pickle=True)
        print(f"[probe]  {m['role']} step {m['j']}: eef {np.round(z['state'][:3], 3).tolist()} grip {np.round(z['state'][6:], 3).tolist()}"
              f" exec disp {m['exec_disp']} -> {m['exec_pointed']} (named {m['named']})")
        for n, p, c in zip(z["cand_names"], z["cand_pos"], m["exec_cos"]):
            print(f"[probe]     {str(n):40s} {np.round(p, 3).tolist()} cos {c:+.3f}")
        imageio.imwrite(os.path.join(OUT, "scratch", m["key"] + ".png"), np.concatenate([z["image"], z["wrist"]], 1))


def load_index():
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "index", "*.json"))):
        x = json.load(open(f, encoding="utf-8"))
        out[x["task"]] = x
    return out


def validate():
    """The readout's validity, from the rollouts alone: on pre-grasp observations of episodes that bound an object, does the
    policy's EXECUTED 10-step window point at the object it then bound?  On post-grasp observations of successful episodes, at
    the own target?  Against chance (1 / number of candidates)."""
    ix = load_index()
    rows = {"pre": [], "post": []}
    for tid, x in ix.items():
        for e in x["episodes"]:
            for m in e["obs"]:
                if m["role"] == "pre" and e["bound"]:
                    rows["pre"].append((tid, m["exec_pointed"] == e["bound"], 1.0 / len(m["cand"]), m["exec_pointed"] is None))
                if m["role"] == "post" and e["success"]:
                    rows["post"].append((tid, m["exec_pointed"] == m["named"][0], 1.0 / len(m["cand"]), m["exec_pointed"] is None))
    out = {}
    for r, v in rows.items():
        if not v:
            continue
        acc = float(np.mean([a for _, a, _, _ in v]))
        ch = float(np.mean([c for _, _, c, _ in v]))
        per = {}
        for t, a, _, _ in v:
            per.setdefault(t, []).append(a)
        out[r] = dict(n=len(v), tasks=len(per), accuracy=acc, chance=ch, none=float(np.mean([z for *_, z in v])),
                      per_task={t: round(float(np.mean(a)), 3) for t, a in per.items()})
        print(f"[validate] {r}: executed-window readout accuracy {acc:.3f} (chance {ch:.3f}, none {out[r]['none']:.3f}) over {len(v)}"
              f" observations, {len(per)} tasks")
        print("[validate]   per task " + json.dumps(out[r]["per_task"]))
    counts = {}
    for tid, x in ix.items():
        for e in x["episodes"]:
            counts[str(e["type"])] = counts.get(str(e["type"]), 0) + 1
    out["episode_types"] = counts
    out["n_obs"] = {r: int(sum(len([m for m in e["obs"] if m["role"] == r]) for x in ix.values() for e in x["episodes"]))
                    for r in ("pre", "post")}
    print("[validate] episode types", counts, "observations", out["n_obs"])
    dump(os.path.join(OUT, "validate.json"), out)
    return out


# ── stage 2: modes and losses (openpi env, GPU) ─────────────────────────────────────────────────────────────────
class Field:
    """v_null + w (v_instr - v_null) as an eps field for score_ops.mode_ascent (w 0: null, 1: instr, CAG_W: CAG)."""
    path = "flow"

    def __init__(self, bk, pre_n, pre_i, w):
        self.bk, self.pre_n, self.pre_i, self.w = bk, pre_n, pre_i, float(w)
        self.device, self.dim = bk.device, bk.dim

    def eps(self, X, u, **_):
        if self.w == 0.0:
            return self.bk.eps(X, u, prefix=self.pre_n)
        if self.w == 1.0:
            return self.bk.eps(X, u, prefix=self.pre_i)
        en = self.bk.eps(X, u, prefix=self.pre_n)
        return en + self.w * (self.bk.eps(X, u, prefix=self.pre_i) - en)


def fm_losses(bk, pre, A, E, batch=256):
    """Flow-matching (velocity) loss of every chunk A [m, dim] at every t of T_GRID with the shared noises E [K, dim].
    -> [len(T_GRID), m] float32 (mean over the K noises and the dims)."""
    import torch
    A_t = torch.as_tensor(A, dtype=torch.float32, device=bk.device)
    E_t = torch.as_tensor(E, dtype=torch.float32, device=bk.device)
    m, K = len(A_t), len(E_t)
    out = np.zeros((len(T_GRID), m), np.float32)
    tgt = (E_t[None] - A_t[:, None]).reshape(m * K, -1)
    with torch.no_grad():
        for ti, t in enumerate(T_GRID):
            X = ((1 - t) * A_t[:, None] + t * E_t[None]).reshape(m * K, -1)
            L = torch.empty(m * K, device=bk.device)
            for s in range(0, m * K, batch):
                v = bk.v(X[s:s + batch], float(t), prefix=pre)
                L[s:s + batch] = ((v - tgt[s:s + batch]) ** 2).mean(-1)
            out[ti] = L.reshape(m, K).mean(1).float().cpu().numpy()
    return out


def modes(shard):
    import torch
    from lifelong_ood.backbones.score_ops import mode_ascent
    from lifelong_ood.policy.backbone import PolicyBackbone
    keys = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(OUT, "obs", "*.npz")))
    if os.environ.get("R2A_PREFIX"):                                    # a late task only (libero_90:16 rolled last)
        keys = [k for k in keys if k.startswith(os.environ["R2A_PREFIX"])]
    i, n = map(int, shard.split("/"))
    mine = keys[i::n]
    md = os.path.join(OUT, "modes")
    if os.environ.get("R2A_SMOKE"):                                     # pipeline smoke: 2 observations, never read
        mine, md = mine[:2], os.path.join(OUT, "scratch", "smoke_modes")
    os.makedirs(md, exist_ok=True)
    bk = PolicyBackbone(CFG, checkpoint_dir=CKPT, device="cuda")
    print(f"[r2a modes] {bk.describe()}; {len(mine)} observations in shard {shard}", flush=True)
    E = np.random.default_rng(SCORE_SEED).standard_normal((K_NOISE, bk.dim)).astype(np.float32)
    t_all, done = time.time(), 0
    for key in mine:
        outp = os.path.join(md, key + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        z = np.load(os.path.join(OUT, "obs", key + ".npz"), allow_pickle=True)
        od = {"observation/image": z["image"], "observation/wrist_image": z["wrist"],
              "observation/state": np.asarray(z["state"], np.float32), "prompt": str(z["text"])}
        with torch.no_grad():
            pre_i = bk.prefix(bk.observation(od))
            pre_n = bk.prefix(bk.observation(dict(od, prompt="")))
            dn = bk.sample(N_DRAFT, pre_n, seed=12)
            di = bk.sample(N_DRAFT, pre_i, seed=11)
            g = np.random.default_rng(0)
            starts = [(u, ((1 - u) * dn + u * g.standard_normal(dn.shape)).astype(np.float32)) for u in LEVELS]
            starts.append((U_NOISE, (U_NOISE * g.standard_normal((N_NOISE, bk.dim))).astype(np.float32)))
            x0 = {}
            for name, w in zip(FIELDS, (0.0, 1.0, CAG_W)):
                fld = Field(bk, pre_n, pre_i, w)
                x0[name] = np.concatenate([mode_ascent(fld, None, u, iters=ITERS, lr=LR, x_init=S0, n_starts=len(S0))[1]
                                           for u, S0 in starts])
            t_modes = time.time() - t0
            chunks = np.concatenate([x0["null"], dn, di, x0["instr"], x0["cag"]]).astype(np.float32)
            ns = len(x0["null"]) + 2 * N_DRAFT                         # scored: null modes + both drafts (instr / CAG modes: mass only)
            Ln = fm_losses(bk, pre_n, chunks[:ns], E)
            Li = fm_losses(bk, pre_i, chunks[:ns], E)
        raw = bk.unnormalize(chunks).reshape(len(chunks), bk.H, -1)
        nm = len(x0["null"])
        sl = dict(null=(0, nm), draft_null=(nm, nm + N_DRAFT), draft_instr=(nm + N_DRAFT, nm + 2 * N_DRAFT),
                  instr=(nm + 2 * N_DRAFT, 2 * nm + 2 * N_DRAFT), cag=(2 * nm + 2 * N_DRAFT, 3 * nm + 2 * N_DRAFT))
        start_u = np.concatenate([np.full(len(S0), u) for u, S0 in starts]).astype(np.float32)
        save_atomic(outp, chunks=chunks, raw=raw.astype(np.float32), L_null=Ln, L_instr=Li, start_u=start_u,
                    slices=np.array(json.dumps(sl)), secs=np.array([t_modes, time.time() - t0]))
        done += 1
        print(f"[r2a modes] {key}: modes {t_modes:.0f}s, total {time.time() - t0:.0f}s ({done} done, {time.time() - t_all:.0f}s)",
              flush=True)
    print(f"[r2a modes] shard {shard} finished: {done} new", flush=True)


# ── stage 3: the read (chunks env) ──────────────────────────────────────────────────────────────────────────────
def obs_readout(m, z, zm):
    """One observation -> dict of 0/1 (or shares) per arm and the coverage."""
    sl = json.loads(str(zm["slices"]))
    eef = z["state"][:3]
    cp = z["cand_pos"]
    cn = [str(x) for x in z["cand_names"]]
    named = set(m["named"])
    raw = zm["raw"]
    hit = lambda idx: (idx >= 0 and cn[idx] in named)                    # noqa: E731
    pts = np.array([pointed(r, eef, cp)[0] for r in raw])
    Ln, Li = zm["L_null"].mean(0), zm["L_instr"].mean(0)                 # uniform weight over T_GRID (velocity loss)
    tw = np.array([(1 - t) ** 2 for t in T_GRID])[:, None]
    Ln_e, Li_e = (zm["L_null"] * tw).mean(0), (zm["L_instr"] * tw).mean(0)   # eps-weighted (descriptive)
    delta, delta_e = Ln - Li, Ln_e - Li_e

    def options(name):
        a, b = sl[name]
        lab = clusters(zm["chunks"][a:b])
        out = []
        for c in range(lab.max() + 1):
            mem = np.flatnonzero(lab == c) + a
            obj = [int(p) for p in pts[mem]]
            vals, cnt = np.unique(obj, return_counts=True)
            o = int(vals[np.argmax(cnt)]) if len(vals) else -1
            if cnt.max() == 1 and len(vals) > 1:
                o = obj[0]
            sc = mem.max() < len(delta)                                  # instr / CAG modes are not scored (mass only)
            out.append(dict(obj=o, mass=len(mem), delta=float(delta[mem].mean()) if sc else None,
                            delta_e=float(delta_e[mem].mean()) if sc else None, li=float(-Li[mem].mean()) if sc else None))
        return out

    nul = options("null")
    top = lambda opts, k: max(range(len(opts)), key=lambda i: (opts[i][k], -i))   # noqa: E731
    r = dict(n_options=len(nul), n_objects=len({o["obj"] for o in nul if o["obj"] >= 0}),
             coverage=float(any(hit(o["obj"]) for o in nul)),
             ratio_top=float(hit(nul[top(nul, "delta")]["obj"])),
             ratio_e_top=float(hit(nul[top(nul, "delta_e")]["obj"])),
             instr_lik_top=float(hit(nul[top(nul, "li")]["obj"])),
             null_mass_top=float(hit(nul[top(nul, "mass")]["obj"])),
             random=float(np.mean([hit(o["obj"]) for o in nul])))
    for name in ("instr", "cag"):
        op = options(name)
        r[f"{name}_mode_top"] = float(hit(op[top(op, "mass")]["obj"]))
        r[f"{name}_coverage"] = float(any(hit(o["obj"]) for o in op))
    a, b = sl["draft_instr"]
    r["policy_sample"] = float(np.mean([hit(p) for p in pts[a:b]]))
    a, b = sl["draft_null"]
    r["null_sample"] = float(np.mean([hit(p) for p in pts[a:b]]))
    r["draft_null_coverage"] = float(any(hit(p) for p in pts[a:b]))
    r["policy_exec"] = float(m["exec_pointed"] in named) if m["exec_pointed"] is not None else 0.0
    r["ratio_top_given_cov"] = r["ratio_top"] if r["coverage"] else None
    r["random_given_cov"] = r["random"] if r["coverage"] else None              # descriptive, added at the read
    r["policy_sample_given_cov"] = r["policy_sample"] if r["coverage"] else None
    r["null_mass_top_given_cov"] = r["null_mass_top"] if r["coverage"] else None
    r["single_option"] = float(r["n_options"] == 1)
    r["top_obj"] = {k: (cn[nul[top(nul, k)]["obj"]] if nul[top(nul, k)]["obj"] >= 0 else None) for k in ("delta", "mass")}
    return r


ARMS = ("ratio_top", "policy_sample", "policy_exec", "instr_lik_top", "null_mass_top", "instr_mode_top", "cag_mode_top", "random",
        "ratio_e_top", "null_sample")


def ci_rate(per_task, B=10000, seed=0):
    """Two-level cluster bootstrap of a pooled mean: per_task {task: [episode values]} (episode = mean over its observations)."""
    rng = np.random.default_rng(seed)
    ts = [t for t in per_task if len(per_task[t])]
    xs = [np.asarray(per_task[t], float) for t in ts]
    n = sum(len(x) for x in xs)
    if not n:
        return dict(mean=None, lo=None, hi=None, n=0, tasks=0)
    s = np.empty(B)
    for i in range(B):
        num, den = 0.0, 0
        for j in rng.integers(0, len(xs), len(xs)):
            m = len(xs[j])
            num += xs[j][rng.integers(0, m, m)].sum()
            den += m
        s[i] = num / den
    return dict(mean=float(sum(x.sum() for x in xs) / n), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)),
                n=int(n), tasks=len(xs))


def read(record):
    from lifelong_ood import stats
    import fm_types as FT
    ix = load_index()
    ax = FT.axes()
    rows = []
    for tid, x in ix.items():
        a = ax[tid]
        for e in x["episodes"]:
            for m in e["obs"]:
                mp = os.path.join(OUT, "modes", m["key"] + ".npz")
                if not os.path.exists(mp):
                    continue
                z = np.load(os.path.join(OUT, "obs", m["key"] + ".npz"), allow_pickle=True)
                r = obs_readout(m, z, np.load(mp, allow_pickle=True))
                r.update(task=tid, trial=e["trial"], role=m["role"], key=m["key"], type=e["type"], success=e["success"],
                         bin=a["bin"], kind=a["kind"], scene_familiar=a["scene_familiar"], instance=a["instance"])
                rows.append(r)
    n_expected = sum(len(e["obs"]) for x in ix.values() for e in x["episodes"])
    print(f"[r2a read] {len(rows)} of {n_expected} observations have modes", flush=True)

    def by_episode(sel, arm):
        """{task: [episode means]} over the selected observation rows."""
        ep = {}
        for r in sel:
            if r.get(arm) is None:
                continue
            ep.setdefault((r["task"], r["trial"]), []).append(r[arm])
        out = {}
        for (t, _), v in ep.items():
            out.setdefault(t, []).append(float(np.mean(v)))
        return out

    def table(sel, arms=ARMS + ("coverage", "draft_null_coverage", "instr_coverage", "cag_coverage", "ratio_top_given_cov", "random_given_cov",
                                     "policy_sample_given_cov", "null_mass_top_given_cov", "single_option")):
        return {arm: ci_rate(by_episode(sel, arm)) for arm in arms}

    def paired(sel, a, b):
        ep = {}
        for r in sel:
            ep.setdefault((r["task"], r["trial"]), []).append((r[a], r[b]))
        pairs = {}
        for (t, _), v in ep.items():
            xa, xb = np.mean([p[0] for p in v]), np.mean([p[1] for p in v])
            pa, pb = pairs.setdefault(t, ([], []))
            pa.append(xa)
            pb.append(xb)
        return {t: (np.asarray(p[0]), np.asarray(p[1])) for t, p in pairs.items()}

    res = dict(n_obs=len(rows), n_expected=n_expected, settings=dict(levels=LEVELS, u_noise=U_NOISE, n_draft=N_DRAFT, n_noise=N_NOISE,
               iters=ITERS, lr=LR, cag_w=CAG_W, t_grid=T_GRID, k_noise=K_NOISE, tol=TOL, d_min=D_MIN, merge=MERGE))
    for role in ("pre", "post"):
        sel = [r for r in rows if r["role"] == role]
        if not sel:
            continue
        rr = dict(all=table(sel))
        for axis, fn in (("bin", lambda r: r["bin"]), ("kind", lambda r: r["kind"]),
                         ("scene", lambda r: "familiar" if r["scene_familiar"] else "new"),
                         ("instance", lambda r: "instance" if r["instance"] else "no_qualifier"),
                         ("zs_type", lambda r: str(r["type"]))):
            rr[axis] = {lv: table([r for r in sel if fn(r) == lv]) for lv in sorted({fn(r) for r in sel})}
        rr["by_task"] = {t: {k: v["mean"] for k, v in table([r for r in sel if r["task"] == t],
                                                              ("coverage", "ratio_top", "policy_sample", "null_mass_top")).items()}
                         for t in sorted({r["task"] for r in sel})}
        rr["n_options_mean"] = float(np.mean([r["n_options"] for r in sel]))
        rr["n_objects_mean"] = float(np.mean([r["n_objects"] for r in sel]))
        res[role] = rr
    # hypotheses (pre-registered): object role
    pre = [r for r in rows if r["role"] == "pre"]
    h = {}
    cov = res["pre"]["all"]["coverage"]
    h["H1"] = dict(coverage=cov, passed=bool(cov["mean"] is not None and cov["mean"] >= 0.8))
    wo = [r for r in pre if r["type"] == "wrong_object"]
    pw = paired(wo, "ratio_top", "policy_sample")
    d = [float(x - y) for t in pw for x, y in zip(*pw[t])]
    bpos, bneg = int(sum(v > 0 for v in d)), int(sum(v < 0 for v in d))
    from math import comb
    p2 = float(sum(comb(bpos + bneg, i) for i in range(bpos, bpos + bneg + 1)) / 2 ** (bpos + bneg)) if bpos + bneg else 1.0
    testable = len(d) >= 10
    h["H2"] = dict(episodes=len(d), tasks=len(pw), positive=bpos, negative=bneg, mean_diff=float(np.mean(d)) if d else None,
                   p_one_sided_sign=p2 if testable else 1.0, testable=testable,
                   ratio_top=ci_rate(by_episode(wo, "ratio_top")), policy_sample=ci_rate(by_episode(wo, "policy_sample")),
                   diff_ci=stats.bootstrap_diff(pw, B=10000, seed=0) if pw else None)
    su = [r for r in pre if r["success"]]
    ps = paired(su, "ratio_top", "policy_sample")
    bd = stats.bootstrap_diff(ps, B=10000, seed=0, return_samples=True)
    p3 = float(np.mean(bd.pop("samples") <= -0.05))
    h["H3"] = dict(diff=bd, p_one_sided=p3, margin=-0.05)
    ps_ = sorted([("H2", h["H2"]["p_one_sided_sign"]), ("H3", p3)], key=lambda x: x[1])
    for rank, (name, p) in enumerate(ps_):
        h[name]["holm_threshold"] = 0.05 / (len(ps_) - rank)
    passed_prev = True
    for rank, (name, p) in enumerate(ps_):
        ok = passed_prev and p < 0.05 / (len(ps_) - rank)
        h[name]["passed"] = bool(ok)
        passed_prev = ok
    res["hypotheses"] = h
    # descriptive: the failures the ratio could fix, and what the ratio's top choice is on them
    res["wrong_object_cards"] = {}
    for t in sorted({r["task"] for r in wo}):
        rs = [r for r in wo if r["task"] == t]
        tops = {}
        for r in rs:
            tops[str(r["top_obj"]["delta"])] = tops.get(str(r["top_obj"]["delta"]), 0) + 1
        res["wrong_object_cards"][t] = dict(obs=len(rs), coverage=float(np.mean([r["coverage"] for r in rs])),
                                            ratio_top=float(np.mean([r["ratio_top"] for r in rs])),
                                            policy_sample=float(np.mean([r["policy_sample"] for r in rs])), ratio_top_objects=tops)
    path = os.path.join(OUT, "read.json")
    dump(path, res)
    for role in ("pre", "post"):
        if role not in res:
            continue
        print(f"\n[r2a read] {role}: " + "  ".join(f"{k} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
                                                    for k, v in res[role]["all"].items() if v["mean"] is not None))
        for axis in ("bin", "kind", "scene", "instance", "zs_type"):
            for lv, tb in res[role][axis].items():
                print(f"[r2a read]   {axis}={lv} (n {tb['coverage']['n']} ep, {tb['coverage']['tasks']} tasks): " +
                      " ".join(f"{k} {tb[k]['mean']:.2f}" for k in ("coverage", "ratio_top", "policy_sample", "null_mass_top",
                                                                     "instr_mode_top", "cag_mode_top", "random")
                               if tb[k]["mean"] is not None))
    print("\n[r2a read] hypotheses", json.dumps({k: {q: v[q] for q in v if q in ("passed", "p_one_sided_sign", "p_one_sided",
                                                                                   "episodes", "positive", "negative", "mean_diff")}
                                                 for k, v in h.items()}, default=str))
    print("[r2a read] H1 coverage", json.dumps(h["H1"]["coverage"]), "H3", json.dumps(h["H3"]["diff"]))
    print("[r2a read] wrong-object cards", json.dumps(res["wrong_object_cards"]))
    if record:
        from lifelong_ood import ledger
        print(ledger.record("r2a_read", bench="libero_90_c", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=res["settings"], script="experiments/r2a_null_modes.py", command=" ".join(sys.argv),
                            backbone=dict(config=CFG, checkpoint=CKPT), artifacts=[path]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("fresh-check", "launch", "extract", "validate", "modes", "read", "dry", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--probe", nargs=2, metavar=("TASK", "TRIAL"))
    a = ap.parse_args()
    if a.fresh_check:
        fresh_check()
    if a.launch:
        launch(a.dry)
    if a.extract:
        extract(a.shard, a.tasks)
    if a.probe:
        probe(*a.probe)
    if a.validate:
        validate()
    if a.modes:
        modes(a.shard)
    if a.read:
        read(a.record)
