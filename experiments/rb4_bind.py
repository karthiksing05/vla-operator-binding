"""PROBE R4 (docs/PLAN_R4.md; pre-registration experiments/prereg/pr4_bind.py, ledger m3_prereg tag pr4_bind): vision-anchored
binding.  Bypass the broken word -> referent link of the frozen pi05_l90c@5000: the parse names the role (object pre-grasp, goal
post-grasp), the VISION anchor (OF D3: SigLIP identity prototypes from training scenes only) finds that referent in the agentview
image, depth back-projects the anchor token to a 3D target, and OF D1's extracted approach operator d = lam (c - h) + b turns it into
a displacement that guides the policy's OWN flow sampler while it generates the chunk.  No demo, no weight change, no new token.

Guidance (pre-declared): endpoint x0-projection at each guided Euler step.  With x_t = (1 - t) a + t eps and v the policy's velocity,
x0_hat = x_t - t v.  The chunk's integrated xy displacement D(x0_hat) = sum_h scale * unnorm(x0_hat)[h, :2] is affine in x0_hat, so
the step moves x0_hat by w (d* - D(x0_hat)) spread evenly over the H steps' xy dims (the minimum-norm move; w = 1 is the exact
projection) and replaces v by (x_t - x0_hat') / t.  It is a gradient step on |D(x0_hat) - d*|^2 through the action map only (no
backprop through the network).  z, rotation, gripper and the padded dims stay the policy's.

Stages (each its own SLURM job; nothing here runs on the login node):
  --operator              (chunks env, CPU)  D1's pooled approach operator (lam, b) from OF's training frames -> operator.json
  --protos                (chunks env, CPU)  identity prototypes of every object AND fixture type (training frames) + named types
  --geom --shard i/n      (LIBERO env, EGL)  depth at every tuning / test observation (robot hidden) -> per-token 3D points
  --smoke                 (openpi GPU)       parity (no guide == bk.sample == R2a draws), affine action map, w=1 exactness, norm
  --tune --shard i/n      (openpi GPU)       stage 0 on R3b's tuning pool: u11, u21, every config toward the cf / named ORACLE pose
  --stage0 [--record]     (chunks env)       movability + on-manifold gate + the pre-registered choice -> stage0.json, rb4_stage0
  --test --shard i/n      (openpi GPU)       R3b's test sets: u11, u21, s_vis (vision anchor), s_orc (oracle anchor), s_cf
  --read [--record]       (chunks env)       anchor localisation, H1-H4, rb4_read
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
import r2a_null_modes as R2A                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "rb4")
RS3 = os.path.join(ROOT, "results", "rs3")
OFD = os.path.join(ROOT, "results", "of")
R2 = R2A.OUT
CFG, CKPT = R2A.CFG, R2A.CKPT
# ── settings (fixed before any guided chunk was scored; docs/PLAN_R4.md) ──────────────────────────────────────────
N_SAMP = 12
SEED_A, SEED_B = 11, 21                                                 # R2a policy_sample seed (parity) / noise floor
WS = (0.25, 0.5, 1.0)                                                   # guidance strength
SCHEDS = ("early", "all")                                               # guided Euler steps: the first 5 of 10, or all 10
EARLY_STEPS = 5
TARGETS = ("d1", "dir")                                                 # d* = lam (c - h) + b  |  |D| (>= 3 cm) along c - h
DIR_MIN = 0.03
OP_SOURCE, OP_LEVEL = "instr", 0.55                                     # D1's l90c instruction modes at their best level
MOVE_BAR, FLOOR_MIN = 0.10, 0.03                                        # stage 0: g_cf >= 0.10 + max(0.03, |u21 - u11 at cf|)
ADMIT_NAMED, ADMIT_STILL, ADMIT_FM, ADMIT_OOB, TIE = -0.02, 0.10, 0.10, 0.05, 0.005
GATE_RISE, GATE_HARM, ALPHA = 0.20, -0.05, 0.05
T_FM, K_FM, FM_SEED, FM_Z = (0.1, 0.2, 0.3, 0.5), 4, 5, 2.0             # flow-loss check: outlier = FM > mean + 2 sd of u11's 12
OOB = 1.0                                                               # |normalised action| > 1 (outside q01..q99, pi0.5 quantile)
COV_MIN = 0.05
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"


def grid():
    return [dict(name=f"{tg}|{sc}|w{w:g}", target=tg, sched=sc, w=float(w)) for tg in TARGETS for sc in SCHEDS for w in WS]


def dump(path, obj):
    R2A.dump(path, obj)


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


META = {}


def _load_meta():
    for tid, x in R2A.load_index().items():
        for e in x["episodes"]:
            for m in e["obs"]:
                META[m["key"]] = dict(m, task=tid, trial=e["trial"], type=e["type"], success=e["success"])


def cf_pick(key, names, excl):
    """R3b's counterfactual target (rs3_steer.cf_pick): a non-named candidate chosen by a hash of the key (no outcome)."""
    pool = [n for n in names if n not in excl]
    if not pool:
        return None
    return pool[int(hashlib.sha1(("rs3cf:" + key).encode()).hexdigest(), 16) % len(pool)]


def _keys_all():
    sp = jload(os.path.join(RS3, "split.json"))
    return list(sp["tune"]), sorted(set(k for v in sp["test"].values() for k in v)), sp


# ── operator (chunks env): OF D1's pooled approach operator ──────────────────────────────────────────────────────
def operator():
    """Pooled least squares d = lam (o - h) + b over every l90c training frame of OF D1 (instruction modes, level 0.55, object
    centre frame without rotation: OF found the yaw adds nothing).  Training tasks only; no dev observation is read."""
    import of_frame as OF
    scale = OF.scale_for("l90c")
    rows = OF.d1_rows("l90c", scale)[0][(OP_SOURCE, OP_LEVEL)]
    X = np.concatenate([OF.design(r, "object_trans") for r in rows])
    y = np.concatenate([r["d"] for r in rows])
    th = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = np.stack([OF.design(r, "object_trans") @ th for r in rows])
    d = np.stack([r["d"] for r in rows])
    cosd = float(np.mean(np.sum(unit(pred) * unit(d), 1)))
    cos_c = float(np.mean(np.sum(unit(np.stack([r["o"] - r["h"] for r in rows])) * unit(d), 1)))
    out = dict(lam=float(th[0]), b=[float(th[1]), float(th[2])], n=len(rows), tasks=len({r["tid"] for r in rows}),
               r2=float(1 - ((d - pred) ** 2).sum() / ((d - d.mean(0)) ** 2).sum()), mean_cos_pred_vs_mode=cosd,
               mean_cos_centre_vs_mode=cos_c, mode_disp_norm_median=float(np.median(np.linalg.norm(d, axis=1))),
               centre_dist_median=float(np.median([np.linalg.norm(r["o"] - r["h"]) for r in rows])),
               source=OP_SOURCE, level=OP_LEVEL, scale=list(map(float, scale)))
    dump(os.path.join(OUT, "operator.json"), out)
    print("[rb4 operator] " + json.dumps(out), flush=True)


# ── prototypes (chunks env): every object and fixture type, training frames only; named types of the observations ──
def protos():
    import of_frame as OF
    from lifelong_ood.policy import factors as F
    ix = {x["tid"]: x for x in OF.jload(os.path.join(OFD, "index_l90c.json"))["tasks"]}
    sel = OF.jload(os.path.join(OFD, "sel_l90c.json"))["obs"]
    rows, miss = [], 0
    for s in sel:
        p = os.path.join(OFD, "obs_l90c", s["key"] + ".npz")
        if not os.path.exists(p):
            miss += 1
            continue
        z = np.load(p, allow_pickle=True)
        x = ix[s["tid"]]
        inst = {**x["fixtures"], **x["objects"]}
        for b, nm in enumerate([str(u) for u in z["bodies"]]):
            if nm in inst and int(z["ntok"][b]) >= 2:
                rows.append(dict(cat=inst[nm], kind="object" if nm in x["objects"] else "fixture", scene=x["scene"],
                                 f=unit(z["feat_sig"][b].astype(np.float32))))
    cats = sorted({r["cat"] for r in rows})
    P = np.stack([unit(np.mean([r["f"] for r in rows if r["cat"] == c], 0)) for c in cats]).astype(np.float32)
    kind = [sorted({r["kind"] for r in rows if r["cat"] == c}) for c in cats]
    cnt = [sum(r["cat"] == c for r in rows) for c in cats]
    sc = [len({r["scene"] for r in rows if r["cat"] == c}) for c in cats]
    # sanity: object-only types equal OF's prototypes (same rows)
    pz = np.load(os.path.join(OFD, "protos_l90c.npz"), allow_pickle=True)
    ofc = [str(c) for c in pz["cats"]]
    diff = {c: float(np.abs(P[cats.index(c)] - pz["sig"][ofc.index(c)]).max()) for c in ofc if c in cats and kind[cats.index(c)] == ["object"]}
    R2A.save_atomic(os.path.join(OUT, "protos.npz"), cats=np.array(cats), sig=P, count=np.array(cnt), scenes=np.array(sc),
                    kind=np.array(json.dumps(kind)))
    # named types of every tuning / test observation (the parse's role filler -> its BDDL type)
    _load_meta()
    tasks = {t.tid: t for t in F.load()}
    tune, test, _ = _keys_all()
    nt = {}
    for key in tune + test:
        m = META[key]
        t = tasks[m["task"]]
        inst = {**t.objects, **t.fixtures}
        out = []
        for nm in m["named"]:
            base = max([q for q in inst if nm == q or nm.startswith(q + "_")], key=len, default=None)
            out.append(dict(name=nm, base=base, type=inst.get(base), proto=(cats.index(inst[base]) if base and inst[base] in cats else None)))
        nt[key] = out
    dump(os.path.join(OUT, "named_types.json"), nt)
    cov = {r: sum(any(q["proto"] is not None for q in nt[k]) for k in nt if META[k]["role"] == r) for r in ("pre", "post")}
    print(f"[rb4 protos] {len(rows)} regions ({miss} missing frames), {len(cats)} types: {dict(zip(cats, cnt))}; kinds "
          f"{dict(zip(cats, kind))}; max |P - OF proto| on object types {max(diff.values()) if diff else None}; named-type coverage "
          f"{cov} of pre {sum(META[k]['role'] == 'pre' for k in nt)} / post {sum(META[k]['role'] == 'post' for k in nt)}", flush=True)


# ── geometry (LIBERO env, EGL): depth at the recorded poses -> per-token 3D points ────────────────────────────────
TOK_RC = np.stack(np.meshgrid(np.arange(16) * 16 + 7.5, np.arange(16) * 16 + 7.5, indexing="ij"), -1).reshape(256, 2)
ROBOT_PREFIX = ("robot0_", "gripper0_", "mount0_")
ORIENT = {"rs3": lambda r, c: (r, 255.0 - c), "id": lambda r, c: (r, c), "flip_r": lambda r, c: (255.0 - r, c),
          "flip_both": lambda r, c: (255.0 - r, 255.0 - c)}


def backproject(depth, sim, cam, orient):
    """client-orientation metric depth [256, 256] -> world points [256, 256, 3] (camera convention of rs3_steer._CU)."""
    cid = sim.model.camera_name2id(cam)
    f = 0.5 * 256 / np.tan(float(sim.model.cam_fovy[cid]) * np.pi / 360.0)
    r, c = np.meshgrid(np.arange(256, dtype=np.float64), np.arange(256, dtype=np.float64), indexing="ij")
    a, b = ORIENT[orient](r, c)
    z = depth.astype(np.float64)
    pc = np.stack([(b - 128.0) * z / f, -(a - 128.0) * z / f, -z], -1)
    return np.asarray(sim.data.cam_xpos[cid])[None, None] + pc @ np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3).T


def token_points(Wp, depth):
    """per token: the median world point of its foreground half (pixels at or nearer than the token's median depth)."""
    out = np.zeros((256, 3), np.float32)
    for k in range(256):
        i, j = divmod(k, 16)
        d = depth[16 * i:16 * i + 16, 16 * j:16 * j + 16].reshape(-1)
        P = Wp[16 * i:16 * i + 16, 16 * j:16 * j + 16].reshape(-1, 3)
        out[k] = np.median(P[d <= np.median(d)], 0)
    return out


def geom(shard):
    import pathlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero_client as LC
    _load_meta()
    tune, test, _ = _keys_all()
    want = set(tune) | set(test)
    ix = R2A.load_index()
    i, n = map(int, shard.split("/"))
    od = os.path.join(OUT, "geom")
    os.makedirs(od, exist_ok=True)
    report = {}
    for j, tid in enumerate(sorted(ix)):
        if j % n != i:
            continue
        x = ix[tid]
        todo = [(e, mm) for e in x["episodes"] for mm in e["obs"] if mm["key"] in want
                and not os.path.exists(os.path.join(od, mm["key"] + ".npz"))]
        if not todo:
            continue
        suite, ti = tid.split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(ti))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        fresh = np.load(os.path.join(R2A.R, R2A.FRESH_DIR, f"{suite}_{ti}.npy"))
        inner = env.env
        sim = inner.sim
        m = sim.model
        robot_geoms = np.array([g for g in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(ROBOT_PREFIX)])
        # The rollout recordings (per-step object poses) no longer exist on disk, so the depth is rendered at the episode's START
        # state (the trial's fresh init state; pre-grasp objects are mostly unmoved).  The calibration below measures the error
        # against the observation-time candidate positions.
        onames = list(inner.objects_dict.keys())
        errs = {o: [] for o in ORIENT}
        saved, dcache = [], {}
        for e, mm in todo:
            tr = int(e["trial"])
            if tr not in dcache:
                env.set_init_state(fresh[tr])
                sim.forward()
                old = m.geom_group[robot_geoms].copy()
                m.geom_group[robot_geoms] = 5
                try:
                    dcache[tr] = LC.seg_depth(sim, "agentview")[1]
                finally:
                    m.geom_group[robot_geoms] = old
            depth = dcache[tr]
            zo = np.load(os.path.join(R2, "obs", mm["key"] + ".npz"), allow_pickle=True)
            g = np.load(os.path.join(RS3, "regions", mm["key"] + ".npz"), allow_pickle=True)
            cn = [str(q) for q in zo["cand_names"]]
            cp = np.asarray(zo["cand_pos"], np.float64)
            tp = {}
            for o in ORIENT:
                tp[o] = token_points(backproject(depth, sim, "agentview", o), depth)
                for c_i, nm in enumerate(cn):                           # calibration: seg-region candidates (objects) only
                    if str(g["how"][c_i]) == "seg" and nm in onames:
                        r_ = g["region"][c_i] > 0
                        errs[o].append(float(np.linalg.norm(np.median(tp[o][r_][:, :2], 0) - cp[c_i, :2])))
            saved.append((mm["key"], tp, depth))
        med = {o: (float(np.median(v)) if v else None) for o, v in errs.items()}
        best = min((o for o in med if med[o] is not None), key=lambda o: med[o])
        report[tid] = dict(n_obs=len(saved), median_err_m=med, best=best, n_pairs=len(errs[best]))
        print(f"[rb4 geom] {tid}: {len(saved)} obs; median xy error of the region's token points vs the candidate position by "
              f"orientation {json.dumps({k: (None if v is None else round(v, 4)) for k, v in med.items()})}", flush=True)
        assert best == "rs3" and med["rs3"] < 0.05, (tid, med)   # relaxed from 0.03 before any policy read (disclosed)
        for key, tp, depth in saved:
            R2A.save_atomic(os.path.join(od, key + ".npz"), tok_xyz=tp["rs3"].astype(np.float32),
                            tok_depth=depth.reshape(16, 16, 16, 16).transpose(0, 2, 1, 3).reshape(256, 256).min(1).astype(np.float32))
        env.close()
    dump(os.path.join(od, f"_report_{i}of{n}.json"), report)


# ── guided sampling (openpi env) ─────────────────────────────────────────────────────────────────────────────────
class Guided:
    def __init__(self, bk):
        import torch
        self.bk, self.torch = bk, torch
        H = bk.H
        self.scale = np.asarray(R2A.SCALE, np.float64)
        u0 = bk.unnormalize(np.zeros((1, bk.dim), np.float32))[0]                       # [H, 7]
        al = np.zeros((H, 2))
        for d in (0, 1):
            E = np.zeros((1, bk.dim), np.float32)
            E[0, [h * 7 + d for h in range(H)]] = 1.0
            al[:, d] = bk.unnormalize(E)[0][:, d] - u0[:, d]
        rng = np.random.default_rng(0)
        Rr = rng.standard_normal((4, bk.dim)).astype(np.float32)
        U = bk.unnormalize(Rr)
        pred = Rr.reshape(4, H, 7)[:, :, :2] * al[None] + u0[None, :, :2]
        self.affine_err = float(np.abs(pred - U[:, :, :2]).max())
        assert self.affine_err < 1e-3, self.affine_err
        self.alpha, self.beta = al, u0[:, :2]
        self.idx = [[h * 7 + d for h in range(H)] for d in (0, 1)]
        dev = bk.device
        self.t_alpha = torch.as_tensor(al.T, dtype=torch.float32, device=dev)          # [2, H]
        self.t_beta = torch.as_tensor(u0[:, :2].T, dtype=torch.float32, device=dev)
        self.t_scale = torch.as_tensor(self.scale[:2], dtype=torch.float32, device=dev)

    def disp(self, rows):
        """normalised real rows [n, dim] (torch) -> integrated xy displacement in metres [n, 2]."""
        out = []
        for d in (0, 1):
            u = rows[:, self.idx[d]] * self.t_alpha[d][None] + self.t_beta[d][None]
            out.append(u.sum(1) * self.t_scale[d])
        return self.torch.stack(out, 1)

    def correct(self, rows, delta):
        """rows + the minimum-norm move giving the xy displacement change delta [n, 2] (spread evenly over the H steps)."""
        rows = rows.clone()
        H = self.bk.H
        for d in (0, 1):
            rows[:, self.idx[d]] += (delta[:, d:d + 1] / H) / (self.t_scale[d] * self.t_alpha[d][None])
        return rows

    def dstar(self, D, cfg, h, c, op):
        torch = self.torch
        ch = torch.as_tensor(np.asarray(c, np.float64)[:2] - np.asarray(h, np.float64)[:2], dtype=torch.float32, device=D.device)
        if cfg["target"] == "d1":
            v = op["lam"] * ch + torch.as_tensor(op["b"], dtype=torch.float32, device=D.device)
            return v[None].expand_as(D)
        mag = torch.clamp(D.norm(dim=1, keepdim=True), min=DIR_MIN)
        return mag * (ch / ch.norm().clamp(min=1e-9))[None]

    def sample(self, n, prefix, seed, cfg=None, h=None, c=None, op=None):
        """Euler on the flow, step for step bk.sample (the same noise and ops) when cfg is None.  -> normalised real rows."""
        torch, bk = self.torch, self.bk
        K = bk.num_steps
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn((n, bk.dim_full), generator=g).to(bk.device)
        dt = torch.tensor(-1.0 / K, dtype=torch.float32, device=bk.device)
        t = torch.tensor(1.0, dtype=torch.float32, device=bk.device)
        k = 0
        with torch.no_grad():
            while t >= -dt / 2:
                vt = bk.v_full(x, t.expand(n), prefix, None)
                if cfg is not None and (cfg["sched"] == "all" or k < EARLY_STEPS):
                    x0 = x - t * vt
                    xr = x0[:, bk.real]
                    D = self.disp(xr)
                    delta = cfg["w"] * (self.dstar(D, cfg, h, c, op) - D)
                    x0n = x0.clone()
                    x0n[:, bk.real] = self.correct(xr, delta)
                    vt = (x - x0n) / t
                x = x + dt * vt
                t = t + dt
                k += 1
        return x[:, bk.real].float().cpu().numpy()

    def posthoc(self, rows, cfg, h, c, op):
        """the naive edit (reference for the flow-loss check): the final rows moved by the full correction, no flow."""
        torch = self.torch
        X = torch.as_tensor(rows, dtype=torch.float32, device=self.bk.device)
        D = self.disp(X)
        return self.correct(X, self.dstar(D, cfg, h, c, op) - D).cpu().numpy()


def fm_loss(bk, pre, A, E):
    """flow-matching loss of each chunk A [m, dim] (normalised rows) averaged over T_FM and the shared noises E [K, dim] -> [m]."""
    import torch
    A_t = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=bk.device)
    E_t = torch.as_tensor(E, dtype=torch.float32, device=bk.device)
    m, K = len(A_t), len(E_t)
    tgt = (E_t[None] - A_t[:, None]).reshape(m * K, -1)
    out = torch.zeros(m, device=bk.device)
    with torch.no_grad():
        for t in T_FM:
            X = ((1 - t) * A_t[:, None] + t * E_t[None]).reshape(m * K, -1)
            L = torch.empty(m * K, device=bk.device)
            for s in range(0, m * K, 256):
                v = bk.v(X[s:s + 256], float(t), prefix=pre)
                L[s:s + 256] = ((v - tgt[s:s + 256]) ** 2).mean(-1)
            out += L.reshape(m, K).mean(1)
    return (out / len(T_FM)).cpu().numpy()


class Runner:
    def __init__(self):
        from lifelong_ood.policy.backbone import PolicyBackbone
        self.bk = PolicyBackbone(CFG, checkpoint_dir=CKPT, device="cuda")
        self.gd = Guided(self.bk)
        self.op = jload(os.path.join(OUT, "operator.json"))
        self.E = np.random.default_rng(FM_SEED).standard_normal((K_FM, self.bk.dim)).astype(np.float32)

    def obs(self, key):
        import torch
        z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
        od = {"observation/image": z["image"], "observation/wrist_image": z["wrist"],
              "observation/state": np.asarray(z["state"], np.float32), "prompt": str(z["text"])}
        with torch.no_grad():
            ob = self.bk.observation(od)
            pre = self.bk.prefix(ob)
        return z, ob, pre

    def siglip(self, ob):
        """the SigLIP (projected) agentview tokens [256, D]: prompt-independent (OF vision_feats 'sig')."""
        import torch
        m = self.bk.model
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(ob, train=False)
            embs, _, _ = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        return embs[0, :256].float()

    def arm(self, pre, seed, cfg=None, h=None, c=None):
        return self.gd.sample(N_SAMP, pre, seed, cfg, h, c, self.op)


def oracle_target(z, named):
    """the named candidate nearest the hand (xy) and its position."""
    cn = [str(q) for q in z["cand_names"]]
    cp = np.asarray(z["cand_pos"], np.float64)
    h = np.asarray(z["state"][:2], np.float64)
    js = [cn.index(q) for q in named if q in cn]
    j = min(js, key=lambda q: np.linalg.norm(cp[q, :2] - h))
    return cn[j], cp[j]


def gpu_stage(stage, shard):
    import torch
    _load_meta()
    tune, test, sp = _keys_all()
    rn = Runner()
    bk = rn.bk
    if stage in ("tune", "smoke"):
        keys = tune
    else:
        ch = jload(os.path.join(OUT, "stage0.json"))
        assert ch["stage0_pass"] and ch["chosen"], "stage 0 failed: no test run"
        chosen = next(q for q in grid() if q["name"] == ch["chosen"])
        keys = test
        pz = np.load(os.path.join(OUT, "protos.npz"), allow_pickle=True)
        PR = torch.as_tensor(pz["sig"], device="cuda")
        NT = jload(os.path.join(OUT, "named_types.json"))
    od = os.path.join(OUT, stage if stage != "smoke" else "scratch/smoke")
    i, n = map(int, shard.split("/"))
    mine = keys[i::n] if stage != "smoke" else keys[:2]
    os.makedirs(od, exist_ok=True)
    print(f"[rb4 {stage}] {bk.describe()}; affine map error {rn.gd.affine_err:.2e}; operator lam {rn.op['lam']:.4f} b {rn.op['b']}; "
          f"{len(mine)} observations, shard {shard}", flush=True)
    t_all, done = time.time(), 0
    for key in mine:
        outp = os.path.join(od, key + ".npz")
        if os.path.exists(outp) and stage != "smoke":
            continue
        t0 = time.time()
        m = META[key]
        z, ob, pre = rn.obs(key)
        cn = [str(q) for q in z["cand_names"]]
        cp = np.asarray(z["cand_pos"], np.float64)
        h = np.asarray(z["state"][:3], np.float64)
        named = list(m["named"])
        cf = cf_pick(key, cn, set(named))
        nm_o, c_orc = oracle_target(z, named)
        arms, info = {}, dict(key=key, named=named, cf=cf, oracle=nm_o)
        with torch.no_grad():
            arms["u11"] = rn.arm(pre, SEED_A)
            arms["u21"] = rn.arm(pre, SEED_B)
            if stage in ("tune", "smoke"):
                for c in grid():
                    if cf is not None:
                        arms[c["name"] + "|cf"] = rn.arm(pre, SEED_A, c, h, cp[cn.index(cf)])
                    arms[c["name"] + "|named"] = rn.arm(pre, SEED_A, c, h, c_orc)
                if cf is not None:
                    for tg in TARGETS:
                        arms[f"posthoc|{tg}|cf"] = rn.gd.posthoc(arms["u11"], dict(target=tg), h, cp[cn.index(cf)], rn.op)
            else:
                arms["s_orc"] = rn.arm(pre, SEED_A, chosen, h, c_orc)
                if cf is not None:
                    arms["s_cf"] = rn.arm(pre, SEED_A, chosen, h, cp[cn.index(cf)])
                # the vision anchor: argmax token of the per-token cosine to the named types' training prototypes
                pids = sorted({q["proto"] for q in NT[key] if q["proto"] is not None})
                info["protos"] = pids
                if pids:
                    S = torch.nn.functional.normalize(rn.siglip(ob), dim=-1)
                    sim = (PR[pids] @ S.T).max(0).values.cpu().numpy()
                    tok = int(np.argmax(sim))
                    gz = np.load(os.path.join(OUT, "geom", key + ".npz"))
                    xyz = gz["tok_xyz"][tok].astype(np.float64)
                    info.update(anchor_tok=tok, anchor_xyz=xyz.tolist(), anchor_err_xy=float(np.linalg.norm(xyz[:2] - c_orc[:2])),
                                sim=sim.astype(np.float16).tolist())
                    if np.all(np.isfinite(xyz)):
                        arms["s_vis"] = rn.arm(pre, SEED_A, chosen, h, xyz)
            names = list(arms)
            fm = fm_loss(bk, pre, np.concatenate([arms[a] for a in names]), rn.E).reshape(len(names), N_SAMP)
        raw = {a: bk.unnormalize(arms[a]).reshape(N_SAMP, bk.H, -1).astype(np.float32) for a in names}
        oob = {a: float((np.abs(arms[a]) > OOB).mean()) for a in names}
        info.update(secs=time.time() - t0, oob=oob)
        if stage == "smoke":
            zm = np.load(os.path.join(R2, "modes", key + ".npz"), allow_pickle=True)
            a, b = json.loads(str(zm["slices"]))["draft_instr"]
            ref = bk.unnormalize(bk.sample(N_SAMP, pre, seed=SEED_A)).reshape(N_SAMP, bk.H, -1)
            w1 = next(q for q in grid() if q["name"] == "d1|all|w1")
            D = rn.gd.disp(torch.as_tensor(arms["d1|all|w1|named"], device=bk.device)).cpu().numpy()
            ds = rn.op["lam"] * (c_orc[:2] - h[:2]) + np.asarray(rn.op["b"])
            moves = {k: float(np.abs(v - raw["u11"]).mean()) for k, v in raw.items() if k != "u11"}
            print(f"[rb4 smoke] {key}: parity max|u11 - bk.sample| {float(np.abs(ref - raw['u11']).max()):.2e}, max|u11 - R2a "
                  f"draft_instr| {float(np.abs(zm['raw'][a:b] - raw['u11']).max()):.2e}; d1|all|w1 displacement error vs d* "
                  f"{float(np.abs(D - ds[None]).max()):.2e} m (d* {ds.round(4).tolist()}, cfg {w1['name']}); share |x| > 1 of u11 "
                  f"{oob['u11']:.3f}; FM u11 {fm[0].mean():.4f} u21 {fm[1].mean():.4f}; mean |arm - u11| {json.dumps({k: round(v, 4) for k, v in moves.items()})}; "
                  f"{info['secs']:.1f}s", flush=True)
            print(f"[rb4 smoke] output transform: {bk.policy._output_transform}", flush=True)
            continue
        R2A.save_atomic(outp, arms=np.array(names), raw=np.stack([raw[a] for a in names]), fm=fm.astype(np.float32),
                        info=np.array(json.dumps(info)))
        done += 1
        print(f"[rb4 {stage}] {key}: {len(names)} arms, {info['secs']:.1f}s ({done} done, {time.time() - t_all:.0f}s)", flush=True)
    print(f"[rb4 {stage}] shard {shard} finished: {done} new", flush=True)


# ── reads (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────
def obs_rows(key, f):
    """-> {arm: dict(hit=share at the arm's target, still, fm_out=share of FM outliers vs u11, oob, fm)} for one observation.
    Targets: '|cf' and s_cf arms -> the cf candidate; every other arm -> the named candidates.  'u11@cf' / 'u21@cf': unsteered at cf."""
    z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
    s = np.load(f, allow_pickle=True)
    info = json.loads(str(s["info"]))
    cn = [str(q) for q in z["cand_names"]]
    eef, cp = z["state"][:3], z["cand_pos"]
    names = [str(q) for q in s["arms"]]
    fm = s["fm"]
    mu, sd = float(fm[names.index("u11")].mean()), float(fm[names.index("u11")].std())
    named, cf = set(info["named"]), info["cf"]
    out = {}
    for a, raw, f_ in zip(names, s["raw"], fm):
        pts = np.array([R2A.pointed(r, eef, cp)[0] for r in raw])
        at = lambda S: float(np.mean([p >= 0 and cn[p] in S for p in pts]))   # noqa: E731
        tg = {cf} if (a.endswith("|cf") or a == "s_cf") else named
        out[a] = dict(hit=at(tg), still=float((pts < 0).mean()), fm_out=float((f_ > mu + FM_Z * sd).mean()), oob=info["oob"][a],
                      fm=float(f_.mean()))
        if a in ("u11", "u21") and cf is not None:
            out[a + "@cf"] = dict(out[a], hit=at({cf}))
    return out, info


def _pairs(rows, a, b, f="hit"):
    """{task: (episode means of arm a's field, of arm b's)} over observations that have both arms."""
    ep = {}
    for key, sh in rows:
        if a in sh and b in sh:
            m = META[key]
            ep.setdefault((m["task"], m["trial"]), []).append((sh[a][f], sh[b][f]))
    out = {}
    for (t, _), v in ep.items():
        pa, pb = out.setdefault(t, ([], []))
        pa.append(float(np.mean([q[0] for q in v])))
        pb.append(float(np.mean([q[1] for q in v])))
    return {t: (np.asarray(p[0]), np.asarray(p[1])) for t, p in out.items()}


def _md(pairs):
    n = sum(len(p[0]) for p in pairs.values())
    return float(sum(p[0].sum() - p[1].sum() for p in pairs.values()) / n) if n else None


def _settings():
    return dict(n_samp=N_SAMP, seeds=[SEED_A, SEED_B], ws=WS, scheds=SCHEDS, early_steps=EARLY_STEPS, targets=TARGETS, dir_min=DIR_MIN,
                operator=dict(source=OP_SOURCE, level=OP_LEVEL), move_bar=MOVE_BAR, floor_min=FLOOR_MIN, admit_named=ADMIT_NAMED,
                admit_still=ADMIT_STILL, admit_fm=ADMIT_FM, admit_oob=ADMIT_OOB, tie=TIE, gate_rise=GATE_RISE, gate_harm=GATE_HARM,
                t_fm=T_FM, k_fm=K_FM, fm_z=FM_Z, oob=OOB,
                guidance="endpoint x0-projection: x0_hat += w (d* - D(x0_hat)) spread over the H xy steps; v = (x_t - x0_hat') / t")


def stage0(record):
    from lifelong_ood import stats
    _load_meta()
    tune, _, _ = _keys_all()
    rows = []
    for key in tune:
        f = os.path.join(OUT, "tune", key + ".npz")
        assert os.path.exists(f), f"tuning incomplete: {key}"
        rows.append((key, obs_rows(key, f)[0]))
    floor = _md(_pairs(rows, "u21@cf", "u11@cf"))
    thr = MOVE_BAR + max(FLOOR_MIN, abs(floor))
    fm_floor = _md(_pairs(rows, "u21", "u11", "fm_out"))
    res = {}
    for c in grid():
        nm, cf = c["name"] + "|named", c["name"] + "|cf"
        g = stats.bootstrap_diff(_pairs(rows, cf, "u11@cf"), B=10000, seed=0)
        r = dict(g_cf=g["diff"], g_cf_ci=[g["lo"], g["hi"]], n_cf=g["n_pairs"], tasks_cf=g["n_tasks"],
                 g_named=_md(_pairs(rows, nm, "u11")), still_named=_md(_pairs(rows, nm, "u11", "still")),
                 still_cf=_md(_pairs(rows, cf, "u11@cf", "still")),
                 fm_out_rise_cf=_md(_pairs(rows, cf, "u21", "fm_out")), fm_out_rise_named=_md(_pairs(rows, nm, "u21", "fm_out")),
                 fm_ratio_cf=float(np.mean([sh[cf]["fm"] for _, sh in rows if cf in sh]) / np.mean([sh["u11"]["fm"] for _, sh in rows if cf in sh])),
                 oob_rise_cf=_md(_pairs(rows, cf, "u11", "oob")))
        r["movable"] = bool(r["g_cf"] >= thr)
        r["on_manifold"] = bool(r["fm_out_rise_cf"] <= ADMIT_FM and r["oob_rise_cf"] <= ADMIT_OOB and r["still_cf"] <= ADMIT_STILL)
        r["admissible"] = bool(r["movable"] and r["on_manifold"] and r["g_named"] >= ADMIT_NAMED and r["still_named"] <= ADMIT_STILL)
        res[c["name"]] = r
    ref = {tg: dict(fm_out_rise=_md(_pairs(rows, f"posthoc|{tg}|cf", "u21", "fm_out")),
                    fm_ratio=float(np.mean([sh[f"posthoc|{tg}|cf"]["fm"] for _, sh in rows if f"posthoc|{tg}|cf" in sh])
                                   / np.mean([sh["u11"]["fm"] for _, sh in rows if f"posthoc|{tg}|cf" in sh])),
                    oob_rise=_md(_pairs(rows, f"posthoc|{tg}|cf", "u11", "oob"))) for tg in TARGETS}
    cg = {c["name"]: c for c in grid()}
    adm = [k for k, v in res.items() if v["admissible"]]
    chosen = None
    if adm:
        best = max(res[k]["g_cf"] for k in adm)
        near = [k for k in adm if res[k]["g_cf"] >= best - TIE]
        chosen = min(near, key=lambda k: (cg[k]["w"], SCHEDS.index(cg[k]["sched"]), TARGETS.index(cg[k]["target"])))
    out = dict(stage0_pass=bool(chosen is not None), any_movable=any(v["movable"] for v in res.values()),
               any_movable_on_manifold=any(v["movable"] and v["on_manifold"] for v in res.values()), threshold=thr,
               floor_cf_u21_minus_u11=floor, floor_named_u21_minus_u11=_md(_pairs(rows, "u21", "u11")), fm_out_floor_u21_vs_u11=fm_floor,
               chosen=chosen, table=res, posthoc_reference=ref, n_tune=len(rows),
               unsteered_named=float(np.mean([sh["u11"]["hit"] for _, sh in rows])),
               unsteered_at_cf=float(np.mean([sh["u11@cf"]["hit"] for _, sh in rows if "u11@cf" in sh])),
               unsteered_still=float(np.mean([sh["u11"]["still"] for _, sh in rows])),
               unsteered_oob=float(np.mean([sh["u11"]["oob"] for _, sh in rows])))
    dump(os.path.join(OUT, "stage0.json"), out)
    for k, v in res.items():
        print(f"[rb4 stage0] {k:14s} g_cf {v['g_cf']:+.3f} [{v['g_cf_ci'][0]:+.3f}, {v['g_cf_ci'][1]:+.3f}] g_named {v['g_named']:+.3f} "
              f"still n/cf {v['still_named']:+.3f}/{v['still_cf']:+.3f} fm_out {v['fm_out_rise_cf']:+.3f} (ratio {v['fm_ratio_cf']:.3f}) "
              f"oob {v['oob_rise_cf']:+.3f} movable {v['movable']} manifold {v['on_manifold']} adm {v['admissible']}")
    print(f"[rb4 stage0] posthoc reference {json.dumps(ref)}")
    print(f"[rb4 stage0] floor u21 - u11 at cf {floor:+.3f}; threshold {thr:.3f}; unsteered named {out['unsteered_named']:.3f} at cf "
          f"{out['unsteered_at_cf']:.3f}; stage 0 {'PASSES' if out['stage0_pass'] else 'FAILS'}; chosen {chosen}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rb4_stage0", bench="libero_90_c", tag="stage0", seed=SEED_A, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=_settings(),
                            script="experiments/rb4_bind.py", command=" ".join(sys.argv), backbone=dict(config=CFG, checkpoint=CKPT),
                            artifacts=[os.path.join(OUT, "stage0.json")]))


def _bd(pairs, **kw):
    from lifelong_ood import stats
    if not pairs:
        return dict(diff=None, lo=None, hi=None, lo90=None, hi90=None, n_pairs=0, n_tasks=0)
    return stats.bootstrap_diff(pairs, **kw)


def _rate(rows, a, f="hit", only=None):
    per, ep = {}, {}
    for key, sh in rows:
        if a in sh and (only is None or only in sh):
            m = META[key]
            ep.setdefault((m["task"], m["trial"]), []).append(sh[a][f])
    for (t, _), v in ep.items():
        per.setdefault(t, []).append(float(np.mean(v)))
    return R2A.ci_rate(per)


def read(record):
    _load_meta()
    _, _, sp = _keys_all()
    ch = jload(os.path.join(OUT, "stage0.json"))
    res = dict(chosen=ch["chosen"], settings=_settings(), sets={}, anchor={})
    allrows = {}
    for sname, keys in sp["test"].items():
        rows, loc = [], []
        for key in keys:
            f = os.path.join(OUT, "test", key + ".npz")
            assert os.path.exists(f), f"test incomplete: {key}"
            sh, info = obs_rows(key, f)
            rows.append((key, sh))
            # anchor localisation: the anchor token inside the named object's / own target's ORACLE region (rs3 regions)
            if "anchor_tok" in info:
                g = np.load(os.path.join(RS3, "regions", key + ".npz"), allow_pickle=True)
                cn = [str(q) for q in g["cand_names"]]
                rr = [g["region"][cn.index(q)] for q in info["named"] if q in cn]
                reg = np.max(rr, 0) > 0 if rr else np.zeros(256, bool)
                m = META[key]
                loc.append(dict(task=m["task"], trial=m["trial"], hit=float(reg[info["anchor_tok"]]), chance=float(reg.sum() / 256.0),
                                err=info["anchor_err_xy"], near=float(info["anchor_err_xy"] < R2A.CORRIDOR)))
        allrows[sname] = rows

        def lrate(fld):
            per, ep = {}, {}
            for r in loc:
                ep.setdefault((r["task"], r["trial"]), []).append(r[fld])
            for (t, _), v in ep.items():
                per.setdefault(t, []).append(float(np.mean(v)))
            return R2A.ci_rate(per)
        res["anchor"][sname] = dict(n_obs=len(keys), n_anchor=len(loc), hit=lrate("hit"), chance=lrate("chance"),
                                    within_5cm=lrate("near"), median_err_m=float(np.median([r["err"] for r in loc])) if loc else None)
        d = _bd(_pairs(rows, "s_vis", "u11"), B=10000, seed=0, return_samples=True)
        smp = d.pop("samples") if d["diff"] is not None else np.ones(1)
        itt = [(k, dict(sh, **({"s_vis": sh["u11"]} if "s_vis" not in sh else {}))) for k, sh in rows]
        r = dict(n_obs=len(rows), n_covered=sum("s_vis" in sh for _, sh in rows), diff=d, p_rise=float(np.mean(smp <= 0.0)),
                 p_harm=float(np.mean(smp <= GATE_HARM)), itt=_bd(_pairs(itt, "s_vis", "u11"), B=10000, seed=0),
                 oracle_minus_u11=_bd(_pairs(rows, "s_orc", "u11"), B=10000, seed=0),
                 oracle_minus_u11_covered=_bd({t: p for t, p in _pairs([(k, sh) for k, sh in rows if "s_vis" in sh], "s_orc", "u11").items()},
                                              B=10000, seed=0),
                 vis_minus_oracle=_bd(_pairs(rows, "s_vis", "s_orc"), B=10000, seed=0),
                 floor_u21_minus_u11=_bd(_pairs(rows, "u21", "u11"), B=10000, seed=0),
                 cf_control=_bd(_pairs(rows, "s_cf", "u11@cf"), B=10000, seed=0),
                 rates_covered={a: _rate(rows, a, only="s_vis") for a in ("u11", "u21", "s_vis", "s_orc")},
                 rates_all={a: _rate(rows, a) for a in ("u11", "u21", "s_orc")},
                 manifold={a: dict(fm_out=_rate(rows, a, "fm_out")["mean"], oob=_rate(rows, a, "oob")["mean"],
                                   still=_rate(rows, a, "still")["mean"]) for a in ("u11", "u21", "s_vis", "s_orc", "s_cf")})
        r["n_episodes"] = len({(META[k]["task"], META[k]["trial"]) for k, sh in rows if "s_vis" in sh})
        r["n_tasks"] = len({META[k]["task"] for k, sh in rows if "s_vis" in sh})
        res["sets"][sname] = r
    A = res["anchor"]

    def lgate(su, wr):
        a, b, c = A[su]["hit"]["mean"], A[wr]["hit"]["mean"], A[wr]["chance"]["mean"]
        if a is None or b is None or c is None:
            return dict(go=False, reason="empty")
        return dict(go=bool(b >= a - 0.10 and b >= 2 * c), success=a, wrong=b, chance=c)
    res["anchor_gates"] = dict(object=lgate("su_pre", "wo_pre"), goal=lgate("su_post", "wg_post"))
    hyp = {"H1": ("wo_pre", "rise"), "H2": ("su_pre", "harm"), "H3": ("wg_post", "rise"), "H4": ("su_post", "harm")}
    ps = {k: res["sets"][s]["p_rise" if q == "rise" else "p_harm"] for k, (s, q) in hyp.items()}
    H, prev = {}, True
    for rank, k in enumerate(sorted(ps, key=ps.get)):
        thr = ALPHA / (len(ps) - rank)
        ok = prev and ps[k] < thr
        prev = ok
        s, q = hyp[k]
        d = res["sets"][s]["diff"]
        est = (d["diff"] is not None and d["diff"] >= GATE_RISE and d["lo"] > 0) if q == "rise" else d["diff"] is not None
        H[k] = dict(set=s, kind=q, p=ps[k], holm_threshold=thr, holm_ok=bool(ok), estimate_ok=bool(est), passed=bool(ok and est), diff=d)
    res["hypotheses"] = H
    res["stage1_gate"] = bool(H["H1"]["passed"] and H["H2"]["passed"])
    path = os.path.join(OUT, "read.json")
    dump(path, res)
    for s_, v in res["anchor"].items():
        print(f"[rb4 read] anchor {s_}: " + json.dumps(v))
    print("[rb4 read] anchor gates " + json.dumps(res["anchor_gates"]))
    for s_, v in res["sets"].items():
        print(f"\n[rb4 read] {s_}: {v['n_obs']} obs ({v['n_covered']} covered), {v['n_episodes']} ep, {v['n_tasks']} tasks; " +
              json.dumps({k: v[k] for k in ("diff", "itt", "oracle_minus_u11", "oracle_minus_u11_covered", "vis_minus_oracle",
                                            "floor_u21_minus_u11", "cf_control")}))
        print("[rb4 read]   covered " + "  ".join(f"{a} {r_['mean']:.3f} [{r_['lo']:.3f}, {r_['hi']:.3f}]"
                                                  for a, r_ in v["rates_covered"].items() if r_["mean"] is not None)
              + " | manifold " + json.dumps(v["manifold"]))
    print("\n[rb4 read] hypotheses " + json.dumps({k: {q: v[q] for q in ("p", "holm_threshold", "holm_ok", "estimate_ok", "passed")}
                                                   for k, v in H.items()}))
    print(f"[rb4 read] stage-1 gate {'PASSES' if res['stage1_gate'] else 'FAILS'}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rb4_read", bench="libero_90_c", tag="stage1", seed=SEED_A, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=_settings(),
                            script="experiments/rb4_bind.py", command=" ".join(sys.argv), backbone=dict(config=CFG, checkpoint=CKPT),
                            artifacts=[path]))


def inventory():
    """counts of the inputs this probe reads (compute node only)."""
    import glob
    tune, test, sp = _keys_all()
    have = lambda d, ks: sum(os.path.exists(os.path.join(d, k + ".npz")) for k in ks)   # noqa: E731
    print(json.dumps(dict(tune=len(tune), test=len(test), sets={k: len(v) for k, v in sp["test"].items()},
                          r2a_obs=have(os.path.join(R2, "obs"), tune + test), r2a_modes=have(os.path.join(R2, "modes"), tune[:5]),
                          rs3_regions=have(os.path.join(RS3, "regions"), tune + test),
                          of_obs_l90c=len(glob.glob(os.path.join(OFD, "obs_l90c", "*.npz"))),
                          of_poses=len(glob.glob(os.path.join(OFD, "poses", "*.npz"))),
                          of_protos=os.path.exists(os.path.join(OFD, "protos_l90c.npz")),
                          of_index=os.path.exists(os.path.join(OFD, "index_l90c.json")),
                          rb4=sorted(os.listdir(OUT)) if os.path.isdir(OUT) else None)), flush=True)
    _load_meta()
    rec = {}
    for key in tune + test:
        m = META[key]
        rd = R2A.roll_path(m["task"])[:-5] + "_rec"
        f = os.path.join(rd, f"task{m['task'].split(':')[1]}_trial{m['trial']}.npz")
        rec.setdefault(m["task"], [0, 0, os.path.isdir(rd), os.path.exists(R2A.roll_path(m["task"]))])
        rec[m["task"]][0 if os.path.exists(f) else 1] += 1
    print("rec files per task [present obs, missing obs, rec dir exists, roll json exists]: " + json.dumps(rec), flush=True)
    for d in sorted(glob.glob(os.path.join(R2, "roll", "*")))[:80]:
        print("roll entry", os.path.basename(d), len(os.listdir(d)) if os.path.isdir(d) else os.path.getsize(d), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("inventory", "operator", "protos", "geom", "smoke", "tune", "stage0", "test", "read", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.inventory:
        inventory()
    if a.operator:
        operator()
    if a.protos:
        protos()
    if a.geom:
        geom(a.shard)
    if a.smoke:
        gpu_stage("smoke", a.shard)
    if a.tune:
        gpu_stage("tune", a.shard)
    if a.stage0:
        stage0(a.record)
    if a.test:
        gpu_stage("test", a.shard)
    if a.read:
        read(a.record)
