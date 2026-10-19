"""H1 (A): what a prototype DONATES to the frozen pi0.5 (exploratory, offline; docs/history/PLAN_H.md section H1.1).

Question.  Each prototype of the LIBERO-90 library (a rank-4 LoRA on the action expert, results/robot/prim_<name>_0.pt; the
object, instance and goal prototypes that e3_plan.json and e5_plan.json name) changes the policy's velocity field during one
phase.  What does that change do?  How large is it, does it point the end effector toward the named object (object and instance
prototypes) or toward the place location (goal prototypes), does it act on WHERE the hand goes (position) or on HOW (orientation,
gripper timing), how much does it lower the frozen policy's flow-matching loss on its phase and on the other phase, do the
prototypes of one object type share a subspace, are goal and object prototypes orthogonal, and how does the action change scale
with the LoRA's weight?

Method.  Matched demo states: for every prototype, up to 4 of the tasks it was fitted on (ledger c0_fit, its `joint` list), and
for each the task's SECOND demonstration (episodes_of_task(t)[1]; every prototype was fitted on demonstration 0 only, so these
frames are held out), cut into the prototype's phase exactly as c0_fit.py cuts it (approach = before the first sustained grasp,
carry = from the grasp to the release).  On 6 evenly spaced frames of its own phase plus 2 frames just before the phase's
gripper event (timing frames), and 3 frames of the other phase, prompted with the task's own clause, with 4 fixed noise draws per
frame shared by both arms:
  * the action chunk sampled by the frozen policy with the prototype off and on (Euler, 10 steps; unnormalised LIBERO actions);
  * the velocity field off and on at x_t = t eps + (1 - t) a_demo for t in 0.9, 0.5, 0.1;
  * the frozen policy's flow-matching loss off and on (8 fixed draws of time and noise per frame; c0_fit.py --score's loss);
  * for 4 prototypes, the action chunk and flow loss at LoRA weights -1 .. 3 (a scale sweep, the Concept-Sliders analogue).
Direction target: the end-effector position at the demonstration's first sustained grasp (object / instance prototypes) or at its
release (goal prototypes).  Subspaces: for every pair of prototypes and every adapted Linear of the action expert, the overlap of
the column spaces of B (what the update writes) and the row spaces of A (what it reads), mean cos^2 of the principal angles
(1 = the same subspace, chance = rank / dimension), and the Frobenius cosine of the two updates B A.
This script's GPU part writes raw arrays (results/robot/h1_donate/); --read computes the metrics, figures and ledger rows.

    sbatch --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw --partition=overcap --account=overcap --qos=short \
        --export=ALL,ENV=openpi,SCRIPT="experiments/h1_donate.py --run" slurm/jobs/robot_gpu.sbatch
    ~/flash/miniconda3/envs/chunks/bin/python experiments/h1_donate.py --read [--record]
"""
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(R, "h1_donate")
FIG = os.path.join(ROOT, "figures", "robot")
K = 4                                   # noise draws per frame
TS = (0.9, 0.5, 0.1)                    # flow times for the field comparison
SWEEP = ("Osc1", "Sch2", "Gscplate", "XG1")
SCALES = (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
MAXTASKS, NOWN, NOTHER = 4, 6, 3


def library():
    """-> {name: dict(kind in object / instance / goal, key)} over the prototypes e3_plan.json and e5_plan.json name."""
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    e5 = json.load(open(os.path.join(R, "e5_plan.json")))["tasks"]
    out = {}
    for v in list(e3.values()) + list(e5.values()):
        inst = bool(v["instance_key"])
        out.setdefault(v["object_proto"], dict(kind="instance" if inst else "object",
                                               key=v["instance_key"] if inst else v["object_key"]))
        ty = v.get("type_proto") or v.get("type")
        if ty:
            out.setdefault(ty, dict(kind="object", key=v["object_key"]))
        g = v.get("goal_proto") or v.get("goal")
        out.setdefault(g, dict(kind="goal", key=v["goal_key"]))
    return out


def fit_row(name):
    fs = sorted(glob.glob(os.path.join(ROOT, "ledger", "c0_fit", f"*joint_{name}__*.json")))
    rows = [json.load(open(f)) for f in fs]
    return max(rows, key=lambda r: r["started"])


def parent_of(name, lib):
    """An instance prototype's type prototype (same object key)."""
    k = lib[name]["key"].rsplit("|", 1)[0]
    return next((n for n, v in lib.items() if v["kind"] == "object" and v["key"] == k), None)


# ── GPU part ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def run():
    import torch
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    os.makedirs(OUT, exist_ok=True)
    bk = PolicyBackbone("pi05_libero", device="cuda")
    norm = D.Norm.load()
    d90 = D.Demos90()
    lib = library()
    state, keys = None, {}
    for n in lib:
        state, keys[n], _ = adapt.load(bk, os.path.join(R, f"prim_{n}_0.pt"), state=state)
    print(f"[h1d] {bk.describe()}; {len(keys)} prototypes loaded", flush=True)
    eps = {}

    def episode(t):
        if t not in eps:
            e = d90.load(d90.episodes_of_task(t)[1])
            e["image"], e["wrist_image"] = d90.frames(e["episode_index"])
            eps[t] = e
        return eps[t]

    def phases(ep, clause, text):
        steps = P.steps_of(text)
        s = steps.index(clause) if clause in steps else 0
        segs = P.segment(ep["actions"], len(steps))
        if segs is None:
            return None
        lo, hi = segs[s]
        tr = P.transport(ep["actions"], lo, hi)
        if tr is None or tr[0] <= lo + 4:
            return None
        g = tr[0]
        rel = [r for r in P.releases(ep["actions"][:, 6], P.RELEASE_RUN) if g < r <= hi]
        rel = rel[0] if rel else hi - 1
        return lo, g, rel, hi

    def onoff(key, w):
        if key is None or w == 0:
            return frozenset()
        return frozenset({key}) if w == 1.0 else {key: float(w)}

    @torch.no_grad()
    def chunks(pre, n, noise, key, w):
        state["on"] = onoff(key, w)
        rows = bk.sample(n * K, pre, noise=noise)
        state["on"] = frozenset()
        return rows

    @torch.no_grad()
    def field(pre, a_full, key, n):
        """v off / on at x_t on the path to the demo chunk; -> [len(TS), 4, n] (|v_off|, |v_on - v_off|, |v_off - u|, |v_on - u|)
        on the real action dimensions, and cos(v_on - v_off, u - v_off)."""
        g = torch.Generator(device="cpu").manual_seed(1)
        out = np.zeros((len(TS), 5, n), np.float32)
        A = torch.as_tensor(a_full, device=bk.device).reshape(n, -1)
        for j, t in enumerate(TS):
            e = torch.randn(A.shape, generator=g).to(bk.device)
            xt = t * e + (1 - t) * A
            u = (e - A)[:, bk.real]
            state["on"] = frozenset()
            v0 = bk.v_full(xt, torch.full((n,), t, device=bk.device), pre)[:, bk.real]
            state["on"] = onoff(key, 1.0)
            v1 = bk.v_full(xt, torch.full((n,), t, device=bk.device), pre)[:, bk.real]
            state["on"] = frozenset()
            dv = v1 - v0
            out[j, 0] = v0.norm(dim=1).cpu().numpy()
            out[j, 1] = dv.norm(dim=1).cpu().numpy()
            out[j, 2] = (v0 - u).norm(dim=1).cpu().numpy()
            out[j, 3] = (v1 - u).norm(dim=1).cpu().numpy()
            out[j, 4] = torch.nn.functional.cosine_similarity(dv, u - v0, dim=1).cpu().numpy()
        return out

    @torch.no_grad()
    def loss(pre, a_full, key, w, n, draws=8):
        """Per-frame flow-matching loss (openpi's: mean over the full padded chunk) over fixed draws of time and noise."""
        gen = torch.Generator(device="cpu").manual_seed(0)
        A = torch.as_tensor(a_full, device=bk.device)
        tot = torch.zeros(n, device=bk.device)
        for _ in range(draws):
            t, e = adapt._flow_batch(n, bk.H, bk.A, gen, bk.device)
            xt = t[:, None, None] * e + (1 - t[:, None, None]) * A
            u = (e - A).reshape(n, -1)
            state["on"] = onoff(key, w)
            v = bk.v_full(xt, t, pre)
            state["on"] = frozenset()
            tot += ((v - u) ** 2).mean(1)
        return (tot / draws).cpu().numpy()

    t_all = time.time()
    for name, info in lib.items():
        f = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        row = fit_row(name)
        goal = bool(row["metrics"].get("goal"))
        joint = row["metrics"]["joint"]
        seen, recs = set(), []
        for j in joint:
            if j["task"] in seen or len(seen) >= MAXTASKS:
                continue
            ep = episode(j["task"])
            ph = phases(ep, j["clause"], texts[j["task"]])
            if ph is None:
                continue
            seen.add(j["task"])
            lo, g, rel, hi = ph
            own = (g, rel) if goal else (lo, g)
            oth = (lo, g) if goal else (g, rel)
            ev = rel if goal else g
            fr = sorted(set(np.linspace(own[0], own[1] - 1, NOWN).round().astype(int).tolist()) | {max(own[0], ev - 6),
                                                                                                    max(own[0], ev - 3)})
            fo = np.linspace(oth[0], oth[1] - 1, NOTHER).round().astype(int).tolist()
            tgt = ep["state"][ev, :3]
            for i in fr:
                recs.append((j["task"], j["clause"], int(i), 1, tgt, int(g), int(rel)))
            for i in fo:
                recs.append((j["task"], j["clause"], int(i), 0, tgt, int(g), int(rel)))
        if not recs:
            print(f"[h1d] {name}: no usable held-out demo frames; skipped", flush=True)
            continue
        n = len(recs)
        obs = []
        for (t, c, i, *_r) in recs:
            o = d90.observation(episode(t), i)
            o["prompt"] = c
            obs.append(o)
        idx_cache = {}
        demo = []
        for (t, c, i, *_r) in recs:
            ep = episode(t)
            if t not in idx_cache:
                idx_cache[t] = D.chunk_indices(len(ep["actions"]), bk.H)[0]
            demo.append(ep["actions"][idx_cache[t][i]])
        demo = np.stack(demo).astype(np.float32)                                   # [n, H, 7] raw
        a_full = D.pad_rows(D.to_rows(demo, norm), bk.H, bk.a, bk.A)               # [n, H, A] normalised
        pre = bk.prefix(bk.observations(obs))
        preK = bk.repeat(pre, K)
        noise = torch.randn((n * K, bk.dim_full), generator=torch.Generator(device="cpu").manual_seed(0))
        off = chunks(preK, n, noise, None, 0)
        on = chunks(preK, n, noise, keys[name], 1.0)
        fld = field(pre, a_full, keys[name], n)
        l_off, l_on = loss(pre, a_full, None, 0, n), loss(pre, a_full, keys[name], 1.0, n)
        extra = {}
        if name in SWEEP:
            sw, sl = [], []
            for w in SCALES:
                sw.append(D.from_rows(chunks(preK, n, noise, keys[name], w), norm, bk.H).reshape(n, K, bk.H, 7))
                sl.append(loss(pre, a_full, keys[name], w, n))
            extra = dict(sweep_actions=np.stack(sw).astype(np.float32), sweep_loss=np.stack(sl), scales=np.array(SCALES))
        np.savez_compressed(
            f, task=np.array([r[0] for r in recs]), frame=np.array([r[2] for r in recs]), own=np.array([r[3] for r in recs]),
            target=np.stack([r[4] for r in recs]).astype(np.float32), grasp=np.array([r[5] for r in recs]),
            release=np.array([r[6] for r in recs]), ee=np.stack([episode(r[0])["state"][r[2], :3] for r in recs]).astype(np.float32),
            demo=demo, rows_off=off.reshape(n, K, -1), rows_on=on.reshape(n, K, -1),
            act_off=D.from_rows(off, norm, bk.H).reshape(n, K, bk.H, 7), act_on=D.from_rows(on, norm, bk.H).reshape(n, K, bk.H, 7),
            demo_rows=D.to_rows(demo, norm), field=fld, loss_off=l_off, loss_on=l_on, goal=np.array(goal),
            kind=np.array(info["kind"]), key=np.array(info["key"]), **extra)
        del pre, preK
        torch.cuda.empty_cache()
        print(f"[h1d] {name} ({info['kind']}, {info['key']}): {n} frames from tasks {sorted(seen)}; loss own "
              f"{l_off[[r[3] == 1 for r in recs]].mean():.4f} -> {l_on[[r[3] == 1 for r in recs]].mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if len(eps) > 40:                                                          # keep host memory bounded
            eps.clear()
    print(f"[h1d] prototypes done in {time.time() - t_all:.0f}s", flush=True)
    subspaces(lib)


def subspaces(lib):
    """Pairwise subspace overlap of the prototypes' updates, per adapted Linear, saved to h1_donate/subspaces.npz."""
    import torch
    names = [n for n in lib if os.path.exists(os.path.join(R, f"prim_{n}_0.pt"))]
    Z = [torch.load(os.path.join(R, f"prim_{n}_0.pt"), map_location="cpu")["banks"] for n in names]
    mods = sorted(set.intersection(*[set(z) for z in Z]))
    P_, M = len(names), len(mods)
    ob, oa, fc = (np.zeros((M, P_, P_), np.float32) for _ in range(3))
    nrm = np.zeros((M, P_), np.float32)
    dims = np.zeros((M, 2), np.int64)
    for m, mod in enumerate(mods):
        Bs = [z[mod]["B"].float().cuda() for z in Z]              # [out, r]
        As = [z[mod]["A"].float().cuda() for z in Z]              # [r, in]
        dims[m] = (Bs[0].shape[0], As[0].shape[1])
        QB = torch.stack([torch.linalg.qr(b)[0] for b in Bs])                         # [P, out, r]
        QA = torch.stack([torch.linalg.qr(a.T)[0] for a in As])                       # [P, in, r]
        r = QB.shape[-1]
        ob[m] = (torch.einsum("pir,qis->pqrs", QB, QB) ** 2).sum((-1, -2)).cpu().numpy() / r
        oa[m] = (torch.einsum("pir,qis->pqrs", QA, QA) ** 2).sum((-1, -2)).cpu().numpy() / r
        BtB = torch.einsum("pir,qis->pqrs", torch.stack(Bs), torch.stack(Bs))         # B_p^T B_q
        AAt = torch.einsum("pri,qsi->pqrs", torch.stack(As), torch.stack(As))         # A_p A_q^T
        ip = torch.einsum("pqrs,qpsr->pq", BtB, AAt)                                  # <B_p A_p, B_q A_q> = tr(B_p^T B_q A_q A_p^T)
        d = torch.sqrt(torch.clamp(torch.diagonal(ip), min=1e-30))
        fc[m] = (ip / d[:, None] / d[None, :]).cpu().numpy()
        nrm[m] = d.cpu().numpy()
    np.savez_compressed(os.path.join(OUT, "subspaces.npz"), names=np.array(names), modules=np.array(mods), overlap_B=ob,
                        overlap_A=oa, frob_cos=fc, norm=nrm, dims=dims)
    print(f"[h1d] subspaces: {P_} prototypes x {M} modules", flush=True)


# ── CPU part: metrics, figures, ledger ───────────────────────────────────────────────────────────────────────────────
def _cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.sum(a * b, -1) / np.maximum(na * nb, 1e-12)


def proto_metrics(z):
    own = z["own"].astype(bool)
    off, on, demo = z["act_off"], z["act_on"], z["demo"]                  # [n, K, H, 7], [n, H, 7]
    ro, rn = z["rows_off"].reshape(*off.shape), z["rows_on"].reshape(*on.shape)   # normalised
    dr = rn - ro
    dmo = z["demo_rows"].reshape(len(demo), 1, *demo.shape[1:])
    vec = z["target"] - z["ee"]                                            # [n, 3] toward the phase's target
    far = np.linalg.norm(vec, axis=-1) > 0.02
    disp = lambda a: a[:, :, :5, :3].sum(2)                                # the 5 executed steps' position deltas
    d_off, d_on, d_demo = disp(off), disp(on), demo[:, None, :5, :3].sum(2)
    dd = d_on - d_off
    v = vec[:, None, :].repeat(off.shape[1], 1)
    m = {}
    for ph, sel in (("own", own), ("other", ~own)):
        s = sel & far
        e_tot = (dr[sel] ** 2).sum((-1, -2))
        m[ph] = dict(
            frames=int(sel.sum()),
            dnorm=float(np.sqrt((dr[sel] ** 2).sum((-1, -2))).mean()),                 # |chunk change|, normalised units
            share_pos=float(((dr[sel][..., :3] ** 2).sum((-1, -2)) / np.maximum(e_tot, 1e-12)).mean()),
            share_rot=float(((dr[sel][..., 3:6] ** 2).sum((-1, -2)) / np.maximum(e_tot, 1e-12)).mean()),
            share_grip=float(((dr[sel][..., 6:] ** 2).sum((-1, -2)) / np.maximum(e_tot, 1e-12)).mean()),
            err_off=float(np.sqrt(((ro[sel] - dmo[sel]) ** 2).sum((-1, -2))).mean()),     # distance to the demo's own chunk
            err_on=float(np.sqrt(((rn[sel] - dmo[sel]) ** 2).sum((-1, -2))).mean()),
            err_pos_off=float(np.sqrt(((ro[sel][..., :3] - dmo[sel][..., :3]) ** 2).sum((-1, -2))).mean()),
            err_pos_on=float(np.sqrt(((rn[sel][..., :3] - dmo[sel][..., :3]) ** 2).sum((-1, -2))).mean()),
            err_rot_off=float(np.sqrt(((ro[sel][..., 3:6] - dmo[sel][..., 3:6]) ** 2).sum((-1, -2))).mean()),
            err_rot_on=float(np.sqrt(((rn[sel][..., 3:6] - dmo[sel][..., 3:6]) ** 2).sum((-1, -2))).mean()),
            err_grip_off=float(np.sqrt(((ro[sel][..., 6:] - dmo[sel][..., 6:]) ** 2).sum((-1, -2))).mean()),
            err_grip_on=float(np.sqrt(((rn[sel][..., 6:] - dmo[sel][..., 6:]) ** 2).sum((-1, -2))).mean()),
            cos_delta_target=float(np.nanmean(_cos(dd[s], v[s]))) if s.any() else None,     # the CHANGE points at the target?
            cos_off_target=float(np.nanmean(_cos(d_off[s], v[s]))) if s.any() else None,
            cos_on_target=float(np.nanmean(_cos(d_on[s], v[s]))) if s.any() else None,
            cos_demo_target=float(np.nanmean(_cos(d_demo[s], v[s]))) if s.any() else None,
            loss_off=float(z["loss_off"][sel].mean()), loss_on=float(z["loss_on"][sel].mean()),
        )
        f = z["field"][:, :, sel]                                           # [T, 5, n]
        m[ph]["field"] = {str(t): dict(rel_change=float((f[j, 1] / np.maximum(f[j, 0], 1e-9)).mean()),
                                       err_off=float(f[j, 2].mean()), err_on=float(f[j, 3].mean()),
                                       cos_change_to_target=float(f[j, 4].mean())) for j, t in enumerate(TS)}
    # gripper timing: frames where the demo's gripper changes inside the chunk (own phase)
    ev_close = not bool(z["goal"])
    def first_change(g):                                                    # g [..., H] gripper channel
        hit = (g > 0) if ev_close else (g < 0)
        return np.where(hit.any(-1), hit.argmax(-1), g.shape[-1])
    gd = first_change(demo[..., 6])
    sel = own & (gd < demo.shape[1]) & (gd > 0)
    if sel.any():
        go, gn = first_change(off[..., 6])[sel], first_change(on[..., 6])[sel]
        m["timing"] = dict(frames=int(sel.sum()), event="close" if ev_close else "open",
                           abs_err_off=float(np.abs(go - gd[sel][:, None]).mean()),
                           abs_err_on=float(np.abs(gn - gd[sel][:, None]).mean()))
    return m


def subspace_metrics(lib):
    f = os.path.join(OUT, "subspaces.npz")
    if not os.path.exists(f):
        return None
    s = np.load(f, allow_pickle=True)
    names = [str(x) for x in s["names"]]
    kind = {n: lib[n]["kind"] for n in names}
    r = 4
    chanceB = float(np.mean(r / s["dims"][:, 0]))
    chanceA = float(np.mean(r / s["dims"][:, 1]))
    oB, oA, fc = s["overlap_B"].mean(0), s["overlap_A"].mean(0), s["frob_cos"].mean(0)
    P_ = len(names)
    groups = {}
    for i in range(P_):
        for j in range(i + 1, P_):
            a, b = names[i], names[j]
            ka, kb = kind[a], kind[b]
            if {ka, kb} == {"goal"}:
                g = "goal-goal"
            elif "goal" in (ka, kb):
                g = "object-goal"
            elif "instance" in (ka, kb):
                ia, ib = (a, b) if ka == "instance" else (b, a)
                pa = parent_of(ia, lib)
                if kind[ib] == "object" and ib == pa:
                    g = "instance-its type"
                elif kind[ib] == "instance" and parent_of(ib, lib) == pa:
                    g = "instance-sibling instance"
                else:
                    g = "instance-other object"
            else:
                g = "object-object (types)"
            groups.setdefault(g, []).append((i, j))
    out = dict(chance_B=chanceB, chance_A=chanceA, n_modules=int(len(s["modules"])), prototypes=P_)
    for g, ps in groups.items():
        ii = np.array(ps)
        out[g] = dict(pairs=len(ps), overlap_B=float(oB[ii[:, 0], ii[:, 1]].mean()), overlap_A=float(oA[ii[:, 0], ii[:, 1]].mean()),
                      frob_cos=float(fc[ii[:, 0], ii[:, 1]].mean()), frob_cos_abs=float(np.abs(fc[ii[:, 0], ii[:, 1]]).mean()),
                      overlap_B_ratio_to_chance=float(oB[ii[:, 0], ii[:, 1]].mean() / chanceB))
    # where the updates live: share of ||BA||_F^2 per layer, objects vs goals
    mods = [str(x) for x in s["modules"]]
    import re
    layer = np.array([int(re.search(r"layers\.(\d+)\.", m).group(1)) if re.search(r"layers\.(\d+)\.", m) else -1 for m in mods])
    part = np.array(["attn" if "self_attn" in m else "mlp" if "mlp" in m else "other" for m in mods])
    nrm2 = s["norm"] ** 2
    for k in ("object", "instance", "goal"):
        cols = [i for i, n in enumerate(names) if kind[n] == k]
        if not cols:
            continue
        tot = nrm2[:, cols].sum(0)
        out[f"layers_{k}"] = [float((nrm2[layer == L][:, cols].sum(0) / tot).mean()) for L in range(int(layer.max()) + 1)]
        out[f"attn_share_{k}"] = float((nrm2[part == "attn"][:, cols].sum(0) / tot).mean())
    out["_matrices"] = dict(names=names, overlap_B=oB, frob_cos=fc)
    return out


def read(record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lib = library()
    per, missing = {}, []
    for n, info in lib.items():
        f = os.path.join(OUT, f"{n}.npz")
        if not os.path.exists(f):
            missing.append(n)
            continue
        z = np.load(f, allow_pickle=True)
        per[n] = dict(kind=info["kind"], key=info["key"], **proto_metrics(z))
    kinds = ("object", "instance", "goal")
    agg = {}
    for k in kinds:
        ps = [v for v in per.values() if v["kind"] == k]
        if not ps:
            continue
        a = dict(prototypes=len(ps))
        for ph in ("own", "other"):
            a[ph] = {m: float(np.mean([p[ph][m] for p in ps if p[ph].get(m) is not None]))
                     for m in ps[0][ph] if m not in ("field", "frames")}
            a[ph]["field"] = {t: {m: float(np.mean([p[ph]["field"][t][m] for p in ps])) for m in ps[0][ph]["field"][t]}
                              for t in ps[0][ph]["field"]}
            a[ph]["loss_drop_share"] = float(np.mean([1 - p[ph]["loss_on"] / p[ph]["loss_off"] for p in ps]))
            a[ph]["n_loss_lower"] = int(sum(p[ph]["loss_on"] < p[ph]["loss_off"] for p in ps))
            a[ph]["n_delta_toward"] = int(sum((p[ph]["cos_delta_target"] or 0) > 0 for p in ps))
        tm = [p["timing"] for p in ps if "timing" in p]
        if tm:
            a["timing"] = dict(prototypes=len(tm), abs_err_off=float(np.mean([t["abs_err_off"] for t in tm])),
                               abs_err_on=float(np.mean([t["abs_err_on"] for t in tm])),
                               n_better=int(sum(t["abs_err_on"] < t["abs_err_off"] for t in tm)))
        agg[k] = a
    sub = subspace_metrics(lib)
    # scale sweep
    sweep = {}
    for n in SWEEP:
        f = os.path.join(OUT, f"{n}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        if "sweep_actions" not in z.files:
            continue
        own = z["own"].astype(bool)
        sa, sc = z["sweep_actions"], [float(x) for x in z["scales"]]
        i0, i1 = sc.index(0.0), sc.index(1.0)
        base, ref = sa[i0][own], sa[i1][own] - sa[i0][own]
        vec = (z["target"] - z["ee"])[own][:, None, :].repeat(sa.shape[2], 1)
        rows = []
        for j, w in enumerate(sc):
            d = sa[j][own] - base
            rows.append(dict(scale=w, dnorm=float(np.sqrt((d ** 2).sum((-1, -2))).mean()),
                             cos_to_unit=float(np.nanmean(_cos(d.reshape(len(d), -1, d.shape[-2] * 7),
                                                               ref.reshape(len(ref), -1, ref.shape[-2] * 7)))) if w != 0 else None,
                             cos_disp_target=float(np.nanmean(_cos(sa[j][own][:, :, :5, :3].sum(2), vec))),
                             loss=float(z["sweep_loss"][j][own].mean())))
        sweep[n] = dict(kind=lib[n]["kind"], key=lib[n]["key"], rows=rows)
    res = dict(missing=missing, by_kind=agg, per_prototype=per, scale_sweep=sweep,
               subspaces={k: v for k, v in (sub or {}).items() if k != "_matrices"})
    os.makedirs(FIG, exist_ok=True)
    COL = dict(object="#1d5e86", instance="#8b5cf6", goal="#a35d2b")
    # figure 1: per-phase flow loss with and without, and the direction of the change
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k in kinds:
        ps = [p for p in per.values() if p["kind"] == k]
        if not ps:
            continue
        ax[0].scatter([p["own"]["loss_off"] for p in ps], [p["own"]["loss_on"] for p in ps], color=COL[k], label=f"{k}: own phase",
                      s=22)
        ax[0].scatter([p["other"]["loss_off"] for p in ps], [p["other"]["loss_on"] for p in ps], facecolors="none",
                      edgecolors=COL[k], label=f"{k}: other phase", s=22)
    lim = [0, max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
    ax[0].plot(lim, lim, color="#9ca3af", lw=1)
    ax[0].set_xlabel("frozen policy's flow loss, prototype OFF")
    ax[0].set_ylabel("flow loss, prototype ON")
    ax[0].set_title("flow loss on held-out demo frames", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)
    labels, xs = [], []
    for j, k in enumerate(kinds):
        ps = [p for p in per.values() if p["kind"] == k]
        for i, (m, lab) in enumerate((("cos_off_target", "policy alone"), ("cos_on_target", "with prototype"),
                                      ("cos_delta_target", "the change"), ("cos_demo_target", "demonstration"))):
            vals = [p["own"][m] for p in ps if p["own"][m] is not None]
            x = j * 5 + i
            ax[1].bar(x, np.mean(vals), color=COL[k], alpha=[0.35, 0.7, 1.0, 0.2][i])
            ax[1].errorbar(x, np.mean(vals), yerr=np.std(vals) / max(1, np.sqrt(len(vals))), color="#111827", lw=1)
            labels.append(f"{k}\n{lab}"); xs.append(x)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels, fontsize=6, rotation=90)
    ax[1].axhline(0, color="#9ca3af", lw=1)
    ax[1].set_ylabel("cosine with (target - end effector)")
    ax[1].set_title("does the executed displacement point at the phase's target?\n(object: grasp point; goal: release point)",
                    fontsize=9)
    w = 0.25
    for j, k in enumerate(kinds):
        if k not in agg:
            continue
        o = agg[k]["own"]
        ax[2].bar(np.arange(3) + (j - 1) * w, [1 - o[f"err_{d}_on"] / o[f"err_{d}_off"] for d in ("pos", "rot", "grip")], w,
                  color=COL[k], label=k)
    ax[2].axhline(0, color="#9ca3af", lw=1)
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(["position (WHERE)", "orientation (HOW)", "gripper (WHEN)"], fontsize=8)
    ax[2].set_ylabel("error to the held-out demo's chunk removed\n(1 - err with / err without the prototype)")
    ax[2].set_title("what the prototype corrects (own phase)", fontsize=10)
    ax[2].legend(fontsize=7, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h1_donate.png"), dpi=150)
    plt.close(fig)
    # figure 2: subspaces
    if sub:
        M = sub["_matrices"]
        order = sorted(range(len(M["names"])), key=lambda i: (kinds.index(lib[M["names"][i]]["kind"]), M["names"][i]))
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
        im = ax[0].imshow(M["overlap_B"][np.ix_(order, order)], cmap="viridis", vmin=0, vmax=max(0.05, float(np.sort(
            M["overlap_B"][np.triu_indices(len(order), 1)])[-3])))
        nm = [M["names"][i] for i in order]
        ax[0].set_xticks(range(len(nm)))
        ax[0].set_xticklabels(nm, fontsize=5, rotation=90)
        ax[0].set_yticks(range(len(nm)))
        ax[0].set_yticklabels(nm, fontsize=5)
        ax[0].set_title(f"output-subspace overlap (mean cos^2 of principal angles of B), chance {sub['chance_B']:.4f}", fontsize=8)
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        gs = [g for g in ("instance-its type", "instance-sibling instance", "instance-other object", "object-object (types)",
                          "goal-goal", "object-goal") if g in sub]
        ax[1].bar(range(len(gs)), [sub[g]["overlap_B_ratio_to_chance"] for g in gs], color="#1d5e86")
        for i, g in enumerate(gs):
            ax[1].text(i, sub[g]["overlap_B_ratio_to_chance"], f"n={sub[g]['pairs']}\ncos {sub[g]['frob_cos']:.3f}", ha="center",
                       va="bottom", fontsize=7)
        ax[1].axhline(1, color="#9ca3af", lw=1, ls="--")
        ax[1].set_xticks(range(len(gs)))
        ax[1].set_xticklabels(gs, fontsize=7, rotation=25)
        ax[1].set_ylabel("output-subspace overlap / chance (rank / dimension)")
        ax[1].set_title("which prototypes share a subspace (mean over the action expert's Linears)", fontsize=9)
        ax[1].spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "h1_subspaces.png"), dpi=150)
        plt.close(fig)
    # figure 3: scale sweep
    if sweep:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for n, s in sweep.items():
            sc = [r["scale"] for r in s["rows"]]
            ax[0].plot(sc, [r["dnorm"] for r in s["rows"]], marker="o", label=f"{n} ({s['kind']}: {s['key']})", color=COL[s["kind"]],
                       alpha=0.6 if n in ("Sch2", "XG1") else 1.0, ls="--" if n in ("Sch2", "XG1") else "-")
            ax[1].plot(sc, [r["loss"] for r in s["rows"]], marker="o", color=COL[s["kind"]], ls="--" if n in ("Sch2", "XG1") else "-")
            ax[2].plot(sc, [r["cos_disp_target"] for r in s["rows"]], marker="o", color=COL[s["kind"]],
                       ls="--" if n in ("Sch2", "XG1") else "-")
        for a, yl in zip(ax, ("|chunk change| from weight 0 (normalised)", "flow loss on own-phase frames",
                              "cos(executed displacement, target - ee)")):
            a.set_xlabel("prototype weight (1 = as fitted)")
            a.set_ylabel(yl, fontsize=8)
            a.axvline(1, color="#9ca3af", lw=1, ls=":")
            a.spines[["top", "right"]].set_visible(False)
        ax[0].legend(fontsize=6, frameon=False)
        fig.suptitle("scale sweep on held-out demo frames: the action change grows with the weight; the flow loss is lowest at weight 0.5-0.75", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "h1_scale.png"), dpi=150)
        plt.close(fig)
    json.dump(res, open(os.path.join(R, "h1_donate.json"), "w"), indent=1)
    for k, a in agg.items():
        o, t = a["own"], a["other"]
        print(f"[h1d] {k:8} n={a['prototypes']:2}: own loss {o['loss_off']:.4f} -> {o['loss_on']:.4f} (lower on {o['n_loss_lower']}); "
              f"other {t['loss_off']:.4f} -> {t['loss_on']:.4f}; |change| own {o['dnorm']:.2f} other {t['dnorm']:.2f}; shares pos "
              f"{o['share_pos']:.2f} rot {o['share_rot']:.2f} grip {o['share_grip']:.2f}; cos change->target {o['cos_delta_target']:.3f}"
              f" (off {o['cos_off_target']:.3f}, on {o['cos_on_target']:.3f}, demo {o['cos_demo_target']:.3f}); "
              f"timing {a.get('timing')}", flush=True)
    if sub:
        print(json.dumps({k: v for k, v in sub.items() if k != "_matrices"}, indent=1)[:3000], flush=True)
    for n, s in sweep.items():
        print(f"[h1d] sweep {n}:", [(r["scale"], round(r["dnorm"], 2), round(r["loss"], 4), round(r["cos_disp_target"], 3))
                                    for r in s["rows"]], flush=True)
    print(f"[h1d] missing {missing}", flush=True)
    if record and not missing:
        from lifelong_ood import ledger
        z0 = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
        print(ledger.record("h1_donate", bench="libero_90", tag="read", seed=0, metrics=res, bytes=z0,
                            config=dict(K=K, TS=TS, sweep=SWEEP, scales=SCALES, maxtasks=MAXTASKS, demo="episodes_of_task(t)[1]"),
                            script="experiments/h1_donate.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    if "--subspaces" in sys.argv:
        subspaces(library())
    if "--read" in sys.argv:
        read("--record" in sys.argv)
