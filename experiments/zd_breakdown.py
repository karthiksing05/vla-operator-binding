"""Plan D, D1 (docs/PLAN_D.md): the ACTION BREAKDOWN -- what a prototype donates and what the one demonstration teaches.
Exploratory, offline (no rollouts).  pi0.5, Z3's 25 LIBERO-90 tasks (both concepts held: pointers, library prototypes, the soft
blend and the per-task leaf all exist).

THREE NESTED SOURCES per action chunk:
    a_policy   the frozen policy under the task's instruction
    a_proto    a composition of held fields, nothing fitted for the task in the network:
                 ptr    Z3's pointers (pfield@a.g): the policy under training instruction a before the grasp, g from the grasp
                        to the release (supervised: the sources hold the task's object / goal concept by name)
                 lib    G8's library (fac@o.g): the object prototype before the grasp, the goal prototype from it to the release
                        (supervised; the prototypes were fitted on demonstration 0 of their tasks, the task's own among them)
                 blend  Z5's soft blend (blend@l90_t; lifelong_ood/policy/blend.py): per contact segment and dim group a convex
                        mix of held fields, the gripper from the segment's best field (unsupervised: units found by likelihood)
                 prog   the unsupervised pointer program (program.py): per contact segment the training instruction with the
                        lowest flow loss on demonstration 0's segment (U1 scores), a hard switch
    a_demo     the leaf fitted on the demonstration ON TOP of a_proto: a fresh rank-16 KeyedLoRA (B = 0, so step 0 IS a_proto)
               switched on in every field of the composition, fitted by flow matching through the composed field.  The same
               fit on a_policy is the plain one-demonstration LoRA (r3_fewshot's recipe: rank 16, lr 1e-4, batch 32, AdamW).
               A leaf cannot copy the weights of a two-prototype composition (--init-best copies ONE unit), and pointers / the
               blend have no weights, so "initialised from a_proto" is implemented as the residual leaf, exact at step 0.
    donation = a_proto - a_policy, taught = a_demo - a_proto.

Phase conditioning offline follows each serving hook's own rule, driven by the demonstration's sustained gripper events
(segment.gripper_truth): PrimHooks (ptr, lib: carrying from a grasp to the next release), ProgHooks / BlendHooks (the segment index
advances at every event, capped).  Training demonstrations: 1 = [0], 2 = [0, 2], 5 = [0, 2, 3, 4, 5]; held out: demonstration 1
(no unit, blend, program or leaf has seen it).  Frames: every 2nd frame (train and held-out, all sources alike).
Readouts per checkpoint (fit steps 0 50 100 200 400 800), on the held-out frames:
  * the sampled chunk (Euler, 10 steps; K = 4 noise draws shared by every source and checkpoint), normalised action units;
  * the composed velocity at x_t = t eps + (1 - t) a_demo1, t in 0.9 / 0.5 / 0.1 (shared eps), and its target u;
  * the held-out flow-matching loss (adapt.fixed_draws, 256 draws, the same for every source).
The existing served leaf (lorasc: r3_lora_<t>[_sc].pt, 1,000 steps from a_policy) is scored once as a reference.

    sbatch ... SCRIPT="experiments/zd_breakdown.py --run --task 31 --group n1"      (one job per task and group; resumable)
    $PY experiments/zd_breakdown.py --launch [--groups n1+n2+n5]   (submits the missing jobs)
    $PY experiments/zd_breakdown.py --read [--record]              (CPU: shares, curves, figures, ledger zd_breakdown)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(R, "zd")
FIG = os.path.join(ROOT, "figures", "robot")

# the 25 Z3 tasks with their grade (results/robot/split_grades.json) -- fixed here so the CPU read needs no dataset access
TASKS = (31, 15, 25, 10, 9, 13, 12, 14, 72, 30, 54, 52, 53, 48, 50, 46, 49, 19, 24, 26, 38, 67, 68, 70, 77)
SOURCES = ("policy", "ptr", "lib", "prog", "blend")
CURVE_SOURCES = ("policy", "ptr", "lib", "blend")
STEPS = (0, 50, 100, 200, 400, 800)
TRAIN_DEMOS = {1: (0,), 2: (0, 2), 5: (0, 2, 3, 4, 5)}
HELD = 1
STRIDE, K, TS = 2, 4, (0.9, 0.5, 0.1)
RANK, LR, BATCH = 16, 1e-4, 32
GROUPS = {"n1": (1, SOURCES), "n2": (2, CURVE_SOURCES), "n5": (5, CURVE_SOURCES)}
GPU_PREFIXES = 450                       # training banks up to this many cached prefixes live on the GPU, larger ones in host memory
COMP = {"translation": [0, 1, 2], "rotation": [3, 4, 5], "gripper": [6]}
PHASES = ("approach", "grasp", "carry", "release", "after")
WIN = 5                                  # frames on each side of a sustained grasp / release that form its phase


def grades():
    g = json.load(open(os.path.join(R, "split_grades.json")))["grades"]
    return {t: g[str(t)]["grade"] for t in TASKS}


def curve_tasks():
    """~12 tasks across grades, by a fixed rule: every G4 task, and evenly spaced task ids (sorted) within G0 (3), G2 (3), G3 (4)."""
    gr = grades()
    out = []
    for g, k in (("G0", 3), ("G2", 3), ("G3", 4), ("G4", 99)):
        ts = sorted(t for t in TASKS if gr[t] == g)
        if k >= len(ts):
            out += ts
            continue
        out += [ts[i] for i in np.unique(np.round(np.linspace(0, len(ts) - 1, k)).astype(int))]
    return sorted(out)


def fpath(t, src, nd):
    return os.path.join(OUT, f"{t}_{src}_n{nd}.npz")


# ── GPU part ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def task_meta(t):
    """The task's sources: pointer instructions, library prototypes, blend row, program instructions (all from existing files)."""
    import z3_l90_pointers as Z3
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    from lifelong_ood.policy import program as PG
    import importlib.util
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    tx40 = D.Demos().tasks
    text = texts[t]
    ptr = Z3.plan()[str(t)]
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    e5 = json.load(open(os.path.join(R, "e5_plan.json")))["tasks"]
    v = e3[str(t)] if str(t) in e3 and e3[str(t)]["lora"] else e5[str(t)]
    lib = (v["object_proto"], v.get("goal_proto") or v["goal"])
    z = np.load(os.path.join(R, "u1", f"l90_{t}_d0.npz"))
    L = np.log(((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2))
    prog = PG.label(L, z["frames"], PG.segments(z["actions"]), list(range(40)))
    rows = json.load(open(os.path.join(R, "zi_blend.json")))["rows"]
    brow = next(r for r in rows if r["name"] == f"l90_{t}")
    from lifelong_ood import ledger
    wrong = set(ledger.load("l90_scenes", tag="map")[-1]["metrics"]["wrong_scene_tasks"])
    lorasc = os.path.join(R, f"r3_lora_{t}_sc.pt" if t in wrong else f"r3_lora_{t}.pt")
    return dict(task=t, text=text, clause=P.steps_of(text)[0], n_steps=len(P.steps_of(text)), grade=grades()[t],
                ptr=[int(ptr[0]), int(ptr[1])], ptr_text=[tx40[int(ptr[0])], tx40[int(ptr[1])]], lib=list(lib),
                prog=[int(i) for i in prog], prog_text=[tx40[int(i)] for i in prog], blend=brow, lorasc=lorasc)


def events(actions):
    from lifelong_ood.policy import segment as S
    g, r = S.gripper_truth(actions)
    return sorted([(int(f), "g") for f in g] + [(int(f), "r") for f in r])


def hook_state(ev, f):
    """-> (number of events at or before frame f, carrying: the last such event is a grasp)."""
    past = [e for e in ev if e[0] <= f]
    return len(past), bool(past and past[-1][1] == "g")


def blend_codes(row):
    """BlendHooks' per-segment weights (experts sorted by id; the gripper one-hot on the segment's best field) -> (ids, [S, E, 70])."""
    from lifelong_ood.policy import blend as BL
    ids = sorted(int(e) for e in row["expert_ids"])
    segs = row["segments"]
    order = [s for s in BL.SEGS if s in segs] or ["whole"]
    W = np.zeros((len(order), len(ids), 10, 7), np.float32)
    for k, s in enumerate(order):
        seg = segs[s]
        for g, dims in BL.GROUP.items():
            w = np.array([float(seg["beta"][g].get(str(e), 0.0)) for e in ids], np.float32)
            w = w / w.sum()
            W[k][:, :, list(dims)] = w[:, None, None]
        gl = {int(e): float(v) for e, v in seg["held"]["gripper"]["single"].items()}
        W[k, min(range(len(ids)), key=lambda i: gl.get(ids[i], np.inf)), :, 6] = 1.0
    prompts = {int(e): p for e, p in zip(row["expert_ids"], row["experts"])}
    return ids, [prompts[e] for e in ids], W.reshape(len(order), len(ids), 70)


def slots_of(src, meta, frames, keys):
    """frames: [(episode dict, frame, events)].  -> list of slots dict(prompts [n], keys {key: weights [n]}, B [n, 70] or None);
    the composed velocity is sum_j B_j * v_j on the real dims (padded dims follow slot 0), each v_j under its prompt and keys."""
    n = len(frames)
    st = [hook_state(ev, f) for _, f, ev in frames]
    if src in ("policy", "lorasc"):
        return [dict(prompts=[meta["text"]] * n, keys={}, B=None)]
    if src == "ptr":
        return [dict(prompts=[meta["ptr_text"][1] if c else meta["ptr_text"][0] for _, c in st], keys={}, B=None)]
    if src == "lib":
        car = np.array([c for _, c in st], np.float32)
        return [dict(prompts=[meta["clause"]] * n, keys={keys["obj"]: 1.0 - car, keys["goal"]: car}, B=None)]
    if src == "prog":
        p = meta["prog_text"]
        return [dict(prompts=[p[min(k, len(p) - 1)] for k, _ in st], keys={}, B=None)]
    if src == "blend":
        ids, prompts, W = blend_codes(meta["blend"])
        seg = np.array([min(k, len(W) - 1) for k, _ in st])
        out = []
        for j in range(len(ids)):
            B = W[seg, j]
            if j > 0 and not B.any():
                continue
            out.append(dict(prompts=[prompts[j]] * n, keys={}, B=B))
        return out
    raise ValueError(src)


def run(task, group, smoke=False):
    import torch
    from lifelong_ood.experts.lora import add_node, drop_node, keys_on
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    import c0_fit as C0
    os.makedirs(OUT, exist_ok=True)
    nd, srcs = GROUPS[group]
    steps = (0, 2, 4) if smoke else STEPS
    out_of = (lambda s: os.path.join(OUT, f"smoke_{task}_{s}_n{nd}.npz")) if smoke else (lambda s: fpath(task, s, nd))
    todo = [s for s in srcs if not os.path.exists(out_of(s))]
    lor = os.path.join(OUT, f"{task}_lorasc.npz")
    want_lorasc = group == "n1" and not smoke and not os.path.exists(lor)
    if not todo and not want_lorasc:
        print(f"[zd] task {task} {group}: done", flush=True)
        return
    T0 = time.time()
    meta = task_meta(task)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    norm = D.Norm.load()
    d90 = D.Demos90()
    eps_of = d90.episodes_of_task(task)
    state, banks = adapt.inject(bk)
    keys = {}
    for nm, which in ((meta["lib"][0], "obj"), (meta["lib"][1], "goal")):
        state, keys[which], _ = adapt.load(bk, os.path.join(R, f"prim_{nm}_0.pt"), state=state)
    real = torch.as_tensor(bk.real, device=bk.device)

    def episode(k):
        e = d90.load(eps_of[k])
        e["image"], e["wrist_image"] = d90.frames(e["episode_index"])
        return e

    def frames_of(demos):
        fr, rows = [], []
        for k in demos:
            ep = episode(k)
            T = len(ep["actions"])
            ev = events(ep["actions"])
            fs = list(range(0, T, STRIDE))
            idx, _ = D.chunk_indices(T, bk.H)
            rows.append(D.pad_rows(C0.chunk_rows(ep, idx[fs], fs, norm), bk.H, bk.a, bk.A))
            fr += [(ep, f, ev) for f in fs]
        return fr, np.concatenate(rows, 0).astype(np.float32)

    def banks_of(slots, frames, store):
        for s in slots:
            obs = []
            for (ep, f, _), p in zip(frames, s["prompts"]):
                o = d90.observation(ep, f)
                o["prompt"] = p
                obs.append(o)
            s["bank"] = adapt.PrefixBank(bk, obs, store=store)
            s["Bt"] = None if s["B"] is None else torch.as_tensor(s["B"], device=bk.device)
            s["Kt"] = {k: torch.as_tensor(w, device=bk.device) for k, w in s["keys"].items()}
        return slots

    def vmix(x, t, slots, idx, leaf, pres=None):
        """The composed velocity [n, H * A] at x [n, H, A], t [n] for bank rows idx (pres: the slots' gathered prefixes)."""
        it = torch.as_tensor(np.asarray(idx), device=bk.device, dtype=torch.long)
        v0, acc = None, 0.0
        for j, s in enumerate(slots):
            on = {k: w.index_select(0, it) for k, w in s["Kt"].items()}
            if leaf is not None:
                on[leaf] = 1.0
            with keys_on(state, on)():
                v = bk.v_full(x, t, s["bank"].take(idx) if pres is None else pres[j])
            if s["Bt"] is None:
                return v
            v0 = v if v0 is None else v0
            acc = acc + s["Bt"].index_select(0, it) * v[:, real]
        out = v0.clone()
        out[:, real] = acc
        return out

    held, A_held = frames_of([HELD])
    nh = len(held)
    gnoise = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((nh * K, bk.dim_full), generator=gnoise)
    gv = torch.Generator(device="cpu").manual_seed(1)
    vnoise = torch.randn((len(TS), nh, bk.dim_full), generator=gv)
    draws = adapt.fixed_draws(nh, bk.H, bk.A)
    Ah = torch.as_tensor(A_held, device=bk.device)
    held_info = dict(frames=np.array([f for _, f, _ in held]), T=len(held[0][0]["actions"]),
                     events=np.array(held[0][2], dtype=object), demo_rows=A_held.reshape(nh, -1)[:, bk.real])

    @torch.no_grad()
    def evaluate(slots, leaf):
        rows = np.zeros((nh, K, 70), np.float32)
        for i in range(0, nh, 16):
            fr = np.arange(i, min(nh, i + 16))
            idx = np.repeat(fr, K)
            x = noise[i * K:(i + len(fr)) * K].to(bk.device).reshape(len(idx), bk.H, bk.A)
            pres = [s["bank"].take(idx) for s in slots]
            t = 1.0
            while t >= 0.05:
                v = vmix(x, torch.full((len(idx),), t, device=bk.device), slots, idx, leaf, pres)
                x = x - 0.1 * v.reshape(x.shape)
                t -= 0.1
            rows[fr] = x.reshape(len(idx), -1)[:, real].float().cpu().numpy().reshape(len(fr), K, 70)
        V = np.zeros((nh, len(TS), 70), np.float32)
        U = np.zeros((nh, len(TS), 70), np.float32)
        for j, tv in enumerate(TS):
            for i in range(0, nh, 32):
                idx = np.arange(i, min(nh, i + 32))
                e = vnoise[j, idx].to(bk.device).reshape(len(idx), bk.H, bk.A)
                a = Ah[idx]
                xt = tv * e + (1 - tv) * a
                v = vmix(xt, torch.full((len(idx),), tv, device=bk.device), slots, idx, leaf)
                V[idx, j] = v[:, real].float().cpu().numpy()
                U[idx, j] = (e - a).reshape(len(idx), -1)[:, real].float().cpu().numpy()
        tot, n = 0.0, 0
        for idx, tt, ee in draws:
            tt, ee = tt.to(bk.device), ee.to(bk.device)
            a = Ah[idx]
            xt = tt[:, None, None] * ee + (1 - tt[:, None, None]) * a
            v = vmix(xt, tt, slots, idx, leaf)
            tot += float(((v - (ee - a).reshape(len(idx), -1)) ** 2).mean()) * len(idx)
            n += len(idx)
        return rows, V, U, tot / n

    def save(path, evs, extra):
        np.savez_compressed(path, steps=np.array(sorted(evs)), rows=np.stack([evs[s][0] for s in sorted(evs)]),
                            v=np.stack([evs[s][1] for s in sorted(evs)]), u=evs[sorted(evs)[0]][2],
                            loss=np.array([evs[s][3] for s in sorted(evs)]), meta=np.array(json.dumps(
                                {k: v for k, v in meta.items() if k != "blend"})), **held_info, **extra)

    if smoke:                                    # parity: the composed sampler on the policy alone is bk.sample bit for bit
        hs = banks_of(slots_of("policy", meta, held, keys), held, None)
        with torch.no_grad():
            ref = bk.sample(4 * K, hs[0]["bank"].take(np.repeat(np.arange(4), K)), noise=noise[:4 * K])
        mine = evaluate(hs, None)[0][:4].reshape(4 * K, 70)
        print(f"[zd] smoke parity: max |composed - bk.sample| = {np.abs(mine - ref).max():.2e} (|ref| {np.abs(ref).mean():.3f})",
              flush=True)
        del hs
    if want_lorasc and os.path.exists(meta["lorasc"]):
        hs = banks_of(slots_of("policy", meta, held, keys), held, "cpu")
        state, lk, _ = adapt.load(bk, meta["lorasc"], state=state)
        r_ = evaluate(hs, lk)
        drop_node(banks, lk)
        save(lor, {1000: r_}, dict(path=np.array(meta["lorasc"])))
        print(f"[zd] task {task}: lorasc held-out loss {r_[3]:.4f}", flush=True)
        del hs
    if todo:
        train, A_train = frames_of(TRAIN_DEMOS[nd])
        At = torch.as_tensor(A_train, device=bk.device)
    for src in todo:
        t0 = time.time()
        hs = banks_of(slots_of(src, meta, held, keys), held, "cpu")
        sl = slots_of(src, meta, train, keys)
        store = None if len(sl) * len(train) <= GPU_PREFIXES else "cpu"
        ts = banks_of(sl, train, store)
        t_cache = time.time() - t0
        leaf = f"zd_{task}_{src}_n{nd}"
        torch.manual_seed(0)
        params = add_node(banks, leaf, RANK)
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        gen = torch.Generator(device="cpu").manual_seed(0)
        rng = np.random.default_rng(0)
        evs, curve = {0: evaluate(hs, leaf)}, []
        for s in range(1, max(steps) + 1):
            idx = rng.integers(0, len(train), size=min(BATCH, len(train)))
            t, e = adapt._flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
            a = At[idx]
            xt = t[:, None, None] * e + (1 - t[:, None, None]) * a
            u = (e - a).reshape(len(idx), -1)
            opt.zero_grad(set_to_none=True)
            if len(ts) == 1:
                v = vmix(xt, t, ts, idx, leaf)
                loss = ((v - u) ** 2).mean()
                loss.backward()
            else:                                   # several fields: the exact gradient, one field's graph at a time
                pres = [sj["bank"].take(idx) for sj in ts]
                with torch.no_grad():                # (dL/dv_mix from a no-grad pass, then each field's B_j v_j . g backward)
                    v = vmix(xt, t, ts, idx, leaf, pres)
                    loss = ((v - u) ** 2).mean()
                    g = 2.0 * (v - u) / v.numel()
                for j, sj in enumerate(ts):
                    it = torch.as_tensor(np.asarray(idx), device=bk.device, dtype=torch.long)
                    on = {k: w.index_select(0, it) for k, w in sj["Kt"].items()}
                    on[leaf] = 1.0
                    with keys_on(state, on)():
                        vj = bk.v_full(xt, t, pres[j])
                    w = sj["Bt"].index_select(0, it) * g[:, real]
                    if j == 0:                       # the padded dims follow field 0
                        pad = torch.ones(vj.shape[1], dtype=torch.bool, device=bk.device)
                        pad[real] = False
                        ((vj[:, real] * w).sum() + (vj[:, pad] * g[:, pad]).sum()).backward()
                    else:
                        (vj[:, real] * w).sum().backward()
                    del vj
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if s % 50 == 0 or s == 1:
                curve.append((s, float(loss)))
            if s in steps:
                evs[s] = evaluate(hs, leaf)
                print(f"[zd] task {task} {src} n{nd} step {s}: train {float(loss):.4f} held-out {evs[s][3]:.4f} "
                      f"(0: {evs[0][3]:.4f}; {time.time() - t0:.0f}s)", flush=True)
        if nd == 1 and not smoke:                  # the 800-step leaf, kept for the rollout stage (prereg zd_rollouts); never fatal
            try:
                adapt.save(banks, leaf, os.path.join(OUT, f"leaf_{task}_{src}_n1.pt"),
                           meta=dict(source=src, task=task, steps=max(steps), rank=RANK, lr=LR, batch=BATCH, train_demos=TRAIN_DEMOS[nd]))
            except Exception as ex:                # noqa: BLE001
                print(f"[zd] leaf save failed: {ex}", flush=True)
        drop_node(banks, leaf)
        save(out_of(src), evs, dict(curve=np.array(curve), slots=np.array(len(sl)), n_train=np.array(len(train)),
                                    secs=np.array([t_cache, time.time() - t0])))
        print(f"[zd] task {task} {src} n{nd}: {len(sl)} fields, {len(train)} train / {nh} held-out frames, cache {t_cache:.0f}s, "
              f"total {time.time() - t0:.0f}s", flush=True)
        del hs, ts, sl, opt, params
        torch.cuda.empty_cache()
    print(f"[zd] task {task} {group}: all done in {time.time() - T0:.0f}s", flush=True)


def launch(groups):
    cur = curve_tasks()
    for g in groups:
        nd, srcs = GROUPS[g]
        for t in (TASKS if g == "n1" else cur):
            if all(os.path.exists(fpath(t, s, nd)) for s in srcs) and (g != "n1" or os.path.exists(os.path.join(OUT, f"{t}_lorasc.npz"))):
                continue
            mem = "120G" if g == "n5" else "80G"
            cmd = ["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short", "--time=04:00:00",
                   "--gres=gpu:a40:1", f"--mem={mem}", "--exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw", f"--job-name=zd{g}_{t}",
                   f"--export=ALL,ENV=openpi,SCRIPT=experiments/zd_breakdown.py --run --task {t} --group {g}",
                   "slurm/jobs/robot_gpu.sbatch"]
            o = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            print(t, g, (o.stdout or o.stderr).strip(), flush=True)


# ── CPU part ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def load(t, src, nd):
    f = fpath(t, src, nd) if src != "lorasc" else os.path.join(OUT, f"{t}_lorasc.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def phase_of(frames, ev, T):
    """Frame -> phase: approach / grasp (WIN frames around the first sustained grasp) / carry / release (around the first release
    after it) / after."""
    g = next((f for f, k in ev if k == "g"), None)
    r = next((f for f, k in ev if k == "r" and g is not None and f > g), None)
    out = []
    for f in frames:
        if g is None or f < g - WIN:
            out.append("approach")
        elif f < g + WIN:
            out.append("grasp")
        elif r is None or f < r - WIN:
            out.append("carry")
        elif f < r + WIN:
            out.append("release")
        else:
            out.append("after")
    return np.array(out)


def sq(rows, demo):
    """rows [n, K, 70], demo [n, 70] -> squared error per frame and component [n, 3], summed over the horizon, mean over draws."""
    d = (rows - demo[:, None]).reshape(rows.shape[0], rows.shape[1], 10, 7) ** 2
    return np.stack([d[..., c].sum((-1, -2)).mean(1) for c in COMP.values()], -1)


def vsq(V, U):
    """V, U [n, len(TS), 70] -> velocity squared error per frame, flow time and component [n, len(TS), 3]."""
    d = ((V - U).reshape(*V.shape[:2], 10, 7)) ** 2
    return np.stack([d[..., c].sum((-1, -2)) for c in COMP.values()], -1)


def task_errors(t):
    """-> per source: dict(sq [S, n, 3] per checkpoint, vsq [S, n, len(TS), 3], loss [S], steps, phase [n]) for n1, plus n2 / n5."""
    out = {}
    for nd in (1, 2, 5):
        for src in SOURCES + ("lorasc",):
            if src == "lorasc" and nd != 1:
                continue
            z = load(t, src, nd)
            if z is None:
                continue
            ev = [(int(f), str(k)) for f, k in z["events"]]
            ph = phase_of(z["frames"], ev, int(z["T"]))
            out[(src, nd)] = dict(steps=z["steps"], loss=z["loss"], phase=ph,
                                  sq=np.stack([sq(r, z["demo_rows"]) for r in z["rows"]]),
                                  vsq=np.stack([vsq(v, z["u"]) for v in z["v"]]))
    return out


def shares(E_pol, E_proto, E_demo):
    """Shares of the policy's error to the held-out demonstration: removed by the prototype, removed only by the demonstration
    fit, left.  They sum to 1."""
    if E_pol <= 0:
        return None
    return dict(proto=(E_pol - E_proto) / E_pol, demo=(E_proto - E_demo) / E_pol, left=E_demo / E_pol,
                e_policy=E_pol, e_proto=E_proto, e_demo=E_demo)


def boot_ci(per_task, fn, B=2000, seed=0):
    """Task-level bootstrap 95% interval of fn(list of per-task items)."""
    rng = np.random.default_rng(seed)
    ks = list(per_task)
    vals = []
    for _ in range(B):
        s = [per_task[ks[i]] for i in rng.integers(0, len(ks), len(ks))]
        v = fn(s)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def stage1(E, fit_step=800):
    """Error shares per scheme x component x phase (pooled sums over tasks and frames), by grade, by flow time."""
    gr = grades()
    res = {}
    for scheme in ("ptr", "lib", "prog", "blend"):
        tasks = [t for t in TASKS if (scheme, 1) in E[t] and ("policy", 1) in E[t]]
        if not tasks:
            continue
        r = dict(tasks=len(tasks))

        def parts(t, sel_phase=None, comp=None, by="sq", tj=None):
            P0 = E[t][("policy", 1)]
            S = E[t][(scheme, 1)]
            j = list(S["steps"]).index(fit_step)
            m = np.ones(len(P0["phase"]), bool) if sel_phase is None else (P0["phase"] == sel_phase)
            c = slice(None) if comp is None else [list(COMP).index(comp)]
            if by == "sq":
                g = lambda X, s: float(X["sq"][s][m][:, c].sum())                                   # noqa: E731
            else:
                g = lambda X, s: float(X["vsq"][s][m][:, tj][:, c].sum())                          # noqa: E731
            return (g(P0, 0), g(S, 0), g(S, j))

        def pooled(items):
            a = np.sum(items, 0)
            return shares(*a)

        for comp in (None,) + tuple(COMP):
            ck = comp or "all"
            r[ck] = {}
            for ph in (None,) + PHASES:
                pk = ph or "all"
                items = {t: parts(t, ph, comp) for t in tasks}
                items = {t: v for t, v in items.items() if v[0] > 0}
                s = pooled(list(items.values()))
                if s is None:
                    continue
                s["n_tasks"] = len(items)
                if ph is None:
                    s["proto_ci"] = boot_ci(items, lambda L: (pooled(L) or {}).get("proto"))
                    s["demo_ci"] = boot_ci(items, lambda L: (pooled(L) or {}).get("demo"))
                    s["tasks_proto_positive"] = int(sum(v[0] > v[1] for v in items.values()))
                    s["tasks_demo_positive"] = int(sum(v[1] > v[2] for v in items.values()))
                r[ck][pk] = s
        r["by_grade"] = {}
        for g in sorted(set(gr.values())):
            ts = [t for t in tasks if gr[t] == g]
            r["by_grade"][g] = {(comp or "all"): pooled([parts(t, None, comp) for t in ts]) for comp in (None,) + tuple(COMP)}
            r["by_grade"][g]["n_tasks"] = len(ts)
        r["by_flow_time"] = {}
        for j, tv in enumerate(TS):
            r["by_flow_time"][str(tv)] = {(comp or "all"): pooled([parts(t, None, comp, "vsq", j) for t in tasks])
                                          for comp in (None,) + tuple(COMP)}
        r["heldout_loss"] = dict(policy=float(np.mean([E[t][("policy", 1)]["loss"][0] for t in tasks])),
                                 proto=float(np.mean([E[t][(scheme, 1)]["loss"][0] for t in tasks])),
                                 demo=float(np.mean([E[t][(scheme, 1)]["loss"][list(E[t][(scheme, 1)]["steps"]).index(fit_step)]
                                                     for t in tasks])))
        r["per_task"] = {str(t): {c: shares(*parts(t, None, None if c == "all" else c)) for c in ("all",) + tuple(COMP)}
                         for t in tasks}
        res[scheme] = r
    # the leaf from the policy alone (the plain LoRA, same recipe) and the served 1,000-step leaf: what one demonstration does alone
    res["leaf_from_policy"] = {}
    for nm, key in (("refit_800", ("policy", 1)), ("served_lorasc", ("lorasc", 1))):
        tasks = [t for t in TASKS if key in E[t] and ("policy", 1) in E[t]]
        d = {}
        for comp in (None,) + tuple(COMP):
            c = slice(None) if comp is None else [list(COMP).index(comp)]
            a = sum(float(E[t][("policy", 1)]["sq"][0][:, c].sum()) for t in tasks)
            j = -1
            b = sum(float(E[t][key]["sq"][j][:, c].sum()) for t in tasks)
            d[comp or "all"] = dict(removed=(a - b) / a if a else None, e_policy=a, e_leaf=b)
        d["n_tasks"] = len(tasks)
        res["leaf_from_policy"][nm] = d
    # unsupervised: segments (BlendHooks' approach / carry / after) whose taught share exceeds the donated one in every component
    res["blend_candidates"] = candidates(E, fit_step)
    return res


def candidates(E, fit_step):
    out, n = [], 0
    for t in TASKS:
        if ("blend", 1) not in E[t]:
            continue
        P0, S = E[t][("policy", 1)], E[t][("blend", 1)]
        j = list(S["steps"]).index(fit_step)
        ph = P0["phase"]
        seg = np.where(np.isin(ph, ("approach", "grasp")), "approach", np.where(np.isin(ph, ("carry", "release")), "carry", "after"))
        for sname in ("approach", "carry", "after"):
            m = seg == sname
            if m.sum() < 3:
                continue
            n += 1
            sh = [shares(float(P0["sq"][0][m][:, c].sum()), float(S["sq"][0][m][:, c].sum()), float(S["sq"][j][m][:, c].sum()))
                  for c in range(3)]
            if all(s is not None and s["demo"] > max(s["proto"], 0.0) for s in sh):
                out.append(dict(task=t, segment=sname, demo=[s["demo"] for s in sh], proto=[s["proto"] for s in sh]))
    return dict(segments=n, candidates=len(out), list=out)


def stage2(E):
    """Learning curves (held-out flow loss and per-component chunk error, n = 1 / 2 / 5 demos x fit steps) per source, and the
    value of a prototype in demonstration units and in fit steps."""
    cur = curve_tasks()
    gr = grades()
    res = dict(tasks=cur, grades={str(t): gr[t] for t in cur}, curves={}, value={})
    have = [t for t in cur if all((s, nd) in E[t] for s in CURVE_SOURCES for nd in (1, 2, 5))]
    res["complete_tasks"] = have
    for src in CURVE_SOURCES:
        res["curves"][src] = {}
        for nd in (1, 2, 5):
            d = {}
            for i, st in enumerate(STEPS):
                ent = dict(loss=float(np.mean([E[t][(src, nd)]["loss"][i] for t in have])))
                for c, ci in zip(COMP, range(3)):
                    ent[c] = float(np.mean([E[t][(src, nd)]["sq"][i][:, ci].mean() for t in have]))
                ent["all"] = float(np.mean([E[t][(src, nd)]["sq"][i].sum(-1).mean() for t in have]))
                d[str(st)] = ent
            res["curves"][src][str(nd)] = d

    def metric(t, src, nd, i, m):
        X = E[t][(src, nd)]
        if m == "loss":
            return float(X["loss"][i])
        if m == "all":
            return float(X["sq"][i].sum(-1).mean())
        return float(X["sq"][i][:, list(COMP).index(m)].mean())

    def demos_needed(t, scheme, i, m):
        """The number of demonstrations the policy start needs at fit step STEPS[i] to match the scheme's start with ONE
        demonstration (linear in log n between 1, 2, 5; '>5' censored at 5 x the slope rule: reported as None + flag)."""
        target = metric(t, scheme, 1, i, m)
        ys = [metric(t, "policy", nd, i, m) for nd in (1, 2, 5)]
        xs = np.log([1, 2, 5])
        if ys[0] <= target:
            return 1.0, False
        for k in range(1, 3):
            if ys[k] <= target:
                f = (ys[k - 1] - target) / max(ys[k - 1] - ys[k], 1e-12)
                return float(np.exp(xs[k - 1] + f * (xs[k] - xs[k - 1]))), False
        return 5.0, True

    def steps_needed(t, scheme, i, m):
        """Fit steps the policy start (1 demo) needs to match the scheme's start (1 demo) at STEPS[i] (log-linear between
        checkpoints; censored at 800)."""
        target = metric(t, scheme, 1, i, m)
        ys = [metric(t, "policy", 1, j, m) for j in range(len(STEPS))]
        if ys[0] <= target:
            return 0.0, False
        xs = np.log1p(STEPS)
        for k in range(1, len(STEPS)):
            if ys[k] <= target:
                f = (ys[k - 1] - target) / max(ys[k - 1] - ys[k], 1e-12)
                return float(np.expm1(xs[k - 1] + f * (xs[k] - xs[k - 1]))), False
        return float(STEPS[-1]), True

    def envelope(t, src, nd, m):
        """The curve's best checkpoint (oracle early stopping on the held-out demonstration, the same rule for every start)."""
        return min(metric(t, src, nd, i, m) for i in range(len(STEPS)))

    def demos_needed_env(t, scheme, m):
        target = envelope(t, scheme, 1, m)
        ys = [envelope(t, "policy", nd, m) for nd in (1, 2, 5)]
        xs = np.log([1, 2, 5])
        if ys[0] <= target:
            return 1.0, False
        for k in range(1, 3):
            if ys[k] <= target:
                f = (ys[k - 1] - target) / max(ys[k - 1] - ys[k], 1e-12)
                return float(np.exp(xs[k - 1] + f * (xs[k] - xs[k - 1]))), False
        return 5.0, True

    res["envelope"] = {src: {str(nd): {m: float(np.mean([envelope(t, src, nd, m) for t in have])) for m in ("loss", "all") + tuple(COMP)}
                             for nd in (1, 2, 5)} for src in CURVE_SOURCES} if have else {}
    res["value_envelope"] = {}
    for scheme in ("ptr", "lib", "blend"):
        res["value_envelope"][scheme] = {}
        for m in ("loss", "all") + tuple(COMP):
            dn = {t: demos_needed_env(t, scheme, m) for t in have}
            if not dn:
                continue
            res["value_envelope"][scheme][m] = dict(
                demos_median=float(np.median([x for x, _ in dn.values()])), demos_censored=int(sum(c for _, c in dn.values())),
                scheme_better_at_1demo=int(sum(envelope(t, scheme, 1, m) < envelope(t, "policy", 1, m) for t in have)),
                per_task={str(t): x for t, (x, _) in dn.items()},
                by_grade={g: float(np.median([dn[t][0] for t in have if gr[t] == g])) for g in sorted({gr[t] for t in have})})

    for scheme in ("ptr", "lib", "blend"):
        v = {}
        for m in ("loss", "all") + tuple(COMP):
            v[m] = {}
            for i, st in enumerate(STEPS):
                dn = {t: demos_needed(t, scheme, i, m) for t in have}
                sn = {t: steps_needed(t, scheme, i, m) for t in have} if st in (0, 50, 100, 200) else None
                ent = dict(demos_median=float(np.median([x for x, _ in dn.values()])),
                           demos_censored=int(sum(c for _, c in dn.values())),
                           scheme_better_at_1demo=int(sum(metric(t, scheme, 1, i, m) < metric(t, "policy", 1, i, m) for t in have)),
                           per_task={str(t): x for t, (x, _) in dn.items()})
                if sn:
                    ent.update(steps_median=float(np.median([x for x, _ in sn.values()])),
                               steps_censored=int(sum(c for _, c in sn.values())))
                v[m][str(st)] = ent
            byg = {}
            for g in sorted({gr[t] for t in have}):
                ts = [t for t in have if gr[t] == g]
                byg[g] = {str(st): float(np.median([demos_needed(t, scheme, i, m)[0] for t in ts])) for i, st in enumerate(STEPS)}
            v[m]["by_grade_demos_median"] = byg
        res["value"][scheme] = v
    return res


def figures(s1, s2, E):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    C_PROTO, C_DEMO = "#2a78d6", "#eb6834"
    schemes = [s for s in ("ptr", "lib", "prog", "blend") if s in s1]
    names = dict(ptr="pointers (sup.)", lib="library (sup.)", prog="program (unsup.)", blend="soft blend (unsup.)")
    # 1: shares per component x phase
    fig, axes = plt.subplots(len(schemes), 4, figsize=(13, 2.4 * len(schemes)), sharey=True, squeeze=False)
    for i, sc in enumerate(schemes):
        for j, comp in enumerate(("all",) + tuple(COMP)):
            ax = axes[i, j]
            phs = [p for p in ("all",) + PHASES if p in s1[sc][comp]]
            x = np.arange(len(phs))
            pr = np.array([s1[sc][comp][p]["proto"] for p in phs])
            de = np.array([s1[sc][comp][p]["demo"] for p in phs])
            ax.bar(x, pr, color=C_PROTO, edgecolor="white", linewidth=1, label="removed by the prototype")
            ax.bar(x, de, bottom=np.where(de >= 0, np.maximum(pr, 0), np.minimum(pr, 0)), color=C_DEMO, edgecolor="white", linewidth=1, label="removed only by the demo")
            ax.axhline(0, color="#444", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(phs, rotation=40, fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            if i == 0:
                ax.set_title(comp, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{names[sc]}\nshare of policy error", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("pi0.5, Z3's 25 tasks, held-out demonstration 1: share of the policy's chunk error removed", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "zd_shares.png"), dpi=140)
    plt.close(fig)
    # 2: by flow time
    fig, axes = plt.subplots(1, len(schemes), figsize=(3.3 * len(schemes), 2.8), sharey=True, squeeze=False)
    for i, sc in enumerate(schemes):
        ax = axes[0, i]
        for comp, col in zip(COMP, ("#2a78d6", "#1baf7a", "#eb6834")):
            ax.plot(TS, [s1[sc]["by_flow_time"][str(t)][comp]["proto"] for t in TS], "-o", color=col, label=f"{comp}: proto")
            ax.plot(TS, [s1[sc]["by_flow_time"][str(t)][comp]["demo"] for t in TS], "--s", color=col, label=f"{comp}: demo only")
        ax.axhline(0, color="#444", lw=0.6)
        ax.invert_xaxis()
        ax.set_xlabel("flow time t (1 = noise)")
        ax.set_title(names[sc], fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("share of policy velocity error")
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "zd_flowtime.png"), dpi=140)
    plt.close(fig)
    # 3: learning curves
    if s2 and s2.get("complete_tasks"):
        cols = dict(policy="#6b6f76", ptr="#1baf7a", lib="#2a78d6", blend="#eb6834")
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.0), squeeze=False)
        for j, m in enumerate(("loss", "translation", "rotation", "gripper")):
            ax = axes[0, j]
            for src in CURVE_SOURCES:
                for nd, ls in ((1, "-"), (2, "--"), (5, ":")):
                    c = s2["curves"][src][str(nd)]
                    ax.plot(np.array(STEPS) + 10, [c[str(s)][m] for s in STEPS], ls, color=cols[src], marker="o", ms=2.5,
                            label=f"{src}, {nd} demo" if m == "loss" else None)
            ax.set_xscale("log")
            ax.set_xlabel("fit steps + 10")
            ax.set_title("held-out flow loss" if m == "loss" else f"{m} chunk error", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=6, frameon=False, ncol=2)
        fig.suptitle(f"leaf fitted from each start, {len(s2['complete_tasks'])} tasks, held-out demonstration 1", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "zd_curves.png"), dpi=140)
        plt.close(fig)


def read(record):
    E = {t: task_errors(t) for t in TASKS}
    n1 = {s: sum((s, 1) in E[t] for t in TASKS) for s in SOURCES + ("lorasc",)}
    print("[zd] n1 files per source:", n1, flush=True)
    s1 = stage1(E)
    s1e = stage1(E, fit_step=100)                # sensitivity: the leaf at 100 steps (held-out flow loss is lowest near there)
    s2 = stage2(E)
    for sc in ("ptr", "lib", "prog", "blend"):
        if sc not in s1:
            continue
        print(f"\n[zd] {sc} ({s1[sc]['tasks']} tasks): share of the policy's chunk error, removed by proto / only by demo / left")
        for comp in ("all",) + tuple(COMP):
            line = "  ".join(f"{p} {s1[sc][comp][p]['proto']:+.2f}/{s1[sc][comp][p]['demo']:+.2f}"
                             for p in ("all",) + PHASES if p in s1[sc][comp])
            print(f"   {comp:12} {line}")
        print("   CI all:", s1[sc]["all"]["all"]["proto_ci"], s1[sc]["all"]["all"]["demo_ci"])
        print("   flow time:", {t: {c: round(s1[sc]["by_flow_time"][t][c]["proto"], 2) for c in COMP} for t in s1[sc]["by_flow_time"]})
        print("   grade:", {g: round(v["all"]["proto"], 2) if v["all"] else None for g, v in s1[sc]["by_grade"].items()})
        print("   held-out loss:", s1[sc]["heldout_loss"])
    print("\n[zd] sensitivity, a_demo at 100 steps: share removed only by the demo (all / translation / rotation / gripper)")
    for sc in ("ptr", "lib", "prog", "blend"):
        if sc in s1e:
            print(f"   {sc:6} " + " ".join(f"{c} {s1e[sc][c]['all']['demo']:+.2f}" for c in ("all",) + tuple(COMP)))
    print("\n[zd] leaf from the policy:", json.dumps({k: {c: round(v[c]["removed"], 3) for c in ("all",) + tuple(COMP)}
                                                    for k, v in s1["leaf_from_policy"].items()}))
    print("[zd] blend candidates:", s1["blend_candidates"]["candidates"], "of", s1["blend_candidates"]["segments"])
    if s2["complete_tasks"]:
        print(f"\n[zd] curves on {len(s2['complete_tasks'])} tasks")
        for src in CURVE_SOURCES:
            for nd in (1, 2, 5):
                c = s2["curves"][src][str(nd)]
                print(f"   {src:7} n{nd}: loss " + " ".join(f"{c[str(s)]['loss']:.4f}" for s in STEPS) +
                      " | all err " + " ".join(f"{c[str(s)]['all']:.2f}" for s in STEPS))
        for sc in ("ptr", "lib", "blend"):
            for m in ("loss", "all") + tuple(COMP):
                v = s2["value"][sc][m]
                print(f"   value {sc:5} {m:11}: demos median " + " ".join(f"{s}:{v[str(s)]['demos_median']:.2f}({v[str(s)]['demos_censored']})"
                                                                      for s in STEPS) +
                      " | steps median " + " ".join(f"{s}:{v[str(s)]['steps_median']:.0f}" for s in (0, 50, 100, 200)))
    if s2.get("value_envelope"):
        print("\n[zd] best checkpoint per curve (oracle early stop): mean over tasks, loss / all chunk error")
        for src in CURVE_SOURCES:
            print(f"   {src:7} " + "  ".join(f"n{nd} {s2['envelope'][src][str(nd)]['loss']:.4f}/{s2['envelope'][src][str(nd)]['all']:.2f}"
                                       for nd in (1, 2, 5)))
        for sc, v in s2["value_envelope"].items():
            print(f"   value (envelope) {sc:5}: " + "  ".join(f"{m} {v[m]['demos_median']:.2f}({v[m]['demos_censored']} cens, "
                                                         f"{v[m]['scheme_better_at_1demo']}/{len(s2['complete_tasks'])} better)" for m in v))
    res = dict(stage1=s1, stage1_step100=s1e, stage2=s2, files=n1, design=dict(tasks=TASKS, steps=STEPS, train_demos=TRAIN_DEMOS, held=HELD, stride=STRIDE,
                                                         K=K, flow_times=TS, rank=RANK, lr=LR, batch=BATCH, win=WIN, a_demo_step=800))
    json.dump(res, open(os.path.join(R, "zd_read.json"), "w", encoding="utf-8"), indent=1, default=float)
    figures(s1, s2, E)
    if record:
        from lifelong_ood import ledger
        z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
        common = dict(bench="libero_90", seed=0, bytes=z, config=res["design"], script="experiments/zd_breakdown.py",
                      command=" ".join(sys.argv))
        if all(n1[s] == len(TASKS) for s in SOURCES):
            print(ledger.record("zd_breakdown", tag="stage1", metrics=s1, **common))
            print(ledger.record("zd_breakdown", tag="stage1_step100", metrics=s1e, **dict(common, config=dict(res["design"], a_demo_step=100))))
        if len(s2["complete_tasks"]) == len(s2["tasks"]):
            print(ledger.record("zd_breakdown", tag="stage2", metrics=s2, **common))


if __name__ == "__main__":
    a = sys.argv
    if "--run" in a:
        run(int(a[a.index("--task") + 1]), a[a.index("--group") + 1] if "--group" in a else "n1", smoke="--smoke" in a)
    if "--launch" in a:
        launch(a[a.index("--groups") + 1].split("+") if "--groups" in a else ["n1"])
    if "--read" in a:
        read("--record" in a)
