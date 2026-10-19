"""Plan C (docs/history/PLAN_PRIMITIVES.md): fit one motion PRIMITIVE per step of a task from its one demonstration.

The instruction's steps (primitives.steps_of) cut the demo at its first sustained release (primitives.segment); each segment's
frames, prompted with the step's clause, fit a keyed LoRA on the action expert by flow matching (adapt.train, the one-demo LoRA's
settings).  Saves results/robot/prim_<task>_<step><tag>.pt (meta: clause, step key, frames); ledger row c0_fit.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/c0_fit.py --task 3" slurm/jobs/robot_gpu.sbatch
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.experts.lora import add_node                         # noqa: E402
from lifelong_ood.policy import adapt                                  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy import primitives as P                        # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--scope", default="expert")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--joint", default="", help="t1+t2+...: ONE primitive for --key fitted on these tasks' demo segments (each "
                                                 "with its own clause), saved as prim_<name>_0.pt")
    ap.add_argument("--key", default="", help="the step key of --joint (with --goal: the goal key, prep|target)")
    ap.add_argument("--goal", action="store_true", help="plan D: match steps by GOAL key and fit only their transport frames "
                                                        "(from the step's first sustained grasp to its end)")
    ap.add_argument("--approach", action="store_true", help="plan E: match steps by OBJECT key (verb|object, qualifier left out) "
                                                            "and fit only their approach frames (before the step's first grasp)")
    ap.add_argument("--init", default="", help="plan E2: start --joint from this primitive's weights (a child concept starts from "
                                                "its parent's prototype and specialises on its own demonstrations)")
    ap.add_argument("--name", default="", help="the file name of --joint")
    ap.add_argument("--source", default="l90", help="--joint: l90 (LIBERO-90 tasks) or d40 (pi0.5's 40 training tasks; plan G4)")
    ap.add_argument("--score", action="store_true", help="--joint: only score the frozen policy's flow loss on the frames (plan I6)")
    ap.add_argument("--ndemo", type=int, default=1, help="--joint: demonstrations per task (from --demo on)")
    ap.add_argument("--backbone", default="pi05_libero", help="plan G3: the policy the primitive is fitted on (pi05_libero, "
                                                                 "pi0_libero, smolvla); its own action normalisation and chunk length")
    ap.add_argument("--bind", action="store_true", help="--joint: fit under attention binding (each task's own demo prototype, "
                                                        "top-4 patches, mass 0.7, before the demo's first grasp), as at test")
    # plan F1 (docs/FIT_TIME.md): fit-time levers; every default leaves the fit exactly as before
    ap.add_argument("--cache", action="store_true", help="--joint: cache each frame's frozen VLM prefix ONCE (adapt.PrefixBank) "
                                                         "instead of recomputing it every step (LazyPrefixBank); the same values")
    ap.add_argument("--cache-store", default="auto", help="--cache: where the cached prefixes live: cuda, cpu, or auto (cuda up to "
                                                          "--cache-gpu-max frames, else cpu)")
    ap.add_argument("--cache-gpu-max", type=int, default=1200)
    ap.add_argument("--init-best", default="", help="--joint: candidate prototypes n1+n2+... (prim_<n>_0.pt); start from the one "
                                                     "with the lowest flow loss on THIS concept's training frames, or from scratch "
                                                     "when none beats the frozen policy (sets --init)")
    ap.add_argument("--heldout", type=int, default=-1, help="--joint: k >= 0 scores the held-out flow loss on demonstration k of "
                                                             "every task of the concept (same phase cut) every --eval-every steps")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-frames", type=int, default=256, help="--heldout: at most this many evenly spaced held-out frames")
    ap.add_argument("--record-as", default="c0_fit", help="the ledger experiment of the fit's row (plan F1 writes f1_fit)")
    a = ap.parse_args()
    if a.joint:
        a.key = a.key.replace("_", " ")                                   # the job wrapper splits arguments on spaces
        return joint(a)
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    text = texts[a.task]
    steps = P.steps_of(text)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    ep_idx = d90.episodes_of_task(a.task)[a.demo]
    ep = d90.load(ep_idx)
    segs = P.segment(ep["actions"], len(steps))
    if segs is None:
        print(f"[c0] task {a.task}: the demo has no sustained release to cut {len(steps)} steps at; fitting one primitive", flush=True)
        steps, segs = [text], [(0, len(ep["actions"]))]
    norm = D.Norm.load()
    state, banks = adapt.inject(bk, scope=a.scope)                       # once: every step's primitive is its own key
    out, total = [], 0
    for s, (clause, (lo, hi)) in enumerate(zip(steps, segs)):
        t0 = time.time()
        obs = []
        for i in range(lo, hi):
            o = d90.observation(ep, i)
            o["prompt"] = clause
            obs.append(o)
        bank = adapt.PrefixBank(bk, obs)
        idx, _ = D.chunk_indices(len(ep["actions"]), bk.H)                # a chunk near the step's end runs into the demo's real
        full = D.pad_rows(D.to_rows(ep["actions"][idx[lo:hi]], norm), bk.H)  # next actions (its release); padding at the step's end
        #                                                                   taught the first step never to open the gripper
        key = f"prim_{a.task}_{s}{a.tag}"
        params = add_node(banks, key, a.rank)
        hist = adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=a.lr, batch=a.batch,
                           log=lambda x: print(x, flush=True))
        path = os.path.join(R, f"prim_{a.task}_{s}{a.tag}.pt")
        meta = dict(task=a.task, step=s, clause=clause, step_key=P.step_key(clause), frames=[int(lo), int(hi)], rank=a.rank,
                    scope=a.scope, demo_episode=int(ep_idx))
        nb = adapt.save(banks, key, path, meta=meta)
        total += nb
        out.append(dict(meta, bytes=nb, first_loss=hist["loss"][0], final_loss=hist["loss"][-1], secs=time.time() - t0, path=path))
        print(f"[c0] task {a.task} step {s} '{clause}' key {meta['step_key']}: frames {lo}-{hi}, {nb / 1e6:.1f} MB, flow "
              f"{hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}", flush=True)
        del bank
    print(ledger.record("c0_fit", bench="libero_90", tag=f"task{a.task}{a.tag}", seed=0, keys=dict(task=a.task),
                        metrics=dict(text=text, steps=out), bytes=dict(experts=total, tree=0, codes=0, samples=0, teacher_copies=0,
                                                                         total=total),
                        config=vars(a), script="experiments/c0_fit.py", command=" ".join(sys.argv)))


def backbone(name):
    """-> (the policy, its action normalisation).  pi0.5 keeps the normalisation every earlier fit used."""
    if name.startswith("smolvla"):
        from lifelong_ood.policy.backbone_smolvla import SmolVLABackbone
        bk = SmolVLABackbone(device="cuda")
        return bk, D.Norm.for_backbone(bk)
    if name.startswith("xvla"):                                         # LeRobot's X-VLA (absolute ee6d actions: backbone_xvla.py)
        from lifelong_ood.policy.backbone_xvla import XVLABackbone
        bk = XVLABackbone(device="cuda")
        return bk, D.Norm.for_backbone(bk)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(name, device="cuda")
    return bk, (D.Norm.load() if name == "pi05_libero" else D.Norm.for_backbone(bk))


def chunk_rows(ep, chunk_idx, frames, norm):
    """Normalised flat rows of the chunks at `frames`; a state-relative checkpoint (pi0) takes each chunk relative to its frame's
    raw state; an absolute-target checkpoint (X-VLA) takes every chunk step from its own frame's state."""
    if getattr(norm, "per_step_state", False):
        st = np.asarray(ep["state"], np.float32)
        return np.stack([norm.normalize(ep["actions"][ci], state=st[ci]).reshape(-1) for ci in chunk_idx])
    if not norm.delta:
        return D.to_rows(ep["actions"][chunk_idx], norm)
    st = np.asarray(ep["state"], np.float32)
    return np.stack([norm.normalize(ep["actions"][ci], state=st[f]).reshape(-1) for ci, f in zip(chunk_idx, frames)])


def score(a, bk, bank, full, used, n_draws=8, batch=64):
    """Plan I6: mean flow-matching loss of the frozen policy (no adapter) on the concept's frames, over n_draws fixed draws of
    (frames, time, noise); recorded in ledger i6_novelty keyed by the concept's name."""
    import torch
    from lifelong_ood import ledger
    gen = torch.Generator(device="cpu").manual_seed(0)
    rng = np.random.default_rng(0)
    A = torch.as_tensor(np.asarray(full, np.float32), device=bk.device)
    losses = []
    with torch.no_grad():
        for _ in range(n_draws):
            idx = rng.integers(0, bank.n, size=min(batch, bank.n))
            t, eps = adapt._flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
            x = A[idx]
            xt = t[:, None, None] * eps + (1 - t[:, None, None]) * x
            u = (eps - x).reshape(len(idx), -1)
            v = bk.v_full(xt, t, bank.take(idx))
            losses.append(float(adapt.flow_loss(bk, v, u, xt, t)))
    m = dict(name=a.name, key=a.key, source=a.source, goal=a.goal, approach=a.approach, frames=int(bank.n), tasks=used,
             loss=float(np.mean(losses)), loss_sd=float(np.std(losses)), draws=losses, backbone=a.backbone)
    print(f"[i6] {a.name} ({a.source}, {a.key}): frozen-policy flow loss {m['loss']:.4f} +- {m['loss_sd']:.4f} on {bank.n} frames",
          flush=True)
    print(ledger.record("i6_novelty", bench="libero", tag="score", seed=0, keys=dict(name=a.name), metrics=m,
                        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                        script="experiments/c0_fit.py", command=" ".join(sys.argv)))


def joint(a):
    """Plan C, consolidation: a step's primitive fitted on SEVERAL arrivals' demos that share its key (different objects and
    rooms), so it can learn the step rather than one object's grasp (docs/COBWEB_UPDATE.md, Proposition 7)."""
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk, norm = backbone(a.backbone)
    if a.source == "d40":                                                   # plan G4: pi0.5's own 40 training tasks (task_index of
        d90 = D.Demos()                                                     # its LeRobot set), a.ndemo demonstrations each
        texts = d90.tasks
        demos_of = lambda t: d90.episodes(t)[a.demo:a.demo + a.ndemo]       # noqa: E731
    else:
        texts, _ = r3p.libero90_texts()
        d90 = D.Demos90()
        demos_of = lambda t: d90.episodes_of_task(t)[a.demo:a.demo + a.ndemo]   # noqa: E731
    obs, rows, used, frame_info = [], [], [], []
    pairs = []
    for t in [int(x) for x in a.joint.split("+")]:
        eps_ = demos_of(t)
        if not eps_:
            print(f"[c0j] task {t}: the dataset has no demonstration under its instruction; skipped", flush=True)
        pairs += [(t, e) for e in eps_]
    for t, e_ in pairs:
        steps = P.steps_of(texts[t])
        match = P.goal_key if a.goal else P.object_key if a.approach else P.step_key
        s = next((i for i, c in enumerate(steps) if match(c) == a.key), None)
        ep = d90.load(e_)
        segs = P.segment(ep["actions"], len(steps))
        if s is None or segs is None:
            print(f"[c0j] task {t}: no step with key {a.key} (or no release to cut at); skipped", flush=True)
            continue
        lo, hi = segs[s]
        if a.goal:                                                          # plan D: the carry and the place only
            tr = P.transport(ep["actions"], lo, hi)
            if tr is None:
                print(f"[c0j] task {t}: its step never grasps; skipped", flush=True)
                continue
            lo, hi = tr
        if a.approach:                                                      # plan E: reaching and grasping the object only
            ap_ = P.approach(ep["actions"], lo, hi)
            if ap_ is None:
                print(f"[c0j] task {t}: its step never grasps; skipped", flush=True)
                continue
            lo, hi = ap_
        idx, _ = D.chunk_indices(len(ep["actions"]), bk.H)
        rows.append(D.pad_rows(chunk_rows(ep, idx[lo:hi], range(lo, hi), norm), bk.H, bk.a, bk.A))
        g = np.asarray(ep["actions"])[:, 6] > 0
        first = int(np.argmax(g)) if g.any() else len(g)
        for i in range(lo, hi):
            o = d90.observation(ep, i)
            o["prompt"] = steps[s]
            obs.append(o)
            frame_info.append((t, i < first))                               # (task, before the demo's first grasp)
        used.append(dict(task=t, clause=steps[s], frames=[int(lo), int(hi)]))
        print(f"[c0j] task {t} step {s} '{steps[s]}': frames {lo}-{hi}", flush=True)
    full = np.concatenate(rows, 0)
    t_frames = time.time()
    bank = cached_bank(bk, obs, a) if a.cache else adapt.LazyPrefixBank(bk, obs)   # ~18 MB per cached prefix: computed per batch
    t_cache = time.time()
    if a.score:                                                             # plan I6: the FROZEN policy's flow loss on the concept's
        return score(a, bk, bank, full, used)                               # phase frames (Wang's missing detector of new primitives)
    gate_of = None
    if a.bind:                                                              # binding-aware primitive: the same attention as at test
        import torch
        from lifelong_ood.policy import attn_bias as AB
        protos = np.load(os.path.join(R, "q4_object_protos.npz"))
        spec_q4 = importlib.util.spec_from_file_location("q4", os.path.join(ROOT, "experiments", "q4_template.py"))
        q4 = importlib.util.module_from_spec(spec_q4)
        spec_q4.loader.exec_module(q4)
        maps = []
        for o, (t, approach) in zip(obs, frame_info):
            if not approach:
                maps.append({})
                continue
            if str(t) not in protos.files:                                  # no silent fallback: a donor without a prototype
                raise SystemExit(f"[c0j] task {t} has no object prototype (q4_template.py); build it before --bind")
            f = q4.feats(bk, o["observation/image"], o["observation/wrist_image"], o["observation/state"])
            k = np.argsort(f @ protos[str(t)])[::-1][:4]
            m = np.zeros(256, np.float32)
            m[k] = 1.0
            maps.append({"agent": m.reshape(16, 16)})
        width = int(bank.take([0]).pad_masks.shape[1])
        gate_of = lambda idx: (lambda: AB.biased(bk, [maps[int(j)] for j in idx], mode="mass", rho=0.7, width=width,   # noqa: E731
                                                  batch=len(idx)))
        print(f"[c0j] binding maps on {sum(bool(m_) for m_ in maps)} of {len(maps)} frames", flush=True)
    state, banks = adapt.inject(bk, scope=a.scope)
    key = f"prim_{a.name}_0"
    params = add_node(banks, key, a.rank)
    choice = None
    if a.init_best and not a.init:                                          # plan F1: warm start from the best-fitting existing unit
        choice = init_best(a, bk, state, bank, full)
        a.init = choice["chosen_path"] or ""
    eval_fn, ev = None, {}
    if a.heldout >= 0:                                                      # plan F1: held-out flow loss along the fit
        ev = heldout_setup(a, bk, norm, state)
        eval_fn = lambda: adapt.heldout_loss(bk, state, {key}, ev["bank"], ev["full"], ev["draws"])   # noqa: E731
    if a.init:                                                              # the parent's prototype, copied into the child's slot
        import torch
        z = torch.load(a.init if os.path.isabs(a.init) else os.path.join(R, a.init), map_location="cpu")
        by_name = {b.name: b for b in banks}
        for n_, w in z["banks"].items():
            by_name[n_].A[key].weight.data.copy_(w["A"].to(by_name[n_].A[key].weight.dtype))
            by_name[n_].B[key].weight.data.copy_(w["B"].to(by_name[n_].B[key].weight.dtype))
        print(f"[c0j] initialised from {a.init}", flush=True)
    ev0 = eval_fn() if eval_fn is not None else None
    t0 = time.time()
    hist = adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=a.lr, batch=a.batch, log=lambda x: print(x, flush=True),
                       gate_of=gate_of, eval_fn=eval_fn, eval_every=a.eval_every if eval_fn is not None else 0)
    t_train = time.time()
    if eval_fn is not None:
        hist["eval_step"].insert(0, 0)
        hist["eval"].insert(0, ev0)
    path = os.path.join(R, f"prim_{a.name}_0.pt")
    meta = dict(joint=used, step_key=a.key, rank=a.rank, scope=a.scope, frames=int(len(obs)), bind=bool(a.bind), goal=bool(a.goal),
                backbone=a.backbone,
                approach=bool(a.approach), init=a.init)
    nb = adapt.save(banks, key, path, meta=meta)
    print(f"[c0j] {a.name}: key {a.key}, {len(used)} demos, {len(obs)} frames, {nb / 1e6:.1f} MB, flow {hist['loss'][0]:.4f} -> "
          f"{hist['loss'][-1]:.4f} in {time.time() - t0:.0f}s", flush=True)
    f1 = {}
    if a.cache or a.init_best or a.heldout >= 0 or a.record_as != "c0_fit":   # plan F1: where the job's time went
        f1 = dict(f1=dict(secs=dict(start_to_frames=t_frames - T_START, cache=t_cache - t_frames, train=t_train - t0,
                                    save=time.time() - t_train, total=time.time() - T_START),
                          frozen_heldout=ev.get("frozen"), heldout_frames=ev.get("frames"), init_choice=choice, cache=bool(a.cache)))
    print(ledger.record(a.record_as, bench="libero_90", tag=f"joint_{a.name}", seed=0, metrics=dict(meta, bytes=nb, curve=hist, **f1),
                        bytes=dict(experts=nb, tree=0, codes=0, samples=0, teacher_copies=0, total=nb), config=vars(a),
                        script="experiments/c0_fit.py", command=" ".join(sys.argv)))


# ── plan F1 (docs/FIT_TIME.md): fit-time levers, used only under their flags ─────────────────────────────────────────────
T_START = time.time()


def cached_bank(bk, obs, a):
    """The frozen VLM prefix of every frame computed once (adapt.PrefixBank, blocks of 32): the values LazyPrefixBank recomputes
    at every step (the prefix does not depend on the adapter, which lives in the action expert only)."""
    store = a.cache_store if a.cache_store != "auto" else ("cpu" if len(obs) > a.cache_gpu_max else None)
    return adapt.PrefixBank(bk, obs, store=None if store == "cuda" else store)


def concept_frames(a, bk, norm, demo, ndemo=1, timer=None):
    """The concept's phase frames exactly as joint() cuts them (same key match, segment, transport / approach cut, chunk rows,
    prompts), from demonstrations demo .. demo + ndemo - 1 of every task of --joint.  -> (obs, full rows, used).
    timer: a dict that accumulates seconds of episode loading (video decode) and observation building."""
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    if a.source == "d40":
        d90 = D.Demos()
        texts = d90.tasks
        demos_of = lambda t: d90.episodes(t)[demo:demo + ndemo]            # noqa: E731
    else:
        texts, _ = r3p.libero90_texts()
        d90 = D.Demos90()
        demos_of = lambda t: d90.episodes_of_task(t)[demo:demo + ndemo]    # noqa: E731
    timer = timer if timer is not None else {}
    obs, rows, used = [], [], []
    pairs = [(t, e) for t in [int(x) for x in a.joint.split("+")] for e in demos_of(t)]
    for t, e_ in pairs:
        steps = P.steps_of(texts[t])
        match = P.goal_key if a.goal else P.object_key if a.approach else P.step_key
        s = next((i for i, c in enumerate(steps) if match(c) == a.key), None)
        t0 = time.time()
        ep = d90.load(e_)
        timer["load"] = timer.get("load", 0.0) + time.time() - t0
        segs = P.segment(ep["actions"], len(steps))
        if s is None or segs is None:
            continue
        lo, hi = segs[s]
        if a.goal:
            tr = P.transport(ep["actions"], lo, hi)
            if tr is None:
                continue
            lo, hi = tr
        if a.approach:
            ap_ = P.approach(ep["actions"], lo, hi)
            if ap_ is None:
                continue
            lo, hi = ap_
        idx, _ = D.chunk_indices(len(ep["actions"]), bk.H)
        rows.append(D.pad_rows(chunk_rows(ep, idx[lo:hi], range(lo, hi), norm), bk.H, bk.a, bk.A))
        t0 = time.time()
        for i in range(lo, hi):
            o = d90.observation(ep, i)
            o["prompt"] = steps[s]
            obs.append(o)
        timer["observation"] = timer.get("observation", 0.0) + time.time() - t0
        used.append(dict(task=t, episode=int(e_), clause=steps[s], frames=[int(lo), int(hi)]))
    return obs, (np.concatenate(rows, 0) if rows else None), used


def heldout_setup(a, bk, norm, state):
    """Held-out frames (demonstration --heldout of every task of the concept; at most --eval-frames, evenly spaced), their cached
    prefixes, fixed draws and the frozen policy's loss on them."""
    obs, full, used = concept_frames(a, bk, norm, a.heldout, 1)
    if not obs:
        raise SystemExit(f"[c0j] no held-out frames at demonstration {a.heldout}")
    keep = np.unique(np.linspace(0, len(obs) - 1, min(a.eval_frames, len(obs))).round().astype(int))
    obs, full = [obs[i] for i in keep], full[keep]
    bank = adapt.PrefixBank(bk, obs)
    draws = adapt.fixed_draws(bank.n, bk.H, bk.A)
    frozen = adapt.heldout_loss(bk, state, set(), bank, full, draws)
    print(f"[c0j] held-out: {len(obs)} frames of demonstration {a.heldout} ({len(used)} demos); frozen-policy flow loss {frozen:.4f}",
          flush=True)
    return dict(bank=bank, full=full, draws=draws, frozen=frozen, frames=len(obs), used=used)


def init_best(a, bk, state, bank, full):
    """Warm start by fit: each candidate prototype's flow loss on this concept's TRAINING frames (fixed draws); the lowest wins if
    it beats the frozen policy (else scratch).  Candidates are loaded under their own keys and dropped afterwards."""
    from lifelong_ood.experts.lora import drop_node
    draws = adapt.fixed_draws(bank.n, bk.H, bk.A, seed=4321)
    losses = {"(policy)": adapt.heldout_loss(bk, state, set(), bank, full, draws)}
    for n_ in [x for x in a.init_best.split("+") if x]:
        p_ = os.path.join(R, f"prim_{n_}_0.pt")
        if not os.path.exists(p_):
            print(f"[c0j] init candidate {n_}: no file; skipped", flush=True)
            continue
        _, k_, banks_ = adapt.load(bk, p_, state=state)
        losses[n_] = adapt.heldout_loss(bk, state, {k_}, bank, full, draws)
        drop_node(banks_, k_)
    best = min(losses, key=losses.get)
    print(f"[c0j] init by fit: {best} ({losses[best]:.4f}; policy {losses['(policy)']:.4f}; {len(losses) - 1} candidates)", flush=True)
    return dict(losses=losses, chosen=best, chosen_path="" if best == "(policy)" else f"prim_{best}_0.pt")


if __name__ == "__main__":
    main()
