"""PLAN M1 (docs/PLAN_M1.md; pre-registration experiments/prereg/pm1_compose_create.py, ledger m3_prereg tag pm1_compose_create):
COMPOSE, CREATE, OR FINE-TUNE on LIBERO-90-C's held-out DEV tasks, with our own checkpoint pi05_l90c main/5000 frozen.

Every arm gets the SAME one demo of the new task (its own-scene IPEC episode 0, l90c.task_episodes; never executed or replayed) and
is rolled from the same never-used start states (results/robot/fresh_m1, LIBERO's sampler at BASE 5700000, 20 per task), noise
seed 13, paired:
  zs           the task's instruction
  compose_cov  demo-free role-coverage pair retrieval (pl2c's rule on pi05_l90c's 73 training-cell instructions, qualifiers kept as
               content words) + TLI (lifelong_ood/policy/tli.py) at lambda 24, latents from pi05_l90c on its own training data
  compose_sel  the one demo's flow loss chooses among {zs} + {blind cover pair, scene-keyed cover pair} x lambda {24, 12, 36}
               (pl2's DemoScorer, K 4, seed 1); zs is kept unless a composition scores strictly lower (the L2 confirm guard)
  create       a KV token (policy/token_fold.py, n 4 entries per layer, 73.7 KB fp16) appended to the instruction's cache, fitted
               on every frame of the demo through the frozen policy (flow matching, Adam lr 1e-2, batch 32, 1000 steps)
  cc           compose-or-create: the same token fitted on top of compose_sel's conditioning (TLI shift per query), initialised
               from the composed prompt's own language entries; = create where compose_sel kept zs
  lora<k>      per-task LoRA r16 on the action expert (r3_fewshot's one-demo recipe: lr 1e-4, batch 32), ONE run of 1000 steps
               with snapshots at k = 50 / 200 / 1000 (the fit curve; budget-to-threshold)
Damage: 10 training cells, stored init states 0-19, seed 0, paired with the parity read of pi05_l90c@5000 (results/l90c/
parity_pi05_l90c_5000_*): the repairs as the store serves them (routed by instruction; differs from the base only on cells whose
text equals a dev task's) and unrouted leaks (the paired dev task's LoRA always on / its token appended to the old prompt).

    sbatch --wait ... --wrap "$PY experiments/m1_compose_create.py --plan"        design table -> results/m1/plan.json
    sbatch ... robot_gpu.sbatch SCRIPT="experiments/m1_compose_create.py --store --shard 0/4"   TLI latents of the pool (GPU)
    $PY experiments/m1_compose_create.py --fresh-check                            the start states are new
    sbatch ... robot_gpu.sbatch SCRIPT="experiments/m1_compose_create.py --fit --task libero_90:9"   one task's arms (GPU)
    $PY experiments/m1_compose_create.py --launch [--dry] | --launch-damage [--dry] | --read [--record]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "m1")
PY = "/coc/flash10/ksingara3/robot/openpi/.venv/bin/python"
CFG = "pi05_l90c"
CKPT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000"
L90C_DATA = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_l90c"
FRESH_DIR, BASE, N_STATES = "fresh_m1", 5700000, 20
USED_BASES = (100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1300000, 3300000, 3400000, 5100000,
              5300000)                                                  # every "seed base" in slurm/*.out on 2026-10-24
SEED = 13
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
LAMS = (24, 12, 36)                                                     # pl2's grid in pl2c's order; compose_cov uses 24
LAM_COV = 24
K_DRAWS, SCORE_SEED = 4, 1                                              # pl2's DemoScorer (K 4, seed 1, 36 frames)
REPLAN = 5
TOK_N, TOK_STEPS, TOK_LR, BATCH = 4, 1000, 1e-2, 32
LORA_RANK, LORA_LR, LORA_STEPS, LORA_SNAP = 16, 1e-4, 1000, (50, 200, 1000)
SMOKE = os.environ.get("M1_SMOKE") == "1"                              # pipeline smoke: results/m1/smoke, short fits, no ledger row
if SMOKE:
    TOK_STEPS, LORA_STEPS, LORA_SNAP = 20, 200, (50, 200)
FITS = os.path.join(OUT, "smoke" if SMOKE else "fits")
ARMS = ("zs", "compose_cov", "compose_sel", "create", "cc", "lora50", "lora200", "lora1000")
REL = r"\b(next to|on top of|on|in|into|onto|at|from|between|under|near|beside|inside|by)\b"   # pl2c's relation words
QUAL = {"top", "front", "back", "left", "right", "middle"}             # instance qualifiers: content words here (docs/PLAN_M1.md)
N_DAMAGE = 10
COLLIDE = {"libero_90:28": "libero_90:0", "libero_90:46": "libero_90:50"}   # training cell -> dev task with the SAME instruction


def slug(tid):
    return tid.replace(":", "_")


def dev_tasks():
    """The 29 DEV held-out tasks (bins 0.1 / 0.3 / 0.5 and the part set) in split.json order.  Sealed tasks are never listed."""
    from lifelong_ood import l90c
    s = l90c.load_split()
    out = []
    for b in ("0.1", "0.3", "0.5", "part"):
        out += [(tid, b) for tid in s["dev"][b]]
    for tid, _ in out:
        l90c.check_task(tid)
    return out


def task_table():
    from lifelong_ood.policy import factors as F
    return {f"{t.suite}:{t.index}": t for t in F.load(("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"))}


# ── the store: pi05_l90c's training-cell instructions ─────────────────────────────────────────────────────────────
def store_pool():
    """-> (tx {l90c task index: text}, scenes {text: sorted scenes of the training cells with that text}).  The 73 distinct
    instructions of lol/libero_l90c (the 76 training tasks; 3 texts are shared across scenes)."""
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    tx = D.Demos(root=L90C_DATA).tasks
    tt = task_table()
    scenes = {}
    for tid in l90c.load_split()["train"]:
        scenes.setdefault(tt[tid].language, set()).add(tt[tid].scene)
    assert set(tx.values()) == set(scenes), "the l90c dataset's texts are not exactly the training cells' instructions"
    return tx, {k: sorted(v) for k, v in scenes.items()}


def lat_path(t):
    return os.path.join(OUT, "tli", f"pi05_l90c_5000_{t}.npy")


# ── the rule: role-filler parse + greedy same-role coverage (pl2c, generalised to every step) ─────────────────────
def content_words(phrase):
    import lifelong_ood.lang.transplant as LT
    stop = LT.STOP - QUAL
    return [w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in stop]


def parse(text):
    """-> dict(pairs {(role, filler)}, frame, words, steps): per step (bind_hooks.split_conj), (V, verb) by primitives.step_key,
    the object phrase split at its first relation word into head (O) and landmark (L), and the goal phrase (G = preposition +
    target, for steps with a preposition).  Articulation steps ("close the top drawer of the cabinet") have no G: their object
    phrase is the fixture."""
    from lifelong_ood.policy import primitives as P
    pairs, frame, steps = set(), [], []
    for c in P.steps_of(text):
        c = c.strip()
        verb, prep, target = P.step_key(c).split("|")
        c2 = re.sub(r"^pick up (.+?) and (place|put) it", r"put \1", c)
        rest = c2[len(verb):].strip() if c2.startswith(verb) else c2
        goal = ""
        if prep and rest.endswith(f"{prep} {target}"):
            obj, goal = rest[: -len(f"{prep} {target}")].strip(), f"{prep} {target}"
        else:
            obj = rest
        m = re.search(REL, obj)
        head, land = (obj[:m.start()].strip(), obj[m.end():].strip()) if m else (obj, "")
        pairs |= {("V", verb)} | {("O", w) for w in content_words(head)} | {("L", w) for w in content_words(land)} \
            | {("G", w) for w in content_words(goal)}
        frame.append(f"{verb} {{O}}" + (f" {prep} {{G}}" if goal else ""))
        steps.append(dict(verb=verb, head=head, landmark=land, goal=goal))
    return dict(pairs=pairs, frame=" ; ".join(frame), words=set(re.findall(r"[a-z]+", text.lower())), steps=steps)


def cover(text, tx, allowed=None, k=2):
    """pl2c's greedy coverage over the pool (every training instruction except one identical to `text`; `allowed` restricts it):
    a source covers (r, w) iff w is one of ITS role-r fillers.  Ties: total pairs covered, same frame, word Jaccard, lower index.
    Slots: t1 = the pick leaning to the object roles ((#O + #L) - #G covered); ties: the first pick is t1."""
    N = parse(text)
    S = {t: parse(s) for t, s in tx.items() if s != text and (allowed is None or t in allowed)}
    picks, covered, trace = [], set(), []

    def key(t):
        c = N["pairs"] & S[t]["pairs"]
        jac = len(N["words"] & S[t]["words"]) / len(N["words"] | S[t]["words"])
        return (len(c - covered), len(c), S[t]["frame"] == N["frame"], round(jac, 6), -t)
    for _ in range(k):
        ks = sorted((t for t in S if t not in picks), key=key, reverse=True)
        trace.append(dict(pick=ks[0], key=list(key(ks[0])), runner_up=ks[1] if len(ks) > 1 else None))
        picks.append(ks[0])
        covered |= N["pairs"] & S[ks[0]]["pairs"]

    def lean(t):
        c = N["pairs"] & S[t]["pairs"]
        return sum(r in ("O", "L") for r, _ in c) - sum(r == "G" for r, _ in c)
    a, b = picks
    t1, t2 = (b, a) if lean(b) > lean(a) else (a, b)
    return dict(t1=int(t1), t2=int(t2), t1_text=tx[t1], t2_text=tx[t2], trace=trace, pairs=sorted(map(list, N["pairs"])),
                uncovered=sorted(map(list, N["pairs"] - covered)), steps=N["steps"],
                covered_by={str(t): sorted(map(list, N["pairs"] & S[t]["pairs"])) for t in picks})


def plan_tasks():
    """The frozen design table: per dev task its text, scene, bin, the blind and the scene-keyed cover pairs, the demo episode."""
    from lifelong_ood import l90c
    tx, scenes = store_pool()
    tt = task_table()
    out = {}
    for tid, b in dev_tasks():
        t = tt[tid]
        blind = cover(t.language, tx)
        sc = [k for k, s in tx.items() if t.scene in scenes[s] and s != t.language]
        scene = cover(t.language, tx, allowed=set(sc)) if len(sc) >= 2 else None
        out[tid] = dict(text=t.language, scene=t.scene, bin=b, demo_episode=int(l90c.task_episodes(tid)[0]),
                        blind=blind, scene_cover=scene, scene_pool=sorted(sc), scene_fallback=scene is None)
    return out


def damage_cells(plan):
    """-> [dict(cell, pair (dev task whose repairs leak), routed)]: the two training cells whose instruction equals a dev task's
    (the store routes that task's repair onto them), then one cell per remaining dev scene (lowest sha1 rank), until N_DAMAGE;
    each paired with its scene's dev task of lowest sha1 rank."""
    from lifelong_ood import l90c
    tt = task_table()
    h = lambda s: hashlib.sha1(s.encode()).hexdigest()                  # noqa: E731
    cells = [dict(cell=c, pair=d, routed=True) for c, d in COLLIDE.items()]
    for c, d in COLLIDE.items():
        assert tt[c].language == plan[d]["text"], (c, d)
    assert not any(tt[c].language == v["text"] for c in l90c.load_split()["train_l90"] for v in plan.values()
                   if c not in COLLIDE), "another instruction collision"
    by_scene = {}
    for d, v in plan.items():
        by_scene.setdefault(v["scene"], []).append(d)
    cand = []
    for sc, ds in by_scene.items():
        cs = [c for c in l90c.load_split()["train_l90"] if tt[c].scene == sc and c not in COLLIDE]
        if cs:
            cand.append(dict(cell=min(cs, key=lambda c: h("m1dmg:" + c)), pair=min(ds, key=lambda d: h("m1pair:" + d)),
                             routed=False))
    cand.sort(key=lambda x: h("m1scene:" + x["cell"]))
    return cells + cand[:N_DAMAGE - len(cells)]


def make_plan():
    p = plan_tasks()
    dm = damage_cells(p)
    os.makedirs(OUT, exist_ok=True)
    res = dict(tasks=p, damage=dm, arms=ARMS, lams=LAMS, lam_cov=LAM_COV, fresh=dict(dir=FRESH_DIR, base=BASE, n=N_STATES),
               seed=SEED, ckpt=CKPT)
    with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for tid, v in p.items():
        sc = v["scene_cover"]
        print(f"{tid:13} {v['bin']:4} {v['scene']:18} '{v['text']}'\n    blind ({v['blind']['t1']}) '{v['blind']['t1_text']}' + "
              f"({v['blind']['t2']}) '{v['blind']['t2_text']}' uncovered {v['blind']['uncovered']}\n    scene "
              + ("FALLBACK (blind)" if sc is None else f"({sc['t1']}) '{sc['t1_text']}' + ({sc['t2']}) '{sc['t2_text']}'"), flush=True)
    print("damage cells:", json.dumps(dm))


def load_plan():
    return json.load(open(os.path.join(OUT, "plan.json"), encoding="utf-8"))


# ── the store build: TLI latents of the pool under pi05_l90c (GPU) ───────────────────────────────────────────────
def backbone():
    from lifelong_ood.policy.backbone import PolicyBackbone
    return PolicyBackbone(CFG, checkpoint_dir=CKPT, device="cuda")


def store(shard):
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import tli as TL
    i, n = map(int, shard.split("/"))
    tx, _ = store_pool()
    todo = [t for t in sorted(tx) if t % n == i and not os.path.exists(lat_path(t))]
    os.makedirs(os.path.dirname(lat_path(0)), exist_ok=True)
    bk = backbone()
    demos = D.Demos(root=L90C_DATA)
    secs = {}
    for t in todo:
        t0 = time.time()
        lat, nf = TL.identify(bk, demos, t)
        tmp = lat_path(t)[:-4] + f".{os.getpid()}.tmp.npy"
        np.save(tmp, lat.astype(np.float16))
        os.replace(tmp, lat_path(t))
        secs[t] = time.time() - t0
        print(f"[m1 store] {t} '{tx[t]}': {nf} frames, {secs[t]:.0f}s", flush=True)
    from lifelong_ood import ledger
    print(ledger.record("m1_store", bench="libero_90_c", tag=f"shard{i}of{n}", seed=0,
                        metrics=dict(texts=len(todo), seconds=secs, frames_per_text=TL.NDEMO * TL.NFRAME),
                        bytes=dict(experts=0, tree=0, codes=int(sum(os.path.getsize(lat_path(t)) for t in todo)), samples=0,
                                   teacher_copies=0, total=int(sum(os.path.getsize(lat_path(t)) for t in todo))),
                        config=dict(ckpt=CKPT, data=L90C_DATA), script="experiments/m1_compose_create.py",
                        command=" ".join(sys.argv)))


# ── per-task fitting (GPU): compose selection, create / cc tokens, the LoRA curve ─────────────────────────────────
def _sync():
    import torch
    torch.cuda.synchronize()
    return time.time()


def init_token(bank, row, n, seed):
    """experiments/pf0_token_capacity.init_token: the mean of the prompt's valid language entries at one frame (pi0.5's 200
    language slots end the prefix), repeated n times, + 1% noise."""
    import torch
    from lifelong_ood.policy import token_fold as TF
    pre0 = bank.take([row])
    pad = pre0.pad_masks[0].cpu().numpy().astype(bool)
    L = len(pad)
    lang = [i for i in range(L - 200, L) if pad[i]]
    init = TF.KVToken.from_prefix(pre0, lang)
    k, v = init.k.mean(2, keepdim=True), init.v.mean(2, keepdim=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    nk = torch.randn((k.shape[0], k.shape[1], n, k.shape[3]), generator=g).to(k.device) * 0.01 * k.std()
    nv = torch.randn((v.shape[0], v.shape[1], n, v.shape[3]), generator=g).to(v.device) * 0.01 * v.std()
    return TF.KVToken(k.repeat(1, 1, n, 1) + nk, v.repeat(1, 1, n, 1) + nv)


def build_bank(bk, obs, comp=None, q=None):
    """adapt.PrefixBank of the demo's frames; with a Composer, each frame's prefix runs under its query's TLI coefficient."""
    from lifelong_ood.policy import adapt
    store_ = "cpu" if len(obs) > 400 else None
    if comp is None:
        return adapt.PrefixBank(bk, obs, store=store_)
    orig, pos = bk.prefix, [0]

    def pref(o):
        nb = int(o.state.shape[0])
        comp.q = np.asarray(q[pos[0]:pos[0] + nb])
        pos[0] += nb
        return orig(o)
    comp.attach()
    bk.prefix = pref
    try:
        return adapt.PrefixBank(bk, obs, store=store_)
    finally:
        bk.prefix = orig
        comp.detach()
        comp.q = None


def fit_task(bk, tid, v):
    import torch
    import lifelong_ood.lang.transplant as LT
    import pl2_compose as PC
    from lifelong_ood import ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import token_fold as TF
    import importlib.util
    spec_c0 = importlib.util.spec_from_file_location("c0", os.path.join(ROOT, "experiments", "c0_fit.py"))
    c0 = importlib.util.module_from_spec(spec_c0)
    spec_c0.loader.exec_module(c0)
    d = os.path.join(FITS, slug(tid))
    done = os.path.join(d, "fit.json")
    if os.path.exists(done):
        print(f"[m1] {tid}: {done} exists; skipped", flush=True)
        return
    os.makedirs(d, exist_ok=True)
    text = v["text"]
    norm = D.Norm.for_backbone(bk)
    tm = {}
    # the demo arrives
    t0 = _sync()
    d90 = D.Demos90()
    dm = PC.load_demo((d90, v["demo_episode"]), bk.H)
    T = dm["T"]
    obs = [dict(dm["_get"](i), prompt=text) for i in range(T)]
    idx, _ = D.chunk_indices(T, bk.H)
    full = D.pad_rows(c0.chunk_rows(dict(actions=dm["actions"], state=dm["state"]), idx, range(T), norm), bk.H, bk.a, bk.A)
    tm["demo"] = _sync() - t0
    # compose: the pairs' latents come from the store; compose_cov needs no GPU
    t0 = _sync()
    lat = {}
    for c in (v["blind"], v["scene_cover"]):
        if c is not None:
            for t in (c["t1"], c["t2"]):
                lat[t] = np.load(lat_path(t)).astype(np.float32)
    specs = {}

    def tli_spec(name, c, lam, token=None):
        p = os.path.join(d, f"{name}.json")
        LT.save_spec(p, [], prompt=text, tli=dict(d=lat[c["t1"]] - lat[c["t2"]], lam=lam, t1=c["t1"], t2=c["t2"]),
                     meta=dict(task=tid, t1_text=c["t1_text"], t2_text=c["t2_text"]), token=token)
        return p
    specs["compose_cov"] = tli_spec("compose_cov", v["blind"], LAM_COV)
    tm["compose_cov"] = _sync() - t0
    # compose_sel: the demo's flow loss over zs + candidates
    t0 = _sync()
    s0 = LT.DemoScorer(bk, dm["frames"], dm["chunks"], dm["states"], dm["carry"], dm["q"], K=K_DRAWS, seed=SCORE_SEED)
    cands = {"zs": None}
    for nm, c in (("blind", v["blind"]), ("scene", v["scene_cover"])):
        if c is None:
            continue
        for lam in LAMS:
            key = f"{c['t1']}.{c['t2']}.{lam}"
            if all(key != f"{x['t1']}.{x['t2']}.{x['lam']}" for x in cands.values() if x):
                cands[f"{nm}@{lam}"] = dict(t1=c["t1"], t2=c["t2"], lam=lam, c=c)
    losses = {}
    for nm, x in cands.items():
        losses[nm] = s0.loss(text) if x is None else s0.loss(text, LT.Composer(bk, (), tli=dict(d=lat[x["t1"]] - lat[x["t2"]],
                                                                                                 lam=x["lam"])))
    best = min(cands, key=lambda k: (losses[k], k != "zs"))             # ties keep zs
    if SMOKE and os.environ.get("M1_FORCE") in cands:                  # smoke only: exercise the cc path
        best = os.environ["M1_FORCE"]
    sel = cands[best]
    tm["compose_sel"] = tm["demo"] + _sync() - t0
    if sel is not None:
        specs["compose_sel"] = specs["compose_cov"] if (sel["t1"], sel["t2"], sel["lam"]) == (v["blind"]["t1"], v["blind"]["t2"], LAM_COV) \
            else tli_spec("compose_sel", sel["c"], sel["lam"])          # the same composition shares compose_cov's rollout
    print(f"[m1] {tid} '{text}': demo {T} frames; selection {json.dumps({k: round(x, 5) for k, x in losses.items()})} -> {best}",
          flush=True)
    del s0
    torch.cuda.empty_cache()
    hist = {}
    # cc: a token on top of compose_sel's conditioning (only where compose_sel composed)
    if sel is not None:
        t0 = _sync()
        comp = LT.Composer(bk, (), tli=dict(d=lat[sel["t1"]] - lat[sel["t2"]], lam=sel["lam"]))
        bank = build_bank(bk, obs, comp, q=np.arange(T) // REPLAN)
        tb = _sync() - t0
        tok = init_token(bank, 0, TOK_N, SEED).to(bk.device)
        hist["cc"] = TF.fit(bk, tok, bank, full, steps=TOK_STEPS, lr=TOK_LR, batch=BATCH, seed=SEED, log_every=200,
                            log=lambda x: print(f"[m1] {tid} cc {x}", flush=True))
        tp = os.path.join(d, "tok_cc.pt")
        torch.save(dict(token=tok.state_dict_fp16(), task=tid, text=text, n=TOK_N, on="compose_sel", sel=best), tp)
        specs["cc"] = tli_spec("cc", sel["c"], sel["lam"], token=tp)
        tm["cc"] = tm["compose_sel"] + tb + (_sync() - t0 - tb)
        tm["cc_bank"] = tb
        del bank, tok
        torch.cuda.empty_cache()
    # create: the token on the instruction's own cache; the same bank then trains the LoRA
    t0 = _sync()
    bank = build_bank(bk, obs)
    tb = _sync() - t0
    tm["bank"] = tb
    t1 = _sync()
    tok = init_token(bank, 0, TOK_N, SEED).to(bk.device)
    hist["create"] = TF.fit(bk, tok, bank, full, steps=TOK_STEPS, lr=TOK_LR, batch=BATCH, seed=SEED, log_every=200,
                            log=lambda x: print(f"[m1] {tid} create {x}", flush=True))
    tp = os.path.join(d, "tok_create.pt")
    torch.save(dict(token=tok.state_dict_fp16(), task=tid, text=text, n=TOK_N, on="instruction"), tp)
    tok_bytes = tok.nbytes()
    tm["create"] = tm["demo"] + tb + _sync() - t1
    p = os.path.join(d, "create.json")
    LT.save_spec(p, [], prompt=text, meta=dict(task=tid), token=tp)
    specs["create"] = p
    p = os.path.join(d, "create_unrouted.json")                         # damage: the token on an OLD prompt (prompt untouched)
    LT.save_spec(p, [], prompt=None, meta=dict(task=tid, unrouted=True), token=tp)
    specs["create_unrouted"] = p
    if sel is None:
        specs["cc"] = specs["create"]
        tm["cc"] = tm["compose_sel"] + tm["create"] - tm["demo"]
    del tok
    torch.cuda.empty_cache()
    # LoRA: one 1000-step run, snapshots at 50 / 200
    state, banks = adapt.inject(bk)
    key = f"m1_{slug(tid)}"
    params = add_node(banks, key, LORA_RANK)
    snaps, cnt = {}, [0]
    t1 = _sync()

    def snap():
        cnt[0] += 1
        s = 50 * cnt[0]
        if s in LORA_SNAP and s != LORA_STEPS:
            ts = _sync()
            snaps[s] = dict(path=os.path.join(d, f"lora{s}.pt"), secs=ts - t1)
            adapt.save(banks, key, snaps[s]["path"], meta=dict(task=tid, text=text, rank=LORA_RANK, steps=s, scope="expert"))
            t1_shift[0] += _sync() - ts                                   # the save is not training time
        return 0.0
    t1_shift = [0.0]
    hl = adapt.train(bk, state, key, params, bank, full, steps=LORA_STEPS, lr=LORA_LR, batch=BATCH, seed=0, eval_fn=snap,
                     eval_every=50, log_every=100, log=lambda x: print(f"[m1] {tid} {x}", flush=True))
    te = _sync() - t1
    p = os.path.join(d, f"lora{LORA_STEPS}.pt")
    lora_bytes = adapt.save(banks, key, p, meta=dict(task=tid, text=text, rank=LORA_RANK, steps=LORA_STEPS, scope="expert"))
    snaps[LORA_STEPS] = dict(path=p, secs=te - t1_shift[0])
    for s, x in snaps.items():
        specs[f"lora{s}"] = x["path"]
        tm[f"lora{s}"] = tm["demo"] + tb + x["secs"]
    hist["lora"] = {k: v_ for k, v_ in hl.items() if k in ("step", "loss")}
    del bank
    torch.cuda.empty_cache()
    res = dict(task=tid, text=text, bin=v["bin"], T=T, demo_episode=v["demo_episode"], selection=dict(losses=losses, best=best,
               cands={k: (None if x is None else dict(t1=x["t1"], t2=x["t2"], lam=x["lam"])) for k, x in cands.items()}),
               specs=specs, seconds=tm, curves=hist,
               bytes=dict(token=tok_bytes, lora=lora_bytes, compose_pointer=len(json.dumps(dict(t1=0, t2=0, lam=24))),
                          tli_d_materialised=int(lat[v["blind"]["t1"]].size * 2)),
               gpu=torch.cuda.get_device_name(0))
    tmp = done + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, done)
    if SMOKE:
        print(f"[m1] SMOKE {tid}: seconds {json.dumps(tm)}", flush=True)
        return
    print(ledger.record("m1_fit", bench="libero_90_c", tag="fit", seed=SEED, keys=dict(task=tid),
                        metrics={k: res[k] for k in ("selection", "seconds", "T", "bin")} | dict(
                            final_loss={k: (h["loss"][-1] if h.get("loss") else None) for k, h in hist.items()}),
                        bytes=dict(experts=lora_bytes, tree=0, codes=tok_bytes * (2 if sel is not None else 1), samples=0,
                                   teacher_copies=0, total=lora_bytes + tok_bytes * (2 if sel is not None else 1)),
                        config=dict(tok=dict(n=TOK_N, steps=TOK_STEPS, lr=TOK_LR, batch=BATCH), lora=dict(rank=LORA_RANK,
                                    lr=LORA_LR, steps=LORA_STEPS, snaps=LORA_SNAP, batch=BATCH), lams=LAMS, ckpt=CKPT),
                        script="experiments/m1_compose_create.py", command=" ".join(sys.argv), artifacts=[done]))
    # the injected LoRA banks stay in this process (off); one task per job keeps every timing clean
    print(f"[m1] {tid} done: seconds {json.dumps({k: round(x, 1) for k, x in tm.items()})}", flush=True)


def fit(task):
    plan = load_plan()
    tasks = [task] if task != "all" else list(plan["tasks"])
    bk = backbone()
    for tid in tasks:
        fit_task(bk, tid, plan["tasks"][tid])


# ── start states ─────────────────────────────────────────────────────────────────────────────────────────────────
def fresh_check():
    """20 new states per dev task, each > 1e-6 from every state of every earlier libero_90 fresh set (results/robot/fresh_*)."""
    out = {}
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in USED_BASES), BASE
    for tid, _ in dev_tasks():
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
    with open(os.path.join(OUT, "fresh_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


# ── closed loop ──────────────────────────────────────────────────────────────────────────────────────────────────
def fit_of(tid):
    f = os.path.join(FITS, slug(tid), "fit.json")
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def mode_of(arm, specs):
    if arm == "zs":
        return "ours"
    if arm == "compose_sel" and "compose_sel" not in specs:
        return "ours"                                                   # the guard kept the instruction
    p = specs[arm]
    return f"lora:{p}" if arm.startswith("lora") else f"lt:{p}"


def roll_out(exp, tid, mode):
    h = hashlib.sha1(mode.encode()).hexdigest()[:10]
    return os.path.join(OUT, "roll", f"{exp}_{slug(tid)}_{h}.json")


def arm_modes(tid):
    """-> {arm: MODE} for a fitted task (identical MODE strings share one rollout)."""
    ft = fit_of(tid)
    return None if ft is None else {a: mode_of(a, ft["specs"]) for a in ARMS}


def _line(tid, mode, exp, tag, fresh, seed):
    suite, i = tid.split(":")
    sp = dict(MODE=mode, SUITE=suite, TASKS=i, TRIALS=N_STATES, FIRST=0, EXPERIMENT=exp, TAG=tag, PROGRAM_BYTES=0,
              OUT=roll_out(exp, tid, mode), BACKBONE_CONFIG=CFG, BACKBONE_CKPT=CKPT,
              BIND="track", RECORD="1", LIGHT="1", OBJECTS="1", ALL_OBJECTS="1")   # failure map (PLAN_STORY 2): per-episode
    # states, actions and every movable object's pose (no images), plus the first object moved while closed and its step
    if seed:
        sp["SERVE_SEED"] = str(seed)
    if fresh:
        sp["FRESH"] = os.path.join(R, FRESH_DIR)
    return sp


def submit(lines, lanes=10, per=4, dry=False):
    import launch_pack as LP
    todo = [s for s in lines if not os.path.exists(s["OUT"])]
    seen, uniq = set(), []
    for s in todo:
        if s["OUT"] not in seen:
            seen.add(s["OUT"])
            uniq.append(s)
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "roll"), exist_ok=True)
    last, jobs = [None] * lanes, []
    for j, i in enumerate(range(0, len(uniq), per)):
        grp = uniq[i:i + per]
        path = os.path.join(OUT, "packs", f"{grp[0]['EXPERIMENT']}_{j:03d}_{int(time.time())}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        lane = j % lanes
        args = ([f"--dependency=afterany:{last[lane]}"] if last[lane] else []) + [
            "--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=04:00:00", "--gres=gpu:a40:1",
            f"--exclude={EXCLUDE}", f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        jid = LP.sbatch(args, dry)
        last[lane] = jid
        jobs.append(jid)
        print(f"pack {os.path.basename(path)} [{', '.join(s['TASKS'] + ':' + s['TAG'] for s in grp)}] lane {lane} -> {jid}", flush=True)
    print(f"{len(uniq)} instances, {len(jobs)} packs: {':'.join(jobs)}")
    return jobs


def smoke_serve(task, dry):
    """The serving smoke (excluded from every number): the smoke fit's create / cc / lora specs, 2 episodes from the task's STORED
    init states (never fresh_m1), one pack.  Outputs results/m1/smoke/roll_*.json."""
    assert SMOKE, "set M1_SMOKE=1"
    ft = fit_of(task)
    lines = []
    for arm in ("cc", "create", "lora200", "compose_cov"):
        m = mode_of(arm, ft["specs"])
        sp = _line(task, m, "m1_smoke", arm, False, SEED)
        sp.update(TRIALS=2, OUT=os.path.join(OUT, "smoke", f"roll_{slug(task)}_{arm}.json"))
        lines.append(sp)
    submit(lines, lanes=1, per=4, dry=dry)


def launch(dry):
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    plan = load_plan()
    lines, missing = [], []
    for tid in plan["tasks"]:
        am = arm_modes(tid)
        if am is None:
            missing.append(tid)
            continue
        for arm, mode in am.items():
            lines.append(_line(tid, mode, "m1_main", arm, True, SEED))
    print("not fitted:", missing)
    submit(lines, dry=dry)


def damage_modes(c):
    """-> {arm: MODE} on a damage cell: base (parity, not rolled), the unrouted leaks, and (collision cells) the routed repairs."""
    ft = fit_of(c["pair"])
    sp = ft["specs"]
    out = {"lora_unrouted": f"lora:{sp['lora1000']}", "create_unrouted": f"lt:{sp['create_unrouted']}"}
    if c["routed"]:
        for a in ("compose_sel", "create", "cc", "lora1000"):
            out[f"{a}_routed"] = mode_of(a, sp)
    return out


def launch_damage(dry):
    plan = load_plan()
    lines = []
    for c in plan["damage"]:
        if fit_of(c["pair"]) is None:
            print("not fitted:", c["pair"])
            continue
        for arm, mode in damage_modes(c).items():
            if mode != "ours":
                lines.append(_line(c["cell"], mode, "m1_damage", arm, False, 0))
    submit(lines, dry=dry)


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _eps(f):
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for t in json.load(open(f))["tasks"].values() for e in t["episodes"]}


def parity_base():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", "l90c", "parity_pi05_l90c_5000_*.json")):
        z = json.load(open(f))
        for k, t in z["tasks"].items():
            out[f"{z['suite']}:{k}"] = {int(e["trial"]): int(bool(e["success"])) for e in t["episodes"]}
    return out


def holm(ps, alpha=0.05):
    order = sorted(ps, key=lambda k: ps[k])
    out, stop = {}, False
    for j, k in enumerate(order):
        thr = alpha / (len(ps) - j)
        ok = (not stop) and ps[k] <= thr
        stop = stop or not ok
        out[k] = dict(p=ps[k], threshold=thr, reject=bool(ok))
    return out


def read(record):
    from lifelong_ood import stats as ST
    plan = load_plan()
    tab, missing, per, fits = {}, [], {}, {}
    for tid, v in plan["tasks"].items():
        am = arm_modes(tid)
        if am is None:
            missing.append(tid)
            continue
        z = {a: _eps(roll_out("m1_main", tid, m)) for a, m in am.items()}
        if any(e is None or len(e) != N_STATES for e in z.values()):
            missing.append(tid)
            continue
        tab[tid] = z
        fits[tid] = fit_of(tid)
        per[tid] = {a: int(sum(e.values())) for a, e in z.items()} | dict(bin=v["bin"], sel=fits[tid]["selection"]["best"])
    res = dict(tasks=len(tab), missing=missing, n_per_task=N_STATES, per_task=per)
    if not tab:
        print("[m1] nothing to read; missing", missing)
        return
    res["totals"] = {a: int(sum(per[t][a] for t in per)) for a in ARMS}
    res["rate_ci"] = {a: {k: x for k, x in ST.bootstrap_rate(tab, a, list(tab)).items() if k != "samples"} for a in ARMS}
    by = {}
    for t in per:
        g = by.setdefault(per[t]["bin"], {a: 0 for a in ARMS} | dict(tasks=0))
        g["tasks"] += 1
        for a in ARMS:
            g[a] += per[t][a]
    res["by_bin"] = by

    def cmp(a, b, margin=None):
        pairs = ST.paired(tab, a, b, list(tab))
        d = ST.bootstrap_diff(pairs, B=10000, return_samples=True)
        s = d.pop("samples", None)
        bb, cc = ST.discordant(pairs)
        out = dict(x=res["totals"][a], y=res["totals"][b], diff=d, discordant=[bb, cc], sign_p=float(ST.sign_test(bb, cc)))
        if margin is not None:
            out["p_noninferior"] = float(np.mean(np.asarray(s) <= -margin)) if s is not None else None
        return out
    h1 = cmp("compose_sel", "zs")
    h2 = cmp("create", "lora1000", margin=0.05)
    h3 = cmp("cc", "create", margin=0.05)
    bytes_ = dict(token=int(np.mean([f["bytes"]["token"] for f in fits.values()])),
                  lora=int(np.mean([f["bytes"]["lora"] for f in fits.values()])))
    h2["bytes_ratio"] = bytes_["token"] / bytes_["lora"]
    p1 = h1["sign_p"] if h1["x"] > h1["y"] else 1.0
    hm = holm(dict(H1=p1, H2=h2["p_noninferior"], H3=h3["p_noninferior"]))
    h1["passes"] = hm["H1"]["reject"]
    h2["passes"] = bool(hm["H2"]["reject"] and h2["bytes_ratio"] <= 0.01)
    h3["passes"] = hm["H3"]["reject"]
    res.update(H1=h1, H2=h2, H3=h3, holm=hm, bytes=bytes_)
    res["secondary"] = {f"{a}_vs_{b}": cmp(a, b) for a, b in (("compose_cov", "zs"), ("create", "zs"), ("cc", "zs"),
                                                              ("lora1000", "zs"), ("lora200", "zs"), ("lora50", "zs"),
                                                              ("cc", "create"), ("cc", "lora1000"), ("compose_sel", "lora1000"),
                                                              ("compose_cov", "compose_sel"))}
    sec = {a: float(np.mean([f["seconds"][a] for f in fits.values()])) for a in ARMS if a != "zs"}
    sec["zs"] = 0.0
    sec["compose_cov"] = float(np.mean([f["seconds"]["compose_cov"] for f in fits.values()]))
    res["seconds"] = sec
    res["selection"] = {k: sum(f["selection"]["best"].split("@")[0] == k for f in fits.values()) for k in ("zs", "blind", "scene")}
    rate = {a: res["totals"][a] / (N_STATES * len(tab)) for a in ARMS}
    curve = [(s, rate[f"lora{s}"], sec[f"lora{s}"]) for s in LORA_SNAP]
    res["budget_to_threshold"] = {ref: next(({"steps": s, "rate": r, "seconds": t} for s, r, t in curve if r >= rate[ref]), None)
                                  for ref in ("zs", "compose_sel", "create", "cc")}
    # damage
    base = parity_base()
    dmg, dtab = {}, {}
    for c in plan["damage"]:
        if fit_of(c["pair"]) is None:
            continue
        b = base.get(c["cell"])
        z = {a: _eps(roll_out("m1_damage", c["cell"], m)) if m != "ours" else b for a, m in damage_modes(c).items()}
        if b is None or any(e is None for e in z.values()):
            continue
        dtab[c["cell"]] = dict(base=b, **z)
    if len(dtab) == N_DAMAGE:
        arms_d = {"lora_unrouted": "lora_unrouted", "create_unrouted": "create_unrouted"}
        for a in ("compose_sel", "create", "cc", "lora1000"):
            arms_d[f"{a}_routed"] = f"{a}_routed"
        for a in arms_d:
            ch = [np.mean(list(z[a].values())) - np.mean(list(z["base"].values())) if a in z else 0.0 for z in dtab.values()]
            dmg[a] = dict(mean_change=float(np.mean(ch)), per_cell={k: (float(np.mean(list(z[a].values())) -
                          np.mean(list(z["base"].values()))) if a in z else 0.0) for k, z in dtab.items()})
        dmg["base_rate"] = float(np.mean([np.mean(list(z["base"].values())) for z in dtab.values()]))
        res["H4"] = dict(compose=dmg["compose_sel_routed"]["mean_change"], create=dmg["create_routed"]["mean_change"],
                         passes=bool(dmg["compose_sel_routed"]["mean_change"] >= -0.01 and dmg["create_routed"]["mean_change"] >= -0.01))
    res["damage"] = dmg
    res["damage_cells"] = len(dtab)
    print(json.dumps({k: res[k] for k in ("tasks", "missing", "totals", "by_bin", "seconds", "bytes", "selection",
                                          "budget_to_threshold", "holm")}, indent=1))
    for h in ("H1", "H2", "H3"):
        x = res[h]
        print(f"[m1] {h} {x['x']} vs {x['y']}, discordant {x['discordant']}, sign p {x['sign_p']:.3g}, diff {x['diff']['diff']:+.3f} "
              f"[{x['diff']['lo']:+.3f}, {x['diff']['hi']:+.3f}], passes {x['passes']}")
    print("[m1] H4", res.get("H4"), "damage", {k: v_["mean_change"] for k, v_ in dmg.items() if isinstance(v_, dict)})
    for t in per:
        print(f"  {t:13} {per[t]['bin']:4} " + " ".join(f"{a} {per[t][a]:2d}" for a in ARMS) + f"  sel {per[t]['sel']}")
    with open(os.path.join(OUT, "read.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("m1_read", bench="libero_90_c", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(states=FRESH_DIR, base=BASE, episodes=N_STATES, arms=ARMS, ckpt=CKPT,
                                        prereg="m3_prereg tag pm1_compose_create"),
                            script="experiments/m1_compose_create.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("plan", "store", "fit", "fresh-check", "launch", "launch-damage", "read", "dry", "record", "smoke-serve"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--task", default="all")
    a = ap.parse_args()
    if a.plan:
        make_plan()
    if a.store:
        store(a.shard)
    if a.fresh_check:
        fresh_check()
    if a.fit:
        fit(a.task)
    if a.launch:
        launch(a.dry)
    if a.smoke_serve:
        smoke_serve(a.task, a.dry)
    if a.launch_damage:
        launch_damage(a.dry)
    if a.read:
        read(a.record)
