"""PLAN L4 (docs/PLAN_L4.md section 7; pre-registration experiments/prereg/pl4_store.py, ledger m3_prereg tag pl4_store):
A CONTINUAL STORE OF CREATED LANGUAGE TOKENS on LIBERO-90-C's 29 dev tasks, pi05_l90c main/5000 frozen.

The unit is a ROLE TOKEN: a KV token (policy/token_fold.KVToken, n 2 entries per layer, 36,864 bytes fp16) appended to the
instruction's cache in one phase of the episode, read from the policy's own gripper (held_ops._Grasp: carrying after 3 closed
queries until it opens):
  O  the open-gripper phase (approach: where the object role binds)      keyed (scene, O, verb + object words) per step
  G  the carrying phase (where the goal role binds)                       keyed (scene, G, preposition + goal words) per step;
                                                                          an articulation step's closed phase: (scene, G, manip:verb + object words)
A task's two role tokens are fitted JOINTLY on its one demo (own-scene IPEC episode 0; never executed or replayed) by flow matching,
each demo frame under the token of its phase (the same _Grasp rule run offline on the demo's gripper command).  An O + G pair has
the bytes of M1's whole-task token (n 4).

Arms (served from experiments/serve_guided.py `lt:<spec>`, role_tokens in lifelong_ood/lang/transplant.py):
  zs        the instruction                       whole     M1's whole-task token (results/m1/fits/*/create.json)
  lora1000  M1's per-task LoRA r16 (1000 steps)   scratch   the task's own role pair, 1000 steps (snapshots 50 / 200)
  reuse     an earlier task's stored role tokens whose key is contained in this task's role items, no fit (zero-shot)
  complete  the reused tokens frozen + new tokens for the uncovered roles, fitted on this task's demo (1000 steps)
  causal    complete with the reused tokens removed (only the new ones)
  init      this task's pair initialised from the partner's pair, 200 steps (snapshots 50 / 200)
The stream (two mirrored orders): practice guard on 5 practice episodes (separate fresh states), reuse first, create only when
nothing passes; see `stream()`.

    sbatch ... cpu.sbatch SCRIPT="experiments/l4_store.py --plan"           design table -> results/l4/plan.json
    ... --fresh-check | --launch-scratch [--dry] | --launch-pairs [--dry] | --launch-rolls [--dry] | --launch-stream [--dry]
    ... --read [--record]
    GPU: --fit-scratch --task libero_90:9 | --fit-request results/l4/requests/<name>.json
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
import m1_compose_create as M1                                          # noqa: E402

R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "l4")
FITS = os.path.join(OUT, "fits")
REQ = os.path.join(OUT, "requests")
SPECS = os.path.join(OUT, "specs")
ROLL = os.path.join(OUT, "roll")
EVAL_DIR, EVAL_BASE, N_EVAL = "fresh_l4", 6100000, 20                    # evaluation start states (new)
PRAC_DIR, PRAC_BASE, N_PRAC = "fresh_l4p", 6300000, 5                    # practice start states (new, disjoint)
SEED = 13
EXCLUDE = M1.EXCLUDE
N_ROLE = 2                                                              # entries per layer per role token (O + G = M1's n 4)
STEPS, SNAPS = 1000, (50, 200)
INIT_STEPS, INIT_SNAPS = 200, (50,)
LR, BATCH = 1e-2, 32
MIN_FRAMES = 10                                                         # a role is fitted only if its phase holds >= 10 demo frames
TAU = 3                                                                 # practice successes (of 5) for a candidate to pass
HEADROOM = 15                                                           # target tasks with M1 zs <= 15 / 20 (fresh_m1, known)
REPLAN = 5
SMOKE = os.environ.get("L4_SMOKE") == "1"
if SMOKE:
    STEPS, SNAPS, INIT_STEPS, INIT_SNAPS = 20, (10,), 10, (5,)
    FITS = os.path.join(OUT, "smoke", "fits")
    REQ = os.path.join(OUT, "smoke", "requests")
    SPECS = os.path.join(OUT, "smoke", "specs")
    ROLL = os.path.join(OUT, "smoke", "roll")


def slug(t):
    return t.replace(":", "_")


def h(s, n=10):
    return hashlib.sha1(s.encode()).hexdigest()[:n]


# ── roles and keys ───────────────────────────────────────────────────────────────────────────────────────────────
def roles(text):
    """-> {"O": sorted items, "G": sorted items}.  Per step (primitives.steps_of / step_key, as M1's parse): O item
    'O|verb|object words' (head + landmark content words, qualifiers kept); G item 'G|prep|goal words', or for a step with no goal
    phrase (articulation) 'G|manip:verb|object words'."""
    from lifelong_ood.policy import primitives as P
    O, G = set(), set()
    for c in P.steps_of(text):
        c = c.strip()
        verb, prep, target = P.step_key(c).split("|")
        c2 = re.sub(r"^pick up (.+?) and (place|put) it", r"put \1", c)
        rest = c2[len(verb):].strip() if c2.startswith(verb) else c2
        if prep and rest.endswith(f"{prep} {target}"):
            obj, goal = rest[: -len(f"{prep} {target}")].strip(), target
        else:
            obj, goal = rest, ""
        ow = " ".join(sorted(set(M1.content_words(obj))))
        O.add(f"O|{verb}|{ow}")
        G.add(f"G|{prep}|{' '.join(sorted(set(M1.content_words(goal))))}" if goal else f"G|manip:{verb}|{ow}")
    return dict(O=sorted(O), G=sorted(G))


def phases(actions, replan=REPLAN):
    """held_ops._Grasp's rule run offline on a demo's gripper command: per query (every `replan` frames) the phase is the carrying
    state BEFORE the query's chunk is observed; closed = the chunk's 5th action closes (> 0); carrying after 3 closed queries,
    released at the first open query.  -> bool [T] (True = G phase)."""
    a = np.asarray(actions)
    T = len(a)
    out = np.zeros(T, bool)
    carrying, run = False, 0
    for f0 in range(0, T, replan):
        out[f0:f0 + replan] = carrying
        if a[min(f0 + 4, T - 1), 6] > 0:
            run += 1
            if run >= 3 and not carrying:
                carrying = True
        else:
            carrying = False
            run = 0
    return out


# ── the design table ─────────────────────────────────────────────────────────────────────────────────────────────
def make_plan():
    from lifelong_ood.policy import data as D
    m1p = M1.load_plan()
    m1r = json.load(open(os.path.join(M1.OUT, "read.json"), encoding="utf-8"))
    d90 = D.Demos90()
    tasks = {}
    for tid, v in m1p["tasks"].items():
        ph = phases(d90.load(v["demo_episode"])["actions"])
        tasks[tid] = dict(text=v["text"], scene=v["scene"], bin=v["bin"], demo_episode=v["demo_episode"], roles=roles(v["text"]),
                          frames=dict(O=int((~ph).sum()), G=int(ph.sum())), m1_zs=int(m1r["per_task"][tid]["zs"]))
        tasks[tid]["fit_roles"] = [r for r in ("O", "G") if tasks[tid]["frames"][r] >= MIN_FRAMES]
    # reuse pairs (e -> l): same scene, and a role token of e (key = all of e's role-r items) is contained in l's role-r items
    pairs = []
    for e, ve in tasks.items():
        for l, vl in tasks.items():
            if e == l or ve["scene"] != vl["scene"]:
                continue
            sh = [r for r in ve["fit_roles"] if set(ve["roles"][r]) <= set(vl["roles"][r])]
            if sh:
                pairs.append(dict(src=e, dst=l, roles=sh, shared=int(sum(len(ve["roles"][r]) for r in sh)),
                                  headroom=vl["m1_zs"] <= HEADROOM))
    canon = {}
    for l in tasks:
        ps = [p for p in pairs if p["dst"] == l and p["headroom"]]
        if ps:
            canon[l] = min(ps, key=lambda p: (-p["shared"], h(f"l4partner:{p['src']}>{l}", 40)))["src"]
    # cross-scene, same text (secondary): the source's whole role pair served on the target
    cross = [dict(src=e, dst=l) for e, ve in tasks.items() for l, vl in tasks.items()
             if e != l and ve["text"] == vl["text"] and ve["scene"] != vl["scene"]]
    order_a = sorted(tasks, key=lambda t: h(f"l4order:{t}", 40))
    orders = dict(A=order_a, B=order_a[::-1])
    partners = {l: sorted({p["src"] for p in pairs if p["dst"] == l}) for l in tasks}
    fwt = {}
    for t in tasks:
        nb = {k: sum(o.index(e) < o.index(t) for e in partners[t]) for k, o in orders.items()}
        if nb["A"] != nb["B"]:
            fwt[t] = dict(late="A" if nb["A"] > nb["B"] else "B", before=nb)
    tt = M1.task_table()
    cells = []
    for c in m1p["damage"]:
        t = tt[c["cell"]]
        cells.append(dict(cell=c["cell"], text=t.language, scene=t.scene, roles=roles(t.language)))
    res = dict(tasks=tasks, pairs=pairs, canonical=canon, cross=cross, orders=orders, partners=partners, fwt=fwt, cells=cells,
               settings=dict(n_role=N_ROLE, steps=STEPS, snaps=SNAPS, init_steps=INIT_STEPS, init_snaps=INIT_SNAPS, lr=LR,
                             batch=BATCH, min_frames=MIN_FRAMES, tau=TAU, n_prac=N_PRAC, headroom=HEADROOM, seed=SEED,
                             eval=dict(dir=EVAL_DIR, base=EVAL_BASE, n=N_EVAL), practice=dict(dir=PRAC_DIR, base=PRAC_BASE, n=N_PRAC)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for t, v in tasks.items():
        print(f"{t:13} {v['scene']:18} frames O {v['frames']['O']:3d} G {v['frames']['G']:3d} fit {v['fit_roles']} {v['roles']}  "
              f"'{v['text']}'", flush=True)
    print("pairs:", len(pairs), "headroom:", sum(p["headroom"] for p in pairs))
    for p in pairs:
        print("  ", p["src"], "->", p["dst"], p["roles"], "headroom" if p["headroom"] else "")
    print("canonical:", json.dumps(canon))
    print("cross:", json.dumps(cross))
    print("order A:", " ".join(t.split(":")[1] for t in order_a))
    print("fwt tasks:", json.dumps(fwt))
    print("cells:", json.dumps(cells))


def load_plan():
    return json.load(open(os.path.join(OUT, "plan.json"), encoding="utf-8"))


def fresh_check():
    """Both new sets: 20 eval and 5 practice states per dev task, each > 1e-6 from every state of every other fresh set."""
    used = M1.USED_BASES + (M1.BASE,)
    for b in (EVAL_BASE, PRAC_BASE):
        assert not any(u <= b < u + 1000 or b <= u < b + 1000 for u in used), b
    out = {}
    for tid in load_plan()["tasks"]:
        i = tid.split(":")[1]
        x = np.load(os.path.join(R, EVAL_DIR, f"libero_90_{i}.npy"))
        y = np.load(os.path.join(R, PRAC_DIR, f"libero_90_{i}.npy"))
        assert len(x) == N_EVAL and len(y) == N_PRAC, (tid, x.shape, y.shape)
        dmin, sets = np.inf, []
        for f in sorted(glob.glob(os.path.join(R, "fresh_*", f"libero_90_{i}.npy"))):
            nm = os.path.basename(os.path.dirname(f))
            z = np.load(f)
            for a, an in ((x, EVAL_DIR), (y, PRAC_DIR)):
                if nm == an or z.shape[1] != a.shape[1]:
                    continue
                dmin = min(dmin, float(np.min(np.linalg.norm(a[:, None] - z[None], axis=-1))))
            sets.append(nm)
        assert dmin > 1e-6, (tid, dmin)
        out[tid] = dict(min_dist=dmin, compared=sets)
        print(tid, f"min distance {dmin:.4f} over {len(sets)} sets", flush=True)
    with open(os.path.join(OUT, "fresh_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


# ── fitting (GPU) ────────────────────────────────────────────────────────────────────────────────────────────────
def load_token(p, device):
    import torch
    from lifelong_ood.policy.token_fold import KVToken
    d = torch.load(p, map_location=device)["token"]
    return KVToken(d["k"].float(), d["v"].float()).to(device)


def demo_bank(bk, v):
    """-> (bank, chunks_full, carry [T], seconds dict) for a task's one demo (M1's recipe: every frame, the instruction)."""
    import importlib.util
    import pl2_compose as PC
    from lifelong_ood.policy import data as D
    sp = importlib.util.spec_from_file_location("c0", os.path.join(ROOT, "experiments", "c0_fit.py"))
    c0 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c0)
    tm = {}
    t0 = M1._sync()
    dm = PC.load_demo((D.Demos90(), v["demo_episode"]), bk.H)
    T = dm["T"]
    obs = [dict(dm["_get"](i), prompt=v["text"]) for i in range(T)]
    idx, _ = D.chunk_indices(T, bk.H)
    norm = D.Norm.for_backbone(bk)
    full = D.pad_rows(c0.chunk_rows(dict(actions=dm["actions"], state=dm["state"]), idx, range(T), norm), bk.H, bk.a, bk.A)
    carry = phases(dm["actions"])
    tm["demo"] = M1._sync() - t0
    t0 = M1._sync()
    bank = M1.build_bank(bk, obs)
    tm["bank"] = M1._sync() - t0
    return bank, full, carry, tm


def fit_roles(bk, bank, full, carry, frozen, new, steps, snaps, save, log):
    """Joint flow matching of the `new` role tokens ({role: KVToken}); each sampled demo frame runs under its phase's tokens
    (frozen[role] + new[role]).  save(step) is called at each snapshot and at the end (not timed)."""
    import torch
    from lifelong_ood.policy.adapt import _flow_batch
    from lifelong_ood.policy.token_fold import attach
    torch.manual_seed(SEED)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    for r in frozen:
        for tk in frozen[r]:
            tk.requires_grad_(False)
    params = [p for tk in new.values() for p in tk.parameters()]
    opt = torch.optim.Adam(params, lr=LR)
    A = torch.as_tensor(np.asarray(full, np.float32), device=bk.device)
    carry = np.asarray(carry, bool)
    hist, shift, t0 = dict(step=[], loss=[]), 0.0, M1._sync()
    secs = {}
    for s in range(1, steps + 1):
        idx = rng.integers(0, bank.n, size=min(BATCH, bank.n))
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        a = A[idx]
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        tot = torch.zeros((), device=bk.device)
        for r, m in (("O", ~carry[idx]), ("G", carry[idx])):
            if not m.any():
                continue
            j = np.nonzero(m)[0]
            pre = bank.take(idx[j])
            for tk in frozen.get(r, []) + ([new[r]] if r in new else []):
                pre = attach(pre, tk)
            jt = torch.as_tensor(j, device=bk.device)
            v = bk.v_full(xt[jt], t[jt], pre)
            tot = tot + ((v - u[jt]) ** 2).sum()
        loss = tot / u.numel()
        if params:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        if s % 50 == 0 or s == steps:
            hist["step"].append(s)
            hist["loss"].append(float(loss))
        if s in snaps or s == steps:
            ts = M1._sync()
            secs[s] = ts - t0 - shift
            save(s)
            shift += M1._sync() - ts
        if s % 200 == 0 or s == steps:
            log(f"[l4] step {s}: flow {float(loss):.4f} ({M1._sync() - t0 - shift:.0f}s)")
    return hist, secs


def save_tok(tok, path, **meta):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(token=tok.state_dict_fp16(), n=tok.n, **meta), path + ".tmp")
    os.replace(path + ".tmp", path)


def fit_scratch(bk, tid):
    """The task's own role pair: O and G (roles with >= MIN_FRAMES phase frames), M1's init (frame 0's language entries + 1%
    noise; G's noise seed + 1), 1000 steps, snapshots at 50 / 200."""
    plan = load_plan()
    v = plan["tasks"][tid]
    d = os.path.join(FITS, slug(tid), "scratch")
    done = os.path.join(d, "fit.json")
    if os.path.exists(done):
        print(f"[l4] {tid} scratch exists", flush=True)
        return
    import torch
    bank, full, carry, tm = demo_bank(bk, v)
    new = {r: M1.init_token(bank, 0, N_ROLE, SEED + (r == "G")).to(bk.device) for r in v["fit_roles"]}
    paths = {}

    def save(s):
        for r, tk in new.items():
            p = os.path.join(d, f"{r}_{s}.pt")
            save_tok(tk, p, task=tid, role=r, items=v["roles"][r], kind="scratch", step=s)
            paths.setdefault(r, {})[s] = p
    hist, secs = fit_roles(bk, bank, full, carry, {}, new, STEPS, SNAPS, save, lambda x: print(f"[l4] {tid} scratch {x}", flush=True))
    res = dict(task=tid, kind="scratch", T=int(len(full)), frames=v["frames"], roles=v["roles"], fit_roles=v["fit_roles"], tokens=paths,
               seconds=dict(tm, **{f"fit{s}": x for s, x in secs.items()}, **{f"total{s}": tm["demo"] + tm["bank"] + x for s, x in secs.items()}),
               curve=hist, bytes={r: tk.nbytes() for r, tk in new.items()}, gpu=torch.cuda.get_device_name(0))
    write_json(done, res)
    record_fit(res, tid, "scratch")


def fit_request(bk, path):
    """A request file: dict(task, kind complete|init, frozen {O: [paths], G: [paths]}, new {role: items covered by the new token},
    init {role: path or None}, name).  complete: 1000 steps, snapshots 50 / 200; init: 200 steps, snapshot 50."""
    rq = json.load(open(path, encoding="utf-8"))
    tid = rq["task"]
    v = load_plan()["tasks"][tid]
    d = os.path.join(FITS, slug(tid), rq["name"])
    done = os.path.join(d, "fit.json")
    if os.path.exists(done):
        print(f"[l4] {rq['name']} exists", flush=True)
        return
    import torch
    bank, full, carry, tm = demo_bank(bk, v)
    frozen = {r: [load_token(p, bk.device) for p in rq["frozen"].get(r, [])] for r in ("O", "G")}
    new = {}
    for r in rq["new"]:
        src = (rq.get("init") or {}).get(r)
        if src:
            new[r] = load_token(src, bk.device)
            new[r].k.requires_grad_(True)
            new[r].v.requires_grad_(True)
        else:
            new[r] = M1.init_token(bank, 0, N_ROLE, SEED + (r == "G")).to(bk.device)
    steps, snaps = (INIT_STEPS, INIT_SNAPS) if rq["kind"] == "init" else (STEPS, SNAPS)
    paths = {}

    def save(s):
        for r, tk in new.items():
            p = os.path.join(d, f"{r}_{s}.pt")
            save_tok(tk, p, task=tid, role=r, items=rq["new"][r], kind=rq["kind"], step=s, request=rq["name"])
            paths.setdefault(r, {})[s] = p
    hist, secs = fit_roles(bk, bank, full, carry, frozen, new, steps, snaps, save,
                           lambda x: print(f"[l4] {tid} {rq['name']} {x}", flush=True))
    res = dict(task=tid, kind=rq["kind"], request=rq, T=int(len(full)), tokens=paths, curve=hist,
               seconds=dict(tm, **{f"fit{s}": x for s, x in secs.items()}, **{f"total{s}": tm["demo"] + tm["bank"] + x for s, x in secs.items()}),
               bytes={r: tk.nbytes() for r, tk in new.items()}, gpu=torch.cuda.get_device_name(0))
    write_json(done, res)
    record_fit(res, tid, rq["name"])


def record_fit(res, tid, tag):
    if SMOKE:
        print(f"[l4] SMOKE {tid} {tag}: seconds {json.dumps(res['seconds'])}", flush=True)
        return
    from lifelong_ood import ledger
    b = int(sum(res["bytes"].values()))
    print(ledger.record("l4_fit", bench="libero_90_c", tag=tag.split("_")[0], seed=SEED, keys=dict(task=tid, name=tag),
                        metrics=dict(seconds=res["seconds"], T=res["T"], final_loss=res["curve"]["loss"][-1] if res["curve"]["loss"] else None),
                        bytes=dict(experts=0, tree=0, codes=b, samples=0, teacher_copies=0, total=b),
                        config=dict(n_role=N_ROLE, steps=STEPS, init_steps=INIT_STEPS, lr=LR, batch=BATCH, ckpt=M1.CKPT),
                        script="experiments/l4_store.py", command=" ".join(sys.argv)), flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def scratch_of(tid):
    f = os.path.join(FITS, slug(tid), "scratch", "fit.json")
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def request_fit(name, tid):
    f = os.path.join(FITS, slug(tid), name, "fit.json")
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


# ── served objects ───────────────────────────────────────────────────────────────────────────────────────────────
def role_mode(tid, text, O, G):
    """A role-token object -> MODE string (lt:<spec>); the spec is written once, named by its content."""
    import lifelong_ood.lang.transplant as LT
    if not O and not G:
        return "ours"
    nm = f"{slug(tid)}_{h(json.dumps(dict(O=list(O), G=list(G), text=text)), 12)}.json"
    p = os.path.join(SPECS, nm)
    if not os.path.exists(p):
        LT.save_spec(p, [], prompt=text, meta=dict(task=tid), role_tokens=dict(O=list(O), G=list(G)))
    return f"lt:{p}"


def roll_out(exp, tid, mode):
    return os.path.join(ROLL, f"{exp}_{slug(tid)}_{h(mode)}.json")


def line(tid, mode, exp, tag, kind):
    """kind: eval (fresh_l4, 20) | prac (fresh_l4p, 5) | damage (stored init states 0-19, seed 0)."""
    if kind == "damage":
        sp = M1._line(tid, mode, exp, tag, False, 0)
    else:
        sp = M1._line(tid, mode, exp, tag, False, SEED)
        sp["FRESH"] = os.path.join(R, EVAL_DIR if kind == "eval" else PRAC_DIR)
        sp["TRIALS"] = N_EVAL if kind == "eval" else N_PRAC
    sp["OUT"] = roll_out(exp, tid, mode)
    return sp


def eps_of(exp, tid, mode, n):
    """{trial: (success, secs)} of a finished rollout (None if missing or short)."""
    if mode == "ours" and exp == "l4_damage":
        return None
    f = roll_out(exp, tid, mode)
    if not os.path.exists(f):
        return None
    z = json.load(open(f))
    out = {int(e["trial"]): (int(bool(e["success"])), float(e.get("secs") or 0.0)) for t in z["tasks"].values() for e in t["episodes"]}
    return out if len(out) == n else None


# ── job submission ───────────────────────────────────────────────────────────────────────────────────────────────
def sbatch(args, dry):
    if dry:
        print("  [dry] sbatch " + " ".join(args))
        return "DRY"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}   # launched from a job: no cpu-bind leak (FM)
    return subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True, env=env).stdout.strip().split(";")[0]


def gpu_job(script, name, dry, hours="01:30:00"):
    return sbatch(["--partition=overcap", "--account=overcap", "--qos=short", "--requeue", f"--time={hours}", "--gres=gpu:a40:1",
                   "--mem=64G", f"--exclude={EXCLUDE}", f"--job-name={name}", f"--export=ALL,SCRIPT={script}",
                   "slurm/jobs/robot_gpu.sbatch"], dry)


def queued_names():
    try:
        r = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True, check=True)
        return set(r.stdout.split())
    except Exception:                                                   # noqa: BLE001
        return set()


def submit_rolls(lines, dry, lanes=10, per=4):
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    os.makedirs(ROLL, exist_ok=True)
    live = set()
    lf = os.path.join(OUT, "packs", "live.txt")
    if os.path.exists(lf):                                              # instances already submitted to packs still queued / running
        q = queued_names()
        for row in open(lf, encoding="utf-8"):
            jn, out = row.rstrip("\n").split("\t")
            if jn in q:
                live.add(out)
    todo, seen = [], set()
    for s in lines:
        if not os.path.exists(s["OUT"]) and s["OUT"] not in seen and s["OUT"] not in live:
            seen.add(s["OUT"])
            todo.append(s)
    last, jobs = [None] * lanes, []
    for j, i in enumerate(range(0, len(todo), per)):
        grp = todo[i:i + per]
        stamp = f"{int(time.time())}_{j:03d}"
        path = os.path.join(OUT, "packs", f"l4_{stamp}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        lane = j % lanes
        jn = f"l4p{stamp}"
        args = ([f"--dependency=afterany:{last[lane]}"] if last[lane] else []) + [
            "--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=04:00:00", "--gres=gpu:a40:1",
            f"--exclude={EXCLUDE}", f"--job-name={jn}", f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        jid = sbatch(args, dry)
        last[lane] = jid
        jobs.append(jid)
        if not dry:
            with open(lf, "a", encoding="utf-8") as f:
                for sp in grp:
                    f.write(f"{jn}\t{sp['OUT']}\n")
        print(f"pack {os.path.basename(path)} [{', '.join(s['TASKS'] + ':' + s['TAG'] for s in grp)}] lane {lane} -> {jid}", flush=True)
    print(f"{len(todo)} instances, {len(jobs)} packs: {':'.join(jobs)}", flush=True)
    return jobs


def launch_scratch(dry):
    q = queued_names()
    for tid in load_plan()["tasks"]:
        nm = f"l4s_{tid.split(':')[1]}"
        if scratch_of(tid) is None and nm not in q:
            print(tid, gpu_job(f"experiments/l4_store.py --fit-scratch --task {tid}", nm, dry), flush=True)


def submit_request(rq, dry):
    """Write the request file and submit its fit (unless done or queued)."""
    p = os.path.join(REQ, f"{rq['name']}__{slug(rq['task'])}.json")
    if request_fit(rq["name"], rq["task"]) is not None:
        return None
    if not os.path.exists(p):
        write_json(p, rq)
    nm = f"l4r_{h(p, 8)}"
    if nm in queued_names():
        return None
    return gpu_job(f"experiments/l4_store.py --fit-request {p}", nm, dry)


# ── the reuse / transfer table ──────────────────────────────────────────────────────────────────────────────────
def cover_from(tid, srcs, plan):
    """Stored role tokens of the source tasks' scratch pairs whose key (all of the source's role items) is contained in tid's
    role items -> ({O: [paths]}, {G: [paths]}, covered items per role)."""
    v = plan["tasks"][tid]
    O, G, cov = [], [], dict(O=set(), G=set())
    for e in srcs:
        se = scratch_of(e)
        if se is None:
            return None
        for r in se["fit_roles"]:
            if set(se["roles"][r]) <= set(v["roles"][r]):
                (O if r == "O" else G).append(se["tokens"][r][str(STEPS)])
                cov[r] |= set(se["roles"][r])
    return O, G, cov


def pair_requests(plan):
    """complete for every headroom pair (and the composite cover of all partners when a target has several), init for the
    canonical partner.  -> list of request dicts (fit-able now)."""
    out = []
    for p in plan["pairs"]:
        if p["headroom"]:
            rq = complete_request(plan, p["dst"], [p["src"]])
            if rq:
                out.append(rq)
    for l, e in plan["canonical"].items():
        se, sl = scratch_of(e), scratch_of(l)
        if se is None or sl is None:
            continue
        v = plan["tasks"][l]
        out.append(dict(task=l, kind="init", name=f"init_{e.split(':')[1]}", frozen={}, new={r: v["roles"][r] for r in v["fit_roles"]},
                        init={r: se["tokens"][r][str(STEPS)] if r in se["tokens"] else None for r in v["fit_roles"]}))
    return out


def complete_request(plan, tid, srcs, cover=None):
    """Freeze the cover; fit a new token for each fit-able role with uncovered items.  None if nothing is uncovered."""
    v = plan["tasks"][tid]
    c = cover if cover is not None else cover_from(tid, srcs, plan)
    if c is None:
        return None
    O, G, cov = c
    new = {r: sorted(set(v["roles"][r]) - set(cov[r])) for r in v["fit_roles"] if set(v["roles"][r]) - set(cov[r])}
    if not new or not (O or G):
        return None
    key = json.dumps(dict(O=O, G=G))
    return dict(task=tid, kind="complete", name=f"complete_{h(key, 8)}", frozen=dict(O=O, G=G), new=new, init={},
                srcs=list(srcs or []))


def table_objects(plan):
    """-> list of (tid, arm, mode, kinds) for the reuse / transfer table (eval and, where the stream may use it, practice)."""
    objs = []
    for tid, v in plan["tasks"].items():
        s = slug(tid)
        objs.append((tid, "zs", "ours", ("eval", "prac")))
        objs.append((tid, "whole", f"lt:{os.path.join(M1.FITS, s, 'create.json')}", ("eval",)))
        objs.append((tid, "lora1000", f"lora:{os.path.join(M1.FITS, s, 'lora1000.pt')}", ("eval",)))
        sc = scratch_of(tid)
        if sc is None:
            continue
        tok = lambda st: ([sc["tokens"]["O"][str(st)]] if "O" in sc["tokens"] else [], [sc["tokens"]["G"][str(st)]] if "G" in sc["tokens"] else [])  # noqa: E731
        objs.append((tid, "scratch", role_mode(tid, v["text"], *tok(STEPS)), ("eval", "prac")))
        if tid in plan["canonical"]:
            for st in SNAPS:
                objs.append((tid, f"scratch{st}", role_mode(tid, v["text"], *tok(st)), ("eval",)))
    for p in plan["pairs"]:
        c = cover_from(p["dst"], [p["src"]], plan)
        if c is None:
            continue
        kinds = ("eval", "prac") if p["headroom"] else ("eval",)
        objs.append((p["dst"], f"reuse@{p['src']}", role_mode(p["dst"], plan["tasks"][p["dst"]]["text"], c[0], c[1]), kinds))
        if p["headroom"]:
            rq = complete_request(plan, p["dst"], [p["src"]])
            ft = request_fit(rq["name"], p["dst"]) if rq else None
            if ft is not None:
                nO = [ft["tokens"]["O"][str(STEPS)]] if "O" in ft["tokens"] else []
                nG = [ft["tokens"]["G"][str(STEPS)]] if "G" in ft["tokens"] else []
                objs.append((p["dst"], f"complete@{p['src']}", role_mode(p["dst"], plan["tasks"][p["dst"]]["text"], c[0] + nO, c[1] + nG),
                             ("eval", "prac")))
                if plan["canonical"].get(p["dst"]) == p["src"]:
                    objs.append((p["dst"], f"causal@{p['src']}", role_mode(p["dst"], plan["tasks"][p["dst"]]["text"], nO, nG), ("eval",)))
    for l, e in plan["canonical"].items():
        ft = request_fit(f"init_{e.split(':')[1]}", l)
        if ft is None:
            continue
        for st in INIT_SNAPS + (INIT_STEPS,):
            O = [ft["tokens"]["O"][str(st)]] if "O" in ft["tokens"] else []
            G = [ft["tokens"]["G"][str(st)]] if "G" in ft["tokens"] else []
            objs.append((l, f"init{st}@{e}", role_mode(l, plan["tasks"][l]["text"], O, G), ("eval",)))
    for c in plan["cross"]:
        sc = scratch_of(c["src"])
        if sc is None:
            continue
        O = [sc["tokens"]["O"][str(STEPS)]] if "O" in sc["tokens"] else []
        G = [sc["tokens"]["G"][str(STEPS)]] if "G" in sc["tokens"] else []
        objs.append((c["dst"], f"cross@{c['src']}", role_mode(c["dst"], plan["tasks"][c["dst"]]["text"], O, G), ("eval",)))
    return objs


def launch_pairs(dry):
    plan = load_plan()
    for rq in pair_requests(plan):
        j = submit_request(rq, dry)
        if j:
            print(rq["task"], rq["name"], j, flush=True)


def launch_rolls(dry):
    plan = load_plan()
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    lines = []
    for tid, arm, mode, kinds in table_objects(plan):
        for k in kinds:
            lines.append(line(tid, mode, f"l4_{k}", arm, k))
    submit_rolls(lines, dry)


# ── the continual stream ─────────────────────────────────────────────────────────────────────────────────────────
class Store:
    """Part entries {key: dict(path, owner, bytes)} with key 'scene||role||items' (items joined by ';'), and task-level
    exceptions {scene||text: dict(O, G)} (the guard's choice where routing by parts would serve something else)."""

    def __init__(self):
        self.parts, self.exc, self.extra = {}, {}, {}                   # extra: tokens held only by an exception {path: bytes}

    @staticmethod
    def key(scene, role, items):
        return f"{scene}||{role}||{';'.join(sorted(items))}"

    def route(self, scene, text, rl, use_exc=True):
        """-> (O paths, G paths, covered items per role): the exception if any, else every part of this scene whose items are
        contained in the request's role items."""
        e = self.exc.get(f"{scene}||{text}")
        if use_exc and e is not None:
            return list(e["O"]), list(e["G"]), None
        O, G, cov = [], [], dict(O=set(), G=set())
        for k, x in sorted(self.parts.items()):
            sc, r, it = k.split("||")
            its = set(it.split(";"))
            if sc == scene and its <= set(rl[r]):
                (O if r == "O" else G).append(x["path"])
                cov[r] |= its
        return O, G, cov

    def bytes(self):
        """Token bytes (parts and exception-only tokens) + the exceptions' pointer lists (their paths, as json)."""
        return int(sum(x["bytes"] for x in self.parts.values()) + sum(self.extra.values())
                   + sum(len(json.dumps(e)) for e in self.exc.values()))


def stream(plan, order, needs):
    """One pass of the arrival rule over `order`.  Missing practice / fits / evals are appended to `needs` and the pass stops at
    the first arrival that cannot be decided.  -> (log, store, complete?).

    Arrival t (instruction + one demo):
      1. route by parts (no exception yet) -> cover; practice zs and (if any) the cover, 5 episodes each.
      2. the best (most practice successes; ties: zs, then the cover) passes if >= TAU of 5: serve it, fit nothing.
      3. else fit: if the cover is non-empty, at least as good as zs on practice, and leaves fit-able items uncovered: COMPLETE
         (cover frozen, new tokens for the uncovered items); otherwise the SCRATCH pair.  Practice it.
      4. serve the best of {zs, cover, fitted} (ties in that order).  If the fitted object wins, its new tokens are stored as
         parts (keys: the items each covers); a key already held stays with its owner and the new token becomes part of t's
         exception.  If routing t by parts would now serve something else than the choice, t's choice is stored as an exception.
    """
    st = Store()
    log = []
    for t in order:
        v = plan["tasks"][t]
        rl, sc, text = v["roles"], v["scene"], v["text"]
        O, G, cov = st.route(sc, text, rl, use_exc=False)
        zmode = "ours"
        z = eps_of("l4_prac", t, zmode, N_PRAC)
        if z is None:
            needs.append(("prac", t, zmode, "zs"))
            return log, st, False
        cands = [("zs", zmode, [], [], sum(x[0] for x in z.values()), z)]
        if O or G:
            cm = role_mode(t, text, O, G)
            c = eps_of("l4_prac", t, cm, N_PRAC)
            if c is None:
                needs.append(("prac", t, cm, "cover"))
                return log, st, False
            cands.append(("reuse", cm, O, G, sum(x[0] for x in c.values()), c))
        best = max(cands, key=lambda x: (x[4], -["zs", "reuse"].index(x[0])))
        prac_secs = sum(sum(x[1] for x in c_[5].values()) for c_ in cands)
        entry = dict(task=t, cover=dict(O=O, G=G), practice={c_[0]: c_[4] for c_ in cands}, fit=None, fit_secs=0.0, new_bytes=0)
        if best[4] >= TAU:
            choice = best
        else:
            reuse = next((c_ for c_ in cands if c_[0] == "reuse"), None)
            rq = None
            if reuse is not None and reuse[4] >= cands[0][4]:
                rq = complete_request(plan, t, None, cover=(O, G, cov))
            if rq is not None:
                ft = request_fit(rq["name"], t)
                if ft is None:
                    needs.append(("fit", t, rq, "complete"))
                    return log, st, False
                nO = [ft["tokens"]["O"][str(STEPS)]] if "O" in ft["tokens"] else []
                nG = [ft["tokens"]["G"][str(STEPS)]] if "G" in ft["tokens"] else []
                fO, fG = O + nO, G + nG
                new = {r: (ft["tokens"][r][str(STEPS)], rq["new"][r], ft["bytes"][r]) for r in ft["tokens"]}
                kind, fsecs = "complete", ft["seconds"][f"total{STEPS}"]
            else:
                sc_ = scratch_of(t)
                if sc_ is None:
                    needs.append(("scratch", t, None, "scratch"))
                    return log, st, False
                fO = [sc_["tokens"]["O"][str(STEPS)]] if "O" in sc_["tokens"] else []
                fG = [sc_["tokens"]["G"][str(STEPS)]] if "G" in sc_["tokens"] else []
                new = {r: (sc_["tokens"][r][str(STEPS)], sc_["roles"][r], sc_["bytes"][r]) for r in sc_["tokens"]}
                kind, fsecs = "scratch", sc_["seconds"][f"total{STEPS}"]
            fm = role_mode(t, text, fO, fG)
            f = eps_of("l4_prac", t, fm, N_PRAC)
            if f is None:
                needs.append(("prac", t, fm, kind))
                return log, st, False
            prac_secs += sum(x[1] for x in f.values())
            cands.append((kind, fm, fO, fG, sum(x[0] for x in f.values()), f))
            entry.update(fit=kind, fit_secs=float(fsecs))
            entry["practice"][kind] = cands[-1][4]
            choice = max(cands, key=lambda x: (x[4], -["zs", "reuse", kind].index(x[0])))
            if choice[0] == kind:
                for r, (p, items, b) in new.items():
                    k = Store.key(sc, r, items)
                    if k in st.parts:                                    # held by its owner: this token lives in t's exception
                        st.extra[p] = int(b)
                    else:
                        st.parts[k] = dict(path=p, owner=t, bytes=int(b))
                    entry["new_bytes"] += int(b)
        rO, rG, _ = st.route(sc, text, rl, use_exc=False)
        if (sorted(rO), sorted(rG)) != (sorted(choice[2]), sorted(choice[3])):
            st.exc[f"{sc}||{text}"] = dict(O=list(choice[2]), G=list(choice[3]))
            entry["exception"] = True
        entry.update(choice=choice[0], mode=choice[1], served=dict(O=choice[2], G=choice[3]), prac_secs=float(prac_secs),
                     store_bytes=st.bytes(), parts=len(st.parts), exceptions=len(st.exc))
        log.append(entry)
    return log, st, True


def stream_all(plan):
    needs, res = [], {}
    for k, o in plan["orders"].items():
        log, st, ok = stream(plan, o, needs)
        res[k] = dict(log=log, complete=ok, parts=st.parts, exc=st.exc, store=st)
    return res, needs


def end_routing(plan, res):
    """Per order: every dev task and damage cell routed by the FINAL store -> mode (dev tasks: compared with the arrival choice)."""
    out = {}
    for k, r in res.items():
        st = r["store"]
        dev = {}
        for e in r["log"]:
            v = plan["tasks"][e["task"]]
            O, G, _ = st.route(v["scene"], v["text"], v["roles"])
            m = role_mode(e["task"], v["text"], O, G)
            dev[e["task"]] = dict(mode=m, arrival=e["mode"], changed=m != e["mode"])
        cells = {}
        for c in plan["cells"]:
            O, G, _ = st.route(c["scene"], c["text"], c["roles"])
            cells[c["cell"]] = dict(mode=role_mode(c["cell"], c["text"], O, G), O=O, G=G)
        out[k] = dict(dev=dev, cells=cells)
    return out


def launch_stream(dry):
    plan = load_plan()
    res, needs = stream_all(plan)
    lines, done = [], set()
    for kind, t, x, tag in needs:
        if kind == "prac":
            lines.append(line(t, x, "l4_prac", f"stream_{tag}", "prac"))
            if x != "ours":
                lines.append(line(t, x, "l4_eval", f"stream_{tag}", "eval"))     # its evaluation, for the read
        elif kind == "fit":
            j = submit_request(x, dry)
            print("fit", t, x["name"], j, flush=True)
        elif kind == "scratch":
            print("scratch missing", t, flush=True)
    for k, r in res.items():
        print(f"order {k}: {len(r['log'])} arrivals decided, complete {r['complete']}", flush=True)
        for e in r["log"]:                                                  # every served choice gets its evaluation
            if e["mode"] != "ours":
                lines.append(line(e["task"], e["mode"], "l4_eval", f"stream_{e['choice']}", "eval"))
    if all(r["complete"] for r in res.values()):
        er = end_routing(plan, res)
        for k, x in er.items():
            for t, d in x["dev"].items():
                if d["changed"] and d["mode"] != "ours":
                    lines.append(line(t, d["mode"], "l4_eval", "end_routed", "eval"))
            for c, d in x["cells"].items():
                if d["mode"] != "ours":
                    lines.append(line(c, d["mode"], "l4_damage", "cell_routed", "damage"))
        print("stream complete; end routing:", json.dumps({k: dict(dev_changed=[t for t, d in x["dev"].items() if d["changed"]],
                                                                    cells=[c for c, d in x["cells"].items() if d["mode"] != "ours"])
                                                           for k, x in er.items()}), flush=True)
    submit_rolls(lines, dry)
    print("needs:", [(a, b, c if isinstance(c, str) else c["name"], d) for a, b, c, d in needs], flush=True)


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────
def read(record):
    from lifelong_ood import stats as ST
    plan = load_plan()
    objs = table_objects(plan)
    tab, miss = {}, []
    for tid, arm, mode, kinds in objs:
        e = eps_of("l4_eval", tid, mode, N_EVAL)
        if e is None:
            miss.append(f"{tid}:{arm}")
            continue
        tab.setdefault(tid, {})[arm] = {k: s for k, (s, _) in e.items()}
    res = dict(missing=miss)
    T = list(plan["tasks"])
    tot = lambda arm, ts: int(sum(sum(tab[t][arm].values()) for t in ts if arm in tab.get(t, {})))   # noqa: E731
    res["totals"] = {a: tot(a, T) for a in ("zs", "whole", "lora1000", "scratch")}
    kinds = {t: ("combination" if plan["tasks"][t]["bin"] != "part" else "part") for t in T}
    res["by_kind"] = {k: {a: tot(a, [t for t in T if kinds[t] == k]) for a in ("zs", "whole", "lora1000", "scratch")}
                      for k in ("combination", "part")}

    def cmp(table, a, b, ts, margin=None):
        pairs = ST.paired(table, a, b, ts)
        d = ST.bootstrap_diff(pairs, B=10000, return_samples=True)
        s = d.pop("samples", None)
        bb, cc = ST.discordant(pairs)
        x = int(sum(p[0].sum() for p in pairs.values()))
        y = int(sum(p[1].sum() for p in pairs.values()))
        out = dict(x=x, y=y, n=int(sum(len(p[0]) for p in pairs.values())), tasks=len(pairs), diff=d, discordant=[bb, cc],
                   sign_p=float(ST.sign_test(bb, cc)))
        out["p_dir"] = out["sign_p"] if x > y else 1.0
        if margin is not None:
            out["p_noninferior"] = float(np.mean(np.asarray(s) <= -margin))
        return out
    # H1: role pair (scratch) non-inferior to the whole-task token
    h1 = cmp(tab, "scratch", "whole", T, margin=0.05)
    # canonical-pair tables (13 headroom targets)
    can = plan["canonical"]
    ct = {}
    for l, e in can.items():
        row = tab.get(l, {})
        m = {"zs": "zs", "scratch": "scratch", "scratch50": "scratch50", "scratch200": "scratch200", "reuse": f"reuse@{e}",
             "complete": f"complete@{e}", "causal": f"causal@{e}", "init50": f"init50@{e}", "init200": f"init200@{e}"}
        ct[l] = {k: row[a] for k, a in m.items() if a in row}
    L = list(can)
    h2 = cmp(ct, "reuse", "zs", L)
    h3 = cmp(ct, "complete", "causal", L)                                # H3: complete > causal (the reused tokens removed)
    h4 = cmp(ct, "init200", "scratch200", L)
    res["canonical_totals"] = {a: int(sum(sum(ct[l][a].values()) for l in L if a in ct[l])) for a in
                               ("zs", "reuse", "complete", "causal", "scratch", "scratch50", "scratch200", "init50", "init200")}
    res["complete_vs_scratch"] = cmp(ct, "complete", "scratch", L, margin=0.05)
    thr = res["canonical_totals"]["scratch"]
    res["budget_to_threshold"] = dict(threshold=thr,
                                      scratch=next((s for s, a in ((50, "scratch50"), (200, "scratch200"), (1000, "scratch"))
                                                    if res["canonical_totals"][a] >= thr), None),
                                      init=next((s for s, a in ((50, "init50"), (200, "init200")) if res["canonical_totals"][a] >= thr), None))
    # secondary: every pair's zero-shot reuse vs zs, the cross-scene same-text reuse
    sec = {}
    for p in plan["pairs"]:
        a = f"reuse@{p['src']}"
        if a in tab.get(p["dst"], {}):
            sec[f"{p['src']}>{p['dst']}"] = dict(reuse=int(sum(tab[p["dst"]][a].values())), zs=int(sum(tab[p["dst"]]["zs"].values())),
                                                 scratch=int(sum(tab[p["dst"]]["scratch"].values())) if "scratch" in tab[p["dst"]] else None,
                                                 complete=int(sum(tab[p["dst"]][f"complete@{p['src']}"].values()))
                                                 if f"complete@{p['src']}" in tab[p["dst"]] else None, headroom=p["headroom"])
    res["pairs"] = sec
    res["cross"] = {f"{c['src']}>{c['dst']}": dict(cross=int(sum(tab[c["dst"]][f"cross@{c['src']}"].values())),
                                                   zs=int(sum(tab[c["dst"]]["zs"].values())), scratch=int(sum(tab[c["dst"]]["scratch"].values())))
                    for c in plan["cross"] if f"cross@{c['src']}" in tab.get(c["dst"], {})}
    # the stream
    sres, needs = stream_all(plan)
    res["stream_needs"] = len(needs)
    er = end_routing(plan, sres) if all(r["complete"] for r in sres.values()) else None
    m1f = {t: M1.fit_of(t) for t in T}
    lora_secs = {t: m1f[t]["seconds"]["lora1000"] for t in T}
    lora_bytes = int(np.mean([m1f[t]["bytes"]["lora"] for t in T]))
    whole_bytes = int(np.mean([m1f[t]["bytes"]["token"] for t in T]))
    S = {}
    for k, r in sres.items():
        succ, gpu, gpu_fit = {}, {}, {}
        for e in r["log"]:
            ev = eps_of("l4_eval", e["task"], e["mode"], N_EVAL) if e["mode"] != "ours" else \
                {kk: (vv, 0) for kk, vv in tab[e["task"]]["zs"].items()}
            succ[e["task"]] = None if ev is None else {kk: s for kk, (s, _) in ev.items()}
            gpu[e["task"]] = e["prac_secs"] + e["fit_secs"]
            gpu_fit[e["task"]] = e["fit_secs"]
        S[k] = dict(log=[{kk: vv for kk, vv in e.items()} for e in r["log"]], succ=succ, gpu=gpu, gpu_fit=gpu_fit,
                    complete=r["complete"], parts=len(r["parts"]), exceptions=len(r["exc"]), store_bytes=r["store"].bytes(),
                    choices={c: sum(e["choice"] == c for e in r["log"]) for c in ("zs", "reuse", "complete", "scratch")},
                    fits={c: sum(e["fit"] == c for e in r["log"]) for c in ("complete", "scratch")})
        ok = [t for t in T if succ.get(t) is not None]
        S[k]["success"] = int(sum(sum(succ[t].values()) for t in ok))
        S[k]["tasks_read"] = len(ok)
        S[k]["gpu_s"] = dict(mean_total=float(np.mean(list(gpu.values()))), mean_fit=float(np.mean(list(gpu_fit.values()))),
                             mean_lora=float(np.mean(list(lora_secs.values()))))
        S[k]["bytes"] = dict(store=r["store"].bytes(), lora_per_task_total=lora_bytes * len(T), whole_per_task_total=whole_bytes * len(T),
                             ratio_vs_lora=r["store"].bytes() / (lora_bytes * len(T)))
        S[k]["curve"] = [(e["task"], e["store_bytes"], e["parts"]) for e in r["log"]]
    # FWT: late vs early arrival
    ft = {t: x for t, x in plan["fwt"].items()}
    fwt_tab = {}
    for t, x in ft.items():
        la, ea = x["late"], "B" if x["late"] == "A" else "A"
        if S.get(la, {}).get("succ", {}).get(t) is not None and S.get(ea, {}).get("succ", {}).get(t) is not None:
            fwt_tab[t] = dict(late=S[la]["succ"][t], early=S[ea]["succ"][t])
    h5 = cmp(fwt_tab, "late", "early", list(fwt_tab)) if fwt_tab else dict(p_dir=1.0)
    # damage: earlier dev tasks re-routed by the final store; the 10 training cells routed by it
    dmg = {}
    if er is not None:
        base = M1.parity_base()
        for k, x in er.items():
            dev_ch, cell_ch = {}, {}
            for t, d in x["dev"].items():
                if not d["changed"]:
                    dev_ch[t] = 0.0
                    continue
                a = S[k]["succ"].get(t)
                b = eps_of("l4_eval", t, d["mode"], N_EVAL) if d["mode"] != "ours" else {kk: (vv, 0) for kk, vv in tab[t]["zs"].items()}
                dev_ch[t] = None if a is None or b is None else float(np.mean([s for s, _ in b.values()]) - np.mean(list(a.values())))
            for c, d in x["cells"].items():
                if d["mode"] == "ours":
                    cell_ch[c] = 0.0
                    continue
                b = eps_of("l4_damage", c, d["mode"], N_EVAL)
                cell_ch[c] = None if b is None else float(np.mean([s for s, _ in b.values()]) - np.mean(list(base[c].values())))
            dmg[k] = dict(dev=dev_ch, cells=cell_ch,
                          dev_mean=float(np.mean([v_ for v_ in dev_ch.values() if v_ is not None])) if dev_ch else None,
                          cells_mean=float(np.mean([v_ for v_ in cell_ch.values() if v_ is not None])) if cell_ch else None,
                          missing=[t for t, v_ in {**dev_ch, **cell_ch}.items() if v_ is None])
    res["damage"] = dmg
    dmg_ok = bool(dmg) and all(d["dev_mean"] is not None and d["dev_mean"] >= -0.01 and d["cells_mean"] >= -0.01 and not d["missing"]
                               for d in dmg.values())
    bytes_ok = bool(S) and all(S[k]["bytes"]["ratio_vs_lora"] <= 0.01 for k in S)
    hm = M1.holm(dict(H1=h1["p_noninferior"], H2=h2["p_dir"], H3=h3["p_dir"], H4=h4["p_dir"], H5=h5["p_dir"]))
    h1["passes"] = hm["H1"]["reject"]
    h2["passes"] = hm["H2"]["reject"]
    h3["passes"] = hm["H3"]["reject"]
    h4["passes"] = hm["H4"]["reject"]
    h5.update(passes=bool(hm["H5"]["reject"] and dmg_ok and bytes_ok), damage_ok=dmg_ok, bytes_ok=bytes_ok)
    res.update(H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, holm=hm, stream={k: {kk: vv for kk, vv in x.items() if kk != "log"} for k, x in S.items()},
               stream_log={k: x["log"] for k, x in S.items()}, fwt=fwt_tab and {t: dict(late=int(sum(x["late"].values())),
                                                                                        early=int(sum(x["early"].values())))
                                                                               for t, x in fwt_tab.items()},
               per_task={t: {a: int(sum(x.values())) for a, x in row.items()} for t, row in tab.items()},
               bytes=dict(role=N_ROLE * 18 * 256 * 2 * 2, whole=whole_bytes, lora=lora_bytes), lora_secs=lora_secs)
    with open(os.path.join(OUT, "read.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("missing", "totals", "by_kind", "canonical_totals", "budget_to_threshold", "holm")}, indent=1))
    for hh in ("H1", "H2", "H3", "H4", "H5"):
        x = res[hh]
        if "x" in x:
            print(f"[l4] {hh} {x['x']} vs {x['y']} (n {x['n']}, tasks {x['tasks']}), discordant {x['discordant']}, sign p {x['sign_p']:.3g}, "
                  f"diff {x['diff']['diff']:+.3f} [{x['diff']['lo']:+.3f}, {x['diff']['hi']:+.3f}] p_ni {x.get('p_noninferior')} passes {x['passes']}")
    print("[l4] stream", json.dumps(res["stream"], indent=1, default=str)[:5000])
    print("[l4] damage", json.dumps(dmg, indent=1)[:3000])
    print("[l4] fwt", json.dumps(res["fwt"]))
    print("[l4] pairs", json.dumps(res["pairs"]))
    print("[l4] cross", json.dumps(res["cross"]))
    print("[l4] per task", json.dumps(res["per_task"]))
    if record:
        from lifelong_ood import ledger
        print(ledger.record("l4_read", bench="libero_90_c", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(plan=plan["settings"], prereg="m3_prereg tag pl4_store", ckpt=M1.CKPT),
                            script="experiments/l4_store.py", command=" ".join(sys.argv)))


def smoke_serve(task, dry):
    """Serving smoke (excluded from every number): the smoke scratch pair, 2 episodes from the task's STORED init states."""
    assert SMOKE
    plan = load_plan()
    sc = scratch_of(task)
    O = [sc["tokens"]["O"][str(STEPS)]] if "O" in sc["tokens"] else []
    G = [sc["tokens"]["G"][str(STEPS)]] if "G" in sc["tokens"] else []
    m = role_mode(task, plan["tasks"][task]["text"], O, G)
    sp = M1._line(task, m, "l4_smoke", "scratch", False, SEED)
    sp.update(TRIALS=2, OUT=os.path.join(OUT, "smoke", f"roll_{slug(task)}.json"))
    submit_rolls([sp], dry, lanes=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("plan", "fresh-check", "fit-scratch", "launch-scratch", "launch-pairs", "launch-rolls", "launch-stream", "read", "record",
              "dry", "smoke-serve"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--task", default=None)
    ap.add_argument("--fit-request", default=None)
    a = ap.parse_args()
    if a.plan:
        make_plan()
    if a.fresh_check:
        fresh_check()
    if a.fit_scratch or a.fit_request:
        bk_ = M1.backbone()
        if a.fit_scratch:
            fit_scratch(bk_, a.task)
        else:
            fit_request(bk_, a.fit_request)
    if a.launch_scratch:
        launch_scratch(a.dry)
    if a.launch_pairs:
        launch_pairs(a.dry)
    if a.launch_rolls:
        launch_rolls(a.dry)
    if a.launch_stream:
        launch_stream(a.dry)
    if a.smoke_serve:
        smoke_serve(a.task, a.dry)
    if a.read:
        read(a.record)
