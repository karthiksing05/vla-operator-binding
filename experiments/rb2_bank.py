"""PLAN R, probe R2 (docs/PLAN_R2.md; pre-registration experiments/prereg/pr2_bank.py, ledger m3_prereg tag pr2_bank):
A BANK OF ROLE TOKENS FROM TRAINING DEMOS ONLY, served zero-shot under ORACLE retrieval on LIBERO-90-C's 15 recombination dev
tasks, pi05_l90c main/5000 frozen.  No target-task (dev) demo is read anywhere in this file.

Bank unit = a ROLE TOKEN (policy/token_fold.KVToken, n 2 entries per layer, 36,864 bytes fp16; L4's unit), one per PART KEY:
  O|<object content words>                  the object role (verb dropped: what the approach goes to)
  G|<prep class>|<goal content words>       the goal role (on / on top of / onto -> on; in / into / inside -> in)
  G|manip:<verb>|<object content words>     an articulation step's goal role (L4's item)
Each token is fitted by flow matching through the frozen VLA on the role-phase frames (held_ops._Grasp rule offline, L4.phases) of
training demos from EVERY training text of pi05_l90c's own training data (lol/libero_l90c) whose parse holds the key, pooled
across scenes and demos (a fixed subset: 40 demos spread over the source texts, <= 1024 frames; docs/PLAN_R2.md 2).

Serving (lt:<spec> with role_tokens, lifelong_ood/lang/transplant.py): a request's O tokens are appended while the policy's own
gripper is open, its G tokens while it carries.  Oracle retrieval = the tokens whose keys the request's parse holds.

    sbatch ... cpu.sbatch SCRIPT="experiments/rb2_bank.py --plan"            design table -> results/rb2/plan.json
    ... --fresh-check                                                        (chunks env) fresh_rb2 is new
    GPU: --fit --key 'O|black bowl'                                          one bank token (+ junk check)
    ... --launch-fits [--dry] | --launch-base [--dry] | --launch-gate [--dry] | --launch-arms [--dry]
    LIBERO env: --type                                                       type every dev episode (fm_types)
    ... --read [--record]
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1                                          # noqa: E402
import l4_store as L4                                                   # noqa: E402

R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "rb2")
SMOKE = os.environ.get("RB2_SMOKE") == "1"
FITS = os.path.join(OUT, "smoke" if SMOKE else "fits")
SPECS = os.path.join(OUT, "specs")
ROLL = os.path.join(OUT, "roll")
FRESH_DIR, BASE, N_STATES = "fresh_rb2", 7500000, 20
SEED = 13
EXCLUDE = M1.EXCLUDE
N_ROLE = 2
STEPS = 20 if SMOKE else 1000
D_TOT, D_MIN, HOLD_PER, HOLD_MAX = 40, 3, 2, 8                         # fit demos per token; held-out demos per source / per token
F_MAX, F_HOLD = 1024, 24                                               # frames per token (fit); per held-out demo (junk check)
MIN_FRAMES = 50                                                         # a key is fitted only if its pooled phase holds >= 50 frames
K_DRAWS = 2                                                             # junk check: fixed (t, eps) draws per frame
GATE_TOL = 0.15                                                         # gate: cell success >= parity - 0.15 (3 of 20)
N_DAMAGE = 8
PREP_CLASS = {"on": "on", "on top of": "on", "onto": "on", "in": "in", "into": "in", "inside": "in"}
LOGDIR = os.path.join(ROOT, "slurm", "rb2")


def h(s, n=10):
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def slug(t):
    return t.replace(":", "_")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(path + ".tmp", path)


def load_json(p):
    return json.load(open(p, encoding="utf-8"))


def keys_of(text):
    """-> {"O": sorted keys, "G": sorted keys} (see the module doc)."""
    rl = L4.roles(text)
    O, G = set(), set()
    for it in rl["O"]:
        w = it.split("|")[2]
        if w:
            O.add(f"O|{w}")
    for it in rl["G"]:
        _, prep, w = it.split("|")
        if not w:
            continue
        G.add(f"G|{prep}|{w}" if prep.startswith("manip:") else f"G|{PREP_CLASS.get(prep, prep)}|{w}")
    return dict(O=sorted(O), G=sorted(G))


def recomb_tasks():
    return [(t, b) for t, b in M1.dev_tasks() if b != "part"]


# ── the design table ─────────────────────────────────────────────────────────────────────────────────────────────
def make_plan():
    """Everything from parses, training-demo gripper commands and the parity file's coverage: no dev demo, no outcome."""
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    dm = D.Demos(root=M1.L90C_DATA)
    tt = M1.task_table()
    split = l90c.load_split()
    base = M1.parity_base()
    tids_of = {}
    for tid in split["train"]:
        tids_of.setdefault(tt[tid].language, []).append(tid)
    train = {}
    for ti, text in sorted(dm.tasks.items()):
        train[str(ti)] = dict(text=text, keys=keys_of(text), episodes=[int(e) for e in dm.episodes(ti)],
                              tids=sorted(tids_of.get(text, [])))
    dev = {}
    for tid, b in recomb_tasks():
        t = tt[tid]
        dev[tid] = dict(text=t.language, scene=t.scene, bin=b, roles=L4.roles(t.language), keys=keys_of(t.language))
    need = sorted({(r, k) for v in dev.values() for r in ("O", "G") for k in v["keys"][r]})
    phase_cache = {}

    def phase_frames(ep, role):
        if ep not in phase_cache:
            phase_cache[ep] = L4.phases(dm.load(ep)["actions"])
        c = phase_cache[ep]
        return np.nonzero(c if role == "G" else ~c)[0]
    bank = {}
    for r, k in need:
        src = [ti for ti, v in train.items() if k in v["keys"][r]]
        ent = dict(role=r, sources=src, fit_demos=[], hold_demos=[], fit_frames=[], hold_frames=[], fitted=False)
        if src:
            n_s = max(D_MIN, int(np.ceil(D_TOT / len(src))))
            pairs = []
            for ti in src:
                eps = train[ti]["episodes"]
                pool = eps[:-HOLD_PER]
                pick = sorted({pool[int(round(x))] for x in np.linspace(0, len(pool) - 1, min(n_s, len(pool)))})
                ent["fit_demos"] += pick
                if len(ent["hold_demos"]) < HOLD_MAX:
                    ent["hold_demos"] += eps[-HOLD_PER:][: HOLD_MAX - len(ent["hold_demos"])]
                for e in pick:
                    pairs += [(int(e), int(f)) for f in phase_frames(e, r)]
            P = len(pairs)
            sel = sorted({int(round(x)) for x in np.linspace(0, P - 1, min(P, F_MAX))}) if P else []
            ent["fit_frames"] = [pairs[i] for i in sel]
            ent["phase_frames_total"] = P
            for e in ent["hold_demos"]:
                fr = phase_frames(e, r)
                if len(fr):
                    ent["hold_frames"] += [(int(e), int(fr[int(round(x))])) for x in
                                           sorted({int(round(y)) for y in np.linspace(0, len(fr) - 1, min(len(fr), F_HOLD))})]
            ent["fitted"] = len(ent["fit_frames"]) >= MIN_FRAMES
        bank[k] = ent
    # oracle retrieval per dev task (keys with a fitted token)
    for tid, v in dev.items():
        v["oracle"] = {r: [k for k in v["keys"][r] if bank[k]["fitted"]] for r in ("O", "G")}
        v["uncovered"] = {r: [k for k in v["keys"][r] if not bank[k]["fitted"]] for r in ("O", "G")}
    fitted = [k for k, e in bank.items() if e["fitted"]]
    # training cells (benchmark ids) with a 20-trial parity baseline, and the bank keys their parse holds
    cells = {}
    for ti, v in train.items():
        for tid in v["tids"]:
            if tid in dev or len(base.get(tid, {})) < 20:
                continue
            ks = [k for r in ("O", "G") for k in v["keys"][r] if k in fitted]
            if ks:
                cells[tid] = dict(text=v["text"], l90c=ti, keys=ks, scene=tt[tid].scene, parity=int(sum(base[tid][i] for i in range(20))))
    # the gate: greedy set cover of the fitted keys by training cells (most uncovered keys, ties lowest sha1)
    U, gate = set(fitted), {}
    while U:
        cand = [c for c in cells if set(cells[c]["keys"]) & U]
        if not cand:
            break
        c = min(cand, key=lambda c: (-len(set(cells[c]["keys"]) & U), h("rb2gate:" + c, 40)))
        gate[c] = sorted(set(cells[c]["keys"]) & U)
        U -= set(gate[c])
    ungated = sorted(U)
    rest = sorted((c for c in cells if c not in gate), key=lambda c: h("rb2dmg:" + c, 40))
    damage = rest[:N_DAMAGE]
    res = dict(dev=dev, bank=bank, fitted=fitted, train=train, cells=cells, gate=gate, ungated=ungated, damage=damage,
               settings=dict(n_role=N_ROLE, steps=STEPS, lr=L4.LR, batch=L4.BATCH, seed=SEED, d_tot=D_TOT, d_min=D_MIN,
                             hold_per=HOLD_PER, hold_max=HOLD_MAX, f_max=F_MAX, f_hold=F_HOLD, min_frames=MIN_FRAMES,
                             k_draws=K_DRAWS, gate_tol=GATE_TOL, n_damage=N_DAMAGE, prep_class=PREP_CLASS,
                             fresh=dict(dir=FRESH_DIR, base=BASE, n=N_STATES), ckpt=M1.CKPT, data=M1.L90C_DATA))
    if not SMOKE:
        write_json(os.path.join(OUT, "plan.json"), res)
    for tid, v in dev.items():
        print(f"{tid:13} {v['bin']} {v['scene']:18} '{v['text']}'  oracle {v['oracle']}  uncovered {v['uncovered']}", flush=True)
    for k, e in bank.items():
        print(f"  {k:36} sources {len(e['sources']):2d} fit demos {len(e['fit_demos']):2d} frames {len(e['fit_frames']):4d} "
              f"(phase total {e.get('phase_frames_total', 0)}) hold demos {len(e['hold_demos'])} frames {len(e['hold_frames'])} "
              f"fitted {e['fitted']}  texts {[train[s]['text'] for s in e['sources']][:3]}", flush=True)
    print("gate:", json.dumps({c: dict(keys=g, text=cells[c]["text"], parity=cells[c]["parity"]) for c, g in gate.items()}, indent=1))
    print("ungated:", ungated)
    print("damage:", json.dumps({c: dict(keys=cells[c]["keys"], text=cells[c]["text"], parity=cells[c]["parity"]) for c in damage}, indent=1))


def load_plan():
    return load_json(os.path.join(OUT, "plan.json"))


def fresh_check():
    """fresh_rb2: 20 states per recombination task, each > 1e-6 from every state of every other libero_90 fresh set."""
    used = tuple(M1.USED_BASES) + (5700000, 6100000, 6300000, 6700000, 7100000)   # + M1, L4 eval / practice, R1, R2a
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in used), BASE
    out = {}
    for tid in load_plan()["dev"]:
        i = tid.split(":")[1]
        x = np.load(os.path.join(R, FRESH_DIR, f"libero_90_{i}.npy"))
        assert len(x) == N_STATES, (tid, x.shape)
        dmin, sets = np.inf, []
        for f in sorted(glob.glob(os.path.join(R, "fresh_*", f"libero_90_{i}.npy"))):
            nm = os.path.basename(os.path.dirname(f))
            if nm == FRESH_DIR:
                continue
            y = np.load(f)
            if y.shape[1] != x.shape[1]:
                continue
            dmin = min(dmin, float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))))
            sets.append(nm)
        assert dmin > 1e-6, (tid, dmin)
        out[tid] = dict(min_dist=dmin, compared=sets)
        print(tid, f"min distance {dmin:.4f} over {len(sets)} sets: {' '.join(sets)}", flush=True)
    write_json(os.path.join(OUT, "fresh_check.json"), out)


# ── fitting a bank token (GPU) ───────────────────────────────────────────────────────────────────────────────────
def key_dir(key):
    return os.path.join(FITS, h("rb2key:" + key, 12))


def _frames_bank(bk, dm, frames, norm, c0):
    """-> (PrefixBank, full chunks [n, H, A], demo id per row) of (episode, frame) pairs; each frame's prompt is its own training
    text (the source cell's instruction)."""
    from lifelong_ood.policy import data as D
    by = {}
    for e, f in frames:
        by.setdefault(e, []).append(f)
    obs, rows, demo = [], [], []
    for e in sorted(by):
        ep = dm.load(e)
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        fr = np.asarray(sorted(by[e]))
        rows.append(c0.chunk_rows(dict(actions=ep["actions"], state=ep["state"]), idx[fr], fr, norm))
        for f in fr:
            obs.append(dm.observation(ep, int(f)))
            demo.append(e)
        ep.pop("image", None)
        ep.pop("wrist_image", None)
    full = D.pad_rows(np.concatenate(rows), bk.H, bk.a, bk.A)
    return M1.build_bank(bk, obs), full, np.asarray(demo)


def _losses(bk, bank, full, tok, draws=K_DRAWS, seed=7):
    """Per-row flow loss without / with the token (fixed draws).  -> (l0 [n], l1 [n])."""
    import torch
    from lifelong_ood.policy.adapt import _flow_batch
    from lifelong_ood.policy.token_fold import attach
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.as_tensor(np.asarray(full, np.float32), device=bk.device)
    l0, l1 = np.zeros(bank.n), np.zeros(bank.n)
    with torch.no_grad():
        for _ in range(draws):
            for i in range(0, bank.n, 32):
                idx = np.arange(i, min(i + 32, bank.n))
                t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
                a = A[idx]
                xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
                u = (eps - a).reshape(len(idx), -1)
                pre = bank.take(idx)
                v0 = bk.v_full(xt, t, pre)
                v1 = bk.v_full(xt, t, attach(pre, tok))
                l0[idx] += ((v0 - u) ** 2).mean(-1).cpu().numpy() / draws
                l1[idx] += ((v1 - u) ** 2).mean(-1).cpu().numpy() / draws
    return l0, l1


def _per_demo(demo, l0, l1):
    g = {}
    for e in np.unique(demo):
        m = demo == e
        g[int(e)] = dict(gain=float(l0[m].mean() - l1[m].mean()), rel=float((l0[m].mean() - l1[m].mean()) / max(l0[m].mean(), 1e-9)),
                         n=int(m.sum()))
    gs = np.array([x["gain"] for x in g.values()])
    return dict(per_demo=g, mean=float(gs.mean()), sd=float(gs.std(ddof=1)) if len(gs) > 1 else 0.0,
                cv=float(gs.std(ddof=1) / abs(gs.mean())) if len(gs) > 1 and gs.mean() != 0 else None,
                positive_share=float((gs > 0).mean()), rel_mean=float(np.mean([x["rel"] for x in g.values()])), demos=len(gs))


def fit_key(bk, key):
    import importlib.util
    import torch
    from lifelong_ood.policy import data as D
    plan = load_plan()
    ent = plan["bank"][key]
    assert ent["fitted"], key
    d = key_dir(key)
    done = os.path.join(d, "fit.json")
    if os.path.exists(done):
        print(f"[rb2] {key} exists", flush=True)
        return
    sp = importlib.util.spec_from_file_location("c0", os.path.join(ROOT, "experiments", "c0_fit.py"))
    c0 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c0)
    dm = D.Demos(root=M1.L90C_DATA)
    # the sources are training texts of lol/libero_l90c (pi05_l90c's own training data); no dev-task demo is read
    norm = D.Norm.for_backbone(bk)
    r = ent["role"]
    tm = {}
    t0 = M1._sync()
    bank, full, demo = _frames_bank(bk, dm, [tuple(x) for x in ent["fit_frames"]], norm, c0)
    tm["bank"] = M1._sync() - t0
    tok = M1.init_token(bank, 0, N_ROLE, SEED + (r == "G")).to(bk.device)
    path = os.path.join(d, "token.pt")

    def save(s):
        L4.save_tok(tok, path, key=key, role=r, kind="bank", step=s, sources=ent["sources"])
    carry = np.full(bank.n, r == "G")
    hist, secs = L4.fit_roles(bk, bank, full, carry, {}, {r: tok}, STEPS, (), save, lambda x: print(f"[rb2] {key} {x}", flush=True))
    tm["fit"] = secs[STEPS]
    # junk check: per-demo flow-loss gain on the fit demos and on held-out demos of the same training texts
    t0 = M1._sync()
    l0, l1 = _losses(bk, bank, full, tok)
    junk = dict(fit=_per_demo(demo, l0, l1))
    del bank
    torch.cuda.empty_cache()
    if ent["hold_frames"]:
        hb, hf, hd = _frames_bank(bk, dm, [tuple(x) for x in ent["hold_frames"]], norm, c0)
        l0, l1 = _losses(bk, hb, hf, tok)
        junk["held_out"] = _per_demo(hd, l0, l1)
        del hb
    tm["junk"] = M1._sync() - t0
    res = dict(key=key, role=r, sources=ent["sources"], n_frames=len(ent["fit_frames"]), n_demos=len(ent["fit_demos"]), token=path,
               seconds=tm, curve=hist, junk=junk, bytes=tok.nbytes(), gpu=torch.cuda.get_device_name(0))
    write_json(done, res)
    print(f"[rb2] {key}: seconds {json.dumps(tm)}; junk fit {junk['fit']['mean']:+.4f} (pos {junk['fit']['positive_share']:.2f}) "
          f"held {junk.get('held_out', {}).get('mean')} (pos {junk.get('held_out', {}).get('positive_share')})", flush=True)
    if SMOKE:
        return
    from lifelong_ood import ledger
    print(ledger.record("rb2_fit", bench="libero_90_c", tag="bank", seed=SEED, keys=dict(key=key),
                        metrics=dict(seconds=tm, n_frames=res["n_frames"], n_demos=res["n_demos"], sources=ent["sources"],
                                     final_loss=hist["loss"][-1] if hist["loss"] else None,
                                     junk={k: {kk: vv for kk, vv in x.items() if kk != "per_demo"} for k, x in junk.items()}),
                        bytes=dict(experts=0, tree=0, codes=res["bytes"], samples=0, teacher_copies=0, total=res["bytes"]),
                        config=dict(n_role=N_ROLE, steps=STEPS, lr=L4.LR, batch=L4.BATCH, ckpt=M1.CKPT, data=M1.L90C_DATA),
                        script="experiments/rb2_bank.py", command=" ".join(sys.argv)), flush=True)


def fit_of(key):
    f = os.path.join(key_dir(key), "fit.json")
    return load_json(f) if os.path.exists(f) else None


# ── served objects ───────────────────────────────────────────────────────────────────────────────────────────────
def role_mode(tid, text, O, G):
    if not O and not G:
        return "ours"
    nm = f"{slug(tid)}_{h(json.dumps(dict(O=list(O), G=list(G), text=text)), 12)}.json"
    p = os.path.join(SPECS, nm)
    if not os.path.exists(p):
        import lifelong_ood.lang.transplant as LT
        os.makedirs(SPECS, exist_ok=True)
        LT.save_spec(p, [], prompt=text, meta=dict(task=tid, probe="rb2"), role_tokens=dict(O=list(O), G=list(G)))
    return f"lt:{p}"


def tokens_for(keys, allowed):
    """-> (O paths, G paths) of the bank tokens for these keys (only `allowed` keys, fitted)."""
    O, G = [], []
    for k in keys:
        if k not in allowed:
            continue
        f = fit_of(k)
        if f is None:
            return None
        (O if k.startswith("O|") else G).append(f["token"])
    return O, G


def gate_result(plan):
    """-> (admitted keys, per gate cell dict) once every gate cell is rolled; None before."""
    adm, per = set(), {}
    base = M1.parity_base()
    for c, gk in plan["gate"].items():
        m = cell_mode(plan, c, set(plan["fitted"]))
        e = eps_of("rb2_gate", c, m, N_STATES)
        if e is None:
            return None
        s = int(sum(e.values()))
        b = int(sum(base[c][i] for i in range(N_STATES)))
        ok = (s - b) / N_STATES >= -GATE_TOL
        per[c] = dict(keys=gk, served=s, parity=b, change=(s - b) / N_STATES, passes=bool(ok))
        if ok:
            adm |= set(gk)
    adm |= set(plan["ungated"])                                         # no training cell with a baseline holds them (none expected)
    return sorted(adm), per


def cell_mode(plan, c, allowed):
    v = plan["cells"][c]
    tk = tokens_for(v["keys"], allowed)
    return None if tk is None else role_mode(c, v["text"], *tk)


def dev_modes(plan, allowed):
    """-> {tid: {arm: mode}} on the dev tasks (None for arms whose tokens are not ready)."""
    out = {}
    for tid, v in plan["dev"].items():
        a = {"zs": "ours", "ceiling": f"lt:{os.path.join(M1.FITS, slug(tid), 'create.json')}"}
        if allowed is not None:
            tO = tokens_for(v["oracle"]["O"], allowed)
            tG = tokens_for(v["oracle"]["G"], allowed)
            if tO is not None and tG is not None:
                a["bank"] = role_mode(tid, v["text"], tO[0], tG[1])
                a["bank_O"] = role_mode(tid, v["text"], tO[0], [])
                a["bank_G"] = role_mode(tid, v["text"], [], tG[1])
        out[tid] = a
    return out


def roll_out(exp, tid, mode):
    return os.path.join(ROLL, f"{exp}_{slug(tid)}_{h(mode)}.json")


def line(tid, mode, exp, tag, kind):
    """kind: eval (fresh_rb2, 20, noise seed 13) | cell (stored init states 0-19, seed 0, as the parity read)."""
    if kind == "cell":
        sp = M1._line(tid, mode, exp, tag, False, 0)
    else:
        sp = M1._line(tid, mode, exp, tag, False, SEED)
        sp["FRESH"] = os.path.join(R, FRESH_DIR)
    sp["TRIALS"] = N_STATES
    sp["OUT"] = roll_out(exp, tid, mode)
    return sp


def eps_of(exp, tid, mode, n):
    f = roll_out(exp, tid, mode)
    if not os.path.exists(f):
        return None
    z = load_json(f)
    out = {int(e["trial"]): int(bool(e["success"])) for t in z["tasks"].values() for e in t["episodes"]}
    return out if len(out) == n else None


# ── job submission ───────────────────────────────────────────────────────────────────────────────────────────────
def sbatch(args, dry):
    if dry:
        print("  [dry] sbatch " + " ".join(args))
        return "DRY"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    return subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True, env=env).stdout.strip().split(";")[0]


def queued_names():
    try:
        r = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True, check=True)
        return set(r.stdout.split())
    except Exception:                                                   # noqa: BLE001
        return set()


def launch_fits(dry):
    plan = load_plan()
    q = queued_names()
    os.makedirs(LOGDIR, exist_ok=True)
    for k in plan["fitted"]:
        nm = f"rb2f_{h('rb2key:' + k, 6)}"
        if fit_of(k) is None and nm not in q:
            j = sbatch(["--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=02:30:00", "--gres=gpu:a40:1",
                        "--mem=96G", f"--exclude={EXCLUDE}", f"--job-name={nm}", f"--output={LOGDIR}/fit_%j.out",
                        f"--error={LOGDIR}/fit_%j.out", f"--export=ALL,SCRIPT=experiments/rb2_bank.py --fit --kid {h('rb2key:' + k, 12)}",
                        "slurm/jobs/robot_gpu.sbatch"], dry)
            print(k, j, flush=True)


def submit_rolls(lines, dry, lanes=6, per=4):
    os.makedirs(os.path.join(OUT, "packs"), exist_ok=True)
    os.makedirs(ROLL, exist_ok=True)
    lf = os.path.join(OUT, "packs", "live.txt")
    live = set()
    if os.path.exists(lf):
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
        path = os.path.join(OUT, "packs", f"rb2_{stamp}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        lane = j % lanes
        jn = f"rb2p{stamp}"
        args = ([f"--dependency=afterany:{last[lane]}"] if last[lane] else []) + [
            "--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=04:00:00", "--gres=gpu:a40:1",
            f"--exclude={EXCLUDE}", f"--job-name={jn}", f"--output={LOGDIR}/pack_%j.out", f"--error={LOGDIR}/pack_%j.out",
            f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
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


def launch_base(dry):
    """zs and the demonstrated ceiling (M1's create token) on fresh_rb2: they do not depend on the bank."""
    plan = load_plan()
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    lines = []
    for tid, a in dev_modes(plan, None).items():
        for arm in ("zs", "ceiling"):
            lines.append(line(tid, a[arm], "rb2_eval", arm, "eval"))
    submit_rolls(lines, dry)


def launch_gate(dry):
    plan = load_plan()
    miss = [k for k in plan["fitted"] if fit_of(k) is None]
    assert not miss, f"fits missing: {miss}"
    lines = [line(c, cell_mode(plan, c, set(plan["fitted"])), "rb2_gate", "gate", "cell") for c in plan["gate"]]
    submit_rolls(lines, dry)


def launch_arms(dry):
    plan = load_plan()
    g = gate_result(plan)
    assert g is not None, "the gate is not read yet"
    adm, per = g
    print("gate:", json.dumps(per), "admitted:", adm, flush=True)
    lines = []
    for tid, a in dev_modes(plan, set(adm)).items():
        for arm in ("bank", "bank_O", "bank_G"):
            if a[arm] != "ours":
                lines.append(line(tid, a[arm], "rb2_eval", arm, "eval"))
    for c in plan["damage"]:
        m = cell_mode(plan, c, set(adm))
        if m != "ours":
            lines.append(line(c, m, "rb2_damage", "damage", "cell"))
    submit_rolls(lines, dry)


# ── failure typing (LIBERO env) ──────────────────────────────────────────────────────────────────────────────────
def type_all():
    import fm_types as FT
    plan = load_plan()
    g = gate_result(plan)
    adm = set(g[0]) if g else None
    ax = FT.axes()

    class Scene(FT.Scene):
        def __init__(self, t):
            super().__init__(t)
            self.fresh = np.load(os.path.join(R, FRESH_DIR, f"{t.split(':')[0]}_{t.split(':')[1]}.npy"))
    for tid, arms in dev_modes(plan, adm).items():
        f = os.path.join(OUT, "types", f"{slug(tid)}.json")
        res = load_json(f) if os.path.exists(f) else dict(task=tid, arms={})
        sc = None
        for arm, m in arms.items():
            rf = roll_out("rb2_eval", tid, m)
            if not os.path.exists(rf):
                continue
            if res["arms"].get(arm, {}).get("mode") == m and len(res["arms"][arm].get("episodes", [])) == N_STATES:
                continue
            if sc is None:
                sc = Scene(tid)
            z = load_json(rf)
            eps = [e for t in z["tasks"].values() for e in t["episodes"]]
            typed = [FT.type_episode(sc, ax[tid], e, FT.load_rec(rf[:-5] + "_rec", tid, int(e["trial"]))) for e in eps]
            res["arms"][arm] = dict(mode=m, episodes=[{k: x.get(k) for k in ("trial", "success", "type", "sub", "bound")} for x in typed])
            print(f"[rb2 types] {tid} {arm}: " + " ".join(f"{x['trial']}:{x['type']}" for x in typed), flush=True)
        if sc is not None:
            sc.env.close()
        write_json(f, res)


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────
def cluster_boot(cols, w, tasks, B=10000, seed=0):
    """Two-level (task, then start state) bootstrap of the pooled linear contrast sum_a w_a * success_a over paired episodes.
    cols: {task: [n, n_arms] array}.  -> dict(est, lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    X = [cols[t] @ np.asarray(w, float) for t in tasks]
    est = float(np.concatenate(X).mean())
    s = np.empty(B)
    for i in range(B):
        num, den = 0.0, 0
        for j in rng.integers(0, len(X), len(X)):
            m = len(X[j])
            num += X[j][rng.integers(0, m, m)].sum()
            den += m
        s[i] = num / den
    return dict(est=est, lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), samples=s)


def read(record):
    from lifelong_ood import stats as ST
    plan = load_plan()
    g = gate_result(plan)
    res = dict(missing=[])
    adm, gate_per = (g if g else (None, None))
    res["gate"] = dict(cells=gate_per, admitted=adm, rejected=sorted(set(plan["fitted"]) - set(adm or [])))
    modes = dev_modes(plan, set(adm) if adm is not None else None)
    tab = {}
    for tid, a in modes.items():
        for arm, m in a.items():
            e = eps_of("rb2_eval", tid, m, N_STATES)
            if e is None:
                res["missing"].append(f"{tid}:{arm}")
                continue
            tab.setdefault(tid, {})[arm] = e
    T = list(plan["dev"])
    arms = ("zs", "bank", "bank_O", "bank_G", "ceiling")
    res["totals"] = {a: int(sum(sum(tab[t][a].values()) for t in T if a in tab.get(t, {}))) for a in arms}
    res["rate_ci"] = {a: ST.bootstrap_rate(tab, a, [t for t in T if a in tab.get(t, {})]) for a in arms
                      if any(a in tab.get(t, {}) for t in T)}
    res["per_task"] = {t: {a: int(sum(x.values())) for a, x in tab.get(t, {}).items()} | dict(oracle=plan["dev"][t]["oracle"])
                       for t in T}
    res["by_bin"] = {b: {a: int(sum(res["per_task"][t].get(a, 0) for t in T if plan["dev"][t]["bin"] == b)) for a in arms}
                     for b in ("0.1", "0.3", "0.5")}

    def cmp(a, b, ts, margin=None, table=None):
        table = table or tab
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
    full = [t for t in T if all(a in tab.get(t, {}) for a in ("zs", "bank", "ceiling"))]
    # H1: gap closure >= 0.5: D = bank - 0.5 zs - 0.5 ceiling > 0 (paired by start state), one-sided cluster-bootstrap p
    cols = {t: np.array([[tab[t]["bank"][i], tab[t]["zs"][i], tab[t]["ceiling"][i]] for i in range(N_STATES)], float) for t in full}
    gap = res["totals"]["ceiling"] - res["totals"]["zs"]
    h1 = dict(tasks=len(full), bank=res["totals"]["bank"], zs=res["totals"]["zs"], ceiling=res["totals"]["ceiling"], gap=gap)
    if full and gap > 0:
        D = cluster_boot(cols, [1.0, -0.5, -0.5], full)
        num = cluster_boot(cols, [1.0, -1.0, 0.0], full, seed=0)
        den = cluster_boot(cols, [0.0, -1.0, 1.0], full, seed=0)
        ratio = num["samples"] / np.where(den["samples"] > 0, den["samples"], np.nan)
        h1.update(closure=(res["totals"]["bank"] - res["totals"]["zs"]) / gap, D=dict(est=D["est"], lo=D["lo"], hi=D["hi"]),
                  p=float(np.mean(D["samples"] <= 0)),
                  closure_ci=[float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))],
                  den_nonpositive_share=float(np.mean(den["samples"] <= 0)))
    else:
        h1.update(closure=None, p=1.0, note="not testable: the ceiling does not exceed zs on fresh_rb2")
    zs_m1, ce_m1 = 180, 234
    h1["closure_vs_m1_reference"] = (res["totals"]["bank"] - res["totals"]["zs"]) / (ce_m1 - zs_m1)
    h2 = cmp("bank", "zs", [t for t in T if "bank" in tab.get(t, {})])
    # H3: binding errors (wrong_object + wrong_goal), bank vs zs, per paired episode
    types = {}
    for t in T:
        f = os.path.join(OUT, "types", f"{slug(t)}.json")
        if os.path.exists(f):
            z = load_json(f)
            types[t] = {a: {int(e["trial"]): e["type"] for e in x["episodes"]} for a, x in z["arms"].items()}
    BIND = ("wrong_object", "wrong_goal")
    cnt = {a: {ty: int(sum(sum(v == ty for v in types[t][a].values()) for t in types if a in types[t]))
               for ty in ("success", "no_commit", "wrong_object", "wrong_goal", "motion", "fixture")} for a in arms}
    res["types"] = cnt
    bt = {t: {a: {i: int(types[t][a].get(i) in BIND) for i in range(N_STATES)} for a in ("zs", "bank") if a in types[t]} for t in types}
    pb = ST.paired(bt, "zs", "bank", [t for t in bt if "zs" in bt[t] and "bank" in bt[t]])
    b_, c_ = ST.discordant(pb)                                          # b_: zs binding error and bank not; c_: the reverse
    h3 = dict(zs_bind=int(sum(p[0].sum() for p in pb.values())), bank_bind=int(sum(p[1].sum() for p in pb.values())),
              discordant=[b_, c_], sign_p=float(ST.sign_test(b_, c_)), tasks=len(pb),
              diff=ST.bootstrap_diff(pb, B=10000) if pb else None)
    h3["p_dir"] = h3["sign_p"] if h3["bank_bind"] < h3["zs_bind"] else 1.0
    res["types_ceiling_bind"] = sum(cnt["ceiling"][x] for x in BIND)
    # H4: damage on the damage cells (gated bank routed by parts) vs parity, non-inferiority at margin 0.05
    base = M1.parity_base()
    dtab, dmiss = {}, []
    for c in plan["damage"]:
        m = cell_mode(plan, c, set(adm or []))
        if m == "ours":
            dtab[c] = dict(served={i: base[c][i] for i in range(N_STATES)}, parity={i: base[c][i] for i in range(N_STATES)})
            continue
        e = eps_of("rb2_damage", c, m, N_STATES)
        if e is None:
            dmiss.append(c)
            continue
        dtab[c] = dict(served=e, parity={i: base[c][i] for i in range(N_STATES)})
    if dtab and not dmiss:
        h4 = cmp("served", "parity", list(dtab), margin=0.05, table=dtab)
        h4["per_cell"] = {c: dict(served=int(sum(v["served"].values())), parity=int(sum(v["parity"].values())),
                                  keys=plan["cells"][c]["keys"]) for c, v in dtab.items()}
        h4["mean_change"] = h4["diff"]["diff"]
    else:
        h4 = dict(p_noninferior=1.0, missing=dmiss)
    hm = M1.holm(dict(H1=h1["p"], H2=h2["p_dir"], H3=h3["p_dir"], H4=h4["p_noninferior"]))
    for k, x in (("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)):
        x["passes"] = bool(hm[k]["reject"])
    res["go"] = bool(h1["passes"])
    res["secondary"] = {f"{a}_vs_{b}": cmp(a, b, [t for t in T if a in tab.get(t, {}) and b in tab.get(t, {})])
                        for a, b in (("bank_O", "zs"), ("bank_G", "zs"), ("bank", "ceiling"), ("ceiling", "zs"), ("bank", "bank_O"),
                                     ("bank", "bank_G"))}
    # junk check and bytes
    jk = {}
    for k in plan["fitted"]:
        f = fit_of(k)
        if f:
            jk[k] = dict(sources=len(f["sources"]), demos=f["n_demos"], frames=f["n_frames"], seconds=f["seconds"],
                         **{w: {kk: vv for kk, vv in x.items() if kk != "per_demo"} for w, x in f["junk"].items()})
    res["junk"] = jk
    res["junk_flagged"] = sorted(k for k, x in jk.items() if x.get("held_out", {}).get("positive_share", 1.0) < 0.5)
    tb = 36864
    res["bytes"] = dict(per_token=tb, bank_fitted=tb * len(plan["fitted"]), bank_admitted=tb * len(adm or []),
                        per_task_served={t: tb * (len(plan["dev"][t]["oracle"]["O"]) + len(plan["dev"][t]["oracle"]["G"])) for t in T},
                        m1_token=73728, m1_lora=18710528)
    res["fit_a40_s"] = dict(total=float(sum(x["seconds"].get("bank", 0) + x["seconds"].get("fit", 0) for x in jk.values())),
                            per_token=float(np.mean([x["seconds"].get("bank", 0) + x["seconds"].get("fit", 0) for x in jk.values()]))
                            if jk else None)
    res.update(H1=h1, H2=h2, H3=h3, H4=h4, holm=hm)
    write_json(os.path.join(OUT, "read.json"), res)
    print(json.dumps({k: res[k] for k in ("missing", "totals", "by_bin", "types", "holm", "go", "gate", "junk_flagged", "bytes",
                                          "fit_a40_s")}, indent=1, default=str))
    print("[rb2] H1", json.dumps(h1, default=str))
    for hh in ("H2", "H4"):
        x = res[hh]
        if "x" in x:
            print(f"[rb2] {hh} {x['x']} vs {x['y']} (n {x['n']}, tasks {x['tasks']}), discordant {x['discordant']}, sign p {x['sign_p']:.3g}, "
                  f"diff {x['diff']['diff']:+.3f} [{x['diff']['lo']:+.3f}, {x['diff']['hi']:+.3f}] p_ni {x.get('p_noninferior')} "
                  f"passes {x['passes']}")
    print("[rb2] H3", json.dumps(h3, default=str))
    print("[rb2] H4 per cell", json.dumps(h4.get("per_cell")))
    print("[rb2] secondary", json.dumps({k: (v["x"], v["y"], v["discordant"], round(v["sign_p"], 4)) for k, v in res["secondary"].items()}))
    print("[rb2] junk", json.dumps({k: (x["fit"]["mean"], x["fit"]["positive_share"], x.get("held_out", {}).get("mean"),
                                        x.get("held_out", {}).get("positive_share"), x.get("held_out", {}).get("cv")) for k, x in jk.items()}))
    for t in T:
        print(f"  {t:13} {plan['dev'][t]['bin']} " + " ".join(f"{a} {res['per_task'][t].get(a, '-')}" for a in arms)
              + f"  oracle {plan['dev'][t]['oracle']}")
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rb2_read", bench="libero_90_c", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=res["bytes"]["bank_admitted"], samples=0, teacher_copies=0,
                                       total=res["bytes"]["bank_admitted"]),
                            config=dict(plan=plan["settings"], prereg="m3_prereg tag pr2_bank", ckpt=M1.CKPT),
                            script="experiments/rb2_bank.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("plan", "fresh-check", "fit", "launch-fits", "launch-base", "launch-gate", "launch-arms", "type", "read", "record", "dry"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--key", default=None)
    ap.add_argument("--kid", default=None)                             # h("rb2key:" + key, 12): keys hold spaces (sbatch splits SCRIPT)
    a = ap.parse_args()
    if a.plan:
        make_plan()
    if a.fresh_check:
        fresh_check()
    if a.fit:
        k_ = a.key or next(k for k in load_plan()["fitted"] if h("rb2key:" + k, 12) == a.kid)
        fit_key(M1.backbone(), k_)
    if a.launch_fits:
        launch_fits(a.dry)
    if a.launch_base:
        launch_base(a.dry)
    if a.launch_gate:
        launch_gate(a.dry)
    if a.launch_arms:
        launch_arms(a.dry)
    if a.type:
        type_all()
    if a.read:
        read(a.record)
