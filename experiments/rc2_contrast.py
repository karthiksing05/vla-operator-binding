"""PLAN R, probe R2c (docs/PLAN_R2C.md; pre-registration experiments/prereg/pr2c_contrast.py, ledger m3_prereg tag pr2c_contrast):
R2's bank of role tokens (experiments/rb2_bank.py, docs/PLAN_R2.md) re-extracted with a CONTRASTIVE MINIMAL-PAIR objective on
TRAINING data only.  Everything but the extraction objective is R2's: the 16 part keys and their key rule, each key's fit demos and
fit frames (R2's plan.json), the token unit (KVToken n 2 per layer), init, frozen pi05_l90c@5000, 1000 steps, Adam lr 1e-2,
batch 32, seed 13.  No dev-task demo is read anywhere in this file; demos are only fitted and scored, never executed or replayed.

Contrast partners of a part key p (role r): training texts that do NOT hold p, hold some other filler in role r, and share a scene
(factors' scene of their training cells) with a source text of p.
  tier A: the partner's other-role key set equals the source's (a minimal pair: only role r's filler differs);
  tier B: same scene, role r's filler differs (the other role may differ too);  used only when a key has no tier-A partner;
  none:   no partner -> the key keeps R2's pooled token (fallback, reported separately; it fails the specificity count).

Objective (per step: R2's p batch of 32 + a contrast batch of 32 role-phase frames of partner demos, each under its own instruction):
  L = FM(tok | p) / Lbar0_p  +  LAMBDA * relu(MARGIN - (FM(tok | c) - FM(none | c)) / FM(none | c))
  Lbar0_p = the no-token flow loss on the fit frames (fixed draws, computed once); FM(none | c) is the no-token loss on the SAME
  contrast batch and draw (no grad).  LAMBDA 1, MARGIN 0.02 (docs/PLAN_R2C.md section 2 gives the principle).

Held-out split (fixed, seeded by the key): 16 p demos from the source texts' episodes R2 never fitted (round robin over sources),
16 partner demos never fitted here; <= 24 role-phase frames each.  Both tokens (R2's and R2c's) are scored on the identical frames
and draws.

    sbatch ... cpu.sbatch SCRIPT="experiments/rc2_contrast.py --plan"     contrast table -> results/rc2/plan.json
    GPU: --fit --kid <h(rc2key:key)>                                      one contrastive token + held-out scores of both tokens
    ... --launch-fits [--dry]
    ... --stage1 [--record]                                               the stage-1 gate -> results/rc2/stage1.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1                                          # noqa: E402
import l4_store as L4                                                   # noqa: E402
import rb2_bank as RB2                                                  # noqa: E402

OUT = os.path.join(ROOT, "results", "rc2")
SMOKE = os.environ.get("RC2_SMOKE") == "1"                              # pipeline smoke: 20 steps, 32 held-out frames, no ledger row, no gain printed
FITS = os.path.join(OUT, "smoke" if SMOKE else "fits")
LOGDIR = os.path.join(ROOT, "slurm", "rc2")
SEED = RB2.SEED
STEPS = 20 if SMOKE else 1000
LAMBDA, MARGIN = 1.0, 0.02
C_TOT, C_MIN, C_FMAX = RB2.D_TOT, RB2.D_MIN, RB2.F_MAX                # partner fit demos / per text / frames: R2's p-side budget
H_DEMOS, H_FRAMES = 16, RB2.F_HOLD                                     # held-out demos per side, frames per held-out demo
E_DRAWS = 4                                                            # held-out scoring: fixed (t, eps) draws per frame
GATE_N = 8                                                             # stage 1: >= 8 of 16 tokens on each criterion
EXCLUDE = M1.EXCLUDE


def h(s, n=10):
    return hashlib.sha1(s.encode()).hexdigest()[:n]


write_json, load_json = RB2.write_json, RB2.load_json


# ── the contrast table (CPU; parses, scenes, training-demo gripper commands only) ───────────────────────────────────────
def make_plan():
    from lifelong_ood.policy import data as D
    rb = RB2.load_plan()
    tt = M1.task_table()
    dm = D.Demos(root=M1.L90C_DATA)
    train = rb["train"]
    scenes = {ti: sorted({tt[tid].scene for tid in v["tids"]}) for ti, v in train.items()}
    phase_cache = {}

    def phase_frames(ep, role):
        if ep not in phase_cache:
            phase_cache[ep] = L4.phases(dm.load(ep)["actions"])
        c = phase_cache[ep]
        return np.nonzero(c if role == "G" else ~c)[0]

    def take_frames(eps, role, cap):
        out = []
        for e in eps:
            fr = phase_frames(e, role)
            if len(fr):
                out += [(int(e), int(fr[int(round(x))])) for x in sorted({int(round(y)) for y in np.linspace(0, len(fr) - 1, min(len(fr), cap))})]
        return out

    def round_robin(groups, n, rng):
        """groups: {text: [episodes]} -> n episodes, one per text in turn (texts in sorted order, episodes in a seeded order)."""
        q = {t: list(rng.permutation(g)) for t, g in sorted(groups.items()) if len(g)}
        out = []
        while len(out) < n and any(q.values()):
            for t in sorted(q):
                if q[t] and len(out) < n:
                    out.append(int(q[t].pop(0)))
        return out
    keys = {}
    for k in rb["fitted"]:
        e = rb["bank"][k]
        r, other = e["role"], ("G" if e["role"] == "O" else "O")
        src = e["sources"]
        cand = {}
        for s in src:
            for ti, v in train.items():
                if ti in src or k in v["keys"][r] or not v["keys"][r]:
                    continue
                shared = sorted(set(scenes[ti]) & set(scenes[s]))
                if not shared:
                    continue
                tier = "A" if v["keys"][other] == train[s]["keys"][other] else "B"
                c = cand.setdefault(ti, dict(tier="B", pairs=[]))
                c["pairs"].append(dict(source=s, scenes=shared, tier=tier))
                if tier == "A":
                    c["tier"] = "A"
        tier = "A" if any(c["tier"] == "A" for c in cand.values()) else ("B" if cand else "none")
        partners = sorted(t for t, c in cand.items() if c["tier"] == tier) if tier != "none" else []
        rng = np.random.default_rng(int(h("rc2hold:" + k, 12), 16))
        fitset = set(e["fit_demos"])
        hold_p = round_robin({s: [x for x in train[s]["episodes"] if x not in fitset] for s in src}, H_DEMOS, rng)
        ent = dict(role=r, sources=src, tier=tier, partners=partners, candidates=cand, hold_p=hold_p,
                   hold_p_frames=take_frames(hold_p, r, H_FRAMES), fit_c=[], fit_c_frames=[], hold_c=[], hold_c_frames=[])
        if partners:
            ent["hold_c"] = round_robin({t: train[t]["episodes"] for t in partners}, H_DEMOS, rng)
            hc = set(ent["hold_c"])
            n_t = max(C_MIN, int(np.ceil(C_TOT / len(partners))))
            pairs = []
            for t in partners:
                pool = [x for x in train[t]["episodes"] if x not in hc]
                pick = sorted({pool[int(round(x))] for x in np.linspace(0, len(pool) - 1, min(n_t, len(pool)))})
                ent["fit_c"] += pick
                for ep in pick:
                    pairs += [(int(ep), int(f)) for f in phase_frames(ep, r)]
            sel = sorted({int(round(x)) for x in np.linspace(0, len(pairs) - 1, min(len(pairs), C_FMAX))}) if pairs else []
            ent["fit_c_frames"] = [pairs[i] for i in sel]
            ent["hold_c_frames"] = take_frames(ent["hold_c"], r, H_FRAMES)
        assert not set(ent["hold_p"]) & fitset and not set(ent["hold_c"]) & set(ent["fit_c"])
        keys[k] = ent
    res = dict(keys=keys, scenes=scenes, rb2_plan_sha256=hashlib.sha256(open(os.path.join(RB2.OUT, "plan.json"), "rb").read()).hexdigest(),
               settings=dict(lam=LAMBDA, margin=MARGIN, c_tot=C_TOT, c_min=C_MIN, c_fmax=C_FMAX, h_demos=H_DEMOS, h_frames=H_FRAMES,
                             e_draws=E_DRAWS, gate_n=GATE_N, steps=STEPS, lr=L4.LR, batch=L4.BATCH, seed=SEED, ckpt=M1.CKPT))
    write_json(os.path.join(OUT, "plan.json"), res)
    for k, e in keys.items():
        print(f"{k:36} tier {e['tier']:4} sources {[train[s]['text'] for s in e['sources']]}", flush=True)
        print(f"    partners {[train[t]['text'] for t in e['partners']]}", flush=True)
        print(f"    fit_c demos {len(e['fit_c'])} frames {len(e['fit_c_frames'])}; hold_p {len(e['hold_p'])} demos "
              f"{len(e['hold_p_frames'])} frames; hold_c {len(e['hold_c'])} demos {len(e['hold_c_frames'])} frames", flush=True)
    print("tiers:", {t: sum(e["tier"] == t for e in keys.values()) for t in ("A", "B", "none")}, flush=True)


def load_plan():
    return load_json(os.path.join(OUT, "plan.json"))


# ── the contrastive fit (GPU) ──────────────────────────────────────────────────────────────────────────────────────────
def key_dir(key):
    return os.path.join(FITS, h("rc2key:" + key, 12))


def _c0():
    import importlib.util
    sp = importlib.util.spec_from_file_location("c0", os.path.join(ROOT, "experiments", "c0_fit.py"))
    c0 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c0)
    return c0


def _baseline(bk, bank, full, draws=1, seed=7):
    """Mean no-token flow loss over the bank's rows (fixed draws)."""
    l = _score(bk, bank, full, [], draws, seed)[0]
    return float(l.mean())


def _score(bk, bank, full, toks, draws=E_DRAWS, seed=7):
    """Per-row flow loss with no token (index 0) and with each token of `toks` (indices 1..), identical draws.  -> [1+len, n]."""
    import torch
    from lifelong_ood.policy.adapt import _flow_batch
    from lifelong_ood.policy.token_fold import attach
    st = torch.get_rng_state()
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.as_tensor(np.asarray(full, np.float32), device=bk.device)
    L = np.zeros((1 + len(toks), bank.n))
    with torch.no_grad():
        for _ in range(draws):
            for i in range(0, bank.n, 32):
                idx = np.arange(i, min(i + 32, bank.n))
                t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
                a = A[idx]
                xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
                u = (eps - a).reshape(len(idx), -1)
                pre = bank.take(idx)
                for j, tk in enumerate([None] + list(toks)):
                    v = bk.v_full(xt, t, pre if tk is None else attach(pre, tk))
                    L[j, idx] += ((v - u) ** 2).mean(-1).cpu().numpy() / draws
    torch.set_rng_state(st)
    return L


def _per_demo(demo, l0, l1):
    rel = []
    for e in np.unique(demo):
        m = demo == e
        rel.append(float((l0[m].mean() - l1[m].mean()) / max(l0[m].mean(), 1e-9)))
    rel = np.asarray(rel)
    return dict(rel_mean=float(rel.mean()) if len(rel) else None, positive_share=float((rel > 0).mean()) if len(rel) else None,
                demos=int(len(rel)), per_demo=[float(x) for x in rel])


def fit_contrast(bk, bank, full, cbank, cfull, tok, lbar, log):
    """R2's p-side sampling (rng / generator / torch seed 13, as L4.fit_roles) + a partner batch on its own RNG streams."""
    import torch
    from lifelong_ood.policy.adapt import _flow_batch
    from lifelong_ood.policy.token_fold import attach
    torch.manual_seed(SEED)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cur = torch.get_rng_state()
    torch.manual_seed(SEED + 1)
    c_state = torch.get_rng_state()
    torch.set_rng_state(cur)
    gen_c = torch.Generator(device="cpu").manual_seed(SEED + 1)
    rng_c = np.random.default_rng(SEED + 1)
    params = list(tok.parameters())
    opt = torch.optim.Adam(params, lr=L4.LR)
    A = torch.as_tensor(np.asarray(full, np.float32), device=bk.device)
    C = torch.as_tensor(np.asarray(cfull, np.float32), device=bk.device)
    hist = dict(step=[], fm_p=[], c_rel_gain=[], hinge=[])
    t0 = M1._sync()
    for s in range(1, STEPS + 1):
        idx = rng.integers(0, bank.n, size=min(L4.BATCH, bank.n))
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        a = A[idx]
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        v = bk.v_full(xt, t, attach(bank.take(idx), tok))
        fm_p = ((v - u) ** 2).mean()
        (fm_p / lbar).backward()
        cur = torch.get_rng_state()
        torch.set_rng_state(c_state)
        jdx = rng_c.integers(0, cbank.n, size=min(L4.BATCH, cbank.n))
        tc, epsc = _flow_batch(len(jdx), bk.H, bk.A, gen_c, bk.device)
        c_state = torch.get_rng_state()
        torch.set_rng_state(cur)
        ac = C[jdx]
        xc = tc[:, None, None] * epsc + (1 - tc[:, None, None]) * ac
        uc = (epsc - ac).reshape(len(jdx), -1)
        prec = cbank.take(jdx)
        with torch.no_grad():
            f0 = ((bk.v_full(xc, tc, prec) - uc) ** 2).mean()
        f1 = ((bk.v_full(xc, tc, attach(prec, tok)) - uc) ** 2).mean()
        rel_raise = (f1 - f0) / f0
        hinge = torch.relu(MARGIN - rel_raise)
        if float(hinge) > 0:
            (LAMBDA * hinge).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 50 == 0 or s == STEPS:
            hist["step"].append(s)
            hist["fm_p"].append(float(fm_p))
            hist["c_rel_gain"].append(float(-rel_raise))
            hist["hinge"].append(float(hinge))
        if s % 200 == 0 or s == STEPS:
            log(f"step {s}: fm_p {float(fm_p):.4f} contrast rel raise {float(rel_raise):+.4f} hinge {float(hinge):.4f} "
                f"({M1._sync() - t0:.0f}s)")
    return hist, M1._sync() - t0


def fit_key(bk, key):
    import torch
    from lifelong_ood.policy import data as D
    rb = RB2.load_plan()
    pl = load_plan()
    ent, rbe = pl["keys"][key], rb["bank"][key]
    d = key_dir(key)
    done = os.path.join(d, "fit.json")
    if os.path.exists(done):
        print(f"[rc2] {key} exists", flush=True)
        return
    r2 = RB2.fit_of(key)
    assert r2 is not None, key
    c0 = _c0()
    dm = D.Demos(root=M1.L90C_DATA)
    norm = D.Norm.for_backbone(bk)
    tm = {}
    path = os.path.join(d, "token.pt")
    hist = None
    if ent["tier"] == "none":
        # fallback: R2's pooled token is the token (same objective, data, seed); not refitted
        tok_path = r2["token"]
    else:
        t0 = M1._sync()
        bank, full, _ = RB2._frames_bank(bk, dm, [tuple(x) for x in rbe["fit_frames"]], norm, c0)
        cbank, cfull, _ = RB2._frames_bank(bk, dm, [tuple(x) for x in ent["fit_c_frames"]], norm, c0)
        tm["bank"] = M1._sync() - t0
        lbar = _baseline(bk, bank, full)
        tok = M1.init_token(bank, 0, RB2.N_ROLE, SEED + (ent["role"] == "G")).to(bk.device)
        hist, tm["fit"] = fit_contrast(bk, bank, full, cbank, cfull, tok, lbar, lambda x: print(f"[rc2] {key} {x}", flush=True))
        hist["lbar0_p"] = lbar
        L4.save_tok(tok, path, key=key, role=ent["role"], kind="rc2_contrast", step=STEPS, sources=ent["sources"],
                    partners=ent["partners"], lam=LAMBDA, margin=MARGIN)
        del bank, cbank
        torch.cuda.empty_cache()
        tok_path = path
    # held-out scores: no token / R2 token / R2c token, identical frames and draws (both loaded from their fp16 files)
    t0 = M1._sync()
    toks = [L4.load_token(r2["token"], bk.device), L4.load_token(tok_path, bk.device)]
    held = {}
    for side, fr in (("p", ent["hold_p_frames"]), ("c", ent["hold_c_frames"])):
        fr = fr[:32] if SMOKE else fr
        if not fr:
            continue
        hb, hf, hd = RB2._frames_bank(bk, dm, [tuple(x) for x in fr], norm, c0)
        L = _score(bk, hb, hf, toks)
        held[side] = dict(r2=_per_demo(hd, L[0], L[1]), rc2=_per_demo(hd, L[0], L[2]), l0_mean=float(L[0].mean()))
        del hb
        torch.cuda.empty_cache()
    tm["score"] = M1._sync() - t0
    res = dict(key=key, role=ent["role"], tier=ent["tier"], token=tok_path, r2_token=r2["token"], curve=hist, held=held, seconds=tm,
               bytes=toks[1].nbytes(), gpu=torch.cuda.get_device_name(0))
    write_json(done, res)
    if SMOKE:
        print(f"[rc2] smoke {key} done (scores not printed)", flush=True)
        return
    print(f"[rc2] {key}: " + json.dumps({s: {a: x[a]["rel_mean"] for a in ("r2", "rc2")} for s, x in held.items()}), flush=True)
    from lifelong_ood import ledger
    print(ledger.record("rc2_fit", bench="libero_90_c", tag="contrast", seed=SEED, keys=dict(key=key),
                        metrics=dict(tier=ent["tier"], seconds=tm, held={s: {a: {kk: vv for kk, vv in x[a].items() if kk != "per_demo"}
                                                                              for a in ("r2", "rc2")} for s, x in held.items()},
                                     final=None if hist is None else {k_: v_[-1] for k_, v_ in hist.items() if isinstance(v_, list) and v_}),
                        bytes=dict(experts=0, tree=0, codes=res["bytes"], samples=0, teacher_copies=0, total=res["bytes"]),
                        config=dict(lam=LAMBDA, margin=MARGIN, steps=STEPS, lr=L4.LR, batch=L4.BATCH, ckpt=M1.CKPT),
                        script="experiments/rc2_contrast.py", command=" ".join(sys.argv)), flush=True)


def fit_of(key):
    f = os.path.join(key_dir(key), "fit.json")
    return load_json(f) if os.path.exists(f) else None


def launch_fits(dry):
    pl = load_plan()
    q = RB2.queued_names()
    os.makedirs(LOGDIR, exist_ok=True)
    for k in pl["keys"]:
        nm = f"rc2f_{h('rc2key:' + k, 6)}"
        if fit_of(k) is None and nm not in q:
            j = RB2.sbatch(["--partition=overcap", "--account=overcap", "--qos=short", "--requeue", "--time=03:00:00", "--gres=gpu:a40:1",
                            "--mem=128G", f"--exclude={EXCLUDE}", f"--job-name={nm}", f"--output={LOGDIR}/fit_%j.out",
                            f"--error={LOGDIR}/fit_%j.out", f"--export=ALL,SCRIPT=experiments/rc2_contrast.py --fit --kid {h('rc2key:' + k, 12)}",
                            "slurm/jobs/robot_gpu.sbatch"], dry)
            print(k, j, flush=True)


# ── stage 1: the offline gate ──────────────────────────────────────────────────────────────────────────────────────────
def stage1(record):
    pl = load_plan()
    per, miss = {}, []
    for k, e in pl["keys"].items():
        f = fit_of(k)
        if f is None:
            miss.append(k)
            continue
        hp, hc = f["held"].get("p"), f["held"].get("c")
        row = dict(role=e["role"], tier=e["tier"])
        for a in ("r2", "rc2"):
            g = hp[a]["rel_mean"] if hp else None
            gc = hc[a]["rel_mean"] if hc else None
            row[a] = dict(gain_p=g, gain_c=gc, spec=(g - gc) if (g is not None and gc is not None) else None,
                          pos_share_p=hp[a]["positive_share"] if hp else None)
        per[k] = row
    res = dict(missing=miss, per_token=per)
    for a in ("r2", "rc2"):
        ng = sum(1 for x in per.values() if x[a]["gain_p"] is not None and x[a]["gain_p"] > 0)
        ns = sum(1 for x in per.values() if x[a]["spec"] is not None and x[a]["spec"] > 0)
        res[a] = dict(n_gain_pos=ng, n_spec_pos=ns, n=len(pl["keys"]),
                      median_gain_p=float(np.median([x[a]["gain_p"] for x in per.values() if x[a]["gain_p"] is not None])) if per else None)
    rc = res["rc2"]
    res["gate"] = dict(gain_pass=rc["n_gain_pos"] >= GATE_N, spec_pass=rc["n_spec_pos"] >= GATE_N, rule=f">= {GATE_N} of 16 each")
    res["stage1_pass"] = bool(not miss and res["gate"]["gain_pass"] and res["gate"]["spec_pass"])
    both = [k for k, x in per.items() if x["r2"]["gain_p"] is not None]
    res["rc2_minus_r2"] = dict(gain_p_better=sum(per[k]["rc2"]["gain_p"] > per[k]["r2"]["gain_p"] for k in both),
                               spec_better=sum(per[k]["rc2"]["spec"] > per[k]["r2"]["spec"] for k in both
                                               if per[k]["rc2"]["spec"] is not None),
                               n=len(both))
    secs = [fit_of(k)["seconds"] for k in pl["keys"] if fit_of(k)]
    res["a40_s"] = dict(total=float(sum(s.get("bank", 0) + s.get("fit", 0) + s.get("score", 0) for s in secs)))
    write_json(os.path.join(OUT, "stage1.json"), res)
    for k, x in per.items():
        print(f"{k:36} {x['tier']:4} " + "  ".join(
            f"{a}: gain_p {x[a]['gain_p'] if x[a]['gain_p'] is None else round(x[a]['gain_p'] * 100, 2)}% "
            f"gain_c {x[a]['gain_c'] if x[a]['gain_c'] is None else round(x[a]['gain_c'] * 100, 2)}% "
            f"spec {x[a]['spec'] if x[a]['spec'] is None else round(x[a]['spec'] * 100, 2)}" for a in ("r2", "rc2")), flush=True)
    print(json.dumps({k: res[k] for k in ("missing", "r2", "rc2", "gate", "stage1_pass", "rc2_minus_r2", "a40_s")}, indent=1), flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rc2_stage1", bench="libero_90_c", tag="stage1", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=36864 * len(per), samples=0, teacher_copies=0, total=36864 * len(per)),
                            config=dict(plan=pl["settings"], prereg="m3_prereg tag pr2c_contrast", ckpt=M1.CKPT),
                            script="experiments/rc2_contrast.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("plan", "fit", "launch-fits", "stage1", "record", "dry"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--kid", default=None)
    a = ap.parse_args()
    if a.plan:
        make_plan()
    if a.fit:
        fit_key(M1.backbone(), next(k for k in load_plan()["keys"] if h("rc2key:" + k, 12) == a.kid))
    if a.launch_fits:
        launch_fits(a.dry)
    if a.stage1:
        stage1(a.record)
