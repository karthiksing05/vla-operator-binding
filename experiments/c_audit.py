"""P0 contamination audit for LIBERO-90-C (docs/PLAN_V4.md 3.1; the I6 method of experiments/i6_novelty.py / c0_fit.py --score).

Was pi05_base (openpi's pretrained base, BEFORE any LIBERO fine-tune) exposed to LIBERO-90?  Score the frozen flow-matching
loss of a policy on demonstration frames, under each task's own instruction, with FIXED draws of (time, noise) per frame shared
by every model:
    groups   train   the split's LIBERO-90 training cells
             test    its DEV combination test tasks (all CD bins; the sealed third is never read)
             part    its DEV part-held-out tasks
             l40     pi05_libero's own 40 fine-tuning tasks (physical-intelligence/libero): the "certainly seen" reference
    models   pi05_base_libero   pi05_base under the LIBERO transforms + pi05_libero's norm stats (lifelong_ood/policy/l90c_config)
             pi05_libero        the reference: SEEN on l40, never fine-tuned on LIBERO-90
Readout.  If pi05_base had seen LIBERO-90, its loss on LIBERO-90 frames would sit near pi05_libero's loss on ITS training data
(l40) rather than near pi05_base's own loss on the unseen reference, and pi05_base's train/test gap would be ~0 either way (the
split is symmetric for a model that saw everything or nothing).  Reported: per-group mean loss per model (real action dims,
and all 32 dims as I6 did), the base's train-vs-test AUC, and the ratio base(l90) / libero(l40).

    $PY experiments/c_audit.py --convert      # once: pi05_base JAX -> PyTorch (GPU node)
    $PY experiments/c_audit.py --score [--record]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "l90c", "c_audit.json")
DEMOS = (0, 1)                 # demonstrations per task
FRAMES = 16                    # evenly spaced frames per demonstration
DRAWS = 4                      # (t, eps) draws per frame
L40_TASKS = None               # all 40


def convert():
    from lifelong_ood.policy import l90c_config as LC
    src, dst = f"{LC.CACHE}/pi05_base", LC.pytorch_dir("pi05_base_libero")
    if os.path.exists(os.path.join(dst, "model.safetensors")):
        print(f"[c_audit] {dst} exists", flush=True)
    else:
        cmd = [sys.executable, "/coc/flash10/ksingara3/robot/openpi/examples/convert_jax_model_to_pytorch.py",
               "--checkpoint_dir", src, "--config_name", "pi05_libero", "--output_path", dst]
        print("[c_audit]", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd="/coc/flash10/ksingara3/robot/openpi")
        if rc:
            sys.exit(rc)
    a = os.path.join(dst, "assets")
    want = os.path.join(a, "physical-intelligence", "libero", "norm_stats.json")
    if not os.path.exists(want):                    # the LIBERO norm stats the audit config names (pi05_libero's)
        if os.path.exists(a):
            shutil.move(a, a + "_pi05_base")        # pi05_base's own robot assets, kept beside
        shutil.copytree(f"{LC.CACHE}/pi05_libero/assets", a)
    print(f"[c_audit] assets: {sorted(os.listdir(a))}", flush=True)


def groups():
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    s = l90c.load_split()
    g = dict(train=[t for t in s["train_l90"]], test=[t for b in s["bins"] for t in s["dev"][b]], part=list(s["dev"]["part"]))
    d40 = D.Demos()
    g["l40"] = [f"pi40:{k}" for k in sorted(d40.tasks)][: L40_TASKS]
    return g


def frames_of(tid, D, d40, d90):
    """-> (observations, padded rows [n, H, 32]) for one task: DEMOS demonstrations x FRAMES evenly spaced frames."""
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D_
    norm = D_.Norm.load()
    obs, rows = [], []
    if tid.startswith("pi40:"):
        src, eps = d40, d40.episodes(int(tid.split(":")[1]))
    else:
        l90c.check_task(tid)                         # refuses a sealed task
        src, eps = d90, d90.episodes_of_task(int(tid.split(":")[1]))
    for e in [eps[k] for k in DEMOS if k < len(eps)]:
        ep = src.load(e)
        T = len(ep["actions"])
        idx, _ = D_.chunk_indices(T, 10)
        fr = np.unique(np.linspace(0, T - 1, FRAMES).round().astype(int))
        rows.append(D_.pad_rows(D_.to_rows(ep["actions"][idx[fr]], norm), 10))
        for i in fr:
            obs.append(src.observation(ep, int(i)))
    return obs, np.concatenate(rows, 0)


def score_model(name, cache, draws_seed=0, batch=16):
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(name, device="cuda")
    print(f"[c_audit] {name} loaded from {bk.checkpoint_dir} in {bk.load_secs:.0f}s", flush=True)
    real = torch.as_tensor(bk.real, device=bk.device)
    out = {}
    for tid, (obs, rows) in cache.items():
        gen = torch.Generator(device="cpu").manual_seed(draws_seed + zlib.crc32(tid.encode()) % 100000)
        lr, lf = [], []
        with torch.no_grad():
            for s in range(0, len(obs), batch):
                pre = bk.prefix(bk.observations(obs[s:s + batch]))
                x = torch.as_tensor(rows[s:s + batch], device=bk.device)
                n = len(x)
                for _ in range(DRAWS):
                    b = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                    t = (b.sample((n,), ) * 0.999 + 0.001).to(bk.device)
                    eps = torch.randn((n, bk.H, bk.A), generator=gen).to(bk.device)
                    xt = t[:, None, None] * eps + (1 - t[:, None, None]) * x
                    u = (eps - x).reshape(n, -1)
                    v = bk.v_full(xt, t, pre)
                    d = (v - u) ** 2
                    lr.append(d[:, real].mean(1).cpu().numpy())
                    lf.append(d.mean(1).cpu().numpy())
        out[tid] = dict(real=float(np.concatenate(lr).mean()), full=float(np.concatenate(lf).mean()), frames=len(obs))
    del bk
    torch.cuda.empty_cache()
    return out


def score(record):
    from lifelong_ood.policy import data as D
    t0 = time.time()
    g = groups()
    d40, d90 = D.Demos(), D.Demos90()
    cache = {}
    for k, tids in g.items():
        for tid in tids:
            cache[tid] = frames_of(tid, D, d40, d90)
        print(f"[c_audit] group {k}: {len(tids)} tasks, frames loaded ({time.time() - t0:.0f}s)", flush=True)
    res = dict(groups=g, demos=list(DEMOS), frames_per_demo=FRAMES, draws=DRAWS, models={})
    for name in ("pi05_base_libero", "pi05_libero"):
        per = score_model(name, cache)
        summ = {}
        for k, tids in g.items():
            for dim in ("real", "full"):
                v = np.array([per[t][dim] for t in tids])
                summ[f"{k}_{dim}_mean"] = float(v.mean())
                summ[f"{k}_{dim}_median"] = float(np.median(v))
        a, b = np.array([per[t]["real"] for t in g["train"]]), np.array([per[t]["real"] for t in g["test"] + g["part"]])
        summ["auc_train_lt_heldout"] = float(np.mean(a[:, None] < b[None, :]))
        res["models"][name] = dict(summary=summ, per_task=per)
        print(f"[c_audit] {name}: " + json.dumps({k: round(v, 4) for k, v in summ.items() if "real_mean" in k or "auc" in k}), flush=True)
    B, L = res["models"]["pi05_base_libero"]["summary"], res["models"]["pi05_libero"]["summary"]
    res["readout"] = dict(
        base_l90_over_libero_l40=(B["train_real_mean"] + B["test_real_mean"]) / 2 / L["l40_real_mean"],
        base_l40_over_libero_l40=B["l40_real_mean"] / L["l40_real_mean"],
        base_l90_over_base_l40=(B["train_real_mean"] + B["test_real_mean"]) / 2 / B["l40_real_mean"],
        libero_l90_over_libero_l40=(L["train_real_mean"] + L["test_real_mean"]) / 2 / L["l40_real_mean"],
        base_test_minus_train=B["test_real_mean"] - B["train_real_mean"],
        base_part_minus_train=B["part_real_mean"] - B["train_real_mean"])
    res["seconds"] = time.time() - t0
    print(json.dumps(res["readout"], indent=1), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("c_audit", bench="libero_90_c", tag="pi05_base", seed=0,
                            metrics=dict(readout=res["readout"], summary={k: v["summary"] for k, v in res["models"].items()},
                                         groups={k: len(v) for k, v in g.items()}),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(demos=list(DEMOS), frames=FRAMES, draws=DRAWS), seconds=res["seconds"],
                            script="experiments/c_audit.py", command=" ".join(sys.argv), artifacts=[OUT]))


if __name__ == "__main__":
    if "--convert" in sys.argv:
        convert()
    if "--score" in sys.argv:
        score("--record" in sys.argv)
