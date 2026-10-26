"""PLAN_SOLVE S0 (docs/PLAN_S0.md): train the binding adapter ONCE on the training cells' demos (s0_data.py), the policy frozen,
by the policy's own flow-matching loss (openpi's: t ~ Beta(1.5, 1), x_t = t eps + (1 - t) a, mean (v - u)^2 over the padded chunk).
openpi venv, GPU; checkpoints every CKPT_EVERY steps and resumes (overcap --requeue).

Examples (frames with frame % 15 != 0; those are the loss-analysis split):
  own   the cell's own prompt + its own oracle role masks (named instance, target region / body)   -> its demo chunk
  cf    a SAME-SCENE training cell's prompt (a different target; libero_90 / libero_10 / libero_goal scenes, whose layouts are
        shared) + THIS cell's oracle masks                                                          -> its demo chunk
cf makes the binding input matter: the prompt names another binding, the masks and the demo agree.  P_CF of each batch is cf.
AdamW lr LR, STEPS steps, batch BATCH, seed 0; the prefix is computed without gradient, the adapter's K/V edit and the action expert
carry the gradient (expert weights frozen).

    .venv/bin/python experiments/s0_train.py [--steps N] [--smoke]  -> results/bind/s0/adapter.pt, train_log.json
    .venv/bin/python experiments/s0_train.py --analyse              -> results/bind/s0/loss_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
OD = os.path.join(ROOT, "results", "bind", "s0")
STEPS, BATCH, LR, WD, P_CF, CKPT_EVERY, SEED, VAL_MOD = 3000, 16, 3e-4, 0.01, 0.5, 250, 0, 15
CF_SUITES = ("libero_90", "libero_10", "libero_goal")


def role_masks(z, i):
    names = [str(x) for x in z["body_names"]] + [str(x) for x in z["site_names"]]
    cov = np.concatenate([z["body_cov"][i], z["site_cov"][i]]).astype(np.float32)
    return cov[names.index(str(z["named"]))], cov[names.index(str(z["target"]))]


def load_data():
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    cells = {}
    for f in sorted(glob.glob(os.path.join(OD, "data", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        tid = str(z["tid"])
        mo, mg = zip(*[role_masks(z, i) for i in range(len(z["frames"]))])
        cells[tid] = dict(tid=tid, scene=tasks[tid]["scene"], prompt=str(z["prompt"]), frames=z["frames"], img=z["img"],
                          wimg=z["wimg"], state8=z["state8"], chunk=z["chunk"], mo=np.stack(mo), mg=np.stack(mg),
                          names=[str(x) for x in z["body_names"]] + [str(x) for x in z["site_names"]],
                          cov=np.concatenate([z["body_cov"], z["site_cov"]], 1))
    # counterfactual prompts: every same-scene TRAINING cell's instruction with a different goal (any verb)
    for tid, c in cells.items():
        c["cf"] = sorted({tasks[t]["lang"] for t, x in tasks.items() if x["set"] == "train" and t != tid and x["scene"] == c["scene"]
                          and tid.split(":")[0] in CF_SUITES and t.split(":")[0] in CF_SUITES and x["goal"] != tasks[tid]["goal"]
                          and x["lang"] != c["prompt"]})
    return cells


def main():
    import torch
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.adapt import _flow_batch
    from lifelong_ood.policy.backbone import PolicyBackbone
    import s0_adapter as SA
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    cells = load_data()
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None)
    norm = D.Norm.for_backbone(bk)
    dev = bk.device
    items = [(t, i) for t, c in cells.items() for i, f in enumerate(c["frames"]) if f % VAL_MOD != 0]
    val = [(t, i) for t, c in cells.items() for i, f in enumerate(c["frames"]) if f % VAL_MOD == 0]
    print(f"[s0 train] {len(cells)} cells, {len(items)} train frames, {len(val)} analysis frames; cf prompts per cell "
          f"{ {t: len(c['cf']) for t, c in cells.items()} }", flush=True)

    def obs(rows):
        return bk.observations([{"observation/image": cells[t]["img"][i], "observation/wrist_image": cells[t]["wimg"][i],
                                 "observation/state": cells[t]["state8"][i], "prompt": p} for t, i, p in rows])

    def chunks(rows):
        c = np.stack([norm.normalize(cells[t]["chunk"][i]) for t, i, _ in rows])            # [B, H, 7]
        return torch.tensor(D.pad_rows(c.reshape(len(rows), -1), bk.H), device=dev)

    def masks(rows, mode="own"):
        mo = np.stack([cells[t]["mo"][i] for t, i, _ in rows])
        mg = np.stack([cells[t]["mg"][i] for t, i, _ in rows])
        if mode == "perm":                                                                     # token-permuted (shuffled) masks
            r = np.random.default_rng(1)
            p = r.permutation(256)
            mo, mg = mo[:, p], mg[:, p]
        elif mode == "zero":
            mo, mg = np.zeros_like(mo), np.zeros_like(mg)
        return torch.tensor(mo, device=dev), torch.tensor(mg, device=dev)
    ad = SA.BindAdapter().to(dev)
    opt = torch.optim.AdamW(ad.parameters(), lr=LR, weight_decay=WD)
    ck = os.path.join(OD, "ckpt.pt")
    step, hist = 0, []
    if a.analyse:
        ad.load_state_dict(torch.load(os.path.join(OD, "adapter.pt"), map_location=dev))
    elif os.path.exists(ck) and not a.smoke:
        s = torch.load(ck, map_location=dev)
        ad.load_state_dict(s["adapter"])
        opt.load_state_dict(s["opt"])
        step, hist = s["step"], s["hist"]
        print(f"[s0 train] resumed at step {step}", flush=True)
    rng = np.random.default_rng(SEED + step)
    gen = torch.Generator(device="cpu").manual_seed(SEED + step)
    steps = 3 if a.smoke else a.steps
    t0 = time.time()
    while not a.analyse and step < steps:
        pick = [items[j] for j in rng.integers(0, len(items), BATCH)]
        rows = []
        for t, i in pick:
            cf = cells[t]["cf"]
            p = cf[rng.integers(len(cf))] if cf and rng.random() < P_CF else cells[t]["prompt"]
            rows.append((t, i, p))
        with torch.no_grad():
            pre = bk.prefix(obs(rows))
        A = chunks(rows)
        mo, mg = masks(rows)
        pe = ad.apply(pre, mo, mg)
        tt, eps = _flow_batch(len(rows), bk.H, bk.A, gen, dev)
        xt = tt[:, None, None] * eps + (1 - tt[:, None, None]) * A
        u = (eps - A).reshape(len(rows), -1)
        loss = ((bk.v_full(xt, tt, pe) - u) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        hist.append(float(loss))
        if step % 25 == 0 or a.smoke:
            print(f"[s0 train] step {step}: loss {np.mean(hist[-25:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % CKPT_EVERY == 0 and not a.smoke:
            torch.save(dict(adapter=ad.state_dict(), opt=opt.state_dict(), step=step, hist=hist), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if not a.analyse:
        if not a.smoke:
            torch.save(ad.state_dict(), os.path.join(OD, "adapter.pt"))
        json.dump(dict(steps=step, loss=hist, bytes=ad.nbytes(), n_train=len(items)), open(os.path.join(OD, "train_log.json" if not a.smoke
                                                                                                         else "train_log_smoke.json"), "w"))
        print(f"[s0 train] done: {step} steps, adapter {ad.nbytes() / 1e6:.1f} MB fp16", flush=True)
        if a.smoke:
            return
    # loss analysis on the held-out frames (fixed noise and times): does the adapter make the policy follow the MASK?
    ad.eval()
    g2 = torch.Generator(device="cpu").manual_seed(123)
    res = {k: [] for k in ("own_base", "own_ad", "cf_base", "cf_ad", "cf_ad_perm", "cf_ad_partner", "cf_ad_partner_own", "own_ad_zero")}
    for g0 in range(0, len(val), BATCH):
        vb = val[g0:g0 + BATCH]
        cfr = [(t, i, cells[t]["cf"][0]) for t, i in vb if cells[t]["cf"]]
        sets = dict(own=[(t, i, cells[t]["prompt"]) for t, i in vb], cf=cfr)
        for name, rows in sets.items():
            if not rows:
                continue
            A = chunks(rows)
            tt, eps = _flow_batch(len(rows), bk.H, bk.A, g2, dev)
            xt = tt[:, None, None] * eps + (1 - tt[:, None, None]) * A
            u = (eps - A).reshape(len(rows), -1)
            with torch.no_grad():
                pre = bk.prefix(obs(rows))
                L = lambda p: ((bk.v_full(xt, tt, p) - u) ** 2).mean(-1).cpu().numpy()          # noqa: E731
                res[f"{name}_base"] += list(L(pre))
                res[f"{name}_ad"] += list(L(ad.apply(pre, *masks(rows))))
                if name == "own":
                    res["own_ad_zero"] += list(L(ad.apply(pre, *masks(rows, "zero"))))
                else:
                    res["cf_ad_perm"] += list(L(ad.apply(pre, *masks(rows, "perm"))))
                    # partner masks: the roles of the cell whose prompt is used, read in THIS frame (same scene): masks that agree
                    # with the prompt and disagree with the demo -> the loss should rise if the policy follows the masks
                    pr, pmo, pmg = [], [], []
                    for (t, i, p) in rows:
                        q = next((x for x in tasks.values() if x["set"] == "train" and x["lang"] == p and x["scene"] == cells[t]["scene"]), None)
                        at = next((g for g in (q or {}).get("goal", []) if g[0] in ("In", "On") and len(g) == 3), None)
                        nm = cells[t]["names"]
                        if at and at[1] in nm and at[2] in nm:
                            pr.append((t, i, p))
                            pmo.append(cells[t]["cov"][i][nm.index(at[1])].astype(np.float32))
                            pmg.append(cells[t]["cov"][i][nm.index(at[2])].astype(np.float32))
                    if pr:
                        ix = [rows.index(r) for r in pr]
                        pre_p = bk.prefix(obs(pr))
                        Lp = ((bk.v_full(xt[ix], tt[ix], ad.apply(pre_p, torch.tensor(np.stack(pmo), device=dev),
                                                                    torch.tensor(np.stack(pmg), device=dev))) - u[ix]) ** 2).mean(-1)
                        res["cf_ad_partner"] += list(Lp.cpu().numpy())
                        res["cf_ad_partner_own"] += list(L(ad.apply(pre, *masks(rows)))[ix])
    out = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in res.items() if v}
    json.dump(out, open(os.path.join(OD, "loss_analysis.json"), "w"), indent=1)
    print(f"[s0 analyse] {json.dumps(out)}", flush=True)


if __name__ == "__main__":
    main()
