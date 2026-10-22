"""X3 (EgoDex) stage 2d/2e: the backbone check and the parity gate of the EgoDex fine-tune of GR00T N1.5.

--check: GrootEgoDexBackbone against the model's own sampler (GR00TN15.get_action) on the same initial noise, 3 test frames:
  max |difference| of the chunks (normalised and in metres); text_span decodes to the instruction; cond_hook sees every encoder
  layer once; a zero shift leaves the prefix unchanged; the flow loss at the ground-truth chunk (t = 0.5); seconds per query.
--parity (default): EgoDex's best-of-K metric on the G0 test episodes, i.e. the official test episodes of the 90 training
  folders, minus basic_pick_place test episodes of a held-out pair or object.  At every stored frame t that is a multiple of
  --every (30 Hz frames) with t + 30 <= T - 1: K samples of the 16-step chunk (frames t + 2 .. t + 32 in C_t); the hand frame at
  offsets 1..30 is linearly interpolated from offsets {0 (the observed state), 2, 4, ..., 32}, exact at even offsets; the
  ground truth is frame48(t + 1 .. t + 30) in C_t.  The metric is compute_metrics.evaluate_distance re-implemented line for line
  (12 points: 2 wrists + 10 fingertips; average over the 30 steps and the points; final = the last step; best of the first k
  samples; mean over data points), K = 1, 5, 10.  Reference: the EgoDex paper's EncDec + FM at H = 30, K = 10: 0.026 / 0.033 m.
  Baselines on the same points: "hold" (the current hand frame repeated) and our own chunk scored at even offsets only.

    python -I experiments/x3_egodex_parity.py --check --ckpt results/robot/egodex/train/run1/final
    python -I experiments/x3_egodex_parity.py --ckpt results/robot/egodex/train/run1/final --every 8 --record
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3_egodex_data import DerivedEpisode  # noqa: E402

DATA = "/coc/flash10/ksingara3/robot/data/egodex"
PTS = list(range(3)) + list(range(9, 27)) + list(range(33, 48))      # compute_metrics: 12 positions, rotations skipped
PAPER_H30_K10 = dict(avg=0.026, final=0.033)


def dists(gt, pred):
    """gt [T, 48], pred [K, T, 48] -> (avg [K], final [K]) as compute_metrics.evaluate_distance."""
    p1 = gt[:, PTS].reshape(gt.shape[0], 12, 3)
    p2 = pred[:, :, PTS].reshape(pred.shape[0], pred.shape[1], 12, 3)
    d = np.linalg.norm(p1[None] - p2, axis=-1)                     # [K, T, 12]
    return d.mean(axis=(1, 2)), d[:, -1, :].mean(axis=1)


def upsample(state, chunk, n=30, stride=2):
    """state [48], chunk [K, 16, 48] at offsets 2..32 -> [K, n, 48] at offsets 1..n (linear)."""
    K, Hh, D = chunk.shape
    xs = np.arange(0, stride * Hh + 1, stride)                     # 0, 2, ..., 32
    full = np.concatenate([np.repeat(state[None, None], K, 0), chunk], 1)   # [K, 17, 48]
    q = np.arange(1, n + 1)
    i = np.clip(np.searchsorted(xs, q) - 1, 0, len(xs) - 2)
    w = ((q - xs[i]) / (xs[i + 1] - xs[i]))[None, :, None]
    return full[:, i] * (1 - w) + full[:, i + 1] * w


def g0_episodes():
    from x3_egodex_split import BPP_HELD_OBJECTS, BPP_HELD_PAIRS, _ot
    prop = json.load(open(os.path.join(ROOT, "results/robot/egodex/split_proposal.json")))
    inv = json.load(open(os.path.join(ROOT, "results/robot/egodex/test_inventory.json")))["tasks"]
    hp, ho = {tuple(p) for p in BPP_HELD_PAIRS}, set(BPP_HELD_OBJECTS)
    out, dropped = [], 0
    for fo in sorted(prop["train"]):
        for e in inv.get(fo, {}).get("episodes_list", []):
            if fo == "basic_pick_place":
                o, t = _ot(e)
                if (o, t) in hp or o in ho or o is None or t is None:
                    dropped += 1
                    continue
            p = os.path.join(DATA, "derived", "test", fo, f"{e['ep']}.npz")
            if os.path.exists(p):
                out.append((fo, p))
    return out, dropped


def obs_of(ep, t):
    return dict(image=ep.image(t), state=ep.frame48(np.array([t]), t)[0], prompt=ep.instruction)


def check(bk, out):
    import torch
    eps, _ = g0_episodes()
    firsts = {}
    for fo, p in eps:                                         # the first G0 episode of three different folders
        firsts.setdefault(fo, p)
    picks = [firsts[f] for f in sorted(firsts)[::max(1, len(firsts) // 3)]][:3]
    res = []
    for p in picks:
        ep = DerivedEpisode(p)
        t = int(ep.fidx[min(10, len(ep.fidx) - 1)])
        o = obs_of(ep, t)
        rec = dict(path=p, t=t, prompt=o["prompt"])
        # the SAME initial noise in both samplers: get_action draws its noise with torch.randn(size, dtype=vl_embs.dtype), so the
        # draw is intercepted and replaced by ours (cast to the dtype it asks for, which is recorded)
        noise = torch.randn((1, bk.H, bk.A), generator=torch.Generator().manual_seed(123)).to(bk.device)
        raw = bk.raw_inputs([o])
        _randn, used = torch.randn, {}

        def _fixed(*args, **kw):
            size = tuple(kw.get("size", args[0] if args else ()))
            if size == tuple(noise.shape):
                used["dtype"] = str(kw.get("dtype"))
                used["noise"] = noise.to(dtype=kw.get("dtype") or noise.dtype, device=kw.get("device") or noise.device)
                return used["noise"].clone()
            return _randn(*args, **kw)
        torch.randn = _fixed
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = bk.gm.get_action(raw)["action_pred"].float()[0, :, :bk.a].cpu().numpy()
        finally:
            torch.randn = _randn
        assert "noise" in used, "get_action did not draw its noise through torch.randn(size=...)"
        rec["get_action_noise_dtype"] = used["dtype"]
        n0 = used["noise"].float()                                  # exactly the values get_action started from
        ob = bk.observation(o)
        t0 = time.time()
        pre = bk.prefix(ob)
        rows = bk.sample(1, pre, noise=n0.reshape(1, -1).cpu().numpy())
        torch.cuda.synchronize()
        rec["secs_per_query"] = time.time() - t0
        ours = rows.reshape(bk.H, bk.a)
        rec["parity_max_abs_norm"] = float(np.abs(ours - ref).max())
        # sensitivity: our sampler from a DIFFERENT noise against get_action (how far apart two noises put the chunks)
        other = torch.randn((1, bk.H, bk.A), generator=torch.Generator().manual_seed(999)).reshape(1, -1).numpy()
        rec["other_noise_max_abs_norm"] = float(np.abs(bk.sample(1, pre, noise=other).reshape(bk.H, bk.a) - ref).max())
        # the same Euler steps with the action state kept in get_action's dtype: isolates float32-vs-bf16 accumulation
        with torch.no_grad():
            x, N = used["noise"].clone(), bk.num_steps
            for k in range(N):
                v = bk.v_full(x.float().reshape(1, -1), torch.full((1,), 1.0 - k / N, device=bk.device), pre)
                x = x + (1.0 / N) * (-v).reshape(x.shape).to(x.dtype)
        rec["parity_same_dtype_state_max_abs_norm"] = float(np.abs(x.float()[0, :, :bk.a].cpu().numpy() - ref).max())
        rec["parity_max_abs_m"] = float(np.abs(bk.unnormalize(rows)[0] - bk._codec.unnormalize(ref))[:, [i for i in range(48) if i in PTS]].max())
        gt = ep.frame48(t + 2 * np.arange(1, bk.H + 1), t)
        rec["gt_dist_ours_m"] = float(dists(gt, bk.unnormalize(rows))[0][0])
        off, nv = bk.text_span(ob)
        rec["span_text"] = bk._tok.decode(ob.eagle["eagle_input_ids"][0].tolist()[off:off + int(nv[0])])
        seen = {}

        def f(li, h, span):
            seen[li] = h.shape[1]
            return None
        hs = bk.cond_hook(f)
        try:
            pre2 = bk.prefix(ob)
        finally:
            for h in hs:
                h.remove()
        rec["cond_layers"], rec["hook_layers_seen"] = len(bk.cond_layers()), len(seen)
        hs = bk.cond_hook(lambda li, h, span: h.clone())
        try:
            pre3 = bk.prefix(ob)
        finally:
            for h in hs:
                h.remove()
        rec["zero_shift_max_abs"] = float((pre3.kv["vl"].float() - pre2.kv["vl"].float()).abs().max())
        a = np.zeros((1, bk.H, bk.A), np.float32)
        a[0, :, :bk.a] = bk._codec.normalize(gt)
        a = torch.as_tensor(a, device=bk.device)
        eps_ = torch.randn(a.shape, generator=torch.Generator().manual_seed(0)).to(bk.device)
        tt = torch.full((1,), 0.5, device=bk.device)
        xt = 0.5 * eps_ + 0.5 * a
        with torch.no_grad():
            v = bk.v_full(xt, tt, pre)
        rec["flow_loss_t05"] = float(bk.flow_loss(v, (eps_ - a).reshape(1, -1), xt, tt))
        print(json.dumps(rec), flush=True)
        res.append(rec)
    json.dump(dict(backbone=bk.describe(), build=bk.build_info, frames=res), open(out, "w"), indent=1, default=str)
    return res


def parity(bk, a):
    import torch
    eps, dropped = g0_episodes()
    Ks = (1, 5, 10)
    K = max(Ks)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    t0, npts, batch = time.time(), 0, []
    # resumable: per-folder results are saved after each folder (<out>.partial.json); a restart skips the finished folders.
    # Seeds depend only on (folder index, batch index within the folder), so a resumed run draws the same noise.
    part = a.partial
    done = {}
    if part and os.path.exists(part):
        done = json.load(open(part))
        for fo, r in done.items():
            per[fo] = collections.defaultdict(list, {k: list(v) for k, v in r.items()})
        print(f"[parity] resumed: {len(done)} folders done", flush=True)
    folders = sorted({fo for fo, _ in eps})

    def flush(batch, fi, bi):
        obs = [b[2] for b in batch]
        ob = bk.observations(obs)
        pre = bk.repeat(bk.prefix(ob), K)
        rows = bk.sample(len(obs) * K, pre, seed=a.seed * 1000003 + fi * 100003 + bi)
        ch = bk.unnormalize(rows).reshape(len(obs), K, bk.H, bk.a)
        for (fo, gt, o), c in zip(batch, ch):
            pred = upsample(o["state"], c)                          # [K, 30, 48]
            av, fi_ = dists(gt, pred)
            hav, hfi = dists(gt, np.repeat(o["state"][None, None], 30, 1))
            even = np.arange(1, 30, 2)                               # offsets 2, 4, ..., 30
            eav, _ = dists(gt[even], pred[:, even])
            r = per[fo]
            for k in Ks:
                r[f"avg@{k}"].append(float(av[:k].min()))
                r[f"final@{k}"].append(float(fi_[:k].min()))
            r["hold_avg"].append(float(hav[0]))
            r["hold_final"].append(float(hfi[0]))
            r["avg_even@10"].append(float(eav.min()))

    by_fo = collections.defaultdict(list)
    for fo, p in eps:
        by_fo[fo].append(p)
    for fi, fo in enumerate(folders):
        if fo in done:
            npts += len(per[fo]["avg@1"])
            continue
        batch, bi = [], 0
        for p in by_fo[fo]:
            ep = DerivedEpisode(p)
            for t in ep.fidx:
                t = int(t)
                if t % a.every or t + 30 > ep.T - 1:
                    continue
                gt = ep.frame48(t + np.arange(1, 31), t)
                batch.append((fo, gt, obs_of(ep, t)))
                npts += 1
                if len(batch) == a.batch:
                    flush(batch, fi, bi)
                    batch, bi = [], bi + 1
        if batch:
            flush(batch, fi, bi)
        if part:
            done[fo] = {k: v for k, v in per[fo].items()}
            tmp = part + ".tmp"
            json.dump(done, open(tmp, "w"))
            os.replace(tmp, part)
        print(f"[parity] {fi + 1}/{len(folders)} {fo}: {len(per[fo]['avg@1'])} points, {npts} total, {time.time() - t0:.0f}s",
              flush=True)
        if a.max_points and npts >= a.max_points:
            break
    allv = collections.defaultdict(list)
    for fo, r in per.items():
        for k, v in r.items():
            allv[k] += v
    summ = {k: float(np.mean(v)) for k, v in allv.items()}
    by_folder = {fo: dict(n=len(r["avg@1"]), **{k: float(np.mean(v)) for k, v in r.items()}) for fo, r in per.items()}
    # folder-balanced mean (each folder's points weigh equally), as a check on the pooled number
    fb = {k: float(np.mean([v[k] for v in by_folder.values()])) for k in summ}
    return dict(points=npts, episodes=len(eps), folders=len(per), bpp_held_test_dropped=dropped, every=a.every, K=Ks,
                pooled=summ, folder_balanced=fb, paper_encdec_fm_h30_k10=PAPER_H30_K10, by_folder=by_folder,
                seconds=time.time() - t0)


def export_ckpt(pt):
    """A training checkpoint (<run>/ckpt.pt) -> <run>/step<N>/ in the final/ layout (head.safetensors, norm.json,
    egodex_config.json), for a parity read of an intermediate step.  -> that directory."""
    import torch
    from safetensors.torch import save_file
    from lifelong_ood.policy.backbone_groot_egodex import BASE
    st = torch.load(pt, map_location="cpu", weights_only=False)
    run = os.path.dirname(os.path.abspath(pt))
    d = os.path.join(run, f"step{int(st['step'])}")
    os.makedirs(d, exist_ok=True)
    save_file({k: v.detach().float().contiguous() for k, v in st["head"].items()}, os.path.join(d, "head.safetensors"))
    json.dump(json.load(open(os.path.join(run, "norm.json"))), open(os.path.join(d, "norm.json"), "w"), indent=1)
    json.dump(dict(base=BASE, H=16, stride=2, action_dim=64, real=48, embodiment=31, steps=int(st["step"]),
                   batch=st["args"]["batch"], lr=st["args"]["lr"], from_ckpt=pt), open(os.path.join(d, "egodex_config.json"), "w"), indent=1)
    print("exported", pt, "->", d, flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", default=None, help="resume file (default <out>.partial.json for the full parity run)")
    a = ap.parse_args()
    from lifelong_ood.policy.backbone_groot_egodex import GrootEgoDexBackbone
    if a.ckpt.endswith(".pt"):
        a.ckpt = export_ckpt(a.ckpt)
    bk = GrootEgoDexBackbone(a.ckpt)
    print(bk.describe(), flush=True)
    name = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    if os.path.basename(os.path.abspath(a.ckpt)) != "final":               # an intermediate step: run1_step14000
        name = f"{name}_{os.path.basename(os.path.abspath(a.ckpt))}"
    if a.check:
        res = check(bk, a.out or os.path.join(ROOT, f"results/robot/egodex/backbone_check_{name}.json"))
        metrics = dict(parity_max_abs_norm=max(r["parity_max_abs_norm"] for r in res),
                       parity_max_abs_m=max(r["parity_max_abs_m"] for r in res),
                       parity_same_dtype_state_max_abs_norm=max(r["parity_same_dtype_state_max_abs_norm"] for r in res),
                       other_noise_max_abs_norm=min(r["other_noise_max_abs_norm"] for r in res),
                       get_action_noise_dtype=sorted({r["get_action_noise_dtype"] for r in res}),
                       secs_per_query=float(np.mean([r["secs_per_query"] for r in res])),
                       span_ok=all(r["span_text"].strip().strip("'\"[]") in r["prompt"] or r["prompt"] in r["span_text"] for r in res),
                       hook_layers=[r["hook_layers_seen"] for r in res], zero_shift=max(r["zero_shift_max_abs"] for r in res),
                       flow_loss_t05=[r["flow_loss_t05"] for r in res])
        tag, exp = "backbone_vs_get_action", "x3_egodex_parity"
    else:
        out = a.out or os.path.join(ROOT, f"results/robot/egodex/parity_{name}_every{a.every}.json")
        if a.partial is None and not a.max_points:
            a.partial = out + ".partial.json"
        res = parity(bk, a)
        json.dump(res, open(out, "w"), indent=1)
        metrics = {k: v for k, v in res.items() if k != "by_folder"}
        print(json.dumps(metrics, indent=1), flush=True)
        tag, exp = "g0_best_of_k", "x3_egodex_parity"
    if a.record:
        from lifelong_ood import ledger
        p = ledger.record(exp, bench="egodex_test_g0", tag=tag, seed=a.seed, metrics=metrics,
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=dict(ckpt=a.ckpt, every=a.every, batch=a.batch), script="experiments/x3_egodex_parity.py",
                          command=" ".join(sys.argv), backbone=bk.describe(), keys=dict(check=tag, run=name))
        print("ledger:", p)


if __name__ == "__main__":
    main()
