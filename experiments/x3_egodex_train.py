"""X3 (EgoDex) stage 2c: fine-tune GR00T N1.5 on the EgoDex training subset (docs/EGODEX.md, "Recipe").

Data: the pre-extracted training episodes (<data>/derived/train/<folder>/<ep>.npz, x3_egodex_fetch.py).  These are exactly the
manifest's episodes (x3_egodex_manifest.py): the 90 training folders, none of the held-out G1 / G4 folders or the excluded ones,
and no basic_pick_place episode of a held-out pair (G1) or object (G2 / G4).  `--check-split` re-verifies that from the files.
A sample = (episode, t) with t a stored frame (even) and the whole chunk inside the episode (t + 2 * 16 <= T - 1), drawn
uniformly over all such samples.
  observation: the 224x224 frame t, the instruction, the state = 48-D hand frame at t (camera frame C_t);
  target:      the chunk [frame48(t + 2k) in C_t, k = 1..16], normalised by EgoDexNorm (q01 / q99 of the training samples,
               computed once by --stats, saved as norm.json), padded to 64; action_mask = the real dims whose keypoint has
               confidence > 0 (x3_egodex_data.action_mask48).
Model: backbone_groot_egodex.build_model (N1.5 base, 64-wide action layers re-initialised, Eagle frozen in bf16); GR00T's own
flow-matching loss (GR00TN15.forward), bf16 autocast; AdamW lr 1e-4 (betas 0.95 / 0.999, wd 1e-5), 5% warm-up, cosine to 0.
Checkpoints: <out>/ckpt.pt (head + optimizer + schedule + step, atomic) every --ckpt-every steps and at --max-minutes; the run
resumes from it.  At the end: <out>/final/{head.safetensors, norm.json, egodex_config.json} (a GrootEgoDexBackbone checkpoint).

    python -I experiments/x3_egodex_train.py --stats --out results/robot/egodex/train/run1        # once (CPU is enough)
    python -I experiments/x3_egodex_train.py --steps 200 --timing --out .../timing                 # the timing job
    python -I experiments/x3_egodex_train.py --steps 30000 --out results/robot/egodex/train/run1   # resumable
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3_egodex_data import DerivedEpisode, action_mask48  # noqa: E402

DATA = "/coc/flash10/ksingara3/robot/data/egodex"
H, STRIDE = 16, 2


def index(split="train", folders=None):
    """[(path, t)] of every valid sample; cached in <data>/derived/<split>_index.npz."""
    cache = os.path.join(DATA, "derived", f"{split}_index.json")
    files = sorted(glob.glob(os.path.join(DATA, "derived", split, "*", "*.npz")))
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d["n_files"] == len(files):
            return d["paths"], np.array(d["samples"], np.int64)
    paths, samples = [], []
    for i, p in enumerate(files):
        z = np.load(p)
        T, fidx = int(z["T"]), z["fidx"]
        paths.append(p)
        for t in fidx:
            if t + STRIDE * H <= T - 1:
                samples.append((i, int(t)))
    json.dump(dict(n_files=len(files), paths=paths, samples=samples), open(cache, "w"))
    return paths, np.array(samples, np.int64)


class EpisodeCache:
    def __init__(self, n=48):
        self.n, self.d = n, collections.OrderedDict()

    def get(self, p):
        if p in self.d:
            self.d.move_to_end(p)
            return self.d[p]
        e = DerivedEpisode(p)
        self.d[p] = e
        if len(self.d) > self.n:
            self.d.popitem(last=False)
        return e


class TrainSet:
    """torch-style dataset over (path, t) samples -> dict(image, instruction, state48, action48 [H, 48], mask [H, 64])."""

    def __init__(self, paths, samples, norm=None):
        self.paths, self.samples, self.norm = paths, samples, norm
        self.cache = EpisodeCache()

    def __len__(self):
        return len(self.samples)

    def raw(self, i, image=True):
        pi, t = self.samples[i]
        ep = self.cache.get(self.paths[pi])
        s, a, pad, conf = ep.sample(int(t), H, STRIDE)
        assert not pad.any()
        return dict(image=ep.image(int(t)) if image else None, instruction=ep.instruction, state=s, action=a,
                    mask=action_mask48(conf, H=H), path=self.paths[pi], t=int(t))

    def __getitem__(self, i):
        r = self.raw(i)
        r["state"] = self.norm.normalize(r["state"])
        r["action"] = self.norm.normalize(r["action"])
        return r


class Collate:
    def __init__(self):
        self.proc = None

    def __call__(self, rows):
        import torch
        from lifelong_ood.policy.backbone_groot_egodex import A_FULL, A_REAL, EMBODIMENT, eagle_batch, eagle_content, eagle_processor, pad_state
        if self.proc is None:
            self.proc = eagle_processor()
        b = eagle_batch(self.proc, [eagle_content(self.proc, r["image"], r["instruction"]) for r in rows])
        st, sm = pad_state(np.stack([r["state"] for r in rows]))
        act = torch.zeros(len(rows), H, A_FULL)
        act[:, :, :A_REAL] = torch.as_tensor(np.stack([r["action"] for r in rows]))
        mask = torch.as_tensor(np.stack([r["mask"] for r in rows]))
        b.update(state=st, state_mask=sm, action=act, action_mask=mask,
                 embodiment_id=torch.full((len(rows),), EMBODIMENT, dtype=torch.long))
        return b


def compute_stats(out, n=40000, seed=0):
    from lifelong_ood.policy.backbone_groot_egodex import EgoDexNorm
    paths, samples = index("train")
    rng = np.random.default_rng(seed)
    ds = TrainSet(paths, samples)
    pick = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    pick.sort()                                               # episode-ordered: the cache hits
    S, A = [], []
    for i in pick:
        r = ds.raw(int(i), image=False)
        S.append(r["state"])
        A.append(r["action"])
    S, A = np.stack(S), np.concatenate(A)
    X = np.concatenate([S, A])
    q01, q99 = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)
    norm = EgoDexNorm(q01, q99)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(norm.to_json(), n_samples=int(len(pick)), n_train_samples=int(len(samples)), n_episodes=len(paths),
                   state_q01=np.quantile(S, 0.01, 0).tolist(), state_q99=np.quantile(S, 0.99, 0).tolist()),
              open(os.path.join(out, "norm.json"), "w"), indent=1)
    print("stats:", len(paths), "episodes", len(samples), "samples; q01..q99 of right wrist xyz",
          np.round(q01[24:27], 3), np.round(q99[24:27], 3))


def check_split():
    """Every training file belongs to a training folder; no basic_pick_place file is a held-out pair / object."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from x3_egodex_split import BPP_HELD_OBJECTS, BPP_HELD_PAIRS, EXCLUDE_FROM_TRAIN, HELD
    prop = json.load(open(os.path.join(ROOT, "results/robot/egodex/split_proposal.json")))
    held = {f for g in HELD.values() for f in g} | EXCLUDE_FROM_TRAIN
    files = glob.glob(os.path.join(DATA, "derived", "train", "*", "*.npz"))
    folders = collections.Counter(os.path.basename(os.path.dirname(p)) for p in files)
    bad = sorted(set(folders) & held)
    assert not bad, bad
    assert set(folders) <= set(prop["train"]), sorted(set(folders) - set(prop["train"]))
    man = {(r["folder"], r["ep"]) for s in glob.glob(os.path.join(ROOT, "results/robot/egodex/fetch/shard*.json")) for r in json.load(open(s))}
    extra = [p for p in files if (os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-4]) not in man]
    assert not extra, extra[:5]
    return dict(files=len(files), folders=len(folders), held_pairs=BPP_HELD_PAIRS, held_objects=BPP_HELD_OBJECTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robot/egodex/train/run1")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--norm", default=None, help="norm.json (default <out>/norm.json)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-minutes", type=float, default=225)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU: one collated batch, shapes and mask fractions")
    a = ap.parse_args()
    if a.stats:
        compute_stats(a.out)
        return
    if a.smoke:
        from lifelong_ood.policy.backbone_groot_egodex import EgoDexNorm
        norm = EgoDexNorm.from_json(json.load(open(a.norm or os.path.join(a.out, "norm.json"))))
        paths, samples = index("train")
        ds = TrainSet(paths, samples, norm)
        t0 = time.time()
        b = Collate()([ds[int(i)] for i in np.random.default_rng(0).choice(len(ds), 8)])
        print({k: (tuple(v.shape), str(v.dtype)) if hasattr(v, "shape") else type(v) for k, v in b.items()})
        print("collate 8 in %.2fs; action range %.2f..%.2f; mask frac %.3f; state range %.2f..%.2f" % (
            time.time() - t0, float(b["action"][..., :48].min()), float(b["action"][..., :48].max()),
            float(b["action_mask"][..., :48].float().mean()), float(b["state"].min()), float(b["state"].max())))
        print("split check", check_split())
        return
    import torch
    from lifelong_ood.policy.backbone_groot_egodex import BASE, EgoDexNorm, build_model
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    split = check_split()
    norm_path = a.norm or os.path.join(a.out, "norm.json")
    norm = EgoDexNorm.from_json(json.load(open(norm_path)))
    paths, samples = index("train")
    print("train:", split, len(samples), "samples", flush=True)
    gm, info = build_model(device="cuda", trainable=True)
    gm.train()
    gm.backbone.eval()
    params = [p for p in gm.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-5)
    warm = max(1, int(a.warmup * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, a.steps - warm)))))
    step, ck = 0, os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck) and not a.timing:
        st = torch.load(ck, map_location="cuda", weights_only=False)
        gm.action_head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = int(st["step"])
        print("resumed at step", step, flush=True)

    def save():
        tmp = ck + ".tmp"
        torch.save(dict(head=gm.action_head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        args=vars(a)), tmp)
        os.replace(tmp, ck)

    ds = TrainSet(paths, samples, norm)
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=10 ** 9, generator=g)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers, collate_fn=Collate(),
                                     persistent_workers=True, prefetch_factor=4)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, tl, losses, times, data_t = time.time(), time.time(), [], [], []
    it = iter(dl)
    while step < a.steps:
        td = time.time()
        b = next(it)
        data_t.append(time.time() - td)
        ts = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = gm.forward(b)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        times.append(time.time() - ts)
        losses.append(float(loss))
        step += 1
        if step % 50 == 0 or step == a.steps:
            rec = dict(step=step, loss=float(np.mean(losses[-50:])), lr=sched.get_last_lr()[0], grad_norm=float(gn),
                       s_per_step=float(np.mean(times[-50:])), data_wait=float(np.mean(data_t[-50:])),
                       mem_gb=torch.cuda.max_memory_allocated() / 1e9, elapsed=time.time() - t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if not a.timing and (step % a.ckpt_every == 0):
            save()
        if (time.time() - t_start) / 60 > a.max_minutes:
            if not a.timing:
                save()
            print("TIME LIMIT at step", step, flush=True)
            sys.exit(3)                                   # the wrapper requeues on 3
    res = dict(steps=step, trainable_params=int(n_tr), s_per_step=float(np.mean(times[10:])) if len(times) > 10 else None,
               data_wait=float(np.mean(data_t[10:])) if len(data_t) > 10 else None, final_loss=float(np.mean(losses[-200:])),
               mem_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=torch.cuda.get_device_name(0), samples=int(len(samples)),
               episodes=len(paths), build=info)
    if a.timing:
        res["gpu_hours_per_30k"] = res["s_per_step"] * 30000 / 3600 if res["s_per_step"] else None
        json.dump(res, open(os.path.join(a.out, "timing.json"), "w"), indent=1, default=str)
        print("TIMING", res, flush=True)
    else:
        save()
    from safetensors.torch import save_file
    fin = os.path.join(a.out, "final")
    os.makedirs(fin, exist_ok=True)
    save_file({k: v.detach().float().contiguous().cpu() for k, v in gm.action_head.state_dict().items()}, os.path.join(fin, "head.safetensors"))
    json.dump(norm.to_json(), open(os.path.join(fin, "norm.json"), "w"), indent=1)
    json.dump(dict(base=BASE, H=H, stride=STRIDE, action_dim=64, real=48, embodiment=31, steps=step, batch=a.batch, lr=a.lr,
                   train_episodes=len(paths), train_samples=int(len(samples)), split=split),
              open(os.path.join(fin, "egodex_config.json"), "w"), indent=1)
    if a.timing:
        return
    json.dump(res, open(os.path.join(a.out, "train_result.json"), "w"), indent=1, default=str)
    print("DONE", res, flush=True)
    if a.record:
        from lifelong_ood import ledger
        tim = json.load(open(os.path.join(os.path.dirname(a.out.rstrip("/")), "timing", "timing.json")))
        p = ledger.record("x3_egodex_train", bench="egodex_train_subset", tag=os.path.basename(a.out.rstrip("/")), seed=a.seed,
                          metrics=dict(final_loss=res["final_loss"], steps=step, s_per_step=res["s_per_step"],
                                       gpu_hours=(res["s_per_step"] or 0) * step / 3600, mem_gb=res["mem_gb"],
                                       timing_s_per_step=tim["s_per_step"], timing_data_wait=tim["data_wait"]),
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0,
                                     head_bytes=int(os.path.getsize(os.path.join(fin, "head.safetensors")))),
                          config=dict(vars(a), trainable_params=int(n_tr), train_episodes=len(paths), train_samples=int(len(samples)),
                                      split=split, build=info),
                          script="experiments/x3_egodex_train.py", command=" ".join(sys.argv),
                          artifacts=dict(final=fin, log=os.path.join(a.out, "log.jsonl")))
        print("ledger:", p)


if __name__ == "__main__":
    main()
