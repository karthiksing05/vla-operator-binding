"""P2 / P3 development pre-checks, step 1 (GPU): every observation's keys, modes, extents and cached fields.

For each observation of results/protos/index.json (experiments/pp_prep.py):
  conditions   ~13 (null, instruction, object / goal / verb parts, their variants; protos.store.conditions_for), one batched
               prefix pass with per-row word splices (protos.store.RowSplice)
  starts       4 draft chunks (2 under the instruction, 2 under null) noised to each level, SHARED by every condition
  modes        mode ascent (backbones.score_ops, Adam, 80 iterations, lr 0.02) at the 6 levels, Tweedie x0, Laplace extent
               (4 probes); the x0 modes unnormalised by the backbone's own output transform
  fields       (train / l90 only) E[eps | x] of every condition and of the 40 training instructions at 4 points per level on the
               flow path of the demonstration's own chunk (protos.store.demo_points), with the true noise: R0 / R1 / Comp are fit
               and scored on these
-> results/protos/obs/<key>.npz.  Resumable (existing outputs are skipped), sharded (--shard i --nshards n).

    sbatch ... --export=ALL,ENV=openpi,SCRIPT="experiments/pp_fields.py --prep-binds" slurm/jobs/robot_gpu.sbatch
    sbatch ... --export=ALL,ENV=openpi,SCRIPT="experiments/pp_fields.py --shard 0 --nshards 4" slurm/jobs/robot_gpu.sbatch
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
OUT = os.path.join(ROOT, "results", "protos")
F0B = os.path.join(ROOT, "results", "robot", "f0b_firstframes")


def log(s):
    print(s, flush=True)


def save_atomic(path, **arrs):
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", default="", help="output subdirectory suffix (another checkpoint: obs_<tag>)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sets", default="train,l90,transfer")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--prep-binds", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from lifelong_ood.policy import bind as B
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.protos import store as S
    ip = os.path.join(OUT, f"index_{a.tag}.json")                  # another checkpoint's own index (pp_prep.py index_l90c)
    idx = json.load(open(ip if a.tag and os.path.exists(ip) else os.path.join(OUT, "index.json")))
    bk = PolicyBackbone(a.config, checkpoint_dir=a.checkpoint, device="cuda")
    log(f"[pp_fields] {bk.describe()}")
    cfgname = a.config if not a.tag else f"{a.config}_{a.tag}"
    demos, d90 = D.Demos(), D.Demos90()
    if a.prep_binds:
        srcs = sorted({int(b[1]) for o in idx["obs"] for c in o["conds"] for b in c["binds"]})
        for k in srcs:
            t0 = time.time()
            B.load_or_identify(bk, cfgname, demos, k)
            log(f"[pp_fields] binding of source task {k}: {time.time() - t0:.0f}s")
        return
    norm = D.Norm.for_backbone(bk)
    splice = S.RowSplice(bk, cfgname, demos)
    train_texts = {int(k): v for k, v in idx["train_texts"].items()}
    instr_conds = [S.Condition(f"i:{k}", "instr", train_texts[k]) for k in sorted(train_texts)]
    sets = set(a.sets.split(","))
    obs = [o for o in idx["obs"] if o["set"] in sets]
    mine = obs[a.shard::a.nshards]
    if a.limit:
        mine = mine[: a.limit]
    od_dir = os.path.join(OUT, "obs" + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(od_dir, exist_ok=True)
    cache = {}
    t_all = time.time()
    done = 0
    for o in mine:
        out = os.path.join(od_dir, o["key"] + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        # the observation
        ep = None
        if o["set"] == "train" and o.get("src") == "ipec":            # index_l90c: a LIBERO-90 training cell (IPEC demo)
            if ("l90", o["episode"]) not in cache:
                cache.clear()
                cache[("l90", o["episode"])] = d90.load(o["episode"])
            ep = cache[("l90", o["episode"])]
            od = d90.observation(ep, o["frame"])
        elif o["set"] == "train":
            if ("tr", o["episode"]) not in cache:
                cache.clear()
                cache[("tr", o["episode"])] = demos.load(o["episode"], images=True)
            ep = cache[("tr", o["episode"])]
            od = demos.observation(ep, o["frame"])
        elif o["set"] == "l90":
            if ("l90", o["episode"]) not in cache:
                cache.clear()
                cache[("l90", o["episode"])] = d90.load(o["episode"])
            ep = cache[("l90", o["episode"])]
            od = d90.observation(ep, o["frame"])
        else:
            z = np.load(os.path.join(F0B, f"{o['suite']}_{o['index']}.npz"), allow_pickle=True)
            s = o["state"]
            if s >= len(z["img"]):
                log(f"[pp_fields] {o['key']}: the first-frame file holds only {len(z['img'])} states; skipped")
                continue
            od = {"observation/image": z["img"][s], "observation/wrist_image": z["wimg"][s],
                  "observation/state": np.asarray(z["state8"][s], np.float32)}
        od = dict(od, prompt=o["language"])
        conds = [S.Condition.from_json(c) for c in o["conds"]]
        with torch.no_grad():
            live = S.Live(bk, od, conds, splice, block=16)
            pre_i = bk.prefix(bk.observation(dict(od, prompt=o["language"])))
            pre_n = bk.prefix(bk.observation(dict(od, prompt="")))
            drafts = np.concatenate([bk.sample(2, pre_i, seed=11), bk.sample(2, pre_n, seed=12)])
            starts = S.shared_starts(drafts, S.LEVELS, seed=0)
            M = live.modes(starts, iters=a.iters, lr=a.lr, probes=a.probes)
            t_modes = time.time() - t0
            C, Sn = len(conds), len(drafts)
            raw = {u: bk.unnormalize(M[u]["x0"].reshape(C * Sn, -1)).reshape(C, Sn, bk.H, -1) for u in S.LEVELS}
            arrs = dict(names=np.array([c.name for c in conds]), kinds=np.array([c.kind for c in conds]),
                        variants=np.array([c.variant for c in conds]), prompts=np.array([c.prompt for c in conds]),
                        targets=np.array([c.target or "" for c in conds]), levels=np.array(S.LEVELS), drafts=drafts,
                        draft_raw=bk.unnormalize(drafts),
                        xt=np.stack([M[u]["xt"] for u in S.LEVELS]), x0=np.stack([M[u]["x0"] for u in S.LEVELS]),
                        ext=np.stack([M[u]["ext"] for u in S.LEVELS]), x0_raw=np.stack([raw[u] for u in S.LEVELS]),
                        splice_log=np.array(splice.log[-20:]))
            if o["set"] in ("train", "l90"):
                T = len(ep["actions"])
                ci, _ = D.chunk_indices(T, bk.H)
                chunk = ep["actions"][ci[o["frame"]]]
                a_demo = D.to_rows(chunk[None], norm)[0]
                X, E = S.demo_points(a_demo, S.LEVELS, n=a.points, seed=0)
                ins = S.Live(bk, od, instr_conds, None, block=20)
                arrs.update(a_demo=a_demo, demo_raw=chunk, points=np.stack([X[u] for u in S.LEVELS]),
                            eps_true=np.stack([E[u] for u in S.LEVELS]),
                            E_core=np.stack([live.eps(X[u], u) for u in S.LEVELS], 1),       # [C, L, n, dim]
                            E_instr=np.stack([ins.eps(X[u], u) for u in S.LEVELS], 1))     # [40, L, n, dim]
        save_atomic(out, **arrs)
        done += 1
        log(f"[pp_fields] {o['key']}: {C} conditions, modes {t_modes:.0f}s, total {time.time() - t0:.0f}s "
            f"({done} done, {time.time() - t_all:.0f}s)")
    log(f"[pp_fields] shard {a.shard}/{a.nshards} finished: {done} new observations")


if __name__ == "__main__":
    main()
