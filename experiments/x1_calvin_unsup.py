"""X1, the unsupervised scheme on CALVIN (offline, SLURM GPU job): docs/UNSUPERVISED.md's two reads on CALVIN's ABC training play.

Sequences: two CONSECUTIVE annotated 64-frame windows (i, j) of task_ABC_D/training/lang_annotations (same recording segment,
start_j in [end_i - 32, end_i + 64], task_i != task_j: the pairing rule of experiments/x1_calvin_split.py), whose frames
start_i .. end_j + 9 are all extracted.  CALVIN annotates a window ending where its subtask is solved, so the true boundary is
b = end_i; segment 1 = [start_i, end_i], segment 2 = [max(start_j, end_i + 1), end_j].

(a) Boundaries from contact events: the frames where the commanded gripper (rel_actions[:, 6], +-1) changes sign.  A boundary
    is recovered at tolerance w if an event lies within w frames of b.  Baseline: the same number of cuts drawn uniformly over the
    sequence's frames (200 draws per sequence), the same hit rule.  Sequences with no event count as misses for both.
(b) Identities from FLOWER's likelihood: frozen FLOWER-CALVIN (FlowerCalvinBackbone, load fix).  At every `stride`-th frame f of
    a segment (f + 9 inside the segment where possible), the rectified-flow loss ||v - u||^2 of the recorded chunk
    rel_actions[f : f + 10] (clamped to [-1, 1], as training) under each of the 34 subtask instructions (the evaluation
    annotation of each subtask), with K shared draws (eps, t; t stratified uniform, FLOWER's sampling_type) per frame; averaged
    over the segment's frames and draws -> the argmin instruction.  Top-1 = it names the segment's annotated subtask (chance
    1 / 34); also the true subtask's rank.  NB: these are FLOWER's own training episodes (in-distribution scenes A-C).

python experiments/x1_calvin_unsup.py --data <task_ABC_D> --flower_root <flower_vla_calvin> --ckpt_dir <snapshot> --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def sequences(data, rng, n, gap=64, overlap=32):
    from x1_calvin_split import load, scene_of, seg_of
    idx, tasks, scene, segs = load(data)
    tr = pathlib.Path(data) / "training"
    have = set()
    for name in os.listdir(tr):
        if name.startswith("episode_") and name.endswith(".npz"):
            have.add(int(name[8:15]))
    order = np.argsort(idx[:, 0], kind="stable")
    seg = np.array([seg_of(int(s), segs) for s in idx[:, 0]])
    cands = []
    for k, i in enumerate(order):
        e = idx[i, 1]
        for j in order[k + 1:]:
            if idx[j, 0] > e + gap or seg[j] != seg[i]:
                break
            if idx[j, 0] >= e - overlap:
                if tasks[i] != tasks[j]:
                    a0, a1 = int(idx[i, 0]), int(idx[j, 1]) + 9
                    if all(f in have for f in range(a0, a1 + 1)):
                        cands.append((int(i), int(j)))
                break
    rng.shuffle(cands)
    out = []
    for i, j in cands[:n]:
        s1 = (int(idx[i, 0]), int(idx[i, 1]))
        s2 = (max(int(idx[j, 0]), int(idx[i, 1]) + 1), int(idx[j, 1]))
        out.append(dict(i=i, j=j, tasks=[tasks[i], tasks[j]], seg1=s1, seg2=s2, b=int(idx[i, 1]),
                        scene=scene_of(s1[0], scene)))
    return out, len(cands)


def load_frames(data, a, b, keys=("rel_actions", "rgb_static", "rgb_gripper", "robot_obs")):
    tr = pathlib.Path(data) / "training"
    fr = {k: [] for k in keys}
    for f in range(a, b + 1):
        z = np.load(tr / f"episode_{f:07d}.npz")
        for k in keys:
            fr[k].append(z[k])
    return {k: np.stack(v) for k, v in fr.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--flower_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_id", action="store_true", help="read (a) only")
    ap.add_argument("--out", default=str(ROOT / "results/x1_calvin/unsup.json"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    seqs, n_cands = sequences(a.data, rng, a.n)
    print(f"[unsup] {len(seqs)} sequences of {n_cands} available consecutive pairs; scenes {Counter(s['scene'] for s in seqs)}",
          flush=True)

    from omegaconf import OmegaConf
    ann = OmegaConf.load(pathlib.Path(a.flower_root) / "conf/annotations/new_playtable_validation.yaml")
    names = sorted(ann.keys())
    prompts = [str(ann[t][0]) for t in names]

    bk = None
    if not a.no_id:
        import torch
        import calvin_env  # noqa: F401
        sys.path.insert(0, a.flower_root)
        from lifelong_ood.policy.backbone_flower_calvin import FlowerCalvinBackbone
        bk = FlowerCalvinBackbone(checkpoint=a.ckpt_dir, device="cuda")
        print("[unsup]", bk.describe(), flush=True)

    tols = (5, 10, 20)
    hits = {w: [] for w in tols}
    rhits = {w: [] for w in tols}
    n_events, dist = [], []
    ident = []
    t0 = time.time()
    for si, s in enumerate(seqs):
        lo, hi = s["seg1"][0], s["seg2"][1]
        fr = load_frames(a.data, lo, hi + 9)
        g = np.sign(fr["rel_actions"][:, 6])
        T = hi - lo + 1
        ev = [lo + f for f in range(1, T) if g[f] != g[f - 1]]
        n_events.append(len(ev))
        b = s["b"]
        dist.append(min([abs(e - b) for e in ev]) if ev else None)
        for w in tols:
            hits[w].append(float(any(abs(e - b) <= w for e in ev)))
            if ev:
                r = rng.integers(lo + 1, hi + 1, size=(200, len(ev)))
                rhits[w].append(float(np.mean(np.any(np.abs(r - b) <= w, axis=1))))
            else:
                rhits[w].append(0.0)
        if bk is not None:
            import torch
            rec = dict(seq=si, tasks=s["tasks"], scene=s["scene"], segs=[])
            for sg, true in ((s["seg1"], s["tasks"][0]), (s["seg2"], s["tasks"][1])):
                last = max(sg[0], sg[1] - 9)
                frames = list(range(sg[0], last + 1, a.stride)) or [sg[0]]
                L = np.zeros((len(frames), len(prompts), a.K), np.float32)
                for q, f in enumerate(frames):
                    k = f - lo
                    obs = [dict(rgb_static=fr["rgb_static"][k], rgb_gripper=fr["rgb_gripper"][k], robot_obs=fr["robot_obs"][k],
                                prompt=p) for p in prompts]
                    x = torch.as_tensor(np.clip(fr["rel_actions"][k:k + 10], -1, 1), dtype=torch.float32, device=bk.device)
                    gen = torch.Generator().manual_seed(7919 * si + f)
                    tt = ((torch.rand(1, generator=gen) + torch.arange(a.K) / a.K) % (1 - 1e-5)).to(bk.device)
                    e = torch.randn((a.K, 10, 7), generator=gen).to(bk.device)
                    xt = (1 - tt[:, None, None]) * x[None] + tt[:, None, None] * e
                    u = e - x[None]
                    P = len(prompts)
                    with torch.no_grad():
                        pre = bk.prefix(bk.observations(obs))
                        v = bk.v_full(xt.repeat(P, 1, 1), tt.repeat(P), bk.repeat(pre, a.K)).reshape(P, a.K, 10, 7)
                    L[q] = ((v - u[None]) ** 2).mean((2, 3)).cpu().numpy()
                m = L.mean((0, 2))
                ti = names.index(true)
                rank = int((m < m[ti]).sum())
                per_frame = float(np.mean(L.mean(2).argmin(1) == ti))
                rec["segs"].append(dict(true=true, pred=names[int(m.argmin())], top1=int(m.argmin() == ti), rank=rank,
                                        n_frames=len(frames), frame_top1=per_frame))
            ident.append(rec)
        if si % 20 == 0:
            print(f"[unsup] {si + 1}/{len(seqs)} {time.time() - t0:.0f}s hit10 {np.mean(hits[10]):.3f} rand {np.mean(rhits[10]):.3f}"
                  + (f" top1 {np.mean([g_['top1'] for r_ in ident for g_ in r_['segs']]):.3f}" if ident else ""), flush=True)

    res = dict(n_sequences=len(seqs), n_candidates=n_cands, scenes=dict(Counter(s["scene"] for s in seqs)),
               events_per_seq=dict(mean=float(np.mean(n_events)), zero=int(sum(x == 0 for x in n_events))),
               boundary={str(w): dict(contact=float(np.mean(hits[w])), random_matched=float(np.mean(rhits[w]))) for w in tols},
               median_event_dist=float(np.median([d for d in dist if d is not None])) if any(d is not None for d in dist) else None)
    # per pair family: does the first subtask involve the gripper at its end
    if ident:
        segs = [g_ for r_ in ident for g_ in r_["segs"]]
        res["identity"] = dict(n_segments=len(segs), top1=float(np.mean([g_["top1"] for g_ in segs])), chance=1 / len(prompts),
                               median_rank=float(np.median([g_["rank"] for g_ in segs])),
                               mean_rank=float(np.mean([g_["rank"] for g_ in segs])),
                               frame_top1=float(np.mean([g_["frame_top1"] for g_ in segs])),
                               first_segment_top1=float(np.mean([r_["segs"][0]["top1"] for r_ in ident])),
                               second_segment_top1=float(np.mean([r_["segs"][1]["top1"] for r_ in ident])),
                               by_task={t: dict(n=sum(g_["true"] == t for g_ in segs),
                                                top1=float(np.mean([g_["top1"] for g_ in segs if g_["true"] == t])))
                                        for t in sorted({g_["true"] for g_ in segs})},
                               confusions=Counter(f"{g_['true']} -> {g_['pred']}" for g_ in segs if not g_["top1"]).most_common(15))
    out = dict(res, sequences=seqs, ident=ident, n_events=n_events, dist=dist, args=vars(a))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str), flush=True)
    if a.record:
        from lifelong_ood import ledger
        p = ledger.record("x1_calvin_unsup", bench="calvin_abc_training", tag="flower_calvin", seed=a.seed, metrics=res,
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=dict(vars(a)), script="experiments/x1_calvin_unsup.py", command=" ".join(sys.argv),
                          seconds=time.time() - t0)
        print("ledger:", p)


if __name__ == "__main__":
    main()
