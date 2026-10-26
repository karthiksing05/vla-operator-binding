"""PLAN_SOLVE S1c (docs/PLAN_S1.md section 8): the binder that keeps S1b's transferable body stage (open-vocabulary detection) and
trains its instance / sub-region head on SIMULATOR boxes, so every instance has its own clean candidate.  openpi venv, GPU.  S1's
hyper-parameters (AdamW 3e-4, 40 epochs, seed 0), no dev-based choice; dev units predicted once.

TRAINING (training cells only; S1's labels).  Per example (role, phrase, targets) with (body, relation) = s1b_detect.split:
  candidates  the frame's bodies whose type the BODY phrase names (the body phrase -> types map is read from the TRAINING labels'
              targets), each as the tight box of its segmentation tokens, jittered (scale x U(0.85, 1.2), shift +-8 px; detector-like),
              plus up to N_DISTRACT boxes of other bodies of the frame (negatives: detector false positives)
  positive    the target instance (any one of a type phrase's instances); for a region target its parent body; for a table region
              its reference body
  mask        the target's coverage at the positive's 8 x 8 crop cells (S1b's crop: the box enlarged 2x, the body's frame)
TEST: the candidates are GroundingDINO's boxes for the body phrase (results/bind/s1b/det.json), exactly as S1b.
Head (s1b_train.build): per candidate the frozen policy's image tokens on the 8 x 8 crop + box geometry and rank among the
candidates; the phrase = the FULL role phrase in the policy's own word embeddings (so a wrong-type box can be rejected).
Loss: multi-positive CE over candidates + (BCE + softmax CE) of the positive's crop mask.

    .venv/bin/python experiments/s1c_train.py [--smoke]  -> results/bind/s1c/pred_dev.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(ROOT, "results", "bind", "s1c")
N_DISTRACT, SEED = 2, 0


def itype(n):
    return re.sub(r"_\d+$", "", n)


def seg_box(cov):
    idx = np.flatnonzero(cov > 0)
    if not len(idx):
        return None
    r, c = idx // 16, idx % 16
    return [float(c.min() * 14), float(r.min() * 14), float((c.max() + 1) * 14), float((r.max() + 1) * 14), 1.0]


def jitter(b, rng):
    x0, y0, x1, y1, s = b
    cx, cy, w, h = (x0 + x1) / 2 + rng.uniform(-8, 8), (y0 + y1) / 2 + rng.uniform(-8, 8), (x1 - x0) * rng.uniform(0.85, 1.2), \
        (y1 - y0) * rng.uniform(0.85, 1.2)
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(rng.uniform(0.2, 0.6))]


def main():
    import torch
    import s1_read as SR
    import s1_train as ST
    import s1b_detect as SD
    import s1b_train as SB
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    det = json.load(open(os.path.join(ROOT, "results", "bind", "s1b", "det.json")))
    lex, det = det["lexicon"], det["det"]
    pz = np.load(os.path.join(S1, "feats", "phrases.npz"), allow_pickle=True)
    pidx = {str(p): i for i, p in enumerate(pz["phrases"])}
    PH, PL = torch.tensor(pz["emb"][:, :12].astype(np.float32)), torch.tensor(pz["len"]).clamp(max=12)   # the head's LMAX
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FT = {s: torch.tensor(np.load(os.path.join(S1, "feats", f"{s}_img.npy")), device=dev) for s in ("train", "dev")}
    fidx = {s: {f: i for i, f in enumerate(json.load(open(os.path.join(S1, "feats", f"{s}_frames.json"))))} for s in ("train", "dev")}
    cache = {}

    def frame(f):
        if f not in cache:
            z = np.load(os.path.join(ROOT, f), allow_pickle=True)
            names, cov = SR.masks(z)
            cache[f] = (names, cov, [str(b) for b in z["body_names"]])
        return cache[f]

    def owner(t, names, bodies):
        """the body a target belongs to: itself, a region's body, a table region's reference body."""
        if t in bodies:
            return t
        tab = next((x for x in SR.TABLES if t.startswith(x + "_")), None)
        if tab:
            _, rt = SR.competitors(t, names, bodies)
            return next((b for b in bodies if itype(b) == rt), None)
        return max([b for b in bodies if t.startswith(b + "_")], key=len, default=None)
    # body phrase -> body types, from the TRAINING labels only
    types = {}
    for e in lab["train"]:
        names, _, bodies = frame(e["frame"])
        body = SD.split(e["phrase"], lex)[0]
        for t in e["targets"]:
            o = owner(t, names, bodies)
            if o:
                types.setdefault(body, set()).add(itype(o))

    def prep_train(e):
        names, cov, bodies = frame(e["frame"])
        body = SD.split(e["phrase"], lex)[0]
        own = {owner(t, names, bodies) for t in e["targets"]} - {None}
        cands = [b for b in bodies if itype(b) in types.get(body, set())]
        if not own or not own <= set(cands):
            return None
        others = [b for b in bodies if b not in cands and seg_box(cov[names.index(b)]) is not None]
        dis = list(rng.choice(others, size=min(N_DISTRACT, len(others)), replace=False)) if others and rng.random() < 0.5 else []
        cand = [c for c in cands if seg_box(cov[names.index(c)]) is not None] + dis
        if not cand or not (own & set(cand)):
            return None
        rng.shuffle(cand)
        boxes = [jitter(seg_box(cov[names.index(c)]), rng) for c in cand][:SB.KMAX]
        cand = cand[:SB.KMAX]
        pos = [j for j, c in enumerate(cand) if c in own]
        if not pos:
            return None
        t = ST.target_of(e, lambda f: np.load(os.path.join(ROOT, f), allow_pickle=True))
        cl = np.stack([SB.cells(b) for b in boxes])
        best = max(pos, key=lambda j: float(t[SB.tok_of(cl[j])].sum()))
        return dict(set="train", fi=fidx["train"][e["frame"]], cl=cl, gm=SB.geom(boxes), rel=pidx[e["phrase"]], boxes=boxes,
                    posset=pos, pos=best, tc=t[SB.tok_of(cl[best])].astype(np.float32))

    def prep_dev(e):
        body = SD.split(e["phrase"], lex)[0]
        boxes = det.get(f"{e['frame']}|{body}", [])[:SB.KMAX]
        if not boxes:
            return None
        cl = np.stack([SB.cells(b) for b in boxes])
        return dict(set="dev", fi=fidx["dev"][e["frame"]], cl=cl, gm=SB.geom(boxes), rel=pidx[e["phrase"]], boxes=boxes, body=body)

    def batch(rs):
        K = max(len(r["boxes"]) for r in rs)
        G2 = SB.G * SB.G
        cl = np.zeros((len(rs), K, G2, 2), np.float32)
        gm = np.zeros((len(rs), K, 8), np.float32)
        km = np.zeros((len(rs), K), bool)
        for i, r in enumerate(rs):
            k = len(r["boxes"])
            cl[i, :k], gm[i, :k], km[i, :k] = r["cl"], r["gm"], True
        fm = FT[rs[0]["set"]][torch.tensor([r["fi"] for r in rs], device=dev)].float().transpose(1, 2).reshape(len(rs), 2048, 16, 16)
        grid = torch.tensor(cl / 224.0 * 2 - 1, device=dev).reshape(len(rs), K * G2, 1, 2)
        crop = torch.nn.functional.grid_sample(fm, grid, align_corners=False).squeeze(-1).transpose(1, 2).reshape(len(rs), K, G2, 2048)
        ri = torch.tensor([r["rel"] for r in rs])
        return crop, torch.tensor(gm, device=dev), PH[ri].to(dev), PL[ri].to(dev), torch.tensor(km, device=dev)
    Head = SB.build(torch)
    os.makedirs(OD, exist_ok=True)
    log, preds = {}, {}
    k_of = lambda e: int(os.path.basename(e["frame"]).rsplit("_", 1)[1].split(".")[0])                # noqa: E731
    for role in ("object", "goal"):
        ex = [(e, prep_train(e)) for e in lab["train"] if e["role"] == role]
        drop = sum(r is None for _, r in ex)
        tr = [r for e, r in ex if r is not None and k_of(e) < SB.VAL_K]
        va = [r for e, r in ex if r is not None and k_of(e) >= SB.VAL_K]
        net = Head().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=SB.LR, weight_decay=SB.WD)
        t0, hist = time.time(), []
        for epoch in range(2 if a.smoke else SB.EPOCHS):
            net.train()
            perm = rng.permutation(len(tr))
            tot = 0.0
            for g in range(0, len(tr), SB.BATCH):
                rs = [tr[i] for i in perm[g:g + SB.BATCH]]
                crop, gm, ph, ln, km = batch(rs)
                il, ml = net(crop, gm, ph, ln)
                il = il.masked_fill(~km, -1e4)
                pm = torch.zeros_like(il, dtype=torch.bool)
                for i, r in enumerate(rs):
                    pm[i, r["posset"]] = True
                ce_i = (torch.logsumexp(il, -1) - torch.logsumexp(il.masked_fill(~pm, -1e4), -1)).mean()
                pos = torch.tensor([r["pos"] for r in rs], device=dev)
                mp = ml[torch.arange(len(rs), device=dev), pos]
                tc = torch.tensor(np.stack([r["tc"] for r in rs]), device=dev)
                has = (tc.sum(-1) > 0).float()
                bce = (torch.nn.functional.binary_cross_entropy_with_logits(mp, tc.clamp(0, 1), reduction="none").mean(-1) * has).sum() / has.sum().clamp(min=1)
                m = (tc > 0).float()
                ce_m = (-(torch.log_softmax(mp, -1) * m / m.sum(-1, keepdim=True).clamp(min=1)).sum(-1) * has).sum() / has.sum().clamp(min=1)
                loss = ce_i + bce + ce_m
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(rs)
            net.eval()
            with torch.no_grad():
                hits = []
                for g in range(0, len(va), 128):
                    rs = va[g:g + 128]
                    crop, gm, ph, ln, km = batch(rs)
                    il, ml = net(crop, gm, ph, ln)
                    ci = il.masked_fill(~km, -1e4).argmax(-1).cpu().numpy()
                    for r, c, mm in zip(rs, ci, ml.cpu().numpy()):
                        hits.append(float(c in r["posset"] and (c != r["pos"] or r["tc"][int(mm[c].argmax())] > 0)))
            hist.append(dict(epoch=epoch, loss=tot / max(len(tr), 1), val_hit=float(np.mean(hits)) if hits else None))
            print(f"[s1c train] {role} epoch {epoch}: loss {hist[-1]['loss']:.4f} within-train val hit {hist[-1]['val_hit']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        log[role] = dict(n_train=len(tr), n_val=len(va), dropped=drop, hist=hist)
        if a.smoke:
            continue
        torch.save(net.state_dict(), os.path.join(OD, f"{role}.pt"))
        for e in [e for e in lab["dev"] if e["role"] == role]:
            r = prep_dev(e)
            key = f"{role}|{e['frame']}"
            if r is None:
                preds[key] = dict(peak=-1, mask=np.zeros(256, bool), cand=-1, body=None, box=None)
                continue
            with torch.no_grad():
                crop, gm, ph, ln, km = batch([r])
                il, ml = net(crop, gm, ph, ln)
            c = int(il[0].argmax())
            mm = ml[0, c].cpu().numpy()
            cl = r["cl"][c]
            msk = np.zeros(256, bool)
            msk[SB.tok_of(cl[1 / (1 + np.exp(-mm)) >= 0.5])] = True
            preds[key] = dict(peak=int(SB.tok_of(cl[int(mm.argmax())])), mask=msk, cand=c, body=r["body"],
                              relation=SD.split(e["phrase"], lex)[1], box=r["boxes"][c])
    if preds:
        keys = sorted(preds)
        np.savez(os.path.join(OD, "pred_dev.npz"), keys=np.array(keys), peak=np.array([preds[k]["peak"] for k in keys]),
                 mask=np.stack([preds[k]["mask"] for k in keys]), cand=np.array([preds[k]["cand"] for k in keys]),
                 meta=np.array([json.dumps({kk: v for kk, v in preds[k].items() if kk != "mask"}) for k in keys]))
    json.dump(dict(log, body_types={k: sorted(v) for k, v in types.items()}),
              open(os.path.join(OD, "train_log.json" if not a.smoke else "train_log_smoke.json"), "w"), indent=1)
    print(f"[s1c train] done: {json.dumps({k: {kk: v for kk, v in d.items() if kk != 'hist'} for k, d in log.items()})}; body types "
          f"{ {k: sorted(v) for k, v in types.items()} }", flush=True)


if __name__ == "__main__":
    main()
