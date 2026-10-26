"""PLAN_SOLVE S1b (docs/PLAN_S1.md section 6): the factorised binder's RELATION / INSTANCE stage, trained once on the training
cells' S1 labels; openpi venv, GPU.  Hyper-parameters fixed in advance (S1's), no dev-based choice; dev units predicted once.

Per role phrase: (body, relation) = s1b_detect.split.  The body's candidate boxes are GroundingDINO's (results/bind/s1b/det.json).
One head per role (object, goal); each sees, per candidate, ONLY
  crop      the frozen policy's agentview image tokens resampled on an 8 x 8 grid over the candidate box enlarged EXP x about its
            centre (the body's own frame; bilinear on the 16 x 16 token map)
  geometry  the box (centre, size, detector score) and its rank position among the body's candidates (min-max over the set):
            what a qualifier ("middle", "on the left", "at the front") needs to pick an instance
  relation  the policy's own word embeddings of the RELATION phrase only (an empty relation -> one zero token)
-> an instance logit per candidate (softmax over the candidates) and a mask logit per crop cell.  The predicted peak = the argmax
cell of the chosen candidate, mapped back to its image token.
Labels: the positive candidate holds the largest share of ONE target instance's coverage (the box itself; if < 0.5, the enlarged box: table
regions lie beside their body); examples whose best share is < 0.5 are dropped (detector miss).  Loss = CE over candidates + (BCE
+ softmax CE) of the crop mask against the target's coverage at the crop cells.

    .venv/bin/python experiments/s1b_train.py [--smoke]  -> results/bind/s1b/pred_dev.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(ROOT, "results", "bind", "s1b")
D, NH, EPOCHS, BATCH, LR, WD, DROP, SEED = 256, 4, 40, 64, 3e-4, 0.01, 0.1, 0
G, EXP, KMAX, LMAX, VAL_K = 8, 2.0, 6, 12, 7


def cells(box):
    """8 x 8 crop cell centres (px, 224 image) over the box enlarged EXP x."""
    x0, y0, x1, y1 = box[:4]
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 4.0) * EXP, max(y1 - y0, 4.0) * EXP
    u = (np.arange(G) + 0.5) / G - 0.5
    return np.stack(np.meshgrid(cx + u * w, cy + u * h, indexing="xy"), -1).reshape(-1, 2)     # [64, (x, y)]


def tok_of(px):
    x, y = np.clip(px[..., 0], 0, 223), np.clip(px[..., 1], 0, 223)
    return (y // 14).astype(int) * 16 + (x // 14).astype(int)


def share(box, t, enlarge):
    x0, y0, x1, y1 = box[:4]
    if enlarge:
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * EXP, (y1 - y0) * EXP
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    c = (np.arange(16) + 0.5) * 14
    inside = ((c[None, :] >= x0) & (c[None, :] <= x1) & (c[:, None] >= y0) & (c[:, None] <= y1)).reshape(-1)
    return float(t[inside].sum() / max(t.sum(), 1e-9))


def geom(boxes):
    b = np.asarray([x[:5] for x in boxes], np.float32)
    cx, cy = (b[:, 0] + b[:, 2]) / 2 / 224, (b[:, 1] + b[:, 3]) / 2 / 224
    w, h = (b[:, 2] - b[:, 0]) / 224, (b[:, 3] - b[:, 1]) / 224
    rk = lambda v: (v - v.min()) / (v.max() - v.min()) if len(v) > 1 and v.max() > v.min() else np.full_like(v, 0.5)  # noqa: E731
    return np.stack([cx, cy, w, h, b[:, 4], rk(cx), rk(cy), np.full_like(cx, len(b) / KMAX)], -1)


def build(torch):
    nn = torch.nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.tin = nn.Linear(2048, D)
            self.tpos = nn.Parameter(torch.zeros(LMAX, D))
            self.tenc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, NH, 4 * D, DROP, batch_first=True), 2)
            self.cln = nn.LayerNorm(2048)
            self.cin = nn.Linear(2048, D)
            self.cpos = nn.Parameter(torch.randn(G * G, D) * 0.02)
            self.gin = nn.Sequential(nn.Linear(8, D), nn.GELU(), nn.Linear(D, D))
            self.fuse = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, NH, 4 * D, DROP, batch_first=True), 2)
            self.inst = nn.Linear(D, 1)
            self.mask = nn.Linear(D, 1)

        def forward(self, crop, gm, ph, ln):
            """crop [B, K, 64, 2048], gm [B, K, 8], ph [B, L, 2048], ln [B] -> inst [B, K], mask [B, K, 64]."""
            B, K = crop.shape[:2]
            L = ph.shape[1]
            pad = torch.arange(L, device=ph.device)[None] >= ln[:, None].clamp(min=1)
            t = self.tenc(self.tin(ph) + self.tpos[:L], src_key_padding_mask=pad)
            x = torch.cat([self.gin(gm).unsqueeze(2), self.cin(self.cln(crop)) + self.cpos], 2).reshape(B * K, 1 + G * G, D)
            t2 = t.unsqueeze(1).expand(B, K, L, D).reshape(B * K, L, D)
            p2 = pad.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
            y = self.fuse(x, t2, memory_key_padding_mask=p2).reshape(B, K, 1 + G * G, D)
            return self.inst(y[:, :, 0]).squeeze(-1), self.mask(y[:, :, 1:]).squeeze(-1)
    return Head


def rel_embeddings(rels):
    """The policy's own word embeddings of each relation phrase (cached)."""
    p = os.path.join(OD, "rel_phrases.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if set(rels) <= set(str(r) for r in z["phrases"]):
            return z
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    from openpi.models import tokenizer as _tok
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None)
    sp = _tok.PaligemmaTokenizer(200)._tokenizer
    emb = np.zeros((len(rels), LMAX, 2048), np.float16)
    ln = np.zeros(len(rels), np.int64)
    for i, r in enumerate(rels):
        ids = sp.encode(r, add_bos=False)[:LMAX] if r else []
        if ids:
            with torch.no_grad():
                e = bk.model.paligemma_with_expert.embed_language_tokens(torch.tensor([ids], device=bk.device))
            emb[i, :len(ids)] = e[0].float().cpu().numpy().astype(np.float16)
        ln[i] = len(ids)
    np.savez(p, emb=emb, len=ln, phrases=np.array(rels))
    del bk
    torch.cuda.empty_cache()
    return np.load(p, allow_pickle=True)


def main():
    import torch
    import s1b_detect as SD
    import s1_train as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    det = json.load(open(os.path.join(OD, "det.json")))
    lex, det = det["lexicon"], det["det"]
    rels = sorted({SD.split(e["phrase"], lex)[1] for e in lab["train"] + lab["dev"]})
    rz = rel_embeddings(rels)
    ridx = {str(p): i for i, p in enumerate(rz["phrases"])}
    RE, RL = torch.tensor(rz["emb"].astype(np.float32)), torch.tensor(rz["len"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FT = {s: torch.tensor(np.load(os.path.join(S1, "feats", f"{s}_img.npy")), device=dev) for s in ("train", "dev")}
    fidx = {s: {f: i for i, f in enumerate(json.load(open(os.path.join(S1, "feats", f"{s}_frames.json"))))} for s in ("train", "dev")}
    cache = {}

    def zc(f):
        if f not in cache:
            cache[f] = np.load(os.path.join(ROOT, f), allow_pickle=True)
        return cache[f]

    def prep(e, s, train):
        body, rel = SD.split(e["phrase"], lex)
        boxes = det.get(f"{e['frame']}|{body}", [])[:KMAX]
        if not boxes:
            return None
        cl = np.stack([cells(b) for b in boxes])                                   # [K, 64, 2]
        gm = geom(boxes)
        r = dict(set=s, fi=fidx[s][e["frame"]], cl=cl, gm=gm, rel=ridx[rel], boxes=boxes, body=body, relation=rel)
        if train:
            t = ST.target_of(e, zc)
            parts = [ST.target_of(dict(e, targets=[n]), zc) for n in e["targets"]]          # any ONE instance of a type phrase
            sh = np.array([max(share(b, q, False) for q in parts) for b in boxes])
            if sh.max() < 0.5:
                sh = np.array([max(share(b, q, True) for q in parts) for b in boxes])
            if sh.max() < 0.5:
                return None
            r["pos"] = int(sh.argmax())
            r["tc"] = t[tok_of(cl[r["pos"]])].astype(np.float32)                 # target coverage at the positive's cells
        return r

    def batch(rs):
        K = max(len(r["boxes"]) for r in rs)
        cl = np.zeros((len(rs), K, G * G, 2), np.float32)
        gm = np.zeros((len(rs), K, 8), np.float32)
        km = np.zeros((len(rs), K), bool)
        for i, r in enumerate(rs):
            k = len(r["boxes"])
            cl[i, :k], gm[i, :k], km[i, :k] = r["cl"], r["gm"], True
        fm = FT[rs[0]["set"]][torch.tensor([r["fi"] for r in rs], device=dev)].float()           # [B, 256, 2048]
        fm = fm.transpose(1, 2).reshape(len(rs), 2048, 16, 16)
        grid = torch.tensor(cl / 224.0 * 2 - 1, device=dev).reshape(len(rs), K * G * G, 1, 2)
        crop = torch.nn.functional.grid_sample(fm, grid, align_corners=False).squeeze(-1).transpose(1, 2)   # [B, K*64, 2048]
        ri = torch.tensor([r["rel"] for r in rs])
        return (crop.reshape(len(rs), K, G * G, 2048), torch.tensor(gm, device=dev), RE[ri].to(dev), RL[ri].to(dev),
                torch.tensor(km, device=dev))
    Head = build(torch)
    os.makedirs(OD, exist_ok=True)
    log, preds = {}, {}
    for role in ("object", "goal"):
        k_of = lambda e: int(os.path.basename(e["frame"]).rsplit("_", 1)[1].split(".")[0])     # noqa: E731
        ex = [(e, prep(e, "train", True)) for e in lab["train"] if e["role"] == role]
        drop = sum(r is None for _, r in ex)
        tr = [r for e, r in ex if r is not None and k_of(e) < VAL_K]
        va = [r for e, r in ex if r is not None and k_of(e) >= VAL_K]
        net = Head().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        t0, hist = time.time(), []
        for epoch in range(2 if a.smoke else EPOCHS):
            net.train()
            perm = np.random.permutation(len(tr))
            tot = 0.0
            for g in range(0, len(tr), BATCH):
                rs = [tr[i] for i in perm[g:g + BATCH]]
                crop, gm, ph, ln, km = batch(rs)
                il, ml = net(crop, gm, ph, ln)
                il = il.masked_fill(~km, -1e4)
                pos = torch.tensor([r["pos"] for r in rs], device=dev)
                ce_i = torch.nn.functional.cross_entropy(il, pos)
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
                        hits.append(float(c == r["pos"] and r["tc"][int(mm[c].argmax())] > 0))
            hist.append(dict(epoch=epoch, loss=tot / max(len(tr), 1), val_hit=float(np.mean(hits)) if hits else None))
            print(f"[s1b train] {role} epoch {epoch}: loss {hist[-1]['loss']:.4f} within-train val hit {hist[-1]['val_hit']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        log[role] = dict(n_train=len(tr), n_val=len(va), dropped_no_match=drop, hist=hist)
        if a.smoke:
            continue
        torch.save(net.state_dict(), os.path.join(OD, f"{role}.pt"))
        for e in [e for e in lab["dev"] if e["role"] == role]:
            r = prep(e, "dev", False)
            key = f"{role}|{e['frame']}"
            if r is None:
                preds[key] = dict(peak=-1, mask=np.zeros(256, bool), cand=-1, body=SD.split(e["phrase"], lex)[0], box=None)
                continue
            with torch.no_grad():
                crop, gm, ph, ln, km = batch([r])
                il, ml = net(crop, gm, ph, ln)
            c = int(il[0].argmax())
            mm = ml[0, c].cpu().numpy()
            cl = r["cl"][c]
            msk = np.zeros(256, bool)
            msk[tok_of(cl[1 / (1 + np.exp(-mm)) >= 0.5])] = True
            preds[key] = dict(peak=int(tok_of(cl[int(mm.argmax())])), mask=msk, cand=c, body=r["body"], relation=r["relation"],
                              box=r["boxes"][c])
    if preds:
        keys = sorted(preds)
        np.savez(os.path.join(OD, "pred_dev.npz"), keys=np.array(keys), peak=np.array([preds[k]["peak"] for k in keys]),
                 mask=np.stack([preds[k]["mask"] for k in keys]), cand=np.array([preds[k]["cand"] for k in keys]),
                 meta=np.array([json.dumps({kk: v for kk, v in preds[k].items() if kk not in ("mask",)}) for k in keys]))
    json.dump(log, open(os.path.join(OD, "train_log.json" if not a.smoke else "train_log_smoke.json"), "w"), indent=1)
    print(f"[s1b train] done: {json.dumps({k: {kk: v for kk, v in d.items() if kk != 'hist'} for k, d in log.items()})}", flush=True)


if __name__ == "__main__":
    main()
