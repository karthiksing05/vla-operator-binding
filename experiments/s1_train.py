"""PLAN_SOLVE S1 (docs/PLAN_S1.md): B1b, our own role-wise binder, trained ONCE on the training cells' labels (results/bind/s1/
labels.json "train"; frozen inputs from s1_feats.py).  openpi venv torch, GPU.  Hyper-parameters are fixed in advance (no dev-based
selection): the dev units are predicted once, after the last epoch.

One head per role (object, goal), each conditioned ONLY on its own role's phrase:
  phrase   the policy's word embeddings [L, 2048] -> Linear 256 + position -> 2-layer Transformer encoder
  image    the policy's agentview image tokens [256, 2048] -> LayerNorm, Linear 256 + learned 16 x 16 position
  fusion   2 Transformer decoder layers: image tokens self-attend and cross-attend to the phrase tokens
  output   one logit per image token (the mask over patches)
Loss: per-token BCE against the target's coverage (soft) + cross-entropy of the token softmax against the target mask (cov > 0)
renormalised.  AdamW lr 3e-4, weight decay 0.01, batch 64, EPOCHS epochs, dropout 0.1, seed 0.  A within-training check holds
out the training frames of states k >= 7 (reported only).

    .venv/bin/python experiments/s1_train.py [--smoke]  -> results/bind/s1/b1b/{obj,goal}.pt, results/bind/s1/b1b/pred_dev.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(ROOT, "results", "bind", "s1")
FE = os.path.join(S1, "feats")
OD = os.path.join(S1, "b1b")
D, NH, EPOCHS, BATCH, LR, WD, DROP, SEED = 256, 4, 40, 64, 3e-4, 0.01, 0.1, 0
VAL_K = 7


def build(torch):
    nn = torch.nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.tin = nn.Linear(2048, D)
            self.tpos = nn.Parameter(torch.zeros(16, D))
            self.tenc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, NH, 4 * D, DROP, batch_first=True), 2)
            self.iln = nn.LayerNorm(2048)
            self.iin = nn.Linear(2048, D)
            self.ipos = nn.Parameter(torch.randn(256, D) * 0.02)
            self.fuse = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, NH, 4 * D, DROP, batch_first=True), 2)
            self.out = nn.Linear(D, 1)

        def forward(self, img, ph, ln):
            L = ph.shape[1]
            pad = torch.arange(L, device=ph.device)[None] >= ln[:, None]
            t = self.tenc(self.tin(ph) + self.tpos[:L], src_key_padding_mask=pad)
            x = self.iin(self.iln(img)) + self.ipos
            x = self.fuse(x, t, memory_key_padding_mask=pad)
            return self.out(x).squeeze(-1)                      # [B, 256]
    return Head


def target_of(e, z_cache):
    z = z_cache(e["frame"])
    names = [str(x) for x in z["body_names"]] + [str(x) for x in z["site_names"]]
    cov = np.concatenate([z["body_cov"], z["site_cov"]]).astype(np.float32)
    t = np.zeros(256, np.float32)
    for n in e["targets"]:
        t = np.maximum(t, cov[names.index(n)])
    return t


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    pz = np.load(os.path.join(FE, "phrases.npz"), allow_pickle=True)
    pidx = {str(p): i for i, p in enumerate(pz["phrases"])}
    PH = torch.tensor(pz["emb"].astype(np.float32))
    PL = torch.tensor(pz["len"])
    feats = {s: np.load(os.path.join(FE, f"{s}_img.npy"), mmap_mode="r") for s in ("train", "dev")}
    fidx = {s: {f: i for i, f in enumerate(json.load(open(os.path.join(FE, f"{s}_frames.json"))))} for s in ("train", "dev")}
    cache = {}

    def zc(f):
        if f not in cache:
            cache[f] = np.load(os.path.join(ROOT, f), allow_pickle=True)
        return cache[f]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FT = {s: torch.tensor(np.asarray(feats[s]), device=dev) for s in feats}                    # fp16 [frames, 256, 2048]
    Head = build(torch)
    os.makedirs(OD, exist_ok=True)
    log = {}
    preds = {}
    for role in ("object", "goal"):
        ex = [e for e in lab["train"] if e["role"] == role]
        k_of = lambda e: int(os.path.basename(e["frame"]).rsplit("_", 1)[1].split(".")[0])   # noqa: E731
        tr = [e for e in ex if k_of(e) < VAL_K]
        va = [e for e in ex if k_of(e) >= VAL_K]
        T = lambda es: torch.tensor(np.stack([target_of(e, zc) for e in es]))             # noqa: E731
        Xi = lambda es, s: torch.tensor([fidx[s][e["frame"]] for e in es])                  # frame indices into FT[s]  # noqa: E731
        Pi = lambda es: torch.tensor([pidx[e["phrase"]] for e in es])                       # noqa: E731
        trI, trT, trP = Xi(tr, "train"), T(tr), Pi(tr)
        vaI, vaT, vaP = Xi(va, "train"), T(va), Pi(va)
        net = Head().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        ep = 2 if a.smoke else EPOCHS
        t0 = time.time()
        hist = []
        for epoch in range(ep):
            net.train()
            perm = torch.randperm(len(tr))
            tot = 0.0
            for g in range(0, len(tr), BATCH):
                b = perm[g:g + BATCH]
                img, tgt, pi = FT["train"][trI[b].to(dev)].float(), trT[b].to(dev), trP[b]
                lo = net(img, PH[pi].to(dev), PL[pi].to(dev))
                bce = torch.nn.functional.binary_cross_entropy_with_logits(lo, tgt.clamp(0, 1))
                m = (tgt > 0).float()
                ce = -(torch.log_softmax(lo, -1) * (m / m.sum(-1, keepdim=True).clamp(min=1))).sum(-1).mean()
                loss = bce + ce
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            net.eval()
            with torch.no_grad():
                hits = []
                for g in range(0, len(va), 256):
                    lo = net(FT["train"][vaI[g:g + 256].to(dev)].float(), PH[vaP[g:g + 256]].to(dev), PL[vaP[g:g + 256]].to(dev))
                    pk = lo.argmax(-1).cpu()
                    hits.append((vaT[g:g + 256][torch.arange(len(pk)), pk] > 0).float())
                vh = float(torch.cat(hits).mean()) if hits else float("nan")
            hist.append(dict(epoch=epoch, loss=tot / len(tr), val_hit=vh))
            print(f"[s1 train] {role} epoch {epoch}: loss {tot / len(tr):.4f} within-train val hit {vh:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
        torch.save(net.state_dict(), os.path.join(OD, f"{role}.pt"))
        log[role] = dict(n_train=len(tr), n_val=len(va), hist=hist, secs=time.time() - t0)
        dv = [e for e in lab["dev"] if e["role"] == role] if not a.smoke else []    # the smoke never predicts a dev unit
        if dv:
            with torch.no_grad():
                dI, dP = Xi(dv, "dev"), Pi(dv)
                lo = torch.cat([net(FT["dev"][dI[g:g + 256].to(dev)].float(), PH[dP[g:g + 256]].to(dev), PL[dP[g:g + 256]].to(dev)).cpu()
                                for g in range(0, len(dv), 256)])
            for e, l in zip(dv, lo.numpy()):
                preds[f"{role}|{e['frame']}"] = l.astype(np.float32)
    keys = sorted(preds)
    if keys:
        np.savez(os.path.join(OD, "pred_dev.npz"), keys=np.array(keys), logits=np.stack([preds[k] for k in keys]))
    json.dump(log, open(os.path.join(OD, "train_log.json" if not a.smoke else "train_log_smoke.json"), "w"), indent=1)
    print("[s1 train] done", flush=True)


if __name__ == "__main__":
    main()
