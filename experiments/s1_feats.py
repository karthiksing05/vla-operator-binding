"""PLAN_SOLVE S1 (docs/PLAN_S1.md): the binder's frozen inputs, from pi05_l90c@5000 itself (openpi venv, GPU).

  image   the agentview's 256 image tokens as the policy's prefix embeds them (embed_prefix(...)[:, :256], 2048-d: SigLIP + the
          PaliGemma projector), exactly the features of the SigLIP prototype anchor (d5_serve._sig) -> feats/<set>_img.npy fp16
  phrase  the policy's OWN word embeddings: PaliGemma's embed_language_tokens of the phrase's SentencePiece ids (no BOS), padded
          to LMAX -> feats/phrases.npz (emb [P, LMAX, 2048] fp16, len [P], phrases)

    BACKBONE_CONFIG=pi05_l90c BACKBONE_CKPT=... .venv/bin/python experiments/s1_feats.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
FE = os.path.join(S1, "feats")
LMAX, BATCH = 16, 16


def main():
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    os.makedirs(FE, exist_ok=True)
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None)
    m = bk.model
    for set_name in ("train", "dev"):
        frames = sorted({e["frame"] for e in lab[set_name]})
        out = np.zeros((len(frames), 256, 2048), np.float16)
        for g in range(0, len(frames), BATCH):
            obs = []
            for f in frames[g:g + BATCH]:
                z = np.load(os.path.join(ROOT, f), allow_pickle=True)
                obs.append({"observation/image": z["img"], "observation/wrist_image": z["wimg"],
                            "observation/state": z["state8"], "prompt": "pick up the object"})
            ob = bk.observations(obs)
            with torch.no_grad():
                images, img_masks, lt, lm, _ = m._preprocess_observation(ob, train=False)
                embs, _, _ = m.embed_prefix(images, img_masks, lt, lm)
            out[g:g + len(obs)] = embs[:, :256].float().cpu().numpy().astype(np.float16)
        np.save(os.path.join(FE, f"{set_name}_img.npy"), out)
        json.dump(frames, open(os.path.join(FE, f"{set_name}_frames.json"), "w"))
        print(f"[s1 feats] {set_name}: {len(frames)} frames", flush=True)
    from openpi.models import tokenizer as _tok
    sp = _tok.PaligemmaTokenizer(200)._tokenizer
    P = lab["phrases"]
    emb = np.zeros((len(P), LMAX, 2048), np.float16)
    ln = np.zeros(len(P), np.int64)
    for i, p in enumerate(P):
        ids = sp.encode(p, add_bos=False)[:LMAX]
        with torch.no_grad():
            e = m.paligemma_with_expert.embed_language_tokens(torch.tensor([ids], device=bk.device))
        emb[i, :len(ids)] = e[0].float().cpu().numpy().astype(np.float16)
        ln[i] = len(ids)
    np.savez(os.path.join(FE, "phrases.npz"), emb=emb, len=ln, phrases=np.array(P))
    print(f"[s1 feats] {len(P)} phrases, max len {ln.max()}", flush=True)


if __name__ == "__main__":
    main()
