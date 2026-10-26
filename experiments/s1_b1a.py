"""PLAN_SOLVE S1 (docs/PLAN_S1.md): B1a, the off-the-shelf binder.  GroundingDINO (transformers' implementation, weights
IDEA-Research/grounding-dino-base from the HF hub; no downloaded code) on the dev frames' 224 agentview image (client orientation).
Query = the role's phrase with its leading goal preposition removed ("on top of the cabinet" -> "top of the cabinet"; "to the right
of the caddy" -> "right of the caddy").  The highest-scoring box (any score) -> a 16 x 16 coverage map over the image's tokens;
the peak token is the token holding the box centre.  Depth is not used: S1 scores image-space masks.

    S1PY=<env with transformers>  experiments/s1_b1a.py  -> results/bind/s1/b1a/pred_dev.npz (keys, cov [U,256], peak [U], score [U])
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(S1, "b1a")
MODEL = "IDEA-Research/grounding-dino-base"
PREPS = ("on top of ", "to the right of ", "to the left of ", "to the front of ", "to the back of ", "under ", "into ", "in ", "on ")


def query(phrase):
    p = phrase
    for q in PREPS:
        if p.startswith(q):
            p = p[len(q):]
            if q.startswith("on top"):
                p = "top of " + p
            elif q.startswith("to the "):
                p = q[len("to the "):-len(" of ")] + " of " + p
            break
    p = re.sub(r"^the ", "", p)
    return p.strip() + "."


def main():
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(dev).eval()
    keys, covs, peaks, scores, qs = [], [], [], [], []
    for e in lab["dev"]:
        z = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)
        im = Image.fromarray(z["img"])
        q = query(e["phrase"])
        inp = proc(images=im, text=q, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.0, text_threshold=0.0,
                                                        target_sizes=[im.size[::-1]])[0]
        cov = np.zeros((224, 224), np.float32)
        if len(r["scores"]):
            i = int(r["scores"].argmax())
            x0, y0, x1, y1 = [float(v) for v in r["boxes"][i]]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(223, x1), min(223, y1)
            cov[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = 1.0
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            pk = int(min(15, cy // 14)) * 16 + int(min(15, cx // 14))
            sc = float(r["scores"][i])
        else:
            pk, sc = -1, 0.0
        keys.append(f"{e['role']}|{e['frame']}")
        covs.append(cov.reshape(16, 14, 16, 14).mean((1, 3)).reshape(-1))
        peaks.append(pk)
        scores.append(sc)
        qs.append(q)
    os.makedirs(OD, exist_ok=True)
    np.savez(os.path.join(OD, "pred_dev.npz"), keys=np.array(keys), cov=np.stack(covs).astype(np.float32), peak=np.array(peaks),
             score=np.array(scores), query=np.array(qs))
    print(f"[s1 b1a] {len(keys)} units; queries {sorted(set(qs))}", flush=True)


if __name__ == "__main__":
    main()
