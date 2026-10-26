"""PLAN_SOLVE S1b (docs/PLAN_S1.md section 6): the factorised binder's BODY stage.  Each role phrase is split into a BODY phrase
(the longest known type / fixture noun phrase in it) and a RELATION phrase (what remains: "in the top drawer of the", "middle",
"at the front", "on the").  GroundingDINO-base proposes every box for the body phrase on the policy's 224 agentview image
(threshold 0.05, NMS IoU 0.5, at most K_MAX boxes by score) -> results/bind/s1b/det.json {"<frame>|<body>": [[x0,y0,x1,y1,score]]}.
Run on every training example's and dev unit's frame (S1's labels).  Chunks env, GPU.

    S1PY=chunks python experiments/s1b_detect.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(ROOT, "results", "bind", "s1b")
MODEL = "IDEA-Research/grounding-dino-base"
THR, NMS, K_MAX = 0.05, 0.5, 6
FIXTURE_NOUNS = ("cabinet shelf", "wine rack", "cabinet", "caddy", "stove", "tray", "basket", "microwave", "rack", "plate", "bowl")


def body_lexicon(lab):
    """Type phrases of the TRAINING tasks (labels.json type_phrases, from factors.object_phrases + its fallback table) plus the
    fixture nouns of the attested goal templates; longest first."""
    lex = set(lab["type_phrases"].values()) | set(FIXTURE_NOUNS)
    return sorted(lex, key=lambda s: (-len(s), s))


def split(phrase, lex):
    """-> (body, relation).  The longest lexicon entry occurring as whole words; the relation is the phrase with it removed."""
    for b in lex:
        m = re.search(rf"\b{re.escape(b)}\b", phrase)
        if m:
            rel = " ".join((phrase[:m.start()] + " " + phrase[m.end():]).split())
            return b, rel
    return phrase, ""


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    lex = body_lexicon(lab)
    want = {}
    for e in lab["train"] + lab["dev"]:
        want.setdefault(e["frame"], set()).add(split(e["phrase"], lex)[0])
    dev = torch.device("cuda")
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(dev).eval()
    out = {}
    for i, (f, bodies) in enumerate(sorted(want.items())):
        im = Image.fromarray(np.load(os.path.join(ROOT, f), allow_pickle=True)["img"])
        for b in sorted(bodies):
            inp = proc(images=im, text=b + ".", return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model(**inp)
            r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=THR, text_threshold=THR,
                                                            target_sizes=[im.size[::-1]])[0]
            bx = sorted(([*map(float, bb), float(s)] for bb, s in zip(r["boxes"], r["scores"])), key=lambda x: -x[4])
            keep = []
            for x in bx:
                if all(iou(x, k) < NMS for k in keep):
                    keep.append(x)
                if len(keep) == K_MAX:
                    break
            out[f"{f}|{b}"] = keep
        if i % 50 == 0:
            print(f"[s1b detect] {i}/{len(want)} frames", flush=True)
    os.makedirs(OD, exist_ok=True)
    json.dump(dict(lexicon=lex, det=out), open(os.path.join(OD, "det.json"), "w"))
    n0 = sum(1 for v in out.values() if not v)
    print(f"[s1b detect] {len(out)} (frame, body) queries; {n0} with no box; lexicon {lex}", flush=True)


if __name__ == "__main__":
    main()
