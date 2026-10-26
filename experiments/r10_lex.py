"""PROBE R10 step 1 (docs/PLAN_R10.md): is there a TRAINED sub-instruction form for :29 / :40 / :16's objects?  CPU only, from the
training-cell instructions of pi05_l90c (m1_compose_create.store_pool: the 73 distinct training texts; split_sealed.json is never
opened).  -> results/r10/lex.json.  No policy output."""
import json
import os
import re
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1

tx, scenes = M1.store_pool()
texts = sorted(set(tx.values()))
objs = {"libero_90:29": ["black bowl", "bowl"], "libero_90:40": ["frying pan", "pan"], "libero_90:16": ["black bowl", "bowl"]}
standalone = [t for t in texts if re.match(r"^(pick up|pick|lift|grab)\b", t) and " and " not in t]
pickup_any = [t for t in texts if t.startswith("pick up")]
out = dict(n_texts=len(texts), standalone_pickup=standalone, pickup_prefixed=pickup_any, per_task={})
print(f"[r10 lex] {len(texts)} distinct training instructions; standalone pick-up instructions: {standalone}", flush=True)
for tid, ws in objs.items():
    hits = {w: [t for t in texts if re.search(r"\b" + re.escape(w) + r"\b", t)] for w in ws}
    pk = {w: [t for t in pickup_any if re.search(r"\bpick up the (\w+ )*?" + re.escape(w) + r"\b", t)] for w in ws}
    out["per_task"][tid] = dict(mentions=hits, pickup=pk)
    for w in ws:
        print(f"[r10 lex] {tid} '{w}': {len(hits[w])} training texts mention it; pick-up forms: {pk[w]}", flush=True)
        for t in hits[w]:
            print(f"      - {t} | scenes {scenes.get(t)}", flush=True)
os.makedirs(os.path.join(ROOT, "results", "r10"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "r10", "lex.json"), "w", encoding="utf-8"), indent=1)
