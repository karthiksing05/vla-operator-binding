"""PROBE R8 stage 0 (docs/PLAN_R8.md): the region lexicon.  Prints the 3 wrong-goal dev tasks' prompts and goal atoms, and every
TRAINING task (the l90c train split; split_sealed.json is never opened) whose goal names a drawer / shelf / cabinet / bowl region,
with D5's training carry observations per task.  -> results/r8/lex.json.  No policy output is computed."""
import json
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1

tt = M1.task_table()
dev = ["libero_90:16", "libero_90:29", "libero_90:40"]
d5 = json.load(open(os.path.join(ROOT, "results", "d5", "index.json"), encoding="utf-8"))
ntr = {}
for o in d5["train"]:
    ntr[o["tid"]] = ntr.get(o["tid"], 0) + 1
out = dict(dev={}, train={})
for t in dev:
    x = tt[t]
    out["dev"][t] = dict(language=x.language, goal=[list(map(str, g)) for g in x.goal], objects=dict(x.objects), fixtures=dict(x.fixtures))
    print("DEV", t, "|", x.language, "|", x.goal, "| fixtures", dict(x.fixtures), flush=True)
for t in sorted(ntr):
    x = tt[t]
    g = json.dumps([list(map(str, q)) for q in x.goal])
    if any(w in g for w in ("drawer", "shelf", "cabinet", "bowl", "top_side", "top_region")):
        out["train"][t] = dict(language=x.language, goal=json.loads(g), fixtures=dict(x.fixtures), n_d5=ntr[t])
        print("TRAIN", t, "|", x.language, "|", x.goal, "| fixtures", dict(x.fixtures), "| d5 obs", ntr[t], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "r8", "lex.json"), "w", encoding="utf-8"), indent=1)
