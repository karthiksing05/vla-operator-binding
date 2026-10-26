"""PLAN_SOLVE S1b: why training examples drop (no detector box holding >= 0.5 of the target) -- per role x body phrase, TRAINING only."""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import s1b_detect as SD, s1b_train as T, s1_train as ST
lab = json.load(open(os.path.join(ROOT, "results/bind/s1/labels.json")))
det = json.load(open(os.path.join(ROOT, "results/bind/s1b/det.json")))
lex, det = det["lexicon"], det["det"]
cache = {}
def zc(f):
    if f not in cache:
        cache[f] = np.load(os.path.join(ROOT, f), allow_pickle=True)
    return cache[f]
st = defaultdict(lambda: Counter())
for e in lab["train"]:
    body, rel = SD.split(e["phrase"], lex)
    boxes = det.get(f"{e['frame']}|{body}", [])
    k = (e["role"], body, e["phrase"])
    if not boxes:
        st[k]["nobox"] += 1; continue
    t = ST.target_of(e, zc)
    parts = [ST.target_of(dict(e, targets=[n]), zc) for n in e["targets"]]
    s1 = max(T.share(b, q, False) for b in boxes for q in parts); s2 = max(T.share(b, q, True) for b in boxes for q in parts)
    st[k]["ok" if max(s1, s2) >= 0.5 else ("lowshare" if t.sum() > 0 else "emptytarget")] += 1
for k, c in sorted(st.items(), key=lambda kv: -(kv[1]["nobox"] + kv[1]["lowshare"] + kv[1]["emptytarget"])):
    bad = c["nobox"] + c["lowshare"] + c["emptytarget"]
    if bad:
        print(k, dict(c), flush=True)
