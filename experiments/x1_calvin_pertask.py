"""Per-subtask success of X1 CALVIN runs.  Chain i's first `result` subtasks succeeded and subtask `result` failed, over
CALVIN's seeded get_sequences(1000) (FLOWER's copy of calvin_agent's multistep_sequences.py, identical but for its import).

python experiments/x1_calvin_pertask.py --flower_root <flower_vla_calvin> --cache results/x1_calvin/sequences.json <run dir> ...
The sequence list is computed once (a few minutes, numpy only) and cached.
"""
import argparse
import contextlib
import glob
import importlib.util
import json
import os
import sys
import types
from collections import Counter

import numpy as np


def sequences(flower_root, cache):
    if cache and os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))

    @contextlib.contextmanager
    def temp_seed(seed):
        st = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(st)
    for name in ("flower", "flower.evaluation"):
        sys.modules.setdefault(name, types.ModuleType(name))
    u = types.ModuleType("flower.evaluation.utils")
    u.temp_seed = temp_seed
    sys.modules["flower.evaluation.utils"] = u
    spec = importlib.util.spec_from_file_location("x1_ms", os.path.join(flower_root, "flower/evaluation/multistep_sequences.py"))
    ms = importlib.util.module_from_spec(spec)
    sys.modules["x1_ms"] = ms
    spec.loader.exec_module(ms)
    seqs = [[s[0], list(s[1])] for s in ms.get_sequences(1000)]
    if cache:
        json.dump(seqs, open(cache, "w", encoding="utf-8"))
    return seqs


def per_task(seqs, d):
    res = {}
    for p in glob.glob(os.path.join(d, "**", "results_*_*.json"), recursive=True):
        r = json.load(open(p, encoding="utf-8"))
        for i, x in enumerate(r["results"]):
            res[r["start"] + i] = x
    ok, fail = Counter(), Counter()
    for i, (_, seq) in enumerate(seqs):
        if i not in res:
            continue
        x = res[i]
        for t in seq[:x]:
            ok[t] += 1
        if x < 5:
            fail[seq[x]] += 1
    return {t: (ok[t], ok[t] + fail[t]) for t in set(ok) | set(fail)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flower_root", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    seqs = sequences(a.flower_root, a.cache)
    out = {os.path.basename(d.rstrip("/")): per_task(seqs, d) for d in a.dirs}
    names = list(out)
    print("| subtask | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for t in sorted(set().union(*out.values())):
        cells = [f"{out[n][t][0]}/{out[n][t][1]} ({100 * out[n][t][0] / out[n][t][1]:.0f})" if t in out[n] else "-" for n in names]
        print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
