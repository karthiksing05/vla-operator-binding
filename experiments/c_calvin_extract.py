"""CALVIN-C P1 data prep (SLURM CPU job): check task_ABC_D against the zip's central directory, then write the
`extracted/` files that FLOWER's CALVIN recipe reads (use_extracted_rel_actions: true).

1. CHECK.  experiments/x1_calvin_fetch.py extracted task_ABC_D.zip member by member; its <data root>/_index.pkl lists every
   member (offset, name, compressed size, size, method).  Every member under task_ABC_D/{training,validation}/ must exist at
   its full size (one os.stat per member, in a process pool; no directory listing).
2. EXTRACT.  The output is byte-for-byte the format of the authors' preprocess/extract_by_key.py (flower_vla_calvin acb32ee):
   <split>/extracted/ep_npz_names.list = the sorted episode ids, one per line; <split>/extracted/ep_rel_actions.npy = the
   stacked `rel_actions` of those episodes in that order.  The authors list the directory (os.listdir over ~1.8M files); the
   episode ids here come from the index instead, after step 1 has shown every indexed episode file is present.  Written to
   a temporary name and renamed.  Then the authors' own spot check: 13 random episodes re-read and compared.

    python experiments/c_calvin_extract.py --root /coc/flash10/ksingara3/robot/data/calvin_abc --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = None


def _stat(row):
    name, size = row
    p = os.path.join(ROOT, name)
    try:
        s = os.stat(p).st_size
    except FileNotFoundError:
        return name, -1
    return name, s if s != size else None


def _rel(path):
    with np.load(path, allow_pickle=True) as z:
        return np.array(z["rel_actions"])


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/coc/flash10/ksingara3/robot/data/calvin_abc")
    ap.add_argument("--task", default="task_ABC_D")
    ap.add_argument("--splits", default="training,validation")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="results/calvin_c/data_check.json")
    a = ap.parse_args()
    ROOT = a.root
    t0 = time.time()
    rows = pickle.load(open(os.path.join(a.root, "_index.pkl"), "rb"))
    report = dict(root=a.root, n_index_members=len(rows))
    with Pool(a.workers) as pool:
        for split in a.splits.split(","):
            pre = f"{a.task}/{split}/"
            mem = [(r[1], r[3]) for r in rows if r[1].startswith(pre)]
            bad = [x for x in pool.imap_unordered(_stat, mem, chunksize=2000) if x[1] is not None]
            eps = sorted(int(n[len(pre) + 8:-4]) for n, _ in mem
                         if n[len(pre):].startswith("episode_") and n.endswith(".npz"))
            r = dict(n_members=len(mem), n_episodes=len(eps), n_missing=sum(b[1] == -1 for b in bad),
                     n_size_mismatch=sum(b[1] != -1 for b in bad), bad_examples=bad[:20],
                     first=eps[0] if eps else None, last=eps[-1] if eps else None, check_seconds=time.time() - t0)
            print(split, json.dumps(r), flush=True)
            report[split] = r
            if bad:
                print(f"[extract] {split}: {len(bad)} members missing or short; not extracting", flush=True)
                continue
            d = os.path.join(a.root, a.task, split)
            out = os.path.join(d, "extracted")
            npy, lst = os.path.join(out, "ep_rel_actions.npy"), os.path.join(out, "ep_npz_names.list")
            if os.path.exists(npy) and os.path.exists(lst):
                print(f"[extract] {split}: {npy} exists; not rewriting", flush=True)
            else:
                os.makedirs(out, exist_ok=True)
                t1 = time.time()
                paths = [os.path.join(d, f"episode_{i:07d}.npz") for i in eps]
                acts = np.stack(list(pool.imap(_rel, paths, chunksize=500)), axis=0)
                with open(npy + ".tmp.npy", "wb") as f:
                    np.save(f, acts)
                with open(lst + ".tmp", "w", encoding="utf-8") as f:
                    f.write("\n".join(str(x) for x in eps))
                os.replace(npy + ".tmp.npy", npy)
                os.replace(lst + ".tmp", lst)
                r.update(extract_seconds=time.time() - t1, shape=list(acts.shape), dtype=str(acts.dtype))
                print(f"[extract] {split}: {acts.shape} {acts.dtype} in {time.time() - t1:.0f}s", flush=True)
            # the authors' check: random episodes re-read against the extracted rows
            acts = np.load(npy, mmap_mode="r")
            names = [int(x) for x in open(lst, encoding="utf-8").read().split()]
            pos = {n: i for i, n in enumerate(names)}
            rng = np.random.default_rng(0)
            pick = [0, len(eps) // 2, len(eps) - 1] + rng.integers(0, len(eps), 10).tolist()
            err = max(float(np.abs(_rel(os.path.join(d, f"episode_{eps[i]:07d}.npz")) - acts[pos[eps[i]]]).max())
                      for i in pick)
            r.update(spot_check_maxdiff=err, n_names=len(names))
            print(f"[extract] {split}: spot check max diff {err}", flush=True)
    report["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    ok = all(report[s]["n_missing"] == 0 and report[s]["n_size_mismatch"] == 0 for s in a.splits.split(","))
    print("[extract] OK" if ok else "[extract] DATA INCOMPLETE", flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
