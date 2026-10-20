"""X3 (EgoDex): per-episode language metadata of TRAINING episodes, read by HTTP range from the CDN zips (no full download).

For each sampled hdf5 member (offsets from x3_egodex_zipdir.py's train_zip_files.json): read its local header, fetch only the first
--head-bytes of the raw-deflate stream, inflate what arrives, pad the inflated prefix with zeros to the member's uncompressed size and
open it with h5py.  The root attributes (llm_description, llm_description2, which_llm_description, llm_verbs, llm_objects, llm_type,
object, environment, task) live in the root object header near the start of the file, so the prefix is enough; when it is not, the
whole member is fetched (counted as `full`).  Frame counts come from the zip-free attribute path only when the dataset header is
readable; otherwise null.  Only metadata is kept (no pose or video data is stored).

    python -I experiments/x3_egodex_trainmeta.py --per-task 60 --all-tasks basic_pick_place --out results/robot/egodex
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import struct
import time
import zlib

import h5py
import numpy as np
import requests

BASE = "https://ml-site.cdn-apple.com/datasets/egodex/"
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))


def rng(url, lo, hi):
    for a in range(6):
        try:
            r = S.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            if r.status_code == 206:
                return r.content
        except Exception:
            pass
        time.sleep(2 + 3 * a)
    raise RuntimeError(f"range failed {url} {lo}-{hi}")


def _s(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    return v if isinstance(v, (str, int, float)) or v is None else str(v)


def read_meta(z, name, off, csize, usize, head):
    url = BASE + z
    lh = rng(url, off, off + 29)
    sig, _, _, meth, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", lh)
    assert sig == 0x04034B50 and meth == 8, (name, hex(sig), meth)
    data0 = off + 30 + nlen + xlen
    full = False
    for nbytes in ([min(head, csize)] + ([csize] if head < csize else [])):
        raw = rng(url, data0, data0 + nbytes - 1)
        d = zlib.decompressobj(-15)
        buf = d.decompress(raw)
        buf = buf[:usize] + b"\0" * (usize - min(len(buf), usize))
        try:
            with h5py.File(io.BytesIO(buf), "r") as f:
                at = {k: _s(f.attrs[k]) for k in f.attrs.keys()}
                try:
                    T = int(f["transforms/camera"].shape[0])
                except Exception:
                    T = None
            if "llm_description" not in at:
                raise KeyError("no attrs")
            return dict(file=name, frames=T, full=full, **at)
        except Exception as e:
            full, err = True, repr(e)
    return dict(file=name, error=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robot/egodex")
    ap.add_argument("--per-task", type=int, default=60)
    ap.add_argument("--all-tasks", default="basic_pick_place", help="task folders read in full (space or + separated)")
    ap.add_argument("--head-bytes", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    files = json.load(open(os.path.join(a.out, "train_zip_files.json")))
    allt = set(a.all_tasks.replace("+", " ").split())
    jobs = []
    for z, lst in files.items():
        by = {}
        for n, off, cs, us, t in lst:
            if n.endswith(".hdf5"):
                by.setdefault(n.split("/")[-2], []).append((n, off, cs, us))
        for task, l in by.items():
            l.sort(key=lambda r: int(os.path.basename(r[0]).split(".")[0]))
            if task in allt or len(l) <= a.per_task:
                pick = l
            else:
                pick = [l[i] for i in np.linspace(0, len(l) - 1, a.per_task).round().astype(int)]
            jobs += [(z, task, *r) for r in pick]
    print("members to read:", len(jobs), flush=True)
    out, t0 = [], time.time()
    with cf.ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(read_meta, z, n, off, cs, us, a.head_bytes): (z, task) for z, task, n, off, cs, us in jobs}
        for i, fu in enumerate(cf.as_completed(futs)):
            z, task = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                r = dict(error=repr(e))
            r.update(zip=z, folder=task)
            out.append(r)
            if i % 500 == 0:
                print(i, round(time.time() - t0), "s", sum("error" in o for o in out), "errors", sum(o.get("full", False) for o in out), "full", flush=True)
    json.dump(out, open(os.path.join(a.out, "train_meta.json"), "w"))
    print("done", len(out), "errors", sum("error" in o for o in out), "full", sum(o.get("full", False) for o in out),
          "s", round(time.time() - t0))


if __name__ == "__main__":
    main()
