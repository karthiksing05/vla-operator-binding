"""X3 (EgoDex) stage 2a: the fetch manifest -- which training episodes to fetch, and the byte range of each.

From split_proposal.json (the 90 training folders and the per-folder caps of the plan), train_zip_files.json (member offsets)
and train_meta.json (basic_pick_place's per-episode object / target), one row per episode: folder, episode, zip, and the
contiguous byte range covering its hdf5 and mp4 members (adjacent in the zips: N.hdf5 then N.mp4).  basic_pick_place takes
CAP_BPP evenly spaced episodes among those NOT in a held-out pair or object and with both fields parsed; every other folder takes
its cap evenly spaced.  Only main-part zips (part1..5); extra.zip is not used.  Rows are dealt round-robin into shards, so each
fetch worker reads only its own small shard file.

    python -I experiments/x3_egodex_manifest.py --shards 8
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3_egodex_split import (BPP_HELD_OBJECTS, BPP_HELD_PAIRS, CAP_BPP, CAP_DEFAULT, CAP_DONOR, DONORS, EXCLUDE_FROM_TRAIN,  # noqa: E402
                             HELD, _ot)

OUT = "results/robot/egodex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--shards", type=int, default=8)
    a = ap.parse_args()
    prop = json.load(open(os.path.join(a.out, "split_proposal.json")))
    train = set(prop["train"])
    held = {f for g in HELD.values() for f in g}
    assert not (train & held) and not (train & EXCLUDE_FROM_TRAIN)
    files = json.load(open(os.path.join(a.out, "train_zip_files.json")))
    eps = collections.defaultdict(dict)                      # folder -> ep -> {zip, hdf5:(name, off, cs, us), mp4:(...)}
    for z, lst in files.items():
        if z == "extra.zip":
            continue
        for n, off, cs, us, t in lst:
            fo, base = n.split("/")[-2], os.path.basename(n)
            if fo not in train:
                continue
            ep, ext = base.split(".")
            d = eps[fo].setdefault(ep, dict(zip=z))
            d[ext] = (n, off, cs, us)
    # basic_pick_place: the allowed episodes
    meta = json.load(open(os.path.join(a.out, "train_meta.json")))
    hp, ho = {tuple(p) for p in BPP_HELD_PAIRS}, set(BPP_HELD_OBJECTS)
    allowed, why = set(), collections.Counter()
    for m in meta:
        if m.get("folder") != "basic_pick_place" or "error" in m or m.get("zip") == "extra.zip":
            continue
        o, t = _ot(m)
        ep = os.path.basename(m["file"]).split(".")[0]
        if o is None or t is None:
            why["unparsed"] += 1
        elif (o, t) in hp:
            why["held_pair"] += 1
        elif o in ho:
            why["held_object"] += 1
        else:
            allowed.add(ep)
    rows = []
    for fo in sorted(eps):
        ks = sorted((k for k, d in eps[fo].items() if "hdf5" in d and "mp4" in d), key=int)
        if fo == "basic_pick_place":
            ks = [k for k in ks if k in allowed]
            cap = CAP_BPP
        else:
            cap = CAP_DONOR if fo in DONORS else CAP_DEFAULT
        if len(ks) > cap:
            ks = [ks[i] for i in np.linspace(0, len(ks) - 1, cap).round().astype(int)]
        for k in ks:
            d = eps[fo][k]
            h, m = d["hdf5"], d["mp4"]
            lo, hi = min(h[1], m[1]), max(h[1], m[1])
            first, second = (h, m) if h[1] < m[1] else (m, h)
            # end of the later member: header (30 + name + extra <= 30 + name + 256) + data, +64 for a data descriptor
            end = second[1] + 30 + len(second[0].encode()) + 256 + second[2] + 64
            rows.append(dict(folder=fo, ep=k, zip=d["zip"], lo=lo, hi=end, hdf5=h, mp4=m))
    tot = sum(r["hdf5"][2] + r["mp4"][2] for r in rows)
    os.makedirs(os.path.join(a.out, "fetch"), exist_ok=True)
    for s in range(a.shards):
        json.dump(rows[s::a.shards], open(os.path.join(a.out, "fetch", f"shard{s}.json"), "w"))
    summ = dict(episodes=len(rows), folders=len(eps), gb=round(tot / 1e9, 2), shards=a.shards,
                bpp=dict(allowed=len(allowed), dropped=dict(why)),
                per_folder={fo: sum(r["folder"] == fo for r in rows) for fo in sorted(eps)})
    json.dump(summ, open(os.path.join(a.out, "fetch", "manifest_summary.json"), "w"), indent=1)
    print({k: v for k, v in summ.items() if k != "per_folder"})


if __name__ == "__main__":
    main()
