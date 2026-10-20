"""Merge X1 CALVIN shard results (results_<start>_<end>.json) and optionally write the ledger row.

python experiments/x1_calvin_merge.py --dir <out dir> [--tag flower|xvla --record --command "..." --config '{...}']
Average length = mean over chains of the number of consecutively completed subtasks (0-5); success at length i = fraction
of chains with >= i completed.  Shards must tile a contiguous range without overlap.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def merge(d):
    rows = [json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in glob.glob(os.path.join(d, "**", "results_*_*.json"),
                                                                                      recursive=True)]
    rows.sort(key=lambda r: r["start"])
    res, covered, secs = [], [], []
    for r in rows:
        assert len(r["results"]) == r["end"] - r["start"], r
        covered.append((r["start"], r["end"]))
        res += r["results"]
        secs.append(r["seconds"])
    for (a0, a1), (b0, b1) in zip(covered, covered[1:]):
        assert a1 == b0, f"shards do not tile: {covered}"
    n = len(res)
    if n == 0:
        raise SystemExit(f"no shard results under {d}")
    sr = [sum(x >= i for x in res) / n for i in range(1, 6)]
    hist = [sum(x == i for x in res) for i in range(6)]
    return dict(n_chains=n, range=[covered[0][0], covered[-1][1]], avg_len=sum(res) / n, chain_sr=sr, hist=hist,
                n_shards=len(rows), shard_seconds_max=max(secs), shard_seconds_sum=sum(secs),
                ep_len=rows[0].get("ep_len"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--command", default="")
    ap.add_argument("--script", default="experiments/x1_calvin_flower_eval.py")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--wall_seconds", type=float, default=None)
    ap.add_argument("--keys", default="{}", help="extra ledger keys (JSON), e.g. the loader variant")
    args = ap.parse_args()
    m = merge(args.dir)
    print(json.dumps(m, indent=1))
    if args.record:
        from lifelong_ood import ledger
        cfg = json.loads(args.config)
        cfg.update(results_dir=str(pathlib.Path(args.dir).resolve()))
        p = ledger.record("x1_calvin_parity", bench="calvin_abc_d", tag=args.tag, seed=0, metrics=m,
                          keys=dict(n_chains=m["n_chains"], ep_len=m["ep_len"], **json.loads(args.keys)),
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=cfg, script=args.script, command=args.command,
                          seconds=args.wall_seconds if args.wall_seconds is not None else m["shard_seconds_max"])
        print("ledger:", p)


if __name__ == "__main__":
    main()
