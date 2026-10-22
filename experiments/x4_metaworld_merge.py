"""Merge X4 Meta-World per-task results (<task>.json from x4_metaworld_pi0_eval.py) and optionally write the ledger row.

Scores as RLinf's metaworld_eval.py does: per-difficulty mean of per-task rates, "Avg." = the mean of the four difficulty
means (so 77.9 / 51.8 / 53.3 / 20.0 -> 50.8), plus the pooled success rate over all episodes.

python experiments/x4_metaworld_merge.py --dir <out dir> [--tag pi0_sft --record --keys '{"max_steps": 160}']
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from x4_metaworld_split import DIFFICULTY  # noqa: E402

PUBLISHED = {   # RLinf docs (examples/embodied/metaworld.rst) and pi_RL (arXiv 2510.25889) Table 6, MT50, % success
    "pi0_sft": dict(easy=77.9, medium=51.8, hard=53.3, very_hard=20.0, avg=50.8),
    "pi05_sft": dict(easy=68.2, medium=37.3, hard=41.7, very_hard=28.0, avg=43.8),
}


def summarise(d):
    rows = {}
    for p in sorted(pathlib.Path(d).glob("*-v3.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows[r["task"][:-3]] = r
    per = {}
    for diff, ts in DIFFICULTY.items():
        have = [t for t in ts if t in rows]
        if have:
            k = sum(rows[t]["successes"] for t in have)
            n = sum(rows[t]["trials"] for t in have)
            per[diff] = dict(tasks=len(have), of=len(ts), mean_task_rate=100 * sum(rows[t]["rate"] for t in have) / len(have),
                             successes=k, episodes=n, se=100 * math.sqrt(max(k / n * (1 - k / n), 1e-9) / n))
    k = sum(r["successes"] for r in rows.values())
    n = sum(r["trials"] for r in rows.values())
    full = len(per) == 4 and all(v["tasks"] == v["of"] for v in per.values())
    return dict(n_tasks=len(rows), episodes=n, successes=k, pooled=100 * k / max(n, 1),
                avg_of_difficulties=(sum(v["mean_task_rate"] for v in per.values()) / 4) if full else None,
                by_difficulty=per, per_task={t: r["successes"] for t, r in rows.items()},
                config=next(iter(rows.values()))["config"] if rows else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="pi0_sft")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--keys", default="{}")
    ap.add_argument("--command", default="")
    a = ap.parse_args()
    m = summarise(a.dir)
    pub = PUBLISHED.get(a.tag)
    print(f"tasks {m['n_tasks']}  episodes {m['episodes']}  pooled {m['pooled']:.1f}%  avg {m['avg_of_difficulties']}")
    for diff, v in m["by_difficulty"].items():
        print(f"  {diff:9s} {v['tasks']:2d}/{v['of']:2d} tasks  {v['mean_task_rate']:5.1f}%  (+- {v['se']:.1f})"
              + (f"   published {pub[diff]}" if pub else ""))
    if pub:
        print("  published avg", pub["avg"])
        m["published"] = pub
    if a.record:
        sys.path.insert(0, str(REPO))
        from lifelong_ood import ledger
        keys = json.loads(a.keys)
        p = ledger.record("x4_metaworld_parity", bench="metaworld_mt50", tag=a.tag, seed=0, metrics=m, keys=keys,
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=dict(dir=a.dir, **(m["config"] or {})), script="experiments/x4_metaworld_merge.py",
                          command=a.command or " ".join(sys.argv))
        print("ledger:", p)


if __name__ == "__main__":
    main()
