"""CALVIN-C P1 parity read of a FLOWER-C checkpoint on the standard 1,000 ABC -> D chains (shards from
slurm/jobs/c_flower_eval.sbatch), against a reference run on the same chains (default: flower_calvin_abc with the load fix,
results/x1_calvin/flower_full_ep360_fix1).

Reports
  * the standard average chain length and success at lengths 1-5 over all 1,000 chains (these chains also contain held-out
    cells, so this number mixes training and held-out subtasks: 438-557 of the 1,000 chains, depending on the seed, contain at
    least one held-out cell);
  * the same over the CLEAN chains (no held-out cell anywhere in the chain; windows_s{k}.json), for the run and the reference;
  * per-subtask success (successes / attempts; chain i's first r subtasks succeeded, subtask r failed) for every subtask that
    is NOT a held-out cell of the split, for the run and the reference.
Held-out cells are not scored here (dev cells are read in later probes; sealed cells only at the final read).

    python experiments/c_calvin_parity.py --run <eval dir> --split_seed 0 [--ref <dir>] [--tag flower_c_s0] [--record]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from lifelong_ood import calvin_c as C  # noqa: E402
from x1_calvin_merge import merge  # noqa: E402
from x1_calvin_pertask import per_task  # noqa: E402


def chains(d):
    import glob
    res = {}
    for p in glob.glob(os.path.join(d, "**", "results_*_*.json"), recursive=True):
        r = json.load(open(p, encoding="utf-8"))
        for i, x in enumerate(r["results"]):
            res[r["start"] + i] = int(x)
    return res


def summarise(res, idx):
    xs = [res[i] for i in idx if i in res]
    n = len(xs)
    return dict(n=n, avg_len=sum(xs) / max(n, 1), sr=[sum(x >= k for x in xs) / max(n, 1) for k in range(1, 6)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--ref", default=str(ROOT / "results/x1_calvin/flower_full_ep360_fix1"))
    ap.add_argument("--seqs", default=str(ROOT / "results/x1_calvin/sequences.json"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    seqs = json.load(open(a.seqs, encoding="utf-8"))
    win = json.load(open(C.RESULTS / f"windows_s{a.split_seed}.json", encoding="utf-8"))
    clean = win["clean_chains"]
    allc = list(range(len(seqs)))
    names = sorted({t for _, s in seqs for t in s})
    held = dict(zip(names, C.held_out_mask(names, a.split_seed)))
    train_tasks = [t for t in names if not held[t]]
    out = dict(split_seed=a.split_seed, run=a.run, ref=a.ref, merge=merge(a.run))
    for nm, d in (("run", a.run), ("ref", a.ref)):
        res = chains(d)
        pt = per_task(seqs, d)
        out[nm] = dict(all=summarise(res, allc), clean=summarise(res, clean),
                       per_subtask={t: dict(succ=pt[t][0], n=pt[t][1], rate=pt[t][0] / max(pt[t][1], 1))
                                    for t in train_tasks if t in pt})
    r, f = out["run"], out["ref"]
    common = [t for t in train_tasks if t in r["per_subtask"] and t in f["per_subtask"]]
    out["per_subtask_gap"] = {t: r["per_subtask"][t]["rate"] - f["per_subtask"][t]["rate"] for t in common}
    out["n_train_subtasks"] = len(common)
    out["worst_train_subtasks"] = sorted(out["per_subtask_gap"].items(), key=lambda kv: kv[1])[:8]
    tag = a.tag or os.path.basename(a.run.rstrip("/"))
    p = C.RESULTS / f"parity_{tag}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"all chains   run {r['all']['avg_len']:.3f} (n={r['all']['n']})  ref {f['all']['avg_len']:.3f}  sr run "
          + " ".join(f"{x * 100:.1f}" for x in r["all"]["sr"]))
    print(f"clean chains run {r['clean']['avg_len']:.3f} (n={r['clean']['n']})  ref {f['clean']['avg_len']:.3f}")
    print("| subtask | run | ref |")
    print("|---|---|---|")
    for t in common:
        x, y = r["per_subtask"][t], f["per_subtask"][t]
        print(f"| {t} | {x['succ']}/{x['n']} ({100 * x['rate']:.0f}) | {y['succ']}/{y['n']} ({100 * y['rate']:.0f}) |")
    print("wrote", p)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("c_calvin_parity", bench=C.BENCH, tag=tag, seed=a.split_seed,
                            metrics={k: v for k, v in out.items()},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(run=a.run, ref=a.ref), script="experiments/c_calvin_parity.py",
                            command=" ".join(sys.argv), artifacts=[str(p)]))


if __name__ == "__main__":
    main()
