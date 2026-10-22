"""B1 stage (d) evaluation: success of the base policies on training cells (parity) and zero-shot on dev test cells.

    --cond free   train cells: the cell's own embedding; dev cells: the nearest embedding (lowest flow loss on the
                  cell's one demonstration, experiments/b1_finetune.nearest_embedding)
    --cond oracle train and dev cells: the cell's factor multi-hot
CPU job; cells sharded with --shard i --nshard n; per-cell JSON in results/b1/eval/<cond>/ (skipped if present).
--read aggregates the per-cell files and records ledger rows b1_policy (one per cond).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifelong_ood import b1, b1_data, b1_eval  # noqa: E402
from experiments.b1_finetune import demo_tensors, load_base, nearest_embedding  # noqa: E402


def cells_of(split, which):
    if which == "train":
        return [("train", c) for c in split["train"]]
    return [(g, c) for g, cs in split["dev"].items() for c in cs]


def run(a):
    split = b1.load_split()
    model, norm, meta = load_base(b1.RESULTS / "policy" / a.cond / "latest.pt")
    torch.set_num_threads(1)
    train_idx = {tuple(c): i for i, c in enumerate(meta["cells"])}
    todo = [x for w in a.which for x in cells_of(split, w)][a.shard::a.nshard]
    out = b1.RESULTS / "eval" / a.cond
    out.mkdir(parents=True, exist_ok=True)
    for grade, cell in todo:
        f = out / f"{b1.cell_name(cell)}.json"
        if f.exists():
            continue
        info = {}
        if a.cond == "oracle":
            cond = dict(task=torch.tensor([b1.multi_hot(cell)], dtype=torch.float32))
        elif cell in train_idx:
            cond = dict(task=torch.tensor([train_idx[cell]]))
        else:
            obs, act, suc = b1_data.load_cell(cell)
            di = int(np.flatnonzero(suc)[0]) if suc.any() else 0
            s, c = demo_tensors(obs[di], act[di], norm)
            k, _ = nearest_embedding(model, s, c)
            info = dict(demo_index=di, nearest_index=k, nearest_cell=meta["cells"][k])
            cond = dict(task=torch.tensor([k]))
        r = b1_eval.rollout(model, norm, cell, cond, a.episodes, seed=1)
        row = dict(cell=list(cell), grade=grade, cond=a.cond, success=float(np.mean(r["success"])), **r, **info)
        f.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        print(grade, b1.cell_name(cell), row["success"], flush=True)


def read(a):
    from lifelong_ood import ledger
    for cond in ("free", "oracle"):
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((b1.RESULTS / "eval" / cond).glob("*.json"))]
        if not rows:
            continue
        by = {}
        for r in rows:
            by.setdefault(r["grade"], []).append(r["success"])
        m = {g: dict(n_cells=len(v), mean=float(np.mean(v)), sem=float(np.std(v) / np.sqrt(len(v)))) for g, v in by.items()}
        done = json.loads((b1.RESULTS / "policy" / cond / "DONE").read_text(encoding="utf-8"))
        eval_s = float(sum(r["seconds"] for r in rows))
        print(cond, json.dumps(m))
        print(ledger.record("b1_policy", bench="composuite", tag="base", seed=0, keys=dict(cond=cond),
                            metrics=dict(success=m, episodes_per_cell=len(rows[0]["success"]),
                                         train_gpu_s=done["gpu_s"], eval_cpu_s=eval_s, final_loss=done["final_loss"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=json.loads((b1.RESULTS / "policy" / cond / "meta.json").read_text(encoding="utf-8"))["args"],
                            script="experiments/b1_eval_policy.py", command=" ".join(sys.argv),
                            artifacts=[str(b1.RESULTS / "eval" / cond)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["free", "oracle"])
    ap.add_argument("--which", nargs="+", default=["train", "dev"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    read(a) if a.read else run(a)


if __name__ == "__main__":
    main()
