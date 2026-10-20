"""X1: the graded compositional split of CALVIN ABC -> D, read from the ABC training language annotations.

Inputs (all small): task_ABC_D/training/lang_annotations/auto_lang_ann.npy (64-frame windows: info.indx = [start, end] global
frame ids, language.task = subtask names), training/scene_info.npy (scene name -> [first, last] frame id),
training/ep_start_end_ids.npy (contiguous recording segments), and the 1,000 standard evaluation chains
(results/x1_calvin/sequences.json, CALVIN's seeded get_sequences(1000); experiments/x1_calvin_pertask.py writes it).

Reads:
  1. which of the 34 subtasks occur in which training scene (window counts per scene);
  2. which ORDERED subtask pairs occur consecutively in training: window j follows window i in the same recording segment, with
     start_j >= end_i - OVERLAP and start_j <= end_i + GAP (the next annotated window after i; primary GAP = 64 frames, also
     reported for 0 / 32 / 128 / 256);  the pair (task_i, task_j) with task_i != task_j counts (also reported with equal ones);
  3. for each evaluation chain (scene D, 5 subtasks, 4 consecutive pairs): the number of pairs never seen consecutively in
     training (G1 relation on the chain).  Scene D itself is unseen in training: every chain is G3 (scene) besides.
  4. with per-chain results of the X1 parity runs: success of step k+1 given step k was reached, split by whether the pair
     (subtask_k, subtask_k+1) was seen consecutively in training.

python experiments/x1_calvin_split.py --data <task_ABC_D> --seqs results/x1_calvin/sequences.json --runs <run dir> ... [--record]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

OVERLAP = 32


def load(data):
    tr = pathlib.Path(data) / "training"
    lang = np.load(tr / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    idx = np.asarray(lang["info"]["indx"])
    tasks = list(lang["language"]["task"])
    scene = np.load(tr / "scene_info.npy", allow_pickle=True).item()
    segs = np.load(tr / "ep_start_end_ids.npy")
    return idx, tasks, scene, segs


def scene_of(frame, scene):
    for k, (a, b) in scene.items():
        if a <= frame <= b:
            return k
    return "unknown"


def seg_of(frame, segs):
    for i, (a, b) in enumerate(segs):
        if a <= frame <= b:
            return i
    return -1


def consecutive_pairs(idx, tasks, segs, gap, same=False):
    order = np.argsort(idx[:, 0], kind="stable")
    seg = np.array([seg_of(int(s), segs) for s in idx[:, 0]])
    pairs = Counter()
    for n, i in enumerate(order):
        e = idx[i, 1]
        for j in order[n + 1:]:
            if idx[j, 0] > e + gap or seg[j] != seg[i]:
                break
            if idx[j, 0] >= e - OVERLAP:
                if same or tasks[i] != tasks[j]:
                    pairs[(tasks[i], tasks[j])] += 1
                break                                       # only the next qualifying window
    return pairs


def chain_results(d):
    res = {}
    for p in glob.glob(os.path.join(d, "**", "results_*_*.json"), recursive=True):
        r = json.load(open(p, encoding="utf-8"))
        for i, x in enumerate(r["results"]):
            res[r["start"] + i] = x
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="results/x1_calvin/split.json")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()

    idx, tasks, scene, segs = load(a.data)
    seqs = json.load(open(a.seqs, encoding="utf-8"))
    eval_tasks = sorted({t for _, s in seqs for t in s})
    all_tasks = sorted(set(tasks) | set(eval_tasks))

    # 1. subtask x scene
    sc = [scene_of(int(s), scene) for s in idx[:, 0]]
    by_scene = defaultdict(Counter)
    for t, s in zip(tasks, sc):
        by_scene[t][s] += 1
    scenes = sorted(scene)
    missing = {s: [t for t in all_tasks if by_scene[t][s] == 0] for s in scenes}

    # 2. consecutive ordered pairs
    sweeps = {}
    for gap in (0, 32, 64, 128, 256):
        p = consecutive_pairs(idx, tasks, segs, gap)
        sweeps[gap] = p
    seen = set(sweeps[64])
    n_possible = len(all_tasks) * (len(all_tasks) - 1)

    # 3. evaluation chains
    per_chain, pair_counts, eval_pairs = [], Counter(), Counter()
    unseen_by_gap = {}
    for gap, p in sweeps.items():
        s_ = set(p)
        unseen_by_gap[gap] = float(np.mean([sum((x, y) not in s_ for x, y in zip(s[:-1], s[1:])) for _, s in seqs]))
    for _, s in seqs:
        prs = list(zip(s[:-1], s[1:]))
        u = [pr not in seen for pr in prs]
        per_chain.append(u)
        for pr, uu in zip(prs, u):
            eval_pairs[pr] += 1
    n_unseen = Counter(sum(u) for u in per_chain)
    distinct_eval = set(eval_pairs)
    first_unseen = [next((k for k, uu in enumerate(u) if uu), None) for u in per_chain]

    # 4. step success by pair grade
    runs = {}
    for d in a.runs:
        res = chain_results(d)
        tab = {"seen": [0, 0], "unseen": [0, 0]}
        by_pos = {k: {"seen": [0, 0], "unseen": [0, 0]} for k in range(1, 5)}
        for i, (_, s) in enumerate(seqs):
            r = res.get(i)
            if r is None:
                continue
            for k in range(1, 5):                     # step k+1 (0-based k) is attempted iff r >= k
                if r < k:
                    break
                g = "unseen" if per_chain[i][k - 1] else "seen"
                ok = int(r >= k + 1)
                tab[g][0] += ok
                tab[g][1] += 1
                by_pos[k][g][0] += ok
                by_pos[k][g][1] += 1
        runs[os.path.basename(d.rstrip("/"))] = dict(
            step_success={g: dict(succ=v[0], n=v[1], rate=v[0] / max(v[1], 1)) for g, v in tab.items()},
            by_position={k: {g: dict(succ=v[0], n=v[1], rate=v[0] / max(v[1], 1)) for g, v in d_.items()}
                         for k, d_ in by_pos.items()})

    out = dict(
        n_windows=int(len(tasks)), n_subtasks_train=len(set(tasks)), n_subtasks_eval=len(eval_tasks),
        scenes={k: [int(v[0]), int(v[1])] for k, v in scene.items()},
        windows_per_scene=dict(Counter(sc)),
        subtask_by_scene={t: {s: int(by_scene[t][s]) for s in scenes} for t in all_tasks},
        subtasks_missing_per_scene=missing,
        pairs=dict(gap=64, overlap=OVERLAP, n_distinct_seen=len(seen), n_possible_ordered=n_possible,
                   n_distinct_by_gap={g: len(p) for g, p in sweeps.items()},
                   n_transitions_by_gap={g: int(sum(p.values())) for g, p in sweeps.items()}),
        eval=dict(n_chains=len(seqs), n_distinct_pairs=len(distinct_eval),
                  n_distinct_pairs_unseen=sum(pr not in seen for pr in distinct_eval),
                  pair_instances=int(sum(eval_pairs.values())),
                  pair_instances_unseen=int(sum(c for pr, c in eval_pairs.items() if pr not in seen)),
                  chains_by_n_unseen_pairs={int(k): int(v) for k, v in sorted(n_unseen.items())},
                  mean_unseen_pairs_per_chain=float(np.mean([sum(u) for u in per_chain])),
                  mean_unseen_pairs_per_chain_by_gap=unseen_by_gap,
                  chains_first_unseen_at_pair={str(k): int(v) for k, v in Counter(first_unseen).items()},
                  unseen_pairs=sorted([f"{x} -> {y}" for x, y in distinct_eval if (x, y) not in seen]),
                  scene_grade="G3 (scene D is not in the training split)"),
        runs=runs,
        per_chain_unseen=[[int(x) for x in u] for u in per_chain])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: v for k, v in out.items() if k not in ("subtask_by_scene", "per_chain_unseen")}
    brief["eval"] = {k: v for k, v in out["eval"].items() if k != "unseen_pairs"}
    print(json.dumps(brief, indent=1))
    print("unseen eval pairs:", len(out["eval"]["unseen_pairs"]))
    for p in out["eval"]["unseen_pairs"][:60]:
        print("  ", p)
    print("subtask x scene:")
    for t in all_tasks:
        print(f"  {t:28s}", " ".join(f"{by_scene[t][s]:5d}" for s in scenes))
    if a.record:
        from lifelong_ood import ledger
        m = {k: v for k, v in out.items() if k != "per_chain_unseen"}
        p = ledger.record("x1_calvin_split", bench="calvin_abc_d", tag="split", seed=0, metrics=m,
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=dict(data=a.data, gap=64, overlap=OVERLAP, runs=a.runs, out=a.out),
                          script="experiments/x1_calvin_split.py", command=" ".join(sys.argv))
        print("ledger:", p)


if __name__ == "__main__":
    main()
