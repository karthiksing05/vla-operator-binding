"""Freeze B1's factorial split on CompoSuite (docs/DESIGN.md section 2, docs/B1_COMPOSUITE.md section 3).

Held-out values (whole values, never in training): object Hollowbox and obstacle ObjectWall.  These are two of the
four "restricted" elements of Hussing, Mendez et al. (arXiv 2307.07091, Table 2), chosen before any result so that
G-new can be set beside the paper's restricted-sampling numbers.

Cells (256 = 4^4):
  base   = no held value: 4 robots x 3 objects x 3 obstacles x 4 objectives = 144
  G-rec  = 36 base cells, a balanced quarter: (r + o + b + j) % 4 == 0 over seed-permuted value indices.
           Every seen value appears 27 times in training and every PAIR of values on two axes appears in training,
           so a G-rec cell is a new 3- or 4-way combination of seen values.
  train  = the other 108 base cells
  G-new  = exactly one held value: 48 Hollowbox (obstacle seen) + 48 ObjectWall (object seen) = 96
  G-new2 = both held values: 16
Each grade (G-new stratified by which value is new) is split dev 2/3 / sealed 1/3 with a fixed seed.  The sealed
cells go to results/b1/split_sealed.json; split.json keeps only salted hashes of them so that loaders can refuse a
sealed cell (lifelong_ood/b1.py).  The hashes stop accidents, not a determined reader.

    python experiments/b1_split.py --record
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood import b1  # noqa: E402

SEED = 20261021
HELD = dict(object="Hollowbox", obstacle="ObjectWall")


def build(seed=SEED):
    rng = random.Random(seed)
    seen = {a: [v for v in b1.VALUES[a] if HELD.get(a) != v] for a in b1.AXES}
    perm = {a: rng.sample(range(len(seen[a])), len(seen[a])) for a in b1.AXES}
    base, rec = [], []
    for cell in itertools.product(*(seen[a] for a in b1.AXES)):
        idx = [perm[a][seen[a].index(v)] for a, v in zip(b1.AXES, cell)]
        (rec if sum(idx) % 4 == 0 else base).append(cell)
    train = base
    new_obj, new_obs, new2 = [], [], []
    for cell in itertools.product(*(b1.VALUES[a] for a in b1.AXES)):
        ho, hb = cell[1] == HELD["object"], cell[2] == HELD["obstacle"]
        if ho and hb:
            new2.append(cell)
        elif ho:
            new_obj.append(cell)
        elif hb:
            new_obs.append(cell)
    strata = {"G-rec": [rec], "G-new": [new_obj, new_obs], "G-new2": [new2]}
    dev, sealed = {}, {}
    for g, groups in strata.items():
        dev[g], sealed[g] = [], []
        for grp in groups:
            grp = sorted(grp)
            rng.shuffle(grp)
            k = round(len(grp) / 3)
            sealed[g] += sorted(grp[:k])
            dev[g] += sorted(grp[k:])
    return sorted(train), dev, sealed, seen, perm


def checks(train, dev, sealed):
    allc = set(itertools.product(*(b1.VALUES[a] for a in b1.AXES)))
    test = [c for g in dev for c in dev[g]] + [c for g in sealed for c in sealed[g]]
    assert len(set(train)) == len(train) and len(set(test)) == len(test)
    assert set(train).isdisjoint(test) and set(train) | set(test) == allc, "cells must partition the 256 tasks"
    tv = {a: {c[i] for c in train} for i, a in enumerate(b1.AXES)}
    out = {}
    for i, a in enumerate(b1.AXES):
        out[f"train_values_{a}"] = sorted(tv[a])
    assert HELD["object"] not in tv["object"] and HELD["obstacle"] not in tv["obstacle"]
    for g in ("G-rec", "G-new", "G-new2"):
        for c in dev[g] + sealed[g]:
            unseen = sum(c[i] not in tv[a] for i, a in enumerate(b1.AXES))
            assert unseen == {"G-rec": 0, "G-new": 1, "G-new2": 2}[g], (g, c, unseen)
    # every pair of seen values on two axes co-occurs in training
    missing = 0
    for (i, a), (j, bb) in itertools.combinations(enumerate(b1.AXES), 2):
        have = {(c[i], c[j]) for c in train}
        missing += sum((x, y) not in have for x in tv[a] for y in tv[bb])
    out["seen_pairs_missing_from_train"] = missing
    out["value_counts_train"] = {a: {v: sum(c[i] == v for c in train) for v in sorted(tv[a])}
                                 for i, a in enumerate(b1.AXES)}
    return out


def main():
    train, dev, sealed, seen, perm = build()
    chk = checks(train, dev, sealed)
    os.makedirs(b1.RESULTS, exist_ok=True)
    if b1.SPLIT.exists() and "--force" not in sys.argv:
        old = json.loads(b1.SPLIT.read_text(encoding="utf-8"))
        assert [tuple(c) for c in old["train"]] == train, "split.json exists and differs: the split is frozen"
        print("split.json exists and matches the rebuild; not rewriting")
        return
    salt = hashlib.sha256(f"b1-{SEED}".encode()).hexdigest()[:16]
    sealed_obj = dict(note="SEALED: read once at the final read, via lifelong_ood.b1.load_sealed(prereg=...)",
                      seed=SEED, sealed={g: [list(c) for c in cs] for g, cs in sealed.items()})
    b1.SEALED.write_text(json.dumps(sealed_obj, indent=1) + "\n", encoding="utf-8")
    split = dict(
        benchmark="CompoSuite (Mendez et al. 2022; offline expert data Hussing, Mendez et al. arXiv 2307.07091)",
        seed=SEED, held_out=HELD, axes=list(b1.AXES), values={a: list(v) for a, v in b1.VALUES.items()},
        rule="G-rec = base cells with sum of seed-permuted value indices % 4 == 0; dev/sealed = 2/3 / 1/3 per stratum",
        perm=perm, train=[list(c) for c in train], dev={g: [list(c) for c in cs] for g, cs in dev.items()},
        n_sealed={g: len(cs) for g, cs in sealed.items()}, hash_salt=salt,
        sealed_hashes=sorted(b1.cell_hash(c, salt) for g in sealed for c in sealed[g]),
        sealed_sha256=hashlib.sha256(b1.SEALED.read_bytes()).hexdigest(), checks=chk)
    b1.SPLIT.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    n = dict(train=len(train), **{f"dev_{g}": len(v) for g, v in dev.items()},
             **{f"sealed_{g}": len(v) for g, v in sealed.items()})
    print(json.dumps(n), "| seen pairs missing from train:", chk["seen_pairs_missing_from_train"])
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("b1_split", bench="composuite", tag="frozen", seed=SEED,
                            metrics=dict(counts=n, held_out=HELD, checks=chk,
                                         split_sha256=hashlib.sha256(b1.SPLIT.read_bytes()).hexdigest(),
                                         sealed_sha256=split["sealed_sha256"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(seed=SEED, held_out=HELD, rule=split["rule"]),
                            script="experiments/b1_split.py", command=" ".join(sys.argv),
                            artifacts=[str(b1.SPLIT), str(b1.SEALED)]))


if __name__ == "__main__":
    main()
