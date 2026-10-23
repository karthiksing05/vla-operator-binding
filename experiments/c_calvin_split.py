"""P0 of plan v4 for CALVIN-C (docs/PLAN_V4.md 3.2, docs/DESIGN.md 3.2, docs/CALVIN_C.md): freeze three split seeds of
CALVIN's 21 block cells, and build each seed's training-window filter over ABC's language annotations.

CELLS (lifelong_ood/calvin_c.py): lift colour x place (9), push colour x direction (6), rotate colour x direction (6).
ATOMS: verb, colour, place (lift only), direction (push and rotate share it).
COMPOUNDS of a cell (v, c, s): verb-colour (v, c), verb-slot (v, s), colour-slot (c, s).  Colour-direction compounds are shared
by push and rotate; a colour-place compound belongs to one lift cell only, so a held-out lift cell always has an unseen
colour-place compound.  CAVEAT: CALVIN's colour also fixes the block's size (red middle, blue small, pink big), so the
colour atom is colour+size.

RULES (fixed before any result).  For split seed k, a held-out set H (the rest of the 21 cells train) must satisfy
  * |H| = 6 or 7;
  * every colour, verb, place and direction occurs in >= 2 TRAINING cells (atoms are never new);
  * PART set: exactly one colour x verb pair, PART[k], is held out entirely (all its cells in H); no other verb-colour
    compound is absent from training;
  * GRADE of a held-out cell = the number of its three compounds absent from training; every grade is 0, 1 or 2;
and among the H that satisfy them, the search maximises (1) the smallest per-grade count over grades 0 / 1 / 2, (2) then
minimises the spread between the largest and smallest grade counts, (3) then prefers |H| = 7, (4) then prefers more verbs among
the non-part held-out cells, (5) then minimises the atom divergence D_A (below); ties are broken by a seeded shuffle.  PART: seed 0 takes the plan's example (pink, rotate); seeds
1 and 2 take the other two verbs and colours in a seeded order, so across the three seeds each verb and each colour is the
part once.
DIVERGENCES (Keysers et al. arXiv 1912.09713): D_A = 1 - C_0.5 over atom frequencies, D_C = 1 - C_0.1 over pairwise-compound
frequencies, training cells vs held-out cells (and per grade); D_A is the last search criterion, D_C is reported only.
SEALED: round(|H| / 3) held-out cells per seed, stratified by grade (floor(n_g / 3) per grade, the rest to the grades with the
largest remainder; never a grade's or the part set's last dev cell), chosen by the smallest salted sha256 within a grade, go to
results/calvin_c/split_s{k}_sealed.json (never committed); split_s{k}.json keeps salted hashes so readers refuse them
(lifelong_ood/calvin_c.py).

WINDOWS (`windows` subcommand, a SLURM CPU job: it reads the 38 MB annotation file).  FLOWER's CALVIN recipe trains on the
language-annotated windows of task_ABC_D/training/lang_clip_resnet50/auto_lang_ann.npy (64-frame windows [start, end]; every
frame start..end-10 of a window is a sample start).  For seed k: drop every window whose subtask is a held-out cell (dev or
sealed, via calvin_c.held_out_mask), then drop every other window that overlaps a held-out window by more than OVERLAP = 16
frames.  Writes results/calvin_c/windows_s{k}.npz (keep: bool per annotation) and windows_s{k}.json (counts; frames of
held-out windows still inside a kept window, which is at most 16 per pair; chains of the 1,000 evaluation chains free of any
held-out cell, for a clean parity read).  Unannotated play (frames outside every window) is not read by the recipe's
language-only dataset, but flower_vla_pret's pretraining is a separate question (the flow-loss audit).

    python experiments/c_calvin_split.py split [--dry] [--record]
    python experiments/c_calvin_split.py windows --seed 0 [--data <task_ABC_D>] [--record]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood import calvin_c as C  # noqa: E402

SEED = 20261022
PART0 = ("pink", "rotate")
OVERLAP = 16
SIZES = (6, 7)


def part_pairs(seed=SEED):
    """(colour, verb) part pair per split seed: seed 0 = PART0, seeds 1 and 2 = a seeded matching of the other verbs and
    colours."""
    rng = random.Random(seed)
    cols = [c for c in C.COLOURS if c != PART0[0]]
    verbs = [v for v in C.VERBS if v != PART0[1]]
    rng.shuffle(cols)
    return {0: PART0, 1: (cols[0], verbs[0]), 2: (cols[1], verbs[1])}


def grades(held):
    train = [c for c in C.CELLS if c not in held]
    seen = {k for c in train for k in C.compounds(c)}
    return {c: sum(k not in seen for k in C.compounds(c)) for c in held}, train


def valid(held, part):
    g, train = grades(held)
    cnt = Counter(a for c in train for a in C.atoms(c))
    all_atoms = {a for c in C.CELLS for a in C.atoms(c)}
    if any(cnt[a] < 2 for a in all_atoms):
        return None
    vc_unseen = {(v, c) for v in C.VERBS for c in C.COLOURS if all(x in held for x in C.CELLS if x[0] == v and x[1] == c)}
    if vc_unseen != {(part[1], part[0])}:
        return None
    if max(g.values()) > 2:
        return None
    return g


def score(held, g, part):
    n = [sum(1 for x in g.values() if x == k) for k in (0, 1, 2)]
    nonpart_verbs = {c[0] for c in held if not (c[0] == part[1] and c[1] == part[0])}
    train = [c for c in C.CELLS if c not in held]
    d_a = 1 - chernoff(freq(train, C.atoms), freq(list(held), C.atoms), 0.5)
    return (min(n), -(max(n) - min(n)), len(held), len(nonpart_verbs), -round(d_a, 6))


def search(k, part, seed=SEED):
    pcells = [c for c in C.CELLS if c[0] == part[1] and c[1] == part[0]]
    rest = [c for c in C.CELLS if c not in pcells]
    best, cands = None, []
    for size in SIZES:
        for extra in itertools.combinations(rest, size - len(pcells)):
            held = tuple(sorted(pcells + list(extra)))
            g = valid(set(held), part)
            if g is None:
                continue
            s = score(held, g, part)
            if best is None or s > best:
                best, cands = s, [held]
            elif s == best:
                cands.append(held)
    cands = sorted(cands)
    random.Random(f"{seed}-{k}").shuffle(cands)
    return list(cands[0]), best, len(cands)


def chernoff(p, q, a):
    keys = set(p) | set(q)
    return sum((p.get(x, 0.0) ** a) * (q.get(x, 0.0) ** (1 - a)) for x in keys)


def freq(cells, fn):
    c = Counter(x for cell in cells for x in fn(cell))
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def divergences(train, test):
    if not test:
        return dict(D_A=None, D_C=None)
    return dict(D_A=1 - chernoff(freq(train, C.atoms), freq(test, C.atoms), 0.5),
                D_C=1 - chernoff(freq(train, C.compounds), freq(test, C.compounds), 0.1))


def seal(held, g, part, salt):
    """-> sealed cells: round(|H| / 3), stratified by grade (see module doc)."""
    k = round(len(held) / 3)
    by = {x: sorted([c for c in held if g[c] == x], key=lambda c: C.cell_hash(c, salt)) for x in (0, 1, 2)}
    take = {x: len(by[x]) // 3 for x in by}
    left = k - sum(take.values())
    order = sorted(by, key=lambda x: (-(len(by[x]) % 3), -len(by[x]), x))
    while left > 0:
        moved = False
        for x in order:
            if left and take[x] < len(by[x]) - 1:        # never a grade's last dev cell
                take[x] += 1
                left -= 1
                moved = True
        if not moved:
            break
    sealed = [c for x in by for c in by[x][:take[x]]]
    pcells = [c for c in held if c[0] == part[1] and c[1] == part[0]]
    if all(c in sealed for c in pcells):                 # keep a dev part cell: swap the last sealed part cell
        p = sorted(pcells, key=lambda c: C.cell_hash(c, salt))[-1]
        alt = [c for c in by[g[p]] if c not in sealed and c not in pcells]
        sealed.remove(p)
        if alt:
            sealed.append(alt[0])
    return sorted(sealed)


def build(seed=SEED):
    out = {}
    for k, part in part_pairs(seed).items():
        held, best, n_tied = search(k, part, seed)
        g, train = grades(set(held))
        salt = hashlib.sha256(f"calvin_c-{seed}-s{k}".encode()).hexdigest()[:16]
        sealed = seal(held, g, part, salt)
        out[k] = dict(part=part, held=held, grades=g, train=sorted(train), sealed=sealed, salt=salt, best=best,
                      n_tied=n_tied)
    return out


def checks(s):
    held, train, g, part = s["held"], s["train"], s["grades"], s["part"]
    assert set(held) | set(train) == set(C.CELLS) and not set(held) & set(train)
    assert len(held) in SIZES
    cnt = Counter(a for c in train for a in C.atoms(c))
    assert all(cnt[a] >= 2 for c in C.CELLS for a in C.atoms(c)), cnt
    g2, _ = grades(set(held))
    assert g2 == g and set(g.values()) <= {0, 1, 2}
    seen = {k for c in train for k in C.compounds(c)}
    assert [(v, c) for v in C.VERBS for c in C.COLOURS if ("vc", v, c) not in seen] == [(part[1], part[0])]
    dev = [c for c in held if c not in s["sealed"]]
    assert any(c[0] == part[1] and c[1] == part[0] for c in dev), "the part set needs a dev cell"
    return dict(atom_counts_train={f"{a[0]}:{a[1]}": cnt[a] for a in sorted(cnt)},
                grade_counts={x: sum(1 for v in g.values() if v == x) for x in (0, 1, 2)},
                divergence_all=divergences(train, held),
                divergence_by_grade={x: divergences(train, [c for c in held if g[c] == x]) for x in (0, 1, 2)},
                unseen_compounds_dev=sorted({"-".join(k) for c in dev for k in C.compounds(c) if k not in seen}),
                n_unseen_compounds_all=len({k for c in held for k in C.compounds(c) if k not in seen}))


def write(seed, s, chk):
    os.makedirs(C.RESULTS, exist_ok=True)
    sp, sl = C.split_path(seed), C.sealed_path(seed)
    part = s["part"]
    seen = {k for t in s["train"] for k in C.compounds(t)}

    def rec(c):
        return dict(cell=list(c), name=C.cell_name(c), grade=s["grades"][c],
                    part=bool(c[0] == part[1] and c[1] == part[0]),
                    unseen=["-".join(k) for k in C.compounds(c) if k not in seen])
    sealed_obj = dict(note="SEALED: read once at the final read, via lifelong_ood.calvin_c.load_sealed(seed, prereg=...)",
                      seed=SEED, split_seed=seed, sealed=[rec(c) for c in s["sealed"]])
    if sp.exists():
        old = json.loads(sp.read_text(encoding="utf-8"))
        assert [tuple(c) for c in old["train"]] == [tuple(c) for c in s["train"]], f"{sp} exists and differs: frozen"
        print(f"{sp.name} exists and matches the rebuild; not rewriting")
        return sp, sl
    tmp = sl.with_suffix(".tmp")
    tmp.write_text(json.dumps(sealed_obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, sl)
    dev = [c for c in s["held"] if c not in s["sealed"]]
    split = dict(
        benchmark="CALVIN-C: CALVIN ABC -> D (Mees et al. 2022) block cells; docs/PLAN_V4.md 3.2",
        seed=SEED, split_seed=seed, part=dict(colour=part[0], verb=part[1],
                                              cells=[C.cell_name(c) for c in s["held"]
                                                     if c[0] == part[1] and c[1] == part[0]]),
        caveat="CALVIN colour also fixes block size (red middle, blue small, pink big): the colour atom is colour+size",
        rule=("|H| in {6,7}; every atom in >= 2 training cells; exactly one verb-colour pair (the part) fully held out; "
              "grade = # of the cell's 3 pairwise compounds absent from training, in {0,1,2}; maximise min per-grade "
              "count, then minimise spread, then |H| = 7, then verbs among non-part cells; seeded tie-break"),
        search=dict(best_score=list(s["best"]), n_tied=s["n_tied"]),
        cells=[C.cell_name(c) for c in C.CELLS], train=[list(c) for c in s["train"]],
        train_names=[C.cell_name(c) for c in s["train"]], dev=[rec(c) for c in dev],
        n_held_out=len(s["held"]), n_sealed=len(s["sealed"]),
        n_sealed_by_grade={x: sum(1 for c in s["sealed"] if s["grades"][c] == x) for x in (0, 1, 2)},
        hash_salt=s["salt"], sealed_hashes=sorted(C.cell_hash(c, s["salt"]) for c in s["sealed"]),
        sealed_sha256=hashlib.sha256(sl.read_bytes()).hexdigest(), checks=chk)
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, sp)
    return sp, sl


def cmd_split(a):
    sets = build()
    rows = {}
    for k, s in sets.items():
        chk = checks(s)
        dev = [c for c in s["held"] if c not in s["sealed"]]
        print(f"split seed {k}: part {s['part']}, held {len(s['held'])} (dev {len(dev)}, sealed {len(s['sealed'])}), "
              f"grades {chk['grade_counts']}, score {s['best']} ({s['n_tied']} tied), D_A {chk['divergence_all']['D_A']:.3f} "
              f"D_C {chk['divergence_all']['D_C']:.3f}")
        for c in dev:
            print(f"    dev  {C.cell_name(c):26s} grade {s['grades'][c]}")
        if a.dry:
            continue
        sp, sl = write(k, s, chk)
        rows[k] = dict(split=str(sp), split_sha256=hashlib.sha256(sp.read_bytes()).hexdigest(),
                       sealed_sha256=hashlib.sha256(sl.read_bytes()).hexdigest(), part=list(s["part"]),
                       n_held_out=len(s["held"]), n_sealed=len(s["sealed"]), grade_counts=chk["grade_counts"],
                       dev=[dict(name=C.cell_name(c), grade=s["grades"][c]) for c in dev],
                       divergence_all=chk["divergence_all"], divergence_by_grade=chk["divergence_by_grade"])
    if a.record and not a.dry:
        from lifelong_ood import ledger
        print(ledger.record("c_calvin_split", bench=C.BENCH, tag="frozen", seed=SEED, metrics=dict(seeds=rows),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(seed=SEED, part0=PART0, sizes=SIZES, overlap=OVERLAP),
                            script="experiments/c_calvin_split.py", command=" ".join(sys.argv),
                            artifacts=[str(C.split_path(k)) for k in rows] + [str(C.sealed_path(k)) for k in rows]))


def overlaps(idx, held_rows):
    """For each window, the largest overlap (frames, inclusive ranges) with any held-out window, and the total number of
    distinct held-out frames inside it."""
    hs = idx[held_rows]
    order = hs[:, 0].argsort()
    hs = hs[order]
    import numpy as np
    best = np.zeros(len(idx), np.int64)
    for i, (s, e) in enumerate(idx):
        lo = np.searchsorted(hs[:, 0], e, side="right")          # held windows starting at or before e
        cand = hs[:lo]
        cand = cand[cand[:, 1] >= s]
        if len(cand):
            ov = np.minimum(cand[:, 1], e) - np.maximum(cand[:, 0], s) + 1
            best[i] = int(ov.max())
    return best


def cmd_windows(a):
    import numpy as np
    tr = os.path.join(a.data, "training")
    lang = np.load(os.path.join(tr, a.lang_folder, "auto_lang_ann.npy"), allow_pickle=True).item()
    idx = np.asarray(lang["info"]["indx"], np.int64)
    tasks = [str(t) for t in lang["language"]["task"]]
    ref = np.load(os.path.join(tr, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    same_windows = bool(np.array_equal(np.asarray(ref["info"]["indx"]), idx)
                        and list(map(str, ref["language"]["task"])) == tasks)
    held = np.array(C.held_out_mask(tasks, a.seed))
    best = overlaps(idx, np.nonzero(held)[0])
    drop_overlap = (~held) & (best > OVERLAP)
    keep = (~held) & ~drop_overlap
    # frames of held-out windows still inside kept windows (sample starts run s..e-10, chunks reach e)
    hf = np.zeros(int(idx[:, 1].max()) + 2, bool)
    for s, e in idx[held]:
        hf[s:e + 1] = True
    kf = np.zeros_like(hf)
    for s, e in idx[keep]:
        kf[s:e + 1] = True
    split = C.load_split(a.seed)
    dev_names = {d["name"] for d in split["dev"]}
    cnt_all = Counter(tasks)
    seqs = json.load(open(a.seqs, encoding="utf-8"))
    hm = set(t for t, h in zip(tasks, held) if h) | {t for _, s in seqs for t in s if C.held_out_mask([t], a.seed)[0]}
    clean = [i for i, (_, s) in enumerate(seqs) if not any(t in hm for t in s)]
    out = dict(
        split_seed=a.seed, lang_folder=a.lang_folder, same_windows_as_lang_annotations=same_windows,
        n_windows=int(len(tasks)), n_held_out_windows=int(held.sum()),
        n_held_out_windows_dev={n: int(cnt_all[n]) for n in sorted(dev_names)},
        n_held_out_windows_sealed_total=int(held.sum()) - sum(int(cnt_all[n]) for n in dev_names),
        overlap_rule=f"drop a non-held-out window that overlaps any held-out window by > {OVERLAP} frames",
        n_dropped_overlap=int(drop_overlap.sum()),
        dropped_overlap_by_task=dict(Counter(t for t, d in zip(tasks, drop_overlap) if d).most_common()),
        n_kept=int(keep.sum()), kept_fraction=float(keep.mean()),
        n_samples_kept=int(sum(e - s - 9 for s, e in idx[keep])), n_samples_all=int(sum(e - s - 9 for s, e in idx)),
        held_out_frames=int(hf.sum()), held_out_frames_in_kept_windows=int((hf & kf).sum()),
        kept_by_task=dict(sorted(Counter(t for t, k in zip(tasks, keep) if k).items())),
        n_eval_chains=len(seqs), clean_chains=clean, n_clean_chains=len(clean))
    os.makedirs(C.RESULTS, exist_ok=True)
    np.savez(os.path.join(C.RESULTS, f"windows_s{a.seed}.npz"), keep=keep, indx=idx)
    p = os.path.join(C.RESULTS, f"windows_s{a.seed}.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(p + ".tmp", p)
    brief = {k: v for k, v in out.items() if k not in ("kept_by_task", "clean_chains")}
    print(json.dumps(brief, indent=1))
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("c_calvin_windows", bench=C.BENCH, tag=f"s{a.seed}", seed=a.seed, metrics=brief,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(data=a.data, lang_folder=a.lang_folder, overlap=OVERLAP),
                            script="experiments/c_calvin_split.py", command=" ".join(sys.argv),
                            artifacts=[p, os.path.join(C.RESULTS, f"windows_s{a.seed}.npz")]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--record", action="store_true")
    w = sub.add_parser("windows")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--data", default=str(C.DATA))
    w.add_argument("--lang_folder", default="lang_clip_resnet50")
    w.add_argument("--seqs", default=os.path.join(os.path.dirname(C.RESULTS), "x1_calvin", "sequences.json"))
    w.add_argument("--record", action="store_true")
    a = ap.parse_args()
    cmd_split(a) if a.cmd == "split" else cmd_windows(a)


if __name__ == "__main__":
    main()
