"""CALVIN-C constants and the guarded split loader (docs/PLAN_V4.md 3.2, docs/DESIGN.md 3.2, docs/CALVIN_C.md).

CALVIN's 21 block subtasks are the cells of three small factorial families:
  lift   colour x place      lift_{red,blue,pink}_block_{table,slider,drawer}    (9)
  push   colour x direction  push_{red,blue,pink}_block_{left,right}              (6)
  rotate colour x direction  rotate_{red,blue,pink}_block_{left,right}            (6)
A cell is (verb, colour, slot), the slot being the place (lift) or the direction (push, rotate).
CAVEAT: in CALVIN a block's colour also fixes its size (red middle, blue small, pink big in scene D), so "colour" is the
atom colour+size; a colour generalisation is also a size generalisation.

experiments/c_calvin_split.py freezes three split seeds, each into two files:
  results/calvin_c/split_s{k}.json          training cells, dev held-out cells with grades and the part set, and only the
                                            COUNT and salted hashes of the sealed cells
  results/calvin_c/split_s{k}_sealed.json   the sealed third (never committed)
`load_split(k)` returns the dev split only.  `load_sealed(k, prereg=...)` opens the sealed file only when it is given the
run_id of an existing ledger row of experiment m3_prereg on bench calvin_c AND the environment variable CALVINC_UNSEAL equals
that same run_id, so a stray import cannot read it.  `check_cell()` refuses a sealed cell the same way; every reader of a
CALVIN-C held-out cell calls it.  `held_out_mask()` tells a training-data filter whether a subtask is held out (dev or
sealed) WITHOUT naming which held-out cells are sealed.  The pattern is lifelong_ood/b1.py's; the hashes stop accidents, not a
determined reader (21 names are easy to enumerate).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "calvin_c"
BENCH = "calvin_c"
SEEDS = (0, 1, 2)

COLOURS = ("red", "blue", "pink")
PLACES = ("table", "slider", "drawer")
DIRECTIONS = ("left", "right")
VERBS = ("lift", "push", "rotate")
SLOTS = dict(lift=PLACES, push=DIRECTIONS, rotate=DIRECTIONS)
CELLS = tuple((v, c, s) for v in VERBS for c in COLOURS for s in SLOTS[v])
DATA = pathlib.Path("/coc/flash10/ksingara3/robot/data/calvin_abc/task_ABC_D")


class SealedError(PermissionError):
    pass


def cell_name(cell):
    """('rotate', 'pink', 'left') -> 'rotate_pink_block_left' (CALVIN's subtask name)."""
    v, c, s = cell
    return f"{v}_{c}_block_{s}"


def parse_cell(name):
    """CALVIN subtask name -> cell tuple, or None for the 13 non-block-cell subtasks."""
    p = str(name).split("_")
    if len(p) == 4 and p[2] == "block" and p[0] in VERBS and p[1] in COLOURS and p[3] in SLOTS[p[0]]:
        return (p[0], p[1], p[3])
    return None


def compounds(cell):
    """The cell's three pairwise compounds: verb-colour, verb-slot, colour-slot."""
    v, c, s = cell
    return (("vc", v, c), ("vs", v, s), ("cs", c, s))


def atoms(cell):
    v, c, s = cell
    return (("verb", v), ("colour", c), ("place" if v == "lift" else "direction", s))


def split_path(seed):
    return RESULTS / f"split_s{int(seed)}.json"


def sealed_path(seed):
    return RESULTS / f"split_s{int(seed)}_sealed.json"


def load_split(seed=0):
    """-> the dev split of split seed `seed` (train, dev held-out cells with grades, part set, n_sealed, ...)."""
    s = json.loads(split_path(seed).read_text(encoding="utf-8"))
    s["train"] = [tuple(c) for c in s["train"]]
    s["dev"] = [dict(d, cell=tuple(d["cell"])) for d in s["dev"]]
    return s


def _check_unseal(prereg):
    from . import ledger
    if not prereg or os.environ.get("CALVINC_UNSEAL") != prereg:
        raise SealedError("CALVIN-C sealed cells are read once, at the final read: pass prereg=<m3_prereg run_id> "
                          "and set CALVINC_UNSEAL to the same run_id")
    rows = [r for r in ledger.load("m3_prereg") if r.get("run_id") == prereg and r.get("bench") == BENCH]
    if not rows:
        raise SealedError(f"no m3_prereg row with run_id {prereg!r} on bench {BENCH}")


def load_sealed(seed=0, prereg=None):
    _check_unseal(prereg)
    s = json.loads(sealed_path(seed).read_text(encoding="utf-8"))
    return [dict(d, cell=tuple(d["cell"])) for d in s["sealed"]]


def cell_hash(cell, salt):
    return hashlib.sha256((salt + cell_name(tuple(cell))).encode()).hexdigest()


def is_sealed(cell, seed=0):
    s = json.loads(split_path(seed).read_text(encoding="utf-8"))
    return cell_hash(tuple(cell), s["hash_salt"]) in set(s["sealed_hashes"])


def check_cell(cell, seed=0, prereg=None):
    """Raise SealedError for a sealed cell unless unsealed (see module doc)."""
    if is_sealed(cell, seed):
        _check_unseal(prereg)
    return tuple(cell)


def held_out_mask(task_names, seed=0):
    """[subtask name] -> [bool]: True when the subtask is a held-out cell of split `seed` (dev or sealed).  Reads only
    split_s{seed}.json (dev cells by name, sealed cells by salted hash), never the sealed file."""
    s = json.loads(split_path(seed).read_text(encoding="utf-8"))
    dev = {cell_name(tuple(d["cell"])) for d in s["dev"]}
    hashes, salt = set(s["sealed_hashes"]), s["hash_salt"]
    out = []
    for t in task_names:
        c = parse_cell(t)
        out.append(c is not None and (cell_name(c) in dev or cell_hash(c, salt) in hashes))
    return out
