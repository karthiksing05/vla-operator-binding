"""B1 (CompoSuite) constants and the guarded split loader (docs/B1_COMPOSUITE.md, docs/DESIGN.md section 2).

The factorial split is frozen by experiments/b1_split.py into two files:
  results/b1/split.json         train cells, dev test cells, and only the COUNT of sealed cells
  results/b1/split_sealed.json  the sealed test cells (one third of each grade)

Nothing may read the sealed file before the final read.  `load_split()` returns the dev split only.
`load_sealed(prereg=...)` opens the sealed file only when it is given the run_id of an existing ledger row
of experiment m3_prereg (bench composuite) AND the environment variable B1_UNSEAL equals that same run_id,
so a stray import cannot read it.  The data loader (`task_dir`) refuses sealed cells the same way.

Factor labels are used only to build the split and to score (DESIGN C5): nothing here hands them to a policy.
"""
from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "b1"
SPLIT = RESULTS / "split.json"
SEALED = RESULTS / "split_sealed.json"
DATA = pathlib.Path("/coc/flash10/ksingara3/robot/data/composuite")
VENV_PY = pathlib.Path("/coc/flash10/ksingara3/robot/composuite/.venv/bin/python")

# value names exactly as composuite.make and the dataset folder names spell them
ROBOTS = ("IIWA", "Jaco", "Kinova3", "Panda")
OBJECTS = ("Box", "Dumbbell", "Plate", "Hollowbox")
OBSTACLES = ("None", "GoalWall", "ObjectDoor", "ObjectWall")
OBJECTIVES = ("PickPlace", "Push", "Shelf", "Trashcan")
AXES = ("robot", "object", "obstacle", "objective")
VALUES = dict(robot=ROBOTS, object=OBJECTS, obstacle=OBSTACLES, objective=OBJECTIVES)


class SealedError(PermissionError):
    pass


def cell_name(cell):
    return "_".join(cell)


def multi_hot(cell):
    """16-dim factor indicator (ORACLE policy only; never given to the free-embedding policy)."""
    v = []
    for axis, val in zip(AXES, cell):
        v += [1.0 if val == x else 0.0 for x in VALUES[axis]]
    return v


def load_split():
    """-> dict(train=[cell...], dev={grade: [cell...]}, n_sealed={grade: n}, ...).  Never the sealed cells."""
    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    s["train"] = [tuple(c) for c in s["train"]]
    s["dev"] = {g: [tuple(c) for c in cs] for g, cs in s["dev"].items()}
    return s


def _check_unseal(prereg):
    from . import ledger
    if not prereg or os.environ.get("B1_UNSEAL") != prereg:
        raise SealedError("B1 sealed cells are read once, at the final read: pass prereg=<m3_prereg run_id> "
                          "and set B1_UNSEAL to the same run_id")
    rows = [r for r in ledger.load("m3_prereg") if r.get("run_id") == prereg and r.get("bench") == "composuite"]
    if not rows:
        raise SealedError(f"no m3_prereg row with run_id {prereg!r} on bench composuite")


def load_sealed(prereg=None):
    _check_unseal(prereg)
    s = json.loads(SEALED.read_text(encoding="utf-8"))
    return {g: [tuple(c) for c in cs] for g, cs in s["sealed"].items()}


def sealed_names_hashed():
    """Salted hashes of sealed cell names (stored in split.json) so loaders can refuse a sealed cell without
    reading the sealed file."""
    return set(json.loads(SPLIT.read_text(encoding="utf-8"))["sealed_hashes"])


def cell_hash(cell, salt):
    import hashlib
    return hashlib.sha256((salt + cell_name(cell)).encode()).hexdigest()


def is_sealed(cell):
    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    return cell_hash(tuple(cell), s["hash_salt"]) in set(s["sealed_hashes"])


def task_dir(cell, kind="expert", prereg=None):
    """Path of a cell's converted data.  Refuses sealed cells unless unsealed (see module doc)."""
    if is_sealed(cell):
        _check_unseal(prereg)
    return DATA / kind / cell_name(cell)
