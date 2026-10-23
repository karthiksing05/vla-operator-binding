"""LIBERO-90-C constants and the guarded split loader (docs/PLAN_V4.md 3.1, docs/L90C.md).

The MCD split is frozen by experiments/c_split.py into two files:
  results/l90c/split.json         training tasks, dev test tasks per CD bin and the part set, only the COUNT of sealed tasks
  results/l90c/split_sealed.json  the sealed third (never committed)

`load_split()` returns the dev split only.  `load_sealed(prereg=...)` opens the sealed file only when it is given the run_id of
an existing ledger row of experiment m3_prereg on bench libero_90_c AND the environment variable L90C_UNSEAL equals that same
run_id, so a stray import cannot read it.  `task_episodes()` (the demo loader for a task) refuses a sealed task the same way.
The pattern is lifelong_ood/b1.py's (CompoSuite).  Task ids are factors.Task.tid strings, e.g. "libero_90:12".
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "l90c"
SPLIT = RESULTS / "split.json"
SEALED = RESULTS / "split_sealed.json"
BENCH = "libero_90_c"


class SealedError(PermissionError):
    pass


def load_split():
    """-> the dev split dict (train, dev={bin: [tid]}, n_sealed, ...).  Never the sealed tasks."""
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def _check_unseal(prereg):
    from . import ledger
    if not prereg or os.environ.get("L90C_UNSEAL") != prereg:
        raise SealedError("LIBERO-90-C sealed tasks are read once, at the final read: pass prereg=<m3_prereg run_id> "
                          "and set L90C_UNSEAL to the same run_id")
    rows = [r for r in ledger.load("m3_prereg") if r.get("run_id") == prereg and r.get("bench") == BENCH]
    if not rows:
        raise SealedError(f"no m3_prereg row with run_id {prereg!r} on bench {BENCH}")


def load_sealed(prereg=None):
    _check_unseal(prereg)
    return json.loads(SEALED.read_text(encoding="utf-8"))["sealed"]


def task_hash(tid, salt):
    return hashlib.sha256((salt + str(tid)).encode()).hexdigest()


def is_sealed(tid):
    s = load_split()
    return task_hash(str(tid), s["hash_salt"]) in set(s["sealed_hashes"])


def check_task(tid, prereg=None):
    """Raise SealedError for a sealed task unless unsealed (see module doc).  Every reader of a LIBERO-90-C test task calls it."""
    if is_sealed(tid):
        _check_unseal(prereg)
    return tid


def task_episodes(tid, prereg=None):
    """The IPEC LeRobot episode indices of a LIBERO-90 task (its own scene's block), refusing sealed tasks."""
    check_task(tid, prereg)
    suite, idx = str(tid).split(":")
    assert suite == "libero_90", tid
    from .policy import data as D
    return D.Demos90().episodes_of_task(int(idx))
