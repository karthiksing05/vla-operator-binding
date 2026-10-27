"""PRE-REGISTRATION of CALVIN-C Stage A (headroom), written 2026-10-23 BEFORE any single-subtask rollout of a held-out cell
by either model (experiments/cc_headroom.py; no results/calvin_c/headroom/ file existed when this row was written).

Known before writing: chain parity on clean chains (FLOWER-C 4.527, control 4.528), the all-chain gap (4.269 vs 4.438, all of
it on chains containing held-out cells, not broken down by cell), and the flow-loss audit (+0.030 held-out gap).  No
per-held-out-cell success of either model had been computed (c_calvin_parity scores training subtasks only).

Models: FLOWER-C seed 0 (results/calvin_c/flower_c_s0/export_stepfinal, held-out windows removed) and the control
flower_abc_full (results/calvin_c/flower_abc_full/export_stepfinal, every window, same recipe), both step-35,000 EMA.
Cells: the 5 DEV held-out cells of split_s0.json; sealed cells are not read.  Plus 3 training sibling cells as a harness
check (not part of the decision).
Episodes: every initial condition of the authors' evaluation generator with the cell's block on the table (96 per cell),
scene D (the only env config the harness has), CALVIN's validation instruction for the cell, ep_len 360, 4 sampling steps,
multistep 10; per-episode seeds sha256(cell|k), identical across models.  The episode list is frozen by its sha256 below.
Measure: per cell, gap = control success - FLOWER-C success over the paired episodes.  Pooled gap = the unweighted mean over
the 5 dev held-out cells.  95% CI: two-stage cluster bootstrap (resample cells with replacement, then paired episodes within
each drawn cell), 10,000 draws, numpy default_rng(0), percentile interval.
Decision: "headroom" iff the CI lower bound > 0.10; otherwise "no headroom" and Stage B (P2/P3 ports) is not run.
Descriptive only (no per-bin claim from < 6 cells): per-cell success, per-grade means, the sibling-cell gap, and what
FLOWER-C did on its failures (tasks the oracle detected at the end)."""
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from lifelong_ood import ledger
import cc_headroom as H

assert not (H.OUT / "flower_c_s0").exists() and not (H.OUT / "flower_abc_full").exists(), "rollouts already exist"
from collections import Counter
cnt = Counter((e["cell"], e["role"], e["grade"]) for e in H.episodes())
print(cnt)
print(ledger.record(
    "m3_prereg", bench="calvin_c", tag="cc_headroom", seed=0,
    metrics={"measure": "per dev held-out cell: control success - FLOWER-C success over paired episodes; pooled = unweighted "
                        "mean over the 5 dev held-out cells",
             "ci": "two-stage cluster bootstrap (cells, then paired episodes), 10000 draws, default_rng(0), 95% percentile",
             "decision": "headroom iff CI lower bound > 0.10; else no headroom and Stage B is not run",
             "models": {k: str(v) for k, v in H.MODELS.items()},
             "cells": [list(c) for c in H.cells()], "episodes_per_cell": {f"{c}|{r}|{g}": n for (c, r, g), n in cnt.items()},
             "episodes_sha256": H.episodes_sha(), "scene": "D (task_ABC_D/validation env config; no ABC env config exists)",
             "rollout": "ep_len 360, num_sampling_steps 4, multistep 10, val instruction [0], per-episode seed sha256(cell|k)",
             "descriptive": "per-cell success, per-grade means, sibling (training) cell gap, FLOWER-C failure modes",
             "known_before": "clean-chain parity 4.527/4.528; all-chain 4.269/4.438; audit +0.030; no per-held-out-cell success",
             "written_before": "any held-out single-subtask rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/cc_headroom.py",
    command="pre-registration: CALVIN-C Stage A headroom (FLOWER-C vs control on dev held-out cells)"))
