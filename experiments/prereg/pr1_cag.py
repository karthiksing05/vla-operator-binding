"""PRE-REGISTRATION of probe R1 (docs/PLAN_R.md section 7; design docs/PLAN_R1.md; code experiments/r1_cag.py), written
2026-10-24 after the frozen design table (results/r1/plan.json) and the serving smoke (libero_90:14, STORED init states 0-1,
2 episodes per arm, excluded from every number), BEFORE any R1 rollout on fresh_r1, fresh_m1 or the damage cells, and before any
R1 read (asserted below).

Question.  Does training-free CAG-VA (Fang et al. 2602.17659 eq. 4: v = v(o, empty) + w (v(o, instr) - v(o, empty)) at every
Euler step, the same frozen VLA) repair pi05_l90c@5000's held-out recombinations, and what does the policy do with NO instruction
(the vision shortcut)?

Sets: dev (w selection) = NEW fresh_r1 (BASE 6700000, 20 per task), noise seed 13, arms zs, cag 1.5 / 2 / 3 / 5; test = M1's
fresh_m1 (BASE 5700000), noise seed 13, cag at every w + empty, paired with M1's zs rollouts; cells = M1's 10 damage cells, stored
init states 0-19, seed 0, cag at every w + empty, paired with the parity read of pi05_l90c@5000.

  w*  = argmax over {1 (= zs), 1.5, 2, 3, 5} of dev-set successes on the 15 recombination tasks; ties to the smaller w.  The
        headline is w*; the other w on the test set are descriptive.
  H1  CAG(w*) > zs on the 15 recombination tasks of the test set: more successes AND the exact two-sided sign test on discordant
      pairs.  If w* = 1, H1's p is 1 (CAG is zs).
  H2  The empty instruction completes a demonstrated task (every In / On atom of a same-scene training task's goal holds at the
      settled final state, not all at the settled first record, and not a subset of the dev task's own goal) in MORE than 25% of
      its episodes on the 19 dev tasks with >= 1 evaluable same-scene training placement goal.  One-sided two-level cluster
      bootstrap (tasks, then states; B 10000, seed 0): p = the share of bootstrap rates <= 0.25.
  Holm over H1-H2 at alpha 0.05.
  H3  (estimate gate, outside Holm) CAG(w*)'s damage, the mean success change over the 10 training cells vs the parity base,
      >= -0.01 (0 by construction if w* = 1).
  Secondary (descriptive): every w vs zs on the test set by recombination / part / all and by bin (cluster-bootstrap CI, sign
  test); failure types (experiments/fm_types.py) of zs, empty and every cag arm; the 'did a demonstrated task instead' rates of
  zs's failures, the empty instruction's episodes and CAG's failures, split by sharing the instructed object / target; the empty
  instruction's success on the 10 training cells vs the parity base (the vision-shortcut rate) and on the dev tasks; the
  agreement of our zs types with results/fm/types.

Disclosure.  M1's and FM's zs success and failure-type numbers on fresh_m1 were read before this design (docs/PLAN_M1.md 6,
docs/PLAN_FM.md 4).  No CAG or empty-instruction rollout of pi05_l90c existed before R1.  Articulated atoms are not recorded by the
rollouts, so the instead test reads placement goals only (articulation-only training tasks are not evaluable)."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r1")
assert not glob.glob(os.path.join(OUT, "roll", "*")), "an R1 rollout already exists"
assert not glob.glob(os.path.join(OUT, "instead", "*")), "an R1 instead file already exists"
assert not ledger.load("r1cag_read"), "an R1 read already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("plan.json", "results/r1/plan.json"), ("script", "experiments/r1_cag.py"),
                                               ("doc", "docs/PLAN_R1.md"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan.json"], encoding="utf-8"))
assert len(plan["tasks"]) == 29 and len(plan["cells"]) == 10
fam = [t for t, c in plan["candidates"].items() if t in plan["tasks"] and any(x["placement"] for x in c["cands"])]
assert len(fam) == 19, fam
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr1_cag", seed=13,
    metrics={"omegas": [1.0] + plan["omegas"], "tasks": {k: v["bin"] for k, v in plan["tasks"].items()}, "cells": plan["cells"],
             "h2_tasks": fam,
             "w_star": "argmax dev-set (fresh_r1) successes on the 15 recombination tasks over w in {1, 1.5, 2, 3, 5}; ties to smaller w",
             "H1": "CAG(w*) > zs on the 15 recombination tasks of fresh_m1: more successes and exact two-sided sign test; p = 1 if w* = 1",
             "H2": "empty instruction completes a demonstrated same-scene training placement goal in > 0.25 of its episodes on the 19 "
                   "evaluable dev tasks; one-sided two-level cluster bootstrap p (B 10000, seed 0)",
             "family": "Holm over H1-H2 at alpha 0.05",
             "H3": "CAG(w*) mean success change over the 10 training cells vs parity base >= -0.01; estimate gate",
             "secondary": "every w vs zs (rec / part / all, by bin); fm_types failure types; instead rates (zs failures, empty, CAG "
                          "failures; shares object / target); empty success on cells vs parity (vision shortcut) and on dev tasks; "
                          "zs type agreement with results/fm/types",
             "states": "dev fresh_r1 BASE 6700000 seed 13; test fresh_m1 BASE 5700000 seed 13 (zs = M1's rollouts); cells stored 0-19 seed 0",
             "disclosure": "M1/FM zs numbers on fresh_m1 known; no CAG / empty rollout before R1; serving smoke on libero_90:14 stored "
                           "states 0-1 excluded; articulated atoms not recorded (placement goals only)",
             "files_sha256": sha, "written_before": "any R1 rollout on fresh_r1 / fresh_m1 / cells and any R1 read"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr1_cag.py",
    command="pre-registration: R1 CAG-VA + empty-instruction probe, design frozen, before any rollout"))
