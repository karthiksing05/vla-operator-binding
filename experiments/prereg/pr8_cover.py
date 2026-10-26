"""PRE-REGISTRATION ROW of probe R8 stage A (docs/PLAN_R8.md section 6; code experiments/r8_cover.py, r8_serve.py, r8_ops.py,
r8_pack.sbatch), written 2026-10-26 AFTER the lexicon (results/r8/lex.json), the unit plan (results/r8/plan.json), the prototypes
(results/r8/protos.npz) and two 2-query mechanics smokes (results/r8/sb_smoke; outcomes not printed), and BEFORE any R8 rollout.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r8")
assert not glob.glob(os.path.join(OUT, "sb", "*", "*.npz")), "R8 rollouts already exist"
assert not os.path.exists(os.path.join(OUT, "read.json")), "read.json already exists"
assert not ledger.load("r8_read"), "an r8_read row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r8_cover.py"), ("serve", "experiments/r8_serve.py"),
                                               ("ops", "experiments/r8_ops.py"), ("pack", "experiments/r8_pack.sbatch"),
                                               ("doc", "docs/PLAN_R8.md"), ("plan", "results/r8/plan.json"),
                                               ("lex", "results/r8/lex.json"), ("protos", "results/r8/protos.npz"),
                                               ("loop", "experiments/r7_cover.py"), ("d5serve", "experiments/d5_serve.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(dict(sha=sha, counts=plan["counts"]), indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr8_cover", seed=0,
    metrics={"status": "PRE-REGISTERED stage A of probe R8 (test-time region re-conditioning, frozen weights, sandboxed coverage)",
             "units": "28 wrong-goal region-error pre-grasp states (:29 19, :40 9; PRIMARY) + 4 controls (:29 1, :40 3); 44 R7 units "
                      "with no spec are not re-rolled (operator inactive)",
             "arms": "c3 (PRIMARY): phrase-token residual replaced at every encoder layer >= 1 by the mean training activation of the "
                     "same phrase (D5 training carry observations of :3/:4/:5 resp. :88); c2w1 / c2w2: v + w (v - mean v_alt) on the "
                     "first 5 of 10 Euler steps, alts = other regions of the same body with a training phrase; C1 no-op (phrases "
                     "already equal training phrases), not run; C4 skipped",
             "rollouts": "R7 run loop, R7 seeds k 0..15 + zs-seed run, from R7 pre-grasp states, to success or budget",
             "bar": "PASS if wrong-goal per-seed success diff vs R7 same seeds >= +0.10 with two-level CI lo > 0 and control per-seed "
                    "diff >= -0.05; on-manifold guard: out-of-bound executed actions share rise > 0.10 flagged",
             "counts": plan["counts"], "specs": plan["specs"], "files_sha256": sha,
             "disclosure": "seen before this row: R7's full reads (incl. per-state success counts at these states), the lexicon, the "
                           "plan, prototype shapes, server parity (c2 w0 exact; c3 own-residual 2-3e-3, bf16 batch noise), two "
                           "2-query smokes (timing only)",
             "written_before": "any R8 rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr8_cover.py",
    command="pre-registration: probe R8 stage A region re-conditioning coverage"))
