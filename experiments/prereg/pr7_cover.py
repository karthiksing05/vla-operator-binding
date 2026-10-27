"""PRE-REGISTRATION ROW of probe R7 stage A (docs/PLAN_R7.md; code experiments/r7_cover.py + r7_pack.sbatch), written 2026-10-25
AFTER the unit plan (results/r7/plan.json) and a 3-query mechanics smoke (results/r7/sb_smoke; no outcome printed or read), and BEFORE
any R7 rollout was launched or read (asserted below).
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r7")
assert not glob.glob(os.path.join(OUT, "sb", "*.npz")), "R7 rollouts already exist"
assert not os.path.exists(os.path.join(OUT, "read.json")), "read.json already exists"
assert not ledger.load("r7_read"), "an r7_read row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r7_cover.py"), ("pack", "experiments/r7_pack.sbatch"),
                                               ("doc", "docs/PLAN_R7.md"), ("plan", "results/r7/plan.json"),
                                               ("sandbox", "experiments/d5_goal.py"), ("serve", "experiments/d5_serve.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(dict(sha=sha, counts=plan["counts"]), indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr7_cover", seed=0,
    metrics={"status": "PRE-REGISTERED stage A of probe R7 (coverage of the frozen policy's own sample distribution)",
             "states": "D5's 42 recombination zs wrong-goal observations (:16 14, :29 19, :40 9; PRIMARY) + D5's 37 zs-success "
                       "observations (control); conditions carry (D5 post-grasp state, PRIMARY) and start (step 0)",
             "rollouts": "K 32 independent noise seeds (41e6 + 1e5 k + q) + 1 zs-seed run per state; D5 sandbox, 10-step Euler, replan 5; "
                         "a run ends at LIBERO success or the episode budget (400 - j) / 5 queries; release recorded, not terminal",
             "readouts": "coverage@K (K 1 4 8 16 32, exact without-replacement estimator from 32), per-seed success, zs-run success; "
                         "pooled two-level task-cluster bootstrap (B 10000, seed 0), per-task state bootstrap",
             "hypothesis": "H-cover: pooled coverage@16 on wrong-goal carry states >= 0.5 (point estimate); decides stage B",
             "counts": plan["counts"], "files_sha256": sha,
             "disclosure": "seen before this row: D5's and R6's full reads; the plan counts; a 3-query mechanics smoke on one unit "
                           "(timing only, outcomes not printed); no R7 rollout outcome read",
             "written_before": "any R7 rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr7_cover.py",
    command="pre-registration: probe R7 stage A coverage@K"))
