"""PRE-REGISTRATION ROW of probe R7 option 1, the pre-grasp coverage ceiling (docs/PLAN_R7.md section 7; code
experiments/r7_pregrasp.py + r7pg_pack.sbatch, run loop r7_cover.run_unit), written 2026-10-25 AFTER the pre-grasp unit plan
(results/r7/plan_pg.json) and R7 stage A's read (section 6), and BEFORE any pre-grasp rollout was launched or read (asserted below).
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
assert not glob.glob(os.path.join(OUT, "sb", "pregrasp_*.npz")), "pre-grasp rollouts already exist"
assert not os.path.exists(os.path.join(OUT, "read_pg.json")), "read_pg.json already exists"
assert not [r for r in ledger.load("r7_read") if r.get("tag") == "pregrasp"], "an r7_read pregrasp row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r7_pregrasp.py"), ("loop", "experiments/r7_cover.py"),
                                               ("pack", "experiments/r7pg_pack.sbatch"), ("doc", "docs/PLAN_R7.md"),
                                               ("plan", "results/r7/plan_pg.json"), ("sandbox", "experiments/d5_goal.py"),
                                               ("serve", "experiments/d5_serve.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(dict(sha=sha, counts=plan["counts"], missing=plan["missing"], gap=plan["gap_steps"]), indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr7_pregrasp", seed=0,
    metrics={"status": "PRE-REGISTERED option 1 of probe R7 (pre-grasp coverage ceiling; coordinator-approved after stage A)",
             "states": "the same 42 recombination zs wrong-goal + 34 of the 37 control episodes (libero_90:1 has no D4 pre "
                       "observation); state = the episode's last D4 `pre` observation (approach query before the bound step)",
             "rollouts": "as stage A: K 32 seeds (41e6 + 1e5 k + q) + zs-seed run, D5 sandbox, to LIBERO success or (400 - j) / 5 queries",
             "hypothesis": "H-pregrasp: pooled coverage@16 on wrong-goal pre-grasp states >= 0.5 (point estimate)",
             "descriptive": "carry vs pre-grasp vs start coverage@K on the same states; paired cov@16 differences (two-level bootstrap); "
                            "per task; zs-run success; carry - pre-grasp gap",
             "counts": plan["counts"], "missing": plan["missing"], "files_sha256": sha,
             "disclosure": "seen before this row: R7 stage A's full read (carry and start coverage on these states), the plan counts; "
                           "no pre-grasp rollout",
             "written_before": "any pre-grasp rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr7_pregrasp.py",
    command="pre-registration: probe R7 option 1 pre-grasp coverage@K"))
