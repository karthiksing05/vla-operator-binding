"""PRE-REGISTRATION ROW of probe R5s, labelled EXPLORATORY (docs/PLAN_R5S.md; code experiments/r5s_smoke.py + r5s_guide.py), written
2026-10-25 AFTER the task selection (plan.json), the bank rebuild, the fresh_r5s states and their check, and the 2-episode serving
smoke (smoke_check.json), and BEFORE any R5s rollout was launched or read (asserted below).  It fixes the design of an exploratory
smoke: no confirmatory claim, no p-value; the decision rule of PLAN_R5S section 5 decides only whether a pre-registered read is
worth running.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r5s")
assert not glob.glob(os.path.join(OUT, "roll", "*.json")), "R5s rollouts already exist"
assert not os.path.exists(os.path.join(OUT, "read.json")), "read.json already exists"
assert not ledger.load("r5s_read"), "an r5s_read row already exists"
sm = json.load(open(os.path.join(OUT, "smoke_check.json"), encoding="utf-8"))
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r5s_smoke.py"), ("guide", "experiments/r5s_guide.py"),
                                               ("doc", "docs/PLAN_R5S.md"), ("plan", "results/r5s/plan.json"),
                                               ("bank", "results/r5s/bank.json"), ("client", "experiments/libero_client.py"),
                                               ("fresh_check", "results/r5s/fresh_check.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(dict(sha=sha, groups=plan["groups"], smoke=sm["reproduces"]), indent=1))
DISCLOSURE = ("EXPLORATORY smoke approved by the user 2026-10-25.  Seen before this row: D4's published results incl. its zs dev "
              "types (the wrong-object tasks), R2's per-task zs on fresh_rb2 (the harm-set rule), R2's damage order (cells rule), "
              "R4-bind / R4b stage-0 tables (the two configs), the 2-episode serving smoke on libero_90:73 (parity of unguided "
              "serving with D4's zs actions; plus a vis / orc code-path check whose outcomes are excluded).  One empty python3 "
              "heredoc ran on the login node during development (rule breach, no computation).")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr5s_smoke", seed=0,
    metrics={"status": "EXPLORATORY (closed-loop smoke; decides whether a pre-registered read is worth running)",
             "sets": plan["groups"],
             "arms": "zs (D4 rows, paired by trial on fresh_d4; re-rolled on fresh_r5s for the 2 cells); vis_e1 PRIMARY (SigLIP "
                     "anchor + live depth, x0-projection first 5 Euler steps w 1); vis_a025 (all steps w 0.25); orc_e1 (sim pose)",
             "guidance": "only open-gripper queries before the first grasp (_Grasp rule); d* = J (c - h) + b, D4 bank (type > "
                         "category > role)",
             "readouts": "success, named grasp, fm_types failure types, paired arm - zs differences with two-level task-cluster "
                         "bootstrap 95% intervals (B 10000 seed 0), harm (4 zs-success recombination tasks), damage (2 training "
                         "cells), anchor localisation (5 cm), guidance bookkeeping, A40-h",
             "decision_rule": "confirmatory read worth running iff wo-set vis_e1 success diff >= +0.10 AND named-grasp diff > 0 "
                              "AND harm diff >= -0.05 AND damage diff >= -0.05 (points)",
             "smoke": sm["reproduces"], "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any R5s rollout beyond the 2-episode serving smoke"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr5s_smoke.py",
    command="pre-registration row (EXPLORATORY): probe R5s closed-loop smoke of vision-anchored D4-bank guidance"))
