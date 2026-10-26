"""PRE-REGISTRATION ROW of probe R6, labelled EXPLORATORY (docs/PLAN_R6.md; code experiments/r6_smoke.py + r6_guide.py), written
2026-10-25 AFTER the task selection (plan.json), the D5 release bank (bank.json) and the 2-episode serving smoke (smoke_check.json),
and BEFORE any R6 rollout was launched or read (asserted below).  It fixes the design of an exploratory smoke; the decision rule of
PLAN_R6 section 5 decides only whether a pre-registered confirmatory read (stage C) is worth running.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r6")
assert not glob.glob(os.path.join(OUT, "roll", "*.json")), "R6 rollouts already exist"
assert not os.path.exists(os.path.join(OUT, "read.json")), "read.json already exists"
assert not ledger.load("r6_read"), "an r6_read row already exists"
sm = json.load(open(os.path.join(OUT, "smoke_check.json"), encoding="utf-8"))
assert sm["reproduces"] == {"smoke_ours": True, "smoke_off": True}, sm["reproduces"]
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r6_smoke.py"), ("guide", "experiments/r6_guide.py"),
                                               ("doc", "docs/PLAN_R6.md"), ("plan", "results/r6/plan.json"), ("bank", "results/r6/bank.json"),
                                               ("client", "experiments/libero_client.py"), ("serve", "experiments/serve_guided.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(dict(sha=sha, groups=plan["groups"], smoke=sm["reproduces"]), indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr6_smoke", seed=0,
    metrics={"status": "EXPLORATORY closed-loop smoke; no confirmatory claim, no p-values",
             "groups": plan["groups"], "arms": "zs (D4 / R5s rows, paired); both (R5s vis_e1 approach + vision-anchored carry guidance "
                                               "via D5's release operator, PRIMARY); goal (carry only); orc (oracle object + goal region)",
             "guidance": "x0-projection, first 5 of 10 Euler steps, w 1; carry: d* = min(|r*-h|, max(|D|, 3 cm)) unit(r*-h), "
                         "r* = h + J (c - h) + b, D5 release bank region > type > category > role",
             "readouts": "success, named-goal placement (own atom at the final pose), failure-type shift, harm (3 zs-success tasks), "
                         "damage (2 cells), closed-loop goal-anchor hit (5 cm of the region site), guidance bookkeeping, cost",
             "decision": "stage C iff `both` raises pooled wrong-goal-set success by >= +0.10 or its paired-difference CI excludes 0",
             "smoke": sm["reproduces"], "files_sha256": sha,
             "disclosure": "seen before this row: D5's full read (docs/PLAN_D5.md section 8), R5s's read, R2's zs types on fresh_rb2 "
                           "(task selection), the bank print; serving smoke on libero_90:29 trials 0-1; no R6 rollout read.  Two "
                           "login-node slips during D5 development are disclosed in PLAN_D5 section 7.",
             "written_before": "any R6 rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr6_smoke.py",
    command="pre-registration (EXPLORATORY): probe R6 closed-loop goal-role guidance smoke"))
