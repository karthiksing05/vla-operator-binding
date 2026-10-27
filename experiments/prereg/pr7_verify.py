"""PRE-REGISTRATION ROW of probe R7 stage B gate (docs/PLAN_R7.md section 8; code experiments/r7_verify.py), written 2026-10-26 AFTER
the pre-grasp coverage read (section 7.1) and BEFORE r7_verify.py was run (asserted below).  Offline, CPU only.
"""
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r7")
assert not os.path.exists(os.path.join(OUT, "verify.json")), "verify.json already exists"
assert not [r for r in ledger.load("r7_read") if r.get("tag") == "verify"], "an r7_read verify row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r7_verify.py"), ("doc", "docs/PLAN_R7.md"),
                                               ("plan", "results/r7/plan_pg.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr7_verify", seed=0,
    metrics={"status": "PRE-REGISTERED stage B gate of probe R7 (offline; one-step selection of the first chunk at the pre-grasp query)",
             "verifiers": "V_ceil (PRIMARY, information ceiling, not deployable): L2 logistic C 1.0 balanced on within-state-centred "
                          "standardised first-chunk features, leave-one-state-out over 42 wrong-goal pre-grasp states (leave-task-out "
                          "descriptive); V_mode / V_anti descriptive",
             "metric": "top-1 gain = picked run success - c/32 per state, two-level task-cluster bootstrap; pick success on covering states",
             "gate": "V_ceil leave-state-out pooled gain >= +0.05 and CI lo > 0",
             "files_sha256": sha,
             "disclosure": "seen before this row: R7 stage A and option 1 reads (coverage, per-state success counts); no chunk feature "
                           "or score computed",
             "written_before": "r7_verify.py run"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr7_verify.py",
    command="pre-registration: probe R7 stage B gate"))
