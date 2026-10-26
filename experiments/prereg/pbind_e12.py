"""PRE-REGISTRATION ROWS of PLAN_BIND E1 and E2 (docs/PLAN_E12.md), written 2026-10-26 AFTER E1's dry run (unit counts only) and
the E2 plan + a 30-step mechanics smoke (results/bind/e2_smoke; no outcome read), and BEFORE either read (asserted below)."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "bind")
assert not os.path.exists(os.path.join(OUT, "e1.json")), "e1.json exists"
assert not os.path.exists(os.path.join(OUT, "e2.json")), "e2.json exists"
assert not glob.glob(os.path.join(OUT, "e2", "*.npz")), "E2 rollouts exist"
assert not ledger.load("bind_e1") and not ledger.load("bind_e2"), "a read row exists"


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


E1F = sha(["experiments/bind_e1.py", "experiments/bind_e1_scene.py", "results/bind/e1_scene.json", "docs/PLAN_E12.md"])
E2F = sha(["experiments/bind_e2.py", "experiments/bind_e2_pack.sbatch", "results/bind/e2_plan.json", "docs/PLAN_E12.md",
           "experiments/d5_goal.py", "experiments/d5_serve.py"])
common = dict(bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={})
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pbind_e1", seed=0, script="experiments/prereg/pbind_e12.py",
    command="pre-registration: PLAN_BIND E1 exemplar binding",
    metrics={"status": "PRE-REGISTERED PLAN_BIND E1 (CPU)", "units": "zs binding failures on the dev tasks (fm types, fresh_m1): "
             "wrong_object (bound object), wrong_goal other_region (BDDL alt), carried_elsewhere (nearest scene site within 10 cm); dry "
             "run: 135 units (63 object, 72 goal), 80 excluded, 43 EX-undefined",
             "accounts": "EX = the scene's most frequent training binding for the role (excl. correct); IS = the most similar training "
                         "instruction's binding (word Jaccard), mapped by name / type key",
             "baseline": "PRIMARY frequency-matched: candidates weighted 1 + global training count of the type key; uniform descriptive",
             "falsified_if": "two-level task bootstrap 95% CI of mean(hit_EX - P_freq) has lo <= 0",
             "files_sha256": E1F, "written_before": "the E1 read"}, **common))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pbind_e2", seed=0, script="experiments/prereg/pbind_e12.py",
    command="pre-registration: PLAN_BIND E2 measuring P",
    metrics={"status": "PRE-REGISTERED PLAN_BIND E2 (a few A40-h)", "tasks": "24 dev tasks with an In / On goal; fresh_m1 states 0-4; "
             "K 8 seeds; D5 sandbox to the first release / success / 400 steps",
             "P_hat": "share of runs binding the named object AND releasing nearest (xy) an own target among the scene candidates",
             "tests": "Spearman(P_hat, zs success over 20 fresh_m1 episodes) with task bootstrap CI; OLS calibration slope with CI",
             "falsified_if": "rho <= 0 or rho CI lo <= 0; or slope CI entirely < 0.5 or entirely > 2.0",
             "files_sha256": E2F, "written_before": "any E2 rollout or read"}, **common))
