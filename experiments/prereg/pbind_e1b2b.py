"""PRE-REGISTRATION ROWS of PLAN_BIND E2b and E1b (docs/PLAN_E12.md sections 5-6), written 2026-10-26 AFTER E1b's dry run (counts)
and E2b's 160-step mechanics smoke (results/bind/e2b_smoke; no outcome read), BEFORE either read (asserted)."""
import glob
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "bind")
assert not os.path.exists(os.path.join(OUT, "e2b.json")) and not os.path.exists(os.path.join(OUT, "e1b.json"))
assert not glob.glob(os.path.join(OUT, "e2b", "*.npz")), "E2b rollouts exist"
assert not [r for r in ledger.load("bind_e1") + ledger.load("bind_e2") if r.get("tag") in ("e1b", "e2b")]


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


common = dict(bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
              script="experiments/prereg/pbind_e1b2b.py")
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pbind_e2b", seed=0, command="pre-registration: PLAN_BIND E2b",
                    metrics={"status": "PRE-REGISTERED PLAN_BIND E2b", "design": "E2 tasks/states/seeds re-rolled; first release "
                             "branched, 20 settle steps, LIBERO In/On predicates vs every candidate; run continues to success / 400 "
                             "steps; P_hat = P_obj x P_goal; competence = success of correct runs; S_hat = P_hat x competence",
                             "falsified_if": "Spearman(P_hat, zs) <= 0 or its task-bootstrap CI lo <= 0",
                             "secondary": "S_hat calibration slope CI entirely < 0.5 or > 2.0 -> FAIL",
                             "files_sha256": sha(["experiments/bind_e2b.py", "experiments/bind_e2b_pack.sbatch", "experiments/bind_e2.py",
                                                  "results/bind/e2_plan.json", "docs/PLAN_E12.md"]),
                             "written_before": "any E2b rollout"}, **common))
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pbind_e1b", seed=0, command="pre-registration: PLAN_BIND E1b",
                    metrics={"status": "PRE-REGISTERED PLAN_BIND E1b", "units": "E1 object-role units with >= 2 candidates: 41 "
                             "(:58 20, :83 18, :1 2, :29 1); low power pre-declared",
                             "accounts": "VIS SigLIP type-prototype similarity to the named type; FREQ global training object-role "
                                         "exposure; SRC nearest training grasp source of the same scene / family (2609.39971)",
                             "baseline": "frequency-matched P_freq (as E1); uniform descriptive",
                             "above_baseline_if": "two-level task bootstrap CI of mean(hit - P_freq) lo > 0",
                             "files_sha256": sha(["experiments/bind_e1b.py", "results/bind/e1b_sources.json", "results/bind/e1.json",
                                                  "docs/PLAN_E12.md"]),
                             "written_before": "the E1b read"}, **common))
