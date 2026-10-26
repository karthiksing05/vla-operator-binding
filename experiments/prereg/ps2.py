"""PRE-REGISTRATION ROW of PLAN_SOLVE S2 smoke (docs/PLAN_S2.md), written 2026-10-26 after S1c's read (FAIL -> binder S1b) and
BEFORE any S2 rollout exists (asserted)."""
import glob
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

RD = os.path.join(ROOT, "results", "bind", "s2", "roll")
assert not glob.glob(os.path.join(RD, "*", "*.npz")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "s2.json"))
assert not ledger.load("s2_smoke")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps2", seed=0, command="pre-registration: PLAN_SOLVE S2 smoke",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S2 smoke (EXPLORATORY; S0 adapter + S1b deployable binder)",
                             "arms": "zs / oracle = S0's dev runs at seeds 0-3 (reused, paired); deploy = adapter + S1b masks per replan, new",
                             "binder": "s1b (S1c FAILED: i 0.40, ii 0.65)",
                             "metrics": "success, binding error, motion (S0 classify); closure (deploy-zs)/(oracle-zs); damage vs S0 train_zs seeds 0-3",
                             "direction": "D1 binding(deploy) < binding(zs); D2 binding closure in [0,1]; D3 damage >= -0.05; no pass bar",
                             "files_sha256": sha(["docs/PLAN_S2.md", "experiments/s2_serve.py", "experiments/s2_sandbox.py",
                                                  "experiments/s2_read.py", "experiments/s2_pack.sbatch", "results/bind/s2/plan.json"]),
                             "written_before": "any S2 rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps2.py"))
