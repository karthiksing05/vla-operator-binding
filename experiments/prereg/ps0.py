"""PRE-REGISTRATION ROW of PLAN_SOLVE S0 (docs/PLAN_S0.md), written 2026-10-26 after the training data and a 3-step training smoke,
BEFORE the adapter is trained or any closed-loop run (asserted)."""
import glob
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OD = os.path.join(ROOT, "results", "bind", "s0")
assert not os.path.exists(os.path.join(OD, "adapter.pt")) and not os.path.exists(os.path.join(OD, "ckpt.pt"))
assert not glob.glob(os.path.join(OD, "roll", "*", "*.npz")) and not ledger.load("s0_adapter")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps0", seed=0, command="pre-registration: PLAN_SOLVE S0",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S0 (binding adapter + oracle masks; dissociation)",
                             "adapter": "scene-wide additive edit of agentview image K/V at prefix layers 10-17 from role masks; zero masks "
                                        "-> zero edit; trained once on 56 training cells' demos by the frozen policy's flow loss, own + "
                                        "same-scene counterfactual prompts, 3000 steps",
                             "arms": "dev (E2b units, 960 runs): zs, oracle, wrong, zero (24-unit parity); damage: 10 training cells x 5 "
                                     "states x 8 seeds, zs vs oracle",
                             "B1": "oracle binding-error rate <= 0.5 x zs AND paired diff CI hi < 0",
                             "B2": "relative motion-failure reduction <= half the relative binding-error reduction",
                             "B3": "training-cell success oracle >= zs - 0.05",
                             "files_sha256": sha(["docs/PLAN_S0.md", "experiments/s0_adapter.py", "experiments/s0_train.py",
                                                  "experiments/s0_data.py", "experiments/s0_serve.py", "experiments/s0_sandbox.py",
                                                  "experiments/s0_read.py", "experiments/s0_pack.sbatch"]),
                             "written_before": "adapter training and any closed-loop run"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps0.py"))
