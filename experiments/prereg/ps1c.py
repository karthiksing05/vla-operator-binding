"""PRE-REGISTRATION ROW of PLAN_SOLVE S1c (docs/PLAN_S1.md section 8), written 2026-10-26 after a
2-epoch trainer smoke (no dev prediction) and BEFORE S1c is trained or any dev unit is scored (asserted)."""
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OD = os.path.join(ROOT, "results", "bind", "s1c")
assert not os.path.exists(os.path.join(OD, "pred_dev.npz")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "s1c.json"))
assert not ledger.load("s1c_bind")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps1c", seed=0, command="pre-registration: PLAN_SOLVE S1c",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S1c (detector body x sim-trained instance / sub-region head)",
                             "design": "GroundingDINO body boxes at test; head trained on SIM segmentation boxes (jittered, + distractors) of training cells, full role phrase",
                             "bar": "PASS iff held-out depth-1 hit >= 0.70 AND all depth-2 seen-body hit >= 0.70 AND depth-0 hit >= GroundingDINO (0.786); caddy out of scope",
                             "power": "(i) 3 tasks, 15 units; (ii) 4 tasks, 20 units",
                             "files_sha256": sha(["docs/PLAN_S1.md", "experiments/s1b_detect.py", "experiments/s1c_train.py",
                                                  "experiments/s1c_read.py", "results/bind/s1b/det.json"]),
                             "written_before": "S1c training and any dev score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps1c.py"))
