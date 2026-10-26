"""PRE-REGISTRATION ROW of PLAN_SOLVE S1b (docs/PLAN_S1.md section 6), written 2026-10-26 after GroundingDINO's proposals and a
2-epoch trainer smoke (no dev prediction) and BEFORE S1b is trained or any dev unit is scored (asserted)."""
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OD = os.path.join(ROOT, "results", "bind", "s1b")
assert not os.path.exists(os.path.join(OD, "pred_dev.npz")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "s1b.json"))
assert not ledger.load("s1b_bind")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps1b", seed=0, command="pre-registration: PLAN_SOLVE S1b",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S1b (factorised binder)",
                             "design": "GroundingDINO body boxes x a trained relation / instance head on the box's 8x8 crop (2x) of the "
                                       "frozen policy's image tokens + box geometry / rank + the relation's own word embeddings",
                             "bar": "PASS iff S1b hit on depth-2 held-out units with a SEEN body type >= 0.70 AND S1b depth-0 hit >= "
                                    "GroundingDINO alone (S1 B1a), point estimates",
                             "power": "(i) is one task (:1, 5 units)",
                             "files_sha256": sha(["docs/PLAN_S1.md", "experiments/s1b_detect.py", "experiments/s1b_train.py",
                                                  "experiments/s1b_read.py", "results/bind/s1b/det.json"]),
                             "written_before": "S1b training and any dev score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps1b.py"))
