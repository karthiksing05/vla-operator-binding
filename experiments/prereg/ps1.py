"""PRE-REGISTRATION ROW of PLAN_SOLVE S1 (docs/PLAN_S1.md), written 2026-10-26 after the frames, labels and frozen features exist and
BEFORE B1b is trained, B1a is run, or any dev unit is scored (asserted)."""
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

S1 = os.path.join(ROOT, "results", "bind", "s1")
assert not os.path.exists(os.path.join(S1, "b1b", "pred_dev.npz")) and not os.path.exists(os.path.join(S1, "b1a", "pred_dev.npz"))
assert not os.path.exists(os.path.join(ROOT, "results", "bind", "s1.json")) and not ledger.load("s1_bind")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="ps1", seed=0, command="pre-registration: PLAN_SOLVE S1",
                    metrics={"status": "PRE-REGISTERED PLAN_SOLVE S1 (binder accuracy; B1b headline)",
                             "b1b": "frozen pi05_l90c agentview image tokens + the policy's own word embeddings; one head per role, "
                                    "own-role phrase only; trained once on training-cell labels (5168, counterfactual within scene); "
                                    "fixed hyper-parameters, 40 epochs, seed 0",
                             "comparisons": "SigLIP prototype anchor (rb4 protos, argmax token); B1a GroundingDINO-base (no depth)",
                             "hit": "peak token in target mask and covered >= every competitor (same-body regions / same-ref table "
                                    "regions / same-type instances)",
                             "bar": "PASS iff B1b depth-2 held-out hit >= 0.70 AND B1b hit >= anchor hit at depths 0, 1, 2 on "
                                    "units with a prototype (point estimates)",
                             "expectation": "depth-2 held-out = :1 + 8 caddy tasks; the caddy is in no training scene; FAIL on (i) likely",
                             "files_sha256": sha(["docs/PLAN_S1.md", "experiments/s1_render.py", "experiments/s1_labels.py",
                                                  "experiments/s1_feats.py", "experiments/s1_train.py", "experiments/s1_b1a.py",
                                                  "experiments/s1_read.py", "results/bind/s1/labels.json"]),
                             "written_before": "B1b training and any dev score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/ps1.py"))
