"""Pre-registration of plan G's read G5b (ledger m3_prereg tag g5b_gating), written before any of its episodes
(experiments/g5b_gating.py).  E3's library on the stream's 30 tasks with both prototypes on for the whole episode, against E3's
gated runs (object until the policy's grasp, goal after it), at E3's fresh start states (set b) and noise seed 13; no acceptance."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g5b_gating", seed=13,
    metrics={"hypotheses": {"H1": "gated composition > ungated (both prototypes on throughout), exact sign test, alpha 0.05",
                            "reported": "ungated against zero-shot"},
             "context": "E3, G0-G3 and G5a were read before this registration", "written_before": "any episode of g5b"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g5b_gating.py", command="pre-registration of plan G's read G5b"))
