"""Pre-registration of plan G's read G17 (ledger m3_prereg tag g17_gated_stream), iteration 14, written after G16 and before any
G17 episode and before start states set f were drawn (experiments/g16_stream_ood.py --gate 8 --tag g17 --fresh fresh_ood_f).
G16 halved practice by reusing stored sources but lost success in three of four streams.  Iteration 14 gates reuse by practice,
the rule the system uses everywhere else: an arrival whose reused pairs reach fewer than 8 of 10 acceptance episodes (the better
operator) re-practises its reused concepts for this task (stage 1 for a reused object concept with our word-rule goal source, stage
2 for a reused goal concept with the object source in hand; per operator), scores the new pairs by 10 acceptance episodes, and
serves the best pair by acceptance (ties: the stored pair, then the switch).  The store keeps its first entry.  Two orders (plan,
reversed), pi0 and pi0.5.  Tested on new start states set f (BASE 900000, 20 per task, noise seed 13), paired with G15's per-task
arm, the policy alone and Li's TLI, all run on set f."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g17_gated_stream", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"], "orders": ["plan", "reversed"], "gate": 8,
             "states": "fresh_ood_f (BASE 900000, 20 per task)",
             "hypotheses": {"H1": "per policy and order, the gated stream's served arms are non-inferior to G15's per-task arms on set f "
                                  "(task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy and order, the gated stream's served arms > the policy alone on set f (exact sign test)",
                            "descriptive": "practice episodes against per-task practice (6,800) and the ungated stream (3,600); "
                                           "re-practised arrivals; against Li's TLI on set f"},
             "written_before": "any G17 episode and set f"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g17_gated_stream.py", command="pre-registration of plan G's read G17 (iteration 14: practice-gated reuse)"))
