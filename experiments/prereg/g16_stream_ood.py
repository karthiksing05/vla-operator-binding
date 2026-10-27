"""Pre-registration of plan G's read G16 (ledger m3_prereg tag g16_stream_ood), iteration 13, written after G15 and before any G16
episode (experiments/g16_stream_ood.py).  G15 chooses each OOD task's held-concept sources by 160 practice episodes per task even
when a concept recurs.  Iteration 13 makes that practice continual: Li's 20 tasks arrive as a stream (two orders: the plan's, and
reversed); a held concept's chosen source is STORED per operator, keyed by (suite, concept words), and a later arrival with that
concept reuses it.  An arrival practises only the unseen concepts: an unseen object concept by stage 1 (16 candidates x 5 episodes
on set a, goal source fixed to our word rule), an unseen goal concept by stage 2 (16 x 5, object source fixed to the stored or
just-chosen one); this per operator (grasp switch, TLI lambda 24), then the operator by 10 practice episodes each on set a (ties
to the switch), as G15.  Tested on start states set e (20 per task, noise seed 13), paired with G15's per-task arm."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g16_stream_ood", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"], "orders": ["plan", "reversed"],
             "hypotheses": {"H1": "per policy and order, the stream's served arms are non-inferior to G15's per-task practice on set e "
                                  "(task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy and order, the stream's served arms > the policy alone on set e (exact sign test, alpha 0.05)",
                            "descriptive": "practice episodes against G15's (stage episodes plus operator choice); concepts reused; "
                                           "against Li's TLI"},
             "written_before": "any G16 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g16_stream_ood.py", command="pre-registration of plan G's read G16 (iteration 13: continual practice on the OOD stream)"))
