"""Pre-registration of plan G's read G15 (ledger m3_prereg tag g15_operator), iteration 12, written after G14 and before any G15
episode and before start states set e were drawn (experiments/g15_operator.py).  G13 and G14 found that the better operator for
held concepts depends on the policy (pi0: TLI; pi0.5: the grasp switch).  Iteration 12 lets practice choose the operator per task:
the grasp switch with G10's practice-chosen pair against TLI with G14's practice-chosen pair (lambda 24), each scored by 10
practice episodes on start states set a (the switch's are G10's acceptance episodes g10_acc; TLI's are run here on the same
states); the higher count wins, ties to the grasp switch.  Tested on set e (BASE 800000, 20 per task, noise seed 13), paired with
the policy alone, Li's TLI and both single operators run on set e.  Per policy (pi0, pi0.5)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g15_operator", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"], "states": "fresh_ood_e (BASE 800000, 20 per task)",
             "hypotheses": {"H1": "per policy, the practice-chosen operator is non-inferior to Li's TLI with hand-chosen pairs on set e "
                                  "(task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy, the practice-chosen operator > the policy alone on set e (exact sign test, alpha 0.05)",
                            "descriptive": "against each single operator; operator counts; per suite"},
             "written_before": "any G15 episode and set e"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g15_operator.py", command="pre-registration of plan G's read G15 (iteration 12: practice chooses the operator)"))
