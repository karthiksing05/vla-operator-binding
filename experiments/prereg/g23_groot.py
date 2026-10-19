"""Pre-registration of plan G's read G23 (ledger m3_prereg tag g23_groot), written before any G23 episode.  G18's design, unchanged,
on a fifth flow-matching policy: GR00T N1.5 (ar0s/groot_libero, fine-tuned on the same 40 LIBERO tasks and frames as pi0.5;
lifelong_ood/policy/backbone_groot.py).  The finished OOD system (practice chooses sources under the grasp switch and under TLI,
then the operator per task), Li's 20 tasks, test on set e (20 per task, noise seed 13), with the policy alone, Li's pairs at the
grasp and Li's TLI (experiments/g18_four_policies.py --backbone groot_libero)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g23_groot", seed=13,
    metrics={"backbones": ["groot_libero"],
             "hypotheses": {"H1": "the system's chosen arms > GR00T alone on set e (exact sign test, alpha 0.05)",
                            "H2": "the system's chosen arms are non-inferior to Li's hand-chosen pairs (TLI, or switched at the grasp if "
                                  "TLI fails its smoke) on set e (task-level bootstrap 95% lower bound above -0.05)",
                            "descriptive": "operator counts; the five-policy table"},
             "written_before": "any G23 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g23_groot.py", command="pre-registration of plan G's read G23 (the OOD system on GR00T N1.5)"))
