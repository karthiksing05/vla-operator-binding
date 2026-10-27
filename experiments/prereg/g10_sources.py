"""Pre-registration of plan G's read G10 (ledger m3_prereg tag g10_sources), written before any G10 episode
(experiments/g4_ood.py --g10-s1, --g10-s2, --g10-test, --g10-read).  G4d found that Li's hand-chosen source pairs carry the
pointer composition (pi0: 305 of 400 at the grasp against 74 for our word-matching rule).  G10 lets PRACTICE choose the sources
with no hand labels and no privileged state: the candidates are the 16 single-step pick-and-place training instructions of
LIBERO-Goal and LIBERO-Spatial; stage 1 tries each as the object source (our goal source fixed), stage 2 each as the goal source
(stage 1's object source fixed), 5 practice episodes per candidate on start states set a (seed 13), the most successes winning
(ties to our rule, then the lowest index); the chosen pair, switched at the grasp, is tested on set c (20 per task) and given
the acceptance test (10 episodes on set a against the policy's recorded episodes).  On pi0 and pi0.5, Li's 20 tasks."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g10_sources", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"],
             "hypotheses": {"H1": "per policy, the practice-chosen pair (unfiltered, grasp switch) is non-inferior to Li's hand-chosen "
                                  "pair at the grasp (G4d li_grasp; task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy, the practice-chosen pair > our word-matching pair (exact sign test, alpha 0.05)",
                            "descriptive": "the chosen pair vs Li's fixed-step pi0^S and vs zero-shot; with acceptance vs zero-shot; how "
                                           "often practice picks Li's pair; practice cost (episodes)"},
             "written_before": "any G10 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g10_sources.py", command="pre-registration of plan G's read G10 (practice-chosen sources)"))
