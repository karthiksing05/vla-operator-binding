"""Pre-registration of plan G's read G4b (ledger m3_prereg tag g4b_field), written after the exploratory screen on start states set
a and before any episode on set c (experiments/g4_ood.py --test-b, --read-b).  Li's 20 compositional OOD tasks.  A concept the
policy already holds is the policy's own field under the instruction of a training task that taught it (the concept's source
task in the test scene's base suite, else the first); a step runs under the object concept's instruction until the policy's own
grasp and under the goal concept's from the grasp to the release (arm pfield).  Acceptance on set a (10 episodes, recorded)
keeps the best of {policy, prototype composition, policy-field composition}, ties to the policy then the prototypes; test on a
NEW set c (LIBERO's sampler, seeds 500000+, 20 per task), noise seed 13, Li's modified LIBERO."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g4b_field", seed=13,
    metrics={"hypotheses": {"H1": "the kept arm > the policy over the 20 tasks on set c (exact sign test, alpha 0.05)",
                            "descriptive": "the policy-field composition alone vs the policy; the prototype composition alone vs "
                                           "the policy; per suite"},
             "screen": "set a, 10 episodes per task: policy 66, prototypes 67, policy field 75 of 200 (gains on 3 tasks, losses "
                       "on 3); the arm and its source rule were fixed before the screen",
             "written_before": "any episode on start states set c"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g4b_field.py", command="pre-registration of plan G's read G4b (policy-field composition)"))
