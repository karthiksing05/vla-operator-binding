"""Pre-registration of plan G's read G5a (ledger m3_prereg tag g5_factors), written before any of its episodes
(experiments/g5_factors.py).  On G0's 53 eligible held-out tasks, with G0's fold concepts: the object concept alone and the goal
concept alone (the policy doing the other phase); acceptance on stored states 0-9 (seed 15), test on G0's fresh start states
(seed 15).  Factor-level selection = the acceptance test's best of {policy, both concepts, object alone, goal alone}, ties to the
policy, then both, then object, then goal."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g5_factors", seed=15,
    metrics={"hypotheses": {"H1": "factor-level selection > selection over {policy, both concepts} (G0's H1b arm), exact sign test, "
                                  "alpha 0.05",
                            "reported": "factor-level selection against zero-shot; each factor alone against zero-shot; which arm kept"},
             "context": "G0 and G2 were read before this registration (recombination supplies a missing skill and interferes with a "
                        "present one: exploratory)",
             "written_before": "any episode of g5"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g5_factors.py", command="pre-registration of plan G's read G5a"))
