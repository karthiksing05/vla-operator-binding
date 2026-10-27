"""Pre-registration of plan G's read G8 (ledger m3_prereg tag g8_leaves), written after an exploratory pass over E3's and E5's
recorded episodes and before any episode on start-state set d (experiments/g8_leaves.py).  The hierarchy with task leaves: the
acceptance test (recorded, stored states 0-9) keeps the best of the policy, the composition of shared concepts and the task's
own one-demonstration adapter (ties: policy, then composition); compared with the per-task LoRA given its own acceptance test,
on 66 tasks, on a NEW set d (sampler seeds 600000+, 20 per task), noise seed 13."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g8_leaves", seed=13,
    metrics={"hypotheses": {"H1": "the hierarchy with leaves is non-inferior to the per-task LoRA with acceptance over the 66 tasks "
                                  "(task-level bootstrap 95% lower bound of the difference above -0.05)",
                            "H1b": "superiority: the same lower bound above 0 (tested only if H1 passes)",
                            "descriptive": "leaves kept; the library without leaves vs LoRA+acc; leaves vs zero-shot and vs the "
                                           "library"},
             "exploratory_before": "on E3's and E5's recorded test episodes (sets b) the rule gave 1,013 against 935 of 1,320 "
                                   "(+0.059 [0.019, 0.104]) with leaves on 24 of 66 tasks; the rule was chosen after that pass",
             "written_before": "any episode on start-state set d"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g8_leaves.py", command="pre-registration of plan G's read G8 (task leaves)"))
