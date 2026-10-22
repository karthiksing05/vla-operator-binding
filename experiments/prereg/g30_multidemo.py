"""Pre-registration of plan D2's read G30b (ledger m3_prereg tag g30_multidemo): the per-task rank-16 LoRA recipe on 5 own-scene
demonstrations (2,000 steps) and on all of them (28 to 50; 3,000 steps), the ceiling: how many demonstrations equal one demonstration plus
the prototypes.  Written after the stage-1 fits of the 5-demonstration LoRA and before any episode on set b; the all-demonstration LoRA is fitted on
all 29 tasks (the user's decision 6, 2026-10-20), and its parity smoke (tasks 13, 36, 52, stored states 30-49) runs alongside, never on set b."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_multidemo", seed=13,
    metrics={"hypotheses": {"H1": "the 5-demonstration LoRA > the 1-demonstration LoRA, both with acceptance (sign test, alpha 0.05): the recipe uses more data",
                            "H2": "E3's library with acceptance is non-inferior to the 5-demonstration LoRA with acceptance (margin 0.05)",
                            "descriptive": "the all-demonstration LoRA (ceiling) against everything; the value of the prototypes in demonstrations: the k in {1, 5, all} at which the LoRA's successes reach the library's, interpolated in log k, per grade"},
             "protocol": "E3's 30 LIBERO-90 tasks (29 with a demonstration; task 51 has none and is served by the policy), E3's fresh start states set b (sampler seeds 200000-200019), 20 episodes per task, policy noise seed 13, paired with E3's arms (library with acceptance 480 of 580, per-task LoRA with acceptance 494 of 580, zero-shot) and G7's DSRL arms; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes, non-inferiority margin 0.05; reported overall and by grade (docs/SPLIT.md)",
             "acceptance": "as E3: 10 practice episodes per task on stored init states 0-9 at seed 13; the arm is kept where it beats the frozen policy there, else the policy serves",
             "parity": "stage 1: the all-demonstration LoRA on tasks 13, 36, 52 (stored states 30-49) against the published LIBERO-90 multi-task level of pi0.5 (92.7, arXiv 2505.23705 Table 1)",
             "written_before": "any G30b episode on set b"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_multidemo.py", command="pre-registration of plan D2's read g30_multidemo"))
