"""Pre-registration of plan D2's read G30a (ledger m3_prereg tag g30_full_sft): full-parameter SFT of pi0.5's action expert on the
task's one own-scene demonstration (experiments/g30_baselines.py full; r3_fewshot.py --full: 1,000 steps, lr 2.5e-5, batch 32), served
with E3's acceptance test.  Written after the stage-1 fits (all 29 tasks fitted) and before any episode on set b; the user approved stage 1
(2026-10-20; whole-model full SFT skipped by decision, the action-expert full SFT stands)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_full_sft", seed=13,
    metrics={"hypotheses": {"H1": "E3's library with acceptance is non-inferior to full SFT with acceptance (margin 0.05; lower bound of the task-level difference > -0.05)",
                            "descriptive": "G8's leaves arm where it exists on set b (exploratory there); full SFT vs the per-task rank-16 LoRA, both with acceptance (is the low-rank adapter the reason prototypes win?); full SFT alone vs zero-shot; storage (1.38 GB per task, the 691M-parameter action expert in bf16, against 18.8 MB for the LoRA)"},
             "protocol": "E3's 30 LIBERO-90 tasks (29 with a demonstration; task 51 has none and is served by the policy), E3's fresh start states set b (sampler seeds 200000-200019), 20 episodes per task, policy noise seed 13, paired with E3's arms (library with acceptance 480 of 580, per-task LoRA with acceptance 494 of 580, zero-shot) and G7's DSRL arms; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes, non-inferiority margin 0.05; reported overall and by grade (docs/SPLIT.md)",
             "acceptance": "as E3: 10 practice episodes per task on stored init states 0-9 at seed 13; the arm is kept where it beats the frozen policy there, else the policy serves",
             "whole_model": "not run (the user's decision 5, 2026-10-20): 3.3B parameters with AdamW need about 53 GB of optimiser state alone, above one A40 (48 GB); the action expert is the full-parameter arm",
             "written_before": "any G30a episode on set b"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_full_sft.py", command="pre-registration of plan D2's read g30_full_sft"))
