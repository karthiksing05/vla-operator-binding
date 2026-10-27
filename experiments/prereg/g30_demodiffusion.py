"""Pre-registration of plan D2's read G30d (ledger m3_prereg tag g30_demodiffusion): DemoDiffusion (arXiv 2506.20668; SDEdit of the
one demonstration's chunk at s* = 0.2 through the frozen pi0.5, 20 integration steps; the M3-era implementation, now on each task's CURRENT
own-scene demonstration, r3_prepare --tag _lib40g30; arm demodiff@g30).  The M3-era read (36 of 420) used the pre-2026-10-04
demonstrations and other tasks.  Re-run by the user's decision 7 (2026-10-20).  Written before any G30d episode."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_demodiffusion", seed=13,
    metrics={"hypotheses": {"H1": "E3's library with acceptance > DemoDiffusion with acceptance (sign test, alpha 0.05)",
                            "descriptive": "DemoDiffusion alone vs zero-shot; by grade"},
             "protocol": "E3's 30 LIBERO-90 tasks (29 with a demonstration; task 51 has none and is served by the policy), E3's fresh start states set b (sampler seeds 200000-200019), 20 episodes per task, policy noise seed 13, paired with E3's arms (library with acceptance 480 of 580, per-task LoRA with acceptance 494 of 580, zero-shot) and G7's DSRL arms; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes, non-inferiority margin 0.05; reported overall and by grade (docs/SPLIT.md)",
             "acceptance": "as E3: 10 practice episodes per task on stored init states 0-9 at seed 13; the arm is kept where it beats the frozen policy there, else the policy serves",
             "written_before": "any G30d episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_demodiffusion.py", command="pre-registration of plan D2's read g30_demodiffusion"))
