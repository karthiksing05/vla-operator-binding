"""Pre-registration of plan D2's read G30c (ledger m3_prereg tag g30_cl): LIBERO's lifelong baselines EWC (e_lambda 50000, gamma 0.9)
and PackNet (equal share per task; and LIBERO's prune_perc 0.75) on one rank-16 LoRA through W1's order a, one demonstration per task, 500
steps per task (+ 500 post-prune steps for PackNet), experiments/g30_cl.py, in G11's protocol.  EWC's final adapter serves every task; PackNet
serves task t with the weights of tasks up to t.  Task 51 has no demonstration, so the stream fits 29 tasks; as in G11 it is still served: by EWC's final adapter, and
for PackNet by the weights of the tasks that arrived before it (task 52's served adapter, its predecessor in order a).
Written after the stage-1 fits, before any episode."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_cl", seed=13,
    metrics={"hypotheses": {"H1": "W1's served library (order a) > EWC's final adapter (exact sign test, alpha 0.05)",
                            "H2": "W1's served library (order a) > PackNet (equal share) with each task's own masks (same test); Holm over H1, H2",
                            "descriptive": "PackNet with prune_perc 0.75; each against zero-shot and against G11's sequential fine-tuning (52) and experience replay (335); successes on the first and last ten arrivals (forgetting); storage"},
             "protocol": "G11's: E3's fresh set b at seed 13, 20 episodes per task on all 30 tasks of order a, paired with W1's served arms (order a); no acceptance (as G11: the CL baselines serve every task); exact sign test on discordant pairs, two-level bootstrap over tasks and episodes", "task_51": "no demonstration: EWC's final adapter serves it; PackNet serves it with the weights of tasks up to its predecessor 52",
             "written_before": "any G30c episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_cl.py", command="pre-registration of plan D2's read g30_cl"))
