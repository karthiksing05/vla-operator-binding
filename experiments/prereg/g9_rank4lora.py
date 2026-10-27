"""Pre-registration of plan G's read G9 (ledger m3_prereg tag g9_rank4lora), written before its fits and episodes
(experiments/g9_rank4lora.py): a rank-4 per-task one-demo LoRA with the acceptance test on E3's 29 tasks, against E3's library."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g9_rank4lora", seed=13,
    metrics={"hypotheses": {"H1": "E3's kept library is non-inferior to the rank-4 per-task LoRA with acceptance (task-level bootstrap "
                                  "95% lower bound above -0.05)",
                            "descriptive": "rank-4 vs rank-16 per-task LoRA with acceptance; bytes of 22 shared rank-4 prototypes "
                                           "against 29 rank-4 per-task LoRAs"},
             "context": "novelty audit: the 1/5 storage ratio compares rank 4 against rank 16; G5c: library rank does not matter",
             "written_before": "any G9 fit or episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g9_rank4lora.py", command="pre-registration of plan G's read G9"))
