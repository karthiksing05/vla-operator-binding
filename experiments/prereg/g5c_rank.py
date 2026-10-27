"""Pre-registration of plan G's read G5c (ledger m3_prereg tag g5c_rank), written before its fits and episodes
(experiments/g5c_rank.py).  E3's 22 prototypes refit at rank 8 and rank 16 (otherwise E3's fit commands, read from the ledger);
acceptance on stored states 0-9 and test on E3's fresh set b at noise seed 13, paired with E3's episodes."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g5c_rank", seed=13,
    metrics={"ranks": [8, 16],
             "hypotheses": {"H1": "at rank 16, the kept library is non-inferior to the per-task LoRA with acceptance (task-level "
                                  "bootstrap 95% lower bound above -0.05), E3's criterion",
                            "descriptive": "rank 8 the same; each rank's kept arm vs E3's rank-4 kept arm and vs zero-shot; bytes"},
             "context": "E3 rank 4: 480 vs 494 of 580 against LoRA+acc (-0.024 [-0.074, 0.024]); the rank ablation was planned in "
                        "docs/PLAN_GENERALIZE.md G5",
             "written_before": "any G5c fit or episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g5c_rank.py", command="pre-registration of plan G's read G5c (rank)"))
