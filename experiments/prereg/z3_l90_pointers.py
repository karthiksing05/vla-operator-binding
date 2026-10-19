"""Pre-registration of plan Z's read Z3 (ledger m3_prereg tag z3_l90_pointers), written before any Z3 episode and before reading
any of the U1 segmentation outcome on these tasks' rollouts.  The zero-fine-tune row on LIBERO-90: on G8's 25 tasks whose object and
goal are both held as words in pi0.5's 40 training instructions, held pointers switched at the grasp, sources chosen by the demo's
flow loss (the demo is a query only), nothing fitted.  Set d, 20 per task, noise seed 13 (experiments/z3_l90_pointers.py)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="z3_l90_pointers", seed=13,
    metrics={"hypotheses": {"H1": "pointers (zero fitting) > pi0.5 alone on set d (exact sign test, alpha 0.05)",
                            "descriptive": "pointers vs the fitted library (G8 hfac) and vs the per-task LoRA (G8): the share of the "
                                           "fitted methods' successes reached with no fitting; per task"},
             "written_before": "any Z3 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/z3_l90_pointers.py", command="pre-registration of plan Z's read Z3 (LIBERO-90 zero-fine-tune row)"))
