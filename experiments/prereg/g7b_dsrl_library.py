"""Pre-registration of plan G's read G7b (ledger m3_prereg tag g7b_dsrl_library), written after a three-episode training smoke and
before any other episode (experiments/g7b_dsrl_library.py).  DSRL (G7's configuration) steering E3's composition on E3's 30
tasks: 100 practice episodes on stored states 0-29, acceptance on states 0-9, test on E3's fresh set b, noise seed 13."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g7b_dsrl_library", seed=13,
    metrics={"hypotheses": {"H1": "DSRL on the library with acceptance is non-inferior to DSRL on the per-task LoRA with acceptance "
                                  "(G7's dsrls; task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "DSRL on the library with acceptance > E3's kept library (exact sign test, alpha 0.05)",
                            "descriptive": "DSRL on the library alone vs zero-shot"},
             "context": "G7: dsrls+acc 503 vs library 480 of 580 (-0.040 [-0.098, 0.012]); G5c: capacity does not close the gap",
             "written_before": "any G7b episode other than two smokes on task 14 at seed 99 (three and four training episodes; the second beside the plain composition)"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g7b_dsrl_library.py", command="pre-registration of plan G's read G7b"))
