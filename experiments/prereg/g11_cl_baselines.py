"""Pre-registration of plan G's read G11 (ledger m3_prereg tag g11_cl_baselines), written before its fits and episodes
(experiments/g11_cl_baselines.py): sequential LoRA fine-tuning and experience replay through W1's order a (one demonstration per
task, rank 16, 500 steps per task), the final adapter serving every task, on E3's fresh set b at seed 13, paired with W1's served
arms."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g11_cl_baselines", seed=13,
    metrics={"hypotheses": {"H1": "W1's served library (order a) > sequential fine-tuning's final adapter (exact sign test, alpha 0.05)",
                            "H2": "W1's served library (order a) > experience replay's final adapter (same test); Holm over H1, H2",
                            "descriptive": "each baseline vs zero-shot; successes on the first and last ten arrivals (forgetting)"},
             "context": "novelty audit item 4; Liu et al. (arXiv 2603.03818) find pretrained VLAs forget little under replay",
             "written_before": "any G11 fit or episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g11_cl_baselines.py", command="pre-registration of plan G's read G11 (CL baselines)"))
