"""DRAFT pre-registration of plan D2's read G30f (ledger m3_prereg tag g30_library), PENDING THE USER'S DECISION (docs/BASELINES.md):
LOTUS (arXiv 2311.02058; runnable code, LIBERO; IsCiL ships no LIBERO setup) as the published skill-library method: base skills from
pi0.5's 40 training tasks (50 demonstrations each, LOTUS's base stage), then W1's order a as the lifelong stream with 1 demonstration per
task (matched) and 10 (LOTUS's published setting).  Its own ResNet-T policy trained from scratch, as published.  NOT YET RECORDED."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_library", seed=13,
    metrics={"hypotheses": {"H1": "W1's served library (order a) > LOTUS at 1 demonstration per task (exact sign test, alpha 0.05)",
                            "descriptive": "LOTUS at 10 demonstrations per task; LOTUS's own FWT / NBT / AUC on the stream; successes on the first and last ten arrivals"},
             "protocol": "G11's: E3's fresh set b at seed 13, 20 episodes per task, paired with W1's served arms; LOTUS's evaluation uses the task id as published",
             "parity": "before the read: LOTUS on LIBERO-OBJECT at its published setting reproduces its FWT 74 / AUC 65 within its seeds' spread",
             "written_before": "any G30f training or episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_library.py", command="pre-registration of plan D2's read g30_library"))
