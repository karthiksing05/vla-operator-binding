"""Pre-registration of plan Z's read Z5 (ledger m3_prereg tag z5_blend), written after the offline blend read (ledger zi_blend
tag read: held-out flow loss only) and before any Z5 episode.  experiments/z5_blend.py."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="z5_blend", seed=13,
    metrics={"hypotheses": {"H1": "the served soft blend (nothing fitted in the network) > pi0.5 alone on Z3's 25 tasks, set d, 20 per task "
                                  "(exact sign test, alpha 0.05)",
                            "H2": "the blend > Z3's hard pointers on the same episodes (exact sign test)",
                            "descriptive": "against G8's fitted library; behavioural donation of the top-3 offline experts per task "
                                           "(ablation on states 0-9) against their offline Shapley donation (Spearman)"},
             "written_before": "any Z5 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/z5_blend.py", command="pre-registration of plan Z's read Z5 (the soft blend served)"))
