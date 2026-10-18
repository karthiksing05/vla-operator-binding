"""Pre-registration of plan G's read G13 (ledger m3_prereg tag g13_tli_sources), iteration 10, written after G12 (pi0: Li's TLI 325
of 400 on set c against 292 for the practice-chosen pair switched at the grasp) and before any G13 episode, before start states
set d were drawn and before the pi0.5 text latents were identified (experiments/g13_tli_sources.py).

Iteration 10 keeps TLI as the operator for held concepts and drops its hand labels: the base-task pair is G10's practice-chosen
pair (object source, goal source; no new practice) and lambda is fixed at 24 for every task (Li's most common value; Li hand-set
12, 14 and 30 on four tasks).  Arms per policy (pi0, pi0.5): the policy alone, TLI with Li's pairs and lambda, TLI with the
practice-chosen pair (lambda 24), and the practice-chosen pair switched at the grasp (G10's arm).  Start states set d (Li's 20
tasks, 20 per task, BASE 700000, never used), noise seed 13, paired.  Also descriptive on set c: TLI with Li's pairs on pi0.5."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g13_tli_sources", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"], "states": "fresh_ood_d (BASE 700000, 20 per task)",
             "hypotheses": {"H1": "per policy, TLI with the practice-chosen pair (lambda 24) is non-inferior to TLI with Li's hand-chosen "
                                  "pairs and lambda on set d (task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy, TLI with the practice-chosen pair > the same pair switched at the grasp on set d "
                                  "(exact sign test, alpha 0.05)",
                            "descriptive": "each arm against the policy alone on set d; TLI with Li's pairs on pi0.5 on set c against "
                                           "G10's 269; by suite"},
             "written_before": "any G13 episode, set d and the pi0.5 latents"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g13_tli_sources.py", command="pre-registration of plan G's read G13 (iteration 10: TLI with practice-chosen sources)"))
