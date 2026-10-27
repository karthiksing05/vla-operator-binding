"""Pre-registration of plan G's read G4d (ledger m3_prereg tag g4d_li), written after a two-episode serving smoke and before any
other G4d episode (experiments/g4_ood.py --test-d, --read-d).  The novelty audit (docs/NOVELTY.md) found Li et al.'s pi0^S
(arXiv 2505.03500): on the same 20 tasks, pi0 under a hand-chosen pair of training instructions switched at a fixed step reaches
0.69 in their setup, against our grasp-switched composition's 98 of 400 on pi0.  G4d runs pi0^S on our harness and crosses its two
choices with ours: the sources (Li's hand-chosen pairs, their examples/libero/mapping.json, against our rule) and the trigger
(Li's fixed step N/2*5 environment steps, query N/2-2 in our client, against the policy's own grasp).  Arms li_fixed (= pi0^S),
li_grasp, ours_fixed, on start states set c (20 per task), noise seed 13, on pi0 and pi0.5; ours_grasp and zero-shot are G4c's
and G4b's recorded episodes on the same states and seed.  Unfiltered (no acceptance), as Li report."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g4d_li", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"],
             "questions": {"Q1": "does pi0^S reproduce Li's reported gain on our harness (li_fixed vs zero-shot)?",
                           "Q2": "which choice carries the gap between pi0^S and ours: sources (li_grasp vs ours_grasp, "
                                 "li_fixed vs ours_fixed) or trigger (li_grasp vs li_fixed, ours_grasp vs ours_fixed)?",
                           "test": "exact sign tests on discordant pairs, per policy; descriptive, no Holm family"},
             "written_before": "any G4d episode other than the smoke (pi0, goal-OOD task 7, two episodes, seed 99)"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g4d_li.py", command="pre-registration of plan G's read G4d (Li's pi0^S)"))
