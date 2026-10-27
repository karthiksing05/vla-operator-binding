"""Pre-registration of plan G's read G14 (ledger m3_prereg tag g14_tli_practice), iteration 11, written after G13's pi0 read (TLI with
G10's pairs, chosen by practice under the grasp switch, 251 of 400 on set d against 326 for Li's pairs) and before any G14 episode
(experiments/g14_tli_practice.py).  Practice chooses the sources UNDER the operator it will use: G10's procedure (object source
over the 16 single-step pick-and-place training instructions of LIBERO-Goal and LIBERO-Spatial with our goal source fixed, then
the goal source; 5 episodes per candidate on start states set a; most successes, ties to our rule's source, then the lowest
index) with TLI, lambda 24, as the operator.  The chosen pair is tested on set d (20 per task, noise seed 13), paired with G13's
arms on the same states.  Per policy (pi0, pi0.5)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g14_tli_practice", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"],
             "hypotheses": {"H1": "per policy, TLI with the pair practice chose under TLI is non-inferior to TLI with Li's hand-chosen pairs "
                                  "and lambda on set d (task-level bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy, it beats TLI with G10's pair (G13 tli_sel) on set d (exact sign test, alpha 0.05)",
                            "descriptive": "against the grasp switch with G10's pair and the policy alone; how often practice picks Li's pair; "
                                           "practice cost"},
             "written_before": "any G14 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g14_tli_practice.py", command="pre-registration of plan G's read G14 (iteration 11: practice under TLI)"))
