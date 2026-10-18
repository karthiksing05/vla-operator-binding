"""Pre-registration of plan G's read G18 (ledger m3_prereg tag g18_four_policies), written before any G18 episode
(experiments/g18_four_policies.py).  The OOD headline (iteration 12, G15) holds on pi0 and pi0.5; SmolVLA and X-VLA stopped at
iteration 5.  G18 runs the finished OOD system, unchanged, on SmolVLA and X-VLA: practice chooses each held concept's sources
under the grasp switch (G10's two stages) and under text-latent interpolation (G14's, lambda 24; TLI ported to each policy's own
conditioning encoder through one interface, lifelong_ood/policy/tli.py), then the operator per task by 10 practice episodes each
on set a (ties to the switch).  If the TLI port fails its smoke check on a policy, that policy's system runs with the switch alone
and its TLI arms are dropped (stated in the read).  Test on set e (20 per task, noise seed 13): the chosen arm, the policy alone,
Li's hand-chosen pairs switched at the grasp, and Li's TLI (Li's pairs and lambda) where ported."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g18_four_policies", seed=13,
    metrics={"backbones": ["smolvla", "xvla"],
             "hypotheses": {"H1": "per policy, the system's chosen arms > the policy alone on set e (exact sign test, alpha 0.05)",
                            "H2": "per policy, the system's chosen arms are non-inferior to Li's hand-chosen pairs (TLI where ported, "
                                  "else switched at the grasp) on set e (task-level bootstrap 95% lower bound above -0.05)",
                            "descriptive": "operator counts; the four-policy table with G15's pi0 and pi0.5"},
             "written_before": "any G18 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g18_four_policies.py", command="pre-registration of plan G's read G18 (the OOD system on SmolVLA and X-VLA)"))
