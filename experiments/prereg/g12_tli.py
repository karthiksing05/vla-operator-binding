"""Pre-registration of plan G's read G12 (ledger m3_prereg tag g12_tli), written before any G12 episode and before the text latents
were identified (experiments/g12_tli.py).  Li et al. (arXiv 2505.03500) report Text Latent Interpolation lifting pi0 from 0.09 to
0.83 on their 20 compositional OOD tasks.  G12 runs TLI on our harness: latents identified on openpi's PyTorch pi0 by Li's
procedure (lifelong_ood/policy/tli.py), Li's base-task pairs and lambda, the test instruction as the prompt; start states set c,
20 per task, noise seed 13, paired with G4d's policy alone and pi0^S and with G10's practice-chosen pair (all on set c)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g12_tli", seed=13,
    metrics={"backbones": ["pi0_libero"],
             "hypotheses": {"H1": "TLI > the policy alone (exact sign test, alpha 0.05): the port reproduces Li's direction",
                            "H2": "G10's practice-chosen pair (grasp switch, no hand labels) is non-inferior to TLI with Li's hand-chosen "
                                  "pairs (task-level bootstrap 95% lower bound above -0.05)",
                            "descriptive": "TLI vs pi0^S (fixed step) and vs Li's pair at the grasp; by suite; TLI's absolute rate "
                                           "against Li's published 0.83"},
             "written_before": "any G12 episode and the latent identification"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g12_tli.py", command="pre-registration of plan G's read G12 (Li's text latent interpolation)"))
