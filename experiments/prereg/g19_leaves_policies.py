"""Pre-registration of plan G's read G19 (ledger m3_prereg tag g19_leaves_policies), written before any G19 episode
(experiments/g19_leaves_policies.py).  G8's task leaves (keep the best of the policy, the shared-concept composition and the
task's own one-demonstration LoRA by the acceptance test; ties to the policy, then the composition) beat the per-task LoRA with
acceptance on pi0.5.  G19 applies the same rule, unchanged, on pi0, SmolVLA and X-VLA, using G3's refit libraries, per-task LoRAs
and acceptance episodes (g3_accept, E3's stored states 0-9), on G3's 29 tasks with a demonstration; test on start states set d
(20 per task, noise seed 13), paired."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g19_leaves_policies", seed=13,
    metrics={"backbones": ["pi0_libero", "smolvla", "xvla"],
             "hypotheses": {"H1": "per policy, task leaves are non-inferior to the per-task LoRA with acceptance on set d (task-level "
                                  "bootstrap 95% lower bound above -0.05)",
                            "H2": "per policy, task leaves > the policy alone on set d (exact sign test, alpha 0.05)",
                            "descriptive": "leaves kept, bytes (hierarchy against per-task adapters), library against LoRA with acceptance"},
             "written_before": "any G19 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g19_leaves_policies.py", command="pre-registration of plan G's read G19 (task leaves on pi0, SmolVLA, X-VLA)"))
