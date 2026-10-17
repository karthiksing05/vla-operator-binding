"""Pre-registration of plan G's read G4c (ledger m3_prereg tag g4c_field_xvla), written after a one-episode serving smoke per policy and
before any other episode (experiments/g4_ood.py --accept-c, --test-c, --read-c --backbone <b>).  G4b's policy-field composition
(each concept the policy's own field under its source training instruction, switched at the grasp; G4b's sources) on two more
flow-matching policies trained on the same LIBERO suites: pi0 (openpi pi0_libero) and SmolVLA (lerobot/smolvla_libero).  Arms
zero-shot and pfield; acceptance on set a (10 episodes) keeps pfield only if it beats the policy; test on set c (20 per task), noise
seed 13, Li's modified LIBERO."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g4c_field_xvla", seed=13,
    metrics={"backbones": ["xvla"],
             "hypotheses": {"H1": "per policy, the kept arm > the policy over the 20 tasks on set c (exact sign test; Holm over the "
                                  "same arms, sources and states)",
                            "descriptive": "pfield alone vs the policy; per suite; pi0's published zero-shot and text-latent "
                                           "interpolation rates (arXiv 2505.03500) as context only"},
             "context": "G4b passed on pi0.5 (174 vs 147 of 400), G4c on pi0 (98 vs 35) and SmolVLA (25 vs 8); X-VLA backbone added (lerobot/xvla-libero)",
             "written_before": "any G4c episode other than the smoke (task goal:7, one episode per arm on X-VLA, seed 99)"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g4c_field_xvla.py", command="pre-registration of plan G's read G4c on X-VLA"))
