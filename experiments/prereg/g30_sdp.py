"""Pre-registration of plan D2's read G30g (ledger m3_prereg tag g30_sdp): the ADAPTED Sparse Diffusion Policy (Wang et al., CoRL 2024,
arXiv 2407.01531; the user's decision 4, 2026-10-20), lifelong_ood/policy/sdp.py + experiments/g30_sdp.py.  SDP's top-2 routers (one
per LoRA bank, i.e. per Linear of pi0.5's action expert, per token) over the frozen per-task one-demonstration LoRAs of every OTHER
LIBERO-90 task that has one (the task's own adapter left out), trained on the task's one own-scene demonstration (router only, as SDP's
task transfer; 1,000 steps, lr 1e-3, batch 32, flow loss + 0.01 x load balancing).  SDP's published code has no LIBERO setting and
trains a diffusion policy from scratch (130-150 h per run on one A6000); the faithful version is not run.  Written before any G30g
episode."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_sdp", seed=13,
    metrics={"hypotheses": {"H1": "E3's library with acceptance is non-inferior to the adapted SDP with acceptance (margin 0.05; lower bound of the task-level difference > -0.05)",
                            "descriptive": "SDP with acceptance vs the per-task LoRA with acceptance (routed held experts vs the task's own adapter); SDP alone vs zero-shot; by grade; storage (the routers, bf16) and fit A40-hours"},
             "protocol": "E3's 30 LIBERO-90 tasks (29 with a demonstration; task 51 has none and is served by the policy), E3's fresh start states set b (sampler seeds 200000-200019), 20 episodes per task, policy noise seed 13, paired with E3's arms (library with acceptance 480 of 580, per-task LoRA with acceptance 494 of 580, zero-shot); exact sign test on discordant pairs, two-level bootstrap over tasks and episodes, non-inferiority margin 0.05",
             "acceptance": "as E3: 10 practice episodes per task on stored init states 0-9 at seed 13; the arm is kept where it beats the frozen policy there, else the policy serves",
             "faithful": "not run (decision 4): no LIBERO code, from-scratch diffusion policy",
             "written_before": "any G30g episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_sdp.py", command="pre-registration of plan D2's read g30_sdp"))
