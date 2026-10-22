"""Pre-registration of plan D2's read G30e (ledger m3_prereg tag g30_rl): RL on the policy's weights, by the user's decisions 1 and 2
(2026-10-20).

(1) Parity, no training: RLinf's released checkpoints in RLinf's own LIBERO harness (slurm/jobs/g30_rlinf_parity.sbatch):
    RLinf-Pi05-LIBERO-SFT on LIBERO-Spatial (published few-shot SFT 84.6), RLinf-Pi0-PPO-LIBERO-spatial on LIBERO-Spatial (published
    Flow-Noise PPO 99.0 for pi0), and, descriptive, RLinf-Pi05-PPO-LIBERO-130 on LIBERO-90 (trained on every LIBERO-90 task, so not
    held-out).
(2) The arm: pi_RL's Flow-Noise PPO (arXiv 2510.25889) with the trainable weights restricted to a rank-16 LoRA on pi0.5's action
    expert (VLM frozen) plus the noise head and a critic on the VLM's prefix output, on one A40, inside our harness
    (lifelong_ood/policy/pirl.py; serve mode pirl:, launch_pack arm pirl_train).  Initialised from each task's one-demonstration
    per-task LoRA (the scene-correct adapter of E3's per-task arm), sparse success reward, practice on stored init states 0-29 only.
    ReinFlow (arXiv 2505.22094) is the same learned-noise flow-policy RL; this arm counts for it (its repository has no LIBERO / VLA
    code and points pi0 LIBERO to RLinf).
    THE PUBLISHED RECIPE IS FULL-MODEL PPO AT A FAR LARGER SCALE: 8 x H100 per suite, 400 epochs x 8 rollout epochs x 64 parallel
    environments, after a few-shot SFT on 40+ trajectories; ours is LoRA PPO on one A40 at the budget below.
Budget (matched): the library's practice per task on these 20 tasks, from the ledger rows e3_accept: its 10 acceptance episodes
(mean 260 s) plus the frozen policy's 10 (317 s) = 577 s of rollout wall-clock, i.e. 0.16 A40-hours with the A40 to itself (the
library's episodes ran two instances per A40, so this is twice its GPU-time).  The pi_RL run's LoRA is saved when its wall-clock
since its first episode (rollouts plus PPO updates) passes 577 s (arm lora@pirlm).  Descriptive: the same run continued to 100
practice episodes, DSRL's budget in G7 (arm lora@pirl).  Neither counts the per-task LoRA fit both start from (about 5 min), which
the library does not need."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g30_rl", seed=13,
    metrics={"hypotheses": {"H1": "E3's library with acceptance is non-inferior to pi_RL (LoRA, matched budget, lora@pirlm) with acceptance on the G3/G4 subset (margin 0.05: lower bound of the task-level difference > -0.05)",
                            "descriptive": "pi_RL at 100 practice episodes (lora@pirl) with acceptance against the library, the per-task LoRA with acceptance and DSRL on the per-task LoRA (G7 dsrls, 100 practice episodes) with acceptance; each pi_RL arm against its own initialisation (the per-task LoRA) and against zero-shot; practice successes (first 10 / last 20 episodes); environment steps, chunks, PPO updates and A40-hours of each arm; storage (the served LoRA, 18.8 MB, as the per-task LoRA)"},
             "protocol": "the G3/G4 tasks of E3 (9, 10, 12, 13, 14, 15, 30; 25, 31, 36, 41, 43, 55, 56, 57, 58, 59, 61, 62, 71), E3's fresh set b (sampler seeds 200000-200019), 20 episodes per task, noise seed 13, paired with E3's arms (library with acceptance, per-task LoRA with acceptance, zero-shot) and G7's dsrls; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes (experiments/e1_read.py compare); RL practice on stored init states 0-29 only (STATE_MOD 30), never set b",
             "acceptance": "as E3: 10 episodes on stored init states 0-9 at seed 13; the arm is kept where it beats the frozen policy there, else the policy serves",
             "recipe": {"noise": "Flow-Noise: ExploreNoiseNet on the expert's final hidden state, sigma in RLinf's [0.08, 0.16] scaled by sqrt(3/10) for 10 Euler steps (RLinf's pi0.5 LIBERO PPO uses 3)", "logprob": "joint over the chain (mean over K+1 steps), executed 5 actions x 7 dims summed", "critic": "ValueHead 2048-1024-512-256-1 on the masked mean of the VLM prefix output (value_after_vlm)", "ppo": "GAE 0.99 / 0.95, normalised advantages, clip 0.2, dual clip 3.0, value clip 0.2, Adam (0.9, 0.95), grad clip 1.0, actor lr 2e-5 (LoRA + noise head), value lr 1e-4, one epoch per 4-episode batch, minibatch 16 chunks", "test": "deterministic ODE through the trained LoRA (noise head off), 10 steps, as every adapter arm"},
             "parity": "RLinf-Pi05-LIBERO-SFT on LIBERO-Spatial in RLinf's harness: 0.79 success (100 trajectories, job 4171082) against the published 84.6, recorded before this pre-registration; the PPO checkpoints' parity runs alongside",
             "written_before": "any G30e training episode or read episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g30_rl.py", command="pre-registration of plan D2's read g30_rl"))
