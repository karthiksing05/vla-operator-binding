# Plan CONV: composition converges faster on unseen tasks (user direction, 2026-10-21)

**Claim to test.** A compositional scheme reaches a given level of success on an unseen task with less learning budget than
the alternatives. The budget is measured four ways: gradient steps, demonstrations, practice episodes, and GPU-seconds.

**The prediction, graded by docs/SPLIT.md:**
- **G1 and G2 (only the relation or binding is new):** composition converges at **zero** budget. The held fields already
  solve the task. Every alternative has to fit something first.
- **G3 and G4 (a new scene or part):** composition still needs a new unit. Starting that fit from the composition should
  reach the threshold in fewer steps than starting from the policy, from scratch, or from RL.

This plan supersedes the "rollout learning curves" arm (3a) of experiments/prereg/zd_rollouts.py. The causal swaps (3b) stay
in that pre-registration.

## What is already known (so the hypotheses are honest)

- **Offline** (docs/INTERPRETATION.md section 5, ledger `zd_breakdown` stage2):
  - The blend start saves about 35 fit steps on chunk error, 66 to 92 on translation, and about 100 on held-out loss.
  - After about 50 steps, every start ends at the same error (about 1.0 demonstration of value).
  - Held-out loss is lowest near 100 steps; 800-step leaves damage the fine (t = 0.1) field.
  - So the advantage is expected early in the curve, and may vanish at the endpoint.
- **Closed loop** (paper, "Results by grade"):
  - At G1 / G2, reuse with nothing fitted beats creation on all five policies (pi0.5: 312 vs 143 of 400).
  - The 200-step fast fit failed (388 vs 480 of 580; docs/FIT_TIME.md), so "fewer steps at the end of the curve" is not
    already established. It must be measured.
- **Endpoints of the baselines** (G2, G6, G7, G11, docs/BASELINES.md): E3 set b. Per-task LoRA with acceptance 494 / 580; the
  library with acceptance 480; DSRL on the adapter 503 at 10x the practice.

## Design

**Tasks.** Six or more per grade (G1, G2, G3, G4) from LIBERO-90 (results/robot/split_grades.json). The set is the one D1
assembles, with the clean `fold` library, for pi0.5. It is replicated on pi0 and SmolVLA for the G1 / G2 zero-budget claim.
Start states are a new fresh set (seeded from LIBERO's sampler and never used before) for every checkpoint and method. Noise
seed 13.

**Methods** (each produces a policy at every budget point):

| arm | family | start | what is trained |
|---|---|---|---|
| C-sup | ours, supervised | `fold` composition (pointers / prototypes without the task) | rank-4 leaf on top (residual, B = 0) |
| C-uns | ours, unsupervised | soft blend of named contact-cut units | rank-4 leaf on top |
| C-sel | ours, as served | the acceptance rule's choice among policy / composition / leaf at each budget | — |
| L-pol | SFT-LoRA (TAIL family) | frozen policy | rank-16 per-task LoRA |
| L-pol4 | SFT-LoRA, matched rank | frozen policy | rank-4 per-task LoRA (same capacity as our leaf) |
| F-pol | full SFT | frozen policy | action expert, all parameters |
| M-k | multi-demo SFT | frozen policy | rank-16 LoRA on k = 2, 5, 50 demos (demonstration axis) |
| TA / AVG | adapter composition | donor per-task LoRAs, summed or averaged | then fitted like L-pol |
| RL | pi_RL Flow-Noise (LoRA), DSRL | policy, and the one-demo LoRA | practice episodes only (practice axis) |
| STRAP | retrieval | policy | LoRA on demo + retrieved data |

**Budget points.**
- Gradient steps: 0, 25, 50, 100, 200, 400, 800, 1600 (checkpoints saved during one fit; resumable).
- Demonstrations: 1, 2, 5, 50.
- Practice episodes (RL, acceptance): 0, 10, 25, 50, 100, 200.
- GPU-seconds: every arm's wall-clock is logged, including prefix caching, practice rollouts and acceptance, and every curve is
  re-plotted on that axis.

**Readout.** At each budget point, closed-loop success on 10 episodes per task. Use 20 episodes at 0, 100, 400 and the
endpoint. The offline held-out flow loss and per-component chunk error at the same checkpoints link this plan to D1.

**Convergence metrics:**
1. **Budget to threshold.** The smallest budget at which a task's success reaches τ, where τ is 80% of that task's
   ceiling (the 50-demo SFT success) and 0.5 absolute is a second threshold. Tasks that never reach τ are right-censored.
   Compare arms with a Kaplan-Meier "fraction of tasks converged vs budget" curve, plus a paired log-rank test, per grade.
2. **Area under the success-vs-log-budget curve**, normalised (paired, task-level bootstrap CI).
3. **Success at fixed small budgets** (0, 50, 100 steps; 1 demo; 10 practice episodes): paired sign tests.
4. **Stream convergence.** On W1 order a (30 arrivals): cumulative success, and demonstrations / steps / GPU-seconds spent per
   arrival, plotted against arrival index. This shows forward transfer: later arrivals need less as the library grows.
   Compare against sequential FT, ER, EWC, PackNet and the per-task LoRA.

**Pre-registered hypotheses (pi0.5):**
- H1 (G1 / G2, zero budget): C-sel at budget 0 is non-inferior (margin 0.05) to L-pol at 1,000 steps, and beats L-pol at 0 / 25 / 50 steps.
- H2 (G3 / G4): C-sup and C-uns reach τ at a smaller median step budget than L-pol and L-pol4 (log-rank, Holm across the two arms).
- H3: the normalised AUC of C-sel exceeds every alternative's AUC on the steps axis and on the GPU-seconds axis. It is tested
  on the GPU-seconds axis as well because a composition costs practice episodes, and those are charged to it.
- H4 (stream): the GPU-seconds spent per arrival fall with arrival index for the library learner (negative Spearman slope)
  and not for the per-task baselines.
- H5 (theory link): steps-to-τ is predicted from step-0 quantities: the taught share (D1) and the held-out flow loss of the
  start. The prediction is fitted on half the tasks and scored on the other half (Spearman > 0.5).
- Expected failure, stated in advance: at large budgets every arm converges to the same endpoint (D1 surprise 2). The claim
  is about speed and budget, not the asymptote.

**Fairness rules:**
- One demonstration per task, the same one, for every one-demo arm.
- Acceptance practice uses the same 10 episodes, stored states 0-9, and is charged to every arm that uses it.
- Learning rates follow each method's own recipe, plus one shared sweep on 3 development tasks that are excluded from the reads.
- Hyperparameters are frozen before the reads.

**Cost estimate** (one E3-style read is about 4.4 A40-h per 600 episodes):
- 24 tasks x 8 step points x 10-20 episodes x about 8 arms is about 25k episodes, or about 180 A40-h.
- The fits add about 40 A40-h. The stream adds about 30 A40-h.
- Everything runs on SLURM overcap in resumable 4-h jobs.
- Run pi0.5 first. The pi0 / SmolVLA replication of H1 adds about 40 A40-h.

## Beyond LIBERO

The same curves apply wherever composition arms exist. On EgoDex (X3) the metric is best-of-K distance instead of success:
budget to reach the G0 parity level on G1 / G4 folders. On CALVIN (X1) it is the length of the chain of five, and on
Meta-World (X4), once our own subset fine-tune exists, the same success curves.

## Files

- New: experiments/conv_curves.py (fits with checkpoints, launch, read), experiments/prereg/conv_main.py (ledger
  m3_prereg tag conv_main), ledger conv_curves, figures/robot/conv_*.png (Kaplan-Meier, AUC, stream).
- Reused: zd_breakdown.py's sources, g30_baselines.py / g30_rl.py fits, r3_fewshot.py, c0_fit.py --cache, launch_pack.py,
  fresh_states.py.
