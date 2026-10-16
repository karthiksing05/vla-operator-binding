# One Example In, New Tasks Out: Composing a Frozen Robot Policy's Object and Goal Prototypes

*Working draft. Every number is read from `ledger/` and audited by `experiments/audit_numbers.py`; sections marked PENDING wait
on pre-registered reads that are running. Citations were checked against their arXiv pages on 2026-10-15.*

## Abstract

A vision-language-action policy fine-tuned on a set of tasks fails most new tasks in the same rooms, although it can reach,
grasp, carry and place. We ask whether a frozen policy can take on new tasks from at most one demonstration each by composing
small prototypes learned from earlier tasks, storing little and damaging nothing. Each step of a task is split at the policy's
own grasp: an OBJECT prototype acts during the approach (an instance child when the instruction names one) and a GOAL
prototype acts during the transport, each a rank-4 adapter consolidated over every demonstration of its concept. Practice
decides between the composition and the policy alone. On pi0.5 and LIBERO-90, on start states no design decision saw, the
kept library succeeds 499 of 600 times against 273 for the policy alone, and level with a separate one-demonstration adapter
per task given the same acceptance test: over 66 tasks, 931 against 935 of 1,320 (difference -0.003, 95% interval -0.063 to
0.058), with 22 prototypes for 30 tasks in place of 29 adapters (102.9 MB against 545.4 MB). Learned as a stream that requests
a demonstration only when practice fails, it needs 17 and 16 demonstrations over 30 arrivals and damages no earlier task.
Concepts learned only from other tasks recombine where the policy lacks the skill (145 against 25 on the tasks it fails) and
interfere where it has it, so practice must choose; factorisation beats a monolithic adapter trained on the same data (365 against
190). On two more flow-matching policies the library lifts SmolVLA from 20 to 203 and pi0 from 85 to 324 of 600 successes; per-task adapters lead on both, by less on pi0.
It beats retrieval (STRAP, 480 against 296 of 580) and noise-space reinforcement learning on the frozen policy (DSRL, 499 against
314 of 600). On a published compositional OOD benchmark with no demonstration of any test task, where every concept is one the
policy already holds, prototypes learn nothing; composing the policy's own fields under each concept's training instruction,
switched at the grasp, lifts 147 to 174 of 400. The policy's own likelihood tells the two kinds of concept apart without overlap
(AUC 1.0), the detector of new primitives Wang et al. lack.

## 1 Introduction

- The failure mode: in familiar scenes the policy runs a trained routine on the wrong object; on the 6 tasks only a per-task
  adapter solves, it grasps the wrong object first in 114 of 120 episodes (Results).
- The idea: a task is a recombination of factors (which object, where it goes); learn each factor once, as a prototype pooled
  over every demonstration of it, and compose.
- Lineage: Wang et al. ([arXiv](https://arxiv.org/abs/2605.07078), our group's companion work) select and compose a frozen generative model's prototypes
  for an out-of-distribution query; their stated gap is a mechanism to detect, consolidate and reuse new primitives. We supply
  one for an embodied policy: likelihood detects a new concept, and practice, not likelihood, selects among concepts.
- Contributions (Limitations states the limits):
  1. a factored, hierarchical library of concepts for a frozen flow-matching VLA, composed at the policy's own grasp: a concept
     the policy lacks is a low-rank prototype, a concept it holds is its own field under the instruction that taught it;
  2. a continual learner that asks for a demonstration only when practice fails, with Cobweb-style updates of every concept on
     the demonstration's path;
  3. a diagnosis of where a VLA fails new tasks (the approach), and of what its prototypes encode;
  4. the policy's own likelihood as a detector of new concepts (Wang et al.'s missing mechanism), and negative results that scope
     the claim: query-likelihood selection among new concepts, demonstration-free transfer to new objects, prototypes of held
     concepts;
  5. a pre-registered, paired, audited protocol on fresh start states, and a defect found and fixed in a public LIBERO-90 port.
- Scope: flow-matching and diffusion policies. A prototype is a low-rank change to the policy's velocity field, and a step
  switches which concept's change is on in each phase. Our companion ICLR-2027 submission finds the same split for image
  diffusion: composing the scores of single-object prompts places objects that one prompt naming them all does not.

## 2 Related work

Composition of generative models (Composable Diffusion, Liu et al., ECCV 2022, [arXiv](https://arxiv.org/abs/2206.01714)) and of
fine-tuned weights (task arithmetic, Ilharco et al., ICLR 2023, [arXiv](https://arxiv.org/abs/2212.04089)); test-time prototype
discovery (Wang et al., [arXiv](https://arxiv.org/abs/2605.07078), our group's companion work); vision-language-action policies
(pi0, [arXiv](https://arxiv.org/abs/2410.24164); pi0.5, CoRL 2025, [arXiv](https://arxiv.org/abs/2504.16054); SmolVLA,
[arXiv](https://arxiv.org/abs/2506.01844)); adapting a policy from few demonstrations by retrieval (STRAP, ICLR 2025,
[arXiv](https://arxiv.org/abs/2412.15182)) and by reinforcement learning in its noise space (DSRL, CoRL 2025,
[arXiv](https://arxiv.org/abs/2506.15799)); continual learning with frozen backbones and per-task modules (PackNet, Piggyback, SupSup,
C-LoRA; docs/RELATED.md); the benchmark (LIBERO, NeurIPS 2023, [arXiv](https://arxiv.org/abs/2306.03310)).

## 3 Method

(docs/METHOD.md; Figures: one composed step; the concept hierarchy; the stream learner.) Object, instance and goal concepts are
read from the instruction; each is a rank-4 LoRA on the action expert fitted by flow matching on the frames of its phase in
every demonstration of the concept, each demonstration from the task's own scene. The object (or instance) prototype is on until
the policy's gripper stays closed for three queries; the goal prototype then acts until the release. Ten practice episodes keep
the composition only if it beats the policy. As a stream, an arrival requests its one demonstration only when neither the
policy nor the composition reaches 8 of 10; the demonstration refits the instance, type and goal concepts it belongs to.

## 4 Protocol

Pre-registered before any episode; paired episodes (same start state and noise for every arm); fresh start states from LIBERO's
own placement sampler at unused seeds; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes;
non-inferiority margin 0.05. The strong baseline is a rank-16 LoRA per task fitted on the task's own scene's demonstration and
given the same acceptance test.

## 5 Results

### Where the policy fails, and factoring the step
Goal prototypes alone do not close the gap: kept 343 against 271 zero-shot, but 324 against 454 for the per-task adapter.
The approach is where it fails (114 of 120 wrong first grasps). Object prototypes fix it: on the corrected data the factored
library keeps 439 against 272 zero-shot and 420 against 478 for the per-task adapter, with 45 of the 58 missing successes on
four tasks that name an instance of the black bowl.

### A hierarchy of concepts
Instance children (screen: 37 of 40 against 13 for the type prototype) close the instance gap. Confirmatory (E3, unseen start
states): 499 against 273 zero-shot; 480 against 494 for the per-task adapter with acceptance (difference -0.024, 95% interval
-0.074 to 0.024); the instance children 74 of 80 against 17 for the type prototype.

### Replication on 37 new tasks
451 of 740 against 151 zero-shot; 451 against 441 for the per-task adapter with acceptance. Pooled over 66 tasks the library
and the per-task adapter are level (931 against 935 of 1,320; difference -0.003, 95% interval -0.063 to 0.058);
non-inferiority at 0.05 is just missed (the lower bound misses it by 0.013).

### The library as a stream
With the request rule (W1): 17 and 16 demonstrations in two orders (21 and 20 under a looser rule), no damage to earlier
arrivals in either order, served 465 and 496 of 600 against 273 zero-shot.

### Does composition generalise?
With each concept refit with no demonstration of a held-out fold (5 folds, 53 eligible tasks), recombined concepts alone succeed
365 of 1,060 times against 354 for the policy: the pre-registered test fails. With practice choosing, 484 against 354. A
monolithic adapter trained on the same demonstrations succeeds 190 times, below the policy, so the factorisation matters (365
against 190). Exploratory: on the 33 tasks the policy fails, recombination gives 145 against 25; on the 20 it performs, 220 against
329. Recombination supplies a missing skill and interferes with a present one; practice separates the two.

### Against adapter composition
On 52 of the held-out tasks, summing the per-task adapters of an object donor and a goal donor collapses (71 of 1,040); averaging
them stays at the policy's level (328 against 336); the recombined concepts reach 345; with practice they reach 466 against 375 for the
better baseline.
### A second policy
The same library refit on SmolVLA, at the same tasks, start states and seed: 203 of 600 against 20 for SmolVLA alone; against
SmolVLA's per-task LoRA with acceptance 189 against 274 of 580 (difference -0.147, 95% interval -0.266 to -0.029). Composition
carries over as a large gain over the frozen policy; the per-task adapter leads by more on the weaker policy.

### Against reinforcement learning in the noise space and retrieval
DSRL steering the frozen policy from 100 practice episodes per task, no demonstration, succeeds 313 of 600 times (314 with
acceptance); the library reaches 499 (difference 0.308, 95% interval 0.188 to 0.433). DSRL steering each task's one-demo
adapter is the strongest arm we ran: 503 of 580 with acceptance against 480 for the library (difference -0.040, 95% interval
-0.098 to 0.012), at ten times the practice and a per-task adapter and actor. STRAP, retrieving from the library's own data and
more, succeeds 192 of 580 alone and 296 with acceptance, against 480 (difference 0.317, 95% interval 0.200 to 0.440).

### Composing the policy's own fields: a published compositional OOD benchmark
On Li's LIBERO-Goal-OOD and LIBERO-Spatial-OOD ([arXiv](https://arxiv.org/abs/2505.03500); 20 tasks, each pairing one
training task's object with another's target, no demonstration of any test task), a concept the policy already holds is the
policy's own field under the instruction of a training task that taught it, switched at the grasp. With practice choosing, the
kept arms succeed 174 of 400 times against 147 for the policy (discordant pairs 30 to 3; task-level difference 0.068, 95%
interval 0.007 to 0.145), in both suites. This is score composition in the sense of our ICLR-2027 companion: the policy
carries each concept under the instruction that taught it, and the composed step reads them one phase at a time.

### The policy's likelihood detects a new concept
Wang et al. name a detector of new primitives as their missing mechanism. The frozen policy's flow-matching loss on a
concept's frames separates the 34 concepts of its own training tasks from the 40 LIBERO-90 concepts it never saw without
overlap (AUC 1.0; held at most 0.0082, new at least 0.026), and the same words are new in new scenes (the cream cheese 0.039
against 0.0040). Likelihood detects novelty although it does not select among new concepts (W0); it decides whether a concept is
stored as a pointer to the policy's own field or fitted as a prototype. Exploratory; the held concepts are scored on their
training demonstrations.

### A third policy
On pi0 the library succeeds 324 of 600 times against 85 for pi0 alone; against pi0's per-task LoRA with acceptance, 308
against 330 of 580 (difference -0.038, 95% interval -0.166 to 0.081): non-inferiority is not established.

### The switch at the grasp
With both prototypes on for the whole episode the library succeeds 281 of 600 times, level with the policy (273); switched at
the policy's own grasp it succeeds 495. The phase gating carries the gain.

### What the prototypes encode
Object prototypes are nearly orthogonal to one another (mean cosine 0.058) and to goal prototypes (0.037); both act in the same
layers, differing in direction; refit without one demonstration a prototype keeps a cosine of 0.64 with the full one.

## 6 What does not work
- Prototypes of concepts the policy already holds (Li's compositional OOD suites, 20 tasks): fitted on pi0.5's own training
  demonstrations they learn nothing (flow loss 0.0024 at the start, 0.0038 at the end, against 0.060 to 0.0062 on LIBERO-90
  demonstrations), and the kept arms equal the policy (154 against 157 of 400). Results, "Composing the policy's own fields",
  gives the remedy.
- Selecting prototypes by a held-out demonstration's likelihood (as Wang et al. do): the right object concept on top for 12 of
  29 queries.
- A new object without a demonstration: borrowed prototypes help on 2 of the 17 tasks the policy fails.
- Re-checking every earlier arrival after each refit (W2): no gain over W1.

## 7 A data defect
The IPEC port of LIBERO-90 merges tasks that share an instruction: 16 tasks were given another scene's demonstration, including
the per-task baselines. Fixed by matching each scene's block of episodes to the task's rendered first frames.

## 8 Limitations
Single-step pick-and-place; flow-matching policies only; one benchmark (LIBERO-90); practice needs a simulator; a new object needs its one demonstration;
non-inferiority to per-task adapters not established at 0.05.
