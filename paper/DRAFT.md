# Iterating a Compositional System: New Tasks for a Frozen Flow-Matching Policy from One Example or None

*Working draft. Every number is read from `ledger/` and audited by `experiments/audit_numbers.py`; sections marked PENDING wait
on pre-registered reads that are running. Citations were checked against their arXiv pages on 2026-10-15.*

## Abstract

A flow-matching vision-language-action policy fine-tuned on a set of tasks fails most new tasks in the same rooms, although it
can reach, grasp, carry and place. We build one compositional system around such a frozen policy and iterate it, each iteration
answering a failure the previous one measured. The system reads a task's concepts from its instruction (which object, which
instance, where it goes) and switches between them at the policy's own grasp; each concept is either a low-rank change to the
policy's velocity field, fitted on every demonstration of it, or, when the policy already holds the concept, the policy's own
field under the instruction that taught it; the policy's likelihood decides which, and practice keeps only what beats the
policy. On pi0.5 and LIBERO-90, on start states no design decision saw, the kept library succeeds 499 of 600 times against 273
for the policy alone, and level with a separate one-demonstration adapter per task given the same acceptance test: over 66
tasks, 931 against 935 of 1,320 (difference -0.003, 95% interval -0.063 to 0.058), with 22 prototypes for 30 tasks in place of
29 rank-16 adapters (102.9 MB against 545.4 MB; against rank-4 adapters, level at 480 against 480 of 580 with 102.9 MB against 135.7 MB). With task leaves at the bottom of the hierarchy, kept only where practice prefers
them, it beats the per-task adapter: 994 against 932 of 1,320 on new start states (difference 0.047, 95% interval 0.008 to
0.091), with 675.9 MB against 902.8 MB. Learned as a stream that requests a demonstration only when practice fails, it needs 17
and 16 demonstrations over 30 arrivals and damages no earlier task. On a published compositional OOD benchmark with no
demonstration of any test task, composing the policy's own fields lifts 147 to 174 of 400 on pi0.5, 35 to 98 on pi0 and 8 to 25
on SmolVLA; when practice chooses each held concept's source instruction, pi0.5 reaches 269 and passes the 222 of a prior method's
hand-chosen pairs. The policy's own likelihood tells held from new concepts without overlap (AUC 1.0), a detector of the kind Wang
et al. leave to future work. The system beats retrieval (STRAP, 480 against 296 of 580) and noise-space reinforcement learning on the
frozen policy (DSRL, 499 against 314 of 600).

## 1 Introduction

- Scope: flow-matching VLAs, whose action expert integrates a learned velocity field (pi0, pi0.5, SmolVLA). Every unit of the
  system is a change to that field for one phase of a step, which makes composition a matter of choosing, per phase, which
  field the policy integrates. Our companion ICLR-2027 submission finds the same split for image diffusion: composing the scores
  of single-object prompts places objects that one prompt naming them all does not.
- The failure mode: in familiar scenes the policy runs a trained routine on the wrong object; on the 6 tasks only a per-task
  adapter solves, it grasps the wrong object first in 114 of 120 episodes (Results).
- The approach: one compositional system, iterated. Each iteration is a measured weakness of the previous version and one
  change motivated by it, pre-registered and tested on start states no earlier decision saw; Results follows the iterations in
  order, then tests the finished system against alternatives.
- Lineage: Wang et al. ([arXiv](https://arxiv.org/abs/2605.07078)) select and compose a frozen
  generative model's prototypes for an out-of-distribution query; their stated gap is a mechanism to detect, consolidate and
  reuse new primitives. We supply one for an embodied flow policy: likelihood detects a new concept, and practice, not
  likelihood, selects among concepts.
- Contributions (Limitations states the limits):
  1. a compositional library for a frozen flow-matching VLA whose units are instruction concepts (object type, instance as a
     child of its type, goal), each a low-rank change to the action expert fitted on the frames of its phase pooled over every
     demonstration of the concept, switched at the policy's own grasp; skill-level adapter banks composed by a learned router
     exist (PRISM), but we know of no prior work that factors a VLA's adapters into object and goal concepts;
  2. for concepts the policy already holds, a zero-byte unit: the policy's own field under a training instruction, switched at
     the grasp. This extends Li's prompt switching (hand-paired base tasks, a fixed switching step, pi0): practice chooses the
     sources with no hand labels (better than Li's pairs on pi0.5, level with pi0^S on pi0), the grasp trigger needs no per-task
     tuning, and the operator runs inside a learner that also stores new concepts, on four policies;
  3. the frozen policy's own flow-matching loss on a concept's phase frames as the decision to point or to fit, with no added
     parameters (earlier continual learners decide expansion with separate models, e.g. CLARE's feature similarity), and the
     finding that likelihood detects novelty but does not select among new concepts; a continual learner that asks for a
     demonstration only when practice fails and refits every concept on its path (cf. LOTUS, which updates and adds skills);
  4. the iteration record itself: each failure and the change it motivated, with negative results that scope the claim
     (query-likelihood selection among new concepts, demonstration-free transfer to new objects, prototypes of held concepts);
  5. a pre-registered, paired, audited protocol on fresh start states, and a defect found and fixed in a public LIBERO-90 port.

## 2 Related work

Composition of generative models (Composable Diffusion, Liu et al., ECCV 2022, [arXiv](https://arxiv.org/abs/2206.01714)) and of
fine-tuned weights (task arithmetic, Ilharco et al., ICLR 2023, [arXiv](https://arxiv.org/abs/2212.04089)); test-time prototype
discovery (Wang et al., [arXiv](https://arxiv.org/abs/2605.07078)); flow-matching vision-language-action
policies (pi0, [arXiv](https://arxiv.org/abs/2410.24164); pi0.5, CoRL 2025, [arXiv](https://arxiv.org/abs/2504.16054); SmolVLA,
[arXiv](https://arxiv.org/abs/2506.01844); GR00T N1, [arXiv](https://arxiv.org/abs/2503.14734); FLOWER, CoRL 2025,
[arXiv](https://arxiv.org/abs/2509.04996); X-VLA, [arXiv](https://arxiv.org/abs/2510.10274)); adapting a policy from few
demonstrations by retrieval (STRAP, ICLR 2025, [arXiv](https://arxiv.org/abs/2412.15182)) and by reinforcement learning in its
noise space (DSRL, CoRL 2025, [arXiv](https://arxiv.org/abs/2506.15799)); continual learning with frozen backbones and per-task
modules (PackNet, Piggyback, SupSup, C-LoRA; docs/RELATED.md); the benchmarks (LIBERO, NeurIPS 2023,
[arXiv](https://arxiv.org/abs/2306.03310); Li's compositional OOD suites, [arXiv](https://arxiv.org/abs/2505.03500)).

**Continual imitation with module libraries.** Per-task adapters on a frozen pretrained policy avoid forgetting (TAIL, ICLR 2024,
[arXiv](https://arxiv.org/abs/2310.05905); our per-task LoRA baseline is of this kind); later work shares modules across tasks
with learned routers (Sparse Diffusion Policy, CoRL 2024, [arXiv](https://arxiv.org/abs/2407.01531); DMPEL,
[arXiv](https://arxiv.org/abs/2506.05985)), grows libraries of discovered skills (LOTUS, ICRA 2024,
[arXiv](https://arxiv.org/abs/2311.02058); IsCiL, NeurIPS 2024, [arXiv](https://arxiv.org/abs/2410.22658)), or expands adapters
only where needed (CLARE, RA-L 2026, [arXiv](https://arxiv.org/abs/2601.09512)); PRISM (NeurIPS 2026) routes frozen low-rank
skill adapters on the action expert per chunk, and PhaseLoRA ([arXiv](https://arxiv.org/abs/2608.15285)) conditions one
action-expert LoRA on the manipulation phase. Our units are instruction concepts, not tasks or skills, switched by the policy's own
grasp, not a learned router, and include zero-byte pointers to the policy's own conditionals. Pretrained VLAs forget little under
fine-tuning with replay ([arXiv](https://arxiv.org/abs/2603.03818)).
**Compositional generalisation of VLAs.** On the suites we use, Li switches pi0's prompt mid-episode and interpolates its text
latents; ACT-VLA ([arXiv](https://arxiv.org/abs/2607.00351)) and AC-VLA ([arXiv](https://arxiv.org/abs/2607.15714)) retrain pi0.5
on synthesised or decomposed compositional demonstrations, training-based references for our frozen policy. Composition of
diffusion and flow policies (PoCo, RSS 2024, [arXiv](https://arxiv.org/abs/2402.02511); GPC, ICLR 2026,
[arXiv](https://arxiv.org/abs/2510.01068)) combines whole policies at every step; we keep one policy frozen and switch its
conditionals per phase, and we compare with Li's operators on the same tasks (G4d).
**Novelty in generative policies.** Denoising error detects out-of-distribution inputs (Graham et al., CVPR-W 2023,
[arXiv](https://arxiv.org/abs/2211.07740)); flow-based densities flag failing policy rollouts (FAIL-Detect, RSS 2025,
[arXiv](https://arxiv.org/abs/2503.08558)); interactive imitation requests help on novel states (ThriftyDAgger,
[arXiv](https://arxiv.org/abs/2109.08273)). We use the policy's own flow loss per concept to decide whether a concept needs new
parameters.

## 3 The system

(docs/METHOD.md; Figures: one composed step; the concept hierarchy; the stream learner.)

**Concepts.** Object, instance and goal concepts are read from the instruction. An instance is a child of its object type.

**Two kinds of unit, one rule to choose.** A concept is scored by the frozen policy's own flow-matching loss on its phase frames
(a detector of the kind Wang et al. leave to future work). A concept the policy holds under its instruction and scene is stored as a
pointer to the policy's own field under the training instruction that taught it (zero bytes); a new concept is fitted as a
prototype, a rank-4 LoRA on the action expert fitted by flow matching on the frames of its phase in every demonstration of the
concept, each demonstration from the task's own scene.

**Composition at the grasp.** The object (or instance) unit is on until the policy's gripper stays closed for three queries; the
goal unit then acts until the release. The learner thus composes held concepts the way composable diffusion composes the
scores of single prompts, and grows the library only where the policy's likelihood says it must.

**Practice and the stream.** Ten practice episodes keep the composition only if it beats the policy. As a stream, an arrival
requests its one demonstration only when neither the policy nor the composition reaches 8 of 10; the demonstration refits the
instance, type and goal concepts it belongs to.

## 4 Protocol

Pre-registered before any episode; paired episodes (same start state and noise for every arm); fresh start states from LIBERO's
own placement sampler at unused seeds; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes;
non-inferiority margin 0.05. The strong baseline is a rank-16 LoRA per task fitted on the task's own scene's demonstration and
given the same acceptance test.

## 5 Iterations of the system

Each iteration names the weakness the previous version showed, the one change, and the pre-registered result.
Figure (figures/robot/iterations.png) summarises the ladder: on LIBERO-90 the system against the per-task adapter at
iterations 1-3 and 8, and on Li's suites the policy alone, word-matched, hand-chosen and practice-chosen sources.

### Iteration 1: goal units after the grasp
Weakness: the frozen policy fails most new tasks. Change: one goal prototype per goal concept, acting from the grasp. Result:
kept 343 against 271 zero-shot, but 324 against 454 for the per-task adapter. The approach is where it fails (114 of 120 wrong
first grasps).

### Iteration 2: object units before the grasp
Change: an object prototype for the approach. Result, on the corrected data: the factored library keeps 439 against 272
zero-shot and 420 against 478 for the per-task adapter, with 45 of the 58 missing successes on four tasks that name an instance
of the black bowl.

### Iteration 3: a hierarchy of concepts
Change: instance children under object types (screen: 37 of 40 against 13 for the type prototype). Confirmatory (E3, unseen
start states): 499 against 273 zero-shot; 480 against 494 for the per-task adapter with acceptance (difference -0.024, 95%
interval -0.074 to 0.024); the instance children 74 of 80 against 17 for the type prototype. Replication on 37 new tasks: 451 of
740 against 151 zero-shot; 451 against 441 for the per-task adapter with acceptance. Pooled over 66 tasks the library and the
per-task adapter are level (931 against 935 of 1,320; difference -0.003, 95% interval -0.063 to 0.058); non-inferiority at 0.05
is just missed (the lower bound misses it by 0.013).

### Iteration 4: learning as a stream
Change: a task asks for its one demonstration only when practice fails (W1). Result: 17 and 16 demonstrations in two orders (21
and 20 under a looser rule), no damage to earlier arrivals in either order, served 465 and 496 of 600 against 273 zero-shot.

### Iteration 5: concepts the policy already holds
Weakness: on Li's compositional OOD suites ([arXiv](https://arxiv.org/abs/2505.03500); 20 tasks, each pairing one training
task's object with another's target, no demonstration of any test task), prototypes fitted on pi0.5's own training
demonstrations learn nothing (flow loss 0.0024 at the start, 0.0038 at the end, against 0.060 to 0.0062 on LIBERO-90
demonstrations), and the kept arms equal the policy (154 against 157 of 400). Change: a held concept is the policy's own field
under the instruction of a training task that taught it, switched at the grasp. Result: with practice choosing, the kept arms
succeed 174 of 400 times against 147 for the policy (discordant pairs 30 to 3; task-level difference 0.068, 95% interval 0.007
to 0.145), in both suites. This is score composition in the sense of our ICLR-2027 companion: the policy carries each concept
under the instruction that taught it, and the composed step reads them one phase at a time. The same composition, unchanged,
lifts pi0 from 35 to 98 and SmolVLA from 8 to 25 of 400 on the same tasks (both pass under Holm).
The operator is Li's prompt switching, which reproduces on our harness: with Li's hand-chosen source pairs, pi0 succeeds 292
of 400 times switched at their fixed step and 305 switched at its own grasp, against 86 and 74 with our automatically chosen
sources (pi0.5: 218 and 222 against 157 and 160). The gap lies entirely in the sources: Li's pairs pick training tasks whose
trained motion reaches where the new object lies, ours match the instruction's words. The grasp trigger matches the tuned fixed
step without per-task tuning. On X-VLA the composition with practice lifts 75 to 120 of 400.

### Iteration 9: practice chooses the sources
Weakness: the word-matching source rule trails Li's hand-chosen pairs. Change: practice chooses each held concept's source
among the 16 single-step pick-and-place training instructions of LIBERO-Goal and LIBERO-Spatial (object source first, then goal
source, 5 episodes per candidate), with no hand labels and no privileged state. Result, on new start states: on pi0.5 the chosen
pairs succeed 269 of 400 times against 222 for Li's hand-chosen pairs at the grasp (task-level difference 0.117, 95% interval
0.022 to 0.240), 160 for our word rule and 147 for the policy alone; on pi0 292 against 305 for Li's pairs (difference -0.033,
95% interval -0.138 to 0.030; level with Li's fixed-step pi0^S at 292), against 74 for the word rule and 35 for the policy.

### Iteration 6: the likelihood decides the kind of unit
Weakness: two kinds of unit need a rule to choose between them. Change: the frozen policy's flow-matching loss on a concept's
frames. It separates the 34 concepts of its own training tasks from the 40 LIBERO-90 concepts it never saw without overlap (AUC
1.0; held at most 0.0082, new at least 0.026), and the same words are new in new scenes (the cream cheese 0.039 against 0.0040).
Likelihood detects novelty although it does not select among new concepts (W0). Exploratory; the held concepts are scored on
their training demonstrations. The routing is right in both directions: on 20 LIBERO-90 tasks whose concepts share words with
held ones, the field composition equals the policy (88 against 85 of 200) and the prototypes the router assigns reach 173.
A text detector that matches a concept's words to held concepts would route 16 of these 40 new concepts to a pointer (28 under
the content-word rule); the likelihood routes none.

### Iteration 7: one learner, both kinds of task
On a joint stream of 50 arrivals (Li's 20 OOD tasks, which have no demonstrations, and W1's 30 LIBERO-90 tasks), one learner
with both kinds of unit requests 17 and 16 demonstrations, all for LIBERO-90 concepts, and serves 641 and 672 of 1,000 against
420 for the policy alone. Pointers carry no weights, so they change no request and damage nothing.

### Iteration 8: task leaves
Weakness: the system stays level with, not above, a per-task adapter. Change: below type and instance, a task the shared
concepts cannot serve keeps its own one-demonstration adapter as a leaf, kept only where practice prefers it. Result, on a new
set of start states over the 66 tasks: 994 of 1,320 against 932 for the per-task adapter with acceptance (discordant pairs 100
to 38; task-level difference 0.047, 95% interval 0.008 to 0.091), superior at the pre-registered criterion, with a leaf on 24
tasks where the per-task baseline keeps an adapter on 48 and with 675.9 MB against 902.8 MB.

## 6 Tests of the system

### Does composition generalise?
With each concept refit with no demonstration of a held-out fold (5 folds, 53 eligible tasks), recombined concepts alone succeed
365 of 1,060 times against 354 for the policy: the pre-registered test fails. With practice choosing, 484 against 354. A
monolithic adapter trained on the same demonstrations succeeds 190 times, below the policy, so the factorisation matters (365
against 190). Exploratory: on the 33 tasks the policy fails, recombination gives 145 against 25; on the 20 it performs, 220
against 329. Recombination supplies a missing skill and interferes with a present one; practice separates the two.

### Other flow-matching policies
The same library refit on SmolVLA, at the same tasks, start states and seed: 203 of 600 against 20 for SmolVLA alone; against
SmolVLA's per-task LoRA with acceptance 189 against 274 of 580 (difference -0.147, 95% interval -0.266 to -0.029). On pi0 the
library succeeds 324 of 600 times against 85 for pi0 alone; against pi0's per-task LoRA with acceptance, 308 against 330 of 580
(difference -0.038, 95% interval -0.166 to 0.081): non-inferiority is not established on either. X-VLA alone already succeeds
534 of 600 times on these tasks; the kept library 542 (not significant), and 526 against 544 of 580 for X-VLA's per-task LoRA
with acceptance: a strong zero-shot policy leaves the library little to add.

### Against adapter composition
On 52 of the held-out tasks, summing the per-task adapters of an object donor and a goal donor collapses (71 of 1,040);
averaging them stays at the policy's level (328 against 336); the recombined concepts reach 345; with practice they reach 466
against 375 for the better baseline.

### Against reinforcement learning in the noise space and retrieval
DSRL steering the frozen policy from 100 practice episodes per task, no demonstration, succeeds 313 of 600 times (314 with
acceptance); the library reaches 499 (difference 0.308, 95% interval 0.188 to 0.433). DSRL steering each task's one-demo
adapter is the strongest arm we ran: 503 of 580 with acceptance against 480 for the library (difference -0.040, 95% interval
-0.098 to 0.012), at ten times the practice and a per-task adapter and actor. DSRL steering the library's composition does not
help: 493 against 499 of 600 for the library alone, and 474 against 503 of 580 for DSRL on the per-task adapter. STRAP,
retrieving from the library's own data and more, succeeds 192 of 580 alone and 296 with acceptance, against 480 (difference
0.317, 95% interval 0.200 to 0.440).

### Ablations
The switch at the grasp: with both prototypes on for the whole episode the library succeeds 281 of 600 times, level with the
policy (273); switched at the policy's own grasp it succeeds 495. The phase gating carries the gain. Capacity: refit at rank 8 or
rank 16 instead of 4, the library is unchanged (498 and 504 against 499 of 600) and stays short of non-inferiority to the
per-task adapter (rank 16: 485 against 494 of 580; difference -0.016, 95% interval -0.064 to 0.033).

### What each unit does to the motion
Recorded trajectories on ten tasks show the division of labour: the first grasp takes the named object in 44 of 50
episodes with the object unit alone and 47 of 50 with both units, against 11 of 50 for the policy alone and 12 of 50 with the
goal unit alone; the transport separates the arms little. Each instance child takes its own bowl first in 15 of 15 episodes;
the type prototype takes the named bowl in 6 of 15.

### What the prototypes encode
Object prototypes are nearly orthogonal to one another (mean cosine 0.058) and to goal prototypes (0.037); both act in the same
layers, differing in direction; refit without one demonstration a prototype keeps a cosine of 0.64 with the full one.

## 7 What does not work
- Selecting prototypes by a held-out demonstration's likelihood (as Wang et al. do): the right object concept on top for 12 of
  29 queries.
- A new object without a demonstration: borrowed prototypes help on 2 of the 17 tasks the policy fails.
- Re-checking every earlier arrival after each refit (W2): no gain over W1.
- Prototypes of concepts the policy already holds (Iteration 5's weakness) and noise-space refinement on top of the
  composition (DSRL on the library).

## 8 A data defect
The IPEC port of LIBERO-90 merges tasks that share an instruction: 16 tasks were given another scene's demonstration, including
the per-task baselines. Fixed by matching each scene's block of episodes to the task's rendered first frames.

## 9 Limitations
Flow-matching policies only (pi0.5, pi0, SmolVLA, X-VLA); Li's text-latent interpolation, reported above prompt switching on
pi0 in their setup, and routed adapter libraries (CLARE) are not rerun here; choosing a held concept's source by practice costs 160 practice episodes per task; single-step pick-and-place; LIBERO-90 and Li's
OOD suites only; practice needs a simulator; a new object needs its one demonstration; non-inferiority to per-task adapters without task leaves not
established at 0.05.
