# Out-of-Distribution Learning Is Composition: One System, Iterated, on Four Frozen Flow-Matching Policies

*Working draft. Every number is read from `ledger/` and audited by `experiments/audit_numbers.py`; sections marked PENDING wait
on pre-registered reads that are running. Citations were checked against their arXiv pages on 2026-10-15.*

## Abstract

A frozen flow-matching vision-language-action policy fails most new tasks in rooms it knows, although it can reach, grasp,
carry and place. We argue that learning such tasks out of distribution is a problem of composition: a new task is a new
arrangement of concepts the policy already holds or can learn from one example, and the work lies in finding those concepts,
storing them cheaply and recombining them. We test the claim with one compositional system, iterated one pre-registered change
at a time, that assumes nothing about the policy's architecture beyond access to its prototypes: its velocity field under an
instruction, a low-rank change to that field, and the latents of the encoder that reads the instruction. The system reads a
task's concepts (object, instance, goal) from its instruction. A concept the policy holds is a zero-byte pointer to its own
field under a training instruction; a new one is a low-rank prototype fitted on every demonstration of it; the policy's
likelihood decides which (AUC 1.0, a detector Wang et al. leave to future work), and practice chooses sources, operators and
what to keep. On a published compositional OOD benchmark with no demonstration of any test task, the same code lifts all four
policies we test, on new start states: pi0 from 35 to 336 of 400 and pi0.5 from 143 to 312; SmolVLA from 5 to 96 and X-VLA
from 80 to 256. On every one it is level with or above a prior method's text-latent interpolation with hand-chosen task pairs
(316 and 245 on pi0 and pi0.5; 0 and 142 on SmolVLA and X-VLA). No fixed operator carries across architectures: that interpolation is the stronger operator on pi0 and fails on
SmolVLA, and practice picks the operator per task. On LIBERO-90 with one demonstration per new concept, task leaves kept by
practice beat a separate adapter per task (994 against 932 of 1,320; difference 0.047, 95% interval 0.008 to 0.091) with
675.9 MB against 902.8 MB. Learned as a stream, the library requests a demonstration only when practice fails and damages no
earlier task; one adapter trained through the same stream by sequential fine-tuning or experience replay reaches 52 or 335 of
600 against 465.

## 1 Introduction

- Thesis: out-of-distribution learning on a frozen policy is a composition problem. The policy already carries most of a new
  task as concepts it learned under other instructions; what it lacks is the arrangement. We show it on four architectures with
  one system whose only architectural assumption is access to the policy's prototypes (Section 3).
- Scope: flow-matching VLAs, whose action expert integrates a learned velocity field (pi0, pi0.5, SmolVLA, X-VLA). Every unit of the
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
     sources and, per task, the operator (grasp switch or Li's text-latent interpolation, ported to every policy's conditioning
     encoder) with no hand labels, level with or above Li's hand-chosen pairs on all four policies; no single operator wins on
     every architecture, so the choice by practice is what makes the system architecture-agnostic;
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

**Operators for held concepts.** A held concept points at a source training instruction, and two operators read the policy
through it. The grasp switch conditions the policy on the object's source instruction until the grasp and on the goal's
source instruction after it. Text-latent interpolation (Li et al.) keeps the test instruction and shifts the residual stream
of the policy's conditioning encoder, at the instruction's tokens, from the object source's latent toward the goal source's.
Practice chooses each concept's source and, per task, the operator.

**What the framework needs from a policy.** Nothing in the system depends on one architecture. It needs three accesses to a
frozen flow-matching policy: its velocity field under a given instruction (a pointer, and the grasp switch), a low-rank
change to that field (a prototype), and the residual stream of the encoder that reads the instruction, with the position of
the instruction's tokens in it (text-latent interpolation). These are the extraction of prototypes. One interface supplies
them for PaliGemma (pi0, pi0.5), SmolVLM (SmolVLA) and Florence-2 (X-VLA); the rest of the system is shared code.

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
Li's text-latent interpolation (TLI) is the stronger operator on pi0. We port it to the PyTorch pi0 and identify its latents
by Li's own procedure; on the same start states it succeeds 325 of 400 times against 35 for the policy alone (Li publish 0.83
in their setup), and it beats the practice-chosen pairs switched at the grasp (292; discordant pairs 76 to 43; the task-level
difference is -0.083 with a 95% interval from -0.237 to 0.060). Non-inferiority to TLI fails. TLI still uses Li's hand-chosen
pairs and a hand-set interpolation length per task; iteration 10 tests TLI with the practice-chosen pairs.

### Iteration 10: TLI with practice-chosen pairs
Weakness: TLI needs hand-chosen pairs. Change: TLI takes the pairs practice chose in iteration 9 and one interpolation length
(24) for every task. Result, on new start states (set d): TLI with the chosen pairs falls behind on both policies. On pi0 it
succeeds 251 of 400 times against 326 for Li's TLI and 287 for the same pairs switched at the grasp (policy alone 38); on
pi0.5, 178 against 259 and 264 (policy alone 155). Pairs that practice chose for the grasp switch do not carry over to
interpolation. The two policies differ in which operator wins: on pi0 Li's TLI leads the grasp switch with practice-chosen
pairs (difference -0.098, 95% interval -0.263 to 0.055), and on pi0.5 the grasp switch is level with it on set d
(difference 0.013, 95% interval -0.170 to 0.182) and ahead on set c (269 against 240; sign p 0.011). Iteration 11 lets practice
choose the pairs under TLI.

### Iteration 11: practice chooses the pairs under TLI
Change: the practice of iteration 9 runs with TLI as the operator. Result, on set d: on pi0 the chosen pairs succeed 307 of
400 times against 326 for Li's TLI (difference -0.048, 95% interval -0.117 to 0.013) and 251 for TLI with the pairs of
iteration 10; on pi0.5, 249 against 259 (difference -0.025, 95% interval -0.142 to 0.092) and 178. Practice recovers most of
the gap to hand-chosen pairs without hand labels, and picks Li's exact pair on 11 of 20 tasks for pi0 and 7 of 20 for pi0.5.
Non-inferiority to Li's TLI is not established on either policy.

### Iteration 12: practice chooses the operator
Weakness: the better operator depends on the policy and the task. Change: per task, 10 practice episodes choose between the
grasp switch and TLI, each with its practice-chosen pair; ties go to the switch. Result, on new start states (set e): on pi0
the system succeeds 336 of 400 times against 316 for Li's TLI with hand-chosen pairs (discordant pairs 41 to 21; task-level
difference 0.050, 95% interval -0.028 to 0.142), which establishes non-inferiority; on pi0.5, 312 against 245 (difference
0.168, 95% interval 0.058 to 0.290), which is better. The policies alone reach 35 and 143. Practice picks TLI on 8 tasks for
pi0 and 5 for pi0.5. No step uses a hand label: the concepts come from the instruction, the sources and the operator from
practice.

### Iteration 13: practice as a stream
Weakness: iteration 12 practises every task's sources from scratch, even when a concept recurs. Change: Li's 20 tasks arrive
as a stream; a held concept's chosen source is stored and reused by later arrivals, so only unseen concepts are practised.
Result: practice falls from 6,800 to 3,600 episodes, and success falls with it in three of four streams. Against per-task
practice (336 on pi0, 312 on pi0.5 on set e) the streams reach 330 and 288 on pi0 and 254 and 297 on pi0.5 in the two
orders; non-inferiority holds only for pi0 in the first order (difference -0.015, 95% interval -0.043 to 0.000). A source
chosen for a concept in one task's context does not always serve the same concept with another partner, and the loss
depends on which task came first.

### Iteration 14: practice gates reuse
Change: the practice gate the system uses everywhere: an arrival whose reused pairs reach fewer than 8 of 10 practice
episodes re-practises its reused concepts. Result, on new start states (set f): against per-task practice (334 on pi0, 308
on pi0.5) the gated streams reach 333 and 332 on pi0 and 292 and 309 on pi0.5, with 4,250 to 5,460 practice episodes in
place of 6,800. Non-inferiority holds in three of four streams and fails for pi0.5 in the first order (95% lower bound
-0.13). Practice that reuses what earlier tasks taught it, checked by practice, keeps the success of practising each task
from scratch with fewer practice episodes in every stream.

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

**The OOD system on all four policies.** The finished OOD system (iteration 12), unchanged, on SmolVLA and X-VLA
(pre-registered G18; Figure: four_policies.png). SmolVLA succeeds 96 of 400 times against 5 alone and 0 for Li's text-latent
interpolation with Li's pairs; Li's pairs switched at the grasp reach 93 (task-level difference 0.007, 95% interval -0.025 to
0.048), and practice never chooses interpolation there. X-VLA succeeds 256 of 400 times against 80 alone, 142 for Li's
interpolation (difference 0.285, 95% interval 0.130 to 0.450) and 210 for Li's pairs at the grasp (difference 0.115, 95%
interval 0.010 to 0.250); practice chooses interpolation on 4 tasks. With pi0 (336 against 316) and pi0.5 (312 against 245),
the system matches or beats the prior method on every policy, with an operator chosen per task.

**Task leaves on all four policies.** The rule of iteration 8, unchanged, on the refit libraries of pi0, SmolVLA and X-VLA
(pre-registered G19; 29 tasks, set d): against each policy's per-task LoRA with acceptance, pi0 succeeds 390 against 338 of 580
(difference 0.090, 95% interval 0.038 to 0.153), SmolVLA 284 against 275 (95% interval -0.022 to 0.059) and X-VLA 551 against
547 (95% interval -0.014 to 0.034): better on pi0 and non-inferior on the other two. With pi0.5 (994 against 932 of 1,320),
the one-demonstration side holds on all four architectures.

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

### Against standard continual learning
One rank-16 adapter trained through the same arrival order, one demonstration per task, serves every task at the end, as
these methods do (pre-registered G11). Sequential fine-tuning succeeds 52 of 600 times and experience replay 335 against 465
for the stream learner (sign tests p 5e-96 and 6e-17; task-level differences 0.688, 95% interval 0.520 to 0.832, and 0.217,
95% interval 0.077 to 0.353). Sequential fine-tuning falls below the frozen policy (273) and fails every one of the first 10
arrivals (0 successes against 28 on the last 10): the forgetting the library avoids by construction. Experience replay keeps
most of the early arrivals (138 on the first 10, 87 on the last 10) but stores every demonstration it replays.

### A second benchmark: LIBERO-PRO
LIBERO-PRO (Zhou et al., [arXiv](https://arxiv.org/abs/2510.03827)) perturbs LIBERO's own suites; its task split gives a training scene a new
instruction and success test, a held object paired with a held goal (pre-registered G22; docs/LIBERO_PRO.md). Its
benchmark reads each task's instruction from the file name, which the task split leaves unchanged, so the published protocol
hands the policy the old instruction. With the old instruction pi0.5 reproduces the published collapse (0.09, 0.00, 0.00 and
0.06 on Goal, Spatial, Object and LIBERO-10, against published 0.0 to 0.01); given the new instruction it succeeds 0.20, 0.52,
0.10 and 0.16. On the 27 of 40 tasks of the system's form (one object, one goal, one grasp) the system, unchanged, succeeds 627
of 1,350 times on LIBERO-PRO's own start states against 366 for the policy (difference 0.193, 95% interval 0.099 to 0.296)
and 243 of 540 against 159 on new start states (difference 0.156, 95% interval 0.059 to 0.261); by suite 0.31, 0.80, 0.22 and
0.17.

### Against other operators for held concepts
Two published demo-free operators, run on the same practice-chosen source pairs as ours so that only the operator varies
(pre-registered G21; set e). Mixing the policy's two velocity fields convexly at every step, after General Policy
Composition (Cao et al., ICLR 2026) with the weight chosen by practice, succeeds 61 of 400 times on pi0.5 and 45 on pi0,
against 282 and 289 for the grasp switch and 143 and 35 for the policies alone. Injecting the action expert's activations
recorded under the source tasks (after Grant et al., 2026) succeeds 151 and 9. Both fail non-inferiority to the system by
wide margins. The same mixing with its weight switched at the grasp (one field before, the other after) succeeds 136 and 148
against 138 and 143 for the grasp switch: the operator works once the fields take turns.

### Ablations
One table per side on pi0.5, every row paired with the full system on the same new start states (pre-registered G20).
LIBERO-90 (66 tasks, set d): the full system succeeds 994 of 1,320 times. Removing task leaves gives 906 (difference 0.067, 95%
interval 0.033 to 0.105); removing practice, so the composition is always served, gives 897 (0.073, 95% interval 0.033 to
0.119); replacing composition with per-task adapters gives 932 (0.047, 95% interval 0.008 to 0.091); the policy alone gives 400.
Inside the composition (897 of 1,320 always on), each unit carries part of the task: the object unit alone reaches 555 and the
goal unit alone 539; both units on for the whole episode reach 489; the switch at the grasp matters as much as either unit;
and on the instance tasks the type prototype in place of the instance child reaches 21 of 80 against 77. Li's OOD suites (set
e): the full system succeeds 312 of 400 times. Word-matched sources in place of practice-chosen ones give 160 (difference 0.380,
95% interval 0.223 to 0.545), the largest single loss; Li's fixed step in place of the grasp gives 259 (0.133, 95% interval
0.050 to 0.233); one operator in place of the choice gives 282 (grasp switch) and 249 (TLI); the object source alone gives 204
and the goal source alone 231. Every component contributes, and the sources contribute most.

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

## 7 What composition is
(docs/INTERPRETATION.md; Figures: h1_factor_ood.png, h1_donate.png, h1_tli_layers.png, h1_switch.png.)

**Composed success is a product of two competences.** The grasp is the only interface between a composed task's two phases.
The approach unit fixes which object the hand takes: arms that share an object source and differ in goal source take the same
first object in 1,803 of 1,805 trial groups on pi0.5. A wrong first grasp ends the task. Composed success is then
g x p, the chance that the approach unit grasps the named object times the chance that the carry unit places it. On 53
held-out LIBERO-90 tasks whose units were learned apart, the product predicts 366 successes against 365 observed. On Li's
suites, estimated from five practice episodes per source pair on one start-state set, the product predicts the grasp switch
on start states practice never saw with Pearson r 0.92 (pi0.5), 0.94 (pi0), 0.91 (SmolVLA) and 0.97 (X-VLA). Units learned
together beat the product: on pi0.5's library the hand-off at the grasp adds 0.117 to success after a named grasp, so the
product is a lower bound. Composition of held concepts is therefore predictable from its parts, and practice that estimates
the parts estimates the whole.

**Concepts compose in time, not in superposition.** Every operator that runs two concepts at once fails: both prototypes on
for the whole episode (489 against 897 of 1,320 when switched), the policy's two fields mixed at every step (61 against 282 of
400 on pi0.5). The same pieces switched at the grasp succeed. The grasp is the interface between a task's concepts, and
composition is the choice of which concept acts on each side of it.

**A prototype corrects where the hand goes, not how it turns.** On a held-out demonstration of each concept, an object
prototype lowers the position error of the policy's chunk from 1.205 to 0.897 and raises the alignment of the motion with the
target from 0.66 to 0.76; a goal prototype lowers position error from 2.00 to 1.24 and release-timing error from 4.3 to 3.3
steps, and raises alignment from 0.22 to 0.53. Orientation barely moves. Units carry places and moments, and the policy
supplies the rest of the motion.

**Why an operator works on one architecture and fails on another.** Text-latent interpolation adds a shift to the residual of
the encoder that reads the instruction. On pi0 and pi0.5 the shift is about a quarter of that residual (0.28 and 0.23); on
X-VLA and SmolVLA it is as large as the residual (1.01 and 1.14) and replaces the instruction's representation, which matches
interpolation's failure on SmolVLA (0 of 400). The grasp switch reads the policy's field under the instruction that taught
each phase and reproduces that phase's trained motion on all four architectures. Practice choosing the operator per task
is what makes the system architecture-agnostic.

## 8 What composition costs
(docs/COST.md; Figures: h2_pareto_libero90.png, h2_pareto_ood.png.) On pi0.5 and LIBERO-90 the system with task leaves stores
10.2 MB per task against 13.6 MB for per-task adapters and succeeds 994 against 932 of 1,320 times. It pays in fitting time:
fitting costs 31.8 A40-minutes per task on average, against 3.5 for a per-task adapter. A unit
served as an unmerged LoRA raises the latency of an action chunk from 292 to 437 ms; the grasp switch and interpolation add no
measurable latency (291 ms). On Li's suites a held concept costs at most 0.3 MB per task and the grasp switch nothing; the
system spends 340 practice episodes per task where Li's methods spend none, and the gated stream of iteration 14 lowers that.
Composition buys success and storage, and pays in fitting time and simulator practice.

**How space and time scale** (docs/SCALING.md; Figure: h3_space.png). A per-task method stores one unit per task; the
library stores one per concept. On LIBERO-90 the number of distinct concepts grows as N^0.68 over 66 tasks, well above the
ideal N^0.5 of a full object-by-goal grid: the tasks cover a small part of that grid, and each needs two or three units. The
library without leaves is smaller than a rank-16 adapter per task from the first task, and draws level with rank-4 adapters
at about 67 tasks (an extrapolation just past the 66 measured); with rank-16 task leaves it stays above rank-4 adapters. On
Li's suites a held concept costs nothing and the latents are capped by the training set. In fitting time the library loses:
a prototype is fitted once per concept but costs about ten per-task adapters, so the library becomes cheaper than rank-16
adapters only past about 20 thousand tasks under the fitted growth. Storage favours composition as concepts recur; fitting time does
not, unless prototypes fit faster.

## 9 What does not work
- Selecting prototypes by a held-out demonstration's likelihood (as Wang et al. do): the right object concept on top for 12 of
  29 queries.
- A new object without a demonstration: borrowed prototypes help on 2 of the 17 tasks the policy fails.
- Re-checking every earlier arrival after each refit (W2): no gain over W1.
- Prototypes of concepts the policy already holds (Iteration 5's weakness) and noise-space refinement on top of the
  composition (DSRL on the library).

## 10 A data defect
The IPEC port of LIBERO-90 merges tasks that share an instruction: 16 tasks were given another scene's demonstration, including
the per-task baselines. Fixed by matching each scene's block of episodes to the task's rendered first frames.

## 11 Limitations
Flow-matching policies only (pi0.5, pi0, SmolVLA, X-VLA); routed adapter libraries (CLARE) are not rerun here; choosing a held concept's source by practice costs 160 practice episodes per task; single-step pick-and-place; LIBERO-90 and Li's
OOD suites only; practice needs a simulator; a new object needs its one demonstration; non-inferiority to per-task adapters without task leaves not
established at 0.05.
