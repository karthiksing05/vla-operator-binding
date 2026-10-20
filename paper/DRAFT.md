# Out-of-Distribution Learning Is Composition: A Supervised and an Unsupervised Scheme on Five Frozen Flow-Matching Policies

*Working draft. Every number is read from `ledger/` and audited by `experiments/audit_numbers.py`; sections marked PENDING wait
on pre-registered reads that are running. Citations were checked against their arXiv pages on 2026-10-15.*

## Abstract

A frozen flow-matching vision-language-action policy fails most new tasks, although it can reach, grasp, carry and place. We
argue that learning such tasks out of distribution is composition: a new task is a new arrangement of parts and relations the
policy already holds or can learn from one example, and the work lies in finding those parts, storing them cheaply and
recombining them. We grade a compositional split of LIBERO by what each test task leaves unseen (a relation, a binding, a scene
or a part) and test the claim with two schemes on one framework. The supervised scheme reads a task's concepts (object,
instance, goal) from its instruction and switches them at the policy's own grasp; a concept the policy holds is a zero-byte
pointer to its own field under a training instruction, a new one is a fitted rank-4 prototype or task leaf, the policy's
likelihood decides which (AUC 1.0, a detector Wang et al. leave to future work), and practice chooses sources, operators and
what to keep. The unsupervised scheme cuts a demonstration at its contact events, names each unit by the policy's likelihood
and serves pointer programs or a soft blend of held fields. The framework needs only access to a policy's prototypes and runs
unchanged on five architectures. Where only relations or bindings are new and the scene is familiar, reuse wins on all five:
on Li's compositional OOD suites, with no demonstration of any test task, pi0 rises from 35 to 336 of 400 and pi0.5 from 143
to 312; SmolVLA rises from 5 to 96 and X-VLA from 80 to 256; GR00T N1.5 rises from 113 to 157. The system is level with or
above a prior method's text-latent interpolation with hand-chosen task pairs on every one (316 and 245 on pi0 and pi0.5; 0 and 142 on SmolVLA and X-VLA; 61 on
GR00T N1.5). No fixed operator carries across architectures, and practice picks the operator per task. Where the scene or a
part is new, nothing reused helps (on LIBERO-90, held fields composed with nothing fitted: 233 of 500 against 245 for the policy
alone) and created units do: task leaves kept by practice beat a separate adapter per task (994 against 932 of 1,320; difference
0.047, 95% interval 0.008 to 0.091) with 675.9 MB against 902.8 MB. Learned as a stream, the library requests a demonstration
only when practice fails and damages no earlier task; one adapter trained through the same stream by sequential fine-tuning or
experience replay reaches 52 or 335 of 600 against 465. The likelihood names units but does not cut them, and what a held field
donates follows behaviour, not concept names.

## 1 Introduction

- Thesis: out-of-distribution learning on a frozen policy is a composition problem. The policy already carries most of a new
  task as concepts it learned under other instructions; what it lacks is the arrangement. We show it on five architectures with
  one framework whose only architectural assumption is access to the policy's prototypes (Section 3).
- The grade of novelty decides the regime. A test task is graded by which parts and relations it leaves unseen (Section 4).
  New relations and bindings in familiar scenes are learned by reuse; new scenes and new parts need units to be created.
- Robotics is an ideal testbed for compositionality. A manipulation task is an explicit composition of parts (an object, its
  location, a goal relation and its location, a scene), each part has a measurable effect on the motion, and success is
  checked by a simulator rather than by a judge. Prior knowledge here donates expertise in unorthodox ways: a field trained to
  put one object in a basket carries the approach to another. This is the systematicity Fodor asked of minds and the
  task-to-task transfer that cognitive psychology documents in people.
- Scope: flow-matching VLAs, whose action expert integrates a learned velocity field (pi0, pi0.5, SmolVLA, X-VLA, GR00T N1.5;
  FLOWER on its single suite). Every unit of the framework is a change to that field for one phase of a step, which makes
  composition a matter of choosing, per phase, which field the policy integrates. Our companion ICLR-2027 submission finds the
  same split for image diffusion: composing the scores of single-object prompts places objects that one prompt naming them all
  does not.
- The failure mode: in familiar scenes the policy runs a trained routine on the wrong object; on the 6 tasks only a per-task
  adapter solves, it grasps the wrong object first in 114 of 120 episodes (Section 6).
- Two schemes. The supervised scheme reads concepts from the instruction and is built as one system, iterated: each iteration
  is a measured weakness of the previous version and one change motivated by it, pre-registered and tested on start states no
  earlier decision saw (Section 6). The unsupervised scheme takes its units from the demonstration alone (Section 7).
- Lineage: Wang et al. ([arXiv](https://arxiv.org/abs/2605.07078)) select and compose a frozen
  generative model's prototypes for an out-of-distribution query; their stated gap is a mechanism to detect, consolidate and
  reuse new primitives. We supply one for an embodied flow policy: likelihood detects a new concept, and practice, not
  likelihood, selects among concepts.
- Contributions (Limitations states the limits):
  1. a compositional split graded by unseen parts and relations (G0 seen, G1 relation, G2 binding, G3 scene, G4 part), and the
     finding that the grade predicts whether a task needs reuse (G1, G2) or creation (G3, G4);
  2. a compositional library for a frozen flow-matching VLA whose units are instruction concepts (object type, instance as a
     child of its type, goal), each a low-rank change to the action expert fitted on the frames of its phase pooled over every
     demonstration of the concept, switched at the policy's own grasp; skill-level adapter banks composed by a learned router
     exist (PRISM), but we know of no prior work that factors a VLA's adapters into object and goal concepts;
  3. for concepts the policy already holds, a zero-byte unit: the policy's own field under a training instruction, switched at
     the grasp. This extends Li's prompt switching (hand-paired base tasks, a fixed switching step, pi0): practice chooses the
     sources and, per task, the operator (grasp switch or Li's text-latent interpolation, ported to every policy's conditioning
     encoder) with no hand labels, level with or above Li's hand-chosen pairs on all five policies; no single operator wins on
     every architecture, so the choice by practice is what makes the system architecture-agnostic;
  4. the frozen policy's own flow-matching loss on a concept's phase frames as the decision to point or to fit, with no added
     parameters (earlier continual learners decide expansion with separate models, e.g. CLARE's feature similarity), and the
     finding that likelihood detects novelty but does not select among new concepts; a continual learner that asks for a
     demonstration only when practice fails and refits every concept on its path (cf. LOTUS, which updates and adds skills);
  5. an unsupervised scheme with no instruction grammar: units cut at contact events and named by likelihood, served as pointer
     programs or as a soft blend of held fields, with a donation analysis of what each held field contributes;
  6. the iteration record itself: each failure and the change it motivated, with negative results that scope the claim
     (query-likelihood selection among new concepts, demonstration-free transfer to new objects, prototypes of held concepts,
     likelihood-placed unit boundaries);
  7. a pre-registered, paired, audited protocol on fresh start states, and a defect found and fixed in a public LIBERO-90 port.

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
action-expert LoRA on the manipulation phase. Our units are instruction concepts or contact segments, not tasks or skills,
switched by the policy's own gripper events, not a learned router, and include zero-byte pointers to the policy's own
conditionals. Pretrained VLAs forget little under fine-tuning with replay ([arXiv](https://arxiv.org/abs/2603.03818)).
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

## 3 The framework

(docs/METHOD.md, docs/UNSUPERVISED.md; Figures: one composed step; the concept hierarchy; the stream learner.)

**Units.** A unit is a change to the policy's velocity field for one phase of a task. The two schemes differ only in where the
units come from:
- **supervised:** object, instance and goal concepts are read from the instruction; an instance is a child of its object type,
  and the object (or instance) unit acts until the policy's grasp, the goal unit after it;
- **unsupervised:** the demonstration is cut at its own contact events (gripper close and open), and each segment is a unit.

**Held or new, by likelihood.** A unit is scored by the frozen policy's own flow-matching loss on its frames (a detector of the
kind Wang et al. leave to future work). A unit the policy holds under its instruction and scene is stored as a pointer to the
policy's own field under the training instruction that best explains it (zero bytes); a new unit is fitted as a prototype, a
rank-4 LoRA on the action expert fitted by flow matching on the frames of its phase in every demonstration of the concept,
each demonstration from the task's own scene. Below the shared concepts, a task they cannot serve keeps its own
one-demonstration adapter as a task leaf.

**Composition at the grasp.** The object (or instance) unit is on until the policy's gripper stays closed for three queries; the
goal unit then acts until the release. The learner thus composes held concepts the way composable diffusion composes the
scores of single prompts, and grows the library only where the policy's likelihood says it must. A pointer program generalises
the switch: a list of training instructions, advanced online at the policy's own gripper events.

**Operators for held units.** A held unit points at a source training instruction, and three operators read the policy
through it. The grasp switch conditions the policy on the object's source instruction until the grasp and on the goal's
source instruction after it. Text-latent interpolation (Li et al.) keeps the test instruction and shifts the residual stream
of the policy's conditioning encoder, at the instruction's tokens, from the object source's latent toward the goal source's.
The soft blend mixes the policy's fields under several training instructions convexly, per segment and per action dimension.

**Practice and the stream.** Ten practice episodes keep the composition only if it beats the policy, and practice chooses each
concept's source and, per task, the operator. As a stream, an arrival requests its one demonstration only when neither the
policy nor the composition reaches 8 of 10; the demonstration refits the instance, type and goal concepts it belongs to.

**What the framework needs from a policy.** Nothing in the framework depends on one architecture. It needs three accesses to a
frozen flow-matching policy: its velocity field under a given instruction (a pointer, the grasp switch and the blend), a
low-rank change to that field (a prototype), and the residual stream of the encoder that reads the instruction, with the
position of the instruction's tokens in it (text-latent interpolation). These are the extraction of prototypes. One interface
supplies them for PaliGemma (pi0, pi0.5), SmolVLM (SmolVLA), Florence-2 (X-VLA), Eagle (GR00T N1.5) and for FLOWER; the rest of the
framework is shared code.

## 4 The compositional split

(docs/SPLIT.md; `experiments/split_grades.py`.) A LIBERO task is a composition of parts in a scene. The training set is the 40
tasks every policy we test was fine-tuned on (LIBERO-Spatial, Object, Goal and LIBERO-10), so we do not choose it. Parts are read
by rule from each task's BDDL file, with one normalisation applied to every task alike:
- **object:** the moved object's type;
- **object location:** its instance qualifier;
- **goal relation:** the placing predicate and target type;
- **goal location:** the region, when it names a location such as a drawer or a side;
- **scene:** the layout. Li's OOD suites use their base suite's table, with objects added.

Following Zekun Wang's criterion ("a set with parts and relations unseen"), every test task is graded by what it leaves unseen.
The grade measures how far prior knowledge must be bent to serve a new task.

| grade | unseen | example |
|---|---|---|
| G0 seen | nothing: the (object, relation) pair in this scene is a training task | LIBERO-90 copies of LIBERO-10 steps |
| G1 relation | the pairing only: object and goal each seen in this scene, never together | Li's "put the cream cheese on the plate" |
| G2 binding | a part never seen in this scene, though seen elsewhere | Li's "put the butter on the plate" (butter only in LIBERO-Object) |
| G3 scene | the scene; every part seen in other scenes | LIBERO-90 kitchens and living rooms not in LIBERO-10 |
| G4 part | an object, goal relation or predicate never seen at all | LIBERO-90's ketchup in the tray, the caddy, the frying pan |

Counts over the 110 evaluated tasks: G0 15; G1 9; G2 20; G3 18; G4 48. Li's 20 compositional OOD tasks are G1 or G2; the
LIBERO-90 tasks span G0 to G4. Section 9 reports every result by grade.

## 5 Protocol

Pre-registered before any episode; paired episodes (same start state and noise for every arm); fresh start states from LIBERO's
own placement sampler at unused seeds; exact sign test on discordant pairs, two-level bootstrap over tasks and episodes;
non-inferiority margin 0.05. The strong baseline is a rank-16 LoRA per task fitted on the task's own scene's demonstration and
given the same acceptance test.

## 6 The supervised scheme: iterations of one system

Each iteration names the weakness the previous version showed, the one change, and the pre-registered result.
Figure (figures/robot/iterations.png) summarises the ladder: on LIBERO-90 the system against the per-task adapter at
iterations 1-3 and 8, and on Li's suites the policy alone, word-matched, hand-chosen and practice-chosen sources.
Iterations 1-4, 6 and 8 work on LIBERO-90, where most tasks are G3 or G4; iterations 5 and 9-14 work on Li's suites (G1, G2);
iteration 7 joins the two.

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

## 7 The unsupervised scheme

(docs/UNSUPERVISED.md, docs/INTERPRETATION.md section 4.) The supervised scheme needs an instruction grammar to name a task's
concepts. The unsupervised scheme takes its units from the demonstration and the policy alone. All reads here are exploratory
except Z3 and Z5, which are pre-registered.

### Boundaries come from contact events, identities from the likelihood
We record pi0.5's flow loss under its 40 training instructions at every 2nd frame of 117 demos. **The likelihood does not find
the boundaries.** On 72 pick-and-place demos, at about 2 cuts per demo, cuts where the best-explaining instruction switches find
the grasp within 10 frames 0.36 of the time against 0.28 for random cuts (release 0.14 against 0.24). Exact 2-cut partitions of
the loss signature (0.24) and the prefix latent (0.25) do no better than random 2 cuts (0.30); the demo's arm actions, a
non-policy baseline, do (0.58). Per frame, the flow loss is dominated by the observation and the noise draw, not by the
instruction. **The likelihood does name the units** once they are cut at contact events: averaged over a contact segment, the
lowest-loss training instruction holds the segment's concept 0.43 (approach) and 0.60 (carry) of the time in familiar scenes,
against chance 0.04 and 0.22, and 0.18 and 0.31 against 0.03 and 0.17 in LIBERO-90's scenes. So boundaries come from the
demonstration's own contact events (gripper close and open; task-agnostic, no grammar), and identities from the policy's
likelihood averaged per segment. In unfamiliar scenes identification is weak: there the policy barely reads its prompt.

### Pointer programs
A pointer program is the list of training instructions that name a demonstration's segments, advanced online at the policy's
own gripper events, with nothing fitted (`lifelong_ood/policy/program.py`). The grasp switch is the case where the grammar names
the two segments. On the 23 LIBERO-90 tasks the grammar cannot express (drawers, knobs, the microwave, stacking, two-step),
programs succeed 43 of 230 times against 37 for the policy alone (exploratory pilot, LIBERO's init states 0-9). Opening the top
drawer reaches 6 and 7 against 0 and turning on the stove 9 against 1, but closing the bottom drawer falls to 0 against 10.

### The soft blend and what it donates
Per contact segment and action dimension, the blend is a convex mix of the frozen pi0.5's fields under its likelihood's top
training instructions (plus the empty prompt and, on LIBERO-90, the task's new instruction), fitted to one demo by flow
matching and scored on held-out frames, over 45 demos; nothing in the network is fitted. Shapley values over the experts give
each one's donation (exploratory; `ledger/zi_blend`).

| held-out flow loss (approach, position) | blend | best single expert | uniform mix | empty prompt | own / new instruction |
|---|---|---|---|---|---|
| familiar scenes (own = the held-out answer) | 0.0375 | 0.0412 | 0.0765 | 0.1115 | 0.0168 |
| LIBERO-90 (new = the task's instruction) | 0.1607 | 0.1854 | 0.2346 | 0.2852 | 0.2627 |

- The blend beats the best single expert on position in 0.78 (familiar) and 0.84 (LIBERO-90) of the demos, and on the gripper
  never (0.00 and 0.04): a discrete command is taken from one expert, not mixed.
- The uniform mix is worse than both: weights that depend on phase and dimension are what make held fields compose.
- In LIBERO-90's scenes the task's new instruction barely improves on the empty prompt, while the blend of other tasks'
  fields does: held fields carry most of a task even where the new sentence carries none.
- By relation to the task (LIBERO-90, approach, position), the donation shares are 0.22 for experts sharing the object, 0.18
  for those sharing the goal, 0.13 for those sharing nothing and 0.15 for the new instruction. Donation follows behaviour, not
  the concept's name, which is the case for an unsupervised scheme: the units that help are found by what they explain.

**The blend served (pre-registered Z5).** On Z3's 25 LIBERO-90 tasks, the blend fitted to each demo with the gripper taken from
the segment's best expert succeeds 253 of 500 times against 245 for pi0.5 alone (H1 fails, sign p 0.45) and 233 for the hard
pointers (H2: sign p 0.047, task-level interval -0.038 to 0.134); the fitted library reaches 406. Across 75 ablations of each
task's 3 largest offline donors, the success lost agrees only weakly with the offline Shapley donation (Spearman 0.23), and only
0.25 of them lose anything: in scenes the policy never saw, held fields are interchangeable.

### Reuse with no fine-tuning
**LIBERO-90 (pre-registered Z3).** On G8's 25 tasks whose object and goal are both held as words in the 40 training
instructions, pointers switched at the grasp, with sources chosen by the demo's flow loss and nothing fitted, succeed 233 of
500 times against 245 for pi0.5 alone (H1 fails, sign p 0.15), 406 for the fitted library and 397 for the per-task LoRA. In
scenes the policy never trained in, composing its held fields at the prompt adds nothing: there the new scene's motion must be
created.

**FLOWER, on its single suite (pre-registered Z1 and Z1b).** FLOWER's LIBERO-Goal checkpoint knows only that suite's 10 tasks;
on the 6 OOD tasks whose two concepts it holds, with nothing fitted and 20 episodes each, our rule's sources succeed 30 of 120
times against 33 for FLOWER alone and 52 for Li's pairs (Z1, both hypotheses fail). Practice-chosen sources inside its suite
reach 48 against 37 alone (sign p 0.071) and 51 for Li's pairs (lower bound -0.092): both narrowly fail (Z1b). The wine bottle
onto the stove rises from 0 to 10; onto the plate and into the bowl it stays at 0 with every source its 10 tasks offer.

**A word-level splice (exploratory pilot).** Composition can also act inside the instruction. The new instruction keeps its
words, and its object phrase's encoder residuals are moved to that phrase's mean under a training instruction that holds it,
likewise its goal phrase's, with nothing fitted. It succeeds 80 of 200 times against 67 for pi0.5 alone (practice set a, 10
episodes). It wins where the source shares the 'put the X' template (the cream cheese onto the plate 9 against 1) and loses
where it comes from LIBERO-Object's 'pick up X and place it in the basket' (the butter 7 against 10): a word's binding carries
its source sentence's placement. The practice-chosen grasp switch reaches 138 on the same states (selection-biased).

## 8 Tests of the system

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

**The OOD system on five policies.** The finished OOD system (iteration 12), unchanged, on SmolVLA and X-VLA
(pre-registered G18; Figure: four_policies.png). SmolVLA succeeds 96 of 400 times against 5 alone and 0 for Li's text-latent
interpolation with Li's pairs; Li's pairs switched at the grasp reach 93 (task-level difference 0.007, 95% interval -0.025 to
0.048), and practice never chooses interpolation there. X-VLA succeeds 256 of 400 times against 80 alone, 142 for Li's
interpolation (difference 0.285, 95% interval 0.130 to 0.450) and 210 for Li's pairs at the grasp (difference 0.115, 95%
interval 0.010 to 0.250); practice chooses interpolation on 4 tasks. On GR00T N1.5, a fifth architecture trained on the same
40 tasks (pre-registered G23, set e), the system succeeds 157 of 400 times against 113 for GR00T alone (H1 passes) and 61 for
Li's interpolation (H2 passes, difference 0.240, 95% interval 0.077 to 0.403); Li's pairs at the grasp reach 108; practice
picks interpolation on 6 tasks. With pi0 (336 against 316) and pi0.5 (312 against 245), the system matches or beats the prior
method on every policy, with an operator chosen per task.

**Task leaves on all four policies.** The rule of iteration 8, unchanged, on the refit libraries of pi0, SmolVLA and X-VLA
(pre-registered G19; 29 tasks, set d): against each policy's per-task LoRA with acceptance, pi0 succeeds 390 against 338 of 580
(difference 0.090, 95% interval 0.038 to 0.153), SmolVLA 284 against 275 (95% interval -0.022 to 0.059) and X-VLA 551 against
547 (95% interval -0.014 to 0.034): better on pi0 and non-inferior on the other two. With pi0.5 (994 against 932 of 1,320),
the one-demonstration side holds on all four architectures it was run on.

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

## 9 Results by grade

(docs/SPLIT.md; successes summed per grade from the per-task reads, `results/robot/split_grades.json`.) Every result above,
stratified by what its tasks leave unseen. The reuse columns fit nothing in the network; the create column fits units on one
demonstration per new concept.

| result | grade | tasks | the policy alone | composition, zero fit | fitted units |
|---|---|---|---|---|---|
| Li OOD, pi0.5 (G15) | G1 / G2 | 6 / 14 | 59 / 84 | 116 (chosen) / 312 over both | — |
| Li OOD, pi0 (G15) | G1 / G2 | 6 / 14 | 11 / 24 | 101 / 235 | — |
| Li OOD, SmolVLA / X-VLA (G18) | G1 | 6 | 4 / 43 | 26 / 97 | — |
| | G2 | 14 | 1 / 37 | 70 / 159 | — |
| Li OOD, GR00T N1.5 (G23) | G1 + G2 | 20 | 113 | 157 | — |
| Li OOD, FLOWER-goal (Z1b) | G1 | 6 | 37 | 48 | — |
| LIBERO-90, pi0.5 (Z3, G8; set d) | G0 | 10 | 173 | below the policy (pointers) | 161 library, 151 LoRA |
| | G2 | 5 | 0 | 0 | 56 library, 70 LoRA |
| | G3 | 8 | 68 | 67 | 154 library, 136 LoRA |
| | G4 | 2 | 4 | 0 | 35 library, 40 LoRA |
| LIBERO-90, pi0.5 (G8, all 66) | G3 | 15 | 90 | — | 225 leaves, 215 library |
| | G4 | 31 | 117 | — | 455 leaves |

When only relations or bindings are new (G1, G2) and the scene is familiar, composing held units, with nothing fitted, recovers
most of the behaviour, on all five flow-matching architectures. When the scene or a part is new (G3, G4), composing held units
does nothing; a unit must be created, and created units compose (the library and leaves beat the per-task LoRA). LIBERO-90's G2
tasks need creation too: there the unseen binding is to a scene the policy knows only from LIBERO-10's two-step tasks. The grade
therefore predicts which regime a task needs: reuse (the supervised pointers, or the unsupervised segment-named programs) or
creation (fitted prototypes and leaves).

## 10 What composition is
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

**Composition is graded for motion and categorical for decisions.** Within a phase, a soft blend of held fields beats the best
single field on position in 0.78 of familiar demos and 0.84 of LIBERO-90 demos, and on the gripper almost never (0.00 and
0.04). Motion can be mixed; a decision such as closing the hand is taken from one field. The fit-time result says the same
(Section 11): a unit must commit to the demonstration's mode.

**A prototype corrects where the hand goes, not how it turns.** On a held-out demonstration of each concept, an object
prototype lowers the position error of the policy's chunk from 1.205 to 0.897 and raises the alignment of the motion with the
target from 0.66 to 0.76; a goal prototype lowers position error from 2.00 to 1.24 and release-timing error from 4.3 to 3.3
steps, and raises alignment from 0.22 to 0.53. Orientation barely moves. Units carry places and moments, and the policy
supplies the rest of the motion.

**Donation follows behaviour, not concept names.** In the blend, fields that share the task's object donate 0.22 of the
improvement, fields that share its goal 0.18, and fields that share nothing 0.13. The policy's prior knowledge reaches a new
task through behavioural similarity more than through the linguistic identity of a concept.

**In new scenes, held fields are interchangeable.** Served on LIBERO-90, removing a task's largest donors loses success in only
0.25 of the ablations, and the loss agrees weakly with the offline donation (Spearman 0.23). Where the policy never saw the
scene, which held field is composed matters little, because none carries the scene's motion.

**Why an operator works on one architecture and fails on another.** Text-latent interpolation adds a shift to the residual of
the encoder that reads the instruction. On pi0 and pi0.5 the shift is about a quarter of that residual (0.28 and 0.23); on
X-VLA and SmolVLA it is as large as the residual (1.01 and 1.14) and replaces the instruction's representation, which matches
interpolation's failure on SmolVLA (0 of 400). The grasp switch reads the policy's field under the instruction that taught
each phase and reproduces that phase's trained motion on every architecture we test. Practice choosing the operator per task
is what makes the system architecture-agnostic.

**The grade predicts reuse or creation.** A new relation or binding in a familiar scene is a recombination of held fields, and
reuse serves it on all five policies. A new scene or a new part has no held field that carries its motion, and only a created
unit serves it (Section 9).

**Composition as transfer.** Systematicity, in Fodor's sense, asks that a learner able to perform one arrangement of parts can
perform the others. Cognitive psychology documents the same in people as transfer from task to task. A frozen policy shows a
measurable version of both: its prior knowledge donates expertise to tasks it was never trained for, often through fields whose
names have nothing to do with the new task, and the reach of that donation ends where the parts or the scene are new. There,
composition has to create.

## 11 What composition costs
(docs/COST.md, docs/FIT_TIME.md; Figures: h2_pareto_libero90.png, h2_pareto_ood.png.) On pi0.5 and LIBERO-90 the system with
task leaves stores 10.2 MB per task against 13.6 MB for per-task adapters and succeeds 994 against 932 of 1,320 times. It pays in
fitting time: fitting costs 31.8 A40-minutes per task on average, against 3.5 for a per-task adapter. A unit
served as an unmerged LoRA raises the latency of an action chunk from 292 to 437 ms; the grasp switch and interpolation add no
measurable latency (291 ms). On Li's suites a held concept costs at most 0.3 MB per task and the grasp switch nothing; the
system spends 340 practice episodes per task where Li's methods spend none, and the gated stream of iteration 14 lowers that.
Composition buys success and storage, and pays in fitting time and simulator practice.

**Fit time (pre-registered F1).** 97% of a unit's 39.4-minute fit recomputed the frozen VLM prefix, which never changes because
the units live in the action expert. Cached once, the same 1,000-step fit takes 4.6 minutes with equal losses: an 8.6-fold saving with nothing traded.
Shorter fits are not free. On held-out flow loss a unit is best near 200 steps, 47% below its 1,000-step value, but the library
refit at 200 steps fails non-inferiority on rollouts: 388 against 480 of 580 (difference -0.159, 95% interval -0.255 to -0.072).
Held-out likelihood rewards a field that hedges, and success needs units that commit. Rank-4 task leaves pass: the rule of
iteration 8 with them succeeds 980 times against 994 of 1,320 with rank-16 leaves (lower bound -0.039), at a quarter of the
bytes.

**How space and time scale** (docs/SCALING.md; Figure: h3_space.png). A per-task method stores one unit per task; the
library stores one per concept. On LIBERO-90 the number of distinct concepts grows as N^0.68 over 66 tasks, well above the
ideal N^0.5 of a full object-by-goal grid: the tasks cover a small part of that grid, and each needs two or three units. The
library without leaves is smaller than a rank-16 adapter per task from the first task, and draws level with rank-4 adapters
at about 67 tasks (an extrapolation just past the 66 measured); with rank-16 task leaves it stays above rank-4 adapters, and with rank-4 leaves, which pass non-inferiority above, it draws level with them at about 590 tasks (an extrapolation of the measured storage slopes). On
Li's suites a held concept costs nothing and the latents are capped by the training set. In fitting time the library loses:
a prototype is fitted once per concept but costs about ten per-task adapters, so the library becomes cheaper than rank-16
adapters only past about 20 thousand tasks under the fitted growth. Storage favours composition as concepts recur. Fitting time favoured per-task adapters, and caching the frozen prefix closes most of
that gap: a prototype then fits in 4.6 minutes, close to a per-task adapter's 3.5.

## 12 What does not work
- Selecting prototypes by a held-out demonstration's likelihood (as Wang et al. do): the right object concept on top for 12 of
  29 queries.
- Placing unit boundaries by the policy's likelihood: no better than random cuts (Section 7).
- A new object without a demonstration: borrowed prototypes help on 2 of the 17 tasks the policy fails.
- Re-checking every earlier arrival after each refit (W2): no gain over W1.
- Prototypes of concepts the policy already holds (Iteration 5's weakness) and noise-space refinement on top of the
  composition (DSRL on the library).
- Composing held fields at the prompt in scenes the policy never trained in (Z3, Z5): no gain over the policy alone.

## 13 A data defect
The IPEC port of LIBERO-90 merges tasks that share an instruction: 16 tasks were given another scene's demonstration, including
the per-task baselines. Fixed by matching each scene's block of episodes to the task's rendered first frames.

## 14 Limitations
Flow-matching policies only (pi0.5, pi0, SmolVLA, X-VLA, GR00T N1.5; FLOWER on its single suite, LIBERO-Goal, where both
pre-registered tests narrowly fail); routed adapter libraries (CLARE) are not rerun here; choosing a held concept's source by
practice costs 160 practice episodes per task; LIBERO-90, Li's OOD suites and LIBERO-PRO only; practice needs a simulator; a new
object needs its one demonstration; non-inferiority to per-task adapters without task leaves not established at 0.05. The
supervised scheme assumes a template: single-step pick-and-place with one object, one goal and one grasp, whose concepts the
instruction names. The unsupervised scheme drops the template, but its identification is weak in unseen scenes (0.18 on the
approach in LIBERO-90), and its reads there are exploratory pilots or pre-registered tests that fail against the policy alone.
