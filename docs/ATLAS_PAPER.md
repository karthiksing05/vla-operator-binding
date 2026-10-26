# Where Compositional Zero-Shot Breaks in a Flow VLA: A Failure Atlas (draft, 2026-10-25)

Every number in this draft is ledger-backed in the commit or plan document cited. The sealed test third of LIBERO-90-C is
untouched; all reads are on the development split.

## Abstract

Vision-language-action (VLA) policies solve the tasks they were post-trained on, yet fail on new combinations of familiar
parts. We build a controlled testbed: a pi0.5 flow-matching policy trained only on the training cells of a
maximum-compound-divergence split of LIBERO-90. On held-out recombinations it succeeds in 180 of 300 episodes, and it never
succeeds on tasks with a never-seen part. We then map why it fails and what does and does not repair it.

1. **The failures are binding errors.** 215 of 339 typed failures grasp the wrong object or place on the wrong goal. The rate
   is flat in compound divergence and worst in new scenes.
2. **The binding error is measurable at the mechanism.** Displacing an object in a rendered counterfactual shows which object
   the action follows. On trained tasks it follows the named object in 0.86 of observations (chance 0.20). On wrong-object
   failures it follows the grasped object (0.69), not the named one (0.27).
3. **The break is the word-to-referent link.** The vision tower still locates the named object on failures (0.62 vs 0.68 on
   successes). The object word's attention does not (0.22-0.46 vs 0.65-0.89).
4. **The approach binding is systematic; the goal binding is weak.** One near-isotropic operator, "move toward the bound
   object", serves all 19 object types: holding a type out costs nothing. The goal role has no comparable operator at our
   readout.
5. **Eleven demo-free, weight-free repairs fail, each for a measured reason.** These include language guidance (CAG-VA),
   selection among the policy's own modes, part tokens extracted from training demos, attention steering, internal-state
   splicing, and guided sampling. One demonstration of the target task does repair it (234 of 300, at 74 KB).

The single explanation: the policy writes the instruction, bound to the scene, into a scene-wide code in its deep image
tokens. No local handle exists, and no correct state of its own exists to borrow for an unseen pair.

## 1. Introduction

The field's default remedy for an out-of-distribution task is post-training on target demonstrations. For new scenes,
objects and phrasings, broad co-training often suffices zero-shot. For new combinations of skills and objects, published
results show single-digit to low-tens success even after fine-tuning (CRAFT 2610.00524).

Compositional zero-shot is therefore the axis where no good default exists. We ask two questions:
- *how* does a flow VLA fail there;
- can a frozen policy be repaired *without* new demonstrations, by composing parts extracted from its own internals?

We answer the first in mechanistic detail. On the second we report a structured negative result, and we isolate the two
missing pieces a method would need.

**Contributions.**
1. A controlled compositional testbed (LIBERO-90-C) with a checkpoint that never saw the held-out combinations.
2. A failure taxonomy read from simulator state, plus an empty-instruction probe and a "did it do a demonstrated task
   instead" probe.
3. A causal counterfactual readout of binding (D4), and a measurement of Fodor-style systematicity *in the mechanism*.
4. A localisation of the break: vision finds the referent, the word does not bind it.
5. Eleven demo-free repairs, each tested against a pre-registered bar, with the reason each fails.

## 2. Testbed

- **Split.** LIBERO-90-C is a maximum-compound-divergence split of LIBERO-90 (Keysers et al. 1912.09713):
  - every test atom appears at least twice in training (atom divergence at most 0.02);
  - test tasks fall in compound-divergence bins 0.1 / 0.3 / 0.5;
  - a part-held-out set contains objects and predicates absent from all training;
  - a sealed third is committed before any read and is untouched here.
- **Policy.** pi05_l90c is pi0.5 trained from pi05_base on the training cells only. It reaches 0.959 parity on training
  cells, so "never seen" holds at the weights.
- **Development set.** 29 held-out tasks: 15 recombinations of held parts and 14 with a never-seen part. Each is run on 20
  fresh start states, checked to be distinct from every earlier set.
- **Failure typing** (experiments/fm_types.py) reconstructs the final state and evaluates the BDDL goal atoms. Each episode
  is typed as one of: success, wrong_object, wrong_goal, motion, no_commit, fixture.
- **Statistics.** Paired episodes, task-cluster bootstrap CIs, and Holm over at most four pre-registered hypotheses per
  probe. Every confirmatory read has a pre-registration row written before its first statistic.

## 3. How zero-shot fails

| finding | evidence | source |
|---|---|---|
| zs solves recombinations, never new parts | 180 / 300 recombination, 0 / 280 part | M1, bdb324b1 |
| failures are mostly binding errors | 215 of 339 typed failures are wrong-object or wrong-goal | FM, b7fb0644 |
| binding errors do not grow with compound divergence; they are worst in new scenes | flat across CD 0.1 / 0.3 / 0.5 | FM |
| the scene alone drives demonstrated behaviour | the empty instruction completes a same-scene training task in 0.466 [0.313, 0.626] of dev episodes, and its own task in 99 / 200 training-cell episodes | R1, fe79413a |
| failures are misbindings, not replays | only 38 / 400 zs failures complete a demonstrated task, nearly all from two tasks | R1 |
| the right object is rarely among the policy's own options | the named object is a null-field mode in 0.534 of pre-grasp observations, 0.19 on wrong-object episodes | R2a, fd474579 |

**Reading.** The policy does not replay the scene's training task, and it does not ignore the instruction. The instruction
moves it off the scene default, but onto the wrong combination.

## 4. Where the binding breaks

### 4.1 Binding is written into the image tokens early

- Restoring the instruction's word positions recovers at most 0.25 of the effect from layer 2 onward (L0b, docs/PLAN_LANG.md).
- Splicing the deep image-token K/V of a same-scene training instruction moves the action toward its object:
  - whole image: +0.44 (the full prompt gives +0.50);
  - the object's own region: -0.005.
- So the binding is a distributed, scene-wide code in layers 10-17, not a local one (R3c, docs/PLAN_R3C.md).

### 4.2 Vision finds the referent; the word does not bind it

| readout | zs successes | zs wrong-object | chance |
|---|---|---|---|
| SigLIP identity-prototype anchor inside the named object's region | 0.68 [0.49, 0.87] | 0.62 [0.10, 0.96] | 0.04 |
| object-word attention argmax inside the named region (layers 3-12) | 0.65-0.89 | 0.22-0.46 | 0.03 |

Source: OF, docs/PLAN_OF.md. The misbinding is upstream of the action but downstream of visual grounding. It sits at the
word-to-referent binding.

### 4.3 A causal readout of binding

For each recorded full simulator state, we render counterfactuals with one object displaced by ±2 cm. We then query the
frozen policy with shared noise. The response J = Δendpoint / Δposition says which object the action follows, and how
strongly. 98-99% of perturbations are physically valid (D4, docs/PLAN_D4.md).

| | result |
|---|---|
| approach role, training tasks | the named object is the bound object in 0.864 [0.804, 0.918] (chance 0.20) |
| place role, training tasks | target bound in 0.480 [0.344, 0.620] (chance 0.21) |
| zs wrong-object failures | the action follows the grasped object 0.686 vs the named 0.271 (5 tasks) |

The pre-registered gates (H1-H3) fail. The bars were 0.8 for both roles and 0.7 for the diagnosis, which the place role and
the five-task diagnosis do not meet. The approach-role readout is clear nonetheless.

### 4.4 Systematicity in the mechanism

- The approach operator is near-isotropic, with gain λ = 0.13-0.25 per 10-step chunk for every one of 19 object types (yellow
  book 0.09).
- Predicting a held-out type's operator from the role's pooled operator gives R² 0.44. That equals the type-label-shuffle
  reference (0.44) and far exceeds a filler-shuffled control (-0.01).
- So holding a type out costs nothing: the approach binding is a single role-level operator. In Fodor and Pylyshyn's sense,
  the same function applies to any filler.
- It explains 44% of the response. 88% of the operator's variance is observation-specific; category, type and instance add
  5%, 2% and 5%.
- The place role has no shared operator (R² about 0). The plate is followed (λ 0.16). Baskets, cabinets, shelves, trays and
  the microwave barely move the carry endpoint at ±2 cm.
- The whole bank is 109 operators in 7,619 bytes.

## 5. What does not repair it (no target demonstration, no weight change)

| intervention | result | why it fails | source |
|---|---|---|---|
| training-free language guidance (CAG-VA) | ω* 1.5: 173 vs 180; no ω beats zs; damage -0.06 | amplifies the wrong binding; high ω turns wrong-goal into no-commit | R1, fe79413a |
| select among null-field modes by the instruction/null ratio | loses to the policy 10:28 on wrong-object episodes | the right option is absent (coverage 0.19), ranking is fine (0.93) | R2a, fd474579 |
| recombine stored training sentences | 150 vs 180 | imports the source sentences' objects | M1, FM |
| role-token bank from training demos, oracle retrieval | 181 vs zs 170 vs ceiling 223 (closure 0.21, bar 0.5); no damage | tokens fit their own demos; held-out gain positive for 3 of 16 | R2, 4eff744a |
| same bank, contrastive minimal-pair objective | 3 of 16 tokens transfer (bar 8) | specificity only by damaging partners; the unit fails, not the loss | R2c, 5bdaaf16 |
| early attention steering to the oracle region | 0.077 -> 0.077 (bar +0.20); counterfactual +0.001 | binding already lives in the image tokens | R3b, 12a9c42e |
| region-wise K/V splice from a training instruction | -0.005 (bar +0.13) | binding is scene-wide; 0 / 163 wrong-object cases have a same-scene source | R3c, deba2362 |
| guided sampling toward a target pose | moves chunks +0.38 to +0.80, but flow-loss outliers +0.29 to +0.80 (bar 0.10) | the action expert treats an unbound target as off-distribution | R4-bind, e8a45e38 |
| the same, re-judged under a target-binding prompt | still +0.26 to +0.71 | guided chunks do not match the model's own bound approach | R4b, 4ff66022 |
| reuse role tokens from earlier tasks, no fit | 8 vs zs 13 of 260 | per-task parts are not reusable | L4, 75deff22 |
| closed-loop approach guidance (vision anchor + D4 bank; exploratory) | success 0 vs zs 2 of 140; grasp fixed on one task (1 -> 16 of 20), which then fails at the goal | goal role unbound; anchor within 5 cm only 0.41 live; even oracle poses barely help | R5s, e40daaa4 |
| closed-loop carry guidance through the D5 release operator (exploratory) | 6 wrong-goal recombination tasks: zs 0.325, goal-only 0.350, oracle 0.358 (CIs include 0) | wrong-goal errors are same-body region errors; the anchor lands within 5 cm of the region in 0.14 | R6, 3d7767fd |
| select among the frozen policy's own samples (oracle verifier ceiling) | coverage@16 from the carry 0.200 (bar 0.5; :40 0.00); from pre-grasp 0.529; from start 0.572 | the region is decided at the grasp; once carried, the error is committed | R7, 9d8b3c25, 1f106dfa |
| deployable selection at the pre-grasp query | learned-ceiling top-1 gain +0.036 [-0.086, +0.161] (bar +0.05) | the first chunk carries no detectable signal of the eventual region | R7 stage B, 96a2c7e3 |
| re-condition the region phrase (activation prototype; region-contrastive guidance) | per-seed 0.042 -> 0.002 (prototype), about -0.02 (contrastive) on 28 region-error states | a training phrase's activations carry their training scene; the contrast removes what the right region shares | R8, 7162a1f8 |
| failure-gated retry | no-op: zs keeps acting about 260 steps after a wrong release; re-grasp :29 0.11, :40 0.00 | a gate that changes nothing leaves the action distribution unchanged | R9, 353ce213 |
| gated retry with a trained pick-up sub-instruction | infeasible: 0 of 73 training instructions is a standalone pick-up | the policy has no trained sub-step to re-condition on | R10, 549c579d |

## 6. What does repair it: one demonstration

| arm | recombination / 300 | part / 280 | bytes | fit |
|---|---|---|---|---|
| zs | 180 | 0 | 0 | none |
| one-demo K/V token | 234 | 117 | 73,728 | 123 s |
| one-demo LoRA r16, 1000 steps | 238 | 149 | 18.7 MB | 145 s |

The token cuts binding errors from 215 to 34 (one demo) and to 9 (five demos). It works because it is fitted on the target
task: it supplies the scene-specific binding directly. That is demonstrated adaptation, not composition.

## 7. One explanation

The policy does not keep object and goal as separable pieces. It reads the instruction holistically against the scene and
writes the result into a scene-wide code in its deep image tokens. Each failed repair follows from this:
- **No local handle.** Attention steering and region splices are inert (R3b, R3c).
- **No correct option inside the frozen model.** The named object is rarely a mode of the policy's field, and guidance
  scales the wrong binding (R2a, R1).
- **No transferable part at the K/V level.** Tokens fitted on training demos capture those demos (R2, R2c, L4).
- **No own correct state to borrow.** Wrong-object failures occur exactly where no training cell binds that object in that
  scene (R3c).
- **What survives is the approach binding operator.** It is real, causal and shared across types (D4). It can redirect the
  grasp in closed loop (R5s, task 83), but without a goal binding and a precise instance anchor it does not lift success.
- **The goal role is bound too, at the release (D5), but the region inside a body is not.** Recombination wrong-goal errors
  are mostly the wrong region of the right body. The frozen policy's own samples contain the right region before the grasp
  (R7, 0.529) and almost never after it (0.200). Nothing visible at the decision point selects it (R7 stage B), and
  re-conditioning the region phrase lowers success (R8).

## 8. Limitations

- One architecture and one benchmark are tested in depth (pi0.5 on LIBERO-90-C). pi05_libero appears only as a cross-model
  check (OF, D4).
- The D4 readout uses ±2 cm displacements and 10-step chunk endpoints, which under-resolve large goal targets.
- The R5s closed loop is exploratory, and its wrong-object tasks contain never-seen parts, which makes success hard
  regardless of the grasp.
- Several probes stop at an offline gate. Their gates are proxies (movability, flow-loss outliers), and R4b shows those
  proxies have their own floors.

## 9. The boundary of test-time recovery on this policy

With frozen weights, no target demonstration and no simulator at decision time, no tested route recovers the recombination
region errors (R5s-R10). Three relaxations remain, and each changes the claim:
1. **A world model or simulator at decision time**, for branched verified search. The oracle ceiling is coverage@16 0.529
   from the pre-grasp query (R7).
2. **A weight change**, such as an adapter. The one-demo token and LoRA (section 6) are the demonstrated ceilings.
3. **An instance- and region-level anchor.** The SigLIP anchor finds the region (0.97 / 0.92 / 0.69, D5) but lands within
   5 cm in only 0.07-0.16, and the policy does not use an external target without breaking its own manifold (R4-bind, R4b).

## Appendix: probe index

| probe | plan document | commits |
|---|---|---|
| M1 | docs/PLAN_M1.md | bdb324b1 |
| FM | docs/PLAN_FM.md | b7fb0644 |
| L4 | docs/PLAN_L4.md | 75deff22 |
| R1 | docs/PLAN_R1.md | fe79413a |
| R2a | docs/PLAN_R2A.md | fd474579 |
| R2 | docs/PLAN_R2.md | 42822d08, 4eff744a |
| R2c | docs/PLAN_R2C.md | e40fb257, 5bdaaf16 |
| R3b | docs/PLAN_R3B.md | 68d20d3d, 12a9c42e |
| R3c | docs/PLAN_R3C.md | deba2362 |
| OF | docs/PLAN_OF.md | 6cb9ea5d ... f7f452a6 |
| R4-bind | docs/PLAN_R4.md | 43b97d14, e8a45e38 |
| R4b | docs/PLAN_R4B.md | 71dca5b0, 4ff66022 |
| D4 | docs/PLAN_D4.md | 2cd1d611, 1ec377e3 |
| R5s | docs/PLAN_R5S.md | a35fb7a9, e40daaa4 |
| D5 | docs/PLAN_D5.md | 8104015c, 08b32296 |
| R6 | docs/PLAN_R6.md | d8b7cbbe, 3d7767fd |
| R7 | docs/PLAN_R7.md | 45d52831, 9d8b3c25, b5ce0e5f, 1f106dfa, fcb15068, 96a2c7e3 |
| R8 | docs/PLAN_R8.md | 0e03e729, 7162a1f8 |
| R9 | docs/PLAN_R9.md | 353ce213 |
| R10 | docs/PLAN_R10.md | 549c579d |
