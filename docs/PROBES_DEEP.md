# What each recovery probe tested, what it rules out, and what it does not (2026-10-26)

Scope of every probe: frozen pi05_l90c@5000 (our pi0.5, trained on the LIBERO-90-C training cells), held-out dev tasks, no
target-task demonstration, no weight change. "Ceiling" = the one-demo K/V token, 234 / 300 on recombinations (zs 180).
Numbers are copied from the cited plan docs and result files; nothing here is new data.

The key point for the paper: every negative below is conditional on **the handles we allowed** (the policy's own internals, its own
samples, a SigLIP prototype anchor, the parse) and on **this one policy** (small training set, about 60 cells). None of them
licenses a blanket claim about test-time composition. Section 3 lists what each probe leaves open.

## 1. Probes, in order

### Phase 1: can the language side be repaired? (R1, R2a, R2, R2c)

**R1. Training-free language guidance (CAG-VA)** (docs/PLAN_R1.md)
- **What:** classifier-free-style guidance that amplifies the instruction against the null prompt, w swept.
- **Result:** best w 1.5 gives 173 vs zs 180. No w beats zs. Damage -0.06 on training cells.
- **Side finding:** with an EMPTY instruction the policy still completes a demonstrated same-scene task in 0.466 of episodes (the
  vision shortcut). But zs failures are rarely such replays (38 / 400): they are misbindings, not "doing the old task".
- **Rules out:** "the language signal is right but too weak". Amplifying it amplifies the wrong binding.
- **Leaves open:** guidance with a DIFFERENT contrast (not null), and guidance from an external model.

**R2a. Selecting among the frozen policy's own modes** (docs/PLAN_R2A.md)
- **What:** gradient-free modes of the null-prompt field, ranked by the instruction/null likelihood ratio.
- **Result:** the named object is a mode in 0.534 of pre-grasp observations but only 0.19 on wrong-object episodes. Ranking given
  coverage is fine (0.93); the ratio loses to the policy 10:28.
- **Rules out:** mode selection for wrong-OBJECT errors. The right option usually is not there.
- **Leaves open:** this is object-level; region errors were not yet isolated (R7 later redid this for regions with rollouts).

**R2 / R2c. A bank of role tokens fitted on training demos** (docs/PLAN_R2.md, docs/PLAN_R2C.md)
- **What:** K/V tokens per part (object, goal) fitted on training-cell demos, composed with oracle retrieval. R2c adds a
  contrastive minimal-pair objective.
- **Result:** bank 181 vs zs 170 vs ceiling 223 (closure 0.21; bar 0.5). No damage. Held-out own-part gain positive for 3 / 16
  tokens in both R2 and R2c.
- **Rules out:** K/V part tokens as transferable parts. They memorise their fit demos.
- **Leaves open:** other units (weights, adapters, vision-side codes). Fitting is not test-time anyway.

### Phase 2: where does binding live, and can we touch it locally? (R3b, R3c, OF)

**R3b. Early-layer attention steering to the named object's patches** (docs/PLAN_R3B.md)
- **Result:** inert, 0.077 -> 0.077 (bar +0.20). Even steering to a counterfactual object moves +0.001.

**R3c. Splicing image-token K/V from a same-scene training instruction** (docs/PLAN_R3C.md)
- **Result:** splicing only the object's region: -0.005. Splicing the WHOLE image's deep K/V: +0.44 (the full prompt itself gives
  +0.50). And for 0 / 163 wrong-object cases does a same-scene training instruction exist to splice from.
- **Rules out:** a local handle. Binding is a distributed scene-wide code in the deep image tokens (layers 10-17).

**OF. Object-frame prototypes** (docs/PLAN_OF.md)
- **D1:** the approach is a pose-parameterised "go to the object" skill (leave-scene-out transfer 0.62 on pi05_libero), but the
  null prompt transfers just as well, so it is vision-driven, not word-driven.
- **D2:** object-word attention misses on wrong-object episodes (0.22-0.46 vs 0.65-0.89 on successes).
- **D3:** a SigLIP prototype anchor finds the named object on failures (0.62 vs 0.68, chance 0.04).
- **Picture:** vision finds the referent, the action skill exists, the word-to-referent binding is what breaks.

### Phase 3: steer the action toward a vision-anchored target (R4, R4b, D4, R5s)

**R4-bind / R4b. Guided sampling toward a target pose** (docs/PLAN_R4.md, docs/PLAN_R4B.md)
- **What:** x0-projection guidance inside the policy's own Euler sampler.
- **Result:** all-step guidance moves chunks to the target (+0.38 to +0.80) but makes them flow-loss outliers (+0.29 to +0.80; bar
  0.10). Early-only guidance stays on-manifold but the last steps pull the chunk back (at most +0.06). R4b re-judged under a
  target-binding prompt: still outliers, with a +0.27 floor of its own.
- **Rules out:** pushing the action map in-sampler. The expert treats an unbound target as off-distribution.

**D4. Is the approach a causal, shared binding operator?** (docs/PLAN_D4.md)
- **Result:** yes for the approach. Displacing the named object moves the chunk endpoint toward it (bound 0.864, chance 0.20;
  gain 0.13-0.25 shared over 19 types). Goal role looked unbound at the first chunk (0.48). On zs wrong-object episodes the
  action follows the grasped object (0.69) rather than the named one (0.27).

**R5s. Closed-loop approach guidance** (exploratory; docs/PLAN_R5S.md)
- **Result:** 7 wrong-object part tasks: success zs 2 vs guided 0 of 140 (oracle 3). Named grasp on :83 1 -> 16 of 20, but the
  goal then fails. Vision anchor within 5 cm only 0.41 in closed loop.

### Phase 4: the goal / region role (D5, R6, R7, R8, R9, R10)

**D5. Is the goal role bound?** (docs/PLAN_D5.md)
- **Result:** yes, at the release point (1.000 of 139 vs chance 0.234; leave-type-out R^2 0.811 vs control 0.017). The first chunk
  alone cannot see it (why D4 missed it).
- **Key finding:** 28 / 42 recombination wrong-goal observations are wrong REGION of the right BODY: :29 cabinet top instead of
  the top drawer; :40 the wrong shelf level. The SigLIP anchor lands in the right region-type 0.97 / 0.92 / 0.69, but within 5 cm
  only 0.07-0.16.

**R6. Closed-loop goal guidance via the D5 operator** (exploratory; docs/PLAN_R6.md)
- **Result:** 6 wrong-goal tasks, zs 0.325, goal-only 0.350, oracle goal 0.358. All CIs include 0.
- **Why:** steering the xy heading of the chunk cannot choose drawer vs top or shelf level. Those differ in height and approach
  path, which the guidance does not control, and late Euler steps pull back. The oracle halves :16's wrong goals (14 -> 7) but they
  become motion failures.

**R7. Is the right behaviour in the policy's own samples?** (docs/PLAN_R7.md)
- **What:** 32 sandboxed rollouts per state on the 42 wrong-goal states. Coverage@K = chance that at least one of K succeeds (the
  ceiling of a perfect verifier).

| from | per-seed | cov@16 | cov@32 |
|---|---|---|---|
| carry (object held) | 0.060 | 0.200 | 0.262 |
| pre-grasp | 0.060 | 0.529 | 0.690 |
| episode start | 0.056 | 0.572 | 0.810 |

- **Stage B:** could a verifier pick the winner at pre-grasp? A logistic regression on the first chunk (10 x 7 actions, integrated
  xyz, rotation, gripper), trained on dev success labels, gains +0.036 [-0.086, +0.161] over random (bar +0.05).
- **Rules out:** selection once carried (ceiling 0.20), and one-step selection from first-chunk geometry at pre-grasp.
- **Does NOT rule out:**
  - multi-step selection (re-selecting at every approach query), which needs branched rollouts;
  - verifiers on richer features: the image, the policy's internal states, a rendered trajectory, a VLM judgement;
  - small signals (power: one binary outcome per pick at a 0.06 base rate, CI half-width about 0.12).

**R8. Re-conditioning the region phrase** (docs/PLAN_R8.md)
- **C1 (reword with the training phrase):** no-op. The dev phrases already ARE the training phrases.
- **C3 (replace the phrase's residual stream with the mean of its training activations):** 0.042 -> 0.002. The activations carry
  their training scene.
- **C2 (contrast against the same body's other regions, early steps):** about -0.02. The alternatives share the body, approach and
  grasp, so the contrast removes what the right answer also needs.
- **Rules out:** internal phrase-level edits of these two kinds.
- **Leaves open:** conditioning from OUTSIDE the policy's language channel, e.g. a goal image, a keypoint, a trace, or a
  differently-structured instruction. Also: only 2 tasks (:29, :40).

**R9. Gate + retry.** No-op: zs keeps acting about 260 steps after a wrong release anyway. Re-grasp is task-dependent (:16 about
1.0, :29 0.11, :40 0.00): on :29 the policy treats a bowl on the cabinet top as done.

**R10. Gate + a trained pick-up sub-instruction.** Infeasible: 0 of 73 training instructions is a standalone pick-up.

## 2. What we need and cannot find (inside the allowed handles)

1. **A pre-grasp signal that predicts the eventual region.** The fix is in the samples (0.53), but the decision is made over
   about 8 queries before the grasp, and nothing in the first chunk tells good from bad. We have not found a deployable predictor
   of "this approach ends in the drawer, not on top".
2. **A region-level anchor.** SigLIP prototypes find the body (cabinet, shelf) but not the sub-region (drawer vs top, shelf
   level): within 5 cm only 0.07-0.16. Region identity is relational ("top drawer of"), and type prototypes cannot express it.
3. **A way to hand the policy a target it accepts.** Guidance in the sampler is pulled back or goes off-manifold (R4, R4b, R6).
   The policy only follows a target when its own conditioning binds it (R3c: whole-prompt / whole-image changes).
4. **A scene-portable region code.** The policy's own activations for "top drawer" carry the scene they came from (R8 C3), and
   the K/V tokens carry their demos (R2). Nothing we extracted is a portable "region" unit.
5. **Control over height and approach path, not just xy heading.** Drawer vs top differ in descent and approach, which our
   guidance (xy x0-projection) cannot set (R6).

## 3. What the negatives do NOT license

- **One policy, small data.** pi05_l90c saw about 60 cells. A model with more varied region phrasing in training may bind
  regions better, and the public pi05_libero was only a cross-check (OF, D4).
- **Few region-error tasks.** The region errors come from 2-3 tasks (:29, :40, partly :16), 42 states. R8's harm read rests on 4
  control states.
- **We excluded external models by design.** No VLM verifier, keypoint model, goal-image generator or LLM decomposition was
  tried; these are exactly what most test-time robotics methods use (docs/LIT_REGION.md).
- **Selection was one-step.** Branched, per-query search was not evaluated.

So the defensible claim is narrow: on this policy, handles extracted from the policy itself (prompt amplification, its own modes,
part tokens, attention, K/V splices, in-sampler guidance, its own samples ranked by its own chunks, phrase activations) do not
recover region-binding errors. Whether external grounding recovers them is open, and the literature suggests it may.
