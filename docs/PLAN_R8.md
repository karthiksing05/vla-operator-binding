# Probe R8: test-time re-conditioning of the region word on the frozen policy (drafted 2026-10-26; stage A pre-registered in section 6)

**Status.** Sections 1-5 are the draft. Section 6 is the stage-A pre-registration (coordinator-approved while the user sleeps; the user may overrule it).

**User direction.** The method is test-time, on the frozen pi05_l90c@5000, with no new demos and no weight change. It uses
prototypes extracted from the model's internals (how language and vision bind to action), and the task structure comes from the
parse.

## 1. Why this route (what R7 closed)

- **Carry phase.** R7 coverage@16 from the carry state is 0.20, and 0 on :40. The policy's own samples do not contain the fix once
  the object is held. R6 found that carry guidance, oracle included, does not raise success.
- **Pre-grasp.** Coverage@16 is 0.53, so the fix IS in the distribution before the grasp, and the region is decided by the grasp
  (pre-grasp - carry +0.33). But the first chunk at the pre-grasp query carries no selectable signal: the learned ceiling gains
  +0.036, with a CI that includes 0.
- **What is left.** The policy's per-seed success on these states is 0.06. To recover the error at test time, the policy must
  produce the right grasp-and-place far more often. That means changing what it is conditioned on (the language / vision binding of
  the REGION), not selecting among its outputs.
- **Earlier evidence on where binding lives (R3c).** Splicing the WHOLE image K/V from a source prompt moves binding +0.44, and the
  full prompt +0.50. Local splices and attention steering are inert (R3b, R3c). So conditioning acts globally, through the prompt
  and the deep image code. A conditioning change is the lever the model responds to.

## 2. Candidate conditioning operators (all frozen weights, no demos, the parse as the only task input)

- **C1, a canonical region phrase.**
  - The parse gives the goal atom, e.g. In(bowl, cabinet.top_drawer) or On(pan, shelf.top_region).
  - The instruction is re-rendered with the region phrase that the TRAINING instructions use for that region predicate. The
    training lexicon is built from the training split only, never split_sealed.json. The named objects are kept.
  - Prototype: the region phrase -> region binding learned in training.
- **C2, region-contrastive prompt guidance in the flow.**
  - At each Euler step, v = v(p) + w [v(p) - v(p_alt)].
  - p_alt is the instruction with the region phrase replaced by the target body's OTHER region that zs chose: the cabinet top, or
    the shelf's other level (from the parse + the scene's region list).
  - This rebinds the region word by contrast, with no vision anchor needed.
  - Risk: R4-bind's off-manifold pull-back, so w is limited to the early steps. That is a pre-declared on-manifold check, as in R4b.
- **C3, a training-scene region prototype in the language stream.**
  - The region word's token embeddings, or their deep K/V, are replaced by a prototype: the mean K/V of that region phrase in
    training prompts where the policy succeeded at that region (D5 training observations; hit 1.0).
  - This is the "prototype extracted from the internals" version of C1.
  - R2 / R2c caution: fitted tokens did not transfer. C3 uses the model's own activations, with no fitting.
- **C4 (lower priority).** A whole-image deep K/V splice from a training scene with the same region goal. This is R3c's +0.44
  lever. It is likely confounded by scene geometry, since :29's white cabinet has no training scene.

## 3. Stage A: offline coverage test of each operator (the R7 sandbox, decisive and cheap)

- **States:** R7's 42 wrong-goal states at pre-grasp and start, plus the 34 controls.
- **Rollouts:** the same 32 seeds as R7 (paired by seed), with the operator applied on every query. Code: R7's run loop with a new
  opt-in server mode; d5_serve.py itself stays unchanged.
- **Readouts:**
  - per-seed success and coverage@16 vs R7's unconditioned numbers on the same seeds: pre-grasp per-seed 0.060, cov@16 0.529;
  - named-region placement (typer);
  - controls' per-seed success (harm).
- **Proposed bar:**
  - per-seed success on wrong-goal pre-grasp states rises by >= +0.10 (from 0.06), with the paired two-level CI excluding 0;
  - control per-seed success drops by <= 0.05.
- **Cost:** about 2 A40-h per operator for pre-grasp only with K 16 (R7 option 1: 4.0 A40-h at K 32).
- **Order:** C1 first (a prompt change only; cheapest), then C2 at 2 values of w, then C3.

## 4. Stage B / C (only if an operator passes A)

- **Stage B:** an exploratory closed-loop smoke on the R6 tasks plus harm and damage tasks, paired with zs, with an oracle-region arm.
- **Stage C:** the confirmatory 15-task read (closure >= 0.5 of the zs-to-ceiling gap) only with the user's approval.

## 5. Open questions for the user

1. Is re-rendering the instruction from the parse (C1) inside the method's scope? It changes the words the policy sees, but uses no
   new data.
2. C2 needs two policy passes per Euler step, which doubles inference cost. Is that acceptable?
3. The budget for stage A.

## 6. Stage A as pre-registered (2026-10-26, before any R8 rollout; m3_prereg `pr8_cover`)

**Coordinator's resolution of section 5 (provisional; the user can overrule it).**
- C3 is the primary method arm, then C2 at 2 values of w.
- C1 would be a DIAGNOSTIC arm only. C4 is skipped.
- Pre-grasp states only, K 16 paired with R7's seeds.
- Budget: about 8 A40-h for stage A.

**What the lexicon showed (results/r8/lex.json, training split only), and the consequences.**
- **C1 is a no-op, so it is not run.** :29's phrase "in the top drawer of the cabinet" is word for word the training phrase of
  :3 / :4 / :5 (wooden cabinet, top_region). :40's "on the cabinet shelf" is word for word :88's (the same shelf type, top_region).
  Re-rendering with the training lexicon would change nothing: the failure is not lexical.
- **The scenes' own training priors point at the wrong region.** In :29's white-cabinet scene, :25 trains "put the black bowl on top
  of the cabinet" (top_side), the region zs chooses. In :40's shelf scene, :43 trains "put the white bowl on top of the cabinet" (the
  shelf's top_side).
- **:16 is out of scope.** Its errors are instance-level (which bowl), not region-level, and its target (a bowl) has no regions. No
  operator applies, and its units are not re-rolled.
- **The spec rule (experiments/r8_ops.py) uses only the parse's goal atom, the scene's fixture types and the training lexicon.** It
  finds a region phrase verbatim in the instruction, with a same-type training entry first and else category level. It applies on
  libero_90:29 (category level: sources :3 / :4 / :5) and :40 (type level: source :88), and on no other dev task in R7's set.

**Units (results/r8/plan.json).**
- 28 wrong-goal region-error pre-grasp states (PRIMARY): :29 19, :40 9.
- 4 controls: :29 1, :40 3.
- The other 44 R7 pre-grasp units have no spec. The operator is never active there, so harm on them is 0 by construction. They are
  not re-rolled.

**Arms (experiments/r8_serve.py, a subclass of d5_serve.BatchPolicy, opt-in request fields; d5_serve.py unchanged).**
- **C3 (PRIMARY method arm): the region-phrase prototype from the model's own training activations.**
  - Before every encoder layer i >= 1 of the prefix pass, the residual stream at the phrase's tokens is REPLACED by the prototype.
  - The prototype is the mean, over D5's training carry observations of the source tasks, of the same layer's input at the same
    phrase tokens. Images and state come from D4's training renders; the training instruction is used. That is 12 observations for
    the drawer phrase (7 tokens) and 4 for the shelf phrase (4 tokens): results/r8/protos.npz.
  - The image tokens attend to the phrase in the bidirectional prefix, so the patch reaches the deep image code (R3c).
- **C2 (method arm): region-contrastive guidance.**
  - On the first 5 of 10 Euler steps, v = v(p) + w [v(p) - mean_a v(p_a)], with w in {1, 2}; afterwards the plain v(p).
  - p_a is the instruction with the phrase replaced by each OTHER region of the same body that has a training phrase:
    - :29: "in the bottom drawer of the cabinet" and "on top of the cabinet";
    - :40: "on top of the cabinet" and "under the cabinet shelf".
  - The noise is the same as the plain sampler's.
- **Server parity, checked at every server start:** C2 at w 0 equals the plain sampler, and C3 patched with the prompt's own residual
  equals the plain prefix.

**Rollouts.** R7's run loop (r7_cover.run_unit) with R7's seeds k = 0..15 plus the zs-seed run, from R7's pre-grasp states. A run ends
at LIBERO success or the episode budget.

**Readouts (per arm).**
- Per-seed success vs R7's unconditioned runs with the same seeds at the same states: the paired per-state difference, pooled with
  the two-level task-cluster bootstrap. With 2 tasks, the interval is wide.
- coverage@16, per task; the controls' per-seed difference.
- **On-manifold guard (descriptive):** the share of executed actions with any |a[0:6]| > 1 (LIBERO's action bound), arm vs R7.
  Flag if it rises by > 0.10.

**Bar (from section 3).** An arm PASSES if:
- the wrong-goal per-seed success rises by >= +0.10, with the paired CI's lower bound > 0; and
- the controls' per-seed success falls by at most 0.05 (point estimate; 4 states, so this is weak, disclosed).

**Next step.** If any method arm passes, a stage B exploratory closed-loop smoke is pre-registered (R6 tasks, harm and damage tasks,
an oracle-region arm; about 4 A40-h). Otherwise R8 stage A is recorded and handed back.

**Cost (planned).** C3 about 0.9 A40-h; C2 about 2 A40-h per w; plus prototypes and smokes. About 5-6 A40-h.

## 7. Stage A result (2026-10-26; ledger `r8_read` tag cover, results/r8/read.json; pre-registration 0e03e729)

**Data.** 3 arms x 32 units x 17 runs, all complete (6 A40 jobs x 3 pairs; 0 crashes, 0 missing). The baseline is R7's
unconditioned runs at the same states with the same seeds (k 0..15). Intervals are two-level task-cluster bootstrap 95% intervals;
with only 2 tasks they are wide.

| arm | wrong-goal per-seed (28) | paired diff | cov@16 | controls per-seed (4) | control diff | out-of-bound actions | verdict |
|---|---|---|---|---|---|---|---|
| R7 (no operator) | 0.042 | | 0.464 | 0.188 | | 0.004 / 0.002 | |
| **C3 prototype (PRIMARY)** | 0.002 | **-0.040 [-0.069, -0.021]** | 0.036 | 0.000 | -0.188 [-0.312, -0.104] | 0.002 / 0.002 | **FAIL** |
| C2 contrast, w 1 | 0.025 | -0.018 [-0.059, +0.002] | 0.286 | 0.078 | -0.109 [-0.156, -0.062] | 0.007 / 0.009 | FAIL |
| C2 contrast, w 2 | 0.022 | -0.020 [-0.076, +0.003] | 0.250 | 0.062 | -0.125 [-0.250, +0.062] | 0.018 / 0.031 | FAIL |

The out-of-bound column is wrong-goal / controls.

- **Per task (wrong-goal per-seed, R7 -> arm):**
  - :29 0.036 -> 0.000 (C3), 0.026 (C2 w 1), 0.033 (C2 w 2);
  - :40 0.056 -> 0.007 / 0.021 / 0.000.
- **No method arm passes. Every arm LOWERS success**, on the wrong-goal states and on the controls. The on-manifold guard never
  trips (out-of-bound actions stay <= 0.031), so the harm is not from out-of-range actions.

### 7.1 Reading

1. **C3 breaks the task.**
   - Replacing the phrase tokens' residual at every layer with the training mean removes almost all success: wrong-goal 0.042 ->
     0.002, controls 0.19 -> 0.00, cov@16 0.46 -> 0.04.
   - The prototype is not a portable "region" code. The phrase's contextual state at the training scenes carries those scenes, and
     forcing it into another scene derails the policy. On :29 the level is category (a wooden cabinet's drawer for a white cabinet);
     on :40 it is the same shelf type and still fails.
   - This matches R3c: binding lives in a distributed deep scene code, not in local token states, and R2 / R2c found that fitted
     part tokens do not transfer.
2. **C2 pushes away from the wrong regions but does not bind the named one.**
   - The contrast against the same body's other regions lowers success at both weights: -0.02 on wrong-goal, -0.11 / -0.13 on
     controls.
   - Subtracting the alternative-region velocity removes components the correct placement also needs (the same body, approach and
     grasp), since the alternatives share everything except the region.
3. **C1 needed no run.** The dev region phrases are already word for word the training phrases. The region word is in the
   instruction, in the training wording; the policy does not ground it in these scenes. That is a grounding failure, not a lexical one.

### 7.2 Where the recovery route stands (R5s-R8)

- **The goal role is bound** (D5: G1 1.0, G2 0.81).
- **None of these recovers the region errors:**
  - steering the carry (R6);
  - selecting among the frozen policy's own samples at carry (R7: ceiling 0.20) or at the pre-grasp query (R7 B: no signal);
  - re-conditioning the region phrase by a training prototype (C3) or by region contrast (C2) (R8).
- **The fix exists only as a rare outcome of the policy's own pre-grasp sampling** (cov@16 0.53, per-seed 0.06). No test-time
  handle tested so far raises its rate without a simulator at decision time.
- **Hence**, on the frozen pi05_l90c, test-time recovery of the recombination REGION errors is not justified by any method tested.
  The remaining options need a user decision:
  - (a) allow a world model / simulator at decision time (branched verified search, ceiling 0.53 at K 16);
  - (b) allow weight change (adapters), which leaves "test-time, frozen";
  - (c) report the region errors as the failure map's boundary of test-time composition, with R6-R8 as the evidence.

**Cost.** R8 6.4 A40-h: C3 1.0, C2 w 1 2.6, C2 w 2 2.6, plus about 0.1 for the prototypes and smokes. Also 6 short CPU jobs.

**Disclosures.**
- Stage A ran while the user slept, under the coordinator's provisional resolution of section 5.
- C1 was not run, because it is a no-op (section 6). :16 is out of scope.
- The control set is 4 states, so the harm read is weak. On the 44 units with no spec, harm is 0 by construction.
- The C3 prototypes come from carry-state training renders but are applied at pre-grasp queries.
- C3's own-residual server parity is 2-3e-3 (bf16 batch noise, as in D5), not exact. C2 at w 0 is exact.
- The read was run once with `--record`.
- Audits after the read: audit_numbers 0 discrepancies over 2,064 checks; audit_coverage 0 uncovered numerals.
