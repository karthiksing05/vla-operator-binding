# Probe R8 (DRAFT, not pre-registered, not run): test-time re-conditioning of the region word on the frozen policy (written 2026-10-26)

**Status: DRAFT for the user's approval.** No code, no ledger row, no rollout. Every number below is a cited past result.

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
