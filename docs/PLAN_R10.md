# Probe R10: a gated re-conditioning retry, feasibility from the training lexicon (2026-10-26; EXPLORATORY; stopped at step 1)

**Proposed (coordinator, while the user sleeps).**
- When a deployable gate fires after a wrong release, switch the policy to a TRAINED pick-up sub-instruction from the parse until a
  sustained grasp, then switch back to the original instruction.
- Constraints: frozen weights, no demos, no simulator lookahead.
- Pre-declared stop rule: if no trained sub-instruction exists for :29 / :40's objects, report it and stop, because an untrained
  phrasing would be prompt engineering.

**Step 1 check.** Code: `experiments/r10_lex.py`, on CPU. The source is the 73 distinct training-cell instructions of pi05_l90c
(m1_compose_create.store_pool); split_sealed.json was never opened. Output: results/r10/lex.json.

## Result

- **Standalone pick-up instructions** ("pick up / lift / grab the <obj>" with no "and ..."): **none** of 73.
- **libero_90:40 (frying pan):** 3 training texts mention it, all placements: "put the frying pan on the stove", "... on top of
  the cabinet", "... under the cabinet shelf". There is **no pick-up form**.
- **libero_90:29 / :16 (black bowl):** 10 pick-up forms, all compound libero_spatial tasks: "pick up the black bowl <location> and
  place it on the plate". Each is another scene's whole pick-and-place task, ending on a plate, not a pick-up sub-step.

## Decision (pre-declared rule): R10 is infeasible within the constraints, and step 2 is not run

The policy was never trained on a pick-up sub-instruction for these objects.
- **:40:** nothing to switch to at all.
- **:29:** the only candidates would issue a different task (place on a plate) in a different scene. That is the prompt engineering
  the rule excludes.
- **A switch would therefore need an untrained phrasing.** R8 already showed that prompt-level re-conditioning on the frozen policy
  can harm: phrase prototypes took controls from 0.19 to 0.00.

**Where the test-time route stands (R5s-R10).** No within-constraints handle recovers the recombination region errors on the
frozen pi05_l90c:
- steering: R5s, R6;
- selection at carry or pre-grasp: R7;
- phrase re-conditioning: R8;
- an ungated retry: R9, which is what the policy already does;
- a trained-sub-instruction retry: R10, for which no such instruction exists.
The options that remain need the user. They are listed in PLAN_R8 section 7.2:
- (a) a world model / simulator at decision time;
- (b) weight change;
- (c) report the region errors as the boundary of test-time composition.

**Cost.** 1 CPU job; 0 A40-h.
