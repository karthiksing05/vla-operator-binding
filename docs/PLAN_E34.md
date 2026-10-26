# PLAN_BIND E3 and E4: frames and binding resolution (pre-registered 2026-10-26, before either read)

From docs/PLAN_BIND.md section 2, on pi05_l90c@5000 (frozen). Ledger: m3_prereg tags `pbind_e3` and `pbind_e4`; read rows `bind_e3` /
`bind_e4` tag read. split_sealed.json is never opened. Demos are never executed: training states are set in the simulator.

## 1. E4, binding resolution (CPU; experiments/bind_e4.py)

**Prediction (b).** At matched scene familiarity, binding error rises from type to instance to relational sub-region.

**Data.** E2b's per-task binding measures (results/bind/e2b.json; already read and reported). E4 is therefore a pre-declared
re-analysis: the coding and the model are fixed here, before any fit.

**Depth coding** (mechanical from the parse and the scene). CAT is a hand-written category table: sauces, groceries, mugs, bowls,
books, cookware.
- **Object role:**
  - 1 (instance) if the named object's type has >= 2 instances among the movable objects (a qualifier picks one), or another movable
    object is in its CAT category (an attribute picks one: :58 ketchup vs tomato sauce);
  - else 0.
- **Goal role:**
  - 2 (relational sub-region) if the target is a table region defined relative to another body ("right of the caddy"), or a region
    of a body with >= 2 goal regions among the scene's candidates (cabinet top side / drawers, shelf levels, caddy compartments);
  - 1 if the target is an object with >= 2 same-type instances (:16 "the bowl in the middle");
  - else 0.

The dry run printed the codes only.
- Object role: depth 1 on :14, :16, :60, :50, :59, :58, :38; depth 0 elsewhere.
- Goal role: depth 2 on :1, :29, :32, :40, :73, :75, :76 (after the relational rule), :77-:82; depth 1 on :16; depth 0 elsewhere.

**Errors.**
- Object role: 1 - P_obj, over 40 runs.
- Goal role: 1 - P_goal, over the runs that bind the named object. The goal role is excluded where fewer than 5 runs do (:38, :58,
  :83).
- Scene familiarity: fm axes `scene_familiar`, i.e. the scene has a training cell.

**Model (PRIMARY).** A binomial GLM, logit(err) = b0 + b_depth x depth + b_fam x familiar + b_role x [goal], fitted by IRLS. A
task-cluster bootstrap (B 2000, seed 0) resamples both roles of a task together.

**Falsified if** b_depth <= 0, or its 95% CI's lower bound <= 0.

**Descriptive.** Mean error per depth level (is it monotone?), within familiar and new scenes.

**Power (stated honestly).**
- About 45 task-role rows on 24 tasks.
- Depth 2 is confounded with the caddy STUDY scenes: all new, and all goal role. The depth effect is identified mainly from the
  familiar tasks (:1, :29, :32, :40 vs the depth 0 / 1 ones), with the familiarity covariate absorbing the new scenes.
- Instance-level object errors rest on :38 and :58. A null result is weak evidence.

## 2. E3, frames (about 0.5-1 A40-h; experiments/bind_e3.py, bind_e3_serve.py, bind_e3_pack.sbatch)

**Predictions.**
- (P1) Without injection, approach endpoints follow the object / scene frame, not the world frame.
- (P2) With 2603.19233-style injection of action-expert layer-16/17 activations from a source task, the endpoint locks to the
  source's program, and the target object no longer moves it.
- (P3) The approach gain holds at 5 and 10 cm, so it is not +-2 cm interpolation.

**Observations.** D4's training APPROACH states (open gripper, before the grasp), one per training task: the last `pre` frame, 54
observations (results/bind/e3_plan.json). Each source observation comes from another training task in a different scene (sha1
pairing).

**Readout.** The first-chunk endpoint e = h + scale x the summed xyz deltas (D4's readout and scale), mean over K 4 shared noises.

**Conditions.** D4's validity rules apply (penetration growth <= 2 mm; pinned 50-step settle within 5 mm of the expected positions).
An invalid condition is skipped.
- **obj:** the named object moved +-x / +-y by 5 and by 10 cm.
- **rig:** every non-table body (movable objects and fixtures) translated +-x / +-y by 5 cm.
- **yaw:** every non-table body rotated +-10 deg about the bodies' xy centroid (positions and orientations).
- **inj (16-17):** the outputs of action-expert decoder layers 16 and 17 are replaced, at every denoising step, by those of the
  source observation under the same noise. Run at base and with the object moved 5 cm.
- **inj16:** layer 16 only.

**Statistics.** Two-level task-cluster bootstrap (d4_read.rate). The gain is <Delta e, d> / |d|^2 (world frame 0, object frame 1).

**Bars (set now).**
- **H1 (P1).** PASS if mean(SS_world - SS_obj) > 0 with its CI's lower bound > 0, AND the 5 cm object gain's lower bound > 0.5.
  - SS_world = |Delta e|^2 (the endpoint ignores the displacement); SS_obj = |Delta e - d|^2 (it follows it, gain 1). Pooled over
    the obj and rig conditions.
  - This is PLAN_BIND's falsifier: "world-frame fit >= object-frame fit without injection".
- **H2 (P2).** PASS if the inj16-17 gain under a 5 cm object displacement has a CI upper bound < 0.5 (locked), AND the injected
  endpoint is closer to the source program than to the target's own endpoint (mean(dist_target - dist_delta_lock) has lower bound
  > 0).
  - The source program, mapped onto the target's hand, is h_tgt + (e_src - h_src). The absolute source endpoint is descriptive.
- **H3 (P3).** PASS if the 10 cm object gain's lower bound > 0.5.

**Descriptive.**
- The rigid-translation gain.
- Yaw: the error vs the world (0), scene (the endpoint rotates with the scene) and object (the endpoint follows the object's
  displacement) predictions.
- inj16 alongside inj16-17.
- Validity counts.

**Server parity (checked at every start).** Injecting a request's own observation as the source reproduces the plain sampler.
