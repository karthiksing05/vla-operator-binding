# Probe D4: binding operators extracted from a frozen flow VLA (written 2026-10-25, before any D4 statistic)

**User direction (2026-10-25).** Prototypes come from the INTERNALS of an off-the-shelf post-trained flow VLA, with no new demos.
Language (the parse) supplies the task structure. The prototypes that matter are HOW language + vision BIND TO ACTION: binding
operators per role, extracted from the model and stored hierarchically (role > object category > instance). Composition = the parse
picks the role operators, and the vision codes fill them.

**What D4 measures.** A model has bound object Z to a role if its action FOLLOWS Z. At an observation's full simulator state, one
body Z is displaced by a small delta. Both cameras are re-rendered and the frozen policy is queried with the SAME noise. The
binding operator of Z is the finite-difference response J_Z = d endpoint / d pos(Z). The bound body is the one with the largest
response.

Code: `experiments/d4_bind.py` (stages) and `experiments/d4_read.py` (statistics). Pre-registration: `experiments/prereg/pd4_bind.py`
(ledger `m3_prereg`, tag `pd4_bind`). Read row: `d4_read` tag `read`. Payloads: `results/d4/` (read.json, fresh_check.json and the
two training indexes are tracked; renders, chunks, rollouts and dev observations stay local).

Models: pi05_l90c@5000 (primary) and the public pi05_libero (training observations only, a cross-model check). Both are pi0.5
checkpoints with absolute OSC deltas. A chunk is integrated from the hand with each model's demo-calibrated scale
(of_frame.scale_for, OF's).

## 1. Observations

**(a) Training tasks** (state and scoring only; demos are never executed or replayed). The tasks are OF's D1 tasks for each model
(results/of/index_<m>.json: a pick-and-place first step on a movable object).
- Approach role (pre-grasp): OF's selected frames (hand >= 5 cm xy from the named object), at most 6 per task, evenly spaced.
- Place role (carry): candidate frames every 3rd frame from the demo's first close + 5 while the gripper command stays closed.
  A frame is kept if the named object is lifted >= 3 cm above its first-frame height, lies within 10 cm of the hand, and the hand
  is >= 6 cm (xy) from the target (R2a's post-grasp clearance). At most 4 per task (the earliest).
- The role filler is the named object (approach) or the target body (place). The target body is the body carrying the goal atom's
  second argument: the longest body name it equals or starts with (`basket_1_contain_region` -> basket_1). Targets that are
  table regions have no body, so those tasks have no place observations (libero_90:34, :37, :69).
- The demo state (results/protos/match, results/robot/p0_demo; the full MjSimState) is set in a fresh LIBERO env and rendered.

**(b) zs dev rollouts recording the full sim state.** R2a's per-step recordings were deleted, so new zs rollouts are needed.
- Tasks (18): the 11 recombination dev tasks with an In / On goal (libero_90:14, 16, 60, 1, 29, 50, 9, 36, 40, 59, 61). The 4
  articulation-only recombination tasks (0, 22, 7, 11) have no object or goal role.
- **Deviation from the brief:** R2a's zs made NO wrong-object error on the 15 recombination tasks (0 / 300; its 56 wrong-object
  episodes all come from part tasks). H3's wrong-object arm is therefore run on the 7 part dev tasks where R2a saw wrong-object
  errors (libero_90:58, 73, 77, 79, 80, 83, 38). The selection used R2a's episode types only.
- Start states: a NEW set `results/robot/fresh_d4`, LIBERO's sampler at BASE 8500000 + k, 20 per task. No logged base is within
  1000 of it (logs list 100000 ... 7100000). `--fresh-check` asserts every state is > 1e-6 from every earlier libero_90 fresh set.
- zs as M1 / R2a served it: MODE ours, plain Euler 10 steps, noise seed 13, replan 5, the bound-object tracker. The client records
  the full MjSimState at every step (an additive, opt-in `REC_SIM=1` in experiments/libero_client.py) plus R2a's images, states,
  actions and object poses.
- Episodes are typed by fm_types' typer on these states. Observation points follow R2a's rules unchanged: pre-grasp = the earliest,
  middle and latest query before the first bind with the gripper open and the hand >= 5 cm from every movable object; post-grasp =
  the first query with the named object bound, lifted 3 cm and the hand >= 6 cm from the own target.
- Gotcha check: the base render at a recorded sim state must reproduce the recorded policy image (mean abs difference and exact
  share reported), and the 8-D state must match.

## 2. Counterfactuals and their validity (pre-declared)

- **Bodies.** Every movable object and every fixture except tables and the floor. In the place role, the held object is excluded.
- **Perturbations.** One body at a time, delta = +-2 cm in x and in y (4 per body). A movable object moves through its free joint.
  A fixture moves through its model body position, and its child bodies (drawers, sites) move with it. No yaw step: OF D1 found the
  yaw adds nothing, and four counterfactuals per body keep the cost down (disclosed).
- **Validity (both must hold, otherwise the perturbation is rejected):**
  1. no new collision: the deepest contact penetration of the body's geoms may grow by at most 2 mm over the base state;
  2. still supported, nothing pushed: from the base and the perturbed state, 0.1 s of physics (50 steps) runs with zero velocities
     and the robot's joints pinned. The body must end within 5 mm of (its base end + delta), and every other movable body within
     5 mm of its base end.
- A body is scored at an observation only if all 4 of its perturbations are valid. The rejection rate is reported by reason, per set.
- The rendered (unsettled) perturbed state is what the policy sees. Both cameras are rendered at 256 and resized with padding to
  224, as the rollout client does. The 8-D state is unchanged (the robot does not move).

## 3. Readout

- Policy queries: every render (base and counterfactuals) is sampled with the SAME 8 noises (set A, seed 11) through the policy's
  own Euler sampler. The base is also sampled with a second noise set (B, seed 21), the noise-floor pair. The batched path is
  checked against the single-observation sampler (`PolicyBackbone.sample`).
- Endpoint: e = h + scale * sum over the 10 chunk steps of the xyz action, averaged over the 8 noises.
- **Binding operator:** J_Z (3 x 2) = [(e(+x) - e(-x)) / 2 delta, (e(+y) - e(-y)) / 2 delta]. The xy block (2 x 2) is primary.
  Offset b_Z = (e0 - h) - J_Z (p_Z - h) in xy (OF D1's aim term). Response norm = ||J_Z,xy||_F.
- **Bound body** = the scored body with the largest response norm.
- **Noise floor** = ||e_B - e_A|| (xy, base) / (sqrt(2) delta), the norm that a pure noise change would read as. The share of
  observations whose bound body's norm exceeds it is reported.

## 4. Hypotheses (Holm over H1, H2, H3a, H3b at alpha 0.05)

The unit is the observation, clustered by task (two-level task-cluster bootstrap, R2a's scheme, B 10,000, seed 0); on dev it is the
episode (mean over its observations). pi05_l90c is the tested model. pi05_libero's H1 / H2 are reported with the same criteria as a
descriptive cross-model check.

- **H1 readout validity** (training observations, pi05_l90c). For EACH role (approach: the named object; place: the target body),
  the filler is the bound body in >= 0.8 of the observations where it is scored, and the 95% CI's lower bound exceeds chance (the
  mean of 1 / scored bodies). p = the one-sided bootstrap share of (hit - chance) <= 0. The two roles are joined as an
  intersection-union test: H1's p = the larger of the two, and both point criteria must hold.
- **H2 systematicity** (training observations, pi05_l90c, each role, joined as H1). Within a role, the filler's operator is shared
  across object types.
  - Prediction: for every observation, the filler's 4 endpoint responses Delta e_k = e(p + delta_k) - e0 (xy) are predicted by
    J-bar delta_k. J-bar is the mean filler operator of the role's observations of OTHER types (leave-type-out; world-aligned axes,
    i.e. the translation-only object frame OF found sufficient).
  - R^2 = 1 - sum ||Delta e - J-bar delta||^2 / sum ||Delta e||^2, pooled over all observations (regression through the origin:
    a body that does not move the action gives Delta e = 0).
  - Control (filler-shuffled): the same pipeline, with J-bar pooled from a hash-chosen NON-filler scored body of each training
    observation, i.e. the role filled by the wrong object.
  - Pass: R^2 >= 0.5, R^2 > R^2_control, and p = the task-cluster bootstrap share of (R^2 - R^2_control) <= 0 (predictions fixed,
    B 2000).
  - Descriptive: the yaw-rotated object frame; and a type-LABEL shuffle reference (labels permuted across observations, 200
    permutations). Under systematicity the true leave-type-out R^2 should come close to that reference, because the reference keeps
    within-type information. The brief's "type-shuffled control" is implemented as the filler-shuffled control above (disclosed).
- **H3 diagnosis** (zs dev observations, pi05_l90c).
  - H3a: on wrong-object episodes (pre-grasp), the bound body is the object the episode actually grasped, not the named one.
  - H3b: on wrong-goal episodes typed `other_region` (post-grasp), the bound body is the body the object was actually placed on or
    in (the typer's `alt`, mapped to its body), not the own target.
  - Each: rate(actual) >= 0.7 (episode means), and p = the one-sided two-level bootstrap share of (actual - named) <= 0. Episodes
    whose actual body is unscored or has no body (carried_elsewhere, table regions) are excluded and counted.
- **H4 descriptive: the hierarchy.** The filler J (2 x 2 entries) is decomposed into nested sums of squares: category > type within
  category > instance (scene / instance) within type > observation within instance. Categories are a pre-declared map one level
  above the BDDL type (`d4_bind.CATEGORY`: bowls, mugs, groceries, books, cookware, ...). The shares define the store's levels.

**Also reported:**
- the per-object-type operator table: mean J, lambda (half the trace), norm, offset and hit rate;
- bytes per stored operator (J 3 x 2 + offset 2, float16: 16 bytes) and for the whole bank (one operator per role, category, type
  and instance node, plus the node keys);
- dev successes: the bound body vs the named one (the readout on zs behaviour);
- **composition (descriptive):** for every dev observation, the parse names the filler, its pose fills the bank's most specific
  operator for its type (type, else category, else role), and e-hat = h + J (p - h) + b. This is scored against the model's own
  endpoint on zs successes, and by where it points (R2a's readout) on wrong-object / wrong-goal episodes. The pose is the ORACLE
  sim pose (D3's vision anchor would supply it; not run here).

No guidance and no closed loop in this probe.

## 5. Cost (planned)

- Renders: about 1 to 1.5 s per observation (32 to 40 counterfactuals), EGL.
- Policy: about 40 renders x 8 noises per observation.
- Training: about 318 + 220 (l90c) and 195 + 136 (libero) observations. Dev: 360 episodes (18 tasks x 20, 5 packs of 4 instances),
  up to about 4 observations each.
