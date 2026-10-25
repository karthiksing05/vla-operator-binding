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

## 6. Result (2026-10-25; ledger `d4_read` tag read, results/d4/read.json; pre-registration m3_prereg `pd4_bind`, commit 2cd1d611)

**Data.**
- Training observations: pi05_l90c 474 (318 approach + 156 carry), pi05_libero 307 (195 + 112).
- zs dev rollouts on fresh_d4 (18 tasks x 20): 135 successes. Typed failures: wrong object 60 (41 other type, 19 wrong instance),
  wrong goal 95 (46 other_region, 49 carried_elsewhere), motion 57, no commit 13. Dev observations: 1,005 pre-grasp + 277 post-grasp.
- **Perturbation validity:** 0.980 of 10,420 (l90c train), 0.983 of 7,556 (libero train), 0.991 of 20,860 (dev). Rejections: by
  penetration 138 / 112 / 174, by settle 71 / 18 / 9. Bodies with all 4 perturbations valid: 0.965 / 0.962 / 0.976.
- **Gotcha check (deviation):** the base render at a recorded sim state is close to the recorded policy image but not bit-exact
  (mean abs difference at most 5.8 of 255, exact on 0 observations). The 8-D state differs by at most 0.0046. Likely cause: robosuite
  computes the observation before the recorded state is read. The policy sees the re-rendered base and counterfactuals alike.
- Batched vs single-observation sampler parity: max |diff| 1.6e-3 to 2.1e-3 action units (smoke).

### 6.1 Verdicts (Holm over H1, H2, H3a, H3b): ALL FOUR FAIL

| hypothesis | statistic (pi05_l90c) | criterion | p | verdict |
|---|---|---|---|---|
| H1 approach | filler bound 0.864 [0.804, 0.918], chance 0.200 (280 obs, 49 tasks; 38 unscored) | >= 0.8, CI lo > chance | 0.0 | passes alone |
| H1 place | 0.480 [0.344, 0.620], chance 0.208 (150 obs, 39 tasks) | >= 0.8 | 0.0 | fails |
| **H1 (both roles)** | | | 0.0 | **FAILS (place < 0.8)** |
| H2 approach | leave-type-out R^2 0.440 [0.394, 0.486] vs control -0.011 (diff +0.451 [+0.404, +0.498]) | R^2 >= 0.5 | 0.0 | fails (R^2 < 0.5) |
| H2 place | R^2 -0.035 vs control 0.006 | | 0.982 | fails |
| **H3a wrong object** | bound = grasped 0.686 [0.244, 0.942] vs named 0.271 (59 obs, 5 tasks; 22 unscored) | >= 0.7 | 0.129 | **FAILS (narrowly, wide CI)** |
| **H3b wrong goal** | bound = placed-on 0.538 vs own target 0.154 (13 obs, 2 tasks; 77 without a body) | >= 0.7 | 0.263 | **FAILS (sparse)** |

**pi05_libero (descriptive cross-model check):** H1 approach 0.606 [0.491, 0.714], place 0.593 [0.450, 0.734] (chance 0.18); H2
approach R^2 0.131 vs control -0.025, place -0.049.

### 6.2 Descriptive

- **Noise floor.** The bound body's norm exceeds the second-seed floor in 0.46 (l90c approach), 0.10 (l90c place), 0.99 / 0.95
  (libero). The l90c floor is large because the mean of 8 samples moves between noise sets. The shared-noise differences are what
  the readout uses, and the approach hit rate is 4x chance.
- **Systematicity in the approach role is real but partial.** The leave-type-out R^2 (0.440) equals the type-label-shuffle
  reference (0.442). Holding a type out costs nothing, so the approach operator is shared across all 19 object types. It explains 44%
  of the endpoint response; the rest varies observation by observation (H4).
- **Per-type approach operators (l90c, mean J, xy):** near-isotropic gains lambda = trace / 2 of 0.13-0.25 for every type (bowls
  0.16-0.21, groceries 0.15-0.25, mugs 0.13-0.17, frypan 0.13, wine bottle 0.14, yellow book 0.09). J_yy (0.10-0.34) is larger
  than J_xx (0.02-0.20) for most types; off-diagonals are at most 0.09. This is OF D1's d = lambda (c - h) + b, measured causally.
- **The place role has no shared operator.** The plate is followed (lambda 0.16, norm 0.31), but basket (0.05), cabinets (0.05),
  shelf (0.03), tray (0.06) and microwave (0.05) barely move the carried endpoint at +-2 cm. The leave-type-out R^2 is about 0. A
  container's interior is a large target, so a 2 cm shift of it need not change a 10-step carry chunk.
- **H4 (share of the J sum of squares, l90c):** approach: category 0.049, type within category 0.024, instance within type 0.047,
  observation 0.879. Place: category 0.192, type 0.003, instance 0.190, observation 0.615. libero approach: 0.071 / 0.040 / 0.067 /
  0.822. The store needs a role level and (for place) a category level. The type and instance levels add little over the
  observation-level variation.
- **Bank:** 16 bytes per operator (J 3x2 + offset 2, float16). The l90c bank has 109 nodes (2 role, 15 category, 29 type, 63
  instance): 1,744 bytes of operators + 5,875 bytes of keys = 7,619 bytes. The libero bank has 80 nodes (5,768 bytes).
- **Dev successes (zs):** the bound body is the named object in 0.846 [0.720, 0.949] (approach) and the own target in 0.629
  [0.285, 0.884] (place).
- **Composition read (oracle pose, bank operator at the most specific level):**
  - approach: on successes, e-hat lies 3.1 cm (median) from the model's own endpoint vs 3.8 cm for "stay"; it points at the named
    object in 0.381 vs the model's 0.601. On wrong-object / wrong-goal episodes, e-hat points at the named object in 0.561 vs the
    model's 0.439.
  - place: on successes, 4.2 cm vs 7.3 cm, and it points at the own target in 0.737 vs 0.812. On failures, 0.311 vs 0.167.

### 6.3 Reading

1. **The approach is a causal, object-bound, type-general operator.** Displacing the named object moves the action toward it in
   0.86 of training observations (chance 0.20). The operator is near-isotropic (gain 0.1-0.25 per 10-step chunk), shared across 19
   types (leave-type-out costs nothing), and stored in about 16 bytes. This is the binding operator the user asked for, and it lives
   at the role level.
2. **The goal role has no such operator** at this readout. Carry chunks follow the plate but not the containers or fixtures.
3. **The zs misbinding is visible in the operator.** On wrong-object episodes the action follows the grasped object (0.69) far more
   than the named one (0.27). This misses the 0.7 bar narrowly, on 5 tasks.
4. **Composition (descriptive).** The parse picks the role operator and the oracle pose fills it. On failures, this points at the
   named object more often than the model does (0.56 vs 0.44 approach; 0.31 vs 0.17 place). It is worse than the model on
   successes, and the operator explains only 44% of the response. So the bank predicts an X-bound heading for an unseen pair only
   coarsely. The model's own fine approach is not captured.

**Disclosures.**
- H3a runs on part tasks (no recombination task had wrong-object errors in R2a). That choice was pre-registered.
- The H3b exclusion count (77) mixes carried_elsewhere episodes (no alt) and alts without a body.
- The render check is not bit-exact (above).
- The read was run once, recorded. A per-type table and the above-floor shares were printed from read.json afterwards
  (descriptive).
- Sessions: the first session ended while the dev rollouts ran. The coordinator's resume found the rollouts complete. Extract,
  dev render and dev GPU were then submitted once each, and nothing was rerun.
- One python heredoc was issued on the login node during development; the harness blocked it before it ran.
