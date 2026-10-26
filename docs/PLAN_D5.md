# Probe D5: a goal-role binding operator read at the release point (written 2026-10-25, before any D5 statistic)

**User direction (2026-10-25).** "Keep iterating until the recovery method has been justified." The method works at test time on the
frozen pi05_l90c@5000, with no new demos and no weight change. It extracts prototypes from the model's internals (how language and
vision BIND to action) and takes the task structure from the parse. On the 15 recombination dev tasks, zs failures are mostly WRONG
GOAL, so the method must bind the goal role.

**Why D5.** D4 (docs/PLAN_D4.md section 6) found a causal, type-general approach operator (bound 0.864, chance 0.20) but no place
operator (bound 0.48). The plate was followed, but containers and fixtures barely moved a 10-step carry chunk at +-2 cm. A
container's interior is a large target, so a 2 cm shift need not change one chunk. D5 changes exactly two things:
1. **larger displacements**: +-5 cm (primary) and +-10 cm (secondary), with D4's validity rules unchanged;
2. **a readout that resolves a goal**: the predicted RELEASE point. From the recorded carry state, a sandboxed copy of the simulator
   is rolled forward. The frozen policy is queried every 5 steps on the renders of that counterfactual world, with shared noise, until
   it first commands the gripper open.

Code: `experiments/d5_goal.py` (stages), `experiments/d5_read.py` (statistics), `experiments/d5_serve.py` (a batched policy server, new
file), `experiments/d5_pack.sbatch` (server + sandbox client pairs on one GPU). Pre-registration: `experiments/prereg/pd5_goal.py`
(ledger `m3_prereg`, tag `pd5_goal`). Read row: `d5_read` tag `read`. Payloads: `results/d5/` (index.json, parity.json and read.json are
tracked; sb/, sb_smoke/ and logs/ stay local).

## 1. Observations

- **Training (the place role):** D4's 156 kept carry observations of pi05_l90c's training tasks (40 tasks; D4's rules: lifted >= 3 cm,
  held within 10 cm of the hand, hand >= 6 cm xy from the target, at most 4 per task). The filler is the target body (D4's
  `target_body`). The demo state is set in the simulator; the demo is never executed.
- **Dev (zs):** D4's fresh_d4 zs recordings (full MjSimState at every step). D4's post-grasp observation of each episode (R2a's rule:
  the first query with the named object bound, lifted 3 cm, the hand >= 6 cm from the own target; one per episode). Kept: every
  wrong-goal episode (90: recombination :16 14, :29 19, :40 9; part :73 6, :77 12, :79 13, :80 16, :83 1) and up to 4 zs successes
  per recombination task (sha1 order; 37). No new dev rollouts are needed: fresh_d4 covers the 11 recombination tasks with an In / On
  goal (the other 4 are articulation-only).

**What the dev wrong-goal failures are (from the typer, seen before this plan).** On :16 ("stack the front bowl on the middle bowl")
zs stacks on the plate or the other bowl: a different BODY. On :29 ("in the top drawer of the cabinet") zs puts the bowl on the
cabinet's top side, and on :40 ("on the cabinet shelf", the shelf's top region) on the shelf's bottom region or top side: the right
body, the wrong REGION. The caddy part tasks (front / back / left / right compartment) are carried elsewhere, also a region-level
goal. A body displacement cannot separate two regions of one body, so G3 is defined on different-body errors only (section 4), and
the region level is read descriptively.

## 2. Counterfactuals and their validity (pre-declared)

- **Bodies:** every movable object and every fixture except tables and the floor; the held object is excluded (D4).
- **Perturbations:** one body at a time, x and y, +-5 cm (PRIMARY) and +-10 cm (secondary). Movable objects move through their free
  joint; fixtures through their model body position, children with them.
- **Validity (D4's rules, unchanged):** the body's deepest contact penetration grows by at most 2 mm; a 50-step pinned-robot settle
  leaves the body within 5 mm of its base end + delta and every other movable body within 5 mm of its base end. A body is scored at a
  magnitude only if all 4 of its perturbations at that magnitude are valid. Rejections are reported by reason and magnitude.

## 3. The sandbox and the readout (pre-declared)

- **State restore.** A sandbox starts from the observation's full MjSimState (one body displaced). Two pieces of robosuite state lie
  outside the MjSimState and are restored too:
  - the Panda gripper's ramp state (`current_action`, +-0.01 per physics substep toward the commanded sign; never reset by env.reset).
    It is reconstructed from the executed gripper commands before the observation: the zs recording's own actions for dev (after the
    10 settle steps); the demo's recorded gripper commands for training (read, not executed);
  - the OSC goal (set to the current pose), a zero solver warm start, the env timestep.
  Several sandboxes share one env: each is snapshotted (MjSimState, body positions, warm start, gripper ramp, OSC goal, timestep) and
  restored before every render and step. `--parity` checks (a) that re-executing the RECORDED zs actions from a restored recorded state
  reproduces the recording, and (b) that interleaved sandboxes equal uninterrupted ones bit for bit.
- **Rollout.** Every 5 steps, both cameras are rendered at 256 and resized with padding to 224 (D4's render), the 8-D state is read,
  and the frozen policy's chunk is sampled by its own Euler sampler (10 steps) from the noise of seed (set base + query index), shared
  across all sandboxes of an observation. The first 5 actions are executed. A sandbox ends at its first executed open command (the
  RELEASE), at task success (LIBERO's done: many In / On predicates already hold while the object is held, so on those tasks success
  comes before any open command; the smoke showed this on :1, where the zs recording itself never opens), or after 24 queries
  (120 steps; flagged). The readout point is the hand at that moment: "release" below means release-or-success.
- **Noise sets.** A1 and A2 for the +-5 cm perturbations and the base; A1 alone for +-10 cm; B1 and B2 on the base only (the noise
  floor). The served batched sampler equals the single-observation sampler for the same seed (checked at server start).
- **PRIMARY readout:** the hand's xy at the release, mean over A1 and A2, at +-5 cm.
  - J_Z (3 x 2) = [(e(+x) - e(-x)) / 2 delta, (e(+y) - e(-y)) / 2 delta]; the xy block is primary.
  - **Bound body** = the scored body with the largest ||J_Z,xy||_F.
  - Noise floor = ||e_B - e_A||_xy / (sqrt(2) delta).
- **Secondary readouts (descriptive, same bound-body rule):** `release_10` (+-10 cm, A1); `release_xyz` (the full 3 x 2 J, z row
  included); `chunk_mean` (the first chunk's mean integrated position, D4's scale); `chunk_end` (D4's first-chunk endpoint, at
  +-5 cm); `obj_release` (the held object's position at the release).

## 4. Hypotheses (Holm over G1, G2, G3 at alpha 0.05; pi05_l90c)

Bootstrap: two-level task-cluster (tasks, then observations), B 10,000, seed 0 (d4_read.boot2).

- **G1 readout validity (training observations).** The target body is the bound body in >= 0.8 of the observations where it is
  scored, and the 95% CI's lower bound exceeds chance (the mean of 1 / scored bodies). p = the bootstrap share of (hit - chance) <= 0.
- **G2 systematicity (training observations).** d4_read.h2 unchanged on the primary readout: leave-type-out (target type) prediction
  of the target's 4 release responses from the mean target operator of the other types; R^2 (through the origin, pooled) >= 0.5 and
  > the filler-shuffled control (operator pooled from a hash-chosen non-target scored body); p = the task-cluster bootstrap share of
  (R^2 - R^2_ctrl) <= 0 (B 2000). The type-label-shuffle reference is descriptive.
- **G3 diagnosis (zs dev wrong-goal observations).** The bound body is the body the object was actually placed on, in >= 0.7.
  - The actual body: the typer's alternatives mapped to bodies, minus the own target body and the held object. Exactly one -> it. None
    on a carried-elsewhere episode -> the body nearest (xy) the zs release hand, if within 10 cm (and not the own target). Otherwise the
    episode is excluded: `same_body_region` (the alternative is a region of the target body: :29, :40), `ambiguous`, `no_body`,
    `unscored`; counts reported.
  - p = the two-level bootstrap share of (rate_actual - rate_target) <= 0.
  - Expected n: mostly :16 (14 episodes) plus the caddy tasks' near bodies. This is small; the CI is reported.

**Also reported (descriptive):**
- validity per magnitude; release share; noise floor;
- per target-type operator table (mean J, lambda, norm, hit);
- every secondary readout's G1 / G2 / G3 numbers;
- dev successes: the bound body is the own target; dev recombination wrong-goal: the same;
- **sandbox fidelity:** a sixth base sandbox per dev observation uses the RECORDED zs noise seeds (13 + 2 warm-up calls + the earlier
  trials' calls + the query index); its release vs the recorded zs release (median xy distance, share within 3 cm, release agreement);
- **region readout:** for the base release point (A mean, xyz), the nearest BDDL region site of the target body vs the own target
  region, on successes and on the same-body wrong-goal episodes (:29, :40);
- **goal-role vision anchor (D3's criterion):** the argmax agentview SigLIP token of the target type's training prototype
  (results/rb4/protos.npz) lies in the target body's segmentation region; also within 5 cm (xy, live depth) and the median error;
  training, dev successes, dev wrong-goal (all, recombination only).

## 5. Decision for stage B (R6, exploratory closed-loop smoke)

R6 is run only if G1 passes (the operator need not pass G2 / G3). Otherwise D5 is recorded and handed back with a diagnosis.

## 6. Cost (planned)

About 66 sandboxes per observation (4-5 bases + about 5 bodies x (4 x 2 at +-5 cm + 4 at +-10 cm)), up to 24 queries each, 283
observations. The training smoke took 145 s for one observation (68 sandboxes, 69 s of it in the server). Plan: 8 A40 jobs x 3
server / client pairs (24 shards), about 5-7 A40-h.

## 7. Development (before this plan was fixed)

- **Index:** 156 training (40 tasks) + 127 dev observations (90 wrong-goal, 37 zs successes; results/d5/index.json).
- **Parity (results/d5/parity.json; 6 dev states, 40 steps):**
  - restoring a recorded zs state (with the reconstructed gripper ramp) and re-executing the RECORDED zs actions keeps the eef within
    0.4-2.7 mm of the recording on 5 of 6 states; on one contact-chaotic :40 state (the frying pan at the shelf) it drifts 8.6 cm;
  - interleaved vs uninterrupted sandboxes: equal to <= 2e-9 (state max) on 4 of 6, 4e-4 on one and 0.17 on the same :40 state.
    So snapshot / restore is not bit-exact (some robosuite / LIBERO Python-side state is not captured), and chaotic contact amplifies
    a 1e-9 difference. The noise-floor pair (B vs A) carries the same kind of variation, and every counterfactual shares it.
  - An earlier snapshot (MjSimState + warm start only) was worse (0.16 / 1.13 maximum); the mj_getState(INTEGRATION) version is used.
- **Server parity:** batched rows equal the single-observation sampler to 1.9e-3 to 2.7e-3 action units (bf16 batching, as D4).
- **Smoke (one observation per set; mechanics only):** training tr_libero_10_1_e12_carry68: 7 bodies, 68 sandboxes, all released,
  145 s. Dev dv_libero_90_1_t6_post125: 4 bodies, 3 rejected for penetration at both magnitudes, 17 sandboxes, every one ended by
  task success before any open command; the zs recording itself never opens. The release-or-success rule (section 3) was fixed
  after this smoke and before any readout was computed.
- **Disclosure:** during development an empty `python3 -` heredoc was issued on the login node; the harness denied it, so it did not
  run. Every computation ran on compute nodes.
