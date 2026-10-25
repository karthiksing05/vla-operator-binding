# Probe R2a: can the frozen VLA's own action modes recover binding without a demo? (written 2026-10-24, before any mode)

**Question.** pi05_l90c@5000 fails its zero-shot dev tasks mostly by binding: the wrong object or the wrong goal, driven by the
scene (docs/PLAN_FM.md section 4). P2 found that its action modes are distinct (separation 0.68) but that part prompts do not
name them. R2a asks whether the scene's afforded options are already among the modes of the null-prompt field, and whether the
instruction can pick the right one by likelihood ratio, with no demo, no gradient on weights and no weight change.

Code: `experiments/r2a_null_modes.py`. Pre-registration: `experiments/prereg/pr2a_null_modes.py` (ledger `m3_prereg`, tag
`pr2a_null_modes`), recorded after the readout validation and before any mode was computed. Ledger rows: `r2a_zs` (rollouts, via
roll_record), `r2a_read`. Payloads: `results/r2a/` (observation dumps, modes and rollouts stay local).

## 1. Method (no demo, frozen policy)

At an observation o (images + state):
1. **Null options.** Draw 12 chunks from the null-prompt policy (empty instruction, images kept), noise them to the flow levels
   u in {0.1, 0.3, 0.55}, and add 12 uninformed starts 0.7·eps at u 0.7 (48 starts). Ascend log p_u under the null field
   (score_ops.mode_ascent: Adam, 80 iterations, lr 0.02) and take the Tweedie x0 modes. Cluster the 48 modes greedily
   (rms 0.15 in normalised action space, P2's coarse tolerance). Each cluster is an afforded option.
2. **Rank the options by the instruction / null likelihood ratio.** For a chunk a:
   - `L(a | o, c) = mean over t in {0.1, ..., 0.9} and K = 8 shared noises of || v_c((1-t) a + t eps_k, t) - (eps_k - a) ||^2`
     (the conditional flow-matching loss the policy was trained with);
   - `Delta(a) = L(a | o, null) - L(a | o, instr)`; an option's score is the mean Delta of its members.
3. **The same starts** are also ascended under the instructed field and under a CAG field
   `v_null + 2 (v_instr - v_null)` (CAG-VA, 2602.17659, Eq. 4, training-free variant).

**Why this estimator.**
- `Delta` is the Diffusion-Classifier estimator (Li et al., ICCV 2023): the ELBO difference of two conditions, which estimates
  `log p(a | o, instr) - log p(a | o, null)` up to the ELBO's t-weighting.
- By Bayes, ranking options by this pointwise mutual information is ranking them by `p(instr | a, o)`: which afforded option the
  instruction refers to. This is the listener's inference the binding failure lacks.
- It uses the policy's own training loss, needs only forward passes, and pairs the noise across conditions and options, so the
  difference has low variance.
- The probability-flow ODE log-likelihood was not chosen. It needs the divergence of v along the 10-step path (Hutchinson
  probes), costs about 10x more passes, and is noisier for a model that was trained only with the flow-matching loss.
- The eps-weighted ELBO variant (weight (1-t)^2) is reported as a robustness read, not tested.

## 2. Data: new zero-shot rollouts that save their observations

- **Tasks.** The 29 dev tasks of results/m1/plan.json (15 recombination, 14 part). split_sealed.json is never opened.
- **Start states.** A NEW set `results/robot/fresh_r2a`: LIBERO's sampler at BASE 7100000 + k, 20 per task. No logged base is
  within 1000 of it, and `--fresh-check` asserts every state is > 1e-6 from every earlier libero_90 fresh set.
- **Rollouts.** zs as M1 served it (MODE ours, plain Euler 10 steps, noise seed 13, replan 5). The client records the policy-input
  images, the state, the actions, every movable object's pose and the bound object at every step.
- **Typing.** Every episode is typed by fm_types' typer (wrong_object / wrong_goal / motion / no_commit / fixture), on these states.
- **Observations** (query steps j, every 5 steps; fixed before any rollout):
  - *pre-grasp* (object role): queries before the first bind with the gripper open and the hand at least 5 cm (xy) from every
    movable object; the earliest, middle and latest such query (at most 3 per episode). The 22 pick-and-place tasks whose first
    step is a pick-and-place: libero_90:1 and :45 articulate a fixture first and have no object-role observation.
  - *post-grasp* (goal role): the named object bound, still held and lifted 3 cm, the hand at least 6 cm (xy) from the own
    target; the first such query (one per episode; two before the GPU smoke, section 7 disclosures).

## 3. The object-assignment readout

- A chunk (10 steps, unnormalised by the policy's own output transform) is integrated into an end-effector path with the
  demo-calibrated scale (pp2_dev_l90c). It points at the candidate whose direction from the hand best matches the path's
  horizontal displacement (cosine; P2's `pointed`). A chunk moving less than 2 cm (xy) points at nothing.
- Candidates:
  - object role: every movable object of the scene, at its recorded pose;
  - goal role: the own target, then every other BDDL region or object a placement could land in (the typer's alternatives),
    with options within 3 cm (xy) of an earlier one merged. Shelves and drawers stacked at the same xy merge, so the goal role
    cannot tell them apart.
- A cluster's object is the majority of its members' objects.
- **Validation before any mode** (`--validate`, the rollouts alone): on pre-grasp observations of episodes that bound an
  object, the policy's EXECUTED 10-step action window must point at the object it then bound; on post-grasp observations of
  successful episodes, at the own target. Plus a hand check of the printed geometry and images (`--probe`).

## 4. Arms compared at each observation (the named object is the first argument of the goal's In / On atom)

| arm | the option chosen |
|---|---|
| ratio_top | the null option with the largest Delta (the method) |
| policy_sample | the instructed policy's own samples (12 draws at o; the share pointing at the named object) |
| policy_exec | the chunk the policy executed in the rollout (10-step window) |
| instr_lik_top | the null option with the largest instructed likelihood alone (-L(a \| o, instr)), no ratio |
| instr_mode_top | the instructed field's own modes from the same starts, the largest cluster |
| cag_mode_top | the CAG field's modes from the same starts, the largest cluster |
| null_mass_top | the largest null cluster (the scene's default) |
| random | a uniform choice among the null options (expected hit) |

coverage = some null option points at the named object.

## 5. Hypotheses (pre-registered; Holm over H2-H3 at alpha 0.05; H1 an estimate gate)

The unit is the episode: an episode's value is the mean over its pre-grasp observations. Rates are pooled with the two-level
task-cluster bootstrap (tasks, then episodes; B 10,000, seed 0).
- **H1 coverage.** On the 22 object-role tasks, the pooled coverage is >= 0.8.
- **H2 the ratio fixes the policy's object choice where zs fails by wrong object.** On zs episodes typed wrong_object,
  ratio_top > policy_sample. One-sided exact sign test on the episodes whose two means differ. With fewer than 10 such
  episodes, H2 is not testable and does not pass.
- **H3 no harm where zs succeeds.** On zs successful episodes of the object-role tasks, ratio_top - policy_sample >= -0.05:
  one-sided bootstrap p = the share of bootstrap differences <= -0.05.

Descriptive: every arm by bin / part / scene / instance qualifier / zs type; coverage of the instructed and CAG modes and of the
null drafts alone; the goal role (post-grasp), the same tables; the eps-weighted ratio; wrong-object cards per task.

**Closed loop only if H1 and H2 pass**: a smoke on about 5 tasks (exploratory): follow the top-ratio mode until the policy's own
grasp, then the instructed policy, gated on the ratio disagreeing with the policy's choice.

## 6. Cost (planned)

580 zs episodes (8 packs of 4 instances on A40s, about 1 h); extraction on CPU; modes: about 2,600 observations x about 15 s,
about 11 A40-h over 6 shards.

Actual: packs of 16 to 58 min (recording images); modes 1,716 observation files x about 44 s, about 21 A40-h over 19 shards.

## 7. Result (2026-10-24; ledger `r2a_read` tag read, results/r2a/read.json; pre-registration m3_prereg `pr2a_null_modes`)

**Data.**
- zs on the 29 dev tasks x 20 fresh_r2a states (580 episodes): 177 successes (M1's zs on fresh_m1: 180).
- Typed: wrong object 56, wrong goal 152, motion 118, no commit 16, fixture 61.
- 1,677 observations: 1,305 pre-grasp (22 tasks, 437 episodes) and 372 post-grasp (21 tasks, one per episode).

**Readout validity** (before any mode; results/r2a/validate.json, 28 tasks):
- pre-grasp: the policy's executed 10-step window points at the object it then bound in 0.645 of 1,221 observations. Chance is
  0.379; 0.228 of the windows move less than 2 cm.
- post-grasp (successful episodes): 0.795 at the own target, chance 0.272.
- So the object-role readout is valid but noisy. That noise applies to every arm alike.

### 7.1 Object role (pre-grasp), episode means, task-cluster bootstrap 95% CI

| arm | all (437 ep, 22 tasks) | zs wrong object (55 ep, 7 tasks) | zs success (132 ep, 11 tasks) |
|---|---|---|---|
| coverage (some null option -> named object) | 0.534 [0.408, 0.659] | 0.19 | 0.56 |
| ratio_top (the method) | 0.500 [0.372, 0.626] | 0.07 | 0.54 |
| policy_sample (instructed policy's own samples) | 0.541 [0.435, 0.640] | 0.08 | 0.61 |
| policy_exec (the executed window) | 0.584 [0.458, 0.700] | | |
| instr_lik_top (no ratio) | 0.502 [0.372, 0.629] | | |
| instr_mode_top | 0.609 [0.474, 0.736] | 0.03 | 0.62 |
| cag_mode_top (w 2) | 0.590 [0.457, 0.717] | 0.02 | 0.64 |
| null_mass_top | 0.494 [0.366, 0.620] | 0.08 | 0.54 |
| random | 0.488 [0.365, 0.611] | 0.10 | 0.52 |

- Coverage of the null drafts before ascent is 0.725 [0.599, 0.839]. The instructed modes cover 0.630, the CAG modes 0.627.
- **The null field has one option at 0.704 of the pre-grasp observations.** Mode ascent collapses the null drafts onto the
  scene's default, so ascent loses options the drafts had.
- Given coverage, ratio_top is 0.927 [0.837, 0.981], random 0.899 [0.830, 0.948] and null_mass_top 0.910. The ratio adds
  little because there is usually little to choose from.
- By axis (coverage / ratio_top / policy_sample):

| axis | coverage | ratio_top | policy_sample |
|---|---|---|---|
| CD 0.1 (60 ep) | 0.39 | 0.39 | 0.54 |
| CD 0.3 (60 ep) | 0.51 | 0.48 | 0.52 |
| CD 0.5 (78 ep) | 0.52 | 0.50 | 0.59 |
| part (239 ep) | 0.58 | 0.53 | 0.53 |
| familiar scene (257 ep) | 0.45 | 0.42 | 0.50 |
| new scene (180 ep) | 0.65 | 0.62 | 0.60 |
| instance qualifier (80 ep) | 0.37 | 0.30 | 0.43 |
| no qualifier (357 ep) | 0.57 | 0.54 | 0.57 |

- Wrong-object cards (coverage / ratio_top / policy_sample over the task's wrong-object observations):
  - libero_90:38 (the right moka pot): 0.31 / 0.04 / 0.10. The ratio's top option is the left pot (moka_pot_2) or a non-moving
    mode.
  - libero_90:58 (ketchup): 0.00 / 0.00 / 0.01. No null option heads to the ketchup; the ratio picks the tomato sauce,
    as zs does.
  - libero_90:83 (book): 0.16 / 0.09 / 0.06. The ratio picks the porcelain mug, as zs does.

### 7.2 Pre-registered verdicts: H1, H2 and H3 all fail

- **H1 coverage FAILS.** 0.534 [0.408, 0.659] against the gate 0.8. On the wrong-object episodes it is 0.19: where zs
  misbinds, the right object is mostly not among the null field's modes.
- **H2 FAILS, in the wrong direction.** On the 55 wrong-object episodes, ratio_top beats policy_sample in 10 and loses in 28
  (one-sided sign p 0.999).
- **H3 FAILS.** On zs successes, ratio_top - policy_sample = -0.066 [-0.230, +0.079]; non-inferiority p 0.564 at margin -0.05.
- The closed-loop smoke was not run: it was gated on H1 and H2.

### 7.3 Goal role (post-grasp; descriptive, xy readout merges stacked shelves and drawers)

| | coverage | ratio_top | policy_sample | random | null_mass_top | single option |
|---|---|---|---|---|---|---|
| all (372 ep, 21 tasks) | 0.425 [0.247, 0.610] | 0.406 | 0.413 | 0.356 | 0.352 | 0.331 |
| given coverage | | 0.956 [0.892, 0.992] | 0.813 | 0.839 | 0.829 | |
| zs wrong goal (144 ep) | 0.08 | 0.06 | 0.13 | 0.05 | 0.05 | |
| new scene (132 ep) | 0.02 | 0.01 | 0.03 | 0.01 | 0.01 | |

- The null field offers more options after the grasp (one option in 0.331 of observations). Given coverage, the ratio picks
  the own target more often than random (0.956 vs 0.839). This is the one place the ratio ranks well.
- But coverage is the bottleneck again. Where zs puts the object in the wrong place (wrong goal), and in every new study
  scene, the own target is almost never a null option (0.08; 0.02). There, the right compartment of the caddy is not afforded.

### 7.4 Reading

1. **The frozen policy's own null field does not hold the right binding where zs fails.** Its modes are the scene's default
   options, and the default is what zs already does. Coverage on the wrong-object episodes is 0.19 for the object, and 0.08 on the
   wrong-goal episodes for the goal.
2. **Ranking is not the problem.** When the named option exists, the instruction/null likelihood ratio picks it in 0.93
   (object) and 0.96 (goal) of observations. The failure is that the option is absent.
3. **Mode ascent removes diversity.** The null drafts cover 0.725 before ascent and 0.534 after; 70% of pre-grasp observations
   ascend to one option. A gradient-free, demo-free selector over the policy's own modes cannot recover a binding the policy
   does not afford. This supports R's premise that the repair must be created in the conditioning (M1/FM: one created token
   removes the binding errors), not selected from the frozen field.

**Disclosures.**
- The readout changed once before any aggregate. A hand check (libero_90:14 trial 13, printed geometry) showed max-cosine
  picking the cabinet behind the plate on the same heading. So a chunk now points at the first candidate along its heading
  (in front of the hand, within a 5 cm corridor), else at the best cosine.
- A 2-observation GPU smoke (post-grasp observations of libero_90:14, results/r2a/scratch/smoke_modes, never read as data) ran
  before the pre-registration. It showed 63 s per observation and a collapsed null field. After it, only the null modes and
  drafts are scored (the instructed and CAG modes are ranked by mass only), and the post-grasp role keeps one observation per
  episode.
- The pre-registration row was written with 28 of 29 tasks rolled and validated; libero_90:16 (the last pack) joined the mode
  stage afterwards with the same code.
- Four descriptive "given coverage" / single-option columns were added at the read, after the first unrecorded print of
  the aggregates. The recorded read is the second run; the tests are unchanged. The first read crashed because the unscored
  instructed / CAG modes had no Delta (fixed: mass only).
- The extraction smoke on libero_90:14 and :9 wrote 39 extra post-grasp observation files under the old 2-per-episode rule.
  They are not in the index, so the read never uses them.
- The object readout's validity is 0.645 against chance 0.379, so every arm's absolute rate is attenuated. The comparisons
  between arms share the same readout.
