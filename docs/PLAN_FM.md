# Plan FM: the failure map of few-shot composition in pi05_l90c (written 2026-10-24, before any typed aggregate or stage-B rollout)

**Question** (docs/PLAN_STORY.md section 2). Where does few-shot composition fail in a traditional VLA (pi05_l90c@5000 on
LIBERO-90-C's 29 dev tasks)? Each failure gets a reading from the theory and a test of the repair:
- the theory: language is compositional (Frege; Fodor and Pylyshyn), but the VLA binds the instruction holistically to the
  scene, and binding leaves the word positions by layer 2;
- the repair, in the language channel: a created input token, or a composition of stored sentences.

M1 (docs/PLAN_M1.md section 6) gives the arms at k = 1. This plan types every M1 episode (stage A, no new rollout). It then adds
the few-shot curve (stage B).

Code: `experiments/fm_types.py` (the typer), `experiments/fm_fewshot.py` (stage-B fits, launch, read, figures).
Pre-registration: `experiments/prereg/pfm_failure_map.py` (ledger `m3_prereg`, tag `pfm_failure_map`). Ledger rows: `fm_fit`
(per task and k), `fm_main` (rollout packs, via roll_record), `fm_read` (with the typer validation). Payloads: `results/fm/`.

## 1. Stage A: the episode typer (offline)

**What M1 recorded.** M1's rollouts recorded these per episode:
- the 8-D EEF and finger state, and the actions;
- every movable object's position and quaternion at every policy step;
- `bound`: the first movable object moved more than 3 cm while the gripper is commanded closed, and its step.

**The goal and its regions come from the task's own BDDL, read by LIBERO's own predicates.**
1. A CPU LIBERO env (`ControlEnv`, no renderer) is set to the episode's start state (`results/robot/fresh_m1`).
2. Every movable object is put at its last recorded pose, at rest.
3. 10 control steps of physics run, with the arm holding its home pose. The last record comes before the last action, so a bowl
   that is still being lowered sits 1 cm above the plate, out of contact.
4. The named object's In / On atoms are then evaluated.

**Types of a failed episode** (precedence in this order; the thresholds are fixed here):

| type | rule | role-filler reading |
|---|---|---|
| fixture | the goal has no In / On atom (open / close a drawer, turn on the stove). Articulated fixtures are read only by success, so these failures stay untyped | (none) |
| no_commit | no movable object is ever bound | (none) |
| wrong_object | the bound object is not the instruction's named object (the first argument of an In / On goal atom). Subtype wrong_instance: another instance of the named object's type | object-role binding error |
| wrong_goal | the named object is bound and either (a) at its final pose it satisfies In / On with another BDDL region or object of the scene (not an init region, not a table, not its own target), or (b) it was carried at least 10 cm (xy) from where it was bound, while getting less than half way (xy) to its target | goal-role binding error |
| motion | the named object is bound and either got at least half way to its target, or was not carried 10 cm (grasp lost, dropped, placement failed). Subtype in_region: its own atom holds at the final pose, so the episode failed on another atom (the articulated step of a two-step task) | missing skill |

**Validation.**
- By hand, before the pre-registration, on 16 episodes of 8 task x arm cells (the traces were printed and read; listed in the
  pre-registration's disclosure).
- By rule, on every successful episode of a pick-and-place task: the named object's own atom must hold at the settled final pose.
  The read reports the share.

**Axes per task.**

| axis | levels | source |
|---|---|---|
| CD bin | 0.1 / 0.3 / 0.5; the part set is "part" | split.json dev bins; task CD from dev_per_task |
| part novelty | combination (15) / part (14) | split.json |
| scene | familiar (at least one training cell of split.json "train" in the scene) / new | factors.py scene |
| instance | the instruction names left / right / front / back / middle, AND the BDDL holds at least 2 instances of a moved or target object's type | factors.py, BDDL objects |

- Familiar scenes: the 20 non-study tasks. New scenes: the 9 STUDY_SCENE1-3 tasks (corrected after the read from "21 / 8"; the axis rule did not change).
- Instance tasks: 14, 16, 60 and 38. Caddy compartments and "right of the caddy" are regions, so they do not count as instances.
- Pick-and-place tasks (with an In / On atom): 24 of 29, of which 11 are combination tasks and 13 are part tasks.

## 2. Stage B: the few-shot curve

| arm | what | settings |
|---|---|---|
| lora_k5_200 / lora_k5_1000 | per-task LoRA from k = 5 demos, snapshots at steps 200 and 1000 | M1's: r16 on the action expert, lr 1e-4, batch 32, seed 0, openpi's flow loss |
| lora_k10_200 / lora_k10_1000 | the same from k = 10 demos | as above |
| create_k5 | the KV token from k = 5 demos | M1's: n 4 entries per layer, 1000 Adam steps, lr 1e-2, batch 32, seed 13, init from the first frame's language entries |

- **The demos** are the task's own-scene IPEC episodes 0 to k-1 (`l90c.task_episodes`). Episode 0 is M1's one demo, so the
  demo sets nest. LIBERO's teleop demos do not start from the evaluation states (fresh_m1 is new: M1's `--fresh-check`).
- **Training data.** Every frame of every demo trains, through a CPU-held PrefixBank.
- **Rollouts.** The same 29 tasks, the same fresh_m1 states (20 per task) and noise seed 13, so every arm pairs with M1's k = 1
  arms (zs, compose_sel, create, lora200, lora1000).
- **Recording.** Rollouts record as M1's did, so the typer types the new arms too.
- **Cost (planned).** 58 fits of about 6 to 15 minutes on an A40, and 2,900 episodes in 37 packs of 4 instances.

## 3. Hypotheses (pre-registered; Holm over F1-F5 at alpha 0.05)

Each failure gets one test of its prediction and a pre-specified repair reading (descriptive, unadjusted).

- **F1 holistic binding.**
  - Test: on the 11 combination pick-and-place tasks, zs's per-task binding-error rate (wrong object + wrong goal, over 20
    episodes) rises with task CD. Statistic: Spearman rho(task CD, rate) > 0, one-sided permutation p (100,000 permutations,
    seed 0).
  - Repair reading: the same rho and the by-bin rates for compose_sel, compose_cov, create, cc, lora1000, create_k5 and
    lora_k10_1000. Also the logistic slope of success on CD per arm (combination tasks, task-cluster bootstrap), the
    systematicity readout.
- **F2 few-shot memorises the demo.** It passes only if (a) and (b) both pass (p = max).
  - (a) lora1000 (k = 1) binds the demo's layout: its wrong-object episodes start farther from the demo's grasp point.
    - Layout distance: the xy distance between the named object's start and the demo's grasp point. The grasp point is the EEF
      at the demo's sustained-close onset nearest the named object's mean start, because two-step demos also close on a knob.
    - Statistic: the within-task centred sum of the layout distances of the wrong-object episodes. One-sided permutation of the
      wrong-object labels within each task (20,000 permutations).
  - (b) more demos reduce it: lora1000 has more wrong-object episodes at k = 1 than at k = 10. Paired by task x start state on
    the pick-and-place tasks; one-sided exact sign test on the discordant pairs.
  - Repair reading: wrong-object counts for zs, compose_sel (no k), create (k = 1, 5), and LoRA at k = 1, 5 and 10. Also the
    share of wrong-object grasps that land on an object nearer the demo's grasp point than the named object.
- **F3 instance averaging.**
  - Test: on the 4 instance tasks, more than half of zs's typed failures are instance-role binding errors. These are a grasp on
    another instance of the named type, or a placement in or on another instance of the target's type. Two-level cluster
    bootstrap; p = the share of bootstrap shares <= 0.5. With fewer than 10 typed zs failures, the test is not testable and
    does not pass.
  - Repair reading: instance errors and successes under compose_sel, create, cc, lora1000, create_k5 and lora_k10_1000 against
    zs (sign test on discordant pairs). The pooled action-space units of E1/E2 are not in this study.
- **F4 a genuinely new part.**
  - Test: on the 13 part pick-and-place tasks, more than half of the typed failures of create AND of lora1000 are motion
    failures (cluster bootstrap each; p = max).
  - Repair reading:
    - zs's failure types on the part tasks;
    - part-task successes per arm, against the known M1 numbers (create 117 vs lora1000 149 of 280, at 1/254 bytes);
    - create_k5 vs lora_k5_1000.
- **F5 interference.**
  - Test: on the composed pick-and-place tasks (compose_sel picked a composition), the induced failures are mostly binding
    errors. Induced failures are the episodes that zs solves and compose_sel fails; the binding errors are wrong object and
    wrong goal. Statistic: share > 0.5, cluster bootstrap. With fewer than 10 induced typed failures, the test is not testable.
  - Repair reading:
    - M1 already showed that the zs guard does not remove the interference (compose_sel 150 vs zs 180; a success-level fact
      known before this plan);
    - the share of induced wrong-object grasps on an object that a source sentence names.

**Secondary (descriptive).**
- Failure-type shares by arm x axis (the figure).
- The success-vs-k curve, overall, by combination / part and by failure type, paired against k = 1.
- create_k5 vs lora_k5_1000.
- Composition cards for the tasks the token fixed (create - zs >= 5 of 20): source sentences, uncovered roles, selection,
  token, and the bound-object distribution under zs vs create.

**Disclosure.**
- M1's success numbers, per task and per arm, were read (results/m1/read.json) before this plan; the typed aggregates were not.
- The typer was checked by hand on 16 episodes before the pre-registration, and their 16 types were seen: listed in the
  pre-registration.
- F5's success-level fact (compose_sel below zs) is already known, so F5 tests the failure type, not the success drop.

## 4. Result (2026-10-24; ledger `fm_read` tag read, results/fm/read.json; figures/robot/fm_types.png, fm_k_curve.png)

29 dev tasks x 20 fresh_m1 states (580 paired episodes per arm), pi05_l90c@5000, noise seed 13.
- Stage A typed M1's 8 arms (4,640 episodes, no new rollout).
- Stage B fitted 58 adapters / tokens (all completed) and rolled 145 x 20 = 2,900 new episodes.

**Typer validation.** On the 3,485 successful pick-and-place episodes, the named object's own atom holds at the settled final pose
in 3,075 (88%). The misses concentrate on the caddy tasks (76-78: the episode succeeds while the mug or book is still in the
gripper, and the reconstruction drops it from the hand) and on bowls 3 cm from the plate centre (On's xy threshold). So the
in-region test, which splits wrong_goal from motion, is approximate. wrong_object and no_commit do not depend on it.

### 4.1 Failure types (episodes of 580 per arm; fixture = articulation-only tasks, success-only)

| arm | success | no commit | wrong object | wrong goal | motion | fixture |
|---|---|---|---|---|---|---|
| zs | 180 | 7 | 63 | 152 | 117 | 61 |
| compose_sel | 150 | 24 | 123 | 120 | 95 | 68 |
| create (token, k 1) | 351 | 68 | 5 | 29 | 105 | 22 |
| lora1000 (k 1) | 387 | 51 | 9 | 15 | 102 | 16 |
| create_k5 | 446 | 25 | 5 | 4 | 96 | 4 |
| lora_k5_1000 | 485 | 20 | 1 | 9 | 57 | 8 |
| lora_k10_1000 | 483 | 6 | 2 | 12 | 67 | 10 |

By axis (success / no commit / wrong object / wrong goal / motion / fixture):

| axis | zs | compose_sel | create k1 | lora1000 k1 | create k5 | lora1000 k5 | lora1000 k10 |
|---|---|---|---|---|---|---|---|
| CD 0.1 (100) | 80/0/0/16/3/1 | 60/0/20/7/4/9 | 86/0/0/0/11/3 | 93/0/0/0/6/1 | 95/0/0/0/5/0 | 97/0/0/0/2/1 | 97/0/0/0/3/0 |
| CD 0.3 (100) | 43/5/3/20/9/20 | 32/2/22/1/23/20 | 75/3/4/0/4/14 | 84/1/6/1/8/0 | 89/1/4/0/3/3 | 93/2/1/0/0/4 | 97/0/2/1/0/0 |
| CD 0.5 (100) | 57/0/0/16/7/20 | 58/0/0/16/7/19 | 73/12/0/3/7/5 | 61/11/0/2/11/15 | 99/0/0/1/0/0 | 91/3/0/2/3/1 | 88/0/0/0/2/10 |
| new part (280) | 0/2/60/100/98/20 | 0/22/81/96/61/20 | 117/53/1/26/83/0 | 149/39/3/12/77/0 | 163/24/1/3/88/1 | 204/15/0/7/52/2 | 201/6/0/11/62/0 |
| familiar scene (400) | 180/5/43/52/59/61 | 150/22/79/26/55/68 | 284/56/5/4/29/22 | 297/38/9/4/36/16 | 348/24/5/1/18/4 | 351/20/1/6/14/8 | 369/6/2/2/11/10 |
| new scene (180) | 0/2/20/100/58/0 | 0/2/44/94/40/0 | 67/12/0/25/76/0 | 90/13/0/11/66/0 | 98/1/0/3/78/0 | 134/0/0/3/43/0 | 114/0/0/10/56/0 |
| instance (80) | 41/0/20/16/3/0 | 29/3/37/7/4/0 | 54/14/1/0/11/0 | 62/5/3/1/9/0 | 63/11/1/0/5/0 | 70/6/0/0/4/0 | 73/3/0/1/3/0 |
| no qualifier (500) | 139/7/43/136/114/61 | 121/21/86/113/91/68 | 297/54/4/29/94/22 | 325/46/6/14/93/16 | 383/14/4/4/91/4 | 415/14/1/9/53/8 | 410/3/2/11/64/10 |

**The shape of the map.**
- **zero-shot fails by binding.** 215 of its 339 typed failures are binding errors (wrong object 63, wrong goal 152). In the new
  study scenes, wrong goal dominates (100 of 180 episodes): the book goes into the wrong caddy compartment.
- **Composing stored sentences adds wrong-object errors** (63 to 123). It fixes none of zs's binding errors.
- **One created token removes almost all binding errors** (215 to 34; 9 at k 5). LoRA does the same (24 at k 1).
- **What remains under token and LoRA is the skill.** These are motion failures (57 to 105 episodes) and no-commits (6 to 68),
  mostly on the new-part and new-scene tasks.

### 4.2 The few-shot curve (successes; paired against k = 1 with the two-level cluster bootstrap, 95% CI)

| arm | k = 1 (M1) | k = 5 | k = 10 | combination (300) k 1 / 5 / 10 | part (280) k 1 / 5 / 10 |
|---|---|---|---|---|---|
| zs | 180 | | | 180 | 0 |
| compose_sel | 150 | | | 150 | 0 |
| LoRA 200 steps | 369 | 436 (+0.116 [+0.014, +0.217]) | 430 (+0.105 [+0.007, +0.200]) | 234 / 287 / 284 | 135 / 149 / 146 |
| LoRA 1000 steps | 387 | 485 (+0.169 [+0.083, +0.259]) | 483 (+0.166 [+0.083, +0.252]) | 238 / 281 / 282 | 149 / 204 / 201 |
| KV token | 351 | 446 (+0.164 [+0.064, +0.266]) | (not run) | 234 / 283 | 117 / 163 |

- **k = 5 saturates.** k = 10 adds nothing for LoRA.
- **On the combination tasks** the 74 KB token stays level with the 18.7 MB LoRA at k = 5 (283 vs 281).
- **On the new parts it trails:** token 163 vs LoRA 204. Pooled, create_k5 - lora_k5_1000 = -0.067 [-0.143, +0.003], discordant
  53 to 92.
- **Fit time, mean A40 seconds from demo arrival:** token k 5 273, LoRA k 5 298 (1000 steps), LoRA k 10 357 (about 1,450
  frames).

### 4.3 Pre-registered verdicts (Holm over F1-F5 at alpha 0.05): F2 passes; F1, F3, F4 and F5 fail

- **F1 holistic binding: FAILS.**
  - The test: rho(task CD, zs binding-error rate) = -0.14, permutation p 0.66.
  - zs's binding-error rate by bin is 0.27 / 0.29 / 0.20. The logistic slope of success on CD has a CI that includes 0 for
    every arm.
  - Theory reading:
    - zs's errors are binding errors, but they do not grow with compound divergence.
    - The VLA's binding is holistic (it misbinds even at CD 0.1), not graded by how new the combination is.
    - What predicts misbinding is the scene. In the new study scenes, zs binds the right book and puts it in the wrong place.
  - Repair: the token sets the binding errors to 0 / 0.05 / 0.04 by bin, flat in CD. Composition raises them at CD 0.1 (0.45).
- **F2 few-shot memorises the demo: PASSES** (p 0.0078 against the Holm threshold 0.010).
  - (a) lora1000's wrong-object episodes start farther from the demo's grasp point: mean 0.063 m vs 0.044 m, permutation
    p 0.0012.
  - (b) k = 10 removes them: discordant 7 to 0, sign p 0.0078.
  - Caveat: the effect is small. LoRA from one demo has only 9 wrong-object episodes in 580, and only 1 of the 9 grasps the
    object nearer the demo's grasp point.
  - So LoRA's layout dependence shows, but it is not the main failure. The main failure is motion and no-commit (153 of 177
    typed).
  - Repair: the token from one demo has 5 wrong-object episodes. That is as few as LoRA's, and flat in k (5 at k 5).
- **F3 instance averaging: FAILS** (p 0.097).
  - 31 of 39 zs typed failures on the instance tasks are instance-role errors (79%), but they come from two tasks.
    - Task 38: all 20 episodes grasp the left moka pot instead of the right one.
    - Task 16: 11 of 18 stack on the wrong bowl.
  - Tasks 14 and 60 barely fail. The two-task cluster bootstrap is too wide (CI 0 to 1).
  - Theory reading: where zs misses an instance, the qualifier word does not reach the binding.
  - Repair: the created token cuts the instance errors from 31 to 1 (discordant 30 to 0, sign p 2e-9; also 1 at k 5).
    lora1000 cuts them to 3. Composition leaves 31.
  - On task 38 the token trades the wrong pot for no-commit (14 of 20). Word-level qualifier prototypes were not built here.
- **F4 a genuinely new part: FAILS.**
  - Motion share of the typed part-task failures: create 83 of 163 (0.51 [0.30, 0.73]); lora1000 77 of 131 (0.59 [0.36, 0.81]).
    The rest is no-commit (53, 39) and wrong goal (26, 12).
  - zs on new parts fails mostly by binding (160 of 260 typed) and not by motion. So the new part breaks the binding first.
  - Repair:
    - only creation or fine-tuning moves the parts (zs and composition: 0);
    - the token reaches 117 at k 1 and 163 at k 5;
    - LoRA reaches 149 and 204, at 254 times the bytes.
- **F5 interference: FAILS** (p 0.41).
  - compose_sel's 29 induced typed failures are 13 wrong object, 3 wrong goal and 13 motion: binding 0.55 [0.09, 0.92].
  - Descriptive: 11 of the 13 wrongly grasped objects are named by one of the composed source sentences. When composition
    misbinds, it imports a source's object.
  - The zs guard does not remove interference (M1: compose_sel 150 vs zs 180).

### 4.4 Composition cards (tasks the token fixed by at least 5 of 20; full cards in results/fm/read.json "cards")

- libero_90:58 "pick up the ketchup and put it in the tray" (new part).
  - Sources: "pick up the alphabet soup ..." and "pick up the cream cheese ... in the tray"; uncovered role: (O, ketchup).
  - zs grasps the tomato sauce in 19 of 20; the token grasps the ketchup in 16 of 20 and succeeds in 13.
- libero_90:83 "... book ... right compartment" (new part). zs grasps the porcelain mug in 18 of 20; the token grasps the book in
  20 and succeeds in 7.
- libero_90:38 "put the right moka pot on the stove".
  - Source: "put the moka pot on the stove"; uncovered role: (O, right).
  - zs grasps moka_pot_2 (the left one) in 20 of 20; the token grasps the right pot in 5, the left in 1, and none in 14.
- libero_90:29, 73, 75, 78, 79, 80: the bound object is already right under zs. The token turns wrong-goal or motion failures into
  successes (29: 1 to 20).

### 4.5 Reading for the story (PLAN_STORY)

- **Where it fails.** The traditional VLA's few-shot failure is a binding failure:
  - zero-shot misbinds the object or the goal in 63% of its typed failures;
  - this is not graded by compound divergence (F1 fails);
  - it is worst in new scenes and on new parts.
- **What repairs it.**
  - A single created input token (no weight change) removes the binding errors at k 1 and k 5. That is the language-channel
    repair.
  - Recombining stored sentences does not repair the binding: it imports the sources' objects.
  - What remains after the token, as after LoRA, is the motion skill on new parts. There LoRA's extra capacity still wins at
    k 5 (204 vs 163).
- **Of the five pre-registered failure readings, only F2 (demo layout) passes.** Its effect is small.

**Disclosures.**
- **Code changes after the pre-registration row** (the tests are unchanged):
  - The fit and rollout launchers strip the parent job's SLURM_* variables. The first 58 fit jobs, submitted from inside a CPU
    job, all failed in 4 s on a cpu-bind error; the fits were resubmitted from the login shell.
  - The figure labels were moved to fix collisions.
  - The scene-axis count in section 1 was corrected (20 / 9; the rule is unchanged).
- The typer's in-region test agrees with the simulator's success on 88% of successes (section 4).
- The stage-A M1 aggregate was first read without a ledger row (an interim `--read`, after the pre-registration and before any
  stage-B rollout). The recorded read is the final one.
- The hand-checked episodes and the `python3 --version` on the login node are listed in the pre-registration.
