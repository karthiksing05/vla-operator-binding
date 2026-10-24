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

- Familiar scenes: the 21 non-study tasks. New scenes: the 8 STUDY_SCENE1-3 tasks.
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
