# LIBERO-PRO in our framing (plan H4, read G22)

LIBERO-PRO (Zhou et al., arXiv 2510.03827; code github.com/Zxy-MLlab/LIBERO-PRO at eafdb80, cloned to
`/coc/flash10/ksingara3/robot/ext/LIBERO-PRO`; BDDL and init files from the Hugging Face dataset `zhouxueyang/LIBERO-Pro` at
c86fc3b, in `/coc/flash10/ksingara3/robot/ext/LIBERO-PRO-data` and copied into the clone's `bddl_files/` and `init_files/`)
perturbs the four LIBERO training suites (Goal, Spatial, Object, 10). Each perturbation rewrites a task's BDDL file
(`perturbation.py`), and LIBERO-PRO ships 50 init states per perturbed task (`.pruned_init`, made by
`notebooks/generate_init_states.py` from the perturbed BDDL).

## How each split perturbs a task, and what it is in our framing

| Split (suite suffix) | What changes (read from `perturbation.py` and the shipped BDDLs) | Held-concept structure | Our class |
|---|---|---|---|
| Task (`_task`) | `TaskPerturbator`: the `(:language)`, `(:goal)` and `(:obj_of_interest)` are replaced by another task in the **same scene** (`libero_ood/ood_task.yaml`); objects and layout unchanged | Goal: another object to the same targets, or the same object to another target (`put the wine bottle on the plate`). Spatial: the *other* bowl, named by a location phrase a different Spatial task taught (`pick the akita black bowl on the stove ...` in the next-to-cookie-box layout; every phrase is true of bowl 2 there, checked against the BDDL init). Object: another object of the scene into the basket (`pick the cream cheese ...` in the alphabet-soup scene). 10: two-object or two-step tasks with swapped objects | **Compositional** (held object or instance x held goal, recombined in a known scene) |
| Position (`_swap`) | `SwapPerturbator`: two objects or fixtures of interest exchange their init regions (e.g. plate and cookie box in Spatial; cabinet and wine rack in Goal); instruction and goal unchanged | The instruction pairs the same object and goal the policy was trained on; only geometry moves (in Spatial the plate leaves the only region it ever occupied in training) | Not compositional: a layout shift of a held task |
| Object (`_object`) | `ObjectReplacePerturbator`: an object of interest swapped for a recoloured or rescaled variant (`yellow_bowl`, `red_cream_cheese`, `green_ketchup`, new meshes in `envs/objects/self_designed_object.py`); language kept | Same concepts, new appearance | Not compositional: visual shift |
| Semantic (`_lan`) | `LanguagePerturbator`: the instruction paraphrased (`libero_ood/ood_language.yaml`); goal unchanged | Same concepts, new words | Not compositional: lexical shift |
| Environment (`_env`) | `EnvironmentReplacePerturbator`: the table replaced (hard-coded to `living_room_table`); not shipped on the HF dataset | Same task, new surroundings | Not compositional: environment shift |
| Seven robustness cases (`01_visual_noise_glare` ... `07_initial_pose_position_angle`, HF dataset) | Glare and noise, camera angle, object moved at run time, texture, occlusion, shape scale, initial pose | Same task | Not compositional: visual or physical shift |

Only the task split matches the system's premise: every new task names concepts that training tasks taught, in a new pairing,
with no demonstration of the test task. The others change what the policy sees or where things are, under the instruction that
taught the task; none recombines concepts, so the system's operators (which read the policy through another training
instruction) have nothing to recombine there. G22 tests the task split only.

## Integration (as Li's suites are integrated)

- Suites `libero_goal_task`, `libero_spatial_task`, `libero_object_task`, `libero_10_task` run in LIBERO-PRO's own LIBERO
  package (its registry, BDDLs and init states): `experiments/libero_client.py` and `experiments/fresh_states.py` put the clone on
  `sys.path` and set `LIBERO_CONFIG_PATH=/coc/flash10/ksingara3/robot/libero_pro_config` when a PRO suite is named, as they do
  for Li's `modified_libero`. For the task split's objects LIBERO-PRO's environment code equals openpi's LIBERO (its diffs add
  objects and replace a `collections.Iterable` check).
- Max steps: LIBERO-PRO's (Goal 300, Spatial 220, Object 280, 10 520).
- **The instruction.** LIBERO-PRO keeps the original file names, and LIBERO's `Task.language` is read from the file name, so
  the benchmark's task object carries the OLD instruction; the paper defines the task perturbation as (l, e, G(T)), the old
  instruction with a new success predicate. A policy given the old instruction is asked to do the old task and scored on the
  new one. Our client sends the BDDL's new `(:language ...)`, lower-cased as every LIBERO prompt
  (`factors.pro_language` on the server side, `libero_client.pro_language` on the client side: the same text); the arm
  `origprompt` gives the policy the old instruction, the protocol of LIBERO-PRO's published table.
- Task map and factors: `lifelong_ood/policy/factors.py` (`PRO_SUITES`, `task_map()`, `parse()`); the server's `prims:` mode
  loads the PRO suites (`experiments/serve_guided.py`).
- Start states: LIBERO-PRO's own 50 init states per task (`.pruned_init`, comparable with the published numbers), and fresh
  states from LIBERO's placement sampler on LIBERO-PRO's BDDLs: set a (seeds 300000+, 10 per task; practice) in
  `results/robot/fresh_pro_a/`, set t (seeds 900000+, 20 per task; test) in `results/robot/fresh_pro_t/`.
- The HF init files are pickles; a static scan of their opcodes (no unpickling) found only numpy and `_codecs` globals before
  any job loaded them.

## Which tasks the system can read

The iteration-12 system composes one object concept and one goal concept around one grasp. A task is eligible when its
instruction is one step with the verb put (or "pick ... and place it"), a preposition, and its BDDL goal is one In/On atom
(`experiments/g22_libero_pro.py eligible`). 27 of the 40 task-split tasks are eligible:

- Goal: 6 of 10 (`put the plate on the stove`, `put the wine bottle in the bowl` twice, `put the plate on the top of the drawer`,
  `put the wine bottle on the plate`, `put the cream cheese on the rack`). Not eligible: `open the bottom drawer` (no grasp;
  the bottom drawer is a new value), `open the top layer ... and put the cream cheese inside` (two steps), `push the cream cheese
  ...` (a push), `turn off the stove` (an operation no training task performs). The plate as a grasped object is held by no
  training task, so two eligible Goal tasks have only a held goal.
- Spatial: 10 of 10. One phrase is new (`not between the plate and the ramekin`); the rest name a held location.
- Object: 10 of 10.
- 10: 1 of 10 (`pick up the cup and place it in the back compartment of the caddy`; the cup is named by no training
  instruction). The other nine move two objects, add a second step, or score a goal the instruction omits (`put the left moka
  pot on the stove` also requires the stove on). They run the policy alone only.

Source pools: G10's (the 16 single-step pick-and-place training instructions of LIBERO-Goal and LIBERO-Spatial) for the Goal and
Spatial splits, unchanged from Li's suites, which are built on the same two suites; for the Object and 10 splits, the same rule
over that suite's own training instructions (10 and 5), because G10's pool holds no instruction from their scenes.

## Published numbers (README leaderboard at eafdb80; 50 episodes per task, LIBERO-PRO's init states)

| Policy | Goal | Spatial | Object | 10 | (unperturbed: Goal / Spatial / Object / 10) |
|---|---|---|---|---|---|
| pi0.5, Task | 0.00 | 0.01 | 0.01 | 0.01 | 0.97 / 0.96 / 0.98 / 0.93 |
| pi0, Task | 0.00 | 0.00 | 0.00 | 0.00 | 0.92 / 0.90 / 0.98 / 0.82 |

These use the old instruction (above); G22's `origprompt` arm reproduces that protocol on the same init states, and the policy
alone under the new instruction is the baseline the system is tested against.
