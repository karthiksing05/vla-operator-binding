# EgoDex (X3): inventory, graded split, contact events, recipe, evaluation (stage 1)

Plan X3 (docs/PLAN_X.md) moves the framework to EgoDex: 30 Hz egocentric video of human hands, with SE(3) poses of the head,
upper body, wrists and 25 joints per hand, and a language description per episode. This page records stage 1: what the data
holds, the graded split, the contact events for the unsupervised scheme, the recipe for GR00T N1.5, and the evaluation plan.
Nothing is trained yet, and nothing is committed.

Licence: CC-BY-NC-ND, research use only. Derived data (poses, frames, metadata extracts) stays under `results/robot/egodex/`,
which is untracked, and is never redistributed.

## 1. Inventory

### Test set (unzipped, `/coc/flash10/ksingara3/robot/data/egodex/test/`)

`experiments/x3_egodex_inventory.py` reads hdf5 headers only and writes `results/robot/egodex/test_inventory.json`.

- 111 task folders, 3,243 episodes, 7.65 h (229 min). All paired hdf5/mp4 files open.
- Episodes per task: min 1, median 20, max 277 (basic_pick_place). 24 tasks have fewer than 10.
- Mean episode length per task: 2.3 s (stack) to 79 s (set_up_clean_up_chessboard). Short tasks are reversible pairs cut into
  halves.
- `llm_type` per episode: reversible 2,240, reset 619, reset-free 384.
- 26 episodes have the description "Description unavailable." (insert_remove_bookshelf mostly).
- Per-episode attributes beyond the README: `llm_verbs`, `llm_objects`, `object` (for example `object:stapler, color:black`),
  `environment` (`table`, `background`, `position`; pick-place tasks also have `from:`, `to:`, `hand:`), `task`,
  `session_name`, `extra` (C1-C9) and `annotated`.
- Poses: 69 transforms and 68 confidence channels. Every transform is in the ARKit origin frame; the camera transform follows
  the OpenCV convention, so after `inv(T_camera) @ T` the hands sit at z ≈ +0.2 to +0.4 m.
- `camera/intrinsic` is fixed (f = 736.6, principal point 960, 540), and the video is 1920x1080.

Per-task episode counts, durations and the most common descriptions are in `test_inventory.json` (`tasks.<name>.descs`).

### Training set (CDN zips, read by range)

`experiments/x3_egodex_zipdir.py` reads only each zip's central directory over HTTP range requests: 92 MB in all, out of
1.86 TB. It writes `train_zip_tasks.json` (per task: zip, episodes, bytes, byte span) and `train_zip_files.json` (per-file
offsets, for the fetcher).

| zip | size | task folders | members |
|---|---|---|---|
| part1 | 321.6 GB | 26 | 92,495 |
| part2 | 327.3 GB | 13 | 190,264 |
| part3 | 326.3 GB | 24 | 107,583 |
| part4 | 329.4 GB | 23 | 88,282 |
| part5 | 331.3 GB | 25 | 151,170 |
| extra | 223.9 GB | 18 | 40,323 |

- **There are 118 folders, not 194.** The 194 tasks of the paper count each reversible folder as two tasks:
  76 reversible pairs give 152, plus 28 reset-free and 14 reset tasks (EgoDex appendix A.2).
- part1-5 hold exactly the 111 test folders: 314,839 episodes. extra.zip holds the 7 folders that are not in the test set
  (assemble_disassemble_jigsaw_puzzle, blowdry_hair, knit_scarf, makeup, play_reversi, stack_unstack_tetra_board,
  stamp_paper), plus 10,917 more episodes of 11 main folders: 20,152 episodes in all.
- **The test set is in-distribution.** It holds 1% of each folder's episodes, from the same sessions as the training episodes.
  A compositional split must therefore hold folders (or units within a folder) out of OUR training subset.
- Every member is deflated (method 8). hdf5: 285 GB compressed for all 334,991 episodes; mp4: 1,574 GB.
- The root attributes of a deflated hdf5 cannot be read from a prefix. h5py fails even at 99% of the file, so metadata needs
  the whole hdf5 member, about 0.85 MB on average.

Per-folder training episodes and bytes are in `task_units.json` (`train_episodes`, `train_gb`).
`experiments/x3_egodex_trainmeta.py` fetched the whole hdf5 of 66,189 training episodes (about 33 GB transferred in 22 min;
nothing stored but the attributes). It took every episode of basic_pick_place, vertical_pick_place, stack and
pick_place_food, and 80 evenly spaced episodes of every other folder. Output: `train_meta.json`.

### Units per task (`experiments/x3_egodex_split.py` → `task_units.json`)

- **Verbs** come from the folder name by a fixed lexicon. Phrasal verbs (`set_up`, `put_away`, `take_out`, `put_in`, `pick_up`)
  are joined, `un`-verbs stay separate (each is a task in the 194), and a small synonym table applies (put_down → place,
  unstake → unstack). The name verbs match the episodes' `llm_verbs` (at least 30% of sampled episodes) exactly in 102 of
  118 folders.
  - In 14 folders `llm_verbs` adds verbs:
    - sub-actions: pick/place in stack_unstack_tupperware, sleeve_unsleeve_cards, play_reversi and stamp_paper;
      fold/unfold/remove in setup_cleanup_table; collect/distribute in play_mancala;
    - synonyms: press for push_pop_toy; put/take for put_in_take_out_glasses;
    - an LLM error: "bagging" in insert_remove_bagging.
  - In 2 more, a name verb never appears in `llm_verbs`: dump (insert_dump_blocks) and smooth (rake_smooth_zen_garden).
- **Objects** are the name's remaining tokens, normalised to a head noun in the singular. Compounds such as `tennis_ball`,
  `bottle_cap`, `square_table`, `rubiks_cube` and `allen_fixture` are kept whole.
- A training folder exposes its name objects plus every `llm_objects` head noun found in at least 30% of its sampled episodes,
  scene nouns (table, tablecloth, background, hand) excluded.
- A held-out folder is graded on its primary objects: the name objects, or, when the name has none, the top `llm_objects`
  (at least 50% of episodes).

Ambiguities, noted rather than resolved:
- Activity names carry no object: color, write, type_keyboard, play_piano, dry_hands, basic_fold, pour, wrap, stack,
  basic_pick_place, vertical_pick_place. Their object comes from the descriptions: tshirt, bowl, plush and so on.
- `use_chopsticks` and `use_rubiks_cube` name a tool; the manipulation verb is implicit.
- FurnitureBench folders: the object is a furniture piece. A "drawer" piece is not the drawer of insert_remove_drawer.
- `llm_objects` is noisy. It lists containers and targets ("lightning" for charge_uncharge_airpods, "bread" for
  open_close_insert_remove_tupperware), and some names are fused or misspelled (graycup, tupperwear, puzzle1).

## 2. The graded split

### How "seen" is defined

"Seen" is defined at the task level, from text, against the TRAINING folders (all part1-5 folders minus the held-out and
excluded ones):
- a verb or object is seen when a training folder's units carry it;
- a verb x object pair is seen when one training folder carries both.

Grades follow docs/SPLIT.md:
- **G1 relation**: every verb and primary object seen, the pair never.
- **G4 part**: a verb or primary object never seen.
- **G2 binding** exists only inside basic_pick_place (below). There, the object is seen in another training folder's text but
  never in a pick-place episode.
- **G3 scene** has no analogue: EgoDex has no fixed scenes, since backgrounds and tablecloths vary per session.

A second, stricter check reads the instructions themselves (regex over llm_description / llm_description2 of the 66,189
sampled training episodes). The model reads instructions, not folder names, so a held-out unit must not appear in any
training instruction either. This check changed two decisions (marked † below).

### Held-out folders (their test episodes are the evaluation; their training episodes are never fetched)

| grade | folder | test eps | test min | unseen part / donors of the seen parts |
|---|---|---|---|---|
| G1 | charge_uncharge_airpods | 58 | 2.5 | charge ← charge_uncharge_device; airpods ← insert_remove_airpods |
| G1 | open_close_insert_remove_tupperware | 41 | 4.9 | open/close ← _box, _case; insert/remove ← 15 folders; tupperware ← stack_unstack_tupperware |
| G1 | stack_unstack_bowls | 36 | 3.3 | stack/unstack ← stack_unstack_plates, stack, ...; bowl ← basic_pick_place (target), boil_serve_egg, ... |
| G1 | stack_unstack_cups | 12 | 0.9 | stack/unstack ← as above; cup ← clean_cups, add_remove_lid, insert_remove_cups_from_rack |
| G4 | pour | 33 | 2.1 | verb pour |
| G4 | lock_unlock_key | 21 | 2.2 | verbs lock/unlock, object key |
| G4 | zip_unzip_bag, zip_unzip_case | 35 + 24 | 3.2 + 5.4 | verbs zip/unzip (bag, case seen) |
| G4 | screw_unscrew_allen_fixture, _bottle_cap, _fingers_fixture | 12 + 29 + 17 | 5.2 + 2.2 + 4.5 | verbs screw/unscrew; the fixtures and the cap |
| G4 | tie_and_untie_shoelace, tie_untie_rubberband | 26 + 42 | 4.5 + 4.8 | verbs tie/untie; shoelace, rubber band |
| G4 | throw_and_catch_ball, throw_collect_objects | 22 + 21 | 4.0 + 4.7 | verbs throw/catch/collect (ball, box, towel seen) |

- Folder-level totals: G1 147 test episodes (11.6 min); G4 282 test episodes (42.8 min).
- The instruction check confirms each one: none of pour, zip, tie, throw, shoelace or rubber band appears in any training
  instruction, and no G1 pair co-occurs in one.
- "lock" appears in 41 basic_pick_place instructions, but as a noun (the object picked up). The verb stays unseen.

Excluded from training, as well as held out:
- fry_bread, fry_egg, wash_fruit, wash_kitchen_dishes: 7-61 training episodes, long multi-step tasks. fry_egg's instructions
  also say "pour".
- **† put_toothpaste_on_toothbrush.** All 80 sampled instructions say "unscrew the toothpaste cap", which would make screw seen.
- **† pick_up_and_put_down_case_or_bag.** It was planned as G1, but "pick ... case" co-occurs in 946 basic_pick_place
  instructions and in all of put_in_take_out_glasses, so it is not a clean G1.

### Inside basic_pick_place (episode level; units = moved object x target container)

The units come from the metadata fields `object:<o>` and `environment ... to:<target>`. 28,644 training episodes give
263 (object, target) pairs over 131 objects and 25 targets. The top targets are bowl 11,449, plate 5,405, tupperware 2,834
and tray 1,661. All 277 test episodes are G0 as shipped. Grades are constructed by dropping training episodes
(`bpp_split.json`):

- **G1 pairs held out**: (block, bowl) 33, (plush, plate) 8, (egg, bowl) 7, (duck, plate) 5, (die, bowl) 5,
  (donut, tupperware) 4.
  - 62 test episodes in all; 4,702 basic_pick_place training episodes dropped.
  - Each held object keeps 281-1,949 training episodes with other targets, and each target 2,647-11,000 with other objects.
- **G2 objects held out of pick-place**, but named in other training folders' instructions:
  - cup 6 (clean_cups, ...), banana 4 (setup_cleanup_table), croissant 4 (wrap_unwrap_food), iphone 4 (measure_objects;
    "phone" in charge_uncharge_device), stapler 2 (staple_paper).
  - 20 test episodes; 1,630 training episodes dropped.
- **G4 object**: mouse, 5 test episodes. Outside pick-place it appears only in zip_unzip_bag, which is held out.

This within-task split is the cleanest test of composition in EgoDex. The verb, scene and hand stay fixed, and only the
binding of object to target changes. It mirrors Li et al.'s LIBERO recombinations.

### What to fetch (the training subset)

90 training folders. Evenly spaced episodes are taken per folder, each by its own byte range (hdf5 + mp4): 150 per folder by
default, 500 for the 10 folders that donate G1 units, and 3,000 for basic_pick_place, taken among the episodes outside the
held-out pairs and objects. Totals: **19,828 episodes, 161.6 GB** (`split_proposal.json`, `plan`).

| cap per folder (all 90) | episodes | GB |
|---|---|---|
| 50 | 4,500 | 45.4 |
| 100 | 9,000 | 91.2 |
| 150 | 13,478 | 136.3 |
| plan (150 / 500 donors / 3,000 bpp) | 19,828 | 161.6 |
| 500 | 43,690 | 421.3 |
| all | 269,957 | 1,414.7 |
| hdf5 only, all episodes (poses + text, no video) | 269,957 | 215.8 |

Every fetch reads members by range from `train_zip_files.json` offsets. The local header is read first (30 bytes plus name and
extra), then the raw-deflate stream. Fetches run inside SLURM jobs into a new directory under `robot/data/egodex/`.

## 3. Contact events (for the unsupervised scheme)

`experiments/x3_egodex_contacts.py` runs on all 3,243 test episodes and writes `test_contacts.json` and
`test_contacts_summary.json`.

**Aperture thresholds fail.** The first definition was hysteresis on the thumb-to-index/middle fingertip aperture, with fixed
(4.5 / 6.5 cm) or per-episode adaptive thresholds. It finds exactly one grasp and one release on the named hand in only 13-22%
of basic_pick_place episodes. Two causes, both seen in the traces:
- fingertip confidences drop to 0.2-0.4 exactly while an object is held (occlusion);
- the relaxed hand's aperture (3-8 cm) is about the width of most grasped objects.

**The chosen definition is wrist stops.** A contact event of a hand is an interior stop of its wrist: the reach → grasp →
transport → release → retract kinematics.
- Signal: wrist speed (30 Hz), Gaussian-smoothed (σ = 2 frames).
- Stop: a local minimum at or below 0.6x the lower of its two flanking maxima, with that maximum at least 0.2 m/s
  (scipy prominence).
- Labels: stops alternate grasp, release, grasp, ...
- Independent check: the aperture trend across each stop (closing should mean grasp).
- Tuning: the parameters were set on "dev" episodes (the first 60 / 30 / 20 / 20 of basic_pick_place / vertical_pick_place /
  stack / pick_place_food) and are reported on the rest ("held").

| task (held episodes) | n | exactly 2 stops (grasp + release) | mode | other hand silent | aperture agrees with label |
|---|---|---|---|---|---|
| basic_pick_place | 217 | 63% (dev 65%) | 2 | 99% | 70% |
| vertical_pick_place | 123 | 73% (dev 90%) | 2 | 96% | 70% |
| stack | 60 | 77% (dev 65%) | 2 | 100% | 76% |
| pick_place_food | 29 | 62% (dev 45%) | 2 | 93% | 49% |

Negative controls, mean stops per episode:

| task | mean stops |
|---|---|
| play_piano | 0.03 |
| push_pop_toy | 0.03 |
| roll_ball | 0.0 |
| use_rubiks_cube | 0.0 |
| type_keyboard | 1.0 |

Multi-object tasks scale with the object count: stack_unstack_cups 5.1 (three cups), add_remove_lid 7.3 (up to four lids).
Long tasks reach 20-34 (chessboard, staple_paper).

Where it fails:
- 10-22% of pick-place episodes show one stop. In the dev episodes most of these end while the hand is still moving
  (7 of 10 in basic_pick_place, 6 of 6 in stack), so the release falls outside the clip.
- stack_unstack_plates shows zero stops in 38% of episodes: slow slides, with no clear stop.
- charge_uncharge_airpods shows 0.3 per episode: 2.6 s clips that start mid-manipulation.

These are kinematic proxies for contact, not detected contact. For the unsupervised scheme they serve as segment cuts (as the
gripper events do on LIBERO). The supervised scheme's switch uses the first stop of the named hand as the grasp.

## 4. Recipe for our models (GR00T N1.5 first)

### What the published recipes do

| source | action / state for human hands |
|---|---|
| EgoDex benchmark (paper §5, `compute_metrics.py`) | 48-D: per hand, wrist xyz + 6D rotation + 5 fingertip xyz, all in the **current camera frame**; a chunk is a relative trajectory; H = 30 / 60 / 90 at 30 Hz (main table H = 60); input = current image (224x224) + current pose + CLIP text; no history |
| H-RDT (AAAI 26) | the same 48-D (bilateral wrist 3 + 6D, 10 fingertip positions); pretrained on all 338k EgoDex trajectories; per-embodiment action encoder/decoder modules for transfer; 30 Hz |
| EgoScale / GR00T N1.7 (NVIDIA 2026) | wrist action relative to the wrist pose at the chunk's first step, ΔW_t = (W_0)^-1 W_t; hands retargeted (CasADi/IPOPT) to a 22-DoF hand; **a learnable placeholder token replaces proprioception** for human data; pretraining 100k steps, batch 8,192 on 256 GB200, then frozen-VL mid-training |
| HuRo (2609.10706, RLWRLD) | human video "robotized" (inpainted robot images, retargeted states and actions), about 56% EgoDex; per PLAN_X, relative-wrist end-effector actions + absolute hand-joint targets. The PDF exceeded the fetch limit, so the details are unverified here |
| Being-H0 | MANO-based part-level motion tokens (GRVQ); EgoDex's 25 joints mapped to MANO's 21 by dropping 4. Not used |

### The representation for our models (implemented in `experiments/x3_egodex_data.py`)

Per hand h ∈ {left, right}, everything in the camera frame C_t of the observation (p^C = inv(T_cam[t]) p^W, the official
`convert_to_camera_frame`):

- **action chunk a_t**, [H, 48]: for k = 1..H, the frame t + s·k. Each frame holds, per hand, wrist xyz (3), wrist 6D
  rotation (6: the first two columns of R), and five fingertips (15).
  - Fingertips default to camera-frame xyz, the official layout `compute_metrics` indexes.
  - With `tips_in_wrist=True` they are expressed in that step's own wrist frame. This is the analogue of HuRo's absolute
    hand-joint targets: hand shape decoupled from arm motion, the same quantity for every embodiment. `to_camera48` inverts
    it exactly (checked to 1e-4).
- **state s_t** (48-D): the same frame at t. GR00T pads it to 64.
- **image**: frame t of the mp4, resized to 224x224 (Eagle's input), decoded by PyAV seek + forward decode.
- **instruction**: llm_description, or llm_description2 when reversible and which_llm_description == "2" (as the official
  loader does).
- **pad and conf**: per-step padding and the 12 keypoints' confidences, for masking occluded targets.

Recommendation: H = 16 at stride s = 2. That is 15 Hz over 1.07 s, the pretrained N1.5 action horizon unchanged, covering
EgoDex's H = 30 window. Predictions are linearly upsampled to 30 frames for the official metric; they are exact at the even
frames. Ablation: H = 32, s = 1. The position embedding table holds 1,024 entries, so the change is a config value, but
positions 16-31 were never trained.

Loader smoke test (SLURM, GR00T venv, 3 test episodes, frames decoded only at the sampled steps):
- 13 samples in 0.9 s;
- rot6d orthonormal;
- consistent with the official camera-frame conversion;
- wrist-frame fingertips 11.7-13.6 cm from the wrist.
The GR00T venv lacks h5py, so h5py 3.12.1 sits in `/coc/flash10/ksingara3/robot/ext/egodex_pylibs` (pip `--target`), appended
to `sys.path` only when the import fails.

### Fine-tuning GR00T N1.5 on the EgoDex training subset (plan; not run)

- **Start from `nvidia/GR00T-N1.5-3B`**, not ar0s/groot_libero: that checkpoint's projector and DiT were tuned to LIBERO's
  7-D OSC deltas.
- **Embodiment**: `new_embodiment` (id 31). The state encoder, action encoder and action decoder are CategorySpecific
  per embodiment, so EgoDex gets its own slot. A later LIBERO / robot slot keeps its own.
- **Action dimension, the one architectural change**: `max_action_dim` is 32 and our action is 48. Raise it to 64 and
  re-initialise the action encoder/decoder. Their input/output widths change, and the new-embodiment rows are untrained in
  the base model anyway. The DiT, the VL backbone and the vlln/self-attention are loaded unchanged.
  - Alternative (no architecture change): a 32-D action, per hand wrist 9 + 7 finger scalars (aperture, five tip-to-wrist
    distances, spread).
  - That alternative cannot be scored by the official fingertip metric, so it is not recommended.
- **State**: the current 48-D pose (EgoDex protocol). For later human-to-robot transfer, also train with p = 0.3 state
  dropout to a learned placeholder (EgoScale), so the head works without proprioception.
- **Normalisation**: per-dimension quantiles (1st/99th percentile → [-1, 1], clipped) computed on the fetched training
  subset. Plain min-max (LeRobot's GR00T default) is too sensitive to ARKit tracking glitches. 6D rotation components are left
  unnormalised; they already lie in [-1, 1].
- **Loss masking**: drop steps whose keypoint confidence is 0, and chunks with padding (as the metric does).
- **Tuning**: LLM and vision frozen; projector + DiT + new embodiment heads tuned (as ar0s/groot_libero did); lr 1e-4,
  warm-up 5%, bf16; batch 32 on one A40.
- **Data**: pre-extract 224x224 JPEG frames at 15 Hz, about 12 KB each. The plan's 19,828 episodes come to about 72 h of
  video by the test set's per-task mean durations: 3.9M frames, about 47 GB. Decoding 1080p H.264 in the training loop would
  starve the GPU.
- **Budget (an estimate, to be measured by a 200-step timing job first)**: frozen-Eagle forward plus DiT backward at batch
  32 is about 0.5-0.8 s/step on an A40. 30k steps (about 1M samples, a quarter pass over the 3.9M frames) comes to about
  5-7 GPU-hours, plus about 4-8 CPU-hours of frame extraction (2-4 parallel jobs).
- **Faster option**: cache the frozen Eagle features once (docs/FIT_TIME.md's prefix cache), about 0.3 MB per frame in bf16,
  about 1.2 TB for 3.9M frames. Too large; cache only a subset or skip it.
- **Parity gate before any composition read**: the fine-tuned model's best-of-K on the G0 test episodes (the 90 training
  folders' test episodes) should approach the EgoDex paper's EncDec + FM numbers at H = 30: K = 10, 2.6 cm average and
  3.3 cm final. Those were trained on all 829 h; a gap is expected and must be reported.

The same loader serves pi0.5 / pi0 / SmolVLA / X-VLA (Ego-Pi fine-tunes pi0.5 on egocentric human demonstrations). Their
padded action widths are also below 48 (pi0 / pi0.5: 32), so they face the same head change; GR00T goes first.

## 5. Evaluation plan

All of it is offline, on test episodes. There is no simulator.

- **Metric**: EgoDex's best-of-K (`compute_metrics.evaluate_distance`), K = 1 / 5 / 10. It reports the average and final
  Euclidean distance (m) over the 12 points (2 wrists + 10 fingertips), on chunks without padding, at every frame of each
  test episode.
  - Report per grade (G0 control, G1, G2-bpp, G4) and per folder.
  - Units: paired per episode (`lifelong_ood/stats.py`: paired sign tests, two-level bootstrap over episodes within tasks).
  - Second read: the flow likelihood of the ground-truth chunk under each conditioning (the per-chunk flow-matching loss
    averaged over a fixed grid of t and noise seeds, as `zu_score`).
- **Arms, per held-out task**:
  1. the policy alone, with the held-out instruction;
  2. supervised composition with nothing fitted;
  3. unsupervised composition with nothing fitted;
  4. created units: the one-demonstration fit;
  5. oracle: the policy fine-tuned with the held-out folder included. An upper bound only, costing one extra fine-tune.
- **Supervised scheme (units from the instruction)**:
  - Each held-out instruction is parsed into its verb and object units.
  - Each unit points to the policy's field under a training instruction that carries it, the donor folders in §2. For
    stack_unstack_cups: object "cups" ← "Insert four cups into a dish rack ..."; verb "stack" ← "Stack four plates ...".
  - The prompt switches at the named hand's first wrist stop, its grasp event: object unit for the reach, verb/target unit
    after the grasp.
  - Offline, the switch time comes from the ground-truth hand (the event detector of §3); closed loop it would come from the
    policy's own predicted hand.
  - Within basic_pick_place the units are object and target: "pick up the duck ... place it in the plate" composes the duck
    unit (reach, grasp) with the plate unit (transport, release).
- **Unsupervised scheme (units from the demonstration)**:
  - One test episode of the held-out task is the demonstration (the one-demonstration rule); the others are evaluated.
  - The demonstration is cut at its wrist stops, and each segment is named by the training instruction (from the 90
    folders' instruction library) under which the policy gives its chunks the highest flow likelihood.
  - The resulting pointer program runs on the other episodes, advanced at each episode's own stops.
  - A segment whose best likelihood falls below the held-unit calibration (G0 episodes) is "new": a rank-4 unit is created
    for it, fitted on that one demonstration (c0_fit), as on LIBERO.
- **What the grade predicts (to pre-register)**:
  - G1 (and the basic_pick_place pairs): composing held units should close most of the gap between the policy alone and the
    oracle, with nothing fitted.
  - G4 (pour, zip, screw, tie, throw, lock): composing held units should not help; created units should.
  - That is the LIBERO finding (docs/SPLIT.md), now tested on human hand motion.

## Decisions for the user

1. **The fetch size**: 161.6 GB (plan above) vs 45-91 GB (cap 50-100) vs pose-only hdf5 for every training episode,
   215.8 GB with no video, useful for action statistics and the unsupervised scheme's segment library. Also: where to store
   it (the flash10 quota is unknown here).
2. **put_toothpaste_on_toothbrush**: exclude it from training, as the proposal does, so screw stays G4. The alternative is to
   keep it and regrade the three screw folders as G1 (verb seen in "unscrew the toothpaste cap").
3. **Action dimension**: raise GR00T's `max_action_dim` to 64 and re-initialise the action encoder/decoder (recommended), or
   use a 32-D compact hand that the official metric cannot score.
4. **Fingertip frame**: camera frame (the official metric's layout) or wrist frame (embodiment-neutral; converted back
   exactly for scoring).
5. **Horizon**: H = 16 at 15 Hz (pretrained horizon; recommended) vs H = 32 at 30 Hz.
6. **G2/G3**: only basic_pick_place gives a G2 (20 + 5 episodes), and G3 has no EgoDex analogue. Is a G1/G4 split plus the
   within-pick-place G1/G2/G4 enough for the paper's grade table?
7. **The 4 G1 folders are few**, 147 episodes. More G1 candidates exist only by holding out further donor-adjacent folders,
   for example insert_remove_drawer or charge_uncharge_device's partner. Each removes a donor, so the trade needs a choice.

## Files

| file | what |
|---|---|
| `experiments/x3_egodex_inventory.py` | test-set inventory (SLURM) |
| `experiments/x3_egodex_zipdir.py` | training zips' central directories by HTTP range (SLURM) |
| `experiments/x3_egodex_trainmeta.py` | training-episode attributes by HTTP range (SLURM) |
| `experiments/x3_egodex_split.py` | units per folder, the graded split, the basic_pick_place split, the fetch plan (SLURM; JSON only) |
| `experiments/x3_egodex_contacts.py` | grasp/release events: wrist stops (chosen) and the aperture variants (SLURM) |
| `experiments/x3_egodex_data.py` | the loader: (image, instruction, state, action chunk, pad, conf); `--smoke` |
| `results/robot/egodex/` (untracked) | test_inventory, train_zip_tasks / _files, train_meta, task_units, split_proposal, bpp_split, test_contacts(_summary), logs/ |
| `/coc/flash10/ksingara3/robot/ext/ml-egodex_src/ml-egodex` | apple/ml-egodex clone (reference only; nothing of it is imported) |

Sources: [EgoDex paper](https://arxiv.org/html/2505.11709v3), [apple/ml-egodex](https://github.com/apple/ml-egodex),
[H-RDT](https://arxiv.org/html/2507.23523v1), [EgoScale](https://arxiv.org/html/2602.16710),
[HuRo](https://arxiv.org/pdf/2609.10706), [Being-H0](https://arxiv.org/pdf/2507.15597).
