# Plan X: the framework beyond LIBERO (user direction, 2026-10-19)

The user's criteria:
- datasets widely reputed in robot learning;
- datasets on which the models we already benchmark are known to work well.

The user chose three. Each one must give a compositional split graded by unseen parts and relations (docs/SPLIT.md). Each one
also runs both schemes: supervised (concepts read from the instruction) and unsupervised (contact-cut units named by likelihood,
with donation analysis).

## Order (best fit to our models first)

| # | dataset | our models with public checkpoints there | split | evaluation | main risk |
|---|---|---|---|---|---|
| X1 | **CALVIN** (ABC → D) | FLOWER (`mbreuss/flower_calvin_abc`, its home benchmark); X-VLA (`2toINF/X-VLA-Calvin-ABC_D`, reported 4.43) | scene D unseen (G3); 34 subtasks recombined in 5-step chains (relations); chains are the natural test of the unsupervised scheme's segments | closed loop in CALVIN's PyBullet environment, the standard 1,000 chains | installing calvin_env; FLOWER's CALVIN code is already at robot/ext/flower_vla_calvin |
| X2 | **Bridge V2 / Open X-Embodiment via SimplerEnv** | X-VLA (`2toINF/X-VLA-WidowX`, reported 95.8%; `X-VLA-Google-Robot`); pi0 (open-pi-zero's Bridge / Fractal checkpoints) | SimplerEnv's WidowX tasks (spoon→towel, carrot→plate, green block→yellow block, eggplant→basket) recombined into unseen object × target pairs, built as Li built LIBERO's OOD suites | closed loop in SimplerEnv (SAPIEN) | needs NVIDIA Vulkan on the GPU nodes (the login node has none); only 4 source tasks, so the split is small |
| X3 | **EgoDex** (Apple, ICLR 2026; replaces EPIC-KITCHENS by user decision 2026-10-20: 3D motion required) | 829 h of 30 Hz egocentric video with SE(3) poses of head, upper body, both wrists and 25 joints per hand, 194 tabletop tasks with language. Our models have no EgoDex checkpoint, but recipes exist for our families: HuRo trains a GR00T-N1.6-architecture VLA on robotized human video (about 56% EgoDex) with relative-wrist end-effector actions plus absolute hand-joint targets; EgoScale / GR00T N1.7 pretrain the GR00T family on human hand motion with a learned placeholder for proprioception; Ego-Pi fine-tunes pi0.5 on egocentric human demos; H-RDT (a diffusion transformer) and Being-H0 ship open EgoDex preprocessing | tasks are verb x object compositions; hold out verb x object pairs by grade (relation: both seen, never paired; part: an object or verb never seen) | offline: EgoDex's own best-of-K trajectory metrics (compute_metrics.py) and the flow likelihood | dataset is CC-BY-NC-ND (research use; no redistribution of derived data); 1.5 TB training set, so extract only the task folders the split needs (the CDN serves byte ranges) |
| X4 | **Meta-World** (MT50 / ML45; added by user request 2026-10-21; docs/METAWORLD.md) | pi0 and pi0.5 (`RLinf/RLinf-Pi0-MetaWorld-SFT`, published MT50 avg 50.8; `RLinf/RLinf-Pi05-MetaWorld-SFT`, 43.8; RL fine-tunes 66-86); SmolVLA (`lerobot/smolvla_metaworld`, `jadechoghari/smolvla_metaworld`; paper avg 57.3, checkpoint unnamed). X-VLA, GR00T N1.5 and FLOWER have none | 50 verb x object tasks in one scene. Every public checkpoint was trained on all 50 (all G0), so a graded read needs our own fine-tune: ML45 (4 G4 + 1 G1) or x4_35 (35 train; 11 G1 + 4 G4) | closed loop in MuJoCo (EGL), 10 trials per task, RLinf's standalone loop (160 steps, chunk 5, 5 flow steps) | the single scene leaves no G2 / G3; the demonstrations predate metaworld 3.0.0, and push / push-back shared one instruction until 2026-09-24 |

## What carries over unchanged

- The backbone interface (`lifelong_ood/policy/backbone*.py`): FLOWER and X-VLA wrappers exist. They need each benchmark's own
  checkpoint, action space and observation conventions.
- Units, pointers and practice; the grasp switch where a gripper exists (all three); pointer programs and the soft blend.
- Pre-registration, paired episodes, the audits.

## First steps

1. A Vulkan probe on a GPU node: done 2026-10-19. The A40 nodes carry the NVIDIA Vulkan ICD and libvulkan, so X2 is feasible (the login node has neither).
2. X1: install CALVIN in its own environment, then wrap FLOWER-CALVIN and X-VLA-CALVIN behind the backbone interface. Reproduce each
   checkpoint's published CALVIN score before any composition read: the parity gate, as H4 did for GR00T and FLOWER.
3. X1 split and pre-registration: grade CALVIN's subtasks and chains by what scene D and each chain leave unseen. Supervised units
   are CALVIN's 34 subtasks as held fields; unsupervised units are cut at contact events and named by likelihood.
