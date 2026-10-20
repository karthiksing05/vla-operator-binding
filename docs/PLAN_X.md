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
| X3 | **EPIC-KITCHENS-100** | none was trained on it. Precedent: EggHand fine-tunes GR00T's flow-matching decoder for egocentric hand forecasting | verb × noun pairs held out by grade; EPIC's unseen-kitchens split as the scene grade | offline: hand-trajectory forecasting error and flow likelihood (no closed loop) | no 3D hand poses (2D hand tracks only); we fine-tune GR00T's action head ourselves |

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
