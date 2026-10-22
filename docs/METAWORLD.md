# Meta-World (X4): inventory, graded split, setup and the parity gate (stage 1)

Plan X4 (docs/PLAN_X.md) adds Meta-World: 50 tabletop tasks on one Sawyer arm, each a verb applied to an object (open or close a
drawer, press a button from the top or the side, push or pick-and-place a puck, ...). This page records stage 1: which of our
policy families have public Meta-World checkpoints, the graded split, the install, and the parity gate. Nothing is committed.

## 1. Inventory (verified 2026-10-21 from the primary sources linked below)

| family | public Meta-World checkpoint | trained on | published Meta-World number | protocol |
|---|---|---|---|---|
| **pi0** | [`RLinf/RLinf-Pi0-MetaWorld-SFT`](https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT) @ d0b8fbe (7.0 GB, openpi PyTorch); RL fine-tunes [`-RL-FlowSDE`](https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE), [`-RL-FlowNoise`](https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise) | all 50 tasks: `lerobot/metaworld_mt50`, no episode filter (RLinf `pi0_metaworld` TrainConfig, `rlinf/models/embodiment/openpi/dataconfig/__init__.py`) | MT50, SFT: easy 77.9 / medium 51.8 / hard 53.3 / very hard 20.0, **avg 50.8**; RL 78.1 (Flow-SDE), 85.8 (Flow-Noise) | pi_RL ([arXiv 2510.25889](https://arxiv.org/abs/2510.25889), Table 6) and RLinf docs (`docs/source-en/rst_source/examples/embodied/metaworld.rst`): 10 trials per task; avg = mean of the 4 difficulty means |
| **pi0.5** | [`RLinf/RLinf-Pi05-MetaWorld-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT) @ 60b2d03 (7.5 GB); RL `-RL-FlowSDE`, `-RL-FlowNoise` | all 50 (`pi05_metaworld` TrainConfig, same dataset) | MT50, SFT: 68.2 / 37.3 / 41.7 / 28.0, **avg 43.8**; RL 70.7 (SDE), 66.1 (Noise) | same as pi0 |
| **SmolVLA** | [`lerobot/smolvla_metaworld`](https://huggingface.co/lerobot/smolvla_metaworld) @ 8d82bdf (0.9 GB; 5,000 steps from `smolvla_base`, LeRobot's CI smoke checkpoint); [`jadechoghari/smolvla_metaworld`](https://huggingface.co/jadechoghari/smolvla_metaworld) @ ef3089e (100,000 steps, batch 32, expert-only training on SmolVLM2-500M-Instruct) | all 50: `train_config.json` dataset `lerobot/metaworld_mt50`, `episodes: null` | paper only: SmolVLA 0.45B: 82.5 / 41.8 / 45.0 / 60.0, **avg 57.3** (0.24B 56.95, 2.25B 68.24; pi0 baselines 50.5 and 47.9 of unstated source) | SmolVLA paper ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844), Table 2): 50 demos per task, 10 trials per task, 100k steps at batch 64; it does not name which public checkpoint, if any, produced the numbers, and the cards report none |
| **X-VLA** | none | — | none: the paper's six simulation benchmarks are LIBERO, Simpler, VLABench, RoboTwin-2.0, CALVIN, NAVSIM ([arXiv 2510.10274](https://arxiv.org/abs/2510.10274)); none of the 15 `2toINF/*` checkpoints, LeRobot's `xvla-*` list ([docs](https://huggingface.co/docs/lerobot/xvla)) or the repo's `evaluation/` dirs covers Meta-World | — |
| **GR00T N1.5** | none official: Isaac-GR00T `examples/` has DROID, LIBERO, SimplerEnv, RoboCasa, SO100, ... and no Meta-World; one third-party N1.7 fine-tune (`namenu/gr00t17-lerobot-metaworld_mt50`, no card) | — | none | — |
| **FLOWER** | none: `mbreuss/*` holds CALVIN and LIBERO checkpoints only; the paper ([arXiv 2509.04996](https://arxiv.org/abs/2509.04996)) evaluates CALVIN, LIBERO, Aloha, SIMPLER | — | none | — |

Other public Meta-World checkpoints (not our families, or third-party with no card): `RLinf/RLinf-OpenVLAOFT-Metaworld-SFT`, `MINT-SJTU/Evo1_MetaWorld`, many `JunnDooChoi/slurm_*_metaworld_*` SmolVLA / pi0.5 runs, `tiantianx/pi05_metaworld`.

### Versions and protocols

- **Meta-World package.** LeRobot pins `metaworld==3.0.0` (Farama, released 2025-06-14) with `gymnasium==1.1.0` (`pyproject.toml`
  extra `metaworld`; [LeRobot Meta-World docs](https://huggingface.co/docs/lerobot/metaworld)); RLinf installs `metaworld==3.0.0`
  (`requirements/install.sh: install_metaworld_env`). Both use the **v3** task names (`assembly-v3`, ...).
- **The demonstrations.** `lerobot/metaworld_mt50` (HF dataset, 2,500 episodes = 50 per task, 204,806 frames at 80 fps; one
  480 x 480 corner2 image, a 4-D state = end-effector xyz + gripper, a 4-D action = end-effector delta xyz + gripper, and a 39-D
  environment state). It was first uploaded on 2025-02-19 (LeRobot codebase v2.0), four months before Farama's metaworld
  3.0.0, so the demonstrations were recorded with an earlier Meta-World. Until the dataset's commit b46a1c1 (2026-09-24, "Fix
  push-back-v3 task_index and instruction") it held **49 task strings**: push and push-back shared the instruction "Push the
  puck to a goal" and one task index. Every checkpoint above was trained before that fix, so for all of them push-back has no
  instruction of its own. RLinf's evaluation prompts keep the old string (`metaworld_config.json`), and LeRobot's now say "Push
  the puck back toward the robot to the goal".
- **LeRobot's protocol** (`src/lerobot/envs/metaworld.py`): `metaworld.MT1(task, seed=42)`, one fixed `train_tasks[0]` with
  `_freeze_rand_vec = False` (object and goal positions re-drawn at each reset), camera corner2 moved to (0.75, 0.075, 0.7) and
  flipped on both axes, `episode_length` 400, success = `info["success"]`; 10 episodes per task are recommended.
- **RLinf's protocol.** Two evaluators share the env construction: `gym.make("Meta-World/MT1", env_name, camera_id=2)` with the
  same camera move and flip, 15 zero-action settle steps after reset, prompt from `rlinf/envs/sim/metaworld/metaworld_config.json`,
  action chunk 5, 5 flow steps. The RL framework's evaluation (`env/metaworld_50.yaml`) uses 100 steps per episode and 500
  episodes (50 x 10). The standalone evaluator (`toolkits/standalone_eval_scripts/openpi/metaworld_eval.py`, documented as the
  one that gives the per-difficulty table) uses **160 steps**, 10 trials per task, and counts a trial as a success if the task
  ever succeeds. The papers do not say which evaluator produced 50.8.
- **Difficulty groups** (Seo et al. 2023, as SmolVLA / LeRobot / RLinf use them): easy 28, medium 11, hard 6, very hard 5.
- **MT50 / ML10 / ML45.** All published VLA numbers are MT50 (one multi-task policy, all 50 tasks, language-conditioned). ML45
  appears once: RLinf's `metaworld_45_ood` evaluates ML45's 5 test tasks at 20 trials each, after RL on the 45 train tasks, starting
  from the MT50 SFT checkpoint (`metaworld_45_ppo_openpi.yaml` loads `RLinf-Pi0-MetaWorld-SFT`), which saw all 50 tasks. pi_RL
  reports only a figure, and no numbers. No family has an ML10 result.

## 2. The graded split (`experiments/x4_metaworld_split.py`; ledger `x4_metaworld_split`; `results/x4_metaworld/split.json`)

Parts are read by rule from each v3 env name: a **verb**, an **object** (none when only the gripper moves: reach, hand-insert), a
**target** (peg, bin, hole, shelf, goal, cabinet, coffee machine, ...) and **modifiers** (wall, side, topdown, back, stick). The
factor table is the `F` dict in the script. Grades follow docs/SPLIT.md:
- **G0 seen:** the task itself is trained.
- **G1 relation:** every part is trained, never in this combination.
- **G4 part:** a verb, object, target or modifier is never trained.

Meta-World has one scene, so G2 and G3 are empty by construction. The task list and the ML45 partition are checked against the
installed metaworld 3.0.0 (`ALL_V3_ENVIRONMENTS`, `ML45`): they match.

**Against the published checkpoints every task is G0.** All six public checkpoints of our families were trained on all 50
tasks (section 1), so they offer no unseen split. A graded read needs our own fine-tune on a training subset, as on EgoDex.

Two subsets:

| split | train | held-out tasks and grades |
|---|---|---|
| **ML45** (Meta-World's) | 45 | bin-picking **G4** (target bin), box-close **G4** (object lid, target box), door-lock **G4** (verb lock), door-unlock **G4** (verb unlock), hand-insert **G1** (insert and hole both trained, in peg-insert-side and sweep-into / pick-out-of-hole) |
| **x4_35** (proposed) | 35 = ML45 train minus 10 | the 5 ML45 tasks (as above: 4 G4, 1 G1) plus 10 **G1** hold-outs, chosen so every verb, object, target and modifier stays trained: door-close, window-close, drawer-open, faucet-open, coffee-push, stick-pull (the verb and the object are each trained, never together), and handle-pull-side, plate-slide-back-side, button-press-topdown-wall, push-wall (the verb-object pair is trained and only the modifier is recombined). Totals: 11 G1, 4 G4 |

Leave-one-out (each task against the other 49): 40 tasks could be G1 and 10 are G4 even alone (basketball, bin-picking,
box-close, dial-turn, door-lock, door-unlock, hammer, lever-pull, peg-unplug-side, shelf-place), because their verb or object
occurs in no other task. This bounds any Meta-World split: at most 10 part-level (G4) tasks, and the rest are relation-level.

Caveats on the factors:
- box-close's "box" shares the word, though not the object, with stick-push / stick-pull's box. Read from language alone, box-close
  would be G4 on the lid only.
- door-lock and door-unlock use their own door-with-lock asset. Their unseen part is the verb, and the door is seen only in
  door-open / door-close.
- Meta-World's instruction is one fixed sentence per task. The relation grade is therefore also a language grade: a held-out
  task's sentence is new.

## 3. Setup

Code, each in its own directory:

| what | where | commit |
|---|---|---|
| RLinf (evaluators, TrainConfigs, prompts; read through PYTHONPATH, not installed) | `/coc/flash10/ksingara3/robot/metaworld/ext/RLinf` | 91a302e (2026-10-20) |
| Farama Metaworld | the venv's `metaworld==3.0.0` wheel | |

Venv `/coc/flash10/ksingara3/robot/metaworld/.venv` (Python 3.11.14, uv; built by `slurm/jobs/x4_metaworld_install.sbatch`,
freeze in `robot/metaworld/build/venv_freeze.txt`): torch 2.7.1, torchvision 0.22.1, rlinf-openpi 0.1.1 (openpi with the PyTorch
pi0, plus its patched transformers `rlinf-transformer-openpi` 4.53.2), jax 0.5.3 (CPU only at run time), metaworld 3.0.0,
mujoco 3.16.0, gymnasium 1.1.1, lerobot 0.3.3, ray 2.59.0, numpy 1.26.4. Two pins are forced by override: gymnasium 1.1.1
(metaworld needs at least 1.1, lerobot 0.3.3 asks for less than 1.0; lerobot is used only for its dataset classes, which the
evaluation does not touch), and torch 2.7.1 (rlinf-openpi's pin; RLinf HEAD's uv override is 2.11). The existing
`robot/venvs/rlinf` venv, which belongs to another agent, is untouched.

The checkpoint `RLinf/RLinf-Pi0-MetaWorld-SFT` @ d0b8fbe is in the HF cache (`robot/data/hf/home`). Only `model.safetensors`
and `lerobot/metaworld_mt50/norm_stats.json` are read; the repo's `metadata.pt` pickle is never loaded. The PaliGemma tokenizer
comes from `OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache`.

Rendering: MuJoCo `MUJOCO_GL=egl`. Each GPU job starts with `experiments/x4_metaworld_probe.py`: it resets three tasks, renders
camera 2, saves the frames, and checks the installed task list against the split. RESULT: PARITY_PENDING_PROBE

## 4. Parity

PARITY_PENDING

## 5. Decisions for the user

DECISIONS_PENDING
