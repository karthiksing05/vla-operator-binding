# CALVIN (ABC → D): setup and the parity gate (X1)

Plan X1 (docs/PLAN_X.md): before any composition read on CALVIN, each public checkpoint must reproduce its published score with
the authors' own evaluation code. This page records the setup, the commands, the published and reproduced numbers, and every
deviation.

## Result

Standard ABC → D evaluation: 1,000 five-instruction chains in scene D (CALVIN's seeded `get_sequences(1000)`), success at
lengths 1-5 and the average chain length (0-5). One run per row (seed 0 for the evaluation; the policies' sampling noise is not
seeded per chain). The standard error of an average length over 1,000 chains is about 0.04.

| checkpoint | evaluation | chains | 1 | 2 | 3 | 4 | 5 | avg. len. |
|---|---|---|---|---|---|---|---|---|
| FLOWER `mbreuss/flower_calvin_abc` | published (README, 1000 chains) | 1000 | 99.3 | 95.9 | 90.5 | 84.8 | 77.5 | **4.54 ± 0.02** |
| | authors' code, as is | 50 | 100 | 100 | 92.0 | 80.0 | 70.0 | 4.42 |
| | authors' code, as is | 1000 | 99.7 | 96.7 | 91.6 | 85.1 | 77.7 | **4.508** |
| | + the two-tensor load fix (below) | 50 | 100 | 100 | 94.0 | 78.0 | 66.0 | 4.38 |
| | + the two-tensor load fix (below) | 1000 | 99.6 | 96.9 | 92.2 | 86.0 | 79.6 | **4.543** |
| X-VLA `2toINF/X-VLA-Calvin-ABC_D` | published (repo README / paper) | 1000 | 97.1 | 92.6 | 88.5 | 84.4 | 78.8 | **4.43** |
| | authors' current client (720 steps per subtask) | 50 | 94.0 | 90.0 | 84.0 | 78.0 | 72.0 | 4.18 |
| | authors' current client (720 steps per subtask) | 1000 | 96.7 | 91.2 | 86.8 | 81.1 | 75.1 | **4.309** |
| | same client, standard 360 steps | 1000 | 96.1 | 90.5 | 85.7 | 79.3 | 72.5 | 4.241 |
| | authors' earlier client (e57bc00, 360 steps) | 1000 | 94.9 | 88.7 | 82.3 | 75.7 | 67.2 | 4.088 |

**FLOWER reproduces.** The authors' code as is gives 4.508 (published 4.54 ± 0.02; 0.03 below, within one standard error of a
1,000-chain run), and every per-length rate is at or above the published one (by 0.2-1.1 points). The published row is not
self-consistent: its five rates sum to 4.48, not 4.54 (the average length equals the sum of the five rates), so on the rates our
run is slightly above the publication and on the headline slightly below. With the load fix it gives 4.543.

**X-VLA does not reproduce cleanly.** The authors' current client gives 4.309 against a published 4.43: 0.12 below, about three
standard errors, and lower at every length (the first-step rate matches, 96.7 vs 97.1; the gap grows along the chain). The gap is
concentrated in one family of subtasks: `push_{red,blue,pink}_block_right` succeed 57 / 53 / 72 % against 95 / 94 / 97 % for the
same pushes to the left; FLOWER succeeds at 95-97 % on both directions in the same environment. Lights, drawer and slider are at
98-100 % for both policies, so the environment's button / light code (the one difference between calvin_env commits, below) is
not the cause. The earlier client (before the 2026-01-05 rewrite of the evaluation) is worse, 4.088, with the same push-right
deficit (45 / 50 / 57 % for red / blue / pink), so the deficit belongs to the checkpoint as served, not to one client version.
We found no setting of the authors' code that gives 4.43. Candidates we could not test: the authors' exact environment stack
(their conda envs `UniAct` / `calvin_venv`, versions unpublished) and calvin_env 1431a46 (its differences, listed below, touch
only button / light resets and relative actions; X-VLA sends absolute actions). The per-subtask table follows.

Per-subtask success, successes / attempts (%), over each full 1,000-chain run (`experiments/x1_calvin_pertask.py`; per-chain
lengths in each run's `results_<start>_<end>.json` under `results/x1_calvin/`):

| subtask | FLOWER, authors' loader | FLOWER, load fix | X-VLA, current client, 720 | X-VLA, current client, 360 | X-VLA, earlier client, 360 |
|---|---|---|---|---|---|
| close_drawer | 223/223 (100) | 229/229 (100) | 205/205 (100) | 210/210 (100) | 205/205 (100) |
| lift_blue_block_drawer | 19/19 (100) | 20/20 (100) | 15/16 (94) | 17/17 (100) | 15/16 (94) |
| lift_blue_block_slider | 123/143 (86) | 126/145 (87) | 126/139 (91) | 127/142 (89) | 106/130 (82) |
| lift_blue_block_table | 199/200 (100) | 202/203 (100) | 177/179 (99) | 170/176 (97) | 169/175 (97) |
| lift_pink_block_drawer | 15/15 (100) | 13/14 (93) | 14/14 (100) | 13/13 (100) | 11/11 (100) |
| lift_pink_block_slider | 138/148 (93) | 137/146 (94) | 130/142 (92) | 131/142 (92) | 122/138 (88) |
| lift_pink_block_table | 178/182 (98) | 182/183 (99) | 167/177 (94) | 165/174 (95) | 153/172 (89) |
| lift_red_block_drawer | 17/18 (94) | 17/18 (94) | 17/19 (89) | 21/21 (100) | 14/16 (88) |
| lift_red_block_slider | 123/144 (85) | 131/148 (89) | 123/132 (93) | 112/123 (91) | 109/131 (83) |
| lift_red_block_table | 194/197 (98) | 194/197 (98) | 175/180 (97) | 168/175 (96) | 163/177 (92) |
| move_slider_left | 258/259 (100) | 263/264 (100) | 257/260 (99) | 250/253 (99) | 252/256 (98) |
| move_slider_right | 306/306 (100) | 300/301 (100) | 289/289 (100) | 287/287 (100) | 280/280 (100) |
| open_drawer | 360/360 (100) | 365/366 (100) | 354/354 (100) | 351/351 (100) | 348/348 (100) |
| place_in_drawer | 195/195 (100) | 194/194 (100) | 178/179 (99) | 175/177 (99) | 165/166 (99) |
| place_in_slider | 295/376 (78) | 295/374 (79) | 328/358 (92) | 325/357 (91) | 296/331 (89) |
| push_blue_block_left | 66/69 (96) | 66/68 (97) | 66/70 (94) | 68/69 (99) | 67/69 (97) |
| push_blue_block_right | 66/69 (96) | 67/69 (97) | 36/68 (53) | 35/70 (50) | 34/68 (50) |
| push_into_drawer | 101/134 (75) | 107/134 (80) | 100/127 (79) | 100/122 (82) | 86/114 (75) |
| push_pink_block_left | 72/73 (99) | 73/73 (100) | 74/76 (97) | 67/76 (88) | 70/76 (92) |
| push_pink_block_right | 59/62 (95) | 62/65 (95) | 48/67 (72) | 39/68 (57) | 38/67 (57) |
| push_red_block_left | 75/77 (97) | 76/79 (96) | 75/79 (95) | 74/79 (94) | 77/79 (97) |
| push_red_block_right | 69/71 (97) | 69/71 (97) | 41/72 (57) | 36/72 (50) | 32/71 (45) |
| rotate_blue_block_left | 66/67 (99) | 67/67 (100) | 66/67 (99) | 65/66 (98) | 64/67 (96) |
| rotate_blue_block_right | 74/78 (95) | 75/78 (96) | 74/76 (97) | 73/77 (95) | 76/77 (99) |
| rotate_pink_block_left | 53/55 (96) | 55/55 (100) | 54/56 (96) | 54/56 (96) | 49/54 (91) |
| rotate_pink_block_right | 70/70 (100) | 67/68 (99) | 68/68 (100) | 68/68 (100) | 69/70 (99) |
| rotate_red_block_left | 63/63 (100) | 64/64 (100) | 62/64 (97) | 63/65 (97) | 60/63 (95) |
| rotate_red_block_right | 73/74 (99) | 71/72 (99) | 73/74 (99) | 72/74 (97) | 73/73 (100) |
| stack_block | 189/208 (91) | 190/210 (90) | 170/197 (86) | 170/189 (90) | 170/187 (91) |
| turn_off_led | 170/171 (99) | 174/176 (99) | 169/172 (98) | 166/170 (98) | 154/161 (96) |
| turn_off_lightbulb | 150/152 (99) | 151/153 (99) | 155/155 (100) | 149/149 (100) | 150/152 (99) |
| turn_on_led | 185/187 (99) | 186/187 (99) | 179/181 (99) | 178/181 (98) | 169/173 (98) |
| turn_on_lightbulb | 189/191 (99) | 185/185 (100) | 183/183 (100) | 184/184 (100) | 177/177 (100) |
| unstack_block | 75/75 (100) | 70/71 (99) | 61/63 (97) | 58/63 (92) | 65/66 (98) |

Time per 1,000 chains (one A40 per job, 5 evaluation processes per GPU, 2 jobs): FLOWER 2,600-2,740 s wall per half (about
45 minutes for the 1,000 chains on 2 GPUs; 24,500 s summed over the 10 processes, about 25 s per chain per process);
X-VLA (one server per process, 720 steps) 3,540 s per half (about 1 hour), 32,300 s summed (32 s per
chain); X-VLA at 360 steps 2,840 s per half, 24,600 s summed. A 50-chain quick check takes 7-8 minutes per job.

## Setup

Disk: nothing of the ~500 GB ABC_D dataset is needed. The evaluation needs only calvin_env (its scene assets ship in the repo,
`calvin_env/data/`), the scene-D env config `task_ABC_D/validation/.hydra/merged_config.yaml`, the task oracle
(`new_playtable_tasks.yaml`) and one annotation per subtask (`new_playtable_validation.yaml`). X-VLA's repo ships exactly these
six files (`evaluation/calvin/ABC_D/validation/`); they are copied to `/coc/flash10/ksingara3/robot/calvin/task_ABC_D/validation`
(with `robot/calvin/ABC_D -> task_ABC_D` for X-VLA's relative path). The two task / annotation files are byte-identical to
FLOWER's `conf/callbacks/rollout_lh/tasks/new_playtable_tasks.yaml` and `conf/annotations/new_playtable_validation.yaml`.

Code (each in its own directory under `/coc/flash10/ksingara3/robot/ext/`):

| what | where | commit |
|---|---|---|
| CALVIN (mees/calvin), for calvin_agent's evaluation utilities | `ext/calvin/calvin` | fa03f01 |
| calvin_env (submodule) | `ext/calvin/calvin/calvin_env` | **797142c** (calvin pins 1431a46; 797142c is calvin_env's main and FLOWER's pin; see Deviations) |
| tacto (calvin_env submodule) | `ext/calvin/calvin/calvin_env/tacto` | dd53360 |
| FLOWER (intuitive-robots/flower_vla_calvin) | `ext/flower_vla_calvin` | acb32ee (untouched) |
| X-VLA (2toinf/X-VLA) | `ext/X-VLA/X-VLA` | 6bc2513 (untouched) |
| X-VLA's earlier CALVIN client | `ext/X-VLA/client_e57bc00/calvin_client.py` | `git show ecbef0c^:evaluation/calvin/calvin_client.py` |

Checkpoints (HF cache `robot/data/hf/home`): `mbreuss/flower_calvin_abc` @ 6424ee4 (4.0 GB; uses `microsoft/Florence-2-large`
from the cache), `2toINF/X-VLA-Calvin-ABC_D` @ d76710e (3.5 GB).

Environments (new; the existing FLOWER and SmolVLA venvs are untouched):

- `/coc/flash10/ksingara3/robot/calvin/.venv` (Python 3.10.22; the simulator, FLOWER, and X-VLA's client): torch 2.10.0+cu128,
  torchvision 0.25.0, transformers 4.46.3, pytorch-lightning 2.4.0, hydra-core 1.3.2, omegaconf 2.3.1, numpy 1.26.4, timm 1.0.11,
  pybullet 3.2.7, gym 0.26.2, pyrender 0.1.45, urdfpy 0.0.22 (--no-deps), pyhash 0.9.3; calvin_env, tacto, calvin_models
  (`calvin_agent`) installed editable with `--no-deps`. Same versions as the LIBERO FLOWER venv where they overlap.
- `/coc/flash10/ksingara3/robot/calvin/.venv-xvla` (X-VLA's server, its own pins): torch 2.10.0+cu128, transformers 4.51.3,
  fastapi, uvicorn 0.34.3, json_numpy 2.1.0, timm 1.0.12, numpy 1.26.3, safetensors 0.4.5, peft 0.17.1, accelerate 1.2.1.

Install notes:
- pyhash 0.9.3 (CALVIN's per-chain block-rotation seed, `fnv1_32`) does not build from PyPI here. Built from FLOWER's bundled copy,
  copied to `robot/calvin/build/pyhash`, with three build-only edits: `-march=native` removed (built on the login node, run on
  other CPUs), `use_2to3` / `setup_requires` removed (setuptools < 58), and `#include <cstdint>` added to the bundled pybind11
  `attr.h` (newer gcc). The hash function itself is unchanged.
- calvin_env's EGL probe (`egl_check/EGL_options.o`) does not build: the cluster has no X11 headers. HulcWrapper's own fallback
  for a failed probe (EGL device 0) is used; each job has one GPU. PyBullet's EGL plugin loads on the GPU nodes.
- `PYOPENGL_PLATFORM=egl` for X-VLA's client (its env config builds the tactile camera with pyrender).

## Commands

Quick check (50 chains, 5 processes on one A40), then the full 1,000 as two 500-chain jobs:

```
S="sbatch --partition=overcap --account=overcap --qos=short --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw"
$S --export=ALL,START=0,END=50,NPROC=5,FIX=0 slurm/jobs/x1_calvin_flower.sbatch
$S --export=ALL,START=0,END=50,NPROC=5,EPLEN=720 slurm/jobs/x1_calvin_xvla.sbatch
for r in "0 500" "500 1000"; do set -- $r
  $S --export=ALL,START=$1,END=$2,NPROC=5,FIX=0,OUT=$PWD/results/x1_calvin/flower_full_ep360_fix0/part_$1_$2 slurm/jobs/x1_calvin_flower.sbatch
  $S --export=ALL,START=$1,END=$2,NPROC=5,EPLEN=720,OUT=$PWD/results/x1_calvin/xvla_full_ep720/part_$1_$2 slurm/jobs/x1_calvin_xvla.sbatch
done
python experiments/x1_calvin_merge.py --dir results/x1_calvin/flower_full_ep360_fix0     # merge shards; --record writes the ledger row
```

`FIX=1` adds the FLOWER load fix; `EPLEN=360` runs X-VLA at the standard horizon;
`CLIENT=/coc/flash10/ksingara3/robot/ext/X-VLA/client_e57bc00/calvin_client.py` runs X-VLA's earlier client.

- `experiments/x1_calvin_flower_eval.py` runs FLOWER's `flower/evaluation/flower_evaluate.py` `main()` (hydra `eval_calvin`:
  seed_everything(0), `load_mode_from_safetensor`, num_sampling_steps 4, multistep 10, ep_len 360, `evaluate_policy` / `rollout` /
  `print_and_save`). Shards are contiguous slices of the 1,000-chain list.
- `experiments/x1_calvin_xvla_client.py` runs X-VLA's `evaluation/calvin/calvin_client.py` (seed_everything(0), `get_env`,
  `ClientModel`, `evaluate_policy(eval_start, eval_end)`) against the authors' server (`python -m deploy --model_path <snapshot>`
  from the repo root), one server per client process, as in their `launch.sh`.
- `experiments/x1_calvin_merge.py` merges `results_<start>_<end>.json` shards (asserting they tile [0, 1000)) and writes the
  ledger row (`ledger/x1_calvin_parity/`, keys n_chains, ep_len, loader / client).
- `experiments/x1_calvin_pertask.py --flower_root <flower_vla_calvin> --cache results/x1_calvin/sequences.json <run dirs>`: the
  per-subtask table (chain i's first `result` subtasks succeeded, subtask `result` failed, over the same `get_sequences(1000)`).

## Deviations from the authors' code (all outside the decision path unless stated)

FLOWER:
1. **No data module.** The authors' `get_default_mode_and_env` builds a HulcDataModule over the ABC_D dataset only to get the
   validation dir (for `get_env`), the val image transforms and the robot_obs normalisation from `statistics.yaml`. A stand-in
   supplies the scene-D config dir, the checkpoint config's `datamodule.transforms.val` instantiated as HulcDataModule.setup does
   (Resize 224, scale, CLIP mean/std), and no robot_obs transform: the checkpoint has `use_proprio: false`, so the state never
   reaches the network.
2. `print_and_save` names the result by `checkpoint.stem.split("=")[1]` (their files are `...avg_seq_len=4.72.ckpt`); the HF file
   is `model.safetensors`, so the key is renamed. `wandb.log` is a no-op (log_wandb false).
3. The clone has an empty `calvin_env/` submodule directory at its root, which shadows the installed calvin_env once the root is on
   `sys.path` (the README's "Common Issues"); the installed calvin_env is imported first.
4. **The load fix (FIX=1 only; a real deviation).** The authors' loader reports 2 missing and 2 unexpected keys on this HF
   checkpoint: the export renamed `vlm.language_model.model.shared.weight` and `vlm.language_model.final_logits_bias` to
   `vlm.language_shared.weight` / `vlm.language_final_logits_bias`, so Florence-2's base token embeddings replace the fine-tuned
   ones, and the `<Flow>` prompt embedding is the freshly resized (random, seed-0) token row. FIX=1 loads the two tensors and sets
   `<Flow>` to the checkpoint's row, as `lifelong_ood/policy/backbone_flower.py` already does for LIBERO. On CALVIN the fix is
   worth +0.035 (4.508 → 4.543), inside the noise; the published 4.54 was presumably measured on the training-time module.

X-VLA:
1. cwd holds the copied `ABC_D/validation` (the client resolves that relative path); videos are not written (`save_video` no-op).
2. numpy's removed `np.float` / `np.int` / `np.bool` aliases are restored as the builtins (urdfpy 0.0.22, imported by the
   tactile camera that the client's env config builds).
3. One warm-up request per server before the evaluation, with a long timeout. The client posts with `timeout=10` and no retry; a
   server's first call (CUDA initialisation, five servers starting on one GPU) exceeded it and killed three 100-chain shards in
   the first full runs (0-100 and 300-400 at 360 steps, 500-600 at 720 steps). Those shards were rerun from fresh processes with
   the warm-up (same seed-0 start as every shard); the failed attempts are kept in `results/x1_calvin_failed/`.
4. The authors' current client runs **720 steps per subtask**; CALVIN's protocol (calvin_agent's `evaluate_policy.py`, FLOWER)
   is 360. Both are reported. The client was changed on 2026-01-05 (commit ecbef0c "update eval for calvin": EP_LEN 360 → 720,
   chunk 15 → 20 executed actions, 5 → 10 flow steps, gripper threshold 0.7 → 0.8, the initial proprio gripper flag
   `robot_obs[-1] < 0.5` → `> 0`, the model reset once per chain instead of once per subtask), after the checkpoint (2025-11-05)
   and the paper; the earlier client is the last row of the table.

Both: calvin_env at 797142c (calvin_env main, FLOWER's pinned commit) rather than 1431a46 (the commit mees/calvin pins). The
difference: actions copied before scaling (`relative_to_absolute`), lights reset before buttons, and a button's state initialised
from its light (a fix for rollouts that start with the LED or lamp on). Both policies are at 98-100 % on every light subtask here.

## Mapping the backbone interface onto the CALVIN checkpoints (step 5; not implemented)

`PolicyBackbone` (`lifelong_ood/policy/backbone.py`) exposes a policy as a flow field over flattened action chunks with a cached
conditioning prefix: `observations / observation`, `prefix`, `text_span`, `cond_layers / cond_hook`, `v_full`, `flow_loss`,
`sample`, `unnormalize`, `action_codec`.

**FLOWER-CALVIN** is the same network class as the LIBERO wrapper (`FLOWERVLA`, `backbone_flower.py`); a CALVIN wrapper is that
class with a different checkpoint and different observation conventions:
- `observations`: CALVIN's raw `rgb_static` (200 × 200) and `rgb_gripper` (84 × 84), uint8, **no flip**; Resize(**224**) (LIBERO
  uses 112), / 255, CLIP mean / std (the checkpoint's `datamodule.transforms.val`). No proprio (`use_proprio: false`). The
  prompt style is `default`: the instruction alone (LIBERO's checkpoint uses `minimal` with the robot description), so
  `text_span`'s instruction offset is 0 into the prompt (the slot after [static DaViT tokens, gripper DaViT tokens, `<Flow>`]).
- `prefix`: `encode_observations` once per observation, as for LIBERO; `cond_layers`: the Florence-2 encoder layers.
- `v_full` / `sample` / `flow_loss`: unchanged (rectified flow, z_t = (1 - t) a + t eps, 4 Euler steps, clamp(-1, 1); H = 10,
  A = 7).
- `unnormalize`: identity + clamp: CALVIN's 7-D **relative** actions (pos 3, euler 3, gripper), which HulcWrapper binarises
  (gripper > 0 → +1); multistep 10 (the whole chunk is executed, no replanning).
- Load: use the FIX=1 weights (the two renamed tensors and the checkpoint's `<Flow>` row), which `_fix_weights` already does.

**X-VLA-CALVIN** is the network of `backbone_xvla.py` (Florence-2 encoder + soft-prompted transformer, x-prediction, the
re-noising native sampler) with CALVIN's conventions:
- `observations`: `image0` = `rgb_static`, `image1` = `rgb_gripper`, both unflipped, through `XVLAProcessor` (resize 224,
  ImageNet mean / std); **domain_id 2** (LIBERO is 3).
- proprio: 20-D = [ee pos 3, rot6d of the euler `robot_obs[3:6]` 6, gripper flag 1] + 10 zeros, and **the client feeds back the
  last executed predicted action as the next proprio** (`self.proprio[:10] = action_predict[:10]`), reset from the real state
  once per chain (current client) or per subtask (earlier client). A wrapper must keep this state outside the prefix.
- `v_full` / `sample`: as `backbone_xvla.py` (v = (x_t - a_hat) / t), 10 steps (5 in the earlier client); chunk 30, of which the
  client executes the first 20 (15 earlier).
- `unnormalize`: **absolute** end-effector targets: pos 3 + rot6d → quaternion (Gram-Schmidt) + gripper (prob < 0.8 → open +1,
  else close -1), executed as calvin_env's `cartesian_abs` tuple action. This differs from LIBERO's absolute axis-angle codec;
  `action_codec` needs CALVIN's own rel → abs map if it is to read CALVIN demonstrations.

Two points for a composition read: X-VLA's proprio feedback makes a chunk depend on the previous chunk's prediction, not only on
the observation (a pointer or a guide changes later prefixes); and X-VLA's push-right deficit (above) must be read as a baseline
weakness, not as an effect of any operator.

## What CALVIN's 34 subtasks give as concepts (for a graded split, docs/SPLIT.md)

The 34 subtasks (the task oracle) factor into a verb, an object, a direction or a source / target place:

| family | subtasks | parts |
|---|---|---|
| rotate | rotate_{red,blue,pink}_block_{left,right} (6) | verb rotate × block colour × direction |
| push | push_{red,blue,pink}_block_{left,right} (6) | verb push × block colour × direction |
| lift | lift_{red,blue,pink}_block_{table,slider,drawer} (9) | verb lift × block colour × source place |
| place | place_in_{slider,drawer} (2) | verb place × target place (the grasped block is implicit) |
| articulated | open / close_drawer, move_slider_{left,right} (4) | verb × articulated object (× direction) |
| switches | turn_{on,off}_{lightbulb,led} (4) | verb × switch × state |
| relations | stack_block, unstack_block, push_into_drawer (3) | block-block or block-place relations, no colour named |

So the parts are: verbs (rotate, push, lift, place, open / close, move, turn on / off, stack / unstack), objects (three block
colours, which also differ in size: red middle, blue small, pink big in scene D; drawer, slider, lightbulb switch, LED button), directions (left /
right), and places (table, slider, drawer). Relations are the chain: each five-step chain is a sequence whose steps are conditioned
on the state the previous steps left (a block lifted is then placed or stacked; a drawer opened is then pushed into).

Grades under ABC → D, by the docs/SPLIT.md rule:
- **G3 scene is the whole benchmark.** Scene D is never seen in training (A, B, C are): its table texture, the positions of the
  drawer, slider, lightbulb and LED, and the block placement all differ. Every one of the 34 subtasks was trained in A-C, so ABC → D
  as published is a pure G3 read (every part seen in other scenes), the regime where SPLIT.md finds composing held units does
  nothing and creation is needed. Both checkpoints already do well here (4.3-4.5 of 5), which makes CALVIN a test of whether
  composition helps where the policy is strong, not of recovering a failing policy.
- **G1 relation / G2 binding need held-out pairs.** The published checkpoints were trained on all 34 subtasks, so a G1 / G2 grade
  only exists for pairings that never occur in training: (verb, colour, direction / place) triples are all present, so held-out
  cells must come from new language-level combinations (e.g. "rotate the block in the drawer", "push the pink block into the
  slider") whose success needs a new task-oracle entry, or from chains. Chains give the natural relation grade: CALVIN's
  `get_sequences` draws 5-step chains whose consecutive pairs (e.g. lift_red_block_slider → stack_block) can be counted against the
  training annotations' consecutive pairs (needs only `training/lang_annotations/auto_lang_ann.npy` of task_ABC_D, not the
  episodes; CALVIN ships it inside the dataset zip, so it has to be found separately) and graded seen / unseen.
- **G4 part** would need a verb, object or place absent from all 34 (none in the benchmark as published), or a new checkpoint
  trained with a subtask held out, which needs the training data (the ~500 GB ABC dataset, or D alone at ~166 GB for D → D).
- For the unsupervised scheme, CALVIN's chains are the natural test of segmentation: one 5-subtask chain is one long trajectory
  whose subtask boundaries are known from the oracle, so contact-cut units can be scored against the oracle's boundaries.

Proposed first read, no training data required: grade the 1,000 chains' consecutive subtask pairs as seen / unseen in ABC's
training annotations (G1-relation on the chain), and report the per-pair success of each policy by grade from the runs above
(per-chain results are on disk); a held-subtask G2 / G4 split needs a fine-tune with a subtask left out and so the dataset.
