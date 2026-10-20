# SimplerEnv (Bridge V2 / WidowX): setup and parity gate (PLAN_X X2)

Status 2026-10-19: SimplerEnv renders on the A40 nodes (SAPIEN 2.2.2 through the NVIDIA Vulkan ICD; the login node has no
Vulkan). Both public checkpoint families run with their authors' evaluation code. open-pi-zero's pi0 reproduces its published
WidowX numbers within sampling noise. X-VLA-WidowX reproduces on spoon, carrot and eggplant but not on blocks, and its published
protocol differs from stock SimplerEnv in three ways (step budget, eggplant scene, best-of-runs). The section "X-VLA protocol"
below gives the details.

## Setup

Downloads (each in its own directory, untouched):

| what | where | commit |
|---|---|---|
| X-VLA code (server `deploy.py`, clients `evaluation/simpler/*`) | `robot/ext/xvla_src/X-VLA` | 6bc2513 (2026-01-28) |
| X-VLA's SimplerEnv fork (255isWhite, absolute-EE control) | `robot/ext/simpler_255isWhite/SimplerEnv` | 4233e3f; ManiSkill2_real2sim 54ae2e0 |
| open-pi-zero (allenzren) | `robot/ext/openpizero_src/open-pi-zero` | c3df7fb (2025-01-28) |
| open-pi-zero's SimplerEnv fork (allenzren, proprio in obs) | `robot/ext/simpler_allenzren/SimplerEnv` | 59ad9e1; ManiSkill2_real2sim 91d154b |
| X-VLA checkpoints | HF cache `2toINF/X-VLA-WidowX` (8d7ea1a), `2toINF/X-VLA-Google-Robot` (afaad7a) | |
| pi0 checkpoints | `robot/data/openpizero/bridge_{beta,uniform}_step19296_*.pt` (11.8 GB each, from `allenzren/open-pi-zero`) | |

The two forks both install as `simpler_env` / `mani_skill2_real2sim` and conflict, so there are two venvs (Python 3.10.22,
uv at `robot/simpler/uvtool`):

- `robot/simpler/.venv`: X-VLA + 255isWhite fork. torch 2.1.2+cu121, transformers 4.51.3, numpy 1.26.3 (X-VLA's
  requirements.txt), sapien 2.2.2, gymnasium 0.29.1, `setuptools<70` (sapien imports `pkg_resources`), imageio-ffmpeg with
  `ffmpeg` symlinked into `.venv/bin` (the clients' `write_video` calls the `ffmpeg` binary; the job puts `.venv/bin` on PATH).
- `robot/simpler/.venv_pi0`: open-pi-zero (`pip install -e .`: torch 2.5.0, transformers 4.47.1, numpy 1.26.4,
  tensorflow 2.15.0) + allenzren fork + sentencepiece.

PaliGemma tokenizer. open-pi-zero loads the tokenizer from `$TRANSFORMERS_CACHE/paligemma-3b-pt-224` (gated on the Hub; the
stored HF token is invalid). The checkpoint `.pt` holds all model weights, so only the tokenizer is needed. I rebuilt it in
`robot/data/paligemma_tok/paligemma-3b-pt-224` from the big_vision `paligemma_tokenizer.model` in the openpi cache. Its sha256
(8986bb4f...a168fc6) equals the Hub's LFS hash of `google/paligemma-3b-pt-224/tokenizer.model`; the `<image>` token maps to
257152 (the config's `image_token_index`), and the four instructions tokenize identically through the HF tokenizer and raw
sentencepiece.

## Commands (all through SLURM; overcap, A40)

    # X-VLA: authors' server + 4 clients in parallel against it, one job
    sbatch --partition=overcap --account=overcap --qos=short --gres=gpu:a40:1 --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw \
      --export=ALL,MODEL=2toINF/X-VLA-WidowX,SUITE=WidowX,TASKS=spoon:carrot:blocks:eggplant,TAG=run1 slurm/jobs/x2_xvla_simpler.sbatch
    # Google robot, visual matching
    ... --export=ALL,MODEL=2toINF/X-VLA-Google-Robot,SUITE=google-VM,TASKS=coke_can:move_near:open_close:place_in,TAG=run0 ...
    # pi0 (open-pi-zero scripts/run.py --config-name=bridge, as in their slurm/eval_simpler_bridge.sh), one job per task
    ... --export=ALL,CKPT=bridge_beta,TASK=widowx_carrot_on_plate,NEP=240,BF16=False slurm/jobs/x2_pi0_simpler.sbatch
    # tables + ledger rows
    python experiments/x2_simpler_summary.py --ledger

Outputs: `results/x2_simpler/xvla_<suite>_<tag>_<job>/` (per-episode JSON rows + videos), `results/x2_simpler/pi0_*` (hydra
logs), `slurm/<job>.out`. Preflight `experiments/x2_simpler_probe.py` (one env reset + render) runs at the top of each X-VLA job.

Time per evaluation (A40):
- X-VLA WidowX, 4 tasks x 24 episodes, clients in parallel: 8 to 9 min wall (server load 15 to 35 s). Successful episodes take
  2 to 6 s; failures run the full 1200 steps (about 1 to 2 min).
- pi0, 240 trials of one task: 31 to 33 min for the 60-step tasks; eggplant (120 steps) 60 to 62 min. The first episode includes
  about 4 min of torch.compile.
- X-VLA Google-robot VM (864 episodes over 4 tasks, clients in parallel): 2 h 22 min wall (place-in is the slowest client).

## Results: WidowX, visual matching

### pi0 (open-pi-zero), float32, 240 trials per task (24 layouts x 10), success at the 60/120-step timeout

| checkpoint | carrot on plate | eggplant in basket | spoon on towel | stack cube | avg |
|---|---|---|---|---|---|
| Bridge-Beta published | 55.8 | 85.4 | 84.6 | 47.9 | 68.4 |
| Bridge-Beta reproduced | 55.4 | 87.9 | 82.5 | 46.2 | 68.0 |
| Bridge-Uniform published | 58.8 | 79.2 | 63.3 | 21.3 | 55.7 |
| Bridge-Uniform reproduced | 60.8 | 77.1 | 64.6 | 26.2 | 57.2 |

One run of 240 trials has a binomial standard error of 2.3 to 3.2 points; every cell lies within 2 SE of the published value.
**Parity holds for pi0.** (The README also reports bf16 rows; I ran float32 only.)

### X-VLA-WidowX, the authors' client, 24 episodes per task, 3 independent runs

| | spoon | carrot | blocks | eggplant | avg |
|---|---|---|---|---|---|
| published (model card / repo README) | 100 | 91.7 | 95.8 | 95.8 | 95.8 |
| run1 | 100 | 100 | 83.3 | 95.8 | 94.8 |
| run2 | 100 | 95.8 | 83.3 | 95.8 | 93.8 |
| run3 | 100 | 95.8 | 79.2 | 95.8 | 92.7 |
| mean of 3 runs | 100 | 97.2 | 81.9 | 95.8 | 93.8 |
| same runs, success within the stock 60/120-step budget | 91.7 | 75.0 | 59.7 | 65.3 | 72.9 |

Spoon, carrot and eggplant reproduce. Blocks does not: 59/72 successes over three runs (81.9%), against 95.8%; at p = 0.958
we would expect about 69/72. The authors write that "the SIMPLER benchmark currently has uncontrollable randomness ... our
reported numbers are taken from the best rollout". Our best blocks run is 83.3 (20/24). The randomness is the server's
unseeded `torch.randn` initial noise.

### X-VLA protocol: how it differs from stock SimplerEnv

1. **Step budget.** The client runs up to 1200 env steps and stops at the first step where the env reports success. Stock
   SimplerEnv registers 60 steps (120 for eggplant) and judges the episode at the timeout. With the stock budget the same
   episodes score 72.9 on average (last row above), not 95.8. X-VLA's 95.8 is not comparable with numbers from the stock
   protocol (pi0's, Octo's, OpenVLA's). The Google-robot clients similarly use 2x the stock steps.
2. **Eggplant scene.** The 255isWhite fork (commit 3c3590a, "enable abs eef control") also edits `WidowXSinkCameraSetupConfig`:
   it moves the 3rd-view camera to the Bridge-dataset camera pose, the robot's init qpos / height to the table setup's, and
   init_y 0.070 -> 0.028. The real `bridge_sink.png` overlay stays the same, so the rendered robot and eggplant no longer line up with the
   photo. `results/x2_simpler/frames/{allenzren,xvlafork}_widowx_put_eggplant_in_basket.png` show the same reset in both forks:
   the eggplant sits about 100 px to the right and the gripper is in a different place. X-VLA's eggplant number comes from this modified scene.
   The carrot / spoon / blocks scenes render identically in both forks (same first frame).
3. **Control.** The fork adds absolute base-frame EE control (`arm_pd_ee_target_base_pose`, targets as xyz Euler) and makes
   it the Bridge default. The client feeds the model a proprio of [ee position w.r.t. base, rot6d of identity, 0] + 10 zeros.
   After every step it overwrites proprio[:10] with the last commanded action, not the measured pose. It executes all 30 actions
   of a chunk open loop, adds +pi/2 to pitch after rot6d -> Euler, and closes the gripper when p < 0.7.

## Results: X-VLA Google robot, visual matching (optional)

`2toINF/X-VLA-Google-Robot`, the authors' google-VM clients and configs (episodes per task as in their json; up to 2x the
stock steps; stop at first success), one run:

| | pick coke can | move near | open / close drawer | open top drawer, place apple | avg |
|---|---|---|---|---|---|
| episodes | 300 | 240 | 216 | 108 | |
| published | 98.3 | 97.5 | 74.5 | 63.8 | 83.5 |
| reproduced (job 4159547) | 99.0 | 98.3 | 74.1 | 54.6 | 81.5 |

Coke, move-near and drawer reproduce. Place-in is 9 points low (59/108; one run's binomial SE is about 4.8, so about 1.9 SE),
the same direction as WidowX blocks. Variant aggregation (VA) was not run.

## Step 5: recombined tasks in the Bridge scenes

### What exists

- Every WidowX task is a `PutOnBridgeInSceneEnv` (`mani_skill2_real2sim/envs/custom_scenes/put_on_in_scene.py`) with a
  `source_obj_name`, a `target_obj_name`, a list of xy layouts (the 24 "episode_id" layouts are 12 ordered pairs of 4 grid
  corners x 2 yaws for spoon/carrot; 24 xy pairs x 1 yaw for blocks; 8 positions x 3 yaws for eggplant), and an instruction.
- The success checker is generic (`PutOnInSceneEnv.evaluate`): source xy within the target's bbox half-diagonal + 3 mm, source
  above the target with z gap <= 2 cm (6 cm for the basket), and (optionally) no contact with anything but the target and robot.
  Spoon and eggplant pass `success_require_src_completely_on_target=False`. It also logs `moved_correct_obj`,
  `is_src_obj_grasped`, `consecutive_grasp`, `src_on_target`, so a partial-credit grade (grasped the right object, wrong
  placement) comes for free.
- Assets (`ManiSkill2_real2sim/data/custom/models`, metadata `info_bridge_custom_v0.json`): carrot, spoon, plate (two sizes),
  towel / table cloth (three), green and yellow 3 cm cubes (plain and baked-texture), eggplant, sink, and an invisible
  `dummy_sink_target_plane`. The Google-robot assets (cans, apple, orange, sponge, bottle) are in the same folder with their own
  json, and could in principle be put on the Bridge table.
- Scenes: the table (`bridge_table_1_v1`, overlay `bridge_real_eval_1.png`) for spoon / carrot / blocks, and the sink
  (`bridge_table_1_v2`, overlay `bridge_sink.png`) for eggplant. **The yellow basket is not a 3D object**: it is part of the
  real photo overlay, and success is "on the invisible target plane" placed where the basket is in the photo. So "X in basket"
  works only in the sink scene, and "eggplant on plate" needs the eggplant on the table scene.

### What a new pair takes

Subclass `PutOnBridgeInSceneEnv` with (source, target, layouts, instruction) and register it under a new id. No downloaded file
changes. `experiments/x2_simpler_recombine.py` does this for six pairs: carrot -> towel, spoon -> plate, eggplant -> plate,
green block -> plate, carrot -> yellow block (table scene); carrot -> basket (sink scene, a `PutEggplantInBasketScene` with the
carrot as source). For every layout it resets, renders, and checks the stock success checker by teleporting the source:
success is False at reset in 24/24 layouts for every pair, and True after the source rests on the target in 24/24 (table pairs)
and 22/24 (carrot in basket; the 11.5 cm carrot sometimes rolls off the 13 x 8 cm plane), in both forks
(`results/x2_simpler/recombine/{allenzren,xvlafork}/recombine_check.json` and first frames). Notes for building the split:

- **Pick the success flags per pair.** Spoon-like (flat, long) sources need `success_require_src_completely_on_target=False`
  on a towel; a long source on a small target (carrot on cube) is geometrically hard to satisfy and should be checked by
  teleport before use.
- **Visual matching is weaker off the stock pairs.** The overlay is a real photo of the empty table, so any new object combination
  on the table is still "visual matching" for the background. The sink overlay contains the basket, so new sources in the sink
  are as well matched as the eggplant.
- **Unseen relative to what?** The checkpoints were trained on all of Bridge V2, whose real scenes contain carrots, spoons,
  towels, plates, eggplants, baskets and blocks in many pairings. A grade (docs/SPLIT.md: G1 relation / G2 binding / G4 part)
  must be read against Bridge V2's instruction annotations (language only; no images needed), not against the 4 SimplerEnv
  tasks. Without that check, "carrot on towel" may well be a seen pair. G4 parts can come from the Google-robot asset set (cans,
  sponge, apple) placed on the Bridge table: objects Bridge's WidowX data likely never names in these scenes.
- **Instruction wording.** `_get_instruction_obj_name` strips asset-name tags ("bridge", "generated", sizes); the stock tasks
  use hand-written strings ("put eggplant into yellow basket"). Paraphrase sensitivity is a confound to fix once per pair.
- **Layouts.** Reusing the stock grid (4 corners, ordered pairs, 2 yaws -> 24) keeps episode counts and seeds comparable with
  the source tasks.
- **Protocol.** For a composition read, use the stock step budget (60 / 120) and the allenzren scene for eggplant/sink tasks
  even when running X-VLA, i.e. X-VLA's controller with the unmodified sink camera. That variant is not built yet: it needs a
  third SimplerEnv checkout with only the controller half of 3c3590a.

### The backbone interface on these checkpoints

`lifelong_ood/policy/backbone.py` (openpi pi0/pi0.5) and `backbone_xvla.py` (LeRobot X-VLA-LIBERO) define: `observations` ->
`prefix` (conditioning computed once) -> `v_full(x_t, t, prefix)` / `sample` / `flow_loss` / `unnormalize`, plus `text_span` and
`cond_hook` on the conditioning encoder's layers.

**X-VLA-WidowX** (`2toINF/X-VLA-WidowX`, same architecture as X-VLA-LIBERO; config `action_mode=ee6d`, `num_actions=30`).
- The checkpoint is in the authors' format (`models/modeling_xvla.py`), not LeRobot's. `XVLABackbone` loads LeRobot's
  `XVLAPolicy`. Either convert the weights the way `lerobot/xvla-libero` mirrors `2toINF/X-VLA-Libero` (key renames; the
  tied BART embedding stored once), or add a loader for the authors' `XVLA.from_pretrained`. The forward methods used
  (`forward_vlm`, `transformer(domain_id, vlm_features, aux_visual_inputs, action_with_noise, proprio, t)`,
  `action_space.preprocess/postprocess`) have the same names in both.
- observations: ONE image (`image0` = the 3rd-view camera, 640x480 -> the processor's resize), no wrist view; BART-tokenized
  instruction; `domain_id = 0` (LIBERO uses 3); proprio = [ee pos w.r.t. base (3), rot6d (6) fixed to identity, 0] + 10 zeros, then
  the last commanded action's first 10 dims after each step.
- prefix / text_span / cond_hook: unchanged (Florence-2 encoder over [image tokens, text]); only the image count differs.
- v_full / sample / flow_loss: unchanged (x-prediction, t = 1 noise; native re-noising sampler, 10 steps; H = 30, A = 20,
  10 real dims).
- unnormalize: rows are ABSOLUTE base-frame targets [pos 3, rot6d 6, gripper prob 1] -> [pos, Euler xyz(rot6d) + (0, pi/2, 0),
  gripper +1 if p < 0.7 else -1]. Only the 255isWhite fork's controller (`arm_pd_ee_target_base_pose`, Euler) accepts these. An
  `action_codec` for Bridge would map Bridge's delta actions to absolute targets with each frame's pose (as `rel2abs` does for LIBERO).

**pi0 (open-pi-zero)**: its own PyTorch implementation, not openpi's, so it needs its own `backbone_openpizero.py`.
- observations: one 224x224 image (Lanczos resize of the 3rd view), instruction (max 20 tokens after 256 image tokens), proprio
  7-D [ee pos, roll/pitch/yaw in Bridge's top-down frame, gripper openness] bound-normalized by p01/p99
  (`config/bridge_statistics.json`). Requires the allenzren fork's `obs["agent"]["eef_pos"]`.
- prefix: the KV cache of the VLM block + the proprio block (`PiZeroInference` builds it once, then 10 action-expert passes).
  text_span = token positions 256 .. 256 + n_text; cond_hook on the Gemma mixture's layers (`joint_model`, mixture "vlm").
- v_full: the action expert's `action_decoder(...)` at time t. **The time convention is reversed vs openpi**: t = 0 noise,
  t = 1 data, x_t = (1 - (1 - sigma_min) t) eps + t a, target velocity a - (1 - sigma_min) eps, sigma_min = 1e-3, Euler with
  +1/10. In backbone terms (t' = 1 - t): v_backbone(x, t') ~= -v_opz(x, 1 - t'), and Tweedie a_hat = x_t + (1 - t) v_opz (exact up to sigma_min).
- rows: H = 4, A = 7 (delta EE pos, delta roll/pitch/yaw, gripper in [0, 1]), dims 0..5 normalized to [-1, 1] by p01/p99
  bounds, gripper unnormalized. The final action is clipped to [-1, 1].
- unnormalize: `BridgeSimplerAdapter.postprocess` (denormalize the bounds, Euler -> axis-angle, gripper > 0.5 -> +1 open);
  all 4 actions executed (act_steps = 4), 5 Hz.
- flow_loss: `PiZero.forward`'s MSE on (v - (a - (1 - sigma_min) eps)), times sampled Beta (pi0-style) or Uniform.

## Ledger

`ledger/x2_simpler_parity/`: tag `xvla` (mean of the 3 runs, per-run values and the stock-budget numbers in metrics) and tag
`pi0` with keys ckpt = bridge_beta / bridge_uniform. `bytes` are all zero (evaluation only).

## The graded split against all of Bridge V2 (2026-10-20; `experiments/x2_bridge_grades.py`, ledger `x2_bridge_grades` tag `read`)

The WidowX checkpoints were trained on all of Bridge V2, so a pair is graded against its 19,974 distinct instructions
(IPEC-COMMUNITY/bridge_orig_lerobot meta/tasks.jsonl; language only). A pair named together anywhere counts as seen. That is an
over-count ("take carrot off plate" counts too), so "never named together" is the reliable signal for an unseen relation.

| grade | pairs |
|---|---|
| seen (named together) | carrot → cloth (35 instructions), spoon → plate (12), eggplant → plate (3), coke can → towel (27), coke can → plate (1), apple → towel (6), apple → plate (3), sponge → plate (2) |
| G1 relation (both parts named, never together) | green block → plate, carrot → yellow block, carrot → basket |
| G4 part (object never named in Bridge) | Pepsi can → towel, Red Bull can → plate, 7up can → towel (Google-robot assets placed on the Bridge table) |

Three of the six recombined pairs built first are therefore seen relations. The Bridge split uses the three G1 pairs and the
G4 cans.
