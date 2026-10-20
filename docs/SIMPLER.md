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
over-count ("take carrot off plate" counts too), so "never named together" is the reliable signal for an unseen relation. Names are
matched on word boundaries; an earlier version matched "coke" inside "chocolate" and "colander" and "apple" inside "pineapple" (fixed 2026-10-20).

| grade | pairs |
|---|---|
| seen (named together) | carrot → cloth (35 instructions), spoon → plate (12), eggplant → plate (3), apple → plate (3), apple → towel (4), sponge → plate (2) |
| G1 relation (both parts named, never together) | green block → plate, carrot → yellow block, carrot → basket |
| G4 part (object never named in Bridge) | Pepsi can → towel, Red Bull can → plate, 7up can → towel, coke can → plate / towel (Google-robot assets on the Bridge table; a generic "can" is named, so these are unseen brands of a seen category, and no instruction puts a can on a towel or plate) |

Three of the six recombined pairs built first are therefore seen relations. The Bridge split uses the three G1 pairs and the
G4 cans.

## The pi0 side of the Bridge study (2026-10-20): graded tasks, backbone, driver, zero-shot baselines

### The six graded tasks (`experiments/x2_simpler_graded.py`)
Registered by subclassing the stock put-on envs (no downloaded file changes). Each keeps the stock success checker and the stock
24 layouts (4 grid corners x ordered pairs x 2 yaws; the sink task keeps the eggplant's 8 positions x 3 yaws). Budgets are stock:
60 steps, or 120 in the sink scene. Use the allenzren fork (stock scenes). The cans are the Google-robot `opened_*` models, pulled
in through `model_db_override` from `info_pick_custom_v0.json`. They use density 50, as in the stock coke-can tasks; the unopened
models (density 1000, about 0.4 kg) are above the WidowX payload. Each can stands upright (Google's `upright` quat), with the
stock yaw applied on top.

| env id | grade | pair | instruction | placement check: success at reset / after placing |
|---|---|---|---|---|
| X2G1GreenBlockOnPlate-v0 | G1 | green block -> plate | put the green block on the plate | 0/24, 24/24 |
| X2G1CarrotOnYellowBlock-v0 | G1 | carrot -> yellow block | put carrot on the yellow block | 0/24, 24/24 |
| X2G1CarrotInBasket-v0 | G1 | carrot -> basket (sink) | put carrot into yellow basket | 0/24, 23/24 |
| X2G4PepsiOnTowel-v0 | G4 | Pepsi can -> towel | put the pepsi can on the towel | 0/24, 24/24 |
| X2G4RedBullOnPlate-v0 | G4 | Red Bull can -> plate | put the red bull can on the plate | 0/24, 21/24 |
| X2G4SevenUpOnTowel-v0 | G4 | 7up can -> towel | put the 7up can on the towel | 0/24, 24/24 |

Each can settles upright: the minimum cosine between its axis and z is 0.9998, and its centre sits at z = 0.934, i.e. standing
on the table at 0.87. Frames are in `results/x2_simpler/graded/validate/`. The placement misses come from tall cans tipping when
dropped onto the plate, and from the carrot rolling off the basket plane. `opened_redbull_can`'s metadata bbox is 17.2 cm tall,
while the rendered can is about 13 cm. This only loosens its z test.

### OPZBackbone (`lifelong_ood/policy/backbone_opz.py`) and its parity check (`experiments/x2_opz_check.py`)
The check covers 16 SimplerEnv frames: 4 stock tasks x 2 layouts x {reset, after 8 policy steps}. Results:
- Our `observations` reproduce `BridgeSimplerAdapter.preprocess` exactly: token ids, mask, pixels and proprio all identical.
- On the same noise, our PolicyBackbone-Euler `sample` matches `PiZeroInference.forward`. The maximum difference is 1.8e-7 in
  normalised units and 1.8e-8 in SimplerEnv action units (float32). The reversed time convention therefore maps exactly:
  v(x, t) = -u(x, 1 - t).
- `text_span` decodes to `<bos>instruction\n`.
- `cond_hook` fires once per Gemma layer, 18 per prefix. The prefix pass is unrolled, because the joint model never calls its
  layer modules and torch hooks would not fire.
- An identity hook changes the chunk by 0.
- Flow loss at the sampled chunk averages 0.118, against 0.521 with the chunk's dims shuffled.
- One query (prefix + 10 steps, eager fp32) takes 0.34 s.

Results are in `results/x2_simpler/opz/opz_check_bridge_beta.json`.

### Rollout driver (`experiments/x2_simpler_rollout.py`)
It follows the stock protocol:
- Query on the 3rd-view image and proprio, execute all 4 actions, and stop only at the time limit. Success is the env flag at the
  last step.
- Flow noise is seeded per (seed, episode, query).
- `--prompt_after` turns on the grasp switch (prompt A until the chunk's gripper command is "close" for 3 consecutive queries,
  then prompt B). This is `program.ProgHooks`' rule.

Smoke test: 4 episodes of green block -> plate with A = "stack the green block on the yellow block" and B = "put carrot on plate".
The switch fired at query 5 to 7 in every episode, and all 4 succeeded (a smoke test only, not a read). One 60-step episode
takes 8 to 9 s.

### pi0 alone (bridge_beta, fp32), 24 layouts x 10 seeds = 240 episodes per task (ledger `x2_simpler_zs`, tag pi0, key task)

| task | success | grasped (ever) | moved the correct object | wall |
|---|---|---|---|---|
| G1 green block -> plate | 78.3 +- 2.7 | 86.7 | 93.3 | 38 min |
| G1 carrot -> yellow block | 27.1 +- 2.9 | 79.6 | 83.3 | 38 min |
| G1 carrot -> basket (sink) | 68.3 +- 3.0 | 81.2 | 93.8 | 75 min |
| G4 Pepsi can -> towel | 0.8 +- 0.6 | 25.4 | 97.5 | 43 min |
| G4 Red Bull can -> plate | 1.2 +- 0.7 | 26.2 | 97.5 | 41 min |
| G4 7up can -> towel | 0.0 | 34.6 | 97.9 | 44 min |
| stock carrot on plate (published 55.8) | 56.7 +- 3.2 | 73.8 | 82.5 | 43 min |
| stock spoon on towel (84.6) | 81.7 +- 2.5 | 92.1 | 91.7 | 44 min |
| stock stack cube (47.9) | 47.5 +- 3.2 | 87.9 | 91.2 | 40 min |
| stock eggplant in basket (85.4) | 83.8 +- 2.4 | 96.7 | 98.3 | 69 min |

The driver reproduces the authors' own evaluation on the 4 stock tasks, with every cell within 1.2 SE. The G1 tasks are far from
floor: pi0 already solves green block -> plate and carrot -> basket most of the time zero-shot, so they leave little headroom.
The G4 cans are at floor, and the failure is physical as well as semantic. The policy reaches the can and knocks it over
("moved the correct object" 97 to 98%). The light, standing can then tips and often rolls off the table (video
`results/x2_simpler/zs/zs_X2G4PepsiOnTowel-v0_ep0_s0_0.mp4`). A real grasp ("grasped") happens in only 25 to 35% of episodes.
A composition arm on these tasks would be fighting the can's physics as well as the missing word. Possible fixes, each a change
to pre-register: a heavier can (about 0.1 to 0.2 kg), a can laid on its side (Google's `laid_vertically` / `lr_switch`
orientations), or a scripted-grasp feasibility check first.

**Grading caveat found while choosing sources.** In `x2_bridge_grades.py` the `coke|cola` regex matches "colander" and
"chocolate". "coke can -> towel (27)" and "coke can -> plate (1)" are therefore false positives: no Bridge instruction names a
coke, cola, pepsi, red bull, 7up, sprite or soda. A generic "can" is named often, though, including 107 can + cloth/towel
co-mentions ("move the can beside the red towel"). So the G4 cans are unseen brands of a seen category, a seen can-towel
relation (moves near, never "put on"), and no can-on-plate instruction (only "place the tall can in the middle of the left
heating plate"). The `apple` pattern also matches "pineapple".

### Grasp-switch sources per graded pair (prompt A until the grasp, prompt B after)

| pair | A (object source) | B (goal source) |
|---|---|---|
| green block -> plate | "stack the green block on the yellow block" (stock) | "put carrot on plate" (stock) |
| carrot -> yellow block | "put carrot on plate" (stock) | "stack the green block on the yellow block" (stock) |
| carrot -> basket (sink) | "put carrot on plate" (stock; table scene) | "put eggplant into yellow basket" (stock; sink scene) |
| Pepsi can -> towel | a Bridge can instruction, e.g. "put can in pot" (no stock task grasps a can) | "put the spoon on the towel" (stock) |
| Red Bull can -> plate | "put can in pot" (Bridge) | "put carrot on plate" (stock) |
| 7up can -> towel | "put can in pot" (Bridge) | "put the spoon on the towel" (stock) |

Bridge alternatives for B: "put eggplant on towel" and "put knife on towel" (towel); "put spoon on plate" and "put banana on
plate" (plate; "put carrot on plate" is itself a Bridge instruction). For green block -> plate, Bridge's own "put the green block
on the table" is another A. Every source above should get its own zero-shot run (the stock four are in the table above) before
the composition read.

## Read X2: the grasp switch on Bridge (pre-registered: experiments/prereg/x2_bridge_switch.py; 2026-10-20)

pi0 = open-pi-zero bridge_beta (float32), run with the stock protocol: 24 layouts x flow-noise seeds 0-9 = 240 episodes per task
and arm. Episodes pair by (layout, seed) with x2_simpler_zs. Rows are in ledger `x2_bridge_switch` (tag pi0, keys task and arm)
and the read is in tag `read`, produced by `experiments/x2_bridge_read.py`.

The G4 can variant is the registered one: each can's density is raised so its SAPIEN mass is 0.15 kg (Pepsi 368.2, Red Bull
419.3, 7up 381.8 kg/m^3; measured by `experiments/x2_can_mass.py`). The env ids are `X2G4*Heavy-v0` in
`experiments/x2_simpler_graded.py`. Re-validation: success at reset 0/24 in all three; after placing, Pepsi 24/24, 7up 24/24 and
Red Bull 19/24 (the tall can tips when dropped onto the plate); cans stand upright (cosine >= 0.9998); mass 0.150 kg.

| task | switch | alone | discordant (switch only / alone only) | sign p | grasped: switch / alone | switch fired (median query) |
|---|---|---|---|---|---|---|
| G1 green block -> plate | 191/240 | 188/240 | 32 / 29 | 0.80 | 80.8 / 86.7 | 100% (6) |
| G1 carrot -> yellow block | 24/240 | 65/240 | 14 / 55 | 6.9e-7 | 46.7 / 79.6 | 49% (6) |
| G1 carrot -> basket | 61/240 | 164/240 | 14 / 117 | 2.0e-21 | 61.3 / 81.2 | 78% (11) |
| G4 Pepsi -> towel (0.15 kg) | 0/240 | 2/240 | 0 / 2 | 0.50 | 24.6 / 39.2 | 11% (12) |
| G4 Red Bull -> plate (0.15 kg) | 0/240 | 3/240 | 0 / 3 | 0.25 | 14.2 / 32.9 | 11% (12) |
| G4 7up -> towel (0.15 kg) | 0/240 | 1/240 | 0 / 1 | 1.0 | 11.7 / 49.2 | 12% (10) |

**H1 fails.** Pooled over the 720 G1 pairs, the switch gets 276 and pi0 alone gets 417, a difference of -19.6 pp (task-level
bootstrap 95% [-41.9, +0.3]). The discordant pairs are 60 vs 201 (sign p 5.8e-19), in the opposite direction from H1. The switch
ties on green block -> plate and loses clearly on the two carrot tasks.
- Under prompt A ("put carrot on plate"), the policy closes the gripper on the carrot for 3 queries only 49% of the time on the
  yellow-block table, and only 78% of the time in the sink scene, where that prompt was never trained.
- Grasp rates drop by 20 to 33 points against the task's own instruction.

**G4 (descriptive).** The 0.15 kg cans do not lift the floor: pi0 alone succeeds in 2, 3 and 1 of 240 episodes, and the switch
in 0. pi0 alone grasps the heavier cans more often than the light ones (33 to 49%, against 25 to 35%). Under "put can in pot" the
switch fires in only 11 to 12% of episodes.
