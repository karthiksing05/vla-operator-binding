# CALVIN-C status (plan v4 P0 / P1; docs/PLAN_V4.md 3.2)

Updated 2026-10-22.

## P0: the split (DONE, 2026-10-22)

`experiments/c_calvin_split.py split --record` (ledger `c_calvin_split`, tag frozen); guard `lifelong_ood/calvin_c.py`;
tests `tests/test_calvin_c_split.py` (10 passed). Files: `split_s{0,1,2}.json` (dev), `split_s{0,1,2}_sealed.json` (sealed
third, git-ignored, read only via `calvin_c.load_sealed(seed, prereg=...)` + `CALVINC_UNSEAL`).

Rules: 7 of the 21 block cells held out per seed; every colour, verb, place and direction in >= 2 training cells; exactly one
colour x verb (the part) held out entirely; grade = number of the cell's 3 pairwise compounds (verb-colour, verb-slot,
colour-slot) absent from training. The search maximises the smallest per-grade count, then evenness, then |H| = 7, then verb
variety, then minimises atom divergence D_A. Caveat: CALVIN colour also fixes block size (red middle, blue small, pink big).

| seed | part | grades (0/1/2, all 7) | dev cells (grade) | sealed (by grade) | D_A | D_C |
|---|---|---|---|---|---|---|
| 0 | pink x rotate | 3 / 3 / 1 | push_blue_right (0), rotate_red_left (0), push_pink_left (1), rotate_pink_right (1, part), rotate_pink_left (2, part) | 1 x g0, 1 x g1 | 0.123 | 0.288 |
| 1 | red x lift | 2 / 2 / 3 | rotate_blue_right (0), push_pink_left (1), rotate_pink_left (1), lift_red_drawer (2, part), lift_red_slider (2, part) | 1 x g0, 1 x g2 | 0.018 | 0.412 |
| 2 | blue x push | 3 / 3 / 1 | push_pink_right (0), rotate_pink_left (0), lift_red_drawer (1), push_blue_right (1, part), push_blue_left (2, part) | 1 x g0, 1 x g1 | 0.123 | 0.288 |

Structural notes:
- A held-out lift cell always has an unseen colour-place compound (each colour-place pair is one cell), so lift cells are never
  grade 0.
- With a push or rotate part, only one grade-2 cell is possible: a second would need a second fully held-out verb-colour pair.
  So seeds 0 and 2 reach 3/3/1 and seed 1 (the lift part) reaches 2/2/3.

Training-window filters (`windows` subcommand, ledger `c_calvin_windows`, `windows_s{k}.{npz,json}`), over ABC's 17,870 annotated
64-frame windows (`lang_clip_resnet50`, identical to `lang_annotations`):

| seed | held-out windows | dropped for > 16-frame overlap | kept windows | kept sample starts | held-out frames still inside kept windows | eval chains free of held-out cells |
|---|---|---|---|---|---|---|
| 0 | 3,678 | 144 | 14,048 (78.6 %) | 697,296 / 893,043 | 2,464 / 107,792 (2.3 %) | 562 / 1000 |
| 1 | 3,676 | 251 | 13,943 (78.0 %) | 691,068 | 3,694 / 126,809 (2.9 %) | 443 |
| 2 | 3,676 | 171 | 14,023 (78.5 %) | 697,073 | 2,308 / 112,299 (2.1 %) | 574 |

The motions of the held-out cells are still present in unannotated play and inside other subtasks' windows (e.g. the colour-free
push_into_drawer / stack_block, which move every block). The recipe's language-only dataset never reads unannotated play.

## P1: FLOWER-C (IN PROGRESS)

- Data: task_ABC_D checked against the zip index (`data_check.json`): training 1,795,045 / 1,795,045 episode files present at
  full size, validation 99,022 / 99,022. The authors' `extracted/ep_rel_actions.npy` + `ep_npz_names.list` are written for
  both splits (`experiments/c_calvin_extract.py`; spot check max diff 0.0).
- flower_vla_pret @ ffd4134 downloaded (360000_model_weights.pt, the file the published run's training.log names). Loading
  it into the CALVIN FLOWERVLA reports 132 missing and 4 unexpected keys (the recipe's 18-layer DiT vs the pretrained 12).
- Recipe = the published checkpoint's own config.yaml + training.log: 4 GPUs x batch 8, 35 epochs x 1000 steps = 35k steps,
  EMA, bf16-mixed, seed 242. The README says 40k; the config and the log say 35k. Wrapper: `experiments/c_train_flower.py`
  (deviations listed in its docstring: the data filter, no in-training scene-D rollouts / no test-set model selection, a full
  resumable checkpoint, the <Flow> embedding saved, CSV logging, Lightning's local launcher, torch 2.10's TF32 API).
- Timing (job 4188948 + diagnoses 4189189 / 4189575):
  - Step time depends on the node. megabot: 4.6 s/step with 4 GPUs (now excluded). crushinator: 0.73 s/step on 1 GPU x 8,
    about 1.1 s/step on 4 x 8 with NCCL_P2P_LEVEL=SYS, about 1.1 s/step on 2 x 16.
  - Without find_unused_parameters, DDP fails: the model has unused parameters.
  - Full checkpoint: 15.2 GB, about 8 min to write (torch.save about 300 s, plus the copy), so one every 4000 steps plus one at
    the 215-min wall budget.
  - Resume tested: 200 -> 300 steps, all states restored. Export tested: 4.0 GB model.safetensors in 43 s.
- Estimated cost: 35k x 1.1 s = 10.7 h on 4 A40, about 12.5 h with checkpoints, so about 50 A40-h per run.
- Running (2026-10-22 05:25): FLOWER-C seed 0 (`flower_c_s0`, job 4189713, self-chaining) and the full-ABC control
  (`flower_abc_full`, job 4189714; the same recipe on every window). Also a 10-chain eval smoke of the step-300 export
  (job 4189715).
- 2026-10-22 10:20 (supervisor resumed after a session restart): seed 0 job 4189713 hit its wall budget at step 10,650 and
  self-chained to job 4190696 (tachikoma, step 12,000 at 10:13, about 1.05 s/step). The control job 4189714 was PREEMPTED at
  07:02 (last.ckpt at step 4,000; preemption kills the self-chain) and was resubmitted as job 4190911 (clippy, CHAIN=8,
  resumed at 4,000, about 1 s/step):
  `sbatch --partition=overcap --account=overcap --qos=short --time=04:00:00 --gres=gpu:a40:4
  --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot --export=ALL,RUN=<repo>/results/calvin_c/flower_abc_full,SEED=0,FULL=1,CHAIN=8,WORKERS=8,BUDGET=215,MAXSTEPS=0,CKEVERY=4000
  slurm/jobs/c_train_flower.sbatch` (the seed-0 run is the same with RUN=.../flower_c_s0,FULL=0).
- 13:39 seed 0 chained 4190696 -> 4192597 (step 22,600); 14:05 control chained 4190911 -> 4192766 (step 17,000, ig-88).
- 2026-10-22 USER HARD RULES: nothing on sky1 (every python call via srun/sbatch); every job names `-p overcap -A overcap`
  (+ --requeue) or `-p tail-lab -A tail-lab --qos=short`; always `--exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw`.
  Jobs 4192597 and 4192766 were set to Requeue=1 with scontrol (sky1 has no GPUs, so the a40 jobs cannot land there).
  `c_train_flower.sbatch` is in use and self-chains without sky1 / --requeue, so each chained job it submits is patched
  with `scontrol update JobId=<id> Requeue=1 ExcNodeList=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot`. NEW
  submissions use `slurm/jobs/c_train_flower_r.sbatch` (the same script, chaining with --requeue and sky1 excluded) with
  `sbatch -p overcap -A overcap --qos=short --requeue --time=04:00:00 --gres=gpu:a40:4 --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot ...`.
  The parity script runs through srun, never on sky1.
- 17:21 FLOWER-C seed 0 FINISHED (35,000 steps; chain 4189713 -> 4190696 -> 4192597 -> 4194149, see flower_c_s0/jobs.txt;
  4194149 ran 18 min on deebot before it could be patched). Post-run jobs: export 4194226, eval shards 4194227-4194230
  (eval_flower_c_s0), audit 4194231. Control at step 24,000 (job 4192766).
- 2026-10-22 21:48 control `flower_abc_full` FINISHED (35,000 steps). Job 4194308 was preempted at 19:03 and requeued
  (Requeue=1), resumed from the step-32,000 checkpoint, and finished. It was requeued once more at exit and reran as a 2-min
  no-op at step 35,000 (last.ckpt unchanged). Post-run jobs: export 4197572, eval 4197573-76 (eval_flower_abc_full), audit 4197577.
- Post-run reads (per finished run): `slurm/jobs/c_flower_export.sbatch` (RUN, TAG=final -> <run>/export_stepfinal), then
  4 x `c_flower_eval.sbatch` shards of 250 chains (NPROC=5, OUT=results/calvin_c/eval_<run>) and `c_calvin_audit.sbatch`
  (MODELS=pret:abc:<name>=<export>), both afterok on the export; then `c_calvin_parity.py --run eval_<run> --record`.

### Parity, FLOWER-C seed 0 (`experiments/c_calvin_parity.py`, `parity_flower_c_s0.json`, ledger row `c_calvin_parity__calvin_c__flower_c_s0__s0__426df7fe`)

Step-35,000 EMA export (`flower_c_s0/export_stepfinal`), the standard 1,000 ABC -> D chains (eval_flower_c_s0, 4 x 250 chains,
jobs 4194227-30, about 25 min each on deebot). Reference = flower_calvin_abc with the load fix (results/x1_calvin/flower_full_ep360_fix1).

| chains | run avg len | ref avg len | run success at 1-5 (%) | ref success at 1-5 (%) |
|---|---|---|---|---|
| all 1,000 (mixes training and held-out subtasks) | 4.269 | 4.543 | 97.0 / 91.8 / 86.3 / 79.4 / 72.4 | 99.6 / 96.9 / 92.2 / 86.0 / 79.6 |
| clean 562 (no held-out cell) | 4.527 | 4.584 | 99.5 / 96.8 / 92.7 / 85.2 / 78.5 | 99.8 / 97.5 / 93.6 / 86.8 / 80.6 |

- On the chains without held-out cells, FLOWER-C is at parity with the published checkpoint (4.53 vs 4.58). The 0.27 gap on
  all chains comes from the held-out cells, which are not scored here.
- Per-subtask success on the 27 non-held-out subtasks (D chains) is within a few points of the reference on almost every
  subtask. The largest gaps are push_pink_block_right 54/67 (81 %) vs 62/65 (95 %), lift_red_block_slider 108/135 (80 %)
  vs 131/148 (89 %), and lift_pink_block_drawer 12/14 vs 13/14 (n small). push_pink_block_right shares pink-push with the
  held-out push_pink_left cell. Full table: the script's output in logs/4194443_parity.out.
- "Training cells in ABC": the harness rolls out in scene D only (x1_calvin_flower_eval.py takes the scene-D validation
  config). The per-subtask numbers above are the training cells' success in D. An ABC-scene rollout read needs a new env
  config and was not run.
- Control `flower_abc_full` (same recipe, every window; `parity_flower_abc_full.json`, ledger
  `c_calvin_parity__calvin_c__flower_abc_full__s0__b57d001f`): all chains 4.438 (success at 1-5: 98.4 / 95.1 / 89.8 / 83.8 / 76.7)
  vs published 4.543; clean chains 4.528 vs 4.584. The reproduction comes in 0.10 under the published 4.54 (one seed, final
  weights, no best-of-test-scene selection: deviation 2).
- FLOWER-C vs control (`parity_flower_c_s0_vs_full.json`, ledger `c_calvin_parity__calvin_c__flower_c_s0_vs_full__s0__b57d001f`):
  clean chains 4.527 vs 4.528 (identical). All chains 4.269 vs 4.438: removing the held-out cells costs 0.17 average length,
  all of it on chains that contain held-out cells.
  - Per subtask (training cells, D) FLOWER-C matches the control within noise. push_pink_block_right is low in the control
    too (50/65, 77 %, vs FLOWER-C 54/67, 81 %), so it is a reproduction effect, not a split effect. The largest remaining gap is
    lift_red_block_slider, 108/135 (80 %) vs 121/136 (89 %).

### Contamination audit, seed 0 (`experiments/c_calvin_audit.py`, ledger `c_calvin_audit`, `audit_s0_pret_abc.json`)

Rectified-flow loss on ABC training windows: the 5 dev held-out cells against the 14 training block cells, 24 windows per cell
and 6 frames per window, with the same 4 (eps, t) draws for every model and instruction. Sealed cells are not read. Delta =
loss with the colour (or slot) changed in CALVIN's evaluation sentence, minus the loss with the true sentence. 95 % CIs are
cell-cluster bootstraps of held-out minus training.

| model | loss, held-out | loss, training | difference [CI] | Delta colour, H / T | Delta slot, H / T |
|---|---|---|---|---|---|
| flower_vla_pret (the recipe's step 0) | 3.414 | 2.447 | +0.967 [+0.40, +1.55] | -0.020 / -0.015 | +0.003 / +0.006 |
| flower_calvin_abc (trained on every cell) | 0.140 | 0.138 | +0.003 [-0.015, +0.020] | +0.156 / +0.185 | +0.260 / +0.114 |

- flower_vla_pret shows no sign of knowing any CALVIN cell: its field does not depend on the instruction (Delta about 0 on
  held-out and training cells alike), and its loss level is near-uninformed (6 of the 18 DiT layers are fresh at step 0).
- Its higher raw loss on the held-out cells is the cell mix (pushes and rotations vs lifts), not knowledge.
- The published checkpoint, which saw every cell, shows no held-out vs training gap and a clear language dependence: the
  "seen" reference.
- FLOWER-C seed 0 (step 35,000 EMA export; job 4194231, `audit_s0_pret_abc_flower_c_s0.json`, ledger row
  `c_calvin_audit__calvin_c__pret_abc_flower_c_s0__s0__426df7fe`; pret and abc rows reproduce the table above):

| model | loss, held-out | loss, training | difference [CI] | Delta colour, H / T (diff [CI]) | Delta slot, H / T (diff [CI]) |
|---|---|---|---|---|---|
| flower_c_s0 (held-out cells removed) | 0.159 | 0.129 | +0.030 [+0.012, +0.049] | 0.150 / 0.184 (-0.034 [-0.063, -0.007]) | 0.300 / 0.119 (+0.182 [+0.035, +0.315]) |
| flower_abc_full (control, every cell; job 4197577, ledger `..._pret_abc_flower_abc_full__s0__b57d001f`) | 0.132 | 0.132 | -0.000 [-0.017, +0.016] | 0.134 / 0.164 (-0.030 [-0.063, -0.001]) | 0.287 / 0.131 (+0.156 [-0.008, +0.309]) |
| flower_calvin_abc (same run) | 0.140 | 0.138 | +0.003 [-0.015, +0.020] | 0.156 / 0.185 (-0.029 [-0.075, +0.014]) | 0.260 / 0.114 (+0.145 [-0.016, +0.298]) |

  - FLOWER-C shows a small held-out gap that excludes 0 (+0.030; the all-cell model has none). It still fits the held-out
    windows nearly as well as a model trained on them (0.159 vs 0.140), which is consistent with the motions appearing in
    other subtasks' windows.
  - The Delta pattern (colour lower on held-out cells, slot higher) matches the all-cell model's pattern, so the Deltas do not
    separate the two models; only the loss-level gap does.
  - The control (the same recipe on every window) has no held-out gap (-0.000), so FLOWER-C's +0.030 comes from the removal.
    Its Delta pattern (colour -0.030, slot +0.156) is the same as FLOWER-C's: the Deltas are a property of the cells.

## Next (waiting for the lead's go): no composition arm and no split seeds 1-2 have been started.
