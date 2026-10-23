# LIBERO-90-C status (P0 split + audit, P1 pi05_l90c)

Read this first after a restart. Plan: docs/PLAN_V4.md 3.1; details: docs/L90C.md.

## Files
- experiments/c_split.py (MCD split; --dry / --record), lifelong_ood/l90c.py (guarded loader), tests/test_c_split.py
- results/l90c/split.json (dev), results/l90c/split_sealed.json (SEALED, never commit, never read)
- experiments/c_data.py (--fetch-base, --plan, --convert, --norm-stats), lifelong_ood/policy/l90c_config.py (pi05_l90c,
  pi05_l90c_lora, pi05_base_libero TrainConfigs, registered at runtime; backbone.py calls register())
- experiments/c_audit.py (--convert pi05_base -> PyTorch; --score [--record])
- slurm/jobs/l90c_cpu.sbatch (CPU, ENV=openpi|chunks)

## Milestones
- 2026-10-22 02:19  pi05_base downloaded: openpi_cache/openpi-assets/checkpoints/pi05_base (12.4 GB). Job 4188046.
- 2026-10-22 02:20  pi05_base -> PyTorch conversion submitted (job 4188100) -> openpi_cache/.../pi05_base_pytorch
- 2026-10-22 02:32  SPLIT FROZEN (job 4188142): results/l90c/split.json + split_sealed.json; ledger
  c_split/c_split__libero_90_c__frozen__s20261022__9149bef8.json. tests/test_c_split.py 6 passed (job 4188143).
  train 76 = 44 LIBERO-90 + 32 suite; bins 8/8/8 (dev 5/5/5, sealed 3/3/3); part 21 (dev 14, sealed 7).
  Bin D_C 0.139 / 0.379 / 0.588. D_A 0.137 (null mean 0.180, p05 0.132): 0.02 infeasible (see docs/L90C.md 1).
- 02:31  pi05_libero 40-task reference parity read launched: jobs 4188238-4188241 (results/l90c/parity_pi05_libero_*.json)
- 02:34  data conversion running (job 4188153; 76 tasks, 3359 episodes) -> norm stats (4188156) -> 200-step timing (4188165)
- 03:09  pi05_base -> PyTorch converted on a CPU node instead (job 4188949; GPU job 4188100 cancelled while pending):
  openpi_cache/openpi-assets/checkpoints/pi05_base_pytorch (assets = pi05_libero's LIBERO norm stats; base's own in assets_pi05_base)
- 04:03  AUDIT DONE (job 4188950): ledger c_audit/c_audit__libero_90_c__pi05_base__s0__9149bef8.json, results/l90c/c_audit.json.
  Flow loss (real dims, 2 demos x 16 frames x 4 draws per task): pi05_base train 0.353 / dev test 0.353 / dev part 0.339 /
  l40 0.377; pi05_libero train 0.356 / test 0.352 / part 0.286 / l40 (its own data) 0.014. -> no differential exposure
  (test-train +0.0001, AUC 0.43) and base sits 25x above the seen level on LIBERO-90 and ON LIBERO-40 alike: clean.

- 04:40  REFERENCE LEVEL READ (jobs 4188238-41, ledger c_parity tag pi05_libero): pi05_libero on its 40 tasks, 20 eps each
  (init 0-19, MODE=ours): 0.969 mean (775/800), median 1.0, min 0.55; spatial 0.985, object 0.99, goal 0.97, libero_10 0.93.
- 05:12  DATA BUILT (job 4188153, 2h39 CPU): /coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_l90c, 3359 episodes,
  495,313 frames, 76 tasks (manifest results/l90c/data_manifest.json); 0 text mismatches IPEC vs benchmark.
- 05:13  norm stats (job 4188156): /coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_l90c/lol/libero_l90c
- 06:02  TIMING b32 (job 4188165, 4 x A40 FSDP 4): 11.41 s/step steady (median of 190), startup 7 min, peak 26.9 GB/GPU,
  checkpoint 42 GB saved in ~70 s. => 12.7 A40-h per 1k steps; 30k steps = 380 A40-h (95 h wall). Ckpt jax/pi05_l90c/timing_b32/199.
- 06:07  export of timing ckpt -> pytorch/pi05_l90c_timing works (job 4189863, CPU 2 min). Smoke rollout job 4189883 (2 tasks x 2).
- 06:22  smoke (job 4189883): PolicyBackbone('pi05_l90c', ckpt pytorch/pi05_l90c_timing) served via roll_pack, 4 eps, 0/4 (200
  steps; pipeline check only). ledger c_smoke.
- 06:33  TIMING b64 (job 4189864): 13.34 s/step, peak 26.9 GB -> 4.8 samples/s (1.7x b32) = 14.9 A40-h / 1k steps.
- 06:3x  MAIN RUN LAUNCHED: exp `main`, batch 64, 20k steps max (constant lr; stop at first kept ckpt reaching parity), save
  every 500, keep every 5000. Chain 4189987 -> 88 -> 89 -> 90 -> 91 -> 92 (afterany; each resumes). Command:
  experiments/c_train_pi05.py --exp main --steps 20000 --batch 64 --save-interval 500 --keep-period 5000 (l90c_train.sbatch, 4 x A40)
  Ckpts: /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_l90c/main/<step>. Timing ckpts deleted.
- 06:49  protos index for l90c (pp_prep.py index_l90c; job 4189998): results/protos/index_l90c.json: 310 train obs (32 suite
  tasks + 30 LIBERO-90 training cells with raw matches; 62 of 73 instructions), 56 l90 obs (14 DEV held-out tasks), 120 transfer.
  Poses (job 4189999) done; only tr38 (no raw match) lacks poses, as in index.json. pp_fields/pp2/pp3 pick index_<tag>.json when
  it exists (additive edits); pp_prep poses take --tag=.
- 06:51  pi05_l90c_fac pipeline check (job 4190000): transforms [FactoredPrompt, LiberoInputs], prompts as intended. NOT launched.
NEXT: at the first kept checkpoint (5000): export (c_train_pi05.py --exp main --export --step 5000 --tag 5000), parity read
  (c_parity.py --model pi05_l90c with BACKBONE_CKPT) and pp_fields --config pi05_l90c --checkpoint <dir> --tag l90c (--prep-binds
  first, then 4 shards), pp2_dev.py --tag=l90c --record, pp3_dev.py --tag=l90c --record.
- COORDINATOR (06:xx): add pp pre-checks on the first usable l90c checkpoint (pp_fields --checkpoint --tag l90c, 4 shards; pp2_dev /
  pp3_dev --tag=l90c --record; pp_prep's index must be adapted to l90c's training cells additively) and PREPARE (do not launch)
  pi05_l90c_fac (factored conditioning: p=0.3 part prompt, p=0.1 null). Config written: l90c_config.py pi05_l90c_fac; table
  results/l90c/fac_prompts.json (73 training texts; checked: 60/10/10/10/10 split of instr/null/obj/goal/verb on a test text).
- 10:20  SUPERVISOR RESUMED (new agent after session restart): chain intact, 4189987 running on dave (step ~930, 13.3 s/step,
  ckpt 500 saved), 4189988-92 pending afterany. ~1000 steps/job -> 5000 in job 4189991 (~+16 h). Chain ends near 6000: extend
  only if parity at 5000 fails. Decision rule (lead): parity >= 0.9 x 0.969 = 0.872 at 5000 -> stop the chain there; else go to 10000.
  pi05_l90c_fac stays NOT launched (lead decides after the pp read).
- 11:09  4189987 TIMEOUT at step ~1050 (ckpt 1000 kept); 4189988 resumed from 1000 on heistotron (14.5 s/step there).
- 13:50  USER HARD RULES 2026-10-22: nothing (not even python JSON reads) on sky1; every job -p overcap -A overcap (+--requeue) or
  -p tail-lab -A tail-lab --qos=short; always --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw. Pending chain jobs
  4189989-92 updated in place (scontrol update ExcNodeList, adds sky1; deps intact; .sbatch untouched). c_parity.py EXCLUDE now
  includes sky1. RESUBMIT COMMAND (if the chain breaks): sbatch --partition=overcap --account=overcap --qos=short --time=04:00:00
  --gres=gpu:a40:4 --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw [--dependency=afterany:<last>]
  --export=ALL,SCRIPT="experiments/c_train_pi05.py --exp main --steps 20000 --batch 64 --save-interval 500 --keep-period 5000"
  slurm/jobs/l90c_train.sbatch
- 14:55  4189988 TIMEOUT on heistotron at ~step 1980: its 2000 save never landed -> only ckpt 1500 kept (~480 steps lost).
  4189989 resumed from 1500 (heistotron, 14.5 s/step). PROBLEM: at 14.5 s/step a 4-h job does ~950 steps after startup, so with
  --save-interval 500 every job nets only 500 steps (dave at 13.3 s/step only just made it: 1050). The current chain then ends at
  ~3500 (4189989 -> 2000, 90 -> 2500, 91 -> 3000, 92 -> 3500); 5000 needs 3 more jobs, i.e. ~28 h more instead of ~16 h.
  PROPOSED (not done, awaiting the lead): replace pending 4189990-92 with the same command at --save-interval 250 (train.py
  saves on step % save_interval in the loop; the orbax manager has no interval, max_to_keep=1 + keep_period=5000, so a resume
  with 250 is safe and 5000 is still kept; overhead ~70 s async per 250 steps, ~2%). That nets ~750-950 steps/job.
  My scancel+resubmit was refused by the session's permission classifier, so the chain is UNCHANGED (verify with squeue).

## Findings so far
- Atom divergence <= 0.02 is NOT feasible on LIBERO-90 at task granularity: random 24-task test sets give D_A 0.18 mean
  (min 0.11 of 200); a swap-local optimum reached 0.117. The measure's sampling floor at ~130 test atom tokens over ~70 atom
  types is ~0.1. The split minimises D_A and reports it beside the null distribution.
- IPEC LeRobot port: task 51 (butter in basket, LIVING_ROOM_SCENE2) has 0 episodes; other tasks have 28-50 (not 50).
- 15:40  LEAD (user approved scancel): cancelled pending 4189990-92; new chain at --save-interval 250 (otherwise identical, overcap,
  --requeue, sky1 excluded): 4193710 -> 4193711 -> 4193713 -> 4193715 -> 4193717, first afterany:4189989 (running, interval 500,
  will save 2000). Expect ~800 steps/job -> 5000 in about the 4th new job.
- 2026-10-22 15:20 (real clock, from date)  SUPERVISOR 2 took over: 4189989
  running on heistotron (resumed from 1500, step ~1590, 14.4 s/step; will save 2000), new chain 4193710-17 pending afterany
  at --save-interval 250. Plan at 5000: export (l90c_cpu.sbatch) -> c_parity --model pi05_l90c --ckpt
  .../pytorch/pi05_l90c_5000 --tag pi05_l90c_5000 -> pp_fields --config pi05_l90c --checkpoint <same> --tag l90c
  (--prep-binds, then 4 shards) -> pp2_dev/pp3_dev --tag=l90c --record.
- 15:35  NOTE: interactive srun must use the default (debug) QOS: `srun -p overcap -A overcap -c 2 --mem=8G -t 00:20:00
  --exclude=...` works (srun has no --requeue; --qos=short is refused for interactive jobs). Compute nodes have sbatch:
  c_parity --launch --dry ran inside srun (job 4193740, asimo): 8 packs x 2 instances for the 76 training tasks.
- 18:53  4189989 TIMEOUT (normal) at ~2480 on heistotron; ckpt 2000 kept (only dir). 4193710 PENDING (Priority), resumes from 2000 at interval 250.
- 10-23 06:39  LEAD: 4193710 hit REQUEUE_HOLD (launch_failure_limit, 5 restarts): every overcap run was preempted ~50 min in,
  before its next 250-step save (~61 min at 14.5 s/step), so nothing past 2250 was kept. Cancelled 4193710-17; new chain at
  --save-interval 100 (~24 min per save; otherwise identical): 4201198 4201199 4201200 4201201 4201202 4201203 4201204 4201205. If REQUEUE_HOLD recurs: scontrol release <id>.
- 10-23 06:42  SUPERVISOR 2: waiter re-armed on 4201198-205 (also exits on REQUEUE_HOLD -> scontrol release). Last kept ckpt 2250.
- 10-23 09:20  SUPERVISOR 3 took over (previous lost in a session restart): 4201198 running on dendrite (resumed 06:41, 5.4 s/step
  there, ckpt 3900 saved 09:14, step ~3950); 4201199-205 pending afterany. 5000 expected ~10:55 (in 4201198 if it survives to its
  10:41 limit, else in 4201199). Waiter re-armed (exits on 5000 dir, REQUEUE_HOLD, or a chain job ending). Plan at 5000: export
  (l90c_cpu.sbatch, --export --step 5000 --tag 5000 -> pytorch/pi05_l90c_5000) -> c_parity --launch --model pi05_l90c --ckpt <that>
  --tag pi05_l90c_5000 (afterok) ; pp_fields --prep-binds then 4 shards (robot_gpu.sbatch, --config pi05_l90c --checkpoint <that>
  --tag l90c) -> pp2_dev/pp3_dev --tag=l90c --record (cpu.sbatch).
- 10-23 12:03  STEP 5000 KEPT: jax/pi05_l90c/main/5000 finalized 12:00:57 (4201198 was preempted/requeued ~10:40, resumed from
  4700 on randotron, 14.5 s/step; still running, 4201199-205 pending). Submitted: export 4204108 (l90c_cpu, -> pytorch/pi05_l90c_5000)
  -> parity launcher 4204109 (c_parity --model pi05_l90c --launch --ckpt pytorch/pi05_l90c_5000 --tag pi05_l90c_5000, afterok)
  ; pp_fields --prep-binds 4204110 (afterok export) -> shards 4204111-14 (--config pi05_l90c --checkpoint <same> --tag l90c)
  -> pp2_dev 4204115 / pp3_dev 4204116 (--tag=l90c --record, cpu.sbatch tail-lab).
- 10-23 12:08  export 4204108 OK (pytorch/pi05_l90c_5000, 2.5 min). Parity packs 4204127-34 (8 x 2 instances, 76 tasks x 20 eps,
  tag pi05_l90c_5000). pp_fields binds 4204110 running (crushinator).
- 10-23 12:25  pp_fields binds 4204110 done (18 min); shards 4204111-14 running (crushinator/omgwth/qt-1/protocol).
- 10-23 12:55  PARITY @5000 (packs 4204127-34, read 4204864; ledger c_parity/c_parity__libero_90_c__read_pi05_l90c_5000__s7__e9b02fe8.json,
  per-pack rows tag pi05_l90c_5000): pi05_l90c 0.959 mean over 76 tasks x 20 eps (1520), CI95 [0.941, 0.974], median 1.0, min 0.65;
  libero_90 0.966, spatial 0.95, object 0.967, goal 0.938, libero_10 0.93. Reference pi05_libero 0.969. Same 32 suite tasks: 0.948 vs 0.977.
  Below level-0.2: libero_10:9, libero_90:23, libero_90:26. DECISION RULE (>= 0.9 x 0.969 = 0.872): MET -> chain MAY stop at 5000
  (not cancelled; lead decides). Strict parity (>= 0.969, the read's `reached`) is false.
- 10-23 12:56  LEAD: parity 0.959 >= 0.872 -> chain stopped at 5000 as pre-agreed (scancel 4201205 4201204 4201203 4201202 4201201 4201200 4201199 4201198 ). Step-5000 export is the model of record.
- 10-23 13:50  P2/P3 ON pi05_l90c@5000 (binds 4204110, shards 4204111-14: 478 obs in results/protos/obs_l90c; 8 transfer obs of
  libero_90:48 skipped, its first-frame file holds 2 states). pp2 4204115 -> ledger pp2_dev/pp2_dev__libero__l90c__s0__e9b02fe8.json;
  pp3 4204116 -> ledger pp3_dev/pp3_dev__libero__l90c__s0__e9b02fe8.json. vs pi05_libero (ledger *__pi05_libero__s0__9149bef8):
  P2 separation frac>2 at best u<=0.35: 0.68 (go flag true; base 0.87, also flagged true) but distinct core-mode clusters at tol 0.05
     stay ~2.3 of 9 (base ~2.3): still degenerate. Attribution obj parts mean 1.36x chance (base 1.17), named 1.74x (base 1.36),
     null->named 1.52x (base 1.36), own goal 2.43x (base 2.41) vs null->goal 2.36x (base 2.29): NO GO (named < 2x; goal at null level).
     Transfer r best 0.026 (base 0.071): NO GO.
  P3 (best variant plain): R0 0.604 (base 0.513) vs Comp-3 0.356 (base 0.588), equal 0.317, R1 0.602, ceiling 0.954; R0 beats Comp-3
     now (base did not). Stability plain 0.967 / tmpl 0.921 (base 0.857 plain). R0 NO GO (< 0.70); stability GO. By suite (plain R0):
     libero_90 0.71, goal 0.79, object 0.79, libero_10 0.47, spatial 0.29. 62 tasks / 310 obs (base 40 / 200).
