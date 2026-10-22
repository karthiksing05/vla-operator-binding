# B1 (CompoSuite) status

Updated after every milestone (sessions restart). Design: docs/DESIGN.md section 6, step 1. Write-up: docs/B1_COMPOSUITE.md.

## Done
- 2026-10-21 (a) venv `/coc/flash10/ksingara3/robot/composuite/.venv` (py3.10, robosuite 1.4.0, mujoco 2.3.7,
  gym 0.25.2, torch 2.4.1+cu121, CompoSuite editable from `robot/composuite/src/CompoSuite`). Install: `slurm/jobs/b1_install.sbatch`.
- 2026-10-21 (a) stepping verified on CPU compute nodes (eva): obs 93-dim, act 8, ~75-83 env steps/s with random
  actions (`results/b1/env_smoke_egl.json`). Offscreen rendering FAILS on CPU nodes (EGL needs a GPU; OSMesa absent).
  Not needed: the datasets are state-only. EGL on a GPU node is untested (the GPU queue was full).
- 2026-10-21 (c) split frozen: `results/b1/split.json` + sealed `results/b1/split_sealed.json`; ledger
  `ledger/b1_split/`. 108 train, dev G-rec 24 / G-new 64 / G-new2 11, sealed 12 / 32 / 5. Held: Hollowbox, ObjectWall.

- 2026-10-21 code ready and smoke-tested end to end on random-action episodes from the real env (not results):
  `lifelong_ood/{b1,b1_data,b1_policy,b1_eval}.py`, `experiments/b1_{fetch,policy,eval_policy,finetune}.py`;
  launcher `slurm/jobs/b1_launch_stage1.sh fetch|policy|eval|finetune`. The task-indicator layout assertion passes
  on real observations. Resume works (oracle 200 -> 300 steps). With the policy in the loop a failed episode is 10 CPU-s.
  `b1_fetch.py` conversion and `b1_eval_policy.py` have not run yet (they need the data).
- Compute so far: about 0.5 CPU-h, 0 A40-h.

## Blocked
- (b) Dryad downloads need API credentials. `api/v2/files/<id>/download` answers 401 "must have current bearer token";
  the web route answers 403 from the AWS WAF. The user must put `{"client_id": ..., "client_secret": ...}` in
  `~/.dryad_api.json` (chmod 600; request API access from help@datadryad.org after logging in to Dryad once), or
  supply direct URLs (`--url-file`). Then run `experiments/b1_fetch.py --robot <r>` for the 4 robots (SLURM, CPU).

## Next
0. user: Dryad API credentials
1. fetch + convert (4 CPU jobs, 64 GB download, ~10 GB kept)
2. `experiments/b1_policy.py --cond free` and `--cond oracle` (GPU)
3. parity rollouts on train cells + zero-shot on dev cells
4. fine-tuning curves (`experiments/b1_finetune.py`), ceiling tau
