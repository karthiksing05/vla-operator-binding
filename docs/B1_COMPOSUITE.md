# B1: CompoSuite (setup, data, split, policy, parity, fine-tuning, cost)

Step 1 of docs/DESIGN.md section 6. Live state: results/b1/STATUS.md.

## 1 Setup (verified 2026-10-21)

| item | value | source |
|---|---|---|
| CompoSuite | github.com/Lifelong-ML/CompoSuite, cloned to `/coc/flash10/ksingara3/robot/composuite/src/CompoSuite`, installed editable (its subpackages have no `__init__.py`, so a non-editable install drops `composuite.tasks`) | README; setup.py read before install |
| robosuite / MuJoCo | robosuite 1.4.0 (the pin in CompoSuite's `requirements_default.txt`), mujoco 2.3.7 bindings, gym 0.25.2, numpy < 2 | requirements_default.txt |
| venv | `/coc/flash10/ksingara3/robot/composuite/.venv` (py 3.10; CompoSuite requires < 3.12); torch 2.4.1+cu121 in the same venv | `slurm/jobs/b1_install.sbatch` |
| runner | `slurm/jobs/b1_cpu.sbatch` (python -I; add `--gres=gpu:a40:1` for GPU) | |
| env | `composuite.make(robot, obj, obstacle, objective, use_task_id_obs=True)`, joint-position controller, horizon 500 | `composuite/env/main.py` |
| obs (93) | object-state 0-14, obstacle-state 14-28, goal-state 28-45, task indicator 45-61 (object, robot, obstacle, subtask one-hots), robot0 proprio 61-93 | `results/b1/env_smoke_egl.json` (env's own observation_positions) |
| action (8) | 7 joint targets + gripper | |
| stepping | 72-83 env steps/s per CPU core with random actions (a 500-step episode is about 6.5 s) | env smoke on node eva |
| rendering | offscreen rendering fails on CPU nodes (EGL needs a GPU; OSMesa is not installed). The datasets are state-only, so B1 needs no rendering. EGL on a GPU node is untested. | |

The task indicator in the observation is a factor label. Every B1 policy drops it (`lifelong_ood/b1_data.strip_task_id`).
The loader asserts that those 16 dims equal the cell's indicator, in the env's own order, before dropping them.

## 2 Data

Hussing, Mendez et al., *Robotic Manipulation Datasets for Offline Compositional RL* (RLJ 2024, arXiv 2307.07091). Dryad
doi:10.5061/dryad.9cnp5hqps, version 2024-06-06.
- Expert data: four per-robot tarballs, `expert-{iiwa,jaco,kinova3,panda}-offline-comp-data.tar.gz`, 15.85-15.96 GB each,
  63.58 GB in total. Each holds 64 task folders `<Robot>_<Object>_<Obstacle>_<Objective>/data.hdf5`. There are no
  per-task files, so all four tarballs are needed, since every value appears in training.
- Per task (Dryad README): observations 1e6 x 93, actions 1e6 x 8, rewards, terminals, timeouts, infos. That is 2000
  episodes x 500 steps from one compositional PPO agent ("trained to achieve 90% success on every task"). The
  observations are state vectors, with no images. Success means any state of the episode had reward 1.
- Conversion (`experiments/b1_fetch.py`): stream each tarball once and keep the first 200 episodes per task in float32
  (`/coc/flash10/ksingara3/robot/data/composuite/expert/<cell>/episodes.npz`, about 10 GB). Raw hdf5 files are deleted
  after conversion, and the tarballs are kept in `raw/`.
- **Access (blocking, 2026-10-21):** every Dryad API file download answers 401 "Unauthorized, must have current
  bearer token", from the login node and from compute nodes. The web `file_stream` route answers 403 from the AWS WAF.
  A Dryad API account is needed: log in to Dryad once, then ask help@datadryad.org for API access. Put
  `{"client_id", "client_secret"}` in `~/.dryad_api.json`. The token comes from `POST https://datadryad.org/oauth/token`
  with grant_type=client_credentials and lasts 10 hours. Instead, direct download URLs can be passed with `--url-file`.

## 3 Split (frozen 2026-10-21; `experiments/b1_split.py`; ledger b1_split)

- Held-out values (never in training): object **Hollowbox** and obstacle **ObjectWall**. These are two of the paper's
  four "restricted" elements, chosen before any result.
- Base cells (no held value): 4 x 3 x 3 x 4 = 144.
  - **G-rec** = 36 of them. A cell is G-rec when the sum of its seed-permuted value indices is 0 mod 4, which gives a
    balanced quarter.
  - **train** = the other 108. In training, every seen value appears 27 times and every pair of seen values on two
    axes co-occurs (0 pairs missing). A G-rec cell is therefore a new 3- or 4-way combination.
- **G-new** = exactly one held value: 48 Hollowbox + 48 ObjectWall = 96.
- **G-new2** = both held values: 16.
- **dev / sealed**: each stratum is split 2/3 / 1/3 with seed 20261021. G-new is stratified by which value is new.

| grade | dev | sealed |
|---|---|---|
| G-rec | 24 | 12 |
| G-new | 64 | 32 |
| G-new2 | 11 | 5 |

- The sealed cells live in `results/b1/split_sealed.json`. `split.json` holds only salted hashes of them, and its sha256
  is in the ledger row.
- `lifelong_ood.b1.load_split()` never returns sealed cells. `load_sealed(prereg=...)` and the data loader `task_dir`
  open a sealed cell only when given the run_id of an `m3_prereg` row on bench composuite AND when the environment has
  `B1_UNSEAL` set to the same id.
- The hashes guard against accidents, not against a determined reader: there are only 256 cells.

## 4 Policy (`lifelong_ood/b1_policy.py`, `experiments/b1_policy.py`)

The policy is a flow-matching action-chunk transformer over the 77-dim state (the indicator removed). It follows
Diffusion Policy (Chi et al. 2023) for chunking and normalisation: H = 16, execute 8, receding horizon. It follows
conditional flow matching with the linear path (Lipman et al. 2023; rectified flow, Liu et al. 2023; as in pi0) for
the objective: t ~ U(0, 1), 10 Euler steps.
- Network: 4 blocks, d = 256, 4 heads, 19 tokens (state, task, time and 16 action tokens), EMA 0.999.
- Training: AdamW 3e-4, cosine schedule, batch 1024, 100k steps, 200 episodes per training cell.
- **free**: a learned 32-dim embedding per training task, with no factor structure.
- **oracle**: a linear map of the 16-dim factor multi-hot.
- Zero-shot on a test cell:
  - The oracle gets the cell's multi-hot.
  - The free policy gets the **nearest embedding**: the training embedding with the lowest flow loss on the cell's one
    demonstration, under common noise draws. The demonstration is used as the query only.

## 5 Parity, zero-shot, fine-tuning curves, cost

Pending the data (section 2). The parity targets come from the paper (numbers read from the arXiv PDF text,
2026-10-21). The paper's results are means of 3 seeds. Success means any step of a trajectory of at most 500 steps
reached reward 1 (paper Appendix C).

| split | method | train success | test success |
|---|---|---|---|
| uniform, 224 train / 32 test (Table 1) | BC | 0.87 ± 0.01 | 0.73 ± 0.02 |
| | compositional BC | 0.97 ± 0.01 | 0.88 ± 0.05 |
| | IQL | 0.65 ± 0.01 | 0.68 ± 0.07 |
| restricted Hollowbox (Table 2) | BC | 0.92 ± 0.03 | 0.08 ± 0.08 |
| | compositional BC | 0.97 ± 0.02 | 0.25 ± 0.05 |
| restricted ObjectWall (Table 2) | BC | 0.88 ± 0.04 | 0.02 ± 0.03 |
| | compositional BC | 0.99 ± 0.01 | 0.10 ± 0.01 |

The paper's BC runs on the observation **with** the 16-dim task indicator (`use_task_id_obs: True` in its
evaluate.py). It is therefore the counterpart of our ORACLE policy, not of the free-embedding one. It is a 2 x 256 MLP
with batch size #tasks x 256 and lr 1e-3 (d3rlpy; paper Appendix B). The paper's restricted setting keeps one training
task that contains the element, while ours keeps none. Its restricted test numbers are therefore an optimistic
reference for our G-new.

Fine-tuning arms (`experiments/b1_finetune.py`):
- The one-demo arms run on every dev cell from the free base: LoRA r4, LoRA r16, full SFT.
  - Each arm trains a new task code z, initialised at the nearest embedding.
  - Checkpoints at 0/25/50/100/200/400/800 steps, 20 episodes each.
- Multi-task SFT on all training cells plus the cell's 200 demonstrations is the ceiling, and τ = 0.8 x its success.

Pipeline smoke test (`experiments/b1_pipeline_smoke.py`, `results/b1/pipeline_smoke.json`): with the 3.39M-parameter
policy in the loop, a failed 500-step episode takes 10 CPU-s (50 steps/s). The fine-tuning arms train in about
165 CPU-s for 800 steps at batch 64. Cost estimate: about 10 CPU-s per failed episode. The one-demo curves need 99 cells x 3 arms x 7
checkpoints x 20 episodes = 41,580 episodes, which is at most about 115 CPU-hours (less with early stop at success).
All code paths (train, resume, nearest embedding, LoRA, full SFT, multi-task SFT, rollouts) ran end to end on
random-action episodes from the real env. That is a code check, not a result.
