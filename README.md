# lifelong-ood-learner

**Compositional continual learning for a frozen robot policy.** A new task arrives with its instruction, its scene and at most ONE
demonstration ([docs/THE_TASK.md](docs/THE_TASK.md)). The frozen policy (pi0.5 on LIBERO-90) takes it on by composing small,
shared prototypes it has learned from earlier tasks; the demonstration is requested only when what the learner already holds fails
its own practice, and it updates every concept it belongs to. After Wang et al. (prototypes of a frozen generative model, selected
and composed per out-of-distribution task) with the mechanism they lack: new primitives detected by practice, consolidated into
concepts, reused by later tasks.

## The method ([docs/METHOD.md](docs/METHOD.md))

- **Factor every step at the policy's own grasp.** Before the grasp an OBJECT prototype acts (how to reach and grasp this object;
  an INSTANCE child when the instruction names one, "the middle black bowl"); from the grasp to the release a GOAL prototype acts
  (where to carry it). Each is a rank-4 LoRA on the action expert, consolidated over every demonstration of its concept.
- **Keep only what practice validates.** Ten practice episodes decide between the policy and the composition; the policy is kept
  wherever the composition does not beat it.
- **Learn as a stream.** An arrival composes from the current library; it asks for its one demonstration only when neither the
  policy nor the composition reaches 8 of 10 in practice; the demonstration refits every concept on its path.

## Where it stands (every number from `ledger/`, audited)

| claim | evidence |
|---|---|
| composition beats the policy on tasks it fails | kept library 499 of 600 against 273 zero-shot, on start states no design decision saw (`ledger/e3_hier`) |
| close to a separate adapter per task, at a fifth of the storage | 480 against 494 of 580 for the per-task LoRA with the same acceptance test (difference -0.024, 95% interval -0.074 to 0.024); 22 prototypes, 102.9 MB, against 29 adapters, 545.4 MB. Non-inferiority at the margin of 0.05 is NOT established |
| learned as a stream, nothing damaged | the request rule of the thesis: 17 and 16 demonstrations in two orders (21 and 20 under the earlier rule); no damage to earlier arrivals in either order (`ledger/w1_stream`) |
| replication on 37 new tasks | 451 of 740 against 151 zero-shot; 451 against 441 for the per-task LoRA with acceptance; pooled over 66 tasks 931 against 935 of 1,320 (difference -0.003, 95% interval -0.063 to 0.058): level, non-inferiority at 0.05 just missed (`ledger/e5_replicate`) |
| recombination of concepts learned only from OTHER tasks (5-fold held out, 53 tasks) | alone, level with the policy (365 against 354 of 1,060: the pre-registered test fails); with practice choosing, 484 against 354; on the tasks the policy fails, 145 against 25 (exploratory); factorisation beats a monolithic adapter on the same data, 365 against 190 (`ledger/g0_recombine`); summing donor adapters collapses (71), averaging them stays at the policy's level (328) (`ledger/g2_baselines`) |
| a second policy (SmolVLA) | the same library refit on SmolVLA: 203 of 600 against 20 for SmolVLA alone; 189 against 274 of 580 for its per-task LoRA with acceptance (`ledger/g3_crossmodel`) |
| limits | a new object needs its one demonstration (object prototypes do not transfer between objects, `ledger/t0_transfer`); selecting prototypes by a demonstration's likelihood, as Wang et al. do, is at chance here (`ledger/w0_select`) |

## How to view everything

| what | where |
|---|---|
| the readable account, every figure and number | the results page, https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk (built by `experiments/paper_page.py`) |
| every read ever run, in order, with its ledger tags | [docs/RESULTS.md](docs/RESULTS.md) |
| the documents, current and retired | [docs/README.md](docs/README.md) |
| what is running now, and which reads are recorded | `python experiments/status.py` |
| the raw rows | `ledger/<experiment>/`; pre-registrations in `ledger/m3_prereg/` (written by `experiments/prereg/*.py` before any episode) |

## Reproduce

    PY=$HOME/flash/miniconda3/envs/chunks/bin/python
    $PY experiments/e3_read.py --plan --fit && $PY experiments/e3_read.py --accept --test && $PY experiments/e3_read.py --read
    setsid nohup $PY experiments/w1_stream.py --run > results/robot/w1_driver.log 2>&1 &     # a stream: resumable, survives logout
    $PY experiments/status.py                                                               # progress of every running read
    $PY experiments/audit_numbers.py && $PY experiments/audit_coverage.py                   # every quoted number against the ledger
    $PY experiments/paper_page.py                                                           # rebuild the results page

Rollouts run on SLURM (`slurm/jobs/robot_gpu.sbatch`, `experiments/launch_pack.py`); generated outputs (videos, packs, logs,
weights) live in `results/robot/` and are not tracked.

## Layout

    lifelong_ood/policy/      the frozen policy (backbone.py), adapters (adapt.py), primitives and their keys (primitives.py),
                              demonstrations per task and scene (data.py), attention binding, serving hooks
    lifelong_ood/ledger.py    the tracked results ledger;  lifelong_ood/stats.py: paired sign tests, two-level bootstrap
    experiments/              one script per read (e*_read.py, *_stream.py, w0/t0 probes), prereg/, audits, launch_pack.py
    experiments/report/       the results page: base.py (helpers, page), charts.py, current.py, history.py, style.css
    docs/                     METHOD, THE_TASK, RESULTS, plans of the current program; docs/history/ the retired plans
    ledger/                   one small JSON per run, tracked
    archive/                  the earlier repositories and programs, with their claim maps
