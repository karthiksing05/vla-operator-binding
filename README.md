# lifelong-ood-learner

**How far can composition carry a frozen policy on new tasks?** A frozen flow-matching robot policy meets tasks it was never
trained on and takes them on by composing what it already holds, creating new units only where nothing it holds explains the
task. In familiar scenes this works; it is not a general account of out-of-distribution learning (re-scoped 2026-10-21 after
[docs/DESIGN_REVIEW.md](docs/DESIGN_REVIEW.md); forward plan [docs/FRAMEWORK.md](docs/FRAMEWORK.md) with its design [docs/DESIGN.md](docs/DESIGN.md): a hierarchical store of reusable
prototypes composed without supervision, as an alternative to fine-tuning).

Robot manipulation is the testbed. A LIBERO task is a composition of parts (an object and its location, a goal relation and its
location, a scene). The policy's 40 training tasks expose the parts, and every test task is graded by which parts and relations
it leaves unseen ([docs/SPLIT.md](docs/SPLIT.md)). The question, after Fodor's systematicity and the task-to-task transfer
documented in cognitive psychology: how does prior knowledge donate expertise to a task it was never learned for, and what must be
created when it cannot?

## Two schemes, one framework

The framework is architecture-agnostic: all it needs from a policy is a way to extract units. It runs on five frozen
flow-matching policies (pi0.5, pi0, SmolVLA, X-VLA, GR00T N1.5; FLOWER on its one suite).

| | units come from | held units | new units |
|---|---|---|---|
| **supervised** ([docs/METHOD.md](docs/METHOD.md)) | the instruction: object, instance, goal, switched at the policy's own grasp | pointers to the policy's field under a training instruction (zero bytes) | rank-4 prototypes and task leaves, fitted on the one demonstration |
| **unsupervised** ([docs/UNSUPERVISED.md](docs/UNSUPERVISED.md)) | the demonstration: cut at its contact events, each segment named by the policy's own likelihood | pointer programs advanced at the policy's gripper events | fitted units, one per unexplained segment |

In both schemes:
- practice chooses sources and operators;
- the policy's likelihood decides held vs new;
- every unit gets a donation analysis: what it contributes, to which axis, in which phase, and by which relation to the task
  ([docs/INTERPRETATION.md](docs/INTERPRETATION.md); `experiments/zi_blend.py`). The supervised library `lib` is left out of
  every donation claim: its prototypes were fitted on demonstrations that include the target task.

Status of the unsupervised scheme: in closed loop neither the pointer programs nor the soft blend yet beats the policy alone
(Z3, Z5), so it is analysis for now. That prototypes speed convergence ([docs/PLAN_CONV.md](docs/PLAN_CONV.md)) is not
supported yet.

## Results by grade (docs/SPLIT.md; descriptive)

The grades were assigned after these results, and grade is confounded with benchmark: every G1/G2 row is one of Li's tasks in a
training scene, every G3/G4 row a LIBERO-90 scene. "Nothing fitted" at G1/G2 still spends 340 rewarded practice episodes per task
([docs/COST.md](docs/COST.md)); Li's 20 tasks were reused across the design iterations, and practice may pick sources by where
their motion goes more than by their concepts. The leaves choose from a larger candidate set with more practice than the per-task
LoRA. "No damage" in the stream holds by construction (frozen policy, frozen units). The novelty detector's AUC scores held
concepts on the policy's own training demonstrations. Threats: [paper/DRAFT.md](paper/DRAFT.md) Section 16.

| unseen | composing held units, nothing fitted | creating units |
|---|---|---|
| a relation or a binding, familiar scene (G1, G2: Li et al.'s 20 recombinations; 340 practice episodes per task) | pi0.5 312 vs 143 of 400 for the policy alone; pi0 336 vs 35; SmolVLA 96 vs 5; X-VLA 256 vs 80; GR00T N1.5 157 vs 113 | not needed |
| a scene shift (G3) | 8 LIBERO-90 tasks: 67 vs 68 for the policy alone | the same 8: library 154; all 15 G3 tasks: leaves 225 vs 90 |
| a new part (G4) | 2 tasks: 0 vs 4 | 31 tasks: leaves 455 vs 117 |

## Where to look

| what | where |
|---|---|
| the paper page, every figure and number | https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk (built by `experiments/paper_page.py`) |
| the paper draft | [paper/DRAFT.md](paper/DRAFT.md) |
| every read, in order, with its ledger tags | [docs/RESULTS.md](docs/RESULTS.md) |
| the documents | [docs/README.md](docs/README.md); the active plan [docs/PLAN_Z.md](docs/PLAN_Z.md) |
| the scripts, by plan | [experiments/README.md](experiments/README.md) |
| what is running now | `python experiments/status.py` |
| the raw rows | `ledger/<experiment>/`; pre-registrations in `ledger/m3_prereg/` (written by `experiments/prereg/*.py` before any episode) |

## Reproduce

    PY=$HOME/flash/miniconda3/envs/chunks/bin/python
    $PY experiments/split_grades.py                                    # the graded split, every result stratified
    $PY experiments/g18_four_policies.py --read                        # the OOD system on each policy (G15 for pi0.5 / pi0)
    $PY experiments/g8_leaves.py --read                                # LIBERO-90: library, leaves, LoRA
    $PY experiments/audit_numbers.py && $PY experiments/audit_coverage.py   # every quoted number against the ledger
    $PY experiments/paper_page.py                                      # rebuild the results page

Rollouts run on SLURM (`slurm/jobs/robot_gpu.sbatch`, `experiments/launch_pack.py`). Unit fits use `experiments/c0_fit.py
--cache` (the frozen prefix cached once; [docs/FIT_TIME.md](docs/FIT_TIME.md)). Generated outputs (videos, packs, logs, weights)
live in `results/robot/` and are not tracked.

## Layout

    lifelong_ood/policy/      the backbones (backbone*.py: pi0 / pi0.5, SmolVLA, X-VLA, GR00T, FLOWER), adapters (adapt.py),
                              concepts and their keys (primitives.py), segmentation (segment.py), pointer programs (program.py),
                              operators (tli.py, held_ops.py, bind.py), serving hooks, demonstrations (data.py), factors.py
    lifelong_ood/ledger.py    the tracked results ledger;  lifelong_ood/stats.py: paired sign tests, two-level bootstrap
    experiments/              one script per read, grouped by plan prefix (experiments/README.md); prereg/, audits, report/
    docs/                     current documents (docs/README.md); docs/history/ the superseded plans
    paper/                    the draft and its audited numbers
    ledger/                   one small JSON per run, tracked
    archive/                  earlier programs and unused scripts;  vendor/wheels/  offline wheels for the figure toolchain
