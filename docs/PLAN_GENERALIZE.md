# Plan G: justify compositionality methodically, across models, benchmarks and baselines (2026-10-14)

The claim to justify: **a task is solved by composing reusable concepts (object, instance, goal) learned from OTHER tasks, and
this composition is what generalises out of distribution** -- not the extra data, not the adapter capacity, not one model or one
benchmark. Each read below isolates one alternative explanation. Every read is pre-registered, paired, on fresh start states, and
recorded in the ledger as before (docs/METHOD.md, "How it is tested").

## What we have, and the gap a methodical case must close

| | evidence | what it does NOT show |
|---|---|---|
| E3 / E5 | the library, with each task's own demonstration inside its concepts, is near a per-task adapter at a fifth of the storage | that COMPOSITION generalises: the task's own demonstration is in both its prototypes |
| E0 (leave-one-out arm, exploratory, before the data fix) | composing concepts learned only from other tasks: 125 vs 117 zero-shot on 25 tasks | weak, and on scene-wrong data without the hierarchy: the central test has not been run properly |
| T0 | a NEW object does not borrow another object's prototype | (a scope limit, not a test of composition of known factors) |

## The reads, in order (each answers one alternative)

| | question (alternative ruled out) | design | compute |
|---|---|---|---|
| **G0** systematic recombination | Does a task whose object and goal were each demonstrated elsewhere, never together, succeed from its concepts alone? (rules out: "it only works because the task's own demonstration is inside") | 5-fold over the 67 single-step pick-and-place LIBERO-90 tasks: each fold's tasks are held out entirely; concepts refit on the other folds' demonstrations; a held-out task is eligible when its object (or instance) and its goal both have demonstrations outside its fold. Arms: zero-shot; composed concepts; acceptance | ~200 fits, ~3,000 episodes |
| **G1** factorisation vs monolithic, same data | Is it the FACTORISATION, or just more data? | the same folds; a monolithic multitask LoRA (rank 16, the same demonstrations, each frame with its own instruction) per fold, run on the held-out tasks; and the library's matched-capacity ablation (one rank-4 adapter per task-key, unfactored) | 5 + ~100 fits |
| **G2** composition baselines from the literature | Does composing ADAPTERS work without our phase gating? | on G0's held-out tasks: task arithmetic (sum of the LoRAs of an object-donor task and a goal-donor task), LoRA averaging, and retrieval of the nearest demonstrated task (STRAP-style); plus the existing per-task LoRA, DSRL and Wang et al.'s selection (W0) | reuses fits |
| **G3** any model | Is the result pi0.5-specific? | E3 and G0 repeated with SmolVLA (backbone_smolvla.py; same LIBERO checkpoint family) and pi0 (openpi pi0_libero): library refit per policy, same tasks, states and seeds | ~2 x (E3 + G0) |
| **G4** any benchmark / distribution shift | Does it survive other scenes and shifts? | (a) LIBERO-90's other task families as they become composable (drawers, stove, caddy, from E5's new goals); (b) a perturbation suite of LIBERO (camera, lighting, object layout: LIBERO-PRO / LIBERO-Plus if installable); (c) a second simulator (ManiSkill3 pick-and-place, with a diffusion policy trained there) -- only after G0-G3 settle the claim on LIBERO | install-dependent |
| **G5** ablations of the mechanism | Which part carries it? | split at the policy's grasp vs at a fixed fraction of the episode; hierarchy on/off (instance children); goal-only, object-only (from E0/E1 data, rerun at G0's design); prototype rank 4 vs 16 | ~100 fits |

## Decision rules fixed now

- G0 is the gate. If composed concepts from other tasks do not beat zero-shot on G0's eligible tasks (sign test, alpha 0.05),
  the paper's claim narrows to "consolidated prototypes with one demonstration per task" and G1-G2 are reported against that claim.
- G1's criterion: the factored library beats the monolithic same-data LoRA on the held-out tasks (paired sign test), or the
  claim that factorisation matters is dropped.
- Every read keeps the per-task LoRA with acceptance as the upper reference (it sees the held-out task's demonstration).
