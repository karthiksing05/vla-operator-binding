# Fit time: why a unit took 39 minutes, and the fast fit

Plan Z / F1. Scripts: `experiments/zf_profile.py`, `experiments/zf_levers.py`, `experiments/zf_fast_fit.py`. Ledger:
`f1_profile` tag `Gscplate_tmp`, `f1_levers` tag `read`; the confirmatory test is pre-registered as `m3_prereg` tag
`f1_fast_fit`.

## Where the time went

Profiled on one goal prototype (Gscplate: 667 frames, 8 demonstrations), on an A40:

| part | seconds |
|---|---|
| imports + loading pi0.5 | 2.4 + 79.0 |
| demonstration frames | 3.3 |
| one step, as the fits ran it (the frozen VLM prefix recomputed for the batch) | 2.13 take + 0.14 forward/backward + 0.01 optimiser |
| one step with every frame's prefix cached once (42.6 s for 667 frames, 11.9 GB) | 0.003 take + 0.14 + 0.01 |
| 1,000 steps | 39.4 min (recomputed) vs 4.6 min (cached) |

97% of a fit recomputed a prefix that never changes, because the units live in the action expert. With the prefix cached, the
losses match the recomputed fit (largest relative difference 5e-3 over 30 steps, bf16), and the recorded prototypes are
reproduced (`recorded_check`: held-out losses within 4%). `c0_fit.py --cache` is installed; the default is unchanged.

## How many steps a unit needs

E5's 26 units, held-out flow loss on each unit's demonstration 1 (never fitted), every 25 steps. Values are the held-out loss
relative to the 1,000-step fit (mean / worst unit):

| arm | 100 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|
| from scratch (the recipe) | −0.40 / +0.01 | **−0.47 / −0.32** | −0.44 / −0.34 | −0.29 / −0.23 | 0 |
| warm start from the nearest unit (lowest loss on the training frames) | −0.38 / −0.18 | −0.28 / −0.08 | −0.21 / +0.04 | −0.06 / +0.23 | +0.15 / +0.44 |
| from the parent type (5 instance children) | −0.25 / −0.01 | −0.16 / +0.10 | −0.12 / +0.16 | −0.03 / +0.23 | +0.10 / +0.38 |

Every unit overfits well before 1,000 steps. Its held-out loss is lowest near step 175 (median), and at 1,000 steps it is 4%
above the frozen policy's on average. A warm start reaches a good loss sooner (by step 50) but overfits sooner, so it buys little
over a scratch fit stopped at 200 steps. In the warm start, the loss-chosen nearest unit is also the subspace-nearest one in only
6 of 26 cases.

**Caution: held-out flow loss is not success.** One-demonstration per-task LoRAs end above the frozen policy's held-out loss too
(rank 4: 0.071 against 0.037 on task 24). Yet they succeed far more often in rollouts. The recipe is therefore confirmed by
rollouts.

## The fast fit and its test (pre-registered; ledger `f1_fast_fit` tag `read`)

The recipe tested was a cached prefix, from scratch, 200 steps: 2.0 min of job time per unit against 39.4.

| hypothesis | result |
|---|---|
| H1: E3's library refit with the recipe, non-inferior to E3's library (29 tasks, set b, 20 each) | **FAILS**: 388 against 480 of 580, difference −0.159 (95% interval −0.255 to −0.072) |
| H2: G8's leaves rule with rank-4 leaves, non-inferior to the rank-16 rule (66 tasks, set d) | **passes**: 980 against 994 of 1,320, difference −0.011 (−0.039 to +0.016) |

**Held-out flow loss is the wrong criterion for how long to fit.** The 800 steps it called overfitting are what make a unit
succeed. A unit must commit to the demonstration's mode, while a held-out likelihood rewards a field that hedges. This is the
same lesson as the categorical gripper in the soft blend (docs/INTERPRETATION.md section 4): decisions must be sharp.

**What is free.** Caching the frozen prefix reproduces the 1,000-step fit (equal losses) at 4.6 min instead of 39.4, an 8.6×
saving with nothing traded. Rank-4 leaves are non-inferior to rank-16 ones at a quarter of the bytes, which moves the storage
crossover against per-task rank-4 LoRAs from never to about 590 tasks (docs/SCALING.md).
