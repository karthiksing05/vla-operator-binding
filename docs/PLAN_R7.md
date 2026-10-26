# Probe R7: does the frozen policy's own sample distribution contain the correct behaviour? (written 2026-10-25, before any R7 rollout outcome was read)

**User direction.** "Keep iterating until the recovery method has been justified." The method works at test time on the frozen
pi05_l90c@5000, with no new demos and no weight change. It uses prototypes extracted from the model's internals and takes the task
structure from the parse.

**Why R7.** D5 showed that the goal role is bound at the release point (G1 1.000, G2 R^2 0.81). It also showed that 28 of 42
recombination wrong-goal observations are same-body REGION errors that the sandbox reproduces (:29 cabinet top instead of the top
drawer; :40 the wrong shelf level). R6 showed that carry guidance does not raise success, not even with the oracle goal: steering a
chunk's xy heading does not control the descent into a region. R6 recommended to select, not steer.

**The decisive question.** Does the frozen policy's OWN sample distribution already contain the correct behaviour on these states?
- If it does, recovery becomes a verifier and selection problem (PLAN_R section 2.4, verified search).
- If it does not, no test-time selection can recover it without changing the conditioning. That is a principled stopping point for
  this route.

Code: `experiments/r7_cover.py` (stages plan / sandbox / read), `experiments/r7_pack.sbatch` (a copy of d5_pack.sbatch: D5's batched
server + the R7 client, 3 pairs per A40). It reuses D5's Sandbox (experiments/d5_goal.py) and server (experiments/d5_serve.py), both
unchanged. Pre-registration: `experiments/prereg/pr7_cover.py` (ledger `m3_prereg`, tag `pr7_cover`). Read row: `r7_read` tag `cover`.
Payloads: `results/r7/` (plan.json and read.json tracked; sb/, sb_smoke/, logs/ local).

## 1. States (results/r7/plan.json)

- **Wrong-goal (PRIMARY):** every D5 dev observation of a zs wrong-goal episode on a recombination task (bins 0.1 / 0.3 / 0.5; the
  part tasks are excluded). That is 42 states: libero_90:16 14, :29 19, :40 9. These are the only recombination tasks with zs
  wrong-goal episodes in fresh_d4 that have a post-grasp observation. Five of :40's 14 fresh_d4 wrong-goal episodes have no D5 post-grasp
  observation (D4's rule), so they are not in the set.
- **Control:** D5's 37 zs-success observations on 11 recombination tasks (up to 4 per task). They test whether the sandbox and the
  seed sampling reproduce success where zs succeeded.
- **Conditions:**
  - `carry` (PRIMARY): D5's recorded post-grasp state at step j (the named object bound and lifted 3 cm, the hand at least 6 cm
    from its own target);
  - `start`: the same episode's recorded first policy state (step 0, after the 10 wait steps).
  Both use D5's restore (MjSimState via mj_getState(INTEGRATION), the gripper ramp rebuilt from the recorded commands, the OSC goal
  at the current pose).

## 2. Rollouts (pre-declared)

- **K = 32 independent noise seeds per state.** Seed k at query q is 41,000,000 + 100,000 k + q, shared across states.
- **A 33rd run per state uses the recorded zs seeds.** It is a fidelity check: does the sandbox reproduce the zs outcome?
- **Every run follows D5's loop.** The run renders both cameras, queries the frozen policy (its own 10-step Euler sampler) and
  executes 5 actions. A run ends at LIBERO task success (`done`: the goal predicates hold, i.e. the named object is in the NAMED
  region) or when the episode budget runs out: (400 - j) / 5 queries, since libero_90 allows 400 policy steps. The release (the first
  executed open command) is recorded but does not end the run, so a regrasp can still succeed.
- **Saved per run:** success, steps, the release hand and object, the final object position, every chunk, and the hand at every query.
  These are the inputs a deployable verifier would have (stage B).

## 3. Readouts and hypothesis (pre-declared)

- **coverage@K** for K in {1, 4, 8, 16, 32}. For each state, take the exact probability that a random K-subset of its 32 runs holds at
  least one success: 1 - C(32 - c, K) / C(32, K), where c is the number of successes. Average it over states.
  - Pooled CIs: the two-level task-cluster bootstrap (d4_read.rate: B 10,000, seed 0). With 3 tasks, these CIs are wide.
  - Per-task CIs: a state-level bootstrap (B 10,000, seed 0).
- **per-seed success rate:** c / 32, averaged over states (the same as coverage@1).
- **zs-box success:** the 33rd run's success. It should be about 0 on wrong-goal states and high on control states, if the sandbox
  reproduces the zs outcome. D5's fidelity is 0.70 within 3 cm of the zs release.
- **Hypothesis H-cover (PRIMARY, decides stage B):** pooled coverage@16 on the wrong-goal `carry` states >= 0.5 (point estimate).
  The CI and every other cell are descriptive.
- **Decision.**
  - If H-cover passes, stage B (a deployable verifier, pre-registered separately before it is read) is designed on these saved runs.
  - If it fails, R7 is recorded and handed back with a diagnosis.
  - If coverage is high from `start` but low from `carry`, the error is fixed by the grasp / carry state. Selection would then have
    to act earlier, and that is reported.

## 4. Known caveats (before the run)

- **The sandbox is not bit-exact (D5 section 7).** Snapshot / restore leaves some Python-side state out, and contact amplifies it.
  The base sandbox under the recorded zs seeds reproduces the zs release within 3 cm in 0.70 of observations, with release agreement
  0.88. The zs-box success is the R7-specific measure of this.
- **Success is LIBERO's goal predicate.** On In / On tasks some predicates already hold while the object is held (D5 section 3), so
  a run can succeed before the release. That is the task's own success definition, as in every rollout.
- **The 42 states are post-hoc failures.** They were selected because zs failed. Coverage measures whether OTHER samples from the
  same state succeed, so this selection does not bias coverage upward.

## 5. Cost (planned)

158 units (42 + 37 states x 2 conditions) x 33 runs, up to 80 queries each. Wrong-goal runs mostly use the full budget. At D5's
speed this is about 5-6 A40-h, run as 4 A40 jobs x 3 pairs (12 shards).
