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

## 6. Result (2026-10-25; ledger `r7_read` tag cover, results/r7/read.json; pre-registration m3_prereg `pr7_cover`, commit 45d52831)

**Data.** 158 units (42 wrong-goal + 37 control states, each under 2 conditions) x 33 runs, all complete (6 A40 jobs x 3 pairs, 18
shards, 72-84 min each; 0 crashes, 0 missing units).

### 6.1 Coverage@K of the frozen policy's own samples (exact from 32 seeds)

The pooled CIs are two-level task-cluster bootstrap 95% intervals. `zs run` is the success of the 33rd run, which uses the recorded
zs seeds.

| states | condition | @1 (= per-seed) | @4 | @8 | @16 | @32 | zs run |
|---|---|---|---|---|---|---|---|
| wrong-goal (42) | **carry (PRIMARY)** | 0.060 [0.00, 0.12] | 0.118 [0.00, 0.21] | 0.153 [0.00, 0.28] | **0.200 [0.00, 0.39]** | 0.262 [0.00, 0.57] | 0.048 |
| wrong-goal (42) | start | 0.056 [0.04, 0.09] | 0.199 [0.14, 0.30] | 0.352 [0.25, 0.50] | 0.572 [0.44, 0.76] | 0.810 [0.67, 0.95] | 0.119 |
| zs success (37) | carry | 0.795 [0.61, 0.93] | 0.915 | 0.947 | 0.965 | 0.973 | 0.865 |
| zs success (37) | start | 0.747 [0.53, 0.91] | 0.876 | 0.920 | 0.955 | 0.973 | 0.838 |

Per task, wrong-goal states (coverage@1 / @4 / @8 / @16 / @32; successes out of 32 per state):

| task | n | carry | carry successes / 32 | start | start successes / 32 |
|---|---|---|---|---|---|
| :16 stack front bowl on middle bowl | 14 | 0.06 / 0.16 / 0.25 / 0.39 / 0.57 | 0 x6, 1 x4, 2 x2, 3, 17 | 0.06 / 0.22 / 0.37 / 0.59 / 0.86 | 0 x2, 1 x5, 2 x4, 3 x2, 10 |
| :29 bowl in the top drawer | 19 | 0.09 / 0.14 / 0.16 / 0.16 / 0.16 | 0 x16, 11, 11, 30 | 0.04 / 0.14 / 0.27 / 0.47 / 0.74 | 0 x5, 1 x8, 2 x3, 3 x3 |
| :40 frying pan on the shelf | 9 | 0.00 at every K | 0 x9 | 0.08 / 0.29 / 0.50 / 0.75 / 0.89 | 0, 2 x4, 3, 4 x3 |

Per-task state-bootstrap CIs are in read.json.

### 6.2 Verdict (pre-declared, section 3)

**H-cover FAILS.** Pooled coverage@16 on the wrong-goal carry states is 0.200 [0.00, 0.39], against a bar of 0.5. Stage B (verifier)
and stage C (closed loop) are not run.

### 6.3 Reading

1. **The sandbox reproduces the outcomes.** The zs-seed run fails on 0.95 of the wrong-goal carry states (success 0.048) and
   succeeds on 0.87 of the control carry states. The control states' own samples succeed 0.80 per seed, with coverage 0.97. The
   measurement can therefore see success where it exists.
2. **Once the object is grasped and carried, the error is mostly committed.**
   - From the carry state, 31 of 42 wrong-goal states have no success in 32 samples. Those are all 9 of :40, 16 of 19 of :29 and 6 of
     14 of :16.
   - The successes concentrate in a few states (17, 30, 11, 11 of 32). Coverage therefore saturates at 0.16 on :29 and 0 on :40,
     even at K = 32.
   - No selection over the policy's own carry-phase samples can recover these states. That holds for any verifier, the oracle
     included, since coverage is an oracle verifier's ceiling.
3. **From the episode start, the correct behaviour IS in the distribution, but rarely.**
   - The per-seed success from the start is 0.056, about what one expects of tasks the policy usually fails.
   - The successes are spread across states (34 of 42 states have at least one success in 32), so coverage grows with K: 0.35 at 8,
     0.57 at 16, 0.81 at 32.
   - On :40, the carry states are 0 / 9 but the start states reach 0.75 at K = 16. The region outcome is decided at or before the
     grasp (how and where the pan is picked up), not in the carry.
   - So a selection route exists only at the episode level, or at least before the grasp. A pre-grasp verifier would have to
     predict the final region from the approach / grasp, about 0.06 base rate, with K >= 16 whole-episode samples per state.
4. **Consequence for verified search at carry queries (PLAN_R 2.4 as designed for R7 stage C).** It is closed: its oracle ceiling
   is 0.20 at K = 16. That is the principled stopping point the question asked for, for the carry-phase route.

### 6.4 Recommended next step (not run)

- **Option 1: an episode- or grasp-level selection.** The ceiling is cov@16 0.57 from the start. It would need a deployable predictor
  of the final region from the first chunks or from the grasp configuration, and K whole-episode samples, which are not deployable
  without a simulator. In closed loop, the selection can only happen at the approach / grasp queries. Before designing a verifier,
  measure the ceiling of a grasp-time selection: coverage from a state just before the grasp. That costs about 1 A40-h with the same
  code (a new condition at D4's pre-grasp step).
- **Option 2: change the conditioning.** The parse's region word enters only through the prompt. A test-time change of what the
  policy is conditioned on (the language / vision binding of the region) is the remaining route for the carry-committed states.
  Weights stay frozen.
- Either option needs the user's approval: option 1 is a new stage, option 2 a new route.

**Cost.** 7.7 A40-h: 6 packs, 1.21-1.40 h each, plus the smoke at about 0.03 h. Also 4 short CPU jobs (plan, pre-registration, read).

**Disclosures.**
- The cost exceeded the plan (5-6 A40-h). The wrong-goal runs use the full 400-step budget.
- Five of :40's 14 fresh_d4 wrong-goal episodes have no D5 post-grasp observation, so they are not in the set (section 1).
- Start and carry states are the same 42 episodes. The `start` condition re-samples the whole episode from its recorded first state.
- A waiter command that contained `ls results/r7/sb | wc -l` was armed on the login node and stopped before it reached that line.
  The line never ran. No python or data inspection ran on the login node: every computation ran via sbatch.
- The read was run once with `--record`.
- Audits after the read: audit_numbers 0 discrepancies over 2,064 checks; audit_coverage 0 uncovered numerals.

## 7. Option 1: the pre-grasp coverage ceiling (pre-registered 2026-10-25, before any pre-grasp rollout; m3_prereg `pr7_pregrasp`)

**Why.** Section 6 placed the region outcome at or before the grasp: carry cov@16 0.20, start 0.57. A selection that acts at the
approach / grasp queries is deployable in closed loop only if the policy's own samples from a state just before the grasp still
contain the fix. The coordinator approved this ceiling check; no verifier is designed until the user decides.

- **States:** the same 42 wrong-goal + 37 control episodes. The pre-grasp state is the episode's LAST D4 `pre` observation
  (experiments/d4_bind.py): an approach query before the bound step, with an open gripper and the hand at least R2A.HAND_CLEAR (xy)
  from every object. The 3 control episodes of libero_90:1 have no `pre` observation, since it is an articulation-first task (D4).
  That leaves 76 units: 42 wrong-goal, 34 control.
- **Rollouts:** as in section 2. The run loop is r7_cover.run_unit, unchanged, with the same 32 seeds and a zs-seed run (the
  recorded zs seed at that query). The budget is (400 - j) / 5 queries, and the gripper ramp is rebuilt from the recorded commands
  before j. Code: `experiments/r7_pregrasp.py`, `experiments/r7pg_pack.sbatch` (a copy of r7_pack.sbatch). Plan: results/r7/plan_pg.json.
- **Hypothesis H-pregrasp:** pooled coverage@16 on the wrong-goal pre-grasp states >= 0.5 (point estimate; same estimator and
  bootstrap as section 3).
- **Also reported (descriptive), on the same states:**
  - carry vs pre-grasp vs start coverage@K;
  - paired differences of coverage@16 (pre-grasp - carry, start - pre-grasp; two-level bootstrap);
  - per task;
  - the zs-run success;
  - the carry - pre-grasp gap in steps.
- **Reading rule.** If H-pregrasp passes, a grasp-time verifier is worth designing (not here: hand back). If it fails while start
  passes, the outcome is fixed during the approach, before the hand nears the object. Selection would then have to act at the first
  queries. That is reported, and the route is handed back.
- **Cost (planned):** 76 units x 33 runs; about 4 A40-h (4 jobs x 3 pairs).
