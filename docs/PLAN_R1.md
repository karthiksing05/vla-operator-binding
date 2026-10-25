# Probe R1: training-free CAG-VA and the empty-instruction probe (pi05_l90c@5000; written 2026-10-24, before any R1 number)

**Question** (docs/PLAN_R.md section 7, R1). How much of pi05_l90c's zero-shot gap on LIBERO-90-C's held-out recombinations
does the cheapest fix for a vision shortcut close? And what does the policy do with no instruction at all?
- **CAG-VA, training-free** (Fang et al. 2602.17659, eq. 4): at every flow-integration step,
  v = v(o, ∅) + ω (v(o, instr) − v(o, ∅)), where ∅ is the empty instruction to the same frozen VLA. ω = 1 is zs.
- **The empty-instruction probe**: serve ∅ alone. For every episode, ask which demonstrated task the policy did instead.

Code: `experiments/r1_cag.py`. Pre-registration: `experiments/prereg/pr1_cag.py` (ledger `m3_prereg`, tag `pr1_cag`), recorded
after the design table and the serving smoke, before any R1 rollout. Ledger rows: `r1cag_dev`, `r1cag_test`, `r1cag_cells`
(rollout packs, via roll_record), `r1cag_read`. Payloads: `results/r1/`.

## 1. Set-up

| item | choice |
|---|---|
| policy | pi05_l90c main/5000 (`checkpoints_l90c/pytorch/pi05_l90c_5000`), frozen; our Euler sampler, 10 steps, as in M1 |
| CAG serving | the existing cfg control of serve_guided.py's factor mode, `factor:<task>?ctrl=cfg&w=<ω>`. It builds `Composition([instr, ""], [ω, 1 − ω])` and mixes the two velocities affinely at every Euler step (conditionals.sample_composed; both prompts in one batched VLM pass). No shared file changed |
| ∅ serving | `promptx:` (the instruction replaced by the empty string) |
| tasks | M1's 29 dev tasks (results/m1/plan.json): 15 recombination tasks (bins 0.1 / 0.3 / 0.5) and 14 part tasks. split_sealed.json is never opened |
| ω grid | {1 (= zs), 1.5, 2, 3, 5} |
| recording | as M1: BIND=track RECORD=1 LIGHT=1 OBJECTS=1 ALL_OBJECTS=1 (every movable object's pose per step, the first object bound) |

**Three sets of start states.**

| set | states | noise seed | arms | role |
|---|---|---|---|---|
| dev | NEW `results/robot/fresh_r1`: LIBERO's sampler at seed BASE 6700000 + k, 20 per task. `--fresh-check` asserts every state is > 1e-6 from every earlier libero_90 fresh set (fresh_m1 included) | 13 | zs, cag 1.5 / 2 / 3 / 5 | choose ω* |
| test | M1's `fresh_m1` (BASE 5700000) | 13 | cag at every ω, ∅; zs = M1's own rollouts (not re-rolled) | the headline, paired with zs |
| cells | M1's 10 damage cells (training cells), stored init states 0-19 | 0 | cag at every ω, ∅ | damage and the vision-shortcut rate, paired with the parity read of pi05_l90c@5000 |

**ω\*** = the ω with the most dev-set successes on the 15 recombination tasks; ties go to the smaller ω. The headline is ω\*. The
other ω run on the test set as well, and are descriptive.

## 2. "Which demonstrated task did it do instead"

This step is offline, in LIBERO's own venv. It reuses fm_types' reconstruction:
1. set the env to the episode's start state;
2. put every movable object at its last recorded pose;
3. run 10 control steps of physics with the arm at home.

At that settled final state, the step evaluates the goal atoms of every **training** task of the same scene (split.json "train",
their BDDL goals through factors.py).
- **Placement goals only.** The rollouts do not record fixture joints, so articulated atoms (Open / Close / Turnon / Turnoff)
  are not read. A training task's placement goal is its In / On atoms. Articulation-only training tasks are not evaluable.
- **Completes a demonstrated task c** when:
  - every placement atom of c holds at the settled final state;
  - they do not all hold at the settled first record;
  - c's placement atoms are not a subset of the instructed task's own goal.
- **Shares the object / the target**: some atom of c has the instructed task's named object as its first argument, or its
  target as its second argument.
- **Coverage.** 19 of the 29 dev tasks have at least one evaluable same-scene training goal. The 9 STUDY_SCENE1-3 tasks have
  no training task in their scene. KITCHEN_SCENE8's only training task is articulated.

The same pass types every failure with fm_types' rule (no_commit / wrong_object / wrong_goal / motion / fixture). It also
retypes M1's zs, so our zs types can be checked against results/fm/types.

## 3. Hypotheses (pre-registered; Holm over H1-H2 at alpha 0.05)

- **H1** CAG(ω\*) > zs on the 15 recombination tasks of the test set.
  - Test: more successes AND the exact two-sided sign test on discordant pairs.
  - If ω\* = 1, the p-value is 1.
- **H2** ∅ completes a demonstrated task in more than 25% of its episodes on the 19 evaluable dev tasks.
  - Test: one-sided two-level cluster bootstrap (B 10,000, seed 0), with p the share of bootstrap rates <= 0.25.
- **H3** (an estimate gate, outside Holm) CAG(ω\*)'s damage is at least −0.01.
  - Damage: the mean success change over the 10 training cells against the parity base.

**Secondary (descriptive):**
- every ω vs zs, by recombination / part / all and by bin;
- failure types per arm;
- the instead rates of zs failures, ∅ episodes and CAG failures, split by shared object / target;
- ∅'s success on the training cells (the vision-shortcut rate) and on the dev tasks.

**Disclosure.**
- M1's and FM's zs success and failure types on fresh_m1 were known before this design.
- No CAG or ∅ rollout of pi05_l90c existed before R1.
- The serving smoke (libero_90:14, stored init states 0-1, 2 episodes per arm) is excluded from every number.

## 4. Cost (planned)

340 rollout instances of 20 episodes (dev 145, test 145, cells 50), in 85 packs of 4 on overcap A40s: about 25 A40-h. CAG pays two
VLM prefix passes per query. The instead / typing pass is CPU.

## 5. Result (2026-10-24; ledger `r1cag_read` tag read, results/r1/read.json)

All 340 rollout instances completed (85 packs, every one exit 0). The instead / typing pass covered all 39 tasks and cells. Our
retyping of M1's zs agrees with results/fm/types on 580 of 580 episodes.

**Dev selection (fresh_r1, recombination successes of 300):** zs 178, cag 1.5 187, cag 2 180, cag 3 152, cag 5 115. So ω\* = 1.5.

### 5.1 Test set (fresh_m1, 20 states per task, paired with M1's zs; two-level cluster bootstrap 95% CI)

| arm | recombination (300) | part (280) | all (580) | vs zs, recombination: diff [CI], discordant, sign p |
|---|---|---|---|---|
| zs (M1) | 180 (0.600 [0.367, 0.817]) | 0 | 180 (0.310) | |
| **CAG ω\* = 1.5 (headline)** | 173 (0.577 [0.350, 0.790]) | 2 | 175 (0.302) | −0.023 [−0.073, +0.033], 14 to 21, p 0.31 |
| CAG ω 2 | 182 (0.607 [0.397, 0.807]) | 1 | 183 (0.316) | +0.007 [−0.057, +0.070], 22 to 20, p 0.88 |
| CAG ω 3 | 168 (0.560) | 4 | 172 (0.297) | −0.040 [−0.143, +0.053], 23 to 35, p 0.15 |
| CAG ω 5 | 110 (0.367) | 4 | 114 (0.197) | −0.233 [−0.403, −0.073], 17 to 87 |
| ∅ (empty instruction) | 68 (0.227 [0.060, 0.420]) | 1 | 69 (0.119) | −0.373 [−0.590, −0.167], 5 to 117 |

By bin (zs / CAG 1.5 / CAG 2 / ∅, of 100 each): CD 0.1: 80 / 73 / 75 / 42. CD 0.3: 43 / 46 / 51 / 21. CD 0.5: 57 / 54 / 56 / 5.
Part tasks: zs 0 and CAG at most 4 of 280.

### 5.2 Failure types (test set, episodes of 580)

| arm | success | no commit | wrong object | wrong goal | motion | fixture |
|---|---|---|---|---|---|---|
| zs | 180 | 7 | 63 | 152 | 117 | 61 |
| CAG 1.5 | 175 | 13 | 58 | 146 | 125 | 63 |
| CAG 2 | 183 | 29 | 53 | 147 | 112 | 56 |
| CAG 3 | 172 | 62 | 47 | 110 | 128 | 61 |
| CAG 5 | 114 | 124 | 59 | 85 | 132 | 66 |
| ∅ | 69 | 27 | 183 | 147 | 77 | 77 |

Guidance shifts no binding error into success. Raising ω trades wrong-goal errors for no-commits (7 at zs, 124 at ω 5).

### 5.3 "Which demonstrated task did it do instead" (placement goals of same-scene training tasks)

| episodes | completes a demonstrated task | of which share the instructed object or target |
|---|---|---|
| zs failures, all 29 tasks | 38 / 400 (9.5%) | 38 (37 the object, 1 the target) |
| zs failures, the 19 evaluable tasks | 38 / 200 (19%) | |
| ∅ episodes, the 19 evaluable tasks (H2) | 177 / 380 (0.466 [0.313, 0.626]) | 128 |
| ∅ failures, the 19 evaluable tasks | 160 / 312 (51%) | 111 (43 the object, 73 the target) |
| CAG 1.5 failures, 19 evaluable tasks | 42 / 205 | 40 |

Where it shows (∅, of 20):
- KITCHEN_SCENE2: tasks 11 / 14 / 16 stack the bowls (training task 17) 14-15 times.
- LIVING_ROOM_SCENE3: tasks 58 / 59 put the alphabet soup in the tray (task 55) 19 times.
- LIVING_ROOM_SCENE4: task 61 stacks the bowls into the tray (task 63) 15 times.
- LIVING_ROOM_SCENE2: task 50 puts the milk in the basket (task 52) 13 times.
- KITCHEN_SCENE4: task 22 puts the wine bottle in the cabinet (task 26) 11 times.

zs's instead episodes concentrate on two tasks:
- task 29 (bowl into the top drawer) puts the bowl on top of the cabinet (training task 31) in 18 of 19 failures;
- task 40 (frying pan on the shelf) does tasks 41 / 42 in 16 of 20.

### 5.4 Training cells (stored states 0-19, seed 0, against the parity base: 188 / 200 = 0.94)

| arm | successes / 200 | mean change | discordant (arm / base) |
|---|---|---|---|
| CAG 1.5 (ω\*) | 176 | −0.060 | 6 / 18, p 0.023 |
| CAG 2 | 173 | −0.075 | 7 / 22 |
| CAG 3 | 147 | −0.205 | 8 / 49 |
| CAG 5 | 78 | −0.550 | 3 / 113 |
| ∅ | 99 (0.495 [0.265, 0.715]) | −0.445 | 2 / 91 |

**The vision-shortcut rate.** With no instruction at all, the policy completes its own training task in about half of the
episodes:
- cells 46, 17, 28 and 52: 16-19 of 20;
- cells 6, 23 and 62 (articulated, or a two-object scene): 0.

CAG 1.5's damage comes from cell 23 (−0.30) and cell 62 (−0.20).

### 5.5 Pre-registered verdicts (Holm over H1-H2 at alpha 0.05): H2 passes; H1 fails; the H3 gate fails

- **H1 FAILS.** CAG(ω\* = 1.5) 173 vs zs 180 on the recombinations: diff −0.023 [−0.073, +0.033], sign p 0.31. No ω beats zs
  on the test set; the best, ω 2, is +2 episodes.
  - The dev-set gain of ω 1.5 (+9 of 300) did not replicate on fresh_m1.
- **H2 PASSES.** ∅ completes a demonstrated training task in 0.466 [0.313, 0.626] of its episodes on the 19 evaluable tasks,
  with one-sided p 0.0023 against 0.25 (Holm threshold 0.025).
- **H3 FAILS.** CAG 1.5's damage on the training cells is −0.060, below the −0.01 gate.

**Reading.**
1. **The cheap fix does not close the gap on pi05_l90c.** Training-free CAG-VA is the cheapest repair of a vision shortcut,
   and it moves nothing on the recombinations at ω 1.5 to 3. It does not touch the new parts, and at ω ≥ 3 it costs success
   and damages the training cells. The R2 bar is therefore zs itself, 180 of 300.
   - Contrast: CAG-VA scores 66.5 on CRAFT's pi0.5 Pick-Place, a different benchmark, and the VA variant needs training.
2. **The vision shortcut is real and large.** With the instruction removed:
   - the policy still finishes its own training task in 99 of 200 cell episodes;
   - on dev scenes it executes a demonstrated task of the scene in 47% of episodes, most of them sharing the instructed object or
     target.
3. **zs failures mostly do not replay a demonstrated combination.** Only 38 of 400 do (19% on the evaluable tasks).
   - zs's binding errors (215) are therefore mostly not "the training task of this scene". They are misplacements to other
     regions (wrong goal 152) and grasps of other objects, without completing another demonstrated goal.
   - The empty instruction shows that the scene alone does drive demonstrated behaviour. The instruction moves the policy off it,
     but not onto the instructed combination.
4. **Why CFG cannot help here.** CAG amplifies v(instr) − v(∅). Where zs fails by binding, the instruction-conditioned velocity
   is already the wrong one, so the amplified difference points to the same wrong binding. The extra weight only adds
   no-commits. This agrees with the failure map: binding leaves the word positions by layer 2.

**Disclosures.**
- **Selection.** ω\* came from the dev set by the pre-registered rule. The other ω on the test set are descriptive. ω 2 being
  best on the test set (+2) is not a finding.
- **Placement goals only.** Articulated atoms are not recorded, so "instead" reads placement goals only. Articulation-only
  training tasks (e.g. 6, 23's drawer part, 28, 35, 39) can never count, which makes the rates lower bounds.
- **Counted successes.** An instructed success can also satisfy another training goal, and it then counts as instead. On
  task 60, 16 of ∅'s 18 successes also satisfy task 63's goal (the bowl ends in the tray on the other bowl). Excluding task 60,
  H2's share is 161 / 360 = 0.447, still above 0.25.
- **Typer approximation.** The reconstruction places only movable objects (fixtures stay at their start configuration). Its own
  success check holds for 133 of zs's 180 successes, the same approximation FM disclosed.
- **Smoke.** The serving smoke (libero_90:14, stored states 0-1, ledger `r1cag_smoke`) is excluded from every number.
- **Process.** Early in the session, one empty `python3 -` invocation ran on the login node by mistake. It executed nothing,
  and it is disclosed here because the rule forbids any python there.
