# Probe R5s: closed-loop smoke of vision-anchored bank guidance (EXPLORATORY; written 2026-10-25, before any R5s rollout was read)

**Status: EXPLORATORY.** The user approved this smoke on 2026-10-25 ("Closed-loop smoke"). It decides whether a pre-registered
closed-loop read is worth running. It makes no confirmatory claim, and it reports no p-values; every interval is descriptive.

**Question.** pi05_l90c@5000 stays frozen. The pre-grasp chunks are guided early in the policy's own flow sampler toward a
vision-anchored target, through D4's extracted approach operator. Does the policy then grasp the NAMED object on the part tasks
where zs grasps the wrong one? And does it still solve the tasks zs solves?

Code: `experiments/r5s_smoke.py` (stages), `experiments/r5s_guide.py` (serving), plus two opt-in additions, both default off:
- the `r5s:` mode in `experiments/serve_guided.py`;
- `R5S_GEOM=1` in `experiments/libero_client.py`, which sends the live agentview depth's per-token 3D points.

Ledger: `m3_prereg` tag `pr5s_smoke` (this design, labelled exploratory) and `r5s_read` tag `smoke` (the read).
Payloads: `results/r5s/`. plan.json, bank.json, specs/, fresh_check.json, smoke_check.json, types/ and read.json are tracked.
The roll/, smoke/ and qlog/ directories, the recordings and the packs stay local.

## 1. Tasks (pre-declared rules; results/r5s/plan.json)

- **Wrong-object set (7):** the part dev tasks libero_90:38, 58, 73, 77, 79, 80, 83, where R2a and D4 saw zs wrong-object
  errors.
- **Harm set (4):** among D4's 11 recombination tasks with an In / On goal, the 4 with the highest R2 zs success on fresh_rb2
  (results/rb2/read.json; ties broken by sha1). The rule uses M1/R2 rates, not D4's paired rows, so the selection does not
  condition on the baseline it is compared with. Result: libero_90:61 (R2 zs 20 / 20), :59 (20), :9 (19) and :50 (19). :14 and
  :60, also at 19, lost the tie.
- **Damage set (2 training cells):** R2's damage-cell order (results/rb2/plan.json, sha1-ranked). The rule takes the first 2
  cells whose goal has an In / On atom on a movable object, so that guidance can act on them. Result: libero_spatial:6 and
  libero_90:70.
- **States.**
  - The 18 D4 tasks use `fresh_d4` (20 states each). The zs rows are D4's, paired by trial.
  - Prep verified the D4 settings from its roll files and packs: client seed 7, replan 5, wait 10, trials 0-19, MODE ours,
    serve seed 13, the pi05_l90c_5000 checkpoint, bind track, all objects, and 20 episodes plus 20 recordings per task.
  - The training cells use a NEW set, `fresh_r5s`: LIBERO's sampler at BASE 8600000 + k, 20 per cell. Their zs arm is
    re-rolled. `--fresh-check` asserts every state is > 1e-6 from every earlier fresh set of that task, and from the set's
    other states.

## 2. Arms

All arms are served exactly as zs except on guided queries. zs here means MODE ours: plain Euler with 10 steps, the noise seed
taken from the server seed (13) plus its call count, replan 5, and the bound-object tracker.
- `zs`: D4's rows where paired; re-rolled on the 2 cells.
- `vis_e1` (PRIMARY): the vision target, x0-projection on the first 5 of the 10 Euler steps, w = 1.
- `vis_a025` (secondary): the vision target, every Euler step, w = 0.25.
- `orc_e1`: the oracle target (privileged upper bound), primary configuration.

The two configurations are R4-bind's nearest misses: d1/early/1 (g_cf +0.16) and d1/all/0.25 (g_cf +0.43). The target form
changes from D1's operator to D4's bank (below). No other configuration is run.

**Phase rule (L4 / held_ops._Grasp).**
- `closed` holds when the 5th commanded gripper of the last chunk is > 0.
- The grasp is `closed` on 3 consecutive queries.
- A query is guided only while the gripper is open AND before the episode's first grasp. Guidance is off for the rest of the
  episode after that, even after a release.

**Guided query.**
1. The chunk comes from rb4_bind.Guided.sample (R4-bind's sampler), using the noise seed the unguided query would have drawn.
2. At each guided Euler step: x0_hat = x_t - t v. The integrated xy displacement D(x0_hat) is moved by w (d* - D), spread evenly
   over the 10 steps' xy dims, and v' = (x_t - x0_hat') / t.
3. The target is **d* = J (c - h) + b**: D4's bank operator for the named object's type, else its category, else the approach
   role.
   - J is the 2x2 xy block and b the xy offset. h is the hand and c the target.
   - The bank is rebuilt with d4_read.bank from D4's training observations; it reproduces D4's 109 nodes and 7,619 bytes
     (asserted).
   - The level used per task: type on 11 tasks. On 2 it is category, because the type is the never-seen part:
     moka_pot on :38 (cookware) and ketchup on :58 (grocery). Five of the 7 wrong-object tasks name black_book (type level).
   - The displacement uses the same metres-per-action-unit scale that D4's J was measured in. That scale equals R2a's scale
     (checked).
4. Every unguided query goes through `GuidedPolicy.infer` unchanged.

**Targets.**
- `vis`, the vision anchor (D3 / R4-bind):
  - The anchor token is the argmax over the policy's agentview SigLIP tokens of the cosine to the named types' training
    prototypes (results/rb4/protos.npz). The prototypes are type-level, so they cannot tell two instances of one type apart.
  - The anchor's 3D point comes from the LIVE agentview depth at 256 with the robot visible (a real depth camera, as R4's
    stage 2 planned). It is back-projected and reduced per token by R4-bind's own `backproject` ('rs3') and `token_points`.
  - The operator used is the anchored prototype's type.
- `orc`: the named object nearest the hand (xy), at its simulator position.

Named objects are the first arguments of the task's In / On goal atoms, which are D4's / R2a's pre-grasp `named` (checked
against D4's index).

## 3. Readouts (EXPLORATORY; descriptive intervals only)

- **Success** per task and arm, pooled per set.
- **Named grasp:** the episode's bound object (the client's tracker: the first movable object displaced > 3 cm while the gripper
  is commanded closed) is a named object.
- **Failure types:** fm_types' typer (no_commit / wrong_object (+ sub) / wrong_goal (+ sub) / motion / fixture). It is run on
  every episode, with each task's own start states. D4's zs recordings are re-typed and must equal D4's types; mismatches are
  reported.
- **Paired differences** (arm - zs, by trial) per set, with the two-level task-cluster bootstrap 95% interval: tasks are
  resampled, then episodes within each task (B 10,000, seed 0). The differences cover success, named grasp and the
  wrong-object rate.
- **Harm:** the success difference on the harm set, plus the share of zs successes that the arm loses (and the zs failures it
  wins).
- **Damage:** the same on the 2 training cells.
- **Anchor localisation in closed loop** (vis arms): on each guided query, a hit means the anchor's xy is within 5 cm of a named
  object's simulator position. The hit rate is reported as episode means per set, together with the median error.
- **Guidance bookkeeping:** guided queries per episode, fallbacks (open pre-grasp queries with no target), operator levels used,
  the median |D - d*| after guidance and the time per guided query.
- **Cost:** A40-hours from the pack logs (smoke included).

## 4. Serving smoke (before any rollout)

On libero_90:73, trials 0-1 (the first two of a fresh process, so the reset count matches D4's), four instances run in one
pack:
- MODE ours with the R5s client flags (LIGHT, no REC_SIM);
- r5s arm=off with R5S_GEOM=1;
- r5s vis and r5s orc, early / w1, as a code-path check only, excluded from every number.

Requirement: the first two instances' executed actions equal D4's recorded zs actions exactly. results/r5s/smoke_check.json
records the outcome.

## 5. Decision rule for a confirmatory read (pre-declared, exploratory)

A pre-registered closed-loop read is recommended only if all four of the following hold (point estimates):
1. on the 7 wrong-object tasks, vis_e1 raises success by >= +0.10 over zs;
2. vis_e1 raises the named-grasp rate there (point > 0);
3. the harm-set success difference is >= -0.05;
4. the damage-set success difference is >= -0.05.

The oracle arm is read alongside. If orc_e1 passes but vis_e1 does not, the anchor is the bottleneck and the next step is the
anchor, not a confirmatory read. If neither passes, the guidance mechanism is the bottleneck.

## 6. Cost (planned)

13 tasks x 3 guided arms + 2 zs re-rolls = 41 instances of 20 episodes, in 11 packs of 4 on one A40 each. D4's packs took about
20-25 min, so the plan is about 5-8 A40-h with the guidance overhead.

## 7. Disclosures (development)

- During development one empty `python3 -` heredoc was issued on the login node, and it RAN (an empty program, no output). This
  broke the hard rule. Every computation ran on compute nodes.
- The damage set includes a libero_spatial training cell (libero_spatial:6), as R2's damage order gives it.
- The fresh check compares fresh_r5s with the earlier `results/robot/fresh_*` sets of the same task.
  - libero_90:70: the minimum distance is 0.0174 over 3 earlier sets.
  - libero_spatial:6 has no earlier fresh set.
  - Distance to LIBERO's stored init states (fresh_states.py's own check): 0.040 and 0.037.
  - Minimum within-set distance: 0.0113 and 0.0171.
- Rollout instances also write the usual per-instance ledger rows through roll.sh / roll_record.py (experiments r5s_<arm>,
  r5s_smoke_*).

## 8. Result (2026-10-25; ledger `r5s_read` tag smoke, results/r5s/read.json; design row m3_prereg `pr5s_smoke`, commit a35fb7a9)

EXPLORATORY. The intervals are two-level task-cluster bootstrap 95% intervals (descriptive). 41 instances ran, all 20 episodes
each, with no crash and no requeue. The zs rows re-typed from D4's recordings equal D4's types (0 mismatches). Every open
pre-grasp query was guided: 0 fallbacks, since every named type has a prototype and every anchor point is finite.

**Serving smoke.** On libero_90:73, trials 0-1, the executed actions of MODE ours and of r5s arm=off with R5S_GEOM=1 equal
D4's zs actions exactly: 400 / 400 steps per episode, max |diff| 0. The opt-in depth render and the mode leave unguided serving
bit-identical.

### 8.1 Success and named grasp (of 20 per task; named grasp = the bound object is the named one)

| task (set) | zs | vis_e1 | vis_a025 | orc_e1 | named grasp zs / vis_e1 / vis_a025 / orc_e1 |
|---|---|---|---|---|---|
| libero_90:38 moka pot (wo) | 0 | 0 | 0 | 0 | 0 / 0 / 0 / 0 |
| libero_90:58 ketchup (wo) | 1 | 0 | 0 | 2 | 1 / 0 / 2 / 3 |
| libero_90:73 black book (wo) | 0 | 0 | 0 | 0 | 20 / 12 / 3 / 17 |
| libero_90:77 black book (wo) | 1 | 0 | 0 | 1 | 18 / 15 / 18 / 20 |
| libero_90:79 black book (wo) | 0 | 0 | 0 | 0 | 20 / 17 / 16 / 20 |
| libero_90:80 black book (wo) | 0 | 0 | 0 | 0 | 18 / 16 / 18 / 20 |
| libero_90:83 black book (wo) | 0 | 0 | 0 | 0 | 1 / 16 / 17 / 15 |
| libero_90:61 (harm) | 20 | 20 | 20 | 20 | 20 / 20 / 20 / 20 |
| libero_90:59 (harm) | 20 | 19 | 18 | 20 | 20 / 20 / 18 / 20 |
| libero_90:9 (harm) | 20 | 20 | 19 | 17 | 20 / 20 / 20 / 20 |
| libero_90:50 (harm) | 15 | 13 | 8 | 15 | 18 / 15 / 8 / 16 |
| libero_spatial:6 (cell) | 20 | 20 | 20 | 20 | 18 / 18 / 20 / 20 |
| libero_90:70 (cell) | 20 | 19 | 16 | 20 | 20 / 19 / 16 / 20 |

### 8.2 Pooled per set, paired differences against zs

| set | arm | success | diff vs zs [95%] | named grasp | diff [95%] | wrong-object diff [95%] |
|---|---|---|---|---|---|---|
| wrong-object (140) | zs | 0.014 | | 0.557 | | |
| | vis_e1 | 0.000 | -0.014 [-0.043, +0.000] | 0.543 | -0.014 [-0.229, +0.271] | -0.043 [-0.314, +0.171] |
| | vis_a025 | 0.000 | -0.014 [-0.043, +0.000] | 0.529 | -0.029 [-0.364, +0.314] | -0.007 [-0.350, +0.336] |
| | orc_e1 | 0.021 | +0.007 [-0.021, +0.043] | 0.679 | +0.121 [-0.036, +0.343] | -0.136 [-0.364, +0.007] |
| harm (80) | zs | 0.938 | | 0.975 | | |
| | vis_e1 | 0.900 | -0.037 [-0.125, +0.025] | 0.938 | -0.037 [-0.125, +0.000] | +0.037 [+0.000, +0.125] |
| | vis_a025 | 0.812 | -0.125 [-0.312, -0.013] | 0.825 | -0.150 [-0.400, +0.000] | +0.125 [+0.000, +0.388] |
| | orc_e1 | 0.900 | -0.037 [-0.138, +0.050] | 0.950 | -0.025 [-0.087, +0.000] | +0.025 [+0.000, +0.087] |
| damage (40) | zs | 1.000 | | 0.950 | | |
| | vis_e1 | 0.975 | -0.025 [-0.100, +0.000] | 0.925 | -0.025 [-0.125, +0.075] | +0.025 [+0.000, +0.100] |
| | vis_a025 | 0.900 | -0.100 [-0.275, +0.000] | 0.900 | -0.050 [-0.275, +0.150] | +0.050 [+0.000, +0.175] |
| | orc_e1 | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | +0.050 [+0.000, +0.175] | +0.000 [+0.000, +0.000] |

Discordant episodes (won from zs failures / lost from zs successes):
- wrong-object set: vis_e1 0 / 2, vis_a025 0 / 2, orc_e1 3 / 2;
- harm set: vis_e1 2 / 5, vis_a025 3 / 13, orc_e1 3 / 6;
- damage set: vis_e1 0 / 1, vis_a025 0 / 4, orc_e1 0 / 0.

**Failure types, wrong-object set (zs -> vis_e1 -> orc_e1):**

| failure type | zs | vis_e1 | orc_e1 |
|---|---|---|---|
| wrong_object | 57 (other type 38, wrong instance 19) | 51 | 38 |
| wrong_goal carried_elsewhere | 47 | 54 | 64 |
| motion | 28 | 20 | 27 |
| no_commit | 5 | 13 | 7 |

Guidance moves failures out of wrong_object and into wrong_goal. It does not move them into success.

### 8.3 Anchor localisation in closed loop

A hit means the anchor's xy is within 5 cm of the named object's simulator position; the rate is an episode mean.

| arm | wrong-object set | harm set | damage set | all |
|---|---|---|---|---|
| vis_e1 | 0.409 [0.264, 0.537] | 0.465 [0.231, 0.698] | 0.697 [0.597, 0.787] | 0.471 [0.351, 0.582] |
| vis_a025 | 0.407 | 0.453 | 0.667 | 0.461 |

- Median anchor error per task (vis_e1): 3.9-4.7 cm on :61, :9, :6 and :70; 6.5-12 cm on the black-book tasks; 14-23 cm on
  :38, :58, :59 and :50. These are not the same criterion as D3's offline token-in-region 0.62.
- Guided queries per episode: about 8-27. The median |D - d*| after guidance is 1.6-5.5 cm.
- Guided queries took about 0.45-0.84 s on 4 instances per A40, against about 0.49 s for unguided serving in the smoke.

### 8.4 Decision (pre-declared rule, section 5): a confirmatory read is NOT warranted

| criterion | value | holds |
|---|---|---|
| wo success gain >= +0.10 | -0.014 | no |
| wo named-grasp gain > 0 | -0.014 | no |
| harm >= -0.05 | -0.037 | yes |
| damage >= -0.05 | -0.025 | yes |

### 8.5 Reading (exploratory)

1. **Approach guidance can fix the grasp where the anchor or the pose is right, but the part tasks then fail at the goal.**
   - On :83, the named grasp rises from 1 / 20 (zs) to 16 (vis_e1), 17 (vis_a025) and 15 (oracle), and wrong-object falls
     from 18 to 3.
   - Success stays 0: those episodes then fail later (wrong goal, motion or no commit). The new part also breaks the goal role,
     and D4 found no place operator there.
2. **Where zs already grasps the named object, a wrong anchor makes it worse.**
   - On :73 the named grasp falls from 20 to 12 (vis_e1) and to 3 (vis_a025). The anchor misses there (median 12 cm), and the
     type-level prototype cannot separate the instance from look-alikes.
   - Pooled over the set, :83's gain and :73's loss cancel (-0.014).
3. **On the never-seen-type tasks (:38 moka pot, :58 ketchup), nothing moves, oracle included (0-3 named grasps).** The
   category-level operator and early guidance do not pull the approach off the misbound object. The policy's late Euler steps
   restore its own choice (R4-bind's "pull-back").
4. **The oracle is a weak upper bound:**
   - named grasp +0.121 [-0.036, +0.343] on the wrong-object set;
   - success +0.007;
   - harm -0.037 (:9 drops 20 -> 17).
   The guidance mechanism itself, not only the anchor, is the bottleneck.
5. **Harm and damage are small for the primary config** (-0.037 and -0.025). The all-step w 0.25 config hurts: harm -0.125, and
   :50 drops from 15 to 8, mostly through wrong-object grasps. This is the off-manifold cost R4-bind measured offline, now visible
   in closed loop.

**Recommendation.** Do not run a pre-registered confirmatory read of this guidance. Both the vision and the oracle arms fail
the gain criteria. The grasp-level effect is real on one task (:83), but it does not reach success, because the place role is
also unbound. Better candidates:
- an object-AND-goal intervention for new parts;
- a better instance-level anchor (hit 0.41 on the wrong-object set);
- and only after those, a closed-loop read.

**Cost.** 3.41 A40-h of rollout packs (11 packs 3.34 h + smoke 0.07 h), plus about 10 short CPU jobs.

**Disclosures.**
- The empty `python3 -` heredoc on the login node (section 7) ran before any rollout. It computed nothing, but it breaks the
  hard rule.
- Two extra background waiters were started by mistake and stopped at once; one waiter was kept.
- The read was run once with `--record`. The per-task named-grasp and type numbers above were printed from its log.
- Localisation uses the 5 cm xy criterion against the simulator position (R4's corridor), not D3's segmentation-region hit.
- The training-cell zs is a re-roll on fresh_r5s; the 18 D4 tasks reuse D4's zs rows (paired by trial, exact serving parity
  shown).
- Per-instance ledger rows (r5s_<arm>, r5s_smoke_*) were written by roll.sh, as for every rollout.
- Audits after the read: audit_numbers 0 discrepancies over 2,064 checks; audit_coverage 0 uncovered numerals.
