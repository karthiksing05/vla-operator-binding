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
