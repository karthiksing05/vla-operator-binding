# Probe R6: closed-loop smoke of goal-role guidance through D5's release operator (EXPLORATORY; written 2026-10-25, before any R6 rollout was read)

**Status: EXPLORATORY.** This smoke runs because D5's G1 passed (docs/PLAN_D5.md section 8). It decides whether a pre-registered
confirmatory closed-loop read (stage C) is worth running. It makes no confirmatory claim and reports no p-values.

**Question.** pi05_l90c@5000 stays frozen. While the policy carries the object, its chunks are guided early in its own flow sampler
toward the release point that D5's extracted goal operator predicts for the vision-anchored target. On the recombination dev tasks
with zs wrong-goal errors, does the policy then place the object where the instruction says? Does it still solve the tasks zs
solves?

Code: `experiments/r6_smoke.py` (stages), `experiments/r6_guide.py` (serving), plus two opt-in additions, both default off:
- the `r6:` mode in `experiments/serve_guided.py`;
- `R6_TGT=<region>` in `experiments/libero_client.py`: the named goal region's simulator position (meta/tgt_xyz), read by the oracle
  arm only and, in the vision arms, only to log the anchor's error.
Ledger: `m3_prereg` tag `pr6_smoke` (this design, exploratory) and `r6_read` tag `smoke`. Payloads: `results/r6/` (plan.json,
bank.json, specs/, smoke_check.json, types/, read.json tracked; roll/, smoke/, qlog/, packs/ local).

## 1. Tasks (pre-declared rules; results/r6/plan.json)

- **Wrong-goal set (6):** among D4's 11 recombination In / On dev tasks, the 6 with the most R2 zs wrong-goal episodes on fresh_rb2
  (results/rb2/types; ties sha1). R2's rows, not D4's, so the selection does not condition on the paired baseline. Result:
  libero_90:29 (19), :40 (18), :16 (14), :1 (3), :60 (1), :36 (1).
- **Harm set (3):** the 3 of the other 5 with the highest R2 zs success (ties sha1): :59 (20), :61 (20), :14 (19).
- **Damage set (2 training cells):** R5s's, libero_spatial:6 and libero_90:70 (fresh_r5s; R5s's zs rows).
- **States:** fresh_d4 for the 9 dev tasks (zs = D4's rows, paired by trial; client seed 7, replan 5, wait 10, serve seed 13,
  bind track: asserted), fresh_r5s for the cells (zs = R5s's rows).

## 2. Arms (every guided query: x0-projection on the first 5 of 10 Euler steps, w = 1, R5s's primary configuration)

| arm | approach (open gripper, before the first grasp) | carry (from the grasp until the first open chunk) |
|---|---|---|
| zs | none | none |
| `both` (PRIMARY) | R5s vis_e1: SigLIP anchor of the named object, D4's approach bank | vision-anchored goal, D5's release operator |
| `goal` | none | as `both` |
| `orc` (privileged upper bound) | R5s orc_e1: the named object's simulator position | the named goal region's simulator position, D5's operator |

**Phases** (r6_guide.Phase): `closed` = the chunk's 5th commanded gripper > 0; the grasp = closed on 3 consecutive queries (R5s); the
carry phase lasts while the chunk stays closed after the grasp; the first open chunk after the grasp is the release, and guidance is
off for the rest of the episode.

**Carry-phase target.**
- The goal anchor c: the argmax agentview SigLIP token of the target type's training prototype (results/rb4/protos.npz), through the
  live agentview depth (R5S_GEOM=1). The oracle arm uses the goal region's site position (R6_TGT).
- D5's release bank (results/r6/bank.json, built from D5's 139 scored training observations, release readout +-5 cm): per node, the
  mean J (2 x 2) and two offsets, b_orc = (e0 - h0) - J (p_region - h0) with the region's site, and b_vis = (e0 - h0) - J (c_anchor - h0)
  with the training observation's own vision anchor. Levels: type + region suffix (from the parse's goal atom: top_region, top_side,
  contain_region, ...) > type > category > role. Levels used: :29 category (the white cabinet is not a training target; the wooden
  cabinet's operator); every other dev task and libero_spatial:6 at the region level (:40 shelf top_region, :1 cabinet top_side, :16
  bowl, plates, trays); :70 role (a table region: it has no prototype, so its vision arms do not guide the carry).
- The predicted release r* = h + J (c - h) + b. The chunk's integrated xy displacement D is steered along r* - h, keeping its own
  length (at least 3 cm) and never going past r*: d* = min(|r* - h|, max(|D|, 3 cm)) unit(r* - h). The x0 projection moves D by
  w (d* - D), as in R4 / R5s.

## 3. Readouts (EXPLORATORY; two-level task-cluster bootstrap 95% intervals, B 10,000, seed 0, descriptive)

- Success per task and arm; pooled per set; paired differences vs zs; discordant episodes (won / lost).
- **Named-goal placement:** the named object's own goal atom holds at its final pose (fm_types' own_atoms_final).
- **Failure-type shift:** fm_types' typer on every episode (zs re-typed from D4's recordings; mismatches reported); wrong-goal and
  binding-error rates and their paired differences.
- **Harm:** the success difference on the harm set; **damage:** on the 2 cells.
- **Goal-anchor hit in closed loop:** on guided carry queries, the anchor's xy within 5 cm of the goal region's site (episode means).
- **Guidance bookkeeping:** guided approach / carry queries, carry fallbacks, operator levels used, time per guided query.
- **Cost:** A40-hours from the pack logs.

## 4. Serving smoke (before any rollout)

On libero_90:29, trials 0-1 (the first two of a fresh process), one pack of four instances: MODE ours with the R6 client flags
(R5S_GEOM, R6_TGT); r6 arm=off; r6 both and r6 orc (code-path check only, excluded from every number). Requirement: the first two
instances' executed actions equal D4's recorded zs actions exactly (results/r6/smoke_check.json).

## 5. Decision rule for stage C (pre-declared, exploratory)

A pre-registered confirmatory read (15 recombination tasks x 20 new fresh states, paired with zs and the one-demo ceiling) is
recommended only if the primary arm `both` raises pooled success on the wrong-goal set by >= +0.10, or its paired-difference
interval excludes 0. Harm and damage are read alongside (the confirmatory GO would need no harm and no damage). The oracle arm is
read alongside: if `orc` gains and `both` does not, the anchor is the bottleneck; if neither gains, the guidance is.

## 6. Cost (planned)

11 tasks x 3 guided arms = 33 instances of 20 episodes, 9 packs of 4 on one A40 each; about 3-4 A40-h (R5s: 3.4).

## 7. Known limits before the run (from D5)

28 of D5's 42 recombination wrong-goal observations are same-body region errors (:29 top side instead of the top drawer; :40 the
shelf's bottom region or top side instead of its top region). The region enters only through the bank's region-level offset and the
oracle's region site; the vision anchor is type-level (it lands on the cabinet, not on the drawer). Large gains on :29 / :40 are
therefore not expected from the vision arms.
