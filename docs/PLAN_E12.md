# PLAN_BIND E1 and E2: exemplar binding and a measurable P (pre-registered 2026-10-26, before either read)

These are the first two falsification tests of docs/PLAN_BIND.md section 2. Both run on pi05_l90c@5000 (frozen) and the 29
LIBERO-90-C dev tasks. Ledger: m3_prereg tags `pbind_e1` and `pbind_e2`; read rows `bind_e1` / `bind_e2` tag read.
split_sealed.json is never opened, and no demo is executed.

## 1. E1, exemplar binding (CPU only; experiments/bind_e1.py, bind_e1_scene.py)

**Prediction (a).** In a zs binding failure, the wrong filler is the binding that the SCENE's training tasks use most.

**Units.** Every zs binding failure on the dev tasks, from results/fm/types (arm zs, fresh_m1, 20 episodes per task).
- **wrong_object:** the role is object; the filler is the bound object.
- **wrong_goal / other_region:** the role is goal; the filler is the BDDL region (else object) where the named object ended. The
  most specific name is used. If more than one distinct name remains, the unit is excluded as ambiguous.
- **wrong_goal / carried_elsewhere:** the role is goal; the filler is the scene candidate site nearest (xy) to the object's final
  position, if it is within 10 cm. That is D5's rule; the positions come from LIBERO, in results/bind/e1_scene.json. Otherwise
  the unit is excluded.

From the dry run, which printed counts only:
- 135 units: 63 object, 72 goal; 21 of the goal fillers are nearest-site labels;
- 80 carried_elsewhere episodes excluded (no site within 10 cm);
- 14 tasks;
- 43 units in scenes with no training cell, where EX is undefined.

**Candidates.**
- Object role: the dev task's movable objects minus the named one.
- Goal role: the scene's BDDL goal candidates for the named object (fm_types.Scene.alternatives) plus the observed fillers, minus the
  correct filler and the held object.

**Accounts.** Each predicts the filler. Ties count fractionally (1 / |tied set|).
- **EX (exemplar binding):** the role's filler used most often across the training cells of the dev task's scene (split.json
  "train", 76 cells), excluding the correct filler. EX is undefined in a scene with no training cell.
- **IS (instruction-selected trajectory):** the role's binding in the training cell whose instruction is most similar (word
  Jaccard, all 76 cells), mapped into the dev scene: the same name, else the unique candidate with the same type key.

**Baseline (PRIMARY, frequency-matched).** Pick a candidate in proportion to 1 + the global training count of its type key for that
role. The expected hit is P(filler). Uniform over the candidates is reported as a descriptive baseline.
- Type key: an object's type is its name without the instance suffix; a region's is its body type + the region suffix.

**Falsification bar (set now).** E1 FAILS if the two-level task-cluster bootstrap 95% CI (B 10,000, seed 0) of
mean(hit_EX - P_freq), over the units where EX is defined, has a lower bound <= 0.

**Separation (descriptive):**
- IS - P_freq;
- EX - IS on the units where both are defined;
- discordant units (EX-only vs IS-only hits);
- each per role.

## 2. E2, measuring P (a few A40-h; experiments/bind_e2.py, bind_e2_pack.sbatch, D5 sandbox and server unchanged)

**Prediction.** P, the policy's mass on the correct binding, can be estimated from its own early samples, and it predicts per-task
zs success, with calibration.

**Tasks and states.**
- Tasks: the 24 dev tasks with an In / On goal atom. The named object is the first In / On atom's object; its own targets are that
  object's In / On targets.
- States: fresh_m1 states k 0-4, the first 5 of the 20 that zs ran from, each followed by the client's 10 wait steps.
- Runs: K 8 noise seeds per state (51e6 + 1e5 k + q), so 40 runs per task, 960 in total.

**Each run.** The frozen policy (10-step Euler, replan 5) runs until the RELEASE (the first executed open command after an object is
bound), LIBERO success, or the 400-step budget.
- **Bound object:** the first movable object moved > 3 cm from its start while the gripper is commanded closed (fm_types' rule).
- **Release site:** among the scene's goal candidates and the own targets, the site nearest (xy) to the bound object at the release.
  A success before any release counts as the own target.
- **Correct:** the bound object is the named object AND the release site is an own target. This is the policy's intent, not the
  physical outcome.
- **P_hat(task):** the share of correct runs. A run with no grasp within the budget counts as not correct.
- **zs success(task):** fresh_m1's 20 zs episodes.

**Statistics.**
- Spearman rho(P_hat, zs success) across tasks, with a task-bootstrap 95% CI (B 10,000, seed 0).
- Calibration: the OLS slope of zs success on P_hat, with its bootstrap CI.
- Descriptive: rho(P_obj, zs), the mean absolute error, and per-task values.

**Falsification bars (set now).**
- E2 FAILS on rank if rho <= 0 or its CI's lower bound <= 0.
- E2 FAILS on calibration if the slope CI lies entirely below 0.5 or entirely above 2.0 ("excludes 1 by a wide margin").

**Known caveats.**
- P_hat's states are 5 of the 20 zs states, with different noise.
- Tasks whose success requires a second step (e.g. :1 closes a drawer first) can bind correctly and still fail. Those failures count
  against calibration, as the theory's competence term predicts.

**Cost (planned).** About 960 runs, mostly ending at the first release: roughly 2-3 A40-h on 2 A40 jobs x 3 pairs. Together with E1
on CPU, this is within the 6 A40-h budget.
