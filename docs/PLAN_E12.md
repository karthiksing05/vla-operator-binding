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

## 3. E1 result (ledger `bind_e1` tag read, results/bind/e1.json; pre-registration b95cabb4)

**Units.** 135 units (63 object, 72 goal) on 14 tasks; 80 carried_elsewhere episodes were excluded. EX is defined on 92 units (7
tasks) and undefined on 43, which lie in scenes with no training cell (the caddy STUDY scenes and :38's KITCHEN_SCENE8).
Intervals are two-level task-cluster bootstrap 95% intervals. With 7 tasks they are very wide.

| statistic | value |
|---|---|
| EX hit rate | 0.368 [0.08, 0.73] (n 92, 7 tasks) |
| frequency-matched baseline on the same units | 0.367 |
| **EX - P_freq (PRIMARY)** | **+0.001 [-0.558, +0.555]** |
| EX - uniform | +0.008 [-0.555, +0.545] |
| IS hit rate | 0.225 (n 135, 14 tasks) |
| IS - P_freq | -0.148 [-0.530, +0.211] |
| EX - IS (where both are defined) | +0.038 [-0.189, +0.252]; 52 discordant units: EX-only 34, IS-only 18 |
| **by role: goal, EX - P_freq** | **+0.483 [+0.132, +0.872]** (49 units, 4 tasks) |
| **by role: object, EX - P_freq** | **-0.549 [-0.969, -0.115]** (43 units, 4 tasks) |
| by role: IS - P_freq, goal / object | +0.245 [-0.142, +0.465] / -0.598 [-0.958, -0.224] |

**Per task (the zs wrong fillers against the predictions):**
- **:29:** the filler is the cabinet top side in 18 of 19. EX names the top side; this scene's training task :25 is "put the black
  bowl on top of the cabinet". IS also names the top side (0.67).
- **:40:** the shelf top side 9, bottom region 7. EX names the top side (:43, "on top of the cabinet"); IS names the bottom region
  (:42).
- **:16:** the third bowl 11, the plate 3. EX ties between the plate, the cabinet top side and bowl_3; IS names bowl_3 (:17).
- **:58:** tomato sauce 19 of 20. EX names alphabet soup / cream cheese; IS is split three ways.
- **:38:** moka_pot_2 in 20. EX is undefined (new scene), and there is a single candidate (P_freq 1.0).
- **:83:** the porcelain mug in 18. EX is undefined (new scene).

**Verdict (pre-declared): E1 FAILS.** EX - P_freq has a lower bound of -0.56 <= 0. Prediction (a) is falsified as stated, for
binding failures pooled over roles.

**Reading (descriptive).**
1. **Goal role (region / goal errors): the exemplar account holds strongly.** EX - P_freq is +0.48 with a CI above 0, on :29, :40
   and :16's plate errors.
   - The zs goal is the region that the scene's own training instruction binds. This is the R8 lexicon observation, now measured.
   - It also beats the instruction-similar account (IS) on goal: +0.48 vs +0.25.
2. **Object role (wrong-object errors): neither account holds; both are below the baseline.**
   - The wrong object is not the scene's most frequent training object, and not the most similar instruction's object.
   - On :58 the policy picks the tomato sauce, which no :58-scene training instruction or nearest instruction names first. On :38
     and :83 the scenes have no training cell at all.
3. **The theory therefore needs a role-specific refinement before E3-E5.**
   - "Binding P is exemplar-like" holds for the goal role.
   - The object role's errors are not explained by scene exemplars and need another account: for example visual similarity /
     salience, or a nearest-type prior. E1 does not test those.
4. **Power.** Only 7 tasks define EX, and two tasks with 18-20 units dominate each role. The pooled interval spans about +-0.55.
