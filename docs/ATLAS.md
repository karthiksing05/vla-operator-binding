# Failure atlas: compositional zero-shot in a flow VLA (synthesis, 2026-10-25)

Policy: pi05_l90c@5000, our pi0.5 trained on the LIBERO-90-C training cells only, so held-out combinations are unseen at the
weights. Dev set: 29 held-out tasks (15 recombinations of held parts, 14 with a never-seen part), 20 fresh start states each.
The sealed third is untouched. Every number below is ledger-backed in the commit cited.

## 1. How zero-shot fails

| finding | number | source |
|---|---|---|
| zs solves recombinations, never new parts | 180 / 300 recombination, 0 / 280 part | M1, bdb324b1 |
| failures are mostly binding errors | 215 of 339 typed failures are wrong-object or wrong-goal | FM, b7fb0644 |
| binding errors do not grow with compound divergence; they are worst in new scenes | flat across CD 0.1 / 0.3 / 0.5 | FM |
| the scene alone drives demonstrated behaviour (vision shortcut) | empty instruction completes a same-scene training task in 0.466 [0.313, 0.626] of dev episodes; its own task in 99 / 200 training-cell episodes | R1, fe79413a |
| zs failures are misbindings, not replays | only 38 / 400 zs failures complete a demonstrated task, nearly all from two tasks | R1 |
| the right object is usually not among the frozen policy's own options | named object is a null-field mode in 0.534 of pre-grasp observations, 0.19 on wrong-object episodes | R2a, fd474579 |
| binding leaves the word positions by layer 2 and lives in the image tokens | restoring word positions recovers at most 0.25 from layer 2 on | L0b, docs/PLAN_LANG.md |
| action-space part modes are not readable | attribution about 1.8x, transfer about 0, R0 at most 0.60 | P2/P3, 5e65bc83 |
| the approach IS object-frame on training tasks, but the part prompt does not carry it | leave-scene-out object-frame transfer pi05_libero 0.62 (world -0.07, control -0.03; pre-reg GO) but null prompt also 0.62; pi05_l90c part 0.23 / instr 0.41 (fail); yaw adds nothing; P2's ~0 was its first-frame data (its own statistic pooled on approach frames 0.66-0.79) | OF D1, docs/PLAN_OF.md |
| the vision tower finds the named object even where zs misbinds; the words do not | SigLIP prototype anchor hit 0.62 on wrong-object vs 0.68 on successes (chance 0.04; gate passes); object-word attention (layers 3-12) 0.22-0.46 vs 0.65-0.89; identity leave-scene-out 0.80 / 0.65 (gate 0.8 fails narrowly), drops 0.1 far from the usual position | OF D2/D3, docs/PLAN_OF.md |
| the approach is a causal binding operator, shared across object types; the goal role has none; zs's action follows the grasped object | displacing the named object +-2 cm moves the chunk endpoint toward it: bound in 0.864 of training observations (chance 0.20), gain lambda 0.1-0.25 for all 19 types, leave-type-out R^2 0.44 = type-shuffle reference 0.44 (control -0.01; bar 0.5 fails); place role bound 0.48, R^2 about 0; on zs wrong-object episodes the action follows the grasped object in 0.69 vs the named in 0.27 (bar 0.7, 5 tasks); bank 109 operators, 7,619 bytes; all four pre-registered tests fail | D4, docs/PLAN_D4.md |
| closed-loop approach guidance (vision anchor + live depth + D4 bank, early x0-projection) fixes the grasp on some part tasks but not success; the oracle pose barely helps (EXPLORATORY) | 7 wrong-object part tasks: success zs 2 vs guided 0 of 140 (oracle 3); named grasp 0.557 -> 0.543 (oracle 0.679); on :83 named grasp 1 -> 16 of 20, still 0 successes; anchor within 5 cm in 0.41; harm -0.037 on 4 zs-success tasks, damage -0.025 on 2 training cells; all-step w 0.25 harms -0.125; confirmatory read not warranted | R5s, docs/PLAN_R5S.md |
| the goal role IS bound, at the release: displacing the target 5 cm moves where the frozen policy releases, for every target type; zs wrong-goal errors are mostly the wrong REGION of the right body | target is the bound body in 1.000 of 139 training carry observations (chance 0.23); leave-type-out R^2 0.81 vs control 0.02; gains lambda 0.30-0.92 (plate 0.80, basket 0.68, cabinet 0.60, shelf 0.43); the first chunk alone gives 0.49-0.54 and R^2 about 0 (why D4's place role looked unbound); on zs wrong-goal: bound = actual body 0.54 of 13 different-body cases (G3 fails), 28 more are same-body region errors (:29 top side vs drawer, :40 shelf levels) that the sandbox reproduces (its release lands in zs's region 0.79, the named one 0.25); SigLIP goal anchor in the target's region 0.97 / 0.92 / 0.69 (train / dev success / dev wrong-goal) | D5, docs/PLAN_D5.md |
| closed-loop goal guidance through the D5 operator does not raise success, even with the oracle goal (EXPLORATORY) | 6 wrong-goal recombination tasks: success zs 0.325, vision both 0.292, goal-only 0.350, oracle 0.358 (diffs -0.033 / +0.025 / +0.033, all CIs include 0); oracle halves :16's wrong goals (14 -> 7), which become motion failures; :29 / :40 region errors barely move; anchor within 5 cm of the region 0.14; goal-only: no harm, no damage | R6, docs/PLAN_R6.md |
| the frozen policy's own samples do not contain the fix once the object is carried; from the episode start they rarely do | coverage@K (32 noise seeds, sandboxed) on 42 recombination wrong-goal states: from the carry state cov@16 0.200 [0.00, 0.39] (:16 0.39, :29 0.16, :40 0.00; 31 of 42 states 0 of 32), so carry-phase selection has an oracle ceiling of 0.20; from the episode start cov@16 0.572, cov@32 0.810 (per-seed 0.056); zs-seed run reproduces the failure on 0.95 of states, controls succeed 0.80 per seed | R7, docs/PLAN_R7.md |
| the fix is in the policy's samples before the grasp, but its first chunk there gives no selectable signal | pre-grasp coverage@16 0.529 [0.35, 0.73] on the same 42 states (carry 0.200, start 0.572; pre-grasp - carry +0.329 [+0.16, +0.65]); top-1-of-32 first-chunk selection: learned ceiling (dev success labels, leave-state-out) +0.036 [-0.086, +0.161], leave-task-out -0.012, mode heuristic +0.083 [-0.076, +0.209]: one-step pre-grasp selection closed; next = re-conditioning (docs/PLAN_R8.md, draft) | R7 option 1 + stage B, docs/PLAN_R7.md |
| re-conditioning the region phrase at test time lowers success instead of raising it | on the 28 :29 / :40 region-error pre-grasp states (R7 seeds 0-15, per-seed 0.042): C3 training-activation phrase prototype -0.040 [-0.069, -0.021] (controls 0.19 -> 0.00), C2 region-contrast w 1 -0.018, w 2 -0.020 (controls -0.11 / -0.13); the dev region phrases are already the training phrases: a grounding failure, not a lexical one | R8, docs/PLAN_R8.md |
| a failure-gated retry with fresh noise and the original instruction is what the policy already does | zs wrong-goal episodes all run the full 400 steps; about 260 steps remain after the wrong release; the policy re-grasps spontaneously in 0.44 of released runs (:16 0.95, :29 0.07, :40 0.43) and 21 of 75 seed successes come after a re-grasp: the specified retry is a no-op by construction | R9, docs/PLAN_R9.md |
| no trained sub-instruction exists for a gated re-pick | of the 73 training instructions none is a standalone pick-up; the frying pan (:40) has no pick-up form at all; the black bowl's 10 pick-up forms are compound libero_spatial pick-and-place-on-the-plate tasks; a switch would be prompt engineering, so R10 stops at step 1 | R10, docs/PLAN_R10.md |
| exemplar binding holds for the goal role, not for the object role (E1, falsified pooled) | 135 zs binding failures: scene-exemplar minus frequency-matched baseline +0.001 [-0.558, +0.555] pooled (7 tasks define it); goal role +0.483 [+0.132, +0.872] (:29 cabinet top 18 of 19, :40 shelf top), object role -0.549 [-0.969, -0.115]; nearest-instruction account -0.148 | PLAN_BIND E1, docs/PLAN_E12.md |
| the policy's own early samples recover the object binding, but a nearest-site release label cannot measure the goal binding (E2, rank FAILS) | 24 tasks x 40 runs: Spearman(P_hat, zs success) +0.341 [-0.090, +0.716], slope 0.453; P_obj 0 on the 3 wrong-object tasks and about 1 elsewhere; the tray region vs body and in-tray objects capture the label (:60 0.05, :61 0.00 vs zs 0.95 / 1.00); competence failures (:32, :45 P 0.98-1.00, zs 0) | PLAN_BIND E2, docs/PLAN_E12.md |
| P is measurable: the predicate-labelled first release of 40 early own-policy runs per task rank-predicts zs success (E2b) | 24 tasks: Spearman(P_hat, zs) +0.763 [+0.441, +0.979]; S_hat = P_hat x competence rho +0.880, slope 1.00 [0.87, 1.13]; first releases reproduce the zs wrong regions (:29 cabinet top, :40 shelf top, :16 third bowl) and separate competence failures (:45 P 0.975, competence 0); caveat: the sandbox's own success already predicts zs at +0.917 | PLAN_BIND E2b, docs/PLAN_E12.md |
| no tested account predicts the wrong OBJECT (E1b) | 41 units (effectively :58, :83): SigLIP type similarity +0.122 [-0.160, +0.438], exposure +0.097, source slot +0.073 over the frequency-matched baseline; :58 ketchup -> tomato sauce unexplained | PLAN_BIND E1b, docs/PLAN_E12.md |
| our own binder resolves trained combinations but not new scenes (S1, FAIL) | B1b (frozen policy image tokens + own word embeddings, one head per role, training-cell labels): depth-2 held-out hit 0.378 [0.111, 0.667] (bar 0.70); seen depth 2 1.00 vs anchor / GroundingDINO 0.00; held-out depth 0 0.24 vs GroundingDINO 0.90; depth 0 below the anchor (0.733 vs 0.867) | PLAN_SOLVE S1, docs/PLAN_S1.md |
| factorised binder (GroundingDINO body x relation / instance head) transfers types to new scenes, not instances or sub-regions (S1b, FAIL) | depth-2 held-out seen-body hit 0.60 (1 task; bar 0.70); depth 0 0.821 vs GroundingDINO 0.786; held-out depth 0 (seen body) 0.911 vs B1b 0.267; instances weak ("middle black bowl" 0.20 vs B1b 1.00); caddy compartments 0.125 | PLAN_SOLVE S1b, docs/PLAN_S1.md |
| a once-trained binding adapter with ORACLE masks fixes binding, not motion (S0, PASS) | 24 dev tasks x 40 runs: binding errors 0.234 -> 0.048 (diff -0.186 [-0.311, -0.072]); motion failures 0.490 -> 0.573; success +0.103 [+0.017, +0.217]; wrong masks -> binds the masked object 0.61 (zs 0.08); training-cell damage -0.030 [-0.070, +0.008]; 8.4 MB adapter on deep image K/V, policy frozen | PLAN_SOLVE S0, docs/PLAN_S0.md |
| an adapter + DEPLOYABLE binder (S1b masks) does not close the oracle gap and damages trained cells (S2 smoke, exploratory) | 24 dev tasks x 20 runs: binding error zs 0.229, oracle 0.052, deploy 0.215 (closure 0.08 [-0.85, 0.48]); success zs 0.277, oracle 0.388, deploy 0.246; training-cell damage -0.18 [-0.29, -0.075] (mostly motion); S1b object masks mostly inside the target in 0.58 of dev units | PLAN_SOLVE S2, docs/PLAN_S2.md |
| binding error rises with referring-expression depth (E4) | 24 tasks, 45 role rows: logit GLM b_depth +2.011 [1.032, 15.39] at matched scene familiarity; mean error type 0.180 / instance 0.381 / relational 0.945; :76 coding miss disclosed (recode 2.00 [1.10, 14.7]) | PLAN_BIND E4, docs/PLAN_E34.md |
| approach endpoints are mostly world-frame; layer-16 expert output carries the program (E3) | 54 approach states: object gain 0.313 [0.269, 0.355] at 5 cm, 0.271 [0.226, 0.314] at 10 cm, rigid 0.317; SS_world - SS_obj -0.0020 [-0.0024, -0.0016]; layer-16 injection locks to the source program (0.45 mm, gain 0.005); 16-17 arm trivial (last layer) | PLAN_BIND E3, docs/PLAN_E34.md |

**Reading.** The policy reads the instruction holistically, keyed on the scene. By layer 2 the instruction has been written
into the scene's image tokens with the wrong filler, and the action expert reads the misbinding from there.

## 2. What does not repair it (no target demo, no weight change)

| intervention | result | why | source |
|---|---|---|---|
| CAG-VA, training-free language guidance | w* 1.5: 173 vs 180; no w beats zs; damage -0.06 on training cells | amplifies the wrong binding; high w turns wrong-goal into no-commit | R1 |
| select among frozen null modes by instruction/null ratio | ratio loses to the policy 10:28 on wrong-object episodes | coverage, not ranking, is the bottleneck (ranking given coverage 0.93) | R2a |
| recombine stored training sentences (TLI-style) | compose_sel 150 vs zs 180 | imports source objects | M1, FM F5 |
| bank of part tokens fitted on training demos, oracle retrieval | 181 vs zs 170 vs ceiling 223 (closure 0.21, bar 0.5); no damage | tokens fit their own demos; held-out gain about 0 (3 of 16 positive) | R2, 4eff744a |
| same bank with a contrastive minimal-pair objective | held-out own-part gain positive for 3 of 16 | specificity only by damaging partners; the unit, not the loss, fails | R2c, 5bdaaf16 |
| early-layer attention steering to the oracle object region | wrong-object hit 0.077 -> 0.077 (bar +0.20); counterfactual target +0.001 | the misbinding already lives in the image tokens; where early queries look does not rewrite it | R3b, 12a9c42e |
| splice the model's own image K/V at the object's region from a same-scene training instruction (oracle source) | counterfactual hit -0.005 (bar +0.13); whole-image splice +0.44, full prompt +0.50 | binding is a distributed scene code in the deep image K/V, not local to the object's patches; no wrong-object case has a same-scene source (0 / 163) | R3c, docs/PLAN_R3C.md |
| guide the policy's own flow sampler toward a target pose (endpoint x0-projection with the OF D1 operator; oracle targets in the stage-0 test, vision anchor never reached) | all-step guidance moves chunks to an arbitrary object (+0.38 to +0.80; bar +0.13) but raises the flow-loss outlier share by +0.29 to +0.80 (bar 0.10); on-manifold settings move it by at most +0.06 | the action expert treats an unbound target as out of distribution; its last denoising steps pull the chunk back to the scene-bound choice | R4-bind, docs/PLAN_R4.md |
| the same guidance re-judged under a prompt that binds the target (same-scene training instruction; null prompt where none exists) | still fails: outlier rise +0.26 to +0.71 (bar 0.10); unguided instruction chunks alone rise +0.27; on the 32 observations with a binding prompt, guided chunks are outliers in 0.73-0.82 although that prompt reaches the target in 0.62 of its own chunks | guiding the action map does not produce the approach the model makes when it binds the target; the judge itself has a prompt-shift floor | R4b, docs/PLAN_R4B.md |
| reuse role tokens from earlier tasks with no fit | 8 vs zs 13 of 260 | per-task parts are not reusable words | L4, 75deff22 |

## 3. What does repair it (one demo of the target task)

| arm | recombination / 300 | part / 280 | bytes | fit |
|---|---|---|---|---|
| zs | 180 | 0 | 0 | none |
| one-demo K/V token | 234 | 117 | 73,728 | 123 s |
| one-demo LoRA r16, 1000 steps | 238 | 149 | 18.7 MB | 145 s |

- The token removes binding errors: 215 -> 34 (k = 1) -> 9 (k = 5); it saturates at k = 5 (token 446, LoRA 485 of 580), and
  the remaining gap is motion on new parts (FM).
- Routed by instruction text it damages a same-text training cell (-0.075); keyed by scene it does not (M1, L4).

## 4. One explanation for every failure

The policy does not hold object and goal as separable pieces. It reads the instruction holistically against the scene and
writes the result into a scene-wide code in the deep image tokens (layers 10-17; R3c: whole-image splice +0.44 of the full
prompt's +0.50, region splice -0.005). Each repair fails for a consequence of this:
- **no local handle:** attention steering (R3b) and region splicing (R3c) are inert;
- **no correct option inside the frozen model:** the named object is rarely a null mode (R2a) and guidance scales the wrong
  binding (R1);
- **no transferable part:** tokens fitted on training demos capture those demos, not the part (R2, R2c, L4);
- **no own correct state to borrow:** wrong-object failures occur exactly where no training cell binds that object in that
  scene (0 / 163, R3c).

The one-demo token works because it is fitted on the target task: it supplies the scene-specific binding directly. That is
demonstrated adaptation, not composition.

## 5. Open choice (for the user)

The demo-free recovery claim of Plan R is not supported on this policy. Candidate papers: (a) this atlas plus the one-demo
token as the cheap fix; (b) the atlas as a negative result on demo-free compositional recovery; (c) change training so that
binding becomes local, then retest demo-free composition (tens of A40-h).
