# Plan SOLVE: repairing binding in a frozen flow VLA (DRAFT for the user's approval, 2026-10-26)

No code, no ledger row, no rollout. Every number is a cited past result.

## 1. What the interpretability results require of a solution

| result | consequence for the design |
|---|---|
| Success = binding x motor competence, calibrated (E2b: S_hat slope 0.997; Spearman 0.880) | repair P only. Motion failures (:45, :32) are out of scope and must be reported as such |
| Binding error rises with referring-expression depth: type 0.18, instance 0.38, relational sub-region 0.945 (E4) | the repair must RESOLVE instances and relational sub-regions, which the policy's own vision prototypes do not (SigLIP region anchor within 5 cm 0.07-0.16, D5) |
| Goal binding is scene-exemplar-keyed (E1 goal role +0.48; :29 18 / 19 go to the training binding of :25) | the failure is predictable per scene, which gives a gate for free |
| Binding lives in a scene-wide deep image-token code (R3c: whole-image splice +0.44, region -0.005); prefix-KV swaps redirect (2609.39971: 109 / 200) | deliver the repaired binding through the deep image tokens, not words, attention or action guidance |
| Action-side steering fails (R4, R4b, R5s, R6) and operators only partly track objects (E3: gain 0.31 at 5 cm, world frame fits better) | do NOT retarget operators from outside (this drops S2). Let the policy's own operator act on a corrected binding |
| Binding is committed by the grasp (R7: 0.53 -> 0.20) | intervene from the first query to the grasp |
| Internal handles without new information fail (C4), and no extracted code transfers (R2, R8 C3) | the repair must add information the model lacks (who / where at instance and relational level) via a module trained once |

## 2. Method: a factorised binder plus a binding adapter (frozen policy)

**B1. Role-wise binder** (resolves the parse's roles to masks, one per role)
- Input: the parse (task structure from language) and the current image.
- Output: an instance mask for the object role and a region mask for the goal role ("the top drawer of the cabinet").
- **Two variants, compared:**
  - (a) an off-the-shelf spatial pointing model (RoboRefer / RoboPoint) or GroundingDINO + depth;
  - (b) our own binder trained ONCE on the training cells, from BDDL / sim labels of which instance and region each training task
    names. This is factorised by construction: one head per role, conditioned only on that role's phrase.
- Variant (b) is the composition test: does a binder trained on training combinations resolve held-out combinations, e.g. a
  "top drawer of" phrase in a scene where training never bound the drawer?

**B2. Binding adapter** (writes the binding where the policy reads it)
- A small module trained ONCE per policy on the training cells' existing demos. It turns the binder's masks into an additive
  edit of the deep image tokens (layers 10-17), so that the FROZEN policy's flow loss on each demo is minimised given the
  correct binding.
- Training uses counterfactual pairs that already exist in the training data: the same scene with different named targets.
- At test time the adapter receives the binder's masks. The policy's own operators then act.
- **Not policy fine-tuning:**
  - the policy weights stay frozen;
  - the adapter's only input is the binding;
  - it is trained once, before any held-out task, with no target-task demos.

**B3. Exemplar gate**
- Predict the scene's exemplar binding from the training lexicon (E1).
- Apply the adapter when the parse's binding disagrees with that prediction, or always. Both settings are reported.

## 3. Why this is novel against the closest methods

| method | what it does | how B1-B3 differs |
|---|---|---|
| ReGuide 2609.38616 | external poses + demo statistics; overrides the arm during transport | never overrides actions; covers relational sub-regions, which ReGuide's scope excludes; acts only on P |
| ECT 2609.39971 / CRAFT 2610.00524 | fine-tune the policy on counterfactual pairs | policy frozen; the binding is an explicit, factorised variable; a held-out-combination test of the binder itself |
| HAMSTER / RT-Trajectory / TraceVLA-style | train the policy to follow traces or marks | no policy training; the delivery site is the measured binding locus; the input is a role-wise binding, not a trajectory |
| CAG / TAG / IGAR | guidance or attention re-weighting | adds information the model lacks (C4 shows re-weighting alone fails) |
| VERITAS / RoboMonkey | sample selection with a verifier | changes conditioning, so it is not bounded by own-sample coverage (R7: 0.20 once carried) |

## 4. Experiments (pre-registered in order; each gate stops the line)

| id | question | arms | bar | cost |
|---|---|---|---|---|
| S0 | E5 dissociation with an oracle binding: does a correct binding delivered by the adapter fix binding errors and leave motion errors? | adapter + oracle masks vs zs, on the dev tasks | binding-error rate falls by >= 50%; motion-failure rate unchanged | adapter training about 10-20 A40-h + reads about 4 |
| S1 | Binder accuracy on held-out combinations | B1a vs B1b vs the SigLIP prototype anchor | mask hit (IoU or in-region) per depth level; B1b >= B1a on depth 2 | B1b training about 2-4 A40-h; reads on CPU / GPU |
| S2 | Closed loop with a deployable binder | adapter + B1a / B1b, with and without the gate | closure >= 0.5 of the zs-to-one-demo-token gap on recombinations; damage <= 0.05 on training cells | about 6-8 A40-h |
| S3 | Baselines | ReGuide (perception), CAG-VA, VERITAS-style, the one-demo token | pre-declared | about 10 A40-h |
| S4 | Cross-model | the adapter recipe on pi05_libero (public) and one other flow VLA | the same direction of effect | about 20 A40-h |

The confirmatory 15-task read and the sealed set follow S2, only with the user's approval.

## 5. Decisions for the user

**Decided (user, 2026-10-26).** Our own binder (B1b, trained once on the training cells) is the HEADLINE. The off-the-shelf
binder (B1a) is a comparison. The adapter is trained once per policy, never per task (clarified to the user).


1. **The adapter is new trained weights** (frozen policy, trained once, no target demos). This relaxes "no weight change". Is
   that acceptable? Without it, R1-R10 show there is no working delivery site.
2. Is an off-the-shelf binder (B1a) acceptable as the main deployable variant, or should B1b (our own binder, trained on
   training cells only) be the headline and B1a a comparison?
3. Budget: S0 is the gate, at about 15-25 A40-h including adapter training.

## 6. S1 result (2026-10-26; docs/PLAN_S1.md section 5; ledger s1_bind tag read)

**S1 FAILS as pre-registered.**
- B1b depth-2 held-out hit: 0.378 [0.11, 0.67], below 0.70.
- At depth 0, B1b is below the SigLIP anchor: 0.733 vs 0.867.

**What failed.**
- B1b is near-perfect on seen phrase x scene combinations at every depth: depth 2 1.00, where the anchor and GroundingDINO score 0.
- It collapses in scenes no training cell covers: the STUDY_SCENE1-3 tasks score depth 0 0.24, and the never-labelled caddy's
  compartments are at chance.
- GroundingDINO is the mirror image: 0.90 on held-out depth 0, but 0.00-0.20 on sub-regions and qualifiers.

## 7. Framing and decisions (user, 2026-10-26)

**Framing: a compositional crutch.** The VLA already holds the necessary ingredients: motor operators, visual features and the
language lexicon. It still cannot always combine them zero-shot, and fine-tuning on the training cells does not teach it to.
A separate, small model stores the representational information and the bindings the VLA lacks, and helps it with input and
output. The pieces:
- the factorised binder: which body, and which part of it;
- the binding adapter: writes the binding where the policy reads it.

The VLA stays frozen and does the acting. The crutch is trained once per policy, never per task.

**Decisions:**
- A once-trained adapter is APPROVED.
- S0 budget of about 20 A40-h is APPROVED.
- Keep iterating until the theory holds.
- Next:
  - S1b: the factorised binder, body by open-vocabulary detection x relation in the body's own frame;
  - S0: adapter + oracle binding, the dissociation test.
