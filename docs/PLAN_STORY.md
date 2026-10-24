# The story: VLAs compose through language (draft, 2026-10-24)

**User direction.** If the language route works (M1, docs/PLAN_M1.md), the paper argues that language is compositional
in the philosophical sense, and that a vision-language-action model can use this property. The paper also maps where
composition fails in a traditional VLA, and explains each failure with the theory and with the solution.

## 1. Thesis

1. **Language is compositional by definition.**
   - Frege's principle: the meaning of a sentence is a function of the meanings of its parts and how they are combined.
   - Fodor and Pylyshyn: systematicity and productivity follow, and role-filler binding is explicit in syntax.
   - A sentence about a new task is built from words and roles the model has seen, so it is in distribution at the level of
     the parts.
2. **A VLA's action space does not inherit that structure.**
   - P2/P3 on pi05_libero, pi05_l90c and pi05_l90c_fac (5e65bc83): part prompts give no readable action modes (attribution
     about 1.8x), no transfer, and R0 at most 0.60.
   - L0/L1: binding leaves the word positions by layer 2 and lives in the image tokens. Scene-specific, distributed,
     holistic.
   - The VLA reads the instruction holistically, keyed on the scene.
3. **So compose where the structure exists: in the language channel.**
   - Store language prototypes: sentence, role-phrase and word states.
   - Compose them by interpolation, transplant and role schedule (L2, L2c).
   - Create a word token only when composition cannot cover a part (M1 create arm).
   - The policy stays frozen. Composition inherits language's systematicity, and creation adds a new lexical item, as
     a learner of a language does.
4. **Prediction.** Composition succeeds where the failure is in *reading* a new combination of held words, and creation is
   needed where a word or motion is genuinely new.
   - The graded split already shows this shape: reuse wins at G1/G2, creation is needed at G3/G4.
   - Training on a factorial set yields composition by itself (CALVIN-C: no headroom).

## 2. Failure map: where few-shot composition breaks in a traditional VLA

**Setup.**
- Arms: pi05_l90c zero-shot, plus per-task LoRA from k demos (k = 1 from M1; add k = 5 and 10), at M1's step budgets.
- Tasks: M1's 29 LIBERO-90-C dev tasks, same fresh states and seed.
- Sealed tasks untouched.

**Axes** (each task gets every axis label from split.json):

| axis | levels |
|---|---|
| compound divergence | bins 0.1 / 0.3 / 0.5 |
| part novelty | all parts held / one part never seen |
| scene | familiar / new |
| instance | names an instance qualifier (front, middle, back) / does not |

**Failure types**, read per episode from the sim state and the gripper events (factors.py, segment.py):

| failure | episode reading | role-filler label |
|---|---|---|
| wrong object | the grasp lands on a non-named object | object-role binding error |
| wrong goal | right object, placed at the wrong target | goal-role binding error |
| motion failure | right object and target, but the grasp or the placement fails | missing skill |
| no-commit | the episode never grasps | — |

**Theory explanations, one testable prediction each:**
- **F1 holistic binding.** The VLA keys the instruction on the trained scene.
  - Prediction: zero-shot wrong-object and wrong-goal errors rise with compound divergence, at fixed atom divergence.
  - The solution's test: language composition removes this rise, because composing roles in language is systematic.
- **F2 few-shot memorises the demo, not the parts.**
  - Prediction: LoRA from one demo binds the demo's layout. Its errors are wrong-object on new start states, and they grow
    with layout distance from the demo.
  - Prediction: more demos reduce F2. Composition is flat in k, since it needs no layout-specific fit.
  - Test: the success-vs-k curve, by failure type.
- **F3 instance averaging.** A prototype pooled over instances averages their approaches (E1/E2).
  - Prediction: instance tasks fail by wrong-instance grasps for pooled units and for zero-shot.
  - The solution's test: word-level prototypes keyed by the qualifier fix them. Language marks the instance explicitly.
- **F4 a genuinely new part.**
  - Prediction: no composition covers a new part; only creation or fine-tuning moves it, and the failures are motion
    failures.
  - The solution's test: compose-or-create matches LoRA here at less than 1/100 of its bytes (M1 H2/H3).
- **F5 interference** (G0 exploratory: recombination hurts where the policy already performs).
  - Prediction: composition lowers success on tasks zero-shot already solves.
  - The solution's test: the zero-shot guard (compose_sel keeps zs unless a composition scores lower) removes it.

**Readouts:**
- Failure-type shares by arm × axis.
- Mixed-effects logistic regression of success on arm × CD (the systematicity test).
- The success-vs-k curve by failure type.
- One composition card per fixed task: source sentences, roles, a created token if any, and the change in the grasped
  object.

## 3. Decoding compositionality through language (C3, specialised)

- **Systematicity:** swap the object and goal roles (X on Y vs Y on X); the composed prompts give the mirrored behaviour.
- **Substitutivity:** synonymous phrasings and instance paraphrases compose to the same field and the same success.
- **Localism:** replacing the goal phrase leaves the object-approach segment unchanged (the L0 trace by segment).
- **Productivity:** two-step LIBERO-10 instructions composed from single-step sentence prototypes with a THEN schedule.
- **Contrast:** the same tests in the action space failed (P2/P3). That contrast is the paper's evidence that composition
  lives in language, not in actions.

## 4. Order

1. M1 read (running). It gates the thesis: compose > zs, and create ≈ LoRA at less than 1/100 bytes.
2. Failure map on M1's rollouts: failure typing needs no new rollouts for zs, compose, create and lora-1. Add k = 5/10 LoRA
   fits (about 29 x 2 fits plus rollouts, roughly 10 A40-h). Pre-register F1-F5 before reading the k = 5/10 arms.
3. Decoding suite (section 3) on the stored prototypes.
4. The continual store (docs/PLAN_L4.md). Key repairs by instruction + scene, because M1 found duplicate instruction texts.
