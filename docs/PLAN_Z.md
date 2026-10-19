# Plan Z: new-task learning without fine-tuning, as compositional OOD lifelong learning

Plan of 2026-10-19 (user direction). It supersedes the fit-time agent (F1) and restarts the unsupervised-segmentation agent (U1)
as tracks of one plan.

## Revision (user, 2026-10-19 night): the paper is a theory of composition with two schemes

Fine-tuning is allowed where it is needed. The priority is the broader theory of composition, shown through two learning schemes
on the same frozen policies and the same graded split (docs/SPLIT.md):

- **Supervised scheme** (what exists): concepts read from the instruction (object, instance, goal); held concepts are
  pointers, new ones are fitted prototypes and leaves; practice chooses sources and operators.
- **Unsupervised scheme** (track U): units cut at the demonstration's own contact events and named by the policy's likelihood;
  held units are pointers, unexplained units are fitted (with the fast fit of F1); a library of discovered units, with no
  instruction grammar.

Each scheme is reported per grade (G1 relation, G2 binding, G3 scene, G4 part). The grade predicts reuse vs create. Every unit
of both schemes gets a donation table (track I).

The user's other ideas stay:
- binding at the VL→A interface as the prototype (track B);
- language compositionality (the splice);
- a highly reusable library with cheap fits (F1);
- the framing: prior knowledge donating expertise in unorthodox ways; Fodor's systematicity and task-to-task transfer in
  cognitive psychology; robotics as the testbed.

## The claim (first version; see the revision above)

Learning a new task from one demonstration needs no fine-tuning when the new task is a composition of what the policy holds.
The demonstration is a query, never a training set:
1. **Segment** it into units with the policy's own likelihood (unsupervised; no instruction grammar).
2. **Explain** each unit by a held binding, chosen by likelihood (no gradient).
3. **Compose** units in time, and words in the prompt (language compositionality at the VL→A binding).
4. **Create** only for a unit that no held binding explains: a new binding at the VL→A interface, sized and timed per unit.
5. **Fold** the new binding back as a word, so later instructions compose it by language.

Fine-tuning (the rank-4 LoRA prototype, 38.9 min; per-task LoRA) becomes the fallback and the baseline. The headline number is
the share of OOD tasks learned with zero gradient steps, and its success relative to fine-tuning.

## What we already know (constraints on the design)

- **Composition runs in time, not superposition.**
  - Grasp-switched pointers (zero fitting) already beat Li's TLI on all four policies (G15/G18).
  - Field averaging (GPC) and ungated mixing fail (G20/G21).
  - So units are temporal segments.
- **Success factorises at the interface:** success ≈ g × p (H1, r ≥ 0.91 on four policies). The grasp is one discovered
  boundary that segmentation must recover.
- **Held vs new is detectable** by the policy's flow loss (I6, AUC 1.0).
- **Learned KV tokens at the VL→A interface were tried, and the binding failed to transfer** (RESULTS rows q7, q11, q12):
  - A demo token gets 140 vs a LoRA's 252.
  - A token fitted on one task learns the task, not the object: reused elsewhere it scores 0/80.
  - A token shared across tasks moves toward the object but carries its sources' placements.

  **Lesson:** a binding must be fitted on one unit (segment) and one word, not on a whole task. That is what Tracks U and B
  change.
- **The policy does not compose language zero-shot.** The new instruction alone (zs) fails, and the policy runs the scene's
  habit (f1_binding). So "write the new sentence" is the baseline, not the method. The binding must be repaired per word.

## Track U: unsupervised segmentation (the units)

| step | what | gate / kill |
|---|---|---|
| U1 | Cost matrix: the frozen pi0.5's flow loss at every 2nd frame of a demo under each of its 40 training instructions (`experiments/zu_score.py`, kept). Demos: LIBERO-90 (one each), LIBERO-10 (two-step, own instruction column excluded), and the 20 OOD tasks' demos | — |
| U1r | Segment with `segment.label_viterbi` (switch cost λ, one λ for all demos via `normalise`). Score boundaries against the gripper's sustained grasps and releases (scoring only) | Kill the label-switch segmenter if grasp recall at ±10 frames < 0.6 on pick-and-place. Fallback: `level_partition` on the best-field loss, or change points of the empty-prompt latent |
| U2 | Units. Each segment is labelled held (its best field's loss within the I6 calibrated threshold → a pointer, 0 bytes) or new. Cluster new segments across tasks (span latent + action statistics) | Report units per task vs the supervised library (48 prototypes for 66 tasks) and the reuse rate. The out-of-template tasks (drawers, knobs, pushes, two-step) are the point |
| U3 | Online, pre-registered: switch held pointers at the DISCOVERED boundaries instead of the grasp, on the 20 OOD tasks (fresh set g). Then on out-of-template LIBERO-90 / LIBERO-10 tasks the grammar cannot express | U3a: non-inferior (−0.05) to the grasp switch. U3b: beats zs and the policy under the task's own instruction |

## Track B: the binding is the prototype (VL→A, language-compositional)

The action expert reads the VLM's per-layer residuals / keys-values at the instruction tokens. That interface is the binding
between language-vision and action. A held concept's binding exists in the policy, at that word's tokens, under the training
instruction where it was learned.

| step | what | gate / kill |
|---|---|---|
| B0 | Binding library: per training instruction, per layer, per instruction token (full span, not Li's first 9), mean over demo frames (`lifelong_ood/policy/bind.py: identify`). Analysis: does a word's binding agree across instructions and cluster by word and role (an unsupervised prototype hierarchy over word bindings)? | — (interpretation) |
| B1 | **Word splice**, training-free. Read the NEW instruction; at each content word's tokens, layers ≥ 1, write the binding that word has under the training instruction that holds it (object word from the object source, goal word from the goal source). Variants: replace, or delta (source − the new sentence's own mean on the demo frames). The 20 OOD tasks have NO demonstration (the OOD system already learns them with zero fitting), so the gate is the online pilot B2; an offline demo-likelihood gate runs on the L90 tasks that have a demo | Kill (L90 offline) if splice's demo loss is not below the new instruction's on ≥ 2/3 of the 26 both-held tasks |
| B2 | Online pilot (practice set a, 10 episodes, 20 OOD tasks): splice alone (one prompt, no switch), splice + grasp switch, vs pfield and zs | Go if splice ≥ zs + 0.1 on the pilot. Splice + switch ≥ switch is the composition-of-composition test |
| B3 | **Create** a binding for a new unit: optimise only the novel word's binding (the KV of its tokens at the interface; the prompt cache once per frame, gradients only through the action expert, per `token_fold.py`), on that unit's frames only (from U). Target fit ≤ 2 min vs 38.9 | Pre-registered NI (−0.05) vs the LoRA prototype on the E-tasks with a new object; fit time and bytes |
| B4 | **Fold:** register a created binding as that word's binding (vocabulary-level). Reuse it in later instructions by language ("put the ⟨w⟩ on the stove"), with no refit | Transfer test the token era failed (q11: 0/80): the same word in another goal / another scene |

## Track I: what each unit donates (user direction, 2026-10-19)

A soft blending operator over the held fields: a convex mix per contact segment and per action dim (Wang-style per-dim PoE
weights; `poe.py`'s form), fitted to the demo by flow loss on stored velocities. Nothing in the network is fitted.

| step | what |
|---|---|
| I1 | Expert velocities on the demos: `experiments/zi_velocities.py`. Familiar scenes l40 (own instruction held out as a ceiling); LIBERO-90 Z3 tasks |
| I2 | `experiments/zi_blend.py`. Held-out loss: blend vs hard switch vs uniform superposition vs each expert. Per expert: weights per segment × dim; exact Shapley donation per segment × {position, rotation, gripper}; donation share by the expert's relation to the task (shares object / goal / both / neither); object × goal Shapley interaction (synergy vs redundancy) |
| I3 | Donation in behaviour: the fitted blend served online (vs hard switch, zs). Per expert, the ablated blend's change in success and in the end effector's path toward the object / the place (h1_donate's measures) |
| I4 | The same donation decomposition for the splice (per layer, per word) and for fitted prototypes (h1_donate), so every operator gets one donation table |

## Track Z: the zero-fine-tune ledger (what composition alone achieves)

On every benchmark already run, count the tasks learned with 0 gradient steps (pointers + switch + splice, chosen by
likelihood or practice). Report success vs the fine-tuned library and vs LoRA, plus wall-clock to learn a task:

| row | measure |
|---|---|
| LIBERO OOD 20 | zs / pointers / splice / fine-tuned prototype |
| L90 69 | 26 tasks have both concepts held as words in the 40 training tasks (E3 18, E5 8): pointers + splice, zero fitting; 33 have one new concept (B3 creates a unit binding); 3 have none |
| LIBERO-PRO | |
| per-policy | the same rows per policy |

The fit-time levers from F1 apply only where creation remains (B3): early stopping, prefix caching, unit-scoped frames.

## Track FL: FLOWER with no fine-tuning

`mbreuss/flower_libero_goal` holds LIBERO-Goal's 10 tasks (training indices 10–19). For 6 of the 20 OOD tasks (goal:4–9: wine
bottle / cream cheese × stove / plate / bowl / cabinet), both concepts are held inside its suite, and our rule's sources
(11/13 × 10/13/14/17) lie in it. So the system runs with zero fitting.
- Arms: zs, pointers at the grasp, Li's pairs at the grasp, splice (after B1).
- Protocol: fresh set e, 20 episodes each, pre-registered.
- The other 14 tasks need concepts FLOWER-goal never saw. They are reported as outside its library, not as failures.

This is the sixth architecture (Florence-2 encoder + DiT), with no new training at all.

## Protocol

- Pre-register each confirmatory read in `ledger/m3_prereg`, with paired episodes and the exact sign test.
- Report the task-level bootstrap; non-inferiority at −0.05.
- New fresh start states: set g (seed 1,000,000) for U3 / B2-test; set h for B3/B4.
- Development is only on practice set a and on demos.
- Audits: `audit_numbers.py`, `audit_coverage.py`. Commits local only.

## Order (cheapest decisive first)

| order | work | cost |
|---|---|---|
| 1 | FL launch | ≈ 6 × 3 arms × 20 episodes |
| 1 | U1 scoring | ≈ 1 A40-h per 20 demos |
| 1 | B0 identify | ≈ 2 A40-h per policy |
| 2 | U1r, B1 | offline, CPU / 1 GPU |
| 3 | B2 pilot, U3 pilot | |
| 4 | B3/B4 | |
| 5 | confirmatory reads, then the other four policies | |
