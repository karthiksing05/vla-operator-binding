# Plan L: language as the compositional scaffold for new trajectories (for review, 2026-10-22)

**Question (user).** Language is compositional. When a zero-shot prompt fails on a new task, can we extract the
prototypes that correspond to its language and use them either to **update the model sparsely** or to **compose working
actions fast**?

This plan complements docs/PLAN_V4.md, where prototypes are action-space modes under part prompts. It answers that plan's
first failure: on pi05_libero, isolated part prompts barely move the action field (P2 dev). The fix is to work where the
VLA's language actually lives: its text latents, its word positions and the prefix keys and values (K/V).

## What we already know (ledger-backed)

| result | what it says |
|---|---|
| Li's text-latent interpolation (TLI) on our harness (G12): pi0 325 / 400 vs 35 alone | **text latents compose**: interpolating two training instructions' latents gives a working novel instruction |
| Word-level splice (Z2 pilot, pi0.5): 80 / 200 vs 67 alone; object edit before the grasp and goal edit after it: 77 | **word latents are partly portable**, and a word's latent carries its source sentence's placement: it wins when the source shares the template ("put the X …") and loses when it comes from "pick up X and place it in the basket" |
| P2 / P3 dev on pi05_libero: part-prompt modes ≈ null; R0 rebuilds 0.51 of the instruction-vs-null gap; spatial clauses 0.10 | in familiar scenes the field responds little to isolated part prompts, so **prototypes must be taken in sentence context** |
| LIBERO-PRO (G22): given the new instruction, pi0.5 recovers part of each suite | instructions carry usable task content |
| Literature: LIBERO-Plus; LangForce (2601.15197); "When instructions retrieve trajectories" (2609.39971); CRAFT (2610.00524) | VLAs often treat an instruction as a **trajectory key**, not a compositional program, so failures are **binding failures** |

## The idea

A failed zero-shot instruction is usually a binding failure: some words are not grounded in this scene or combination,
so the VLA retrieves a familiar trajectory. The plan runs in four stages:

1. **Diagnose.** Find which words are ungrounded by measuring each word's causal effect on the action field (a lexicon audit).
2. **Extract language prototypes** for the failing words. A prototype is the word's latent and K/V state in a context where
   the VLA *does* ground it (a training sentence), together with the action-field effect that state produces.
3. **Repair**, cheapest first:
   - **(a) Fast composition, no gradient.** Transplant or interpolate grounded word states into the new sentence. The source
     is chosen by the one demo's flow loss.
   - **(b) Sparse update.** Edit only that word's representation: one inverted token (about 2 KB), or a rank-1
     locate-and-edit of the layer that binds it.
4. **Reuse.** A repaired word is reusable in every later sentence that contains it. We repair the **lexicon**, not the
   policy. That is what makes this continual and compositional.

## Stages

**L0. Lexicon audit.**
- For every content word (object, goal, verb, spatial relation), measure its causal effect:
  - ‖eps(instr) − eps(instr with the word swapped for a same-category alternative)‖, normalised by the instruction-vs-null gap;
  - whether the change points at the referent (endpoint attribution from `protos/store.py`).
- **Causal tracing**, ROME-style (Meng et al. 2202.05262): corrupt the word, then restore single (layer, position) states.
  This locates where its grounding lives, in the VLM residual stream and in the prefix K/V the action expert reads.
- Output: a grounding score per word × scene, and a binding map per layer × position.

**L1. Does grounding predict zero-shot failure?** (pre-registered)
- Predict held-out zero-shot success (Li's OOD suites and the LIBERO-90-C dev set) from the minimum and mean word grounding.
- Go: AUC ≥ 0.75. The audit is then a free failure detector, run before any rollout.

**L2. Fast composition from language prototypes (no gradient).**
- **Operators**, applied to the ungrounded words:
  1. **Word transplant**: generalise `policy/bind.py`. The source sentence is chosen by the demo's flow loss, preferring one
     with the same template.
  2. **Template-matched transplant**: re-express the instruction in the grounded word's training template first. This fixes
     Z2's placement leak.
  3. **TLI**: interpolate whole-sentence latents of the two training tasks that hold the parts, with weights chosen on the
     demo.
  4. **Per-role schedule**: the object transplant before the grasp and the goal transplant after it. This matches the
     relation readout from P3, where the object role is AND and the goal role is THEN.
- **Interpretation**, for every repair: the source sentence, the layer span, the change in the action mode (endpoint,
  nearest object, gripper timing), and the drop-one effect in closed loop.

**L3. Sparse update, used only if the best L2 operator fails** the demo's flow-loss threshold (calibrated leave-task-out).
1. **Token inversion** (Gal et al. 2208.01618): one new embedding for the failing word, fitted on the one demo through the
   frozen VLA. About 2 KB, and no weights change.
2. **Rank-1 binding edit** (ROME / MEMIT, 2202.05262 / 2210.07229): a rank-1 update at the layer L0 located. It is closed
   form, and ROME's locality constraint applies.
3. **Escalation:** a rank-4 LoRA leaf, only if both fail.

The tests that make "sparse" meaningful:
- no damage to earlier tasks;
- the repaired word works in combinations never seen with the demo;
- edit size and GPU-seconds, against per-task LoRA.

**L4. Continual lexicon.**
- Each arrival: audit → L2 → L3 only for words still ungrounded.
- Repairs go into a lexicon store, keyed by word.
- Metrics:
  - how often repaired words are reused;
  - forward transfer (fewer repairs needed over time);
  - backward transfer (old tasks after many edits; mass rank-1 edits are known to interfere, so it is measured);
  - bytes per word.

**L5. Decoding compositionality of the language scaffold** (plan v4, section 6). Each property is checked on word-effect
matrices, binding maps and behaviour:
- **systematicity**: a word repaired in one sentence works in the others;
- **substitutivity**: grounded synonyms act alike;
- **localism**: an edit to "ketchup" changes no other word's effect;
- **productivity**: two-step LIBERO-10 instructions built from repaired words.

## Benchmarks and baselines

**Benchmarks:**
- Li's OOD suites (zero-shot prompts that fail);
- LIBERO-90-C dev (pi05_l90c);
- LIBERO-PRO `_task`;
- CALVIN-C (FLOWER's language is a CLIP-style vector, so a second kind of language interface).

**Baselines:**
- the new instruction, zero-shot;
- TLI with Li's pairs;
- ACT-VLA (synthesise, then SFT);
- per-task LoRA from the one demo;
- LLM prompt rewriting (the obvious cheap alternative);
- "When instructions retrieve trajectories" (2609.39971), if code exists.

Every arm gets one demo; practice is charged if used.

## Probes, cheapest decisive first (all on SLURM)

| | question | go / kill | A40-h |
|---|---|---|---|
| L0a | word-effect audit on pi05_libero / pi0 (training + Li OOD sentences) | descriptive: words split into grounded and ungrounded | 2-3 |
| L0b | causal tracing per word kind | binding sits in a small layer × position span (else drop the rank-1 edit) | 2-4 |
| L1 | grounding predicts zero-shot failure | AUC ≥ 0.75 | about 1 |
| L2 | offline demo flow loss per operator, then closed loop on 8 dev tasks × 20 episodes | the best operator beats zero-shot (sign p < 0.05) and matches TLI with Li's pairs, with no practice | 6-10 |
| L3 | token inversion and rank-1 edit on tasks L2 cannot fix: success, locality on 20 old tasks, transfer of the repaired word | non-inferior to per-task LoRA at ≤ 1/100 of its parameters; old-task change ≥ −0.01 | 8-12 |
| L4 | continual lexicon stream | reuse > 0; FWT > 0; BWT ≥ −0.01 | 15-20 |

## Files

**New:**
- `lifelong_ood/lang/audit.py`: word effects and tracing;
- `lifelong_ood/lang/transplant.py`;
- `lifelong_ood/lang/invert.py`;
- `lifelong_ood/lang/edit.py`;
- `lifelong_ood/lang/lexicon.py`;
- `experiments/pl*_*.py` and `experiments/prereg/pl_*.py`.

**Reuse:**
- `policy/bind.py`, `policy/tli.py` (Li's latents across five policies), `policy/token_fold.py`;
- `protos/store.py` (attribution);
- `experiments/z2_pilot.py`, `g12_tli.py`, `fresh_states.py`, `launch_pack.py` / `serve_guided.py`;
- the ledger and prereg system.

## How it fits plan v4

Language prototypes (grounded word states) are the **fillers**, and the sentence template is the **relation**: the VLA's
own syntax. The two plans meet at the per-role schedule (object before the grasp, goal after it). If pi05_l90c still ignores
part prompts at step 5,000, plan L is the cheaper alternative to the factored-conditioning retrain: it repairs only the
words that fail.

## Results (2026-10-22: L0a, L0b, L1)

Code: `lifelong_ood/lang/audit.py`; `experiments/pl0_render.py` (OOD first frames), `pl0_audit.py`, `pl0_trace.py`, `pl1_predict.py`;
pre-registration `experiments/prereg/pl1_predict.py` (`ledger/m3_prereg` tag pl1_predict, recorded before any grounding score).
Ledger rows: `ledger/pl0_audit` (read_pi05, read_pi0), `ledger/pl0_trace` (read_pi05), `ledger/pl1_predict` (read). Payloads
`results/lang/pl0_audit.json`, `pl0_trace.json`, `pl1_predict.json`. Figures `figures/robot/pl_audit_kinds.png`,
`pl_audit_ood_heat.png`, `pl_trace_map.png`, `pl_l1_predict.png`.

**Set-up.** 60 sentences (40 training, Li's 20 OOD). Observations of each sentence's own scene: training sentences at 10 first
frames (stored init states 0-9) plus the 5 demonstration frames of `results/protos/index.json`; OOD sentences at the 10 first
frames of practice start-state set a. At each observation about 34 prompts. G = ‖eps(swap) − eps(instr)‖ / ‖eps(instr) −
eps(null)‖, at shared points (2 drafts × 2 noises × t ∈ {0.2, 0.4, 0.6, 0.8}), averaged over alternatives and observations.
Binding direction: cosine between the endpoint shift of paired samples (same noise) and the referent → alternative vector.
Chance is 0. Object swaps are scored at pre-grasp frames and goal swaps at post-grasp frames; the OOD scenes have no post-grasp
frames, so OOD goal direction is not measured.

**L0a: grounding by word kind** (mean G; training sentences at first frames only, so they compare with OOD):

| kind | pi0.5 train | pi0.5 OOD | pi0 train | pi0 OOD |
|---|---|---|---|---|
| article "the" (control) | 0.10 | 0.12 | 0.08 | 0.15 |
| verb | 0.27 | 0.49 | 0.22 | 0.56 |
| goal preposition | 0.17 | 0.32 | 0.23 | 0.55 |
| spatial relation | 0.47 | 0.45 | 0.56 | 0.65 |
| goal phrase | 0.42 | 0.85 | 0.38 | 0.83 |
| object | 0.95 | 1.33 | 0.66 | 0.94 |
| whole instruction (ceiling) | 1.59 | 2.13 | 1.38 | 1.51 |

- Object direction: pi0.5 cos +0.38 (train, n 1770) and +0.47 (OOD, n 700); pi0 +0.29 and +0.27.
- Goal direction (training, post-grasp): pi0.5 +0.40, pi0 +0.54.
- The swapped endpoint ends closer to the alternative's referent in 61-80% of swaps.
- Words do not split into grounded and ungrounded by size. Every OOD object, goal and verb moves the field by more than twice its
  sentence's article control (both policies). The only words that fall below that floor are the goal preposition (pi0.5: 28/35
  training, 6/20 OOD) and pi0's verb on training sentences (14/39).
- OOD words move the field MORE than training words. The normaliser is not the cause: the instruction-vs-null gap is about the same
  (pi0.5: 1.12 training, 1.21 OOD).
- Binding direction is where the failing sentences differ: on pi0.5 the object-swap direction is about 0 on "the bowl on cookie
  box" and "the bowl at table center" (−0.14 to +0.13), where zero-shot success is 0-17%.

**L0b: binding map** (pi0.5, 20 sentences × 3 kinds × 2 observations = 120 traces; the noise moves the field by 2.0-2.5× the
instruction-vs-null gap):

- Residual stream. Restoring the word's own positions recovers the field only at layer inputs 0-1. At the word's last token this is
  0.66 for object, 0.80 for verb and 0.19 for goal (goal phrases span several tokens). From layer 2 onward it recovers ≤ 0.25.
- The word's effect moves into the **image tokens** by layer 2. Restoring all image positions recovers 0.18-0.33 at layer 2 and
  peaks at 0.48 (object), 0.33 (goal) and 0.48 (verb) at layers 8-11.
- Text tokens other than the word (before it, after it, BOS) carry about 0. pi05_libero's prompt holds no state tokens
  (discrete_state_input is off).
- Prefix K/V (what the action expert reads). Any single (layer, word position) recovers ≤ 0.20. Restoring the word's K/V at every
  layer recovers only 0.10 (object / verb) and 0.05-0.22 (goal).
- The largest single K/V cell is the image positions at layer 14: 0.34 object, 0.19 goal, 0.33 verb. Image K/V at every layer
  recovers 0.58 / 0.40 / 0.60.
- Training and OOD sentences give the same map.

**L1: grounding predicts zero-shot failure** (pre-registered; 40 units = 2 policies × 20 tasks; 80 zero-shot episodes per unit
over sets c-f; 31 failures):

| | AUC(−min G) [95% CI over tasks] | AUC(−mean G) | Spearman(min G, success) | Spearman(mean G, success) |
|---|---|---|---|---|
| pooled (primary) | **0.459** [0.24, 0.67] | 0.627 | −0.08 (p 0.61) | +0.22 (p 0.18) |
| pi0.5 | 0.505 | 0.670 | −0.02 (p 0.94) | +0.35 (p 0.13) |
| pi0 | 0.556 | 0.500 | +0.13 (p 0.60) | +0.06 (p 0.79) |

**The pre-registered GO (AUC ≥ 0.75) FAILS.** The minimum is almost always the verb "put", which is the same in every OOD sentence.
Exploratory, not pre-registered: the object binding direction gives AUC 0.64 and Spearman +0.40 (p 0.08) on pi0.5, but AUC 0.31
on pi0.
LIBERO-90-C dev: no zero-shot dev episodes of pi05_l90c were used. This is left for later.

**Reading for L2 / L3.**
- **L1 is killed** as a free failure detector. Word-effect size does not predict zero-shot failure on either policy. Swap-effect
  size is therefore the wrong diagnosis signal, and L2 must not route repairs by it. Every OOD word already moves the field, often
  more than in training. The failure is in **where** the words send the action: the binding direction, about 0 on the failing
  spatial tasks.
- If L2 runs, it needs a direction-based diagnosis, with that metric pre-registered on fresh tasks. Today that signal is weak and
  policy-specific (0.64 / 0.31).
- **L0b undercuts the rank-1 locate-and-edit plan of L3.** Binding does not sit in a small (layer, word position) span. After layer
  1 the word lives in the image tokens: a scene-specific, distributed state, spread over layers 2-15. The prefix K/V at the word's
  positions carries ≤ 0.1-0.2 of the effect. A ROME-style rank-1 edit keyed on the word's mid-layer state has no site to target:
  KILL the rank-1 edit as specified.
- Two operators remain consistent with the map:
  - an edit of the word's layer-0 / 1 representation (token inversion, L3 item 1). The word's input state does determine its
    effect: restoring it recovers 0.66-0.80.
  - a transplant into the early layers (L2 operators 1-2).
- An edit of the image-token K/V, layers 8-14, is also consistent with the map. It is scene-specific, so it is not a lexicon repair.
- Recommendation: run at most a cheap L2 pilot (transplant / TLI at layers 0-2) and token inversion. Drop the rank-1 edit. Do not
  build the L4 lexicon on swap-magnitude grounding.

## L2 results (2026-10-22: fast composition from language prototypes, no gradient, no practice)

Pre-registered as `ledger/m3_prereg` tag pl2_compose (experiments/prereg/pl2_compose.py) before any L2 number. Code:
`lifelong_ood/lang/transplant.py`, `experiments/pl2_compose.py`. Ledger rows: `pl2_offline`, `pl2_dev`. Figures:
`figures/robot/pl2_offline.png`, `figures/robot/pl2_dev.png`. Served through additive arms: `lt@<name>` in launch_pack.py, and
`lt:<spec>` in serve_guided.py.

**Protocol.**
- **Demos.** Li's tasks have no demos, so each task gets two recorded successes on a new start-state set (fresh_ood_pl2d,
  BASE 3300000). They come from the *other* policy's strongest set-d arm (G13). The policy's own arm is used only where the other
  policy failed: pi0 goal:3, spatial:6 and spatial:7. So these are policy-generated demos, not teleoperation. Demo 0 selects;
  demo 1 is held out.
- **Selection.** It uses only the flow loss on demo 0 (36 frames, 4 fixed draws): the object source, then the goal source, then
  the layer span from {1-17, 1-6, 7-12, 13-17}. Layer 0 is never edited. TLI picks its pair at lambda 24, then lambda from
  {12, 24, 36}.
- **l2sel**, the tested arm, is the operator with the lowest demo-0 loss on that task.
- **Controls** use l2sel's form:
  - rand: a seeded random sentence with no word overlap;
  - pos (T8): the no-overlap sentence whose training grasp or release is nearest demo 0's.

**Offline: held-out demo-1 flow loss.** "Ratio" is the geometric mean of the arm's loss over zs's; "wins" counts tasks where the
arm beats zs.

| | pi0.5 Li-20: ratio / wins | pi0 Li-20: ratio / wins | pi0.5 LIBERO-90-16: ratio / wins |
|---|---|---|---|
| null prompt | 1.20 / 5 | 1.11 / 6 | 0.99 / 13 |
| TLI, Li's pairs | 0.92 / 15 | 1.03 / 9 | — |
| word | 0.91 / 15 | 0.91 / 17 | 0.96 / 15 |
| tmpl | 0.90 / 14 | 0.90 / 19 | 0.94 / 16 |
| tli_sel | 0.83 / 17 | 0.82 / 16 | 0.91 / 11 |
| role | 0.91 / 15 | 0.94 / 16 | 0.96 / 12 |
| **l2sel** | **0.83 / 16** (sign p 0.012) | **0.79 / 18** (p 0.0004) | **0.88 / 14** (p 0.004) |
| rand | 1.16 / 6 | 1.08 / 6 | 1.07 / 6 |
| pos (T8) | 1.30 / 2 | 1.16 / 6 | 1.00 / 6 |

- l2sel picks TLI on 13/20 tasks for both policies; word, tmpl or role on the rest.
- The template step rarely matters. Li's sentences already use "put the X Y", so tmpl equals word on most tasks.
- The layer span picked for word is mostly 1-17 on pi0.5 (13/20); on pi0 it is spread across spans.
- pi0 on LIBERO-90-16: l2sel 0.87 / 12 (sign p 0.077), word 0.94 / 16, tli_sel 0.90 / 10, rand 1.11 / 2, pos 0.99 / 7.

**Closed loop: 8 dev tasks × 20 episodes, fresh_ood_pl2 (never used), seed 13, paired.**

| arm | pi0.5 success | g / p\|g | pi0 success | g / p\|g |
|---|---|---|---|---|
| zs | 63 / 160 | 0.85 / 0.46 | 11 / 160 | 0.51 / 0.11 |
| TLI, Li's pairs | 84 | 0.86 / 0.61 | **106** | 0.93 / 0.70 |
| **l2sel** | **91** | 0.78 / 0.71 | 39 | 0.65 / 0.37 |
| word | 80 | 0.75 / 0.65 | 2 | 0.26 / 0.05 |
| tmpl | 80 | 0.75 / 0.65 | 2 | 0.25 / 0.05 |
| tli_sel | 81 | 0.82 / 0.60 | 40 | 0.78 / 0.32 |
| role | 73 | 0.84 / 0.54 | 3 | 0.29 / 0.04 |
| rand | 39 | 0.61 / 0.39 | 0 | 0.37 / 0.00 |
| pos (T8) | 15 | 0.29 / 0.33 | 15 | 0.34 / 0.27 |

Pre-registered tests:
- **pi0.5.**
  - H1 PASSES: 91 vs 63, discordant 34:6, sign p 8e-6. The task-level bootstrap interval still includes 0 (+0.175, -0.037 to
    +0.419): the gain sits on 3-4 tasks.
  - H2 PASSES: difference to Li's TLI +0.044, 95% interval -0.031 to +0.150, so the lower bound is above -0.05.
  - The controls fall far below l2sel (rand 39, pos 15).
- **pi0.**
  - H1 PASSES: 39 vs 11, sign p 6e-5.
  - H2 FAILS badly: difference -0.419, 95% interval -0.675 to -0.169, against Li's TLI at 106.

**Reading.**
1. **The pi0.5 result holds only conditionally: a weak operator choice fails safe.** With no practice and one demo, the
   demo-selected composition matches TLI with Li's hand pairs. Nearly all of the gain is from TLI with a demo-chosen pair:
   20/20 on goal:5, goal:6 and goal:7.
2. **Word transplants help pi0.5 but break pi0 in closed loop.** On pi0, word, tmpl and role reach 2-3 / 160, below zs's 11,
   although offline they beat zs on 16-19/20 tasks. Lower demo-1 flow loss does not transfer to rollouts for word edits on pi0.
   This is the main negative. For pi0, the demo flow loss is not a safe selector between operator families.
3. **The selector is not robust across tasks.** On pi0.5 goal:1 it picked role (11/20, zs 15). On goal:2 TLI's demo-1 loss was
   already worse than zs's (overfit selection). On pi0 goal:1 / goal:4, role's goal source was "turn on the stove": the word
   matched, but its role was wrong.
4. **T8 control: word grounding, not position, on average.**
   - The position-matched no-overlap source is the worst arm on pi0.5 (15/160) and well below l2sel on pi0 (15 vs 39).
   - Exception: pi0 goal:1, where pos reaches 15/20 while every word-matched arm gets 0. There, position or trajectory
     selection beats word grounding.
5. **Interpretation.**
   - Mode readouts at demo-1 frames, from protos/store.py interpret (`results/lang/pl2/offline_read.json`, keys modes and
     repairs), barely change the approach endpoint.
   - The repairs act mostly in the carrying phase, on where the motion points. For example, pi0.5 goal:7 (TLI "put the cream
     cheese in the bowl" → "put the bowl on the plate", lambda 12) moves the carrying direction from the bowl to the plate,
     and success goes from 8 to 20.
   - Gripper timing is unchanged. This agrees with p|g rising (0.46 → 0.71 on pi0.5) while g does not (0.85 → 0.78).
6. **Layer span vs the L0b trace.** The only spans tried start at layer 1, and spans 1-17 / 1-6 contain layers 1-2, where the trace
   locates the binding. Demo selection chose 1-17 or 1-6 on 13/20 (pi0.5) and 12/20 (pi0) tasks for word.

**Not yet done:**
- the confirm read on the other 12 tasks (rules frozen);
- a teleop-style demo source (these demos come from a policy, and on pi0.5 the source is TLI on pi0, which may favour the TLI
  family);
- a guard that falls back to zs when no operator's demo-0 loss beats it.

## L2 confirm read (pi0.5, 12 held-out Li tasks, 2026-10-22; ledger pl2_confirm, prereg addendum pl2_confirm)

| arm | success / 240 |
|---|---|
| zero-shot | 93 |
| Li's hand TLI pairs | 169 |
| l2sel (demo-0 selection, frozen) | 128 |
| tli_sel (demo-chosen TLI pair) | 143 |
| word / tmpl / role | 83 / 92 / 107 |
| rand / pos controls | 4 / 17 |

- H1 (l2sel > zero-shot) passes: discordant 77:42, sign p 0.0017; the task-bootstrap CI includes 0 (+0.146 [-0.171, +0.442]).
- H2 (l2sel non-inferior to Li's pairs) fails: -0.171 [-0.375, +0.004]. The dev non-inferiority does not replicate.
- Failure mode: with no zero-shot fallback the selector breaks tasks zero-shot already solves (spatial:0 20 -> 0 with tmpl;
  spatial:3 20 -> 7). The demo-chosen TLI pair is the strongest selected operator but stays below Li's hand pairs.
- Next: L2b chooses a selection rule on dev only; the confirm rollouts (all 8 arms rolled on every task) then test that rule
  once, as held-out data.

## L2b: which one-demo criterion predicts closed-loop success? (2026-10-23; ledger pl2b_criteria, prereg pl2b_rule)

- Demo flow loss is a weak ranker: within-task Spearman with closed-loop success +0.19 (demo-0), +0.32 (even/odd split), and
  the demo-1 loss is no better (+0.08). A minimal-edit (MDL) penalty points the wrong way (TLI moves the field most and is the
  best family). A fitted zero-shot guard is not identified at n = 8.
- Rule frozen on dev (even/odd split-demo validation, no constant): 135 / 320 on dev vs 130 for demo-0; on pi0.5 confirm
  129 / 240 vs 128. Always using the demo-chosen TLI pair gets 143; Li's hand pairs 169.
- Reading: the signal is between operator families, not within them. TLI with the right pair is the robust operator, and the
  pair matters more than any demo criterion. Li's pairs are chosen by which training sentences hold the new sentence's parts.

## L2c: demo-free role-coverage pair retrieval (2026-10-23; prereg pl2c_cover, ledger pl2c_read; fresh starts BASE 5100000)

| 20 Li tasks x 20 | zs | Li's hand pairs | auto-cover (Li lambda) | auto-cover (demo lambda) |
|---|---|---|---|---|
| pi0.5 | 146 | 249 | 193 | 211 |
| pi0 | 34 | 317 | 152 | 153 |

- H2 (auto > zero-shot) passes on both policies; H1 (non-inferior to Li's pairs) fails on both.
- Coverage matches Li exactly on 9 / 20 tasks. On the 11 others Li's sources come from the OOD task's own scene and template
  family and often hold none of the new words; coverage picks scene-blind word holders (libero_object "pick up X and place it
  in the basket") and loses (pi0 13 vs 178 / 220 on those tasks).
- Reading: matching the words in their role is not enough; the source must come from the same scene. This agrees with L0b
  (a word's effect moves into the scene's image tokens by layer 2): language prototypes are scene-bound keys.

## L2d: hierarchical scene key for demo-free TLI pairs (2026-10-23; prereg pl2d_scene, ledger pl2d_picks / pl2d_read; LIBERO-PRO task split, fresh starts BASE 5300000)

Rule (`experiments/pl2d_scene.py`): candidates = pl2c pool prompts in the new task's scene (factors.parse scene rule, base
suite), else its base suite, else all (each only with >= 2); inside, pl2c's same-role coverage, 2 sources, pl2c slots; lambda 24
(Li's modal lambda; Li's map has no entry for new tasks). Arms: zs, scene_cover, blind_cover (pl2c), scene_random (2 seeded
draws from scene_cover's candidates).

- **Design check (Li's 20, descriptive):** the scene rule reproduces Li's exact pair on 3 / 20 (pl2c: 9 / 20). Li's two sources
  are both in-scene on only the 10 Goal-OOD tasks; every Spatial-OOD pair takes a LIBERO-Goal source. So the L2c reading "Li's
  sources come from the task's own scene" holds only for Goal-OOD.
- **Test set:** 21 G22-eligible LIBERO-PRO task-split tasks with pi0.5 G22 fresh-t zs <= 15 / 20 (all qualifying; Goal 6, Object
  9, Spatial 5, LIBERO-10 1); used before only by G22. Offline, scene_cover = blind_cover on 20 / 21 tasks (only object:0 differs).

| 21 tasks x 20 | zs | scene_cover | blind_cover | scene_random |
|---|---|---|---|---|
| pi0.5 | 34 | 23 | 15 | 25 |
| pi0 | 12 | 4 | 4 | 11 |

- pi0.5: H1 fails (23 vs 34, discordant 13:24, p 0.099); H2 passes under Holm (8:0, p 0.0078), all on object:0 (8 vs 0) alone; H3
  fails (23 vs 25, p 0.87).
- pi0: H1 is significant in the WRONG direction (4 vs 12, 0:8, p 0.0078); H2 has no discordant pairs; H3 fails (4 vs 11, p 0.039,
  not Holm-significant, wrong direction).
- Reading: on new instructions in the training scenes, TLI with a coverage pair at lambda 24 does not beat zero-shot and on Goal
  breaks what zs solves (pi0.5 Goal: zs 17, scene_cover 0). The scene key only matters where blind coverage leaves the scene, which
  LIBERO-PRO's in-scene wording almost never triggers. Fixed-lambda TLI is not a safe default here.
