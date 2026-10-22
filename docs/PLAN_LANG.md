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
