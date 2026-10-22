# Experimental design (v4, 2026-10-22)

The framework is in docs/FRAMEWORK.md, and the full approved plan, with its probes, in docs/PLAN_V4.md. This file supersedes
the 2026-10-21 design, which had CompoSuite as primary.

## 1. Claims (each can fail)

| | claim | primary evidence |
|---|---|---|
| C1 | **Test-time recomposition.** A never-seen task with all its parts held is solved from one demo by inference alone, at a success close to per-task fine-tuning and at a fraction of its cost, *and better than published composition methods* | success and part correctness on combination cells, as a function of compound divergence (CD) at fixed atom divergence; GPU-seconds to readiness |
| C2 | **Robust continual learning follows.** No forgetting (the policy and relation are frozen; routing is measured). New parts are created once and **reused by later recompositions**. Cost per arrival falls. Bytes grow with parts, not tasks | streams that interleave part tasks before the recombinations that use them: FWT / NBT / AUC, causal reuse, bytes vs parts |
| C3 | **The system is compositional, by decoding**: systematicity, productivity, substitutivity, localism, overgeneralisation; TRE, topographic similarity, multiplicativity, role orthogonality, modularity | the decoding suite (section 6). No robot paper measures these, which the survey confirmed |

## 3. Benchmarks: chosen to make held-out tasks guaranteed unseen *and* recomposable

The surveys of datasets and baselines (2026-10-22) found that public checkpoints contaminate every factorial split:
FLOWER-90 and the CALVIN checkpoints saw everything. **So we train our own checkpoints on training cells.** That is the only
way "never seen" holds at the weights.

### 3.1 Primary: LIBERO-90-C (our split of LIBERO-90; very well known; 50 human demos for every task)

- **Atoms per task** (read by `policy/factors.py`): object type, instance qualifier, target or region, verb, scene.
- **Compounds:** object×target, verb×object, object×scene, target×scene.
- **Candidate pool:**
  - (i) the 90 LIBERO-90 tasks, with the LIBERO-10 two-step tasks decomposed into their steps;
  - (ii) a **CRAFT-style 4×4 cell scene** authored in BDDL (cube colour × plate colour; CRAFT 2610.00524). It is the
    cleanest fully factorial sanity cell. Train on the 4 same-colour cells and test the 12 others. Demos are made by
    MimicGen (2310.17596) or the repo's v2 generator and labelled as generated;
  - (iii) optionally, about 40 authored recombinations in LIBERO-90 scenes, with generated demos.
- **Split construction (MCD; Keysers et al. 1912.09713):**
  - Greedily search a training set with **atom divergence ≤ 0.02** (every test atom seen at least twice in training).
  - Grade the combination test tasks into **CD bins 0.1 / 0.3 / 0.5**, at least 8 tasks per bin.
  - A separate **part-held-out set**: two objects and one predicate, e.g. ketchup, the caddy, "turn on", absent from every
    training task.
  - A **sealed third**, stratified, chosen by a seeded hash and committed to the ledger before any read.
- **Training set.**
  - Train our own **pi05_l90c** from `pi05_base` on the training cells, all 50 demos each.
  - Suite tasks are added only if none of their compounds is held out. LIBERO-10's two-step tasks would leak.
  - Also train **FLOWER from `flower_vla_pret`**, never `flower_libero_90`, for architecture generality.
  - Prototypes and R come only from these checkpoints and the training-cell prompts.
  - **Contamination audit:** flow loss of `pi05_base` on held-out demos against its loss on training demos (the I6 method).
- **Test set.** Each held-out task gets exactly one of its 50 demos as the query. Evaluation uses 50 never-used start
  states, never the query's state.
- **Cost:**
  - pi0.5 multitask over about 60 cells: about 40-100 A40-h (action expert + VLM LoRA), or about 150-300 with whole-model FSDP;
  - FLOWER: about 30-60;
  - reads: about 4.4 A40-h per 600 episodes.

### 3.2 Secondary: CALVIN-C (very well known; factorial blocks; long chains for productivity)

- **Cells.** 21 block cells: lift colour×place (9), push colour×direction (6), rotate colour×direction (6).
- **Held out:** 6-7 cells, keeping every colour, verb, place and direction in training at least twice.
- **Grading:** by how many of a cell's pairwise compounds are unseen (0 / 1 / 2). That is a CD ladder at zero atom divergence.
- **Part set:** one colour×verb held out entirely, e.g. all pink rotations.
- **Three split seeds.** Caveat: in CALVIN colour also encodes size.
- **Training.**
  - Fine-tune FLOWER from `flower_vla_pret` on ABC's annotated windows minus the held-out cells.
  - Drop windows that overlap held-out windows by more than about 16 frames.
  - Unannotated play still contains the motions; state this and run the flow-loss audit.
  - Recipe: 4 GPUs, batch 8, 40k steps, about 50-80 A40-h per split seed. The ABC data is already on disk.
- **Test.** Held-out cells in ABC (familiar scene, pure composition) and in D (composition plus scene shift: the
  scene-familiarity contrast *inside one benchmark*). One held-out window is the query. 5-step chains containing held-out
  cells give the productivity test, at about 1.5 A40-h per 1,000 chains.

### 3.3 Controls and reference reads (cheap, existing harnesses)

- LIBERO-PRO `_task` (combination, our existing checkpoints).
- Li et al.'s OOD suites, which are what ReGuide and ACT-VLA compare on.
- One LIBERO-Plus perturbed-layout read for robustness.

**Not added:**
- RoboCasa365. It is a strong official unseen-composite and lifelong benchmark, but very heavy, and its unseen composites
  sit in unseen kitchens, a confound. Kept as a possible future scale-up.
- GemBench / AGNOSTOS (RLBench, mixed levels) and CompoSuite (state-only, Dryad): cited as precedents.

### 3.4 Continual streams

- Arrivals in **two mirrored random orders**, mixing combination tasks and part tasks.
- **Each new part arrives before the recombinations that use it.** This directly tests "created units compose", which the
  design review marked as not shown.
- Re-test every earlier task every 5-10 arrivals.
- Report LIBERO's native FWT / NBT / AUC, the cost per arrival against its index, and causal reuse (ablate the created
  filler).

## 4. Baselines: recent published robotics methods, chosen for the claim each would refute

| baseline | what it does | why it is the right foil | how we run it |
|---|---|---|---|
| **ReGuide** (Zhang et al., 2609.38616, Sep 2026) | training-free wrapper on frozen pi0.5: rebinds the instructed object; steers the end effector with object poses plus training-demo statistics | the only published *training-free, frozen-pi0.5* compositional method (pi0.5 39.0 → 94.5 with oracle poses, 87.5 with perception) | reimplemented (no code); two versions, oracle sim poses and an off-the-shelf detector |
| **ACT-VLA** (Peng et al., 2607.00351, Jul 2026) | interpolate part text latents (Li's TLI), roll out, filter successes, SFT pi0.5 on about 300-400 synthetic demos | "bake compositions into the weights" (Spatial-OOD 35.5 → 88.2; Goal-OOD 46.6 → 95.6) vs our test-time composition | reimplemented from our `pi0-text-latent` port plus the SFT pipeline; report its rollout and GPU budget and its effect on old tasks |
| **MergeVLA-style** (Fu et al., 2511.18810, CVPR 2026) | task-masked sparse LoRA + TIES merging + training-free router | the strongest parameter-space, compose-only method; isolates field-space composition with a relation | reproduce mask + TIES + router over pi0.5 per-task LoRAs |
| **CLARE** (Römer et al., 2601.09512, RA-L 2026) | exemplar-free continual VLA: adapters expanded where feature similarity says so, autoencoder router with no task id | the direct foil for "create only the new part"; strong LIBERO continual numbers (Goal AUC 89.3) | code and LIBERO checkpoint released; run natively + a PEFT port to the pi0.5 action expert |
| **DMPEL** (Lei et al., 2506.05985, TMLR 2026) | progressive library of low-rank experts + learned mixing router | the closest "compose library experts for a new task" robot method; its codebase also yields LOTUS, IsCiL and TAIL in one harness | MIT code + a LIBERO-90 checkpoint; native + port |
| **ER on pretrained VLA** (Liu et al., 2603.03818) | experience replay; pretrained VLAs barely forget (pi0 NBT −0.016) | without it, "no forgetting" is not a selling point; it also re-checks our SeqFT | their recipe on pi05_l90c; SeqFT alongside |
| **Equal-weight part superposition** (composable diffusion 2206.01714 / GPC 2510.01068 style) + Comp-1/3 (ICLR-27's retrieval baseline) | sum (or a convex mix of) the same part-prompt fields with no distilled relation; or retrieve the nearest trained instructions | ablation-baselines that prove *the distilled relation and the discovered fillers* matter, not just the prompts | our machinery with R replaced |
| **Per-task LoRA** (the fine-tuning replaced) + **DemoDiffusion** (SDEdit; 2506.20668) | per-task fit with its full fit curve; one-demo SDEdit | the speed comparison for C1; the "edit the demo" reference | existing code |

**Optional:**
- **TraceFlow** (2609.20646): frozen pi0.5 action-expert correction from rewarded rollouts, with code. The practice-matched
  comparator.
- **DiMaS** (2607.14280): its "decodable but not steerable" result is the bar C3 must clear.
- **RV-ICL** (2610.06843): one demo, frozen VLA, LLM agent. Optional if code appears.

**Fairness:**
- One demo per new task in every arm; practice episodes charged when used.
- The same tuning sweep on dev cells for every arm.
- All arms on our own checkpoints; ReGuide's oracle-pose variant is reported as privileged.

## 5. Value proposition (honest)

**What only this system does:**
1. **It composes from the policy's own field, with no extra machinery.** ReGuide needs poses or a detector plus demo
   statistics; RV-ICL needs an LLM; GPC / PoCo need several policies.
2. **The composition rule is *extracted* from a VLA never trained to compose**, as per-role weights. GPC uses a scalar.
   Composable diffusion and Factored DP (2605.22596) use equal sums, and Factored DP must be trained for it.
3. **One demo, no gradient, and creation only of the missing part, in one loop.**
   - In-context methods (RICL, RA-VLA, Instant Policy) keep nothing.
   - Library methods (CLARE, DMPEL, MergeVLA, AtomicVLA) train per task from many demos and never solve an unseen
     combination before training.
   - ACT-VLA needs synthetic data plus SFT.
4. **Forgetting-free by construction, with growth proportional to new parts.**
5. **Decodable compositionality** (C3). No robot paper tests Hupkes' properties.

**Where competitors are stronger, and what we must show:**
- ReGuide reports 94.5 (oracle poses) and 87.5 (perception), and ACT-VLA 88-96, on pi0.5 compositional suites. Our earlier
  G0 recombination failed (365 vs 354). We must at least match ReGuide's perception variant without poses, or show where
  each wins: **we predict ReGuide wins on pure re-binding of position, and we win when a part's motion or gripper timing must
  change, and in continual reuse.**
- CLARE and DMPEL have tuned multi-demo continual numbers, and ER barely forgets. Our edge must be **cost per arrival and
  reuse**, not forgetting alone.
- VLAs under-use language (LIBERO-Plus, CRAFT, LangForce), so part-prompt fields may be weak. P1 tests this first. Training
  our own checkpoint on factorial cells may *increase* language reliance, which is also measured.

## 6. Decoding compositionality (C3)

The framework is Hupkes et al. 1908.08351. The measures come from Andreas (TRE, 1902.07181), Okawa et al. 2310.09336,
Uselis et al. 2602.24264, Lepori et al. 2301.10884 and Keysers et al. 1912.09713. Each property is tested at the
representation, relation and behaviour levels, against holistic controls (per-task LoRA, MLC-direct):

- **Systematicity.**
  - (X → Y) vs (Y → X) swaps.
  - Success flat in CD at fixed atom divergence (the CD bins in 3.1 / 3.2 exist for this).
- **Productivity.**
  - LIBERO-10 two-step tasks composed from single-step parts.
  - CALVIN chains longer than any trained span.
  - Success ≈ the product of part competences (Okawa's multiplicativity).
- **Substitutivity.** Synonymous fillers (instances of one type, paraphrased prompts) give the same composed field and the
  same success.
- **Localism.**
  - Swapping the goal filler leaves the object-role components unchanged.
  - R's cross-role terms ≈ 0.
  - Drop-one or swap-one changes only that role's competence (g or p).
- **Overgeneralisation.** On exception tasks (an unusual grasp or approach), NEW fires; otherwise the system over-regularises.
- **TRE, topographic similarity, role orthogonality, structural modularity** (a double dissociation between the object and
  goal roles).
- **Composition cards per task:** segments, roles, fillers (each with path, object and gripper timing), R weights, NEW flags.

## 7. Metrics and statistics

- **C1:**
  - success on new start states;
  - part correctness (g, the object grasped; p, placed given the grasp; on CALVIN, block and direction or place);
  - success vs CD bin;
  - GPU-seconds from demo arrival to ready, and gradient steps (0 unless NEW);
  - bytes.
- **C2:** FWT / NBT / AUC, the cost per arrival against its index, causal reuse, and bytes vs parts.
- **C3:** section 6.
- **Statistics:**
  - paired episodes, sign tests, task-cluster bootstrap CIs;
  - the CD × arm interaction by mixed-effects logistic regression;
  - 20 episodes per task in development, 50 in confirmatory reads;
  - no per-bin claim from fewer than 6 tasks;
  - Holm over at most 6 pre-listed confirmatory hypotheses;
  - sealed sets read once.

## 8. Probes, cheapest decisive first (all on SLURM)

| | question | go / kill | A40-h |
|---|---|---|---|
| P0 | Build LIBERO-90-C and CALVIN-C splits (MCD search), commit them + the sealed third; contamination audit of pi05_base | splits meet the atom-divergence and bin targets; audit clean | 1-2 |
| P1 | Train pi05_l90c (and FLOWER-C); parity on training cells (≥ the 40-task checkpoint's per-task level) | parity reached | 40-100 (+30-60, +50-80 CALVIN) |
| P2 | Fillers: are action modes under part prompts distinct, interpretable and transferable? (on our checkpoints) | separation > 2 at u ≤ 0.35 on ≥ 60% of observations; attribution ≥ 2× chance; transfer correlation ≥ 0.5 | 4-6 |
| P3 | Relation extraction: R0 on leave-cell-out training cells; stability; R1 / R2 only if needed; AND / OR / THEN readout | R0 explains ≥ 70% of the instruction-vs-null gap; template stability ≥ 0.8 | 8-12 |
| P4 | Offline decoding suite + offline pipeline vs equal-weight superposition and Comp-1/3 | beats both (CI excludes 0) | 3-5 |
| P5 | Closed loop on dev combination cells (CRAFT cell first) | beats Comp-3 and equal superposition; raises g; flat-ish in CD | 10-15 |
| P6 | Baseline setup: ReGuide (two versions), ACT-VLA, MergeVLA-style, CLARE, DMPEL, ER; parity smokes | each runs and reproduces its own reported trend | 30-50 |
| P7 | Streams (C2) and confirmatory reads (C1), then the sealed sets once, then ablations and controls | pre-registered | about 200 |

P0-P3 decide the line. Training our checkpoints dominates the cost; the probes P2-P4 are about 15-25 A40-h.

