# Design review: is the experimental design valid, and what a compositional fine-tuning framework needs

Written 2026-10-21 by a reviewer pass. It covers only reading and web research: no job was run. Every arXiv ID below was checked
against the arXiv API on 2026-10-21 (title and first author match). Repository evidence is cited by file and ledger name. The
numbers come from `paper/DRAFT.md`, `docs/RESULTS.md`, `docs/COST.md`, `docs/INTERPRETATION.md` and
`results/robot/split_grades.json`.

## 0 Bottom line

- The **statistical machinery** is stronger than the field norm. It has paired episodes, fresh start states, task-level
  bootstrap, exact sign tests, a pre-registration ledger and audited numbers. PhAIL (arXiv 2605.29710) surveys 13 VLA papers and
  finds that none reports confidence intervals or paired tests. Most of the threats below concern **what is compared and on
  which tasks**, not how the comparison is computed.
- The four most serious threats:
  1. **The grade is confounded with the benchmark.** Every G1/G2 "reuse wins" number comes from Li's 20 tasks in training
     scenes. Every G3/G4 "creation needed" number comes from LIBERO-90 scenes. LIBERO-90's own G2 tasks contradict the rule.
     The grades were also defined after the results existed.
  2. **"Zero budget" reuse at G1/G2 spends 340 rewarded practice episodes per task in the test environment** (`docs/COST.md`,
     G15 row). The comparators spend 0 (the policy, Li's TLI) or 100-120 (DSRL). This is search with reward, and it needs
     practice-matched baselines.
  3. **The 20 G1/G2 tasks were reused across 14 design iterations.** Fresh start states protect against state-level
     overfitting. They do not protect against task-level forking paths, and the task-level CIs assume the tasks are a sample.
  4. **The mechanism at G1/G2 may be trajectory or position selection, not concept composition.** Practice-chosen sources reach
     312 against 160 for word-matched (concept-correct) sources. On pi0.5, practice picks Li's semantically chosen pair on only
     4 of 20 tasks. LIBERO-PRO and LIBERO-Plus show that LIBERO policies ignore language and memorise layouts.
- **The cleanest compositional test in the repository fails.** G0 recombines concepts refit with no demonstration of the
  held-out fold: 365 against 354 for the policy, so the pre-registered H1 fails (`docs/RESULTS.md` row "DOES COMPOSITION
  GENERALISE?"). The G3/G4 headline (994 against 932) comes from task leaves, which are per-task adapters, chosen by practice
  from a larger candidate set. So "created units compose" is not shown.
- **Missing baselines a reviewer will ask for first:**
  - LLM-style adapter composition: LoraHub, Arrow routing, TIES/DARE merging, LoRA soups (CAT), PHATGOOSE, MoLE / LoRA-Flow
    gating, LoraRetriever;
  - robot-side merging and routing: MergeVLA, CLARE;
  - **practice-matched** baselines at G1/G2: reward-filtered self-imitation and DSRL at 340 episodes;
  - a language-free or random-source control.

## 1 Survey of experimental designs in this space

### 1a Compositional generalisation benchmarks and splits

| work | arXiv / venue | split design | what it implies for us |
|---|---|---|---|
| SCAN (Lake & Baroni) | [1711.00350](https://arxiv.org/abs/1711.00350), ICML 2018 | Held-out primitive ("jump" only in isolation), held-out length, held-out template combinations | The canonical *systematic* split: a primitive seen alone, tested in new combinations. Our G1 is the analogue. G4 (a never-seen word) is **not** a SCAN-style split. SCAN never tests unseen primitives. |
| COGS (Kim & Linzen) | [2010.05465](https://arxiv.org/abs/2010.05465), EMNLP 2020 | Separates **lexical** generalisation (a known word in a new role) from **structural** generalisation (new structures) | Our G2 ("binding" of a seen part to a new scene) is lexical-in-context. G1 is structural (a new pairing). G3/G4 are neither: they are distribution shift and new vocabulary. |
| CFQ / MCD (Keysers et al.) | [1912.09713](https://arxiv.org/abs/1912.09713), ICLR 2020 | Maximum compound divergence: atoms distributed alike in train and test, compounds maximally different; divergence is a *continuous* measure | The right way to *grade* a split: report atom divergence and compound divergence per task, rather than a hand-ordered 5-level label. G3/G4 have high **atom** divergence, which MCD deliberately keeps low. |
| Hupkes et al. "Compositionality decomposed" | [1908.08351](https://arxiv.org/abs/1908.08351), JAIR 2020 | Five tests: systematicity, productivity, substitutivity, localism, overgeneralisation | Our grades test systematicity (G1) and substitutivity-in-context (G2). Productivity (longer or more-step tasks) is untested. Localism maps onto the D1 donation analysis. |
| CompoSuite (Mendez et al.) | [2207.04136](https://arxiv.org/abs/2207.04136), CoLLAs 2022 | 4 axes (robot x object x obstacle x objective), 256 tasks; train on a subset, test on held-out *combinations*; also a "restricted" split where one element is seen only in one combination | The robotics template for a factorial held-out-combination split, with every atom seen. It also includes a restricted, few-exposure split, which is close to our G2. |
| Factor-World (Xie et al.) | [2307.03659](https://arxiv.org/abs/2307.03659), CoRL 2023 | Varies one factor at a time (lighting, texture, camera, position, distractors...) and measures the gap per factor | A design for attributing a gap to one factor. A G3 "scene" bundles many factors, and Factor-World shows they differ by an order of magnitude in difficulty. |
| KitchenShift (Xing et al.) | NeurIPS 2021 DistShift workshop ([OpenReview DdglKo8hBq0](https://openreview.net/forum?id=DdglKo8hBq0)); no arXiv | Zero-shot domain shifts (lighting, camera, texture, object pose) | Domain shift, not composition. Our G3 is closer to this than to SCAN. |
| THE COLOSSEUM (Pumacay et al.) | [2402.08191](https://arxiv.org/abs/2402.08191), RSS 2024 | 14 perturbation axes on 20 RLBench tasks | Perturbation robustness, as above. |
| GemBench (Garcia et al.) | [2410.01345](https://arxiv.org/abs/2410.01345), ICRA 2025 | 4 graded levels: L1 new placements, L2 new rigid objects, L3 new articulated objects, L4 long-horizon new compositions | The closest **graded** precedent. Its grades are ordered by type of novelty, as ours are. Like our G4, its L2/L3 are new-vocabulary levels, and GemBench calls them generalisation, not composition. |
| VLABench (Zhang et al.) | [2412.18194](https://arxiv.org/abs/2412.18194), ICCV 2025 | Language-conditioned long-horizon tasks with mesh/texture/semantic/commonsense dimensions | A broad capability benchmark, not a held-out combination split. |
| RoboCasa | [2406.02523](https://arxiv.org/abs/2406.02523), RSS 2024 | Atomic and composite tasks, held-out kitchen scenes | Composite tasks are built from atomic skills: a productivity axis we lack. |
| GenSim | [2310.01361](https://arxiv.org/abs/2310.01361), ICLR 2024 | LLM-generated tasks; held-out task generalisation | A route to many new tasks per grade (see Fix F3). |
| ManiSkill3 | [2410.00425](https://arxiv.org/abs/2410.00425) | GPU-parallel simulation | A cheap simulator for the practice-heavy arms. |
| LIBERO (Liu et al.) | [2306.03310](https://arxiv.org/abs/2306.03310), NeurIPS 2023 | Suites by knowledge type (spatial, object, goal, entangled); lifelong metrics FWT/NBT/AUC | Our training set is fixed by it. LIBERO-90 was designed as a *pretraining* set for LIBERO-10, so its tasks are not a designed compositional split. |
| LIBERO-PRO (Zhou et al.) | [2510.03827](https://arxiv.org/abs/2510.03827) | Task, position, object, semantic and environment perturbations of the training suites | Shows that LIBERO policies succeed by memorisation: they collapse to about 0 under a task change. We found its protocol feeds the old instruction (`docs/LIBERO_PRO.md`). Its task split is the closest independent G1 test we have. |
| LIBERO-Plus (Fei et al.) | [2510.13626](https://arxiv.org/abs/2510.13626) | 7 perturbation dimensions | "models tend to ignore language instructions completely"; camera shifts drop success from 95% to below 30%. This is a direct threat to any language-conditioned composition claim on LIBERO. |
| LIBERO-X | [2602.06556](https://arxiv.org/abs/2602.06556) | Hierarchical levels of progressive difficulty (spatial, object, instruction) | A graded protocol on LIBERO scenes. It is a candidate second LIBERO-family benchmark. |
| AGNOSTOS (Zhou et al.) | [2505.15660](https://arxiv.org/abs/2505.15660) | 23 unseen tasks against 18 seen tasks on LIBERO assets, zero-shot cross-task | The same scenes and assets as ours, with unseen tasks defined at task level. A direct comparator for the split. |
| Li, "VLAs are confined yet capable..." | [2505.03500](https://arxiv.org/abs/2505.03500) | 20 OOD tasks pairing one training task's object with another's goal in the base scene | Our entire G1/G2 evidence base. It is small (20 tasks) and all in training scenes. |
| "Robust Skills, Brittle Grounding" (Emukpere et al.) | [2602.24143](https://arxiv.org/abs/2602.24143) | Held-out object-location pairings that "break familiar associations without increasing spatial difficulty"; SmolVLA and pi0.5 | It argues that few-demo successes "may reflect object-location shortcuts". It recommends **decomposed metrics**: primitive execution versus instruction-conditioned success. Our first-grasp counts (Section 9) are this metric and should be reported per grade. |
| ReGuide, "Correcting WHERE, Preserving HOW" (Zhang et al.) | [2609.38616](https://arxiv.org/abs/2609.38616) | Compositional shifts in objects, destinations and backgrounds; a training-free wrapper on a frozen VLA | **A direct competitor**: frozen VLA, training-free, compositional shift. It finds that VLAs "fail at global grounding while retaining local manipulation skills", which is our failure mode (wrong first grasp). It must be cited and, if code exists, run. |
| RoboHiMan | [2510.13149](https://arxiv.org/abs/2510.13149) | Hierarchical evaluation of compositional generalisation in long-horizon manipulation | Productivity / long-horizon axis. |
| ATOM-Bench | [2606.16826](https://arxiv.org/abs/2606.16826) | Controls the atomic training distribution; tests held-out compositions on real robots | The design that removes the confound we have: **train the atoms yourself**, then hold out compositions. |
| RoboCombine (Zhang et al.) | [2610.15425](https://arxiv.org/abs/2610.15425) | Train on atomic skills only; test unseen ordered compositions; 40.0% on covered combinations against 7.7% on recombinations; failures at hand-off states | Same lesson as our grasp switch: transitions are where composition fails. It supports phase switching as the right locus. |
| ACG-Bench | [2610.06184](https://arxiv.org/abs/2610.06184) | Dual-arm, 6 in-domain conditions and 17 unseen compositions | Arm-wise composition, out of our scope. |
| Data-flywheel factorisation | [2603.25583](https://arxiv.org/abs/2603.25583) | High-dimensional factorisation and composition of data | Data-side composition, a related framing. |

**Methodological consensus.** A compositional split:

- (i) has every atom seen in training (SCAN, CompoSuite, MCD, ATOM-Bench);
- (ii) controls atom divergence while maximising compound divergence (MCD);
- (iii) holds the scene or environment fixed when it tests composition, and treats scene change as a separate robustness axis
  (Factor-World, COLOSSEUM, LIBERO-PRO);
- (iv) when the base model was trained elsewhere, either trains the atoms itself or audits what the base model saw (ATOM-Bench,
  RoboCombine).

Our G1 meets (i) and (iii). G2 meets (i) partly. G3 and G4 break (i) and (iii) by design. They measure robustness to new
scenes and new vocabulary, which is useful, but it is not compositional novelty.

### 1b Few-shot and one-shot adaptation protocols

| work | arXiv | protocol |
|---|---|---|
| TAIL (Liu et al.) | [2310.05905](https://arxiv.org/abs/2310.05905), ICLR 2024 | Per-task adapters on a frozen model across a LIBERO sequence; FWT reported. Our per-task LoRA is this family. |
| OpenVLA-OFT (Kim et al.) | [2502.19645](https://arxiv.org/abs/2502.19645) | Fine-tuning recipe on LIBERO with 50 demonstrations per task: the de-facto VLA fine-tuning baseline protocol. |
| Knowledge Insulation (Driess et al.) | [2505.23705](https://arxiv.org/abs/2505.23705) | pi0.5 on LIBERO-90 with all demonstrations (92.7 / 96.0): the all-demo ceiling cited in `docs/BASELINES.md`. |
| RoboCat (Bousmalis et al.) | [2306.11706](https://arxiv.org/abs/2306.11706) | Few-shot (100-1000 demos) adaptation plus self-generated data, the standard "self-improvement" protocol. It is the closest published analogue of practice: it fine-tunes on its own successful rollouts. |
| DemoDiffusion (Park et al.) | [2506.20668](https://arxiv.org/abs/2506.20668) | One human demonstration, no training, SDEdit-style. Already a baseline. |
| ViVLA (Chen et al.) | [2512.07582](https://arxiv.org/abs/2512.07582) | Holds out 2 of 10 tasks per LIBERO suite; one video demonstration at test. A LIBERO one-shot protocol with a held-out-task split. |
| HyperVLA | [2510.04898](https://arxiv.org/abs/2510.04898) | 50 demonstrations per task (OpenVLA setup). |
| STRAP (Memmel et al.) | [2412.15182](https://arxiv.org/abs/2412.15182), ICLR 2025 | Few-shot plus retrieval of sub-trajectories. Already a baseline. |
| DSRL (Wagenmaker et al.) | [2506.15799](https://arxiv.org/abs/2506.15799) | RL in the noise space of a frozen diffusion/flow policy. Already a baseline at 100 episodes. |
| pi_RL (Chen et al.) | [2510.25889](https://arxiv.org/abs/2510.25889); RLinf-VLA [2510.06710](https://arxiv.org/abs/2510.06710) | Few-shot SFT, then online RL on LIBERO with pi0 and pi0.5; budgets of thousands of episodes. |

**Consensus.** The field reports the number of demonstrations and, rarely, the number of rollouts. Almost no one charges
online practice episodes to a few-shot method. Our design charges them (`docs/COST.md`), which is better than the field. The
paper's text still says "zero budget" and "no demonstration" for arms that use 340 rewarded episodes. Practice-based selection
is a form of online RL, and RoboCat and pi_RL are its published comparators.

### 1c Continual and lifelong imitation protocols

| work | arXiv | protocol and metrics |
|---|---|---|
| LIBERO | [2306.03310](https://arxiv.org/abs/2306.03310) | FWT, NBT and AUC over a task sequence; ER, EWC and PackNet baselines; ResNet-T/ViT-T from scratch, 50 demos per task. |
| TAIL | [2310.05905](https://arxiv.org/abs/2310.05905) | Per-task adapters, so no forgetting by construction; FWT. |
| LOTUS (Wan et al.) | [2311.02058](https://arxiv.org/abs/2311.02058), ICRA 2024 | Unsupervised skill discovery and a growing library on LIBERO; FWT/NBT/AUC. |
| IsCiL (Lee et al.) | [2410.22658](https://arxiv.org/abs/2410.22658), NeurIPS 2024 | Retrievable skill prototypes and incremental adapters; Franka Kitchen and Meta-World. |
| M2Distill (Roy et al.) | [2410.00064](https://arxiv.org/abs/2410.00064) | Multi-modal distillation for lifelong imitation on LIBERO; FWT/NBT/AUC. |
| DMPEL (Lei et al.) | [2506.05985](https://arxiv.org/abs/2506.05985) | A progressive library of parameter-efficient experts with a dynamic mixture, for lifelong robot learning on LIBERO. **The closest published "compositional fine-tuning" method in robotics**, and not yet run. |
| Sparse Diffusion Policy (Wang et al.) | [2407.01531](https://arxiv.org/abs/2407.01531), CoRL 2024 | Mixture-of-experts diffusion policy with a router; multitask and continual. |
| CLARE (Römer et al.) | [2601.09512](https://arxiv.org/abs/2601.09512) | Adapter routing and expansion for VLA continual learning. Not rerun (DRAFT Section 15). |
| Liu et al. (forgetting) | [2603.03818](https://arxiv.org/abs/2603.03818) | Pretrained VLAs resist forgetting under continual learning. **This weakens the "no damage" contribution**: sequential fine-tuning of a pretrained VLA is a weaker strawman than LIBERO's from-scratch numbers suggest. Our own SeqFT (52/600) disagrees with this paper and should be checked for a recipe problem (learning rate, steps). |
| PhaseLoRA | [2608.15285](https://arxiv.org/abs/2608.15285) | A phase-conditioned action-expert LoRA: the nearest neighbour to grasp-switched units. |

**Consensus.** LIBERO-style protocols report FWT, NBT and AUC over a fixed sequence with multiple orders and seeds. We report
end-of-stream success and per-arrival damage. We should also report FWT/NBT/AUC so that the numbers can be placed against
LOTUS, M2Distill and DMPEL.

### 1d Compositional and modular fine-tuning in LLMs and diffusion

| work | arXiv | mechanism | role for us |
|---|---|---|---|
| Task arithmetic (Ilharco et al.) | [2212.04089](https://arxiv.org/abs/2212.04089), ICLR 2023 | Add or subtract fine-tuning deltas | Run (G2: summing collapses to 71/1,040). |
| TIES-Merging (Yadav et al.) | [2306.01708](https://arxiv.org/abs/2306.01708), NeurIPS 2023 | Trim, elect sign, disjoint merge | **Missing**. It fixes the interference that collapsed our task-arithmetic arm. |
| DARE (Yu et al.) | [2311.03099](https://arxiv.org/abs/2311.03099), ICML 2024 | Random drop and rescale of deltas before merging | **Missing** (cheap). |
| AdapterSoup (Chronopoulou et al.) | [2302.07027](https://arxiv.org/abs/2302.07027), EACL 2023 | Average the adapters of retrieved related domains | Close to our "LoRA averaging" arm (G2). |
| LoraHub (Huang et al.) | [2307.13269](https://arxiv.org/abs/2307.13269), COLM 2024 | Gradient-free (CMA-ES) search of LoRA mixing weights on a few examples | **Missing, and the most direct LLM analogue** of "one demonstration plus practice selects the composition". |
| LoRA Soups / CAT (Prabhakar et al.) | [2410.13025](https://arxiv.org/abs/2410.13025), COLING 2025 | Learned per-layer concatenation weights on few data; built for *skill composition* tasks | **Missing**. It is the LLM paper whose framing ("compose skills learned separately") is ours. |
| Arrow / modular LLMs (Ostapenko et al.) | [2405.11157](https://arxiv.org/abs/2405.11157), ICML 2024 | A library of LoRAs; zero-shot per-token routing by each LoRA's top singular vector | **Missing**. A zero-shot routing baseline that needs no labels, the analogue of our likelihood/practice source choice. |
| PHATGOOSE (Muqeeth et al.) | [2402.05859](https://arxiv.org/abs/2402.05859), ICML 2024 | Post-hoc learned gates per expert, zero-shot routing | Missing; an alternative to Arrow. |
| LoraRetriever (Zhao et al.) | [2402.09997](https://arxiv.org/abs/2402.09997) | Retrieve LoRAs by input embedding, then compose | Missing; the analogue of instruction-matched sources (our "word rule"). |
| Polytropon (Ponti et al.) | [2202.13914](https://arxiv.org/abs/2202.13914), EACL 2023 | A shared inventory of skill adapters with a learned task-skill allocation matrix | The multitask *training-time* version: train an inventory and a sparse allocation. A candidate design for "creating units that compose". |
| MHR (Caccia et al.) | [2211.03831](https://arxiv.org/abs/2211.03831), NeurIPS 2023 | Multi-head routing over adapters; few-shot adaptation of the router alone | Shows that adapting only the router is competitive. It is an ablation we should copy: fit only the composition weights on the one demo. |
| MoLE (Wu et al.) | [2404.13628](https://arxiv.org/abs/2404.13628), ICLR 2024 | Learned hierarchical gate over LoRA layers | Missing; the learned-gate counterpart of the grasp switch. |
| LoRA-Flow (Wang et al.) | [2402.11455](https://arxiv.org/abs/2402.11455) | Per-token dynamic fusion weights | As MoLE. |
| ZipLoRA (Shah et al.) | [2311.13600](https://arxiv.org/abs/2311.13600); K-LoRA (Ouyang et al.) [2502.18461](https://arxiv.org/abs/2502.18461) | Merge subject and style LoRAs in diffusion (ZipLoRA learned, K-LoRA training-free) | The diffusion side: composition of two concept LoRAs, as in our object x goal. K-LoRA's training-free top-k selection per layer is a cheap baseline. |
| Composable Diffusion (Liu et al.) | [2206.01714](https://arxiv.org/abs/2206.01714), ECCV 2022 | Sum of conditional scores (product of experts) | The theory behind "held fields compose". GPC and our convex mixing are its policy forms. |
| Wang et al. | [2605.07078](https://arxiv.org/abs/2605.07078) | Test-time concept discovery and composition in diffusion | Our stated lineage. |
| PoCo (Wang et al.) | [2402.02511](https://arxiv.org/abs/2402.02511), RSS 2024 | Compose diffusion policies across tasks, domains and behaviours at sampling | A published policy-composition baseline. Its every-step mixing is our GPC arm, which failed (61/400). |
| GPC (Cao et al.) | [2510.01068](https://arxiv.org/abs/2510.01068), ICLR 2026 | Test-time distribution-level composition of diffusion/flow policies | Run (G21). |
| MergeVLA (Fu et al.) | [2511.18810](https://arxiv.org/abs/2511.18810) | Cross-skill merging of VLA experts; finds naive merging gives near-zero success; task-masked sparse LoRA plus a test-time router; LIBERO and LIBERO-Plus | **Missing. The robot-side merging baseline**, and its negative result on naive merging matches our G2 collapse. |
| Robust fine-tuning via parameter merging (Yadav et al.) | [2512.08333](https://arxiv.org/abs/2512.08333) | Merge the fine-tuned and the pretrained VLA to keep generality | A preservation baseline for leaves (WiSE-FT-style interpolation). |

**Consensus in the LLM literature.** These papers evaluate composition:

- (1) on held-out *tasks*, with a library built only from other tasks (LoraHub, Arrow, PHATGOOSE: zero-shot or few-shot,
  leave-task-out);
- (2) against a **multitask model trained on the union of the library's data**, the upper reference Arrow and PHATGOOSE report;
- (3) at a fixed number of target examples.

Our G0 recombination test has the right shape: leave-fold-out library, plus a monolithic adapter on the same pool. It is the
experiment a modular-LLM reviewer would recognise, and it currently fails H1.

### 1e Evaluation methodology for robot policies

| work | arXiv | guidance |
|---|---|---|
| rliable (Agarwal et al.) | [2108.13264](https://arxiv.org/abs/2108.13264), NeurIPS 2021 | Stratified bootstrap CIs over tasks; IQM; performance profiles; probability of improvement; few-run regimes are unreliable |
| Henderson et al. | [1709.06560](https://arxiv.org/abs/1709.06560), AAAI 2018 | Seeds, hyperparameter tuning parity, reporting |
| Colas et al. | [1806.08295](https://arxiv.org/abs/1806.08295) | Power analysis for the number of seeds |
| Kress-Gazit et al. | [2409.09491](https://arxiv.org/abs/2409.09491) | Robot learning as an empirical science: report initial-condition distributions, blind A/B evaluation, failure-mode taxonomy, report all trials, define the evaluated population |
| Vincent et al. | [2405.05439](https://arxiv.org/abs/2405.05439), RSS 2024 | Statistical bounds on policy success from finite rollouts; generalisation claims only to the sampled condition distribution |
| STEP (Snyder et al.) | [2503.10966](https://arxiv.org/abs/2503.10966) | Near-optimal sequential test for comparing two policies; stops early; controls type I error |
| PhAIL (Arkhangelskiy) | [2605.29710](https://arxiv.org/abs/2605.29710) | Survey: modal N is 10-20 per condition with no CIs or paired tests; a 5 pp paired difference needs 600-1,500 paired rollouts per cell with binary outcomes |

**Our protocol against these.**

| guidance | ours | gap |
|---|---|---|
| Task-level stratified bootstrap (rliable) | yes, two-level | none |
| Paired comparisons | yes, same start state and noise | none |
| Pre-registration and type-I control | ledger with 132 pre-registrations; Holm sometimes | Holm is applied within a read, not across the 14 iterations or across five policies x grades |
| Population defined (Kress-Gazit, Vincent) | the start-state population is LIBERO's sampler | the **task** population is undefined: tasks are hand-curated (Li) or rule-filtered and reused |
| Episodes per task | 20 (10 at many budget points in PLAN_CONV) | per-task SE about 0.11 at p = 0.5; pooled reads are fine, per-grade cells with 2-6 tasks are not |
| Sequential testing (STEP) | not used | STEP would cut cost on the many reads that end far from the margin |
| Tuning parity (Henderson) | the method iterated 14 times; baselines stock | see threat T5 |

## 2 Validity threats

Severity: **H** means it can overturn a headline claim, **M** means it forces a qualification, **L** means a reporting fix.
Costs use the repository's own rate: one E3 read of one arm is 600 episodes, about 4.4 A40-h, so about 0.0073 A40-h per
episode (`docs/BASELINES.md`).

| # | threat | sev. | evidence in the repo | fix | cost (A40-h) |
|---|---|---|---|---|---|
| T1 | **The grade is confounded with the benchmark and the scene.** All reuse evidence at G1/G2 is from Li's suites (training scenes, no demos). All creation evidence at G3/G4 is from LIBERO-90 (new scenes). | H | `split_grades.json`: Li G1 6, G2 14; LIBERO-90 G0 15, G1 **3**, G2 6, G3 18, G4 48. DRAFT Section 10: "LIBERO-90's G2 tasks need creation too", so the rule fails inside one benchmark. | A factorial design within one benchmark (Section 3.2): scene familiar/new x part held/new x pairing held/new, about 6 tasks per cell. Pre-register the grade x arm interaction. | 150-250 (F3) |
| T2 | **The grades were assigned after the results.** "The grade predicts the regime" is a post-hoc stratification. | H | `docs/SPLIT.md` dates from the 2026-10-19 direction and stratifies existing reads; no pre-registration names the grades. | State it as descriptive now. Pre-register the prediction on the factorial tasks of T1. | in F3 |
| T3 | **G3/G4 measure vocabulary and scene novelty, not compositional novelty.** | M | G4 = "an object, goal relation or predicate never seen at all" (48 of 110 tasks); G3 = new scene. SCAN, COGS and MCD all keep atoms seen. | Rename: G1/G2 are compositional; G3 is "scene shift"; G4 is "new atom". Report MCD-style atom and compound divergence per task. Claim composition only where atoms are seen. | 0 |
| T4 | **The practice budget is not charged in the claims.** G1/G2 "reuse with nothing fitted" uses 340 rewarded test-environment episodes per task. | H | `docs/COST.md`: practice-chosen operator 340 episodes per task; grasp switch 170. Iteration 13: 6,800 episodes for 20 tasks. Comparators: policy 0, Li's TLI 0 (hand labels), DSRL 100-120. | Practice-matched baselines at 340 episodes: (a) reward-filtered self-imitation LoRA on the policy's own successes (RoboCat-style); (b) DSRL at 340; (c) random-source pairs with the same practice selection; (d) instruction-paraphrase search with the same practice. Plot success against practice episodes. | 110-160 (F2) |
| T5 | **Baseline tuning parity.** The system went through 14 iterations; the per-task LoRA baseline is a stock recipe. | H | M3-era rows (`docs/VALIDITY.md`): LoRA plus practice successes 314, round 3 334, plus failure negatives 342 of 460, against the stock 292. These practice-using LoRA variants beat the stock LoRA by up to 0.11 and are absent from E3/G8. G0's monolithic multitask LoRA falls **below** the frozen policy (190 against 354), which suggests an under-tuned baseline. | Give every baseline the same practice budget as the system (30 episodes in G8) and a small pre-registered sweep (learning rate x steps x rank) on 3 dev tasks excluded from reads. This is already in PLAN_CONV's fairness rules; apply it to the headline reads too. Retune the G0 monolithic LoRA until it beats the policy on its own training tasks. | 30-50 (F4) |
| T6 | **The system's candidate set is a superset of the baseline's.** Iteration 8 keeps {policy, composition, leaf}; the baseline keeps {policy, LoRA}. Leaves are per-task LoRAs. | M | `docs/COST.md`: hierarchy with leaves 30 practice episodes per task, against 20 for the LoRA baseline. 994 against 932, difference 0.047 [0.008, 0.091]. | Control: per-task LoRA with a second candidate (a rank-4 or second-seed LoRA) and acceptance among three, at 30 practice episodes. If it reaches 994, the gain is selection, not composition. | 15-20 (F5) |
| T7 | **Task reuse and forking paths.** The same 20 Li tasks were read in iterations 5-14, the same 29-30 E3 tasks in about 20 reads (set b: E3, G0, G2, G3, G5, G6, G7, G9, G11, W1, W2, G30...), and the same 66 tasks in G8/G19/G20. | H | Fresh start-state sets a-f exist (`results/robot/fresh_ood_*`, `fresh_states_*`), but every design choice was made after reading earlier sets *on the same tasks*. The task-level bootstrap treats the tasks as a random sample. | Seal a held-out task set now and never read it before the final confirmatory run. Candidates: new Li-style G1/G2 tasks authored in BDDL; AGNOSTOS's unseen tasks; LIBERO-PRO's task split (already read once in G22, with the system unchanged, so it counts as the strongest current check). Report every iteration's read in an appendix, with the number of comparisons. | in F3, plus 30 for a final read |
| T8 | **The mechanism at G1/G2 may be trajectory or position selection.** | H | Practice-chosen sources 312 against word-matched 160 (G20 ablation); "Li's pairs pick training tasks whose trained motion reaches where the new object lies" (iteration 5); practice picks Li's pair on 4/20 (pi0.5) and 10/20 (pi0) tasks; `docs/RESULTS.md` f1_binding: in living room 1 the policy "picks up the alphabet soup first in 10 of 10 episodes whichever object the instruction names". LIBERO-Plus: models "ignore language instructions completely". | Controls on Li's tasks and LIBERO-PRO: (a) sources chosen by practice, but the scored instruction blanked or garbled; (b) the best source pair under an *object-position-matched* rule with no concept match; (c) report the share of chosen sources whose object or goal concept matches the instruction; (d) relocate the target object to a position no source trajectory reaches (LIBERO-PRO's position split). If (b) matches the system, the claim is "trajectory reuse", not concept composition. | 20-30 (F6) |
| T9 | **The library leak in D1.** `lib` prototypes were fitted on demos that include the target task. | M | `docs/INTERPRETATION.md` Section 5: "the target task is among them (`obj_tasks` / `goal_tasks`)"; the flat one-demo curve confirms it. | Fit the leave-task-out `fold` library (PLAN_CONV already names it). Drop `lib` from every donation or speed claim. Keep it only as "served library". | 5-10 |
| T10 | **Demonstration accounting.** "One demonstration per task" is not equal across arms. | M | `docs/COST.md` demos per task: library 1.86 (E3), hierarchy 1.15 (G8), W1 0.57, per-task LoRA 1. Concept prototypes pool demos of *other test tasks* in the same set. | Report demos per task in every main table, and compare against a multitask LoRA on the same pooled demos (G0-style) at matched tuning. Or restrict prototypes to demos of earlier stream arrivals only. | in F4 |
| T11 | **20 episodes per task, and tiny cells by grade.** | M | Z3 grades: G4 has 2 tasks, G2 5, G1 0. G23 GR00T: 157 against 113, a small effect on 20 tasks. Per-task 95% CI half-width about 0.22 at 20 episodes. | Confirmatory reads at 50 episodes per task (LIBERO convention), or STEP sequential tests. At least 8 tasks per reported grade cell. No per-grade claims on cells with fewer than 6 tasks. | x2.5 on the final reads, about 60 |
| T12 | **Pretraining contamination.** | L-M | pi0.5: I6 shows LIBERO-90 concepts have 3-10x the flow loss of training concepts (held at most 0.0082, new at least 0.026; AUC 1.0), so LIBERO-90 was not memorised. For X-VLA, GR00T N1.5 and SmolVLA no audit exists, and public model cards do not list the base pretraining mix (openpi's README says only "10k+ hours of robot data"). X-VLA's zero-shot success is very high (534/600 on E3 tasks), which is a warning sign. Some published VLAs pretrain on LIBERO-90 itself (`docs/BASELINES.md`; search results). | Run I6's flow-loss audit on every policy: training demos against LIBERO-90 demos against Li's tasks. Record each checkpoint's documented training data in a table. Treat X-VLA results as G0-suspect until audited. | 4-8 (offline) |
| T13 | **The LIBERO shortcut also inflates zero-shot baselines.** Stored and fresh start states come from the same placement regions, so a layout-memorising policy is favoured. | M | LIBERO-PRO position split; "Robust Skills, Brittle Grounding". Our fresh states use LIBERO's own regions. | Add one perturbed-layout read (LIBERO-PRO `_swap`, or wider placement regions) for the final system and the main baselines. | 10-15 |
| T14 | **Endpoints converge, so the speed claim rests on early-curve gains.** | H for the speed claim | D1: about 35-100 fit steps saved, the same end error from every start; held fields are worth about one demonstration's fit, not extra demos; the 200-step fast fit failed (388 against 480); a cached 1,000-step fit costs 4.6 min, while 10 practice episodes cost about 9 A40-min. | Pre-register the speed claim on the GPU-seconds axis **including practice** (PLAN_CONV H3 does). Expect it to fail at G3/G4, because practice dominates. Recast it as "demonstration-free at G1/G2" and "storage- and forgetting-efficient at G3/G4". | 0 to re-scope; PLAN_CONV about 290 as written |
| T15 | **PLAN_CONV is infeasible as specified.** | M | It asks for "six or more per grade (G1, G2, G3, G4) from LIBERO-90", but LIBERO-90 has 3 G1 tasks. Li's G1/G2 tasks have no demonstrations, so the SFT and multi-demo arms cannot run there. | Author new G1/G2 tasks with demos (scripted or generated with the v2 retargeting generator, labelled as generated), or restrict H1 to "zero demo against RL" on Li's tasks. | in F3 |
| T16 | **The forgetting baselines may be strawmen.** | M | SeqFT 52/600 falls below the frozen policy, while Liu et al. ([2603.03818](https://arxiv.org/abs/2603.03818)) find that pretrained VLAs resist forgetting. TAIL-style per-task LoRA (which forgets nothing either) is the real comparator. | Report FWT/NBT/AUC; run the per-task LoRA stream and PackNet (already fitted, `docs/BASELINES.md`); check SeqFT's recipe (learning rate, steps per arrival) against Liu et al. | 15 |
| T17 | **Stale validity document.** | L | `docs/VALIDITY.md` is M3-era: 23 tasks, init states 30-49 shared by about 50 arms, and arms with **simulator poses** (an oracle). It is titled as the current validity record. | Regenerate it or mark it historical; never cite its oracle-pose arms as results. | 0 |
| T18 | **The novelty detector's AUC 1.0 is partly by construction.** | L | Held concepts are scored on the policy's **training** demonstrations, which it memorised; new ones on unseen demos (DRAFT iteration 6 says so). | Score held concepts on held-out demonstrations of training tasks (LIBERO ships 50 per task; the policy saw them all, so use LIBERO-PRO `_lan` or fresh-state rollouts of training tasks). | 2-4 (offline) |
| T19 | **Sim-only.** | M | LIBERO, Li, LIBERO-PRO; CALVIN, Meta-World and EgoDex are in progress (`docs/PLAN_X.md`). Practice needs a simulator or a real reset budget. | One non-LIBERO simulator with a designed split (CALVIN or Meta-World ML45, already planned) is the minimum for a venue. In the paper, state that practice costs real resets on hardware. | per PLAN_X |
| T20 | **Multiplicity across policies and iterations.** | L-M | Five policies x several hypotheses x 14 iterations. Holm is applied within reads only. | One family-wise correction over the confirmatory set listed in advance (e.g. 5 policies x 2 hypotheses). | 0 |
| T21 | **Li's operators were ported by us.** "Level with or above Li's TLI on all five" compares against our port. | M | SmolVLA TLI 0/400; GR00T TLI 61. Li published pi0 only. | State that the port was not validated beyond pi0 (it reproduces Li's 0.83 on pi0: 325/400). Drop "beats Li" for the other architectures, or validate the port on Li's own base tasks for each policy. | 5 |

### On "one demo plus practice is comparable across arms"

It is comparable only where every arm gets the same three things: the same demo, the same practice episodes and the same
acceptance rule. Today:

- The per-task LoRA gets acceptance (20 episodes). The system gets 30 (G8) or 340 (Li's tasks).
- The library pools 1.15-1.86 demos per task.
- STRAP, SeqFT and ER were read with and without acceptance in different places.

Practice is also used two ways. In the system it is a **search over discrete compositions with reward**. In the baselines it is
only a keep/reject filter. A baseline that also *learns* from practice (self-imitation, DSRL) is the right parity class. The
repository has these numbers from the M3 era, and they were strong.

## 3 Claim-by-claim verdicts

| # | claim (DRAFT) | verdict | reason |
|---|---|---|---|
| C1 | "OOD learning on a frozen policy is composition" (thesis) | **Not supported as a general claim** | The held-out recombination test (G0) fails H1: 365 against 354. In new scenes, held fields add nothing (Z3 233 against 245; Z5 253 against 245). The G3/G4 wins come from per-task leaves. Supported only as "in familiar scenes, recombining the policy's own conditionals per phase recovers new pairings". |
| C2 | Reuse wins at G1/G2 on five architectures (pi0 35 to 336, pi0.5 143 to 312, SmolVLA 5 to 96, X-VLA 80 to 256, GR00T 113 to 157) | **Valid as measured; needs qualification** | Paired, pre-registered, fresh states. It needs: "with 340 rewarded practice episodes per task"; "on 20 hand-built tasks reused across the design iterations"; the T8 controls, because the mechanism may be trajectory selection; GR00T's effect is small. |
| C3 | Level with or above Li's TLI with hand-chosen pairs on every policy | **Qualify** | pi0 non-inferior (difference 0.050 [-0.028, 0.142]); pi0.5 better. On SmolVLA, X-VLA and GR00T the comparator is our port (T21). The budgets differ: hand labels and 0 episodes against 340 episodes. |
| C4 | "The grade predicts whether a task needs reuse (G1, G2) or creation (G3, G4)" | **Not supported** | Post hoc (T2); confounded with benchmark (T1); contradicted by LIBERO-90 G2; G1 in LIBERO-90 is n = 3 and G4 in Z3 is n = 2. A defensible version now: "scene familiarity separates the regimes in our data". |
| C5 | Where the scene or a part is new, nothing reused helps and created units do; leaves beat per-task adapters (994 against 932) | **First half valid; second half needs qualification** | Z3/Z5 support "reuse does not help in new scenes". 994 against 932 passes its pre-registered criterion, but: leaves are per-task adapters; the candidate set is larger (T6); the practice budget is 30 against 20; demos 1.15 against 1. "Created units compose" is not shown (C1). |
| C6 | Created units compose, and the factorisation matters (G0 H2: 365 against 190 for the monolithic LoRA) | **Weak** | The monolithic LoRA falls below the frozen policy, which suggests a tuning problem (T5). H1 against the policy fails. |
| C7 | The stream damages nothing; SeqFT 52 and ER 335 against 465 | **Valid but expected by construction** | A frozen policy plus separate units cannot forget. The comparison that matters is against other modular learners (per-task LoRA stream, PackNet, LOTUS, DMPEL, CLARE). SeqFT's collapse conflicts with Liu et al. 2026 (T16). |
| C8 | Likelihood detects novelty (AUC 1.0) and names units but does not cut them | **Qualify** | The AUC uses training demos for held concepts (T18). "Does not cut them" and "does not select among new concepts" are clean negative results and valid. |
| C9 | LIBERO-PRO: the system, unchanged, 627 against 366 of 1,350; LIBERO-PRO's published protocol gives the old instruction | **Valid; the strongest current evidence** | New tasks and an unchanged system. It restricts to 27 of 40 eligible tasks (state it), and practice is still charged at 340 per task. The protocol finding (old instruction from the file name) is verified in code (`docs/LIBERO_PRO.md`) and is a contribution in itself. |
| C10 | The framework is architecture-agnostic (five policies) | **Valid with scope** | All five were fine-tuned on the same 40 LIBERO tasks. FLOWER's two tests narrowly fail. "No fixed operator carries across architectures" is honest and valid. |
| C11 | The unsupervised scheme serves pointer programs or a soft blend | **Not supported in closed loop** | Z3 H1 and Z5 H1 fail against the policy. Offline donation is real but small (blend +0.04 [-0.01, 0.08]). Present it as analysis, not as a working scheme. |
| C12 | Prototypes speed convergence (PLAN_CONV) | **Not supported yet; at risk** | Offline: about 35-100 steps saved, the same endpoint. Closed loop: the 200-step fit failed. On the GPU-seconds axis, practice dominates (T14). |
| C13 | What a held field donates follows behaviour, not concept names | **Valid (exploratory)**, and it is evidence for T8 | Donation shares: shared object 0.22, shared goal 0.18, shared nothing 0.13. This undercuts a *concept*-composition reading of the supervised scheme. |
| C14 | Data defect in the IPEC LIBERO-90 port | **Valid** | Fixed and documented; useful to the community. |

## 4 A compositional fine-tuning framework

### 4.1 Problem statement

- **Inputs.**
  - A frozen base policy pi_0 trained on a set of atoms.
  - A **unit library** L, which is a set of reusable parameter-space or conditioning-space modules (held conditionals, adapters).
  - A stream of new tasks T_1..T_n, each given with its instruction and scene and a **budget vector** B = (demos d, practice
    episodes p, gradient steps g, bytes s).
- **What the learner does.** For each T_k it chooses a composition c_k over L, which can be a routing, merging or phase
  schedule. Optionally it creates a new unit u_k from the budget and adds it to L.
- **Objectives**, each reported separately:
  - (i) success on T_k at budget B;
  - (ii) **compositional transfer**: success on *held-out* tasks built only from atoms in L, with **no** unit trained on them;
  - (iii) preservation on T_<k and on pi_0's own tasks;
  - (iv) library cost, which should grow with the number of atoms, not the number of tasks.
- **Claim form.** At matched B, compose-then-create beats create-only (per-task fine-tuning) and compose-only (merging or
  routing without creation). The advantage grows with |L|, which is forward transfer.

This keeps everything the repository has built: units, phases, practice and leaves. It also states composition as a falsifiable
property (ii), not a framing.

### 4.2 Protocol

**Splits.** Factorial, within one benchmark, with atoms controlled:

| axis | levels | how it is built in LIBERO |
|---|---|---|
| pairing (compound novelty) | held / new | Li-style object x goal recombinations, authored in BDDL in **both** training scenes and LIBERO-90 scenes |
| atom novelty | held / new object / new relation | add a LIBERO-90 object to a training scene (new atom, familiar scene); keep held atoms in a LIBERO-90 scene |
| scene | training / new | the same pairs placed in a training scene and in a LIBERO-90 kitchen |

- Use about 6-8 tasks per cell, 8 cells, so about 60 tasks.
- Grade each task by MCD atom and compound divergence against the 40 training tasks, in addition to G1-G4.
- Seal one third of the tasks as the final test, never read during development.
- Demonstrations for authored tasks: scripted or generated (the v2 retargeting generator), labelled as such. Keep the human
  LIBERO-90 demos for the existing tasks.

**Budgets.** Every arm is run on a grid and reported as curves, not points:

- demonstrations {0, 1, 5};
- practice episodes {0, 10, 50, 340};
- gradient steps {0, 100, 1,000};
- GPU-hours including practice.

The headline is the area under success against log(budget) and the budget to threshold (PLAN_CONV's Kaplan-Meier analysis is
right), per cell.

**Metrics.**

- Success, plus the decomposed metrics: first-grasp correctness (grounding) and execution given the correct grasp. This follows
  Emukpere et al. and is already measured in Section 9's trajectories.
- FWT/NBT/AUC for streams.
- Bytes.

**Statistics.**

- Keep the paired design, the two-level bootstrap and the sign tests.
- Add rliable IQM and the probability of improvement across tasks.
- 50 episodes per task, or STEP, at the confirmatory read.
- One family-wise correction over the pre-listed confirmatory hypotheses.

**Controls specific to composition.**

- Shuffled concept-to-unit assignment (does the *right* unit matter?).
- A random library unit.
- A language-blanked instruction.
- A position-matched non-concept source (T8).
- A leave-task-out library everywhere.

### 4.3 Required baselines

| family | baselines | status |
|---|---|---|
| create-only | per-task LoRA (TAIL family; rank 4 and 16), full SFT, multi-demo SFT, **LoRA plus practice self-imitation** (RoboCat-style) | first three run; the last is missing from current reads |
| compose-only, LLM-style | **LoraHub** (CMA-ES over library weights on the demo's flow loss, then on practice reward), **Arrow** (zero-shot SVD routing per layer), **TIES** and **DARE** merging of donor units, **LoRA soups/CAT** (learned per-layer weights on the demo), **PHATGOOSE**, **MoLE / LoRA-Flow** gate, **LoraRetriever** | task arithmetic and averaging run; the rest missing |
| compose-only, robot-side | Li's switch and TLI, GPC, PoCo-style mixing, activation injection, **ReGuide** (frozen wrapper), **MergeVLA** | the first four run |
| compose-then-create, published | **DMPEL**, SDP (adapted router), **CLARE**, LOTUS, IsCiL | none run (`docs/BASELINES.md` decisions pending) |
| RL | DSRL (at matched practice), pi_RL (LoRA on the action expert, matched GPU-h) | DSRL run at 100-120 episodes |
| upper references | multitask LoRA on the union of all demos (Arrow/PHATGOOSE's reference); all-demo SFT | run in G0/F1 but under-tuned |

**Units for the LLM-style baselines.**

- For held concepts (G1/G2), the library is the policy's own conditionals. The LLM analogue is soft-prompt or instruction
  composition, so LoraHub-style search over mixing weights of the 16 source conditionals is the direct baseline. It is cheap,
  because our blend machinery (`blend.py`) already mixes fields.
- For G3/G4, the library is leave-task-out per-task or per-concept LoRAs of the other LIBERO-90 tasks, which already exist.
  Arrow, TIES, DARE and CAT run on these with no new data.

### 4.4 What a convincing paper must show

1. On sealed held-out recombinations, a leave-task-out composed library beats **both** the frozen policy and a *tuned* multitask
   adapter on the same pooled data, at the same budget. This is G0 redone with tuning parity. It is the one result a
   modular-LLM reviewer looks for first, and it currently fails.
2. The advantage depends on the factorial cell in a pre-registered way (grade x arm interaction), within one benchmark.
3. The shuffled-assignment and position-matched controls fall well below the system, so the *concept* identity of a unit matters.
4. Budget curves, not points, with practice charged: the system dominates create-only arms at small budgets and stays level at
   large ones.
5. Forward transfer: the budget needed per arrival falls with library size (PLAN_CONV H4).
6. LoraHub, Arrow and TIES/DARE, which are the LLM field's answers, do worse, and the paper explains why: phase gating. The G20
   ablation (the switch at the grasp matters as much as either unit) is the mechanism.
7. One second benchmark with a designed split (CALVIN or Meta-World ML45) and one perturbed-layout read.

### 4.5 Mapping existing pieces onto the framework

| framework element | existing piece | gap |
|---|---|---|
| unit library L | pointers (zero-byte held conditionals); rank-4 concept prototypes; instance children; leaves | the library must be leave-task-out for transfer claims (`fold`) |
| composition operator c | grasp switch (a phase router); TLI; convex mix (GPC); soft blend (per-segment, per-dim weights) | add the learned gate (MoLE) and SVD routing (Arrow) as alternative operators |
| composition search | practice choosing sources and operators (a reward-based LoraHub); likelihood choosing sources (Z3) | a LoraHub-style search on *demo loss*, with no practice, separates "needs reward" from "needs a demo" |
| create | leaves; refit concepts (W1) | a Polytropon/MHR-style shared inventory trained to compose, not per-task |
| novelty / create decision | flow-loss detector (I6); practice gate (W1, iteration 14) | validate the detector on held-out demos (T18) |
| stream and preservation | W1, W2, iterations 13/14, G11 | report FWT/NBT/AUC; per-task LoRA stream as the modular comparator |
| donation analysis | D1 breakdown, causal swaps | redo on `fold`; tie it to closed-loop swaps |
| budget accounting | `docs/COST.md`, `experiments/h2_cost.py` | move the budget vector into every main table and every claim sentence |
| split | `experiments/split_grades.py` | factorial authored tasks; MCD divergences; a sealed test set |
| statistics | `lifelong_ood/stats.py`, ledger, audit | IQM and probability of improvement; STEP; family-wise correction |

## 5 Prioritised fixes

| priority | fix | addresses | cost (A40-h) |
|---|---|---|---|
| 1 | **Re-scope the text now**: charge practice in every claim; rename G3/G4; call the grade table descriptive; mark `docs/VALIDITY.md` historical; drop `lib` from donation claims; state which comparisons use our port of Li | T2, T3, T4 (text), T9, T17, T21, C4, C11 | 0 |
| 2 | **F6 mechanism controls on Li's 20 tasks and LIBERO-PRO**: blanked instruction; position-matched non-concept source; share of concept-matching chosen sources (offline, from the ledgers); a relocated target object | T8, C2, C13 | 20-30 |
| 3 | **F4 parity on the headline reads**: LoRA plus practice self-imitation at the system's practice budget; a pre-registered small sweep for the per-task and multitask LoRA; G0 redone with a tuned multitask LoRA | T5, T10, C5, C6 | 30-50 |
| 4 | **LLM-style composition baselines on existing units**: TIES, DARE, Arrow (zero fit), CAT and LoraHub (demo-loss search) on leave-task-out LoRAs (G3/G4, set d), and LoraHub over the 16 held conditionals (Li's tasks, set e) | missing baselines, Section 4.3 | 40-60 (reads 6 arms x about 1,300 episodes; fits under 5) |
| 5 | **F5 candidate-set control**: per-task LoRA with three candidates and acceptance at 30 episodes | T6, C5 | 15-20 |
| 6 | **F2 practice-matched RL and search at G1/G2**: DSRL at 340 episodes; random-pair plus practice; paraphrase search; on pi0.5 and pi0 | T4, C2, C3 | 110-160 |
| 7 | **Contamination and detector audits (offline)**: flow loss of every policy on training, LIBERO-90 and Li demos; held concepts on held-out frames; a table of each checkpoint's training data | T12, T18, C8 | 6-12 |
| 8 | **F3 factorial authored split** with a sealed third (about 60 tasks, generated demos); pre-register grade x arm; this also makes PLAN_CONV feasible | T1, T2, T7, T15, C4 | 150-250 |
| 9 | **Confirmatory reads at 50 episodes per task, or STEP**, on the sealed set, with one family-wise correction | T11, T20 | about 60 |
| 10 | **Published modular learners**: DMPEL or CLARE (whichever has code), the adapted SDP router, the PackNet stream (fitted), and FWT/NBT/AUC | T16, C7 | 40-80 |
| 11 | **One perturbed-layout read** (LIBERO-PRO `_swap`) for the system and the main baselines | T13 | 10-15 |
| 12 | **PLAN_CONV, re-scoped** after fixes 3, 8 and 9: the GPU-seconds axis with practice charged; the zero-demo hypothesis tested against RL, not SFT | T14, T15, C12 | about 200 (down from about 290) |

Fixes 1-5 and 7 cost about 120-180 A40-h and settle most of the H-severity threats on the existing tasks. Fix 8 is the one that
can make "the grade predicts the regime" a result rather than a reading.

## 6 Verified references (arXiv API, 2026-10-21)

- Benchmarks and splits:
  - Lake & Baroni [1711.00350](https://arxiv.org/abs/1711.00350)
  - Kim & Linzen [2010.05465](https://arxiv.org/abs/2010.05465)
  - Keysers et al. [1912.09713](https://arxiv.org/abs/1912.09713)
  - Hupkes et al. [1908.08351](https://arxiv.org/abs/1908.08351)
  - CompoSuite [2207.04136](https://arxiv.org/abs/2207.04136)
  - Factor-World [2307.03659](https://arxiv.org/abs/2307.03659)
  - KitchenShift ([OpenReview](https://openreview.net/forum?id=DdglKo8hBq0), NeurIPS 2021 workshop; no arXiv)
  - COLOSSEUM [2402.08191](https://arxiv.org/abs/2402.08191)
  - GemBench [2410.01345](https://arxiv.org/abs/2410.01345)
  - VLABench [2412.18194](https://arxiv.org/abs/2412.18194)
  - RoboCasa [2406.02523](https://arxiv.org/abs/2406.02523)
  - GenSim [2310.01361](https://arxiv.org/abs/2310.01361)
  - ManiSkill3 [2410.00425](https://arxiv.org/abs/2410.00425)
  - LIBERO [2306.03310](https://arxiv.org/abs/2306.03310)
  - LIBERO-PRO [2510.03827](https://arxiv.org/abs/2510.03827)
  - LIBERO-Plus [2510.13626](https://arxiv.org/abs/2510.13626)
  - LIBERO-X [2602.06556](https://arxiv.org/abs/2602.06556)
  - AGNOSTOS [2505.15660](https://arxiv.org/abs/2505.15660)
  - Li [2505.03500](https://arxiv.org/abs/2505.03500)
  - Emukpere et al. [2602.24143](https://arxiv.org/abs/2602.24143)
  - ReGuide [2609.38616](https://arxiv.org/abs/2609.38616)
  - RoboHiMan [2510.13149](https://arxiv.org/abs/2510.13149)
  - ATOM-Bench [2606.16826](https://arxiv.org/abs/2606.16826)
  - RoboCombine [2610.15425](https://arxiv.org/abs/2610.15425)
  - ACG-Bench [2610.06184](https://arxiv.org/abs/2610.06184)
  - Data flywheel [2603.25583](https://arxiv.org/abs/2603.25583)
- Policies:
  - pi0 [2410.24164](https://arxiv.org/abs/2410.24164)
  - pi0.5 [2504.16054](https://arxiv.org/abs/2504.16054)
  - Knowledge Insulation [2505.23705](https://arxiv.org/abs/2505.23705)
  - SmolVLA [2506.01844](https://arxiv.org/abs/2506.01844)
  - GR00T N1 [2503.14734](https://arxiv.org/abs/2503.14734)
  - X-VLA [2510.10274](https://arxiv.org/abs/2510.10274)
  - FLOWER [2509.04996](https://arxiv.org/abs/2509.04996)
- Few-shot and RL:
  - TAIL [2310.05905](https://arxiv.org/abs/2310.05905)
  - OpenVLA-OFT [2502.19645](https://arxiv.org/abs/2502.19645)
  - RoboCat [2306.11706](https://arxiv.org/abs/2306.11706)
  - DemoDiffusion [2506.20668](https://arxiv.org/abs/2506.20668)
  - ViVLA [2512.07582](https://arxiv.org/abs/2512.07582)
  - HyperVLA [2510.04898](https://arxiv.org/abs/2510.04898)
  - STRAP [2412.15182](https://arxiv.org/abs/2412.15182)
  - DSRL [2506.15799](https://arxiv.org/abs/2506.15799)
  - pi_RL [2510.25889](https://arxiv.org/abs/2510.25889)
  - RLinf-VLA [2510.06710](https://arxiv.org/abs/2510.06710)
- Continual:
  - LOTUS [2311.02058](https://arxiv.org/abs/2311.02058)
  - IsCiL [2410.22658](https://arxiv.org/abs/2410.22658)
  - M2Distill [2410.00064](https://arxiv.org/abs/2410.00064)
  - DMPEL [2506.05985](https://arxiv.org/abs/2506.05985)
  - SDP [2407.01531](https://arxiv.org/abs/2407.01531)
  - CLARE [2601.09512](https://arxiv.org/abs/2601.09512)
  - Liu et al. (forgetting) [2603.03818](https://arxiv.org/abs/2603.03818)
  - PhaseLoRA [2608.15285](https://arxiv.org/abs/2608.15285)
- Composition and merging:
  - Task arithmetic [2212.04089](https://arxiv.org/abs/2212.04089)
  - TIES [2306.01708](https://arxiv.org/abs/2306.01708)
  - DARE [2311.03099](https://arxiv.org/abs/2311.03099)
  - AdapterSoup [2302.07027](https://arxiv.org/abs/2302.07027)
  - LoraHub [2307.13269](https://arxiv.org/abs/2307.13269)
  - LoRA Soups [2410.13025](https://arxiv.org/abs/2410.13025)
  - Arrow [2405.11157](https://arxiv.org/abs/2405.11157)
  - PHATGOOSE [2402.05859](https://arxiv.org/abs/2402.05859)
  - LoraRetriever [2402.09997](https://arxiv.org/abs/2402.09997)
  - Polytropon [2202.13914](https://arxiv.org/abs/2202.13914)
  - MHR [2211.03831](https://arxiv.org/abs/2211.03831)
  - MoLE [2404.13628](https://arxiv.org/abs/2404.13628)
  - LoRA-Flow [2402.11455](https://arxiv.org/abs/2402.11455)
  - ZipLoRA [2311.13600](https://arxiv.org/abs/2311.13600)
  - K-LoRA [2502.18461](https://arxiv.org/abs/2502.18461)
  - Composable Diffusion [2206.01714](https://arxiv.org/abs/2206.01714)
  - Wang et al. [2605.07078](https://arxiv.org/abs/2605.07078)
  - PoCo [2402.02511](https://arxiv.org/abs/2402.02511)
  - GPC [2510.01068](https://arxiv.org/abs/2510.01068)
  - MergeVLA [2511.18810](https://arxiv.org/abs/2511.18810)
  - VLA parameter merging [2512.08333](https://arxiv.org/abs/2512.08333)
- Evaluation:
  - rliable [2108.13264](https://arxiv.org/abs/2108.13264)
  - Henderson et al. [1709.06560](https://arxiv.org/abs/1709.06560)
  - Colas et al. [1806.08295](https://arxiv.org/abs/1806.08295)
  - Kress-Gazit et al. [2409.09491](https://arxiv.org/abs/2409.09491)
  - Vincent et al. [2405.05439](https://arxiv.org/abs/2405.05439)
  - STEP [2503.10966](https://arxiv.org/abs/2503.10966)
  - PhAIL [2605.29710](https://arxiv.org/abs/2605.29710)

**Not verified.**

- Venue tags (ICLR, CoRL and so on) are taken from the papers' own pages or the existing drafts. Only arXiv titles and authors
  were checked here.
- The pretraining mixtures of pi0.5-base, X-VLA, GR00T N1.5 and SmolVLA: the public READMEs and model cards fetched do not list
  LIBERO-90 either way.
- Whether ReGuide, DMPEL and MergeVLA release runnable code.
