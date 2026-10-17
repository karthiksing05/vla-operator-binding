# Novelty audit (2026-10-17)

Scope: the contributions in `paper/DRAFT.md` (Section 1) and `docs/METHOD.md`, numbered here as in the audit request:

1. object / instance / goal concept hierarchy; each concept a rank-4 action-expert LoRA "prototype" fitted on its PHASE frames
   pooled over every demonstration of the concept;
2. composition by switching units at the policy's own grasp (object unit before, goal unit after);
3. held concepts as zero-byte pointers: the policy's own conditional field under the training instruction that taught the
   concept, switched per phase (lifts pi0.5, pi0, SmolVLA on Li's OOD suites);
4. the frozen policy's flow-matching likelihood as a new-concept detector (AUC 1.0) that picks pointer vs prototype; likelihood
   SELECTION among concepts fails;
5. the continual learner: a demonstration only when 10 practice episodes fail, Cobweb-style refit of every concept on the
   demonstration's path, no damage, about 1/5 the storage of per-task LoRAs at parity;
6. pre-registered paired protocol on fresh start states; the IPEC LIBERO-90 LeRobot port data defect.

Every entry below was found in a web or arXiv search on 2026-10-17 and its abstract page (or proceedings / OpenReview page) was
opened. Two NeurIPS 2026 papers (PRISM, CARVE) have only a NeurIPS / OpenReview page and no arXiv id that we could find. Where a
detail came only from the abstract it is marked "(abstract)".

## 1 Prior work table

| prior work | year / venue | arXiv | what it does | overlaps | how we differ |
|---|---|---|---|---|---|
| **Li, "VLAs are Confined yet Capable of Generalizing to Novel Instructions"** (IJCAI title: "...Novel Tasks"), Q. Li | IJCAI 2026 (pp. 7583-7591) | [2505.03500](https://arxiv.org/abs/2505.03500) | Introduces libero-goal-ood / libero-spatial-ood (20 tasks). Text Latent Interpolation (TLI): average a base task's text-token hidden states over its demos; for an OOD task built from two base tasks, inject the latent difference with a schedule alpha = i/lambda, so task 1's behaviour fades into task 2's. pi0: 0.09 -> 0.83. Also a **prompt-switching baseline pi0^S: task 1's prompt, then task 2's prompt after step lambda/2: 0.72 Goal-OOD, 0.66 Spatial-OOD, 0.69 overall** (10 runs per task). pi0 only (pi0-fast partial). | **3 (strongly), 2** | Ours switches at the policy's own grasp, not at a fixed step; picks the source tasks automatically from concepts read off the instruction (Li hand-pairs base tasks); gates by practice; runs on three policies. **But on the same suites and the same policy our pointer composition reaches 98/400 (0.245) on pi0, against Li's 0.69 (fixed-step prompt switch) and 0.83 (TLI).** Our pi0 baseline (35/400 = 0.088) matches Li's 0.09, so a reviewer will read the numbers as comparable. This is the most dangerous prior work. |
| **ACT-VLA, "Unleashing More Actions via Action Compositional Training for VLA Models"**, Peng, Lu, Peng | arXiv 2026-07 | [2607.00351](https://arxiv.org/abs/2607.00351) | Synthesises compositional demos for the OOD tasks with TLI rollouts, keeps successful ones (about 300-400 per suite), retrains the VLA by BC. pi0.5: Spatial-OOD 35.5 -> 88.2, Goal-OOD 46.6 -> 95.6; pi0 ablation 2.0/16.0 -> 85.9/90.5; compares against pi0+TLI. | 3, 5 | Retrains the whole policy on synthesised test-task data; ours keeps the policy frozen and stores zero bytes for held concepts. Its pi0.5 baseline (35.5 / 46.6) matches ours (147/400), so our 174/400 sits far below its 88-96%. Must be cited as the training-based upper reference. |
| **AC-VLA, "Robust Out-of-Distribution Action Execution via Compositional Learning"**, Peng et al. | arXiv 2026-07 | [2607.15714](https://arxiv.org/abs/2607.15714) | On pi0.5, LLM instruction decomposition + proprioceptive aligner that **cuts sub-tasks at gripper open/close transitions**; trains on full + decomposed demos; masks the wrist view in closed-gripper phases. Spatial-OOD 35.5 -> 64.2, Goal-OOD 46.6 -> 73.3 on Li's suites. | 2, 3 | Uses gripper events to segment training data and condition a retrained policy; ours uses the grasp at inference to switch frozen-policy units. Anticipates "the grasp is the natural phase boundary" on the same benchmark and policy. |
| **PRISM, "Primitive Routing via In-context Skill Mixing for Lifelong VLA"**, Tao, Zhang, Wang, Liu, Yan | NeurIPS 2026 ([OpenReview DwiSjXGZ79](https://openreview.net/forum?id=DwiSjXGZ79)) | none found | Each skill family is a frozen low-rank **action-expert adapter** in a primitive bank; a per-chunk router (trained on action-derived oracle labels) composes them for long-horizon tasks; compared with a pi0.5-LoRA two-stage fine-tune on RoboCasa365. | **1, 2, 5** | Its units are skill families (verbs), routed by a learned router per chunk; ours are object / instance / goal concepts read from the instruction, switched by the grasp, no router training, plus a held-concept pointer and a likelihood detector. Closest "LoRA bank on the action expert, composed per segment" paper. |
| **CLARE, "Continual Learning for VLA Models via Autonomous Adapter Routing and Expansion"**, Roemer, Zhang, Li, Schoellig | RA-L 2026 | [2601.09512](https://arxiv.org/abs/2601.09512) | Flow-matching DiT policy pretrained on LIBERO-90; per new task adds adapters **only in layers where an autoencoder discriminator's reconstruction-error z-score exceeds a threshold for every earlier discriminator** (expand vs link to existing); AE router at deployment, no task id; exemplar-free; about 2% parameters per task. LIBERO-Long/Goal/Spatial streams. | **4, 5**, 1 | The closest "detect novelty, then expand or reuse" mechanism in robot CL. Its novelty signal is a separate autoencoder on features; ours is the frozen policy's own flow-matching loss, and our "reuse" branch is a zero-byte pointer to the policy's own field, not a shared adapter. Unit = task (per layer), not concept. |
| **DMPEL, "Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning"**, Lei, Mao, Zhou, Zhang, Li, Luo | TMLR 2026 | [2506.05985](https://arxiv.org/abs/2506.05985) | Progressively grows a low-rank expert library; a light router mixes experts into one policy; coefficient replay against forgetting; no test-time task id; LIBERO; minimal parameters and storage. | 1, 5 | Experts are mixed per input by a learned router; ours are concept units chosen by instruction parse and grasp. A direct storage-vs-success competitor on LIBERO. |
| **TAIL, "Task-specific Adapters for Imitation Learning with Large Pretrained Models"**, Liu, Zhang, Asadi, Liu, Zhao, Sabach, Fakoor | ICLR 2024 | [2310.05905](https://arxiv.org/abs/2310.05905) | Per-task adapters (LoRA best, about 1% of parameters) on a frozen pretrained policy (pretrained on 40 LIBERO-90 Kitchen tasks), continual adaptation without forgetting. | 5, 1 | Our strong baseline (rank-16 LoRA per task, acceptance-gated) **is** TAIL-LoRA; say so by name. Ours shares units across tasks; TAIL needs a task id and one adapter per task. |
| **CORAL, "Scalable Multi-Task Robot Learning via LoRA Experts"**, Luo, Chen, Liang, Li | arXiv 2026-03 | [2603.09298](https://arxiv.org/abs/2603.09298) | Frozen VLA + one LoRA expert per task in an expandable library; the language instruction routes to the expert; sequential addition without forgetting; LIBERO, WidowX, Google Robot, real robot. | 5, 1 | Same as TAIL with instruction routing; per task, no sharing or composition. |
| **LOTUS, "Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery"**, Wan, Zhu, Shah, Zhu | ICRA 2024 | [2311.02058](https://arxiv.org/abs/2311.02058) | Segments demos into skills with open-vocabulary vision features, **updates existing skills and adds new ones** as tasks arrive, a meta-controller composes skills; LIBERO. | **5**, 1, 4 | Anticipates "a new demo updates the skills it touches and adds new ones" (our Cobweb-style refit). Its skills are discovered by visual clustering, not read from the instruction; trained from scratch, not a frozen VLA; no pointer units. |
| **IsCiL, "Incremental Learning of Retrievable Skills For Efficient Continual Task Adaptation"**, Lee, Yoo, Kim, Choi, Woo | NeurIPS 2024 | [2410.22658](https://arxiv.org/abs/2410.22658) | Adapter per skill, **prototype-based memory** retrieves the skill adapter from state embeddings; continual imitation; Franka-Kitchen, Meta-World. | 1, 5 | Their "prototype" is a retrieval key in state space; ours is the LoRA itself. Name clash: define our term explicitly. |
| **Sparse Diffusion Policy (SDP)**, Wang, Zhang, Huo et al. | CoRL 2024 | [2407.01531](https://arxiv.org/abs/2407.01531) | MoE inside a transformer diffusion policy; experts as skills, router as planner; adds experts for continual learning with little forgetting. | 1, 5 | Learned per-token routing inside a trained-from-scratch policy; ours is a frozen VLA with instruction-read concept units. |
| **Primitive Prompt Learning, "Think Small, Act Big"**, Yao, Liu et al. | CVPR 2025 | [2504.00420](https://arxiv.org/abs/2504.00420) | Pretrains primitive prompts shared across skills; new skills add prompts trained alongside frozen ones; motion-aware query selects prompts for a diffusion transformer policy. | 1, 5 | Prompts, not LoRA; selection by a learned query; no phase switching. |
| **M2Distill**, Roy, Dissanayake, Tidd, Moghadam | ICRA 2025 | [2410.00064](https://arxiv.org/abs/2410.00064) | Multi-modal latent distillation against forgetting; LIBERO-Object/Goal/Spatial. | 5 (baseline family) | Regularisation-based; no modules. Context only. |
| **LiMoDE**, Gu, Wang | arXiv 2026-06 | [2606.26183](https://arxiv.org/abs/2606.26183) | Dynamic MoE pretraining + lifelong MoE adaptation combining new and frozen experts; LIBERO (abstract). | 1, 5 | Learned routing; CLIP-based small backbone, not a flow VLA. |
| **SCE, "Skill-Compositional Experts for Embodied Continual Learning"**, Zhang, Wang, Tang, Wu, Nie | arXiv 2026-06 | [2606.15685](https://arxiv.org/abs/2606.15685) | Decomposes demos into a skill base; execution and transition experts compose skills for new tasks; appends an expert per newly discovered skill; LIBERO. | 1, 5 | Skill decomposition learned; no frozen VLA, no pointer, no likelihood detector (abstract). |
| **SPECI**, Xu, Nie | arXiv 2025-04 | [2504.15561](https://arxiv.org/abs/2504.15561) | Hierarchical continual imitation with an expandable skill codebook and attention-based skill selection. | 1, 5 | Soft attention over a codebook; no concept factorisation. |
| **T2S, "Tokenized Skill Scaling for Lifelong Imitation Learning"**, Zhang et al. | arXiv 2025-08 | [2508.01167](https://arxiv.org/abs/2508.01167) | New skills = new learnable tokens; language-guided sharing; LIBERO, NBT 1.0%. | 5 | Token growth, not concept units. |
| **AtomicVLA**, Zhang et al. | CVPR 2026 | [2603.07648](https://arxiv.org/abs/2603.07648) | Planner emits atomic-skill abstractions; skill-guided MoE with one expert per atomic skill; new experts for new skills (continual); beats pi0 on LIBERO / LIBERO-Long. | 1, 5 | Skills are verbs from a planner; trained experts; no frozen-policy pointer. |
| **Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts (Task-aware MoILE)**, Jia, Wang, Qu, Yang, Wang | ACL 2025 | [2506.04595](https://arxiv.org/abs/2506.04595) | Embodied CL split into high-level instruction and low-level action levels; task recognition by clustering visual-text embeddings; task- and token-level routers over incremental LoRA experts; SVD protection. | 1, 4, 5 | Novelty / task recognition by embedding clustering, not policy likelihood; embodied-agent, not a flow VLA on LIBERO. |
| **PhaseLoRA**, Guo, Wu, Duan, Ding, Han | arXiv 2026-08 | [2608.15285](https://arxiv.org/abs/2608.15285) | LoRA on the action expert whose left factor is modulated per action chunk by predicted "fine-control" and "event/boundary" descriptors; motivation: approach / grasp / transport / place phases; +12.2 on LIBERO over parameter-matched LoRA. | **1, 2** | Continuous modulation of one LoRA, learned descriptors, single-task fine-tuning; ours switches discrete concept units at the observed grasp. Anticipates "phase-specific LoRA on the action expert". |
| **PHASER**, Chen et al. | arXiv 2026-06 | [2606.03598](https://arxiv.org/abs/2606.03598) | Phase-aware experience replay for VLA continual learning; phases from action change points + VLM verification; three VLA backbones on LIBERO CL. | 2, 5 | Phases organise replay memory, not modules. |
| **Emergent Compositional Skills in MoE VLAs**, Shah et al. | ICML 2026 workshop (Compositional Learning) | [2607.20771](https://arxiv.org/abs/2607.20771) | LoRA-style MoE on the action head of pi0 / SmolVLA; experts specialise into reusable low-level behaviours, router sequences them. | 1, 2 | Learned end to end, not continual, no concept read. |
| **MergeVLA**, Fu et al. | CVPR 2026 | [2511.18810](https://arxiv.org/abs/2511.18810) | Merges per-task VLA experts with task-masked sparse LoRA and a cross-attention-only action expert; test-time task router; naive merging gives near-zero success. | 1, 5 | Merging per-task experts; no held-out compositions. |
| **CARVE, "Skill-Preserving Continual Merging of VLA Experts"**, Park, Kwon | NeurIPS 2026 ([page](https://neurips.cc/virtual/2026/163795)) | none found | Continual merging of VLA experts into a shared core + task-specific spectral residuals; 96.3% vs 96.7% for the full expert bank at 45% storage on continual LIBERO. | 5 (storage) | A storage-vs-parity result on LIBERO comparable in kind to our 1/5 claim. |
| **Compose Your Policies! (GPC)**, Cao et al. | ICLR 2026 | [2510.01068](https://arxiv.org/abs/2510.01068) | Training-free convex combination of the scores / velocities of several pretrained diffusion or flow policies (incl. VLAs), with test-time weight search. | 3 | Composes whole policies at every step; ours composes conditionals of one policy, switched per phase. |
| **PoCo, "Policy Composition from and for Heterogeneous Robot Learning"**, Wang, Zhao, Du, Adelson, Tedrake | RSS 2024 | [2402.02511](https://arxiv.org/abs/2402.02511) | Composes trajectory-level diffusion policies by summing scores (task, behaviour, domain level). | 3 | Score sums at every step; no phase switching, no frozen VLA conditionals. |
| **Flexible Multitask Learning with Factorized Diffusion Policy**, Liu, Chen, Hoeg, Yao, Li, Hauser, Du | arXiv 2025-12 (RA-L DOI listed) | [2512.21898](https://arxiv.org/abs/2512.21898) | Composition of specialised diffusion components per behaviour sub-mode; new tasks by adding / fine-tuning components. | 1, 5 | Sub-mode components, trained jointly; not concept-keyed. |
| **Factored Diffusion Policies**, Mitra, Yuceel, Giles, Pai | arXiv 2026-05 | [2605.22596](https://arxiv.org/abs/2605.22596) | One network with per-factor null-token dropout; additive factor scores generalise to held-out factor combinations (drone racing). | 3 | Conditional factor composition at every step (summed), not per phase; drones. |
| **Generative Skill Chaining**, Mishra, Xue, Chen, Xu | CoRL 2023 | [2401.03360](https://arxiv.org/abs/2401.03360) | One diffusion model per skill; chains skills by sampling them jointly with transition constraints. | 2, 3 | Planning-level chaining of separate models, not switching conditionals of one VLA. |
| **DeCo**, Nanjing Univ. et al. | arXiv 2025-05 | [2505.00527](https://arxiv.org/abs/2505.00527) | Splits demos into atomic tasks **at gripper-object interactions**; a VLM schedules atomic skills for novel compositions, tracking progress by gripper interactions; RLBench. | **2** | Prior use of gripper events as the composition boundary for zero-shot compositions; ours switches adapters / prompts of a frozen VLA. |
| **Diagnosing Compositional Generalization in Sequential Robot Tasks**, Wang et al. (Tomizuka group) | arXiv 2026-07 | [2607.29687](https://arxiv.org/abs/2607.29687) | Decomposes the OOD gap; sparse training fails by "instruction steering rather than missing low-level skills"; one demo per task raises OOD success 0.4% -> 54.7%. | 3, 5 | Diagnostic; supports our G4b reading (the policy holds the skills; the instruction misroutes them). Cite as support. |
| **Hi Robot**, Shi, Ichter et al. | ICML 2025 | [2502.19417](https://arxiv.org/abs/2502.19417) | A VLM emits the next atomic command to a low-level pi0-style policy, i.e., the low-level policy's conditional is switched per subtask. | 3 | Switches among commands the low-level policy was trained to follow per subtask; ours switches to another task's training instruction at the grasp with no new training. Cite as the hierarchical prompt-switching precedent. |
| **FAIL-Detect, "Can We Detect Failures Without Failure Data?"**, Xu et al. (TRI / Stanford) | RSS 2025 | [2503.08558](https://arxiv.org/abs/2503.08558) | Sequential OOD detection for diffusion and flow-matching policies; best score is a flow-based density (logpZO); conformal thresholds. | **4** | Detects failing rollouts at run time; ours detects whether a CONCEPT is new from its demo frames. Same family of signal. |
| **VLA-FAIL**, Seligmann, Gospodinov, Dincer, Neumann | CoRL 2026 | [2606.21386](https://arxiv.org/abs/2606.21386) | Mahalanobis OOD on last-layer features of flow-matching VLAs + chunk consistency for failure detection. | 4 | Baseline detector a reviewer will ask for. |
| **The Geometry of Flow-Matching Uncertainty (accel)**, Rao et al. | arXiv 2026-07 | [2607.27933](https://arxiv.org/abs/2607.27933) | Cost-free uncertainty proxy from flow-matching velocity curvature; flags failing VLA rollouts. | 4 | Another flow-intrinsic signal; run-time, not concept-level. |
| **Denoising diffusion models for OOD detection**, Graham et al. | CVPR-W 2023 | [2211.07740](https://arxiv.org/abs/2211.07740) | Multi-noise-level denoising reconstruction error as an OOD score. | 4 | The generic "denoising loss detects novelty" precedent; ours applies it to a VLA's action field per concept. |
| **ThriftyDAgger**, Hoque et al. | CoRL 2021 (PMLR 164) | [2109.08273](https://arxiv.org/abs/2109.08273) | Requests human intervention only on novel or risky states under a budget. | 5 | Precedent for "ask for a demonstration only when needed"; ours gates per task by practice success. |
| **MiDAS, "Adaptation of Generalist Robot Policies with Minimal Data"**, Kowshik et al. (Kumar group) | arXiv 2026-08 | [2608.11363](https://arxiv.org/abs/2608.11363) | One-demonstration BC anchor for a VLA, then online RL on a residual; LIBERO, RoboCasa, real. | 5 | One demo + autonomous practice to improve; ours uses practice only to accept / reject and to trigger requests. |
| **Pretrained VLAs are Surprisingly Resistant to Forgetting in Continual Learning**, Liu, Kim, Liu, Liu, Zhu | arXiv 2026-03 | [2603.03818](https://arxiv.org/abs/2603.03818) | Pretrained VLAs (pi0 among them) forget little under sequential fine-tuning with small replay on LIBERO. | 5 | Weakens "no damage" as a selling point; a fine-tune + replay baseline is now cheap and strong. |
| **Test-Time Compositional Generalization in Diffusion Models via Concept Discovery**, Wang, Gupta, Zhu, MacLellan | arXiv 2026-05 | [2605.07078](https://arxiv.org/abs/2605.07078) | Mode-ascent prototype discovery, submodular likelihood selection, PoE teacher; images. Names detection / consolidation of new primitives as future work. | 3, 4 | Our lineage; we add detection and a kept library in an embodied flow policy. |
| **Composable Diffusion**, Liu, Li, Du, Torralba, Tenenbaum | ECCV 2022 | [2206.01714](https://arxiv.org/abs/2206.01714) | Sum of conditional scores composes concepts. | 3 | Ours is a per-phase switch in environment time, not a sum in denoising time. |
| LIBERO lifelong baselines (ER, EWC, PackNet), Liu et al. | NeurIPS 2023 D&B | [2306.03310](https://arxiv.org/abs/2306.03310) | Benchmark + sequential, ER, EWC, PackNet. | 5 | Standard CL baselines missing from our tables. |

Also checked and judged not close: CoStream ([2606.26423](https://arxiv.org/abs/2606.26423), SE(3) blending of behaviours), Learning
Diffusion Policy from Primitive Skills ([2601.01948](https://arxiv.org/abs/2601.01948), router over 8 motion primitives), HSR
([2608.24042](https://arxiv.org/abs/2608.24042), demo retrieval), TraceFlow ([2609.20646](https://arxiv.org/abs/2609.20646),
guidance of a frozen flow VLA from success / failure traces; cite near DSRL as frozen-policy adaptation), RETAIN
([2512.08333](https://arxiv.org/abs/2512.08333), weight interpolation).

The IPEC LIBERO-90 LeRobot defect: no public report was found (searches on Hugging Face discussions and the web returned
nothing). Check the dataset's Hugging Face discussion tab once more before claiming it as first report.

## 2 Verdict per contribution

| # | verdict | closest work | what remains ours |
|---|---|---|---|
| 1 concept hierarchy of phase LoRA prototypes | **partially anticipated** | PRISM (action-expert low-rank skill bank composed per chunk), PhaseLoRA (phase-conditioned action-expert LoRA), IsCiL / LOTUS / AtomicVLA (skill libraries), DMPEL | units keyed by instruction concepts (object type, instance child, goal) rather than skills or tasks; a type -> instance hierarchy; each unit pooled over every demo of its concept across tasks; no learned router. We found no prior work factoring a VLA's adapters into object and goal concepts. |
| 2 switching at the policy's own grasp | **partially anticipated** | Li's pi0^S (prompt switch at a fixed step lambda/2), AC-VLA and DeCo (gripper transitions as sub-task boundaries), PHASER (phase segmentation) | the trigger is the frozen policy's own sustained gripper closure at inference, and it switches adapters or conditionals of one frozen policy. Small as a mechanism; the ablation that the switch carries the gain is the defensible part. |
| 3 held concepts as pointers to the policy's own field, per phase | **anticipated in mechanism** (Li 2025/2026, same benchmark, same pi0) | Li's pi0^S and TLI; Hi Robot (per-subtask commands to a low-level policy); GPC / PoCo / Composable Diffusion (score composition) | automatic choice of source instructions from the concept read; grasp trigger instead of a fixed step; practice gating; results on pi0.5 and SmolVLA (Li reports pi0 only). **Our pi0 result (0.245) trails Li's published prompt switch (0.69) and TLI (0.83) on the same suites; this must be addressed before submission.** |
| 4 likelihood detects new concepts; selection fails | **partially anticipated** | CLARE (AE reconstruction z-score decides expand vs reuse), FAIL-Detect / VLA-FAIL / accel (flow-based OOD signals for VLAs), Graham et al. (denoising loss as OOD), Task-aware MoILE (embedding clustering) | the frozen policy's own flow-matching loss, computed per concept on its phase frames, as the expand-vs-pointer router, with zero extra parameters. The negative result on likelihood SELECTION for trajectories is new as far as we found. Caveat already in RESULTS: held concepts are scored on training demos (memorisation not excluded). |
| 5 practice-gated continual learner with path refits, 1/5 storage | **partially anticipated** | LOTUS (update existing skills + add new ones), CLARE (expand only where needed), DMPEL, TAIL / CORAL (per-task LoRA), ThriftyDAgger (request help only when needed), MiDAS (one demo + practice), CARVE (storage at near parity) | demo requests triggered by per-task practice success; refits of every concept on the demo's path (instance, type, goal) on a frozen flow VLA; measured damage on earlier arrivals. **The storage ratio is mostly a rank ratio** (rank-4 prototypes, 4.7 MB each, against rank-16 per-task LoRAs, 18.8 MB each; 22 units against 29): against a rank-4 per-task LoRA the saving would be about 29/22 = 1.3x, unless rank-4 per-task LoRAs lose accuracy. |
| 6 pre-registered paired protocol; data defect | **novel as practice** (no comparable pre-registration found in this literature) | none | keep; it is a methodology contribution, not a method one. |

## 3 Recommended changes

### 3.1 Contribution list (replace Section 1's list)

1. "A compositional library for a frozen flow-matching VLA whose units are instruction concepts (object type, instance as a child
   of its type, goal), each a low-rank change to the action expert fitted on the frames of its phase pooled over every
   demonstration of the concept. Skill-level adapter banks composed per segment exist (PRISM; PhaseLoRA conditions one LoRA on
   phase); to our knowledge no prior work factors a VLA's adapters into object and goal concepts and switches them at the
   policy's own grasp."
2. "For concepts the policy already holds, a zero-byte unit: the policy's own field under a training instruction, switched at the
   grasp. This extends Li's prompt switching and text-latent interpolation (which hand-pair base tasks and switch on a fixed
   schedule, pi0 only) with automatic source selection, a grasp trigger and practice gating, on three policies." Do not present
   this as new composition; present it as making Li's operator part of a learner that also stores new concepts.
3. "The frozen policy's own flow-matching loss on a concept's phase frames as the expand-or-point decision. Earlier robot CL
   decides expansion with a separate novelty model (CLARE's autoencoders; embedding clustering in Task-aware MoILE), and
   flow-based densities flag failing rollouts (FAIL-Detect); here the policy itself is the detector and no parameters are added.
   Likelihood detects novelty but does not select among new concepts for trajectories." Keep the memorisation caveat in the
   sentence that reports AUC 1.0.
4. "A learner that requests one demonstration only when practice fails, refitting every concept on its path (cf. LOTUS, which
   updates and adds skills, and ThriftyDAgger, which requests help on novelty)." Report storage against a rank-matched per-task
   LoRA, or say "1/5 of rank-16 per-task LoRAs (TAIL-style)" explicitly.
5. Protocol and data defect: unchanged.

Drop "the detector of new primitives Wang et al. lack" from the abstract, or soften to "a detector of the kind Wang et al. leave
to future work": the detector idea itself (denoising error as novelty) is old; what is new is where it is used.

### 3.2 Related-work paragraph (draft text)

"**Continual imitation with module libraries.** Per-task adapters on a frozen pretrained policy avoid forgetting (TAIL, ICLR
2024; CORAL 2026); later work shares modules across tasks with learned routers (Sparse Diffusion Policy, CoRL 2024; DMPEL, TMLR
2026; LiMoDE 2026; AtomicVLA, CVPR 2026), grows libraries of discovered skills (LOTUS, ICRA 2024; IsCiL, NeurIPS 2024; SCE 2026),
or expands only where a novelty model fires (CLARE, RA-L 2026). PRISM (NeurIPS 2026) stores skill families as frozen low-rank
action-expert adapters and routes them per chunk, and PhaseLoRA (2026) conditions one action-expert LoRA on manipulation phase.
Our units are instruction concepts rather than tasks or skills, are switched by the policy's own grasp rather than a learned
router, and include zero-byte pointers to the policy's own conditionals. Pretrained VLAs forget little under fine-tuning with
replay (Liu et al. 2026), so we report replay and sequential fine-tuning alongside parameter isolation.
**Compositional generalisation of VLAs.** On the suites we use, Li (IJCAI 2026) switches pi0's prompt mid-episode and
interpolates text latents; AC-VLA and ACT-VLA (2026) retrain pi0.5 on decomposed or synthesised compositional demonstrations,
AC-VLA cutting sub-tasks at gripper transitions, as DeCo (2025) does for RLBench. Score-level composition of diffusion and flow
policies (PoCo, RSS 2024; GPC, ICLR 2026) sums whole policies at every step. We keep the policy frozen and switch its
conditionals per phase, and we compare with Li's operators on the same tasks.
**Novelty in generative policies.** Denoising error detects out-of-distribution inputs (Graham et al. 2023), flow-based densities
and features flag failing VLA rollouts (FAIL-Detect, RSS 2025; VLA-FAIL, CoRL 2026), and interactive imitation requests help on
novel states (ThriftyDAgger, CoRL 2021). We use the policy's own flow loss per concept to decide whether a concept needs new
parameters."

Remove "our group's companion work" (twice in Section 1, once in Section 2) for a double-blind venue; cite 2605.07078 in the third
person.

### 3.3 Experiments and baselines a reviewer will demand

In order of risk:

1. **Li's operators on our harness (critical).** Run pi0^S (task 1 prompt, switch to task 2 at lambda/2) and TLI on pi0, and the
   prompt switch on pi0.5 and SmolVLA, on the same 20 tasks, the same set-c seeds and Li's modified LIBERO. Two outcomes:
   (a) Li's numbers reproduce (about 0.69 / 0.83 on pi0): then our pointer composition is dominated and the paper must explain
   the gap (source-task choice, grasp trigger, scene matching) or adopt Li's base-task pairs as the pointer sources; (b) they do
   not reproduce: report that, with the checkpoint and episode counts (Li runs 10 episodes per task), so the reader is not left
   to compare 0.245 with 0.83. Also run our grasp switch with Li's hand-chosen base-task pairs, and the fixed-step switch with
   our automatically chosen sources, to separate trigger from source choice. (The pre-registration lists Li's TLI rates "as
   context only"; that is not enough once a reviewer sees the same suites.)
2. **Rank-matched per-task LoRA.** A rank-4 per-task LoRA (TAIL-style) with the same acceptance test, so the storage claim
   measures sharing, not rank. Existing G5c shows rank does not change the library's success; the per-task side is the missing
   arm.
3. **A routed adapter library without task ids.** CLARE (code public per its paper) or DMPEL on the same 30-task stream, with
   its own expansion rule, reporting success, storage and backward transfer. This is the closest "detect, then expand or reuse"
   competitor.
4. **Standard CL baselines on pi0.5.** Sequential LoRA fine-tuning and ER (LIBERO's protocol; Liu et al. 2026 show ER is strong
   for pretrained VLAs), so "no damage" is measured against something that can damage.
5. **Detector baselines.** Against the flow-loss detector: CLARE-style autoencoder reconstruction z-score, last-layer
   Mahalanobis (VLA-FAIL), and an instruction-embedding nearest-neighbour test, on the same 34 held / 40 new concepts, ideally on
   held-out demonstrations of training tasks (collect a few by scripted or teleoperated replay) to remove the memorisation caveat.
6. **Trigger ablation.** Grasp switch against a fixed-step switch and against PhaseLoRA-style continuous modulation (or a learned
   per-chunk router as in PRISM), with the same units, on LIBERO-90 and Li's suites.
7. **Training-based reference.** Quote ACT-VLA (pi0.5: 88.2 / 95.6) and AC-VLA (64.2 / 73.3) in the Li-suite table as the
   retrain-the-policy upper references, stating that both use the full policy and synthesised or decomposed training data.
