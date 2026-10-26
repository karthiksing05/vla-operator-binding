# Novelty-collision check (2026-10-26)

Web reading only. Sources are the arXiv HTML full texts: 2603.19233v4 (read in three offset passes, 0-262k chars), 2610.10178, 2605.28527, 2609.38616v1 (0-171k), and 2606.18247v1 (0-100k of 103k).
Where a paper does not contain something, this note says "not reported". Quotes are verbatim fragments.
Caveat: the reads went through a summarising fetch tool. Check each quoted number against the PDF before citing it.

Our setting: our own pi0.5, trained only on the LIBERO-90-C training cells, so held-out combinations are unseen at the weights.
None of the five papers uses a checkpoint with held-out combinations excluded by construction (see each "Checkpoints" line).

## 1. 2603.19233, "Not All Features Are Created Equal" (six VLAs)

**What it claims.**
- Fine-tuned VLAs run "coordinate-bound motor programs" in the action-expert pathway, and a visual-language pathway selects which program runs.
- The main test is cross-task activation injection (Sec. 3.2, 4.3, Table 3). The residual stream from a source episode (task A, scene A) overwrites the destination's expert activations at layers 16-17, at the 50 action-token positions, on every step.
- pi0.5 follows the source task in 99.6% of episodes (n=1,968; destination success 2.6%). Other models: OFT 96.4%, X-VLA 99.8%, SmolVLA 52.1%, GR00T N1.5 57.0%, ACT 0.0%.

**Models.** pi0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T N1.5, ACT.

**Tasks.**
- LIBERO goal/object/spatial/10 (40 tasks), MetaWorld, SimplerEnv, ALOHA, and a physical UR10e.
- Over 420k rollouts.

**Checkpoints.**
- pi0.5 is only described as "fine-tuned on LIBERO". The checkpoint name and whether the suites were trained jointly are not reported.
- The GR00T checkpoints are public community fine-tunes (App. K.6).
- All are public, standard-suite checkpoints. Each test task is a training task, so there are no held-out combinations.

**What "spatially bound" means.**
- The phrase occurs only in the v1 abstract (19 Mar 2026): "spatially bound motor programs tied to scene coordinates rather than abstract task representations".
- v4 replaces it with "bound to absolute workspace coordinates and ignores the visible scene". This is a WORLD/workspace frame.
- Image, pixel or screen coordinates are never mentioned.
- The frame is never isolated. App. B fits the data to "absolute workspace coordinates but do[es] not exclude scene-relative coordinates". It says a rigid whole-world transform "would separate the two, and we have not run it". The checklist confirms that no coordinate-frame control was run.

**How it was tested.**
1. Injection in sim: "The robot reaches toward where the source objects would have been" (Sec. 4.3). The images and the prompt are unchanged; only the expert activations are replaced.
2. Physical displaced-object test (App. O, 50 trials, source object moved 10-15 cm). The injected arm "drove to the recorded coordinates every time and closed on empty air". The key control: "Without injection and with the source prompt, the policy reaches the displaced object", so "the coordinate locking belongs to the injected program".
3. Noun swap in a fixed scene (App. K.3): "Object identity does not predict reach target". One-way ANOVA of final end-effector XYZ against the named object: libero_object F(10,209)=1.27, p=0.247; goal F(9,170)=0.30, p=0.973; spatial F(5,54)=2.20, p=0.067. No objects were displaced in this test.
4. LIBERO-PRO position perturbations take pi0.5 from 0.92-0.98 to 0.08-0.38 on all four suites ("coordinate-cluster brittleness", K.3). Horizontal and vertical image flips give 0% for every model.

**Is it a real contradiction of C2? No.** The settings differ in three ways:
- Their workspace lock is a property of expert activations transplanted across tasks. The uninjected policy tracks the displaced object, which is our C2 regime.
- Their frame claim is untested against scene- or object-relative alternatives, by their own statement (App. B).
- They use public suite-trained checkpoints with fixed layouts, and test within-suite and within-task on training tasks. Our data are 2-5 cm displacements on our checkpoint, plus leave-scene-out transfer (0.62 object frame vs -0.07 world frame) on the public pi05_libero.

The genuine tension is narrower. Under large layout shifts (LIBERO-PRO) the public pi0.5 does not follow objects. Our gain of 0.13-0.25 (well below 1) is compatible with that: the endpoint tracks the object, but only partially.

**Other relevant points.**
- App. G: an edit at the instruction positions, layers 2 and 5, "restores goal selection" on pi0.5 (113/150; true instruction 139/150; random direction 6/150). A pilot at layers 9 and 13 failed. Scope, verbatim: the edits restore a goal the policy already executes and do not override a competing instruction.
- Phase-specific steering localises pi0.5 sensitivity to the transport phase (p=0.013, App. H). No grasp-phase analysis is reported.
- Relational sub-region errors and K/V splicing are not reported.

**C1-C5 for paper 1.**
- C1, PARTIAL (mostly OPEN).
  - Binding is not localised to image versus word tokens. Object-patch locality is not tested.
  - K.3 shows the public pi0.5 does not use the noun to choose its reach. This fits "scene decides", but there is no binding effect to localise there.
  - App. G (word positions at layers 2/5 suffice to restore a goal) is in mild tension with our "binding leaves word positions by layer 2; restoring words recovers ≤0.25". Their restored goal is a trained, uncontested one, so the settings differ.
- C2, PARTIAL, apparent CONTRADICTION resolved above. Their App. O uninjected control supports object tracking. Pose parameterisation, release, per-type gains and the object frame are not reported.
- C3, OPEN (transport-phase steering sensitivity only).
- C4, OPEN. Their steering targets are goal subspaces, not recovery of binding errors.
- C5, OPEN.

## 2. 2610.10178, "Do VLAs Understand Instructions?" (pi0.5, GR00T N1.7)

**Checkpoints.** Public LIBERO-10 fine-tunes: `lerobot/pi05_libero_finetuned_v044` and `nvidia/gr00t17-lerobot-libero_10-640` (Sec. 5).

**Data.** Offline only: 10,000 steps sampled from `lerobot/libero_10_image`, which is the training data itself (Sec. 4.1). No rollouts and no task success are reported.

**Method.**
- Attribution patching on all layers, with activation patching on a subset of layers for verification.
- Patch site: the action-expert residual stream after o_proj (Sec. 4.3). Metric: flow-matching L2 loss against the demonstration.
- Five synthetic prompt perturbations. The Figure 1 caption says six.
- Word-versus-image token positions are not separated.

**Findings.**
- "Language conditioning is concentrated in the early layers" (Sec. 6.1). For pi0.5 this is layer 1 for directional perturbations, with attribution around layer 6 and past layer 12.
- Both models are "comparatively insensitive to abstract rephrasing and to referencing non-existent objects".
- Attribution does not predict activation patching for pi0.5 (Pearson 0.089, p=0.49; Table 1). For GR00T it does (0.946).

**C1-C5 for paper 2.**
- C1, PARTIAL at most: "early layers" is broadly consistent with binding leaving the words early, but it is measured as a loss sensitivity, with no localisation to image versus word tokens, no K/V splice and no held-out combinations.
- C2-C5, OPEN. Displacement, timing, interventions and failure classes are all not reported (they acknowledge "perturbations are synthetic" and that no closed-loop rollouts were run, Sec. 8).

## 3. 2605.28527, "What Frozen VLAs Already Know About Success" (probe over pi0.5 samples)

**Checkpoint.** "the official Pi0.5 LIBERO policy" (Sec. 4.1); the identifier is not reported. Test tasks are LIBERO-Goal training tasks.

**Method.**
- A ridge probe on frozen features predicts Monte-Carlo value. Pi0.5 vision encoder R² is 0.7377 on the demo split and 0.5510 on the task split (Table 1). Same-step pairwise accuracy is 94.22% (Table 2).
- Online, pi0.5 samples K=16 chunks. For each prefix the selector restores the simulator, executes the prefix, then scores the result with the probe. This is privileged simulator lookahead.

**Results (Table 3).**

| Task | Greedy | Random | Value | Note |
|---|---|---|---|---|
| Push plate | 26.7% | 33.7% | 44.3% | p=0.003 vs random |
| Wine rack | 35.7% | 36.0% | 44.0% | p=0.026 vs random |
| Open middle drawer | 38.67% | 37.33% | 39.33% | "a no-gain case" |

- Sec. 4.3: "The better continuations were already inside the sampling distribution."
- Coverage@K, phase analysis (before or after grasp), binding errors and token positions are not reported.

**C1-C5 for paper 3.**
- C1, C2, C5: OPEN. The middle-drawer no-gain case is not diagnosed as a wrong sub-region.
- C3, PARTIAL. It makes the qualitative claim that good behaviour is in the policy's own samples, but gives no coverage number and no timing. Our pre-grasp 0.53 versus post-grasp 0.20 split is new.
- C4, PARTIAL and context-dependent. Its selector works (+8 to +11 pp) only with simulator-reset lookahead, on training tasks. Our failed first-chunk verifier, which uses no lookahead, is a different handle. Positioning: lookahead-free selection fails on binding errors; their privileged lookahead is not available at deployment.

## 4. 2609.38616, ReGuide, "Correcting WHERE, Preserving HOW" (skim, 0-171k)

**Checkpoints.** pi0.5 is the "official LIBERO checkpoints of openpi" (Table 7). The authors fine-tuned GR00T N1.7 and OTTER on the four suites.

**Benchmark.** 16 composition cells (Target Region, Target Object, Unseen Object, Task Scene) recombining training tasks.

**Method.** A training-free wrapper.
- Semantic rebinding predicts the endpoint from the unexecuted chunk (Eqs. 2-3).
- Geometric rebinding transports the end-effector to a demonstration-derived pre-contact set, then hands control back (Eqs. 4-6).
- It uses oracle object state in sim; perception-based grounding gives 87.5% versus 94.5%.

**Theory.** Eq. 1 factors success into binding β and competence γ. Evidence:
- Pre-contact transport raises grasp success from 64.5% to 95.8% (App. C.1).
- Replacing post-hand-back actions drops success from 94.5% to 38.1% or 28.0% (Sec. 4.2).

**Baselines.** CAG attention steering gives 41.4% versus 39.0% bare on composition, and hurts original tasks (94.0% to 81.4%) (Table 17).

**Top-drawer cells.** A2 and A3 ("open top drawer"): bare pi0.5 5/50 and 42/50; ReGuide 48/50 and 46/50. They are not analysed as a failure class.

**C1-C5 for ReGuide.**
- C1: OPEN. There is no internal analysis.
- C2: PARTIAL. "Skills preserved" is shown behaviourally, but object frame and pose are not tested.
- C3: PARTIAL. The where-error is fixable before contact, but there is no coverage-by-phase.
- C4: PARTIAL COLLISION for attention steering only (CAG fails). The other handles are OPEN.
- C5: OPEN. The A2 data exist but are not framed as a failure class.

**Positioning.** ReGuide is the closest competitor to our recovery story. It fixes WHERE with external oracle grounding plus demo statistics. Our C1/C3/C4 explain mechanistically why internal handles cannot fix it.

## 5. 2606.18247, VERITAS (skim)

**Checkpoints.** π0-Bridge in SIMPLER, π0-DROID and π0.5-DROID on a real arm. No LIBERO.

**Method.** Best-of-N selection (N=5) with a VLM pixel-waypoint verifier and SAM-2 tracking. Sim gain +12.6%, real +35% (Sec. 6.2).

**Not reported.** Coverage metric, timing, binding errors and internals.

**C1-C5 for VERITAS.**
- C1, C2, C5: OPEN.
- C3: PARTIAL. It only says "gains depend on sample diversity".
- C4: OPEN. An external VLM verifier is not an internal handle.

## Matrix

| Claim | P1 2603.19233 | P2 2610.10178 | P3 2605.28527 | ReGuide | VERITAS |
|---|---|---|---|---|---|
| C1 scene-wide binding in deep image tokens | PARTIAL/OPEN (App. G mild tension) | PARTIAL ("early layers", no token split) | OPEN | OPEN | OPEN |
| C2 causal, pose-param., shared operators | PARTIAL (App. O uninjected tracks object; not a contradiction) | OPEN | OPEN | PARTIAL (behavioural "skills preserved") | OPEN |
| C3 region decided at grasp (cov 0.53 vs 0.20) | OPEN | OPEN | PARTIAL (qualitative "inside sampling dist.") | PARTIAL (pre-contact fix) | PARTIAL (diversity) |
| C4 internal handles fail | OPEN | OPEN | PARTIAL (lookahead selection works) | PARTIAL (CAG fails) | OPEN |
| C5 relational sub-region class | OPEN | OPEN | OPEN (drawer no-gain undiagnosed) | OPEN (A2 data only) | OPEN |

No cell is a full COLLIDES. No paper contradicts C2 once the settings are matched.

## Verdict and positioning

**C1 is novel.** No paper localises referent binding to image versus word tokens, uses K/V splices, or tests object-patch locality.
- Against P1: cite App. G, and note that their edits restore an uncontested trained goal. Ours rebind held-out combinations.
- Against P2: their pi0.5 patching is offline loss on training data, and attribution failed to predict activation patching (r=0.089).

**C2 is novel in its form:** the release operator, per-type gains, and the object-frame versus world-frame transfer.
- Do not frame it as overturning P1. Frame it as completing their unrun App. B control: the program frame is object-relative when vision is intact, and workspace-locked only when expert activations are transplanted.

**C3 is novel quantitatively.** Present it as the timing that explains why ReGuide-style pre-contact correction and P3-style sampling help only before the grasp.

**C4 is mostly novel.** Concede that CAG has already failed (ReGuide Table 17). Contrast our handles with P3's privileged simulator-reset lookahead.

**C5 is novel as a named class.** Cite ReGuide's A2 cell (bare pi0.5 5/50) as independent corroborating data.

**All papers use public suite-trained checkpoints on training tasks or recombinations of training tasks.** Our checkpoint, with held-out cells unseen at the weights, is a differentiator on every row; state it up front.

## One experiment to sharpen novelty

On our checkpoint in sim, reproduce P1's protocol: cross-task expert injection at layers 16-17 over the 50 action tokens, plus their App. K.3 noun-swap ANOVA. Add the rigid whole-scene transform that P1's App. B says it "has not run".

Score the chunk endpoint in world, scene and object frames, with and without injection. The expected result has three parts:
- uninjected follows the object frame (our C2);
- injected is workspace-locked (their result);
- the noun swap moves the reach on our checkpoint but not on public pi05_libero (our C1 binding effect).

This turns P1 from a perceived contradiction into a resolved special case, in one table.
