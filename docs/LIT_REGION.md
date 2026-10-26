# LIT_REGION: can test-time methods fix "wrong region of the right object"?

Literature review, 2026-10-26. Web search and reading only. Every entry below was checked against its arXiv abstract or HTML full text on this date. "Abstract-only" means I did not read the full text, so details beyond the abstract are not claimed.

## 1. Verdict

As a blanket statement, "region errors are the limit of test-time composition" does not hold. The closest prior work is ReGuide (arXiv 2609.38616, 2026). It is training-free, wraps a frozen pi0.5, and is evaluated on LIBERO. Its "Target Region" cells include our failure: "put the cream cheese in the top drawer of the wooden cabinet". On that cell, bare pi0.5 succeeds 5/50 times; with ReGuide it succeeds 48/50. Over all four Target Region cells it goes from 81/200 to 192/200.

So region errors are fixable at test time, without fine-tuning and without target-task demonstrations. The cost is resources we did not use:
- an external grounding module: oracle simulator poses, or a detector plus depth (87.5% vs 94.5% with oracle on pi0.5);
- per-destination-class pose statistics taken from the *training* demonstrations;
- an action override that drives the end-effector to a demo-supported pre-contact pose, then hands control back to the policy.

Three analysis papers trace the failure to training-distribution shortcuts, not to a fundamental limit:
- CRAFT (2610.00524): vision overrides the instruction. pi0.5 executes a demonstrated task that shares the object 38.4% of the time on undemonstrated LIBERO-Goal combinations. CRAFT fixes this by fine-tuning on existing demonstrations only.
- CAG (2602.17659): the same vision shortcut.
- LIBERO-Plus (2510.13626): models "tend to ignore language instructions completely".

In VLMs and text-to-image models, binding and spatial-relation failures are real and widespread (2210.01936, 2310.19785, 2411.00238). Even there, inference-time attention intervention partly fixes them (Attend-and-Excite, 2301.13826).

**Correctly scoped claim:** for a frozen flow VLA whose only test-time resources are its own samples, its own activations and its own language channel, region errors resisted every policy-internal probe we tried: guidance, self-selection with a learned first-chunk verifier, activation and contrastive re-conditioning, and retry. The fix requires an external spatial grounding signal (object or region pose, a VLM point or trace, or a learned outcome model) that tells the system *where*, while the frozen policy keeps supplying *how*. Our negative result is about what the policy itself contains, not about test-time methods in general. The oracle 0.53@16 also shows that the right behavior is already in the policy's sample distribution. What fails is identifying the right sample, not producing it.

## 2. Methods table

Abbreviations: TT = test-time; Fz = policy frozen.

| Method | What it fixes | TT? | Fz? | Extra resource needed | Verified source |
|---|---|---|---|---|---|
| ReGuide (Zhang et al.) | Global grounding under recombination (wrong referent or destination, incl. top drawer); keeps local skill. pi0.5 composition 39.0 to 94.5; real xArm 15 to 90% | yes | yes | Object poses (sim oracle, or open-vocab detector + depth); training-demo EE/object pose statistics per destination class; action override until a pre-contact set, then hand-back | arXiv 2609.38616 (2026), HTML read |
| CAG (Fang et al.) | Vision-overrides-language "counterfactual" failures on LIBERO-CF. pi0.5: +9.7 grounding / +3.6 success (training-free); +15.5 / +8.5 with a VA model | yes | yes (VLA) | CFG-style mixing against a *language-dropped* branch (omega=1.5). Training-free variant drops language in the same VLA; better variant trains a language-free VA policy on the same LIBERO data | arXiv 2602.17659 (2026), HTML read |
| VERITAS (Zhang, Shah) | General success of pi0 / pi0.5-DROID: +12.6% SimplerEnv, +35% real (not region-specific) | yes | yes (steering) | VLM (Gemini 2.5) proposes a pixel waypoint trace once per episode, plus detection/tracking. Each of N=5 sampled chunks is projected into the image and scored by distance to the trace. Needs camera calibration | arXiv 2606.18247 (2026), HTML read |
| RoboMonkey (Kwok et al.) | OOD success of OpenVLA: +25 pts real WidowX, +9 SIMPLER; LIBERO-Long 49.8 to 56.5 | yes | yes | 7B LLaVA reward-model verifier trained on 20M synthetic action preferences from Bridge V2 (ranked by RMSE to expert action). Gaussian perturbation + voting over samples. LIBERO needed verifier fine-tuning | arXiv 2506.17811 (2025), HTML read |
| CoVer / CoVer-VLA (Kwok et al.) | "Intention-action gap": +22% ID / +13% OOD SIMPLER, +45% real | yes | not stated in abstract | Contrastive instruction-action verifier (training data not in abstract), VLM-generated instruction rephrasings, joint selection over rephrasings x action samples | arXiv 2602.12281 (2026), abstract only |
| V-GPS (Nakamoto et al.) | General success of 5 generalist policies on 12 tasks | yes | yes (black-box) | Value function learned by offline RL on robot data. Authors note it was trained on 2 embodiments only and is limited on unseen language and objects | arXiv 2410.13816 (2024), abstract + summary |
| ITPS (Wang et al.) | Steering toward a user-chosen sub-goal or mode | yes | yes | Human points, sketches or corrections. Finding: plain guided diffusion leaves the data manifold (incoherent at high guide ratio); stochastic MCMC sampling balances alignment vs constraint violation best | arXiv 2411.16627 (2024), HTML read |
| DynaGuide (Du, Song) | Steering a diffusion policy toward a target behavior: CALVIN articulated ~70%; 5.4x over goal-conditioning with weak goals | yes | yes | Latent dynamics model (DINOv2 patches -> far-future latent) trained on play/robot data. Positive and negative goal images. Tested with DDIM diffusion; flow matching untested | arXiv 2506.13922 (2025), HTML read |
| FOREWARN (Wu et al.) | Filtering a generative policy's candidate plans by predicted outcome | yes | not stated in abstract | Latent world model + VLM aligned to its latent predictions | arXiv 2502.01828 (2025), abstract only |
| MOKA | Open-world manipulation from point affordances | yes | n/a (no learned policy; VLM + motion primitives) | VLM with mark-based visual prompting (candidate points/grids) | arXiv 2403.03174 (2024) |
| PIVOT | Zero-shot spatial action choice by iterative visual QA | yes | n/a | VLM (GPT-4V default); numbered arrow candidates drawn on the image | arXiv 2402.07872 (2024) |
| ReKep | Relational keypoint constraints solved as closed-loop optimization | yes | n/a (planner, not a VLA) | VLM-generated constraints, keypoint tracker, rigidity model, staged task | arXiv 2409.01652 (2024) |
| RoboPoint | Spatial affordance points from language ("free space left of X"). Beats GPT-4o/PIVOT by 21.8% accuracy, 30.5% task success | yes (as a module) | n/a | Fine-tuned VLM (synthetic data); depth to lift points | arXiv 2406.10721 (2024) |
| RoboRefer | Spatially constrained point referring (31 relations), placement location; +12.4% over Gemini-2.5-Pro on RefSpatial-Bench | yes (as a module) | n/a | SFT+RFT VLM with depth encoder; combined with downstream control policies | arXiv 2506.04308 (2025) |
| HAMSTER | OpenVLA generalization (+20% abs) via a 2D path | yes | no: low level is *trained* to follow paths | Fine-tuned VLM path predictor + path-conditioned 3D policy | arXiv 2502.05485 (2025) |
| RT-Trajectory | Task generalization via a 2D trajectory sketch | yes | no: policy trained on hindsight sketches | Sketch source (human, video, or foundation model) | arXiv 2311.01977 (2023) |
| RT-Affordance | Novel tasks via a keyframe-pose affordance plan (>50% over prior) | yes | no: policy trained on affordance conditioning | Affordance predictor | arXiv 2411.02704 (2024) |
| SuSIE | Subgoal-image planning (CALVIN SOTA at the time) | yes | no: goal-conditioned low level trained | InstructPix2Pix fine-tuned on video | arXiv 2310.10639 (2023) |
| MolmoAct | Depth tokens -> editable image-space trace -> actions | yes (trace editing) | no: the model is trained to produce and obey traces | Its own trained trace head | arXiv 2508.07917 (2025) |
| Hi Robot | Open-ended prompts and feedback, via a VLM choosing the next step in language | yes | low level is a VLA | High-level VLM, trained with synthetic data in the paper | arXiv 2502.19417 (2025) |
| CRAFT (Yang et al.) | Undemonstrated skill combinations (vision shortcut) on LIBERO-built Pick-Place(-Press); pi0/pi0.5/GR00T | no | no: fine-tuning | Counterfactual instruction pairs from *existing* demos; VLM skill segmentation (training only) | arXiv 2610.00524 (2026), HTML read |
| LIBERO-PRO | Diagnosis only: >90% LIBERO -> 0.0% under object/state/instruction/environment perturbations; near-identical trajectories regardless | -- | -- | -- | arXiv 2510.03827 (2025) |
| LIBERO-Plus | Diagnosis only: camera and initial-state brittleness; language largely ignored | -- | -- | -- | arXiv 2510.13626 (2025) |
| Attend-and-Excite | T2I subject neglect and attribute mis-binding, fixed at inference via cross-attention excitation | yes | yes | Token-level attention access + per-step latent optimisation | arXiv 2301.13826 (2023) |

Name traps:
- arXiv 2610.00524 is the CRAFT *method* paper ("Same Scene, Different Task: Skill Alignment..."); its benchmarks are called Pick-Place and Pick-Place-Press. A different CRAFT, arXiv 2602.12532, is force-aware fine-tuning.
- Search engines surface a different "ReGuide", arXiv 2606.28939 (Lin et al.; guided rollouts used as fine-tuning data, Robomimic). Ours is 2609.38616.

Spatial-failure reporting:
- ReGuide reports and fixes region errors, but it does not separate wrong-region from wrong-object counts in the text I read. Its scope excludes relational or ambiguous instructions and multi-factor changes. It does not say whether shelf levels are covered.
- CRAFT's Table A.4 is the closest to a wrong-receptacle count. "Shares target" is 2.8% for pi0.5 and 48.8% for pi0. The example is pi0.5 putting the wine bottle on the rack when asked for the stove.
- CAG's LIBERO-CF measures grounding by contact with the instructed target, with no region metric. Its real-world "goal targeting" scene shows wrong-goal placement, which CAG improves.

Fundamental-limit literature:
- VLMs act like bags of words on relations and order (ARO, 2210.01936). The fix is hard-negative training.
- 18 VLMs are poor at left/right/on/under (What'sUp, 2310.19785). The authors attribute this to scarce relation data; simple data up-weighting was not enough.
- Campbell et al. (2411.00238) frame multi-object failures as a binding problem from the lack of serial processing.

None of these papers claims VLA region failures cannot be fixed. The VLA papers above all attribute them to dataset shortcuts.

## 3. Which failed probe each extra resource would have rescued

**A. Object or region pose plus training-demo statistics (ReGuide).**
- *Rescues motion/goal guidance.* Our guidance pulled the flow sampler toward a vision anchor, so the samples left the policy's manifold (ITPS documents the same effect). ReGuide does not guide the sampler. It (i) reads the predicted endpoint of the unexecuted chunk and triggers only when that endpoint commits to a competing referent, (ii) drives the end-effector, by bounded override, into a pre-contact set built from demo statistics for that *destination class* (median entry altitude, grasp offset), then (iii) hands back. After hand-back the policy is in an in-distribution configuration for the right region, so its local placing skill is enough.
- *Rescues retry.* ReGuide allows two transports per stage. A retry that starts from the same wrong-region state is a no-op; a retry that starts from the demo-supported pre-contact pose is not.

**B. VLM point or trace for the instructed region (VERITAS, RoboPoint, RoboRefer, MOKA, PIVOT).**
- *Rescues self-selection.* The oracle shows that the right sample exists (0.53 within 16). Our learned verifier read first-chunk geometry, which does not yet encode the region choice. A VLM point for "inside the top drawer" against "on top of the cabinet", lifted with depth, gives an external score: the distance from each candidate's predicted endpoint or projected trace to that point. VERITAS does exactly this with N=5 and gains 35% on average on real pi0-DROID and pi0.5-DROID. It also provides the target that probe A lacks: an image point that specifies the sub-region (drawer interior), not just the object.

**C. Outcome model (DynaGuide latent dynamics, FOREWARN world model + VLM).**
- *Rescues selection made before the grasp.* A far-future latent predictor scores where the episode ends from the current action, before the region is visible in the first chunk. That is the information our first-chunk verifier was missing.
- *Rescues guidance.* Guidance becomes "toward the predicted outcome of goal images g+, away from g-". In LIBERO-90, training frames of *other* objects in that top drawer can serve as g+ without target-task demos. DynaGuide reports 5.4x over goal-conditioning when goals are weak.

**D. Language-dropped vision prior as the negative branch (CAG).**
- *Rescues contrastive re-conditioning.* We contrasted against a wrong-region phrase, which is still a language condition near the instructed one. CAG contrasts against the *language-free* prediction, i.e. the vision shortcut itself, and does so at a moderate scale (omega=1.5; large omega hurts).
- Caveat: the training-free variant needs a policy that behaves sensibly with language dropped. Our pi0.5 was not trained with language dropout, so expect the stronger CAG-VA variant to be required. That variant trains a language-free VA policy on the same LIBERO-90 data, which needs no new demos and leaves the VLA frozen.

**E. Learned value function or VLM reward verifier (V-GPS, RoboMonkey).**
- *Would probably not rescue selection.* Both are trained on other datasets (Bridge, two embodiments). RoboMonkey's labels are RMSE to the expert action, which carries no region-binding signal on novel compositions. V-GPS reports weakness on unseen language and objects. These limits are consistent with our verifier getting no signal. They would need region-relevant training data.

**F. Hierarchical or intermediate-representation conditioning (HAMSTER, RT-Trajectory, RT-Affordance, SuSIE, MolmoAct, Hi Robot).**
- *Would not rescue anything with a frozen pi0.5.* Each requires a policy trained to accept the intermediate representation. The exception is Hi Robot-style language sub-steps, but LIBERO-Plus and CRAFT show that language is the channel the policy under-uses. That matches our failed activation re-conditioning.

## 4. Ranked untried test-time routes

1. **ReGuide-style rebinding with perception grounding.** Most likely to work; it is reported on pi0.5, LIBERO and top-drawer cells.
   - Method: detector + depth, or sim poses as an upper bound. Endpoint check on each chunk. When the endpoint commits to the wrong region, apply a bounded EE override to the destination-class pre-contact set taken from LIBERO-90 training-demo statistics, then hand back.
   - Cost: a grounding module; demo EE/object pose logs (we have them); one controller.
   - Constraints: oracle grounding counts as "simulator at decision time", so the honest number uses perception (ReGuide loses 7 points). The override is an external controller, not composition inside the policy. Ships with ReGuide as baseline anyway.
2. **External-target verifier over the policy's own samples (VERITAS-style).**
   - Method: one VLM/RoboRefer point query per episode for the instructed sub-region, lifted with depth. Score each of N in {5..16} samples by the distance from its *predicted endpoint* (integrate the full chunk, or re-query near transport) to the point. Select at the step where the region choice commits, not at the first chunk.
   - Cost: N x policy calls per decision; one VLM call; camera intrinsics/extrinsics.
   - Constraints: the ceiling is the 0.53@16 oracle; it fails if no sample reaches the drawer interior.
3. **CAG with a language-free VA branch.**
   - Method: train a language-free pi0.5 VA on LIBERO-90 (no new demos), then mix at omega ~1.5. Also try the training-free variant with dropped language first; it is cheap.
   - Cost: one extra training run of about 30k steps; 2x inference.
   - Constraints: reported gains are modest (+3.6 to +8.5 success), and it targets "ignored language", not sub-region precision.
4. **Outcome-model ranking or guidance (DynaGuide/FOREWARN).**
   - Method: train a DINOv2-latent far-future predictor on LIBERO-90. Rank samples, or guide with ITPS-style stochastic sampling (not plain gradient guidance), toward g+ images of objects in the instructed sub-region, taken from training frames.
   - Cost: training the world model; per-step image encoding.
   - Constraints: DynaGuide is validated only on DDIM diffusion. FOREWARN reports that most of its errors come from imprecise world-model imagination.
5. **VLM rephrasing x sampling with a contrastive verifier (CoVer-VLA).**
   - Cost: a contrastive verifier trained on LIBERO-90 instruction-action pairs, plus a VLM.
   - Constraints: lowest prior. Rephrasing works through the language channel the policy ignores (LIBERO-Plus), and the verifier faces the same compositional-shift problem as ours.

## Coordinator spot-check (2026-10-26)

Re-read arXiv 2609.38616 HTML, Appendix C.2 Table 9 (pi0.5, of 50):
- A2 "put the cream cheese in the top drawer of the wooden cabinet": 5 -> 48;
- A3 "put the black bowl in the top drawer of the wooden cabinet": 42 -> 46;
- A1 (stove) 1 -> 50; A4 (stove) 33 -> 48.

No Target Region cell involves a shelf. Perception grounding is GroundingDINO-tiny + RGB-D + declared asset geometry, registered
once at instruction time: Target Region 86.0% vs oracle 96.0%. The baseline is the public pi0.5, not a split-trained checkpoint.
During transport ReGuide REPLACES the policy's actions with its own bounded commands, then hands back. The region is thus
supplied by an external controller, not composed by the policy.
