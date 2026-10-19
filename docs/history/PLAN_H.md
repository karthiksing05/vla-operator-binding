# Plan H: ratify, interpret, compare (2026-10-19)

The user's direction (2026-10-18/19): present the framework as an architecture-agnostic account of composition as the mechanism of
OOD learning; ratify it on more models, methods and benchmarks; interpret what each prototype and operator contributes and what
that says about composition in general; ablate every part on pi0.5; compare on accuracy, space and time; benchmark the way robotics
papers do. Literature and code search: subagent report of 2026-10-19 (summarised below, with its unverified items marked).

## Running
- G19 (pre-registered): task leaves on pi0, SmolVLA, X-VLA (29 LIBERO-90 tasks each, set d).
- G20 (pre-registered): ablation on pi0.5. LIBERO-90 on set d (leaves, practice, composition, object unit, goal unit, instance
  child, the grasp switch); OOD on set e (operator choice, source practice, grasp trigger, each held source).

## H1 Interpretation (the priority)
1. What each prototype contributes: the velocity-field difference it makes along trajectories at matched states, in end-effector
   space and per phase; the frozen policy's per-phase flow loss with and without it; principal angles between prototype subspaces;
   a scale sweep (the Concept Sliders analogue); the reusable-versus-memorised score of Swann et al. (arXiv 2603.19183).
2. What the operators do: TLI per layer and token, shift/residual ratio against success across the four architectures (why
   interpolation fails on SmolVLA); the grasp switch as a change of conditional field.
3. What composition says in general: does composed success factor into the component competences (P(correct grasp) x P(correct
   place | grasp), each estimated from its source task)? Where does it fail, and is the failure predicted by the policy's
   likelihood of the concept? Relate to Grant et al.'s finding (arXiv 2603.19233) that goal selection and motor generation split
   between the VLM and the action expert.

## H2 Cost: accuracy, space, time
Per task: trainable parameters, bytes (bf16), demos, fit A40-minutes, practice episodes; at test time extra forward passes,
control latency (100 queries, batch 1, A40, as OpenVLA-OFT reports), peak memory. Pareto plots of success against bytes and
against practice episodes plus demos, every baseline on them; for the stream, CLARE's AUC / FWT / NBT alongside demo requests.

## H3 Methods (baselines)
- GPC (Cao et al., arXiv 2510.01068, ICLR 2026): convex mixing of the frozen policy's velocity fields under two held
  instructions, weight searched by practice. Velocity-level composition against our instruction-level and latent-level operators.
- Activation injection (Grant et al., arXiv 2603.19233): a held task's action-expert activations injected; a third demo-free
  operator.
- CLARE (arXiv 2601.09512): frozen backbone, grown adapters, autoencoder router; port to our 1-demo stream (code location
  unverified).
- ACT-VLA (arXiv 2607.00351): retrains pi0.5 by BC on filtered TLI rollouts; our practice episodes are such rollouts. Report its
  published numbers on Li's suites; reimplement if compute allows (it changes the policy's weights).

## H4 Benchmarks and models
- LIBERO-PRO (arXiv 2510.03827; code public): perturbed objects, states, instructions; runs with our LIBERO checkpoints.
  LIBERO-Gen (compositional levels) if its code is public (unverified).
- CALVIN ABC->D with X-VLA (2toINF/X-VLA-Calvin-ABC_D) and FLOWER (arXiv 2509.04996).
- Models: GR00T N1.5 (LeRobot, flow), FLOWER (LIBERO-90 checkpoint).

## Protocol additions (from how robotics papers benchmark)
Report in-distribution numbers on LIBERO's standard 50 init states beside our fresh states; 3 fitting seeds for the LoRA arms;
keep pre-registration, paired episodes, exact tests and task-level bootstrap (no paper surveyed reports any of these).
