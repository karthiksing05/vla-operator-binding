# The framework (v3, 2026-10-22): action-space prototypes and a relation distilled from the VLA, composed at test time

The experimental design is in docs/DESIGN.md; the approved plan, with its probes, is in docs/PLAN_V4.md. This file
supersedes framework v2 (the TRELLIS/Cobweb store; its results stay in docs/S0_SYNTH.md as history).

**Story.** Decompose a VLA once into **prototypes** plus the **relation** the VLA itself uses to combine them. Generalising
at test time by composing them gives OOD continual learning that is robust: fast, with nothing forgotten, and with every
new part reused. Both expensive steps, extraction and relation, happen *before* any task, so adaptation is inference plus,
at most, the creation of one missing part.

## 1. Cognitive-science grounding

| idea | operationalised as |
|---|---|
| Principle of compositionality (Frege; Fodor & Pylyshyn, *Cognition* 1988) | composed field = R(prototype fields); R is the fixed, extracted mode of combination |
| Role-filler binding (Smolensky, *AI* 1990; Plate, *IEEE TNN* 1995; Hummel & Holyoak, LISA, *Psych. Rev.* 1997) | roles are R's weight templates over (dimension group × chunk step × phase × level); fillers are prototypes; binding is a filler weighted by its role template |
| Motor primitives superpose linearly (Bizzi, Mussa-Ivaldi & Giszter, *Science* 1991; Mussa-Ivaldi et al., *PNAS* 1994) | R's first form is a linear superposition of fields: composable diffusion (Liu et al. 2206.01714) with structured weights |
| Recognition by components (Biederman, *Psych. Rev.* 1987) | parsing a demo means classifying its segments into (role, filler) |
| Structure mapping (Gentner, *Cog. Sci.* 1983); basic level (Rosch 1976) | a missing filler falls back to its nearest coarser prototype (the noise-level hierarchy) before anything is created |
| Primitives plus learned rules of combination (Lake et al., *Science* 2015, *BBS* 2017; MLC, *Nature* 2023) | R is extracted once; a meta-trained bounded merger is used only if the closed forms leave a gap |

## 2. Extracted once per model, before any task

**Prototypes (fillers), from ICLR-27** (Wang, Gupta, Zhu & MacLellan, arXiv 2605.07078).
- **The key.** A prototype is a key P = (condition c, level u). The condition c is one of:
  - null;
  - a training-cell instruction;
  - an object, goal or verb part prompt (`policy/bind.py` splice);
  - a created unit.
- **Live field, mode and extent.** At the current observation o the key gives a live field eps(x_t, t | prefix(o, c)). From
  it come:
  - a mode μ_P(o): `backbones/score_ops.mode_ascent`, from the policy's draft chunk;
  - an extent Σ_P(o): `laplace_diag`, 4 Hutchinson probes.
- **Re-evaluation.** Keys are re-evaluated at every o. That is the fix for the earlier failure, where action-space
  prototypes held over a phase lost "where" (q3_pilot, 3 of 60).
- **Hierarchy.** Along u (coarse to fine, as ICLR's noise levels) and along condition granularity: null ⊂ part ⊂
  instruction ⊂ created unit.
- **Interpretation.**
  - The integrated path of the mode gives the end-effector displacement, the nearest object at the endpoint and the gripper
    timing.
  - The per-dimension Laplace precision gives which dimensions a filler pins (position, rotation, gripper; early or late in
    the chunk).

**The relation R, distilled from the VLA itself (self-supervised; no demos, no task labels).** On training-cell
instructions the VLA composes correctly, so R is the map from its part fields to its whole field:

`eps(· | o, "pick up X and place it on Y") ≈ R( eps(·|o, obj X), eps(·|o, goal Y), eps(·|o, verb), eps(·|o, null) ; t, phase )`

- **R0.** Closed-form least-squares superposition weights per role, over (dimension group × chunk half × phase × level).
  About 10² numbers, no network, every weight interpretable.
- **R1.** R0 with context-dependent weights: a GLM on t, gripper state and time since the last event.
- **R2.** A bounded set-transformer merger, used only if R0/R1 leave a gap:
  - weights only over frozen fields, with no free action output;
  - initialised at R0, which equals Wang's PoE when its weights are the precision ratios;
  - trained once, leave-task-out, MLC-style, and reused across runs.
- **Roles** are R's templates, learned without supervision. They are checked against the earlier h1_donate reading: object
  fixes position before the grasp; goal fixes position and gripper after it; neither fixes orientation.
- **Mode of combination.** R's weights say whether each role combines as AND (≈ 1, a product of experts), OR (sum to 1) or
  THEN (a switch at the grasp).
- **Stability.** R is stored with the model; its stability across fit runs is measured.

## 3. Test time: inference only

| stage | how | gradient? |
|---|---|---|
| Segment | candidate cuts at gripper events (the best cue measured) plus every 10th frame; keep the cuts minimising flow loss under R plus a per-segment cost | no |
| Classify roles | likelihood of a segment under each role template | no |
| Select fillers | Wang's submodular coverage restricted to the role's dimensions (`compose.loglik_perdim`, `submodular_select`), over keys at all levels | no |
| Recognise | coverage likelihood against a split-conformal threshold calibrated on leave-task-out training segments → held / coarse fallback / NEW | no |
| Create | only when NEW: a KV token for that role only (`token_fold.fit`, 18.4 KB), anchored on R's composition of the other roles; escalate to a rank-4 leaf if needed; R is never refit | that part only |
| Execute | re-evaluate the selected keys at o; compose with R; sample in 10 Euler steps; fillers switch at the policy's own gripper events; the demo is never executed | no |
| Continual | created fillers join the store; the policy and R stay frozen | no |

## 4. Decoding compositionality

The framework is Hupkes et al. ([1908.08351](https://arxiv.org/abs/1908.08351)). The measures come from Andreas (TRE,
[1902.07181](https://arxiv.org/abs/1902.07181)), Okawa et al. ([2310.09336](https://arxiv.org/abs/2310.09336)),
Uselis et al. ([2602.24264](https://arxiv.org/abs/2602.24264)), Lepori et al. ([2301.10884](https://arxiv.org/abs/2301.10884))
and Keysers et al. ([1912.09713](https://arxiv.org/abs/1912.09713)).

Each property is tested at three levels: representation (fillers and fields), relation (R) and behaviour (closed loop).
The holistic controls are a per-task LoRA and an MLC-direct imitator.
- **Systematicity:** swaps, and success flat in compound divergence.
- **Productivity:** multi-step and chain tasks, with success ≈ the product of part competences.
- **Substitutivity:** synonymous fillers are interchangeable.
- **Localism:** a role's contribution is unchanged by the other fillers; R's cross terms ≈ 0; drop-one effects confined to
  the role.
- **Overgeneralisation:** NEW fires on exceptions.
- **TRE.**
- **Topographic similarity.**
- **Role orthogonality and linear decodability.**
- **Structural modularity:** a double dissociation between object and goal roles.

Every task also gets a **composition card**: segments, roles, fillers with interpreted paths, R weights and NEW flags.
