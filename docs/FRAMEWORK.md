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

## 5. Development pre-checks (P2, P3 on the existing pi05_libero checkpoint, 2026-10-22)

These are development reads, not the probes themselves. pi05_libero saw all 40 training tasks, so "leave-task-out" here
masks the task's instruction from the relation fit and from the splice sources, not from the weights. The same code
reruns on our split checkpoint (`experiments/pp_fields.py --checkpoint ... --tag ...`, then `pp2_dev.py --tag=...`,
`pp3_dev.py --tag=...`). Ledger rows `pp2_dev` and `pp3_dev` (tag `pi05_libero`); payloads `results/protos/pp2_dev.json`
and `pp3_dev.json`; status `results/protos/STATUS.md`.

**Setup.**
- Observations:
  - 200 frames from the 40 training tasks (5 evenly spaced frames of each task's first demo);
  - 100 frames from 25 LIBERO-90 tasks (4 frames each);
  - 112 first frames over 12 tasks for transfer (10 init states each; LIBERO-90 task 48 has only 2 stored).
- Conditions: 13 per observation. They are null, the instruction, 3 object parts ("the X": the named object and 2 scene
  distractors), 2 goal parts (own, 1 distractor) and 2 verb parts. Variants of the named parts are added: unspliced (plain)
  and templated ("pick up the X", "put it G"). Splices use bind's observation-free replace mode, sourced from a different
  training task.
- Keys: 6 levels (u = 0.1 to 0.7) and 4 starts per key. The starts are shared by all conditions: 2 instruction drafts and
  2 null drafts, noised to u. Each key gets 80 Adam steps of mode ascent and a 4-probe Laplace extent.
- Object positions: the raw LIBERO states replayed from HDF5. 39 of 40 training demos match their raw demo exactly (task 38
  has no match). The replayed end effector agrees with the dataset's to a few mm (5 cm at most).
- Displacement scale (calibrated on the demos): 0.011 m per unit position command per step.

**P2: modes under part prompts.**

| u | separation, median | share > 2 | distinct modes (rms tol 0.05 / 0.15) | object part points at its object (named / distractors / chance) | null points at the named object | goal part points at its goal (own / chance) | null points at the own goal |
|---|---|---|---|---|---|---|---|
| 0.1 | 10.3 | 0.83 | 2.4 / 1.2 | 0.42 / 0.29, 0.25 / 0.32, 0.29, 0.23 | 0.42 | 0.49 / 0.21 | 0.47 |
| 0.3 | 13.4 | 0.87 | 2.3 / 1.2 | 0.42 / 0.30, 0.25 / 0.32, 0.29, 0.23 | 0.43 | 0.47 / 0.21 | 0.47 |
| 0.55 | 11.1 | 0.99 | 2.3 / 1.2 | 0.42 / 0.32, 0.25 / 0.32, 0.29, 0.23 | 0.44 | 0.50 / 0.21 | 0.47 |

- **Separation passes, but trivially.** At u = 0.3 the ascent from 4 starts collapses to one point (within spread 0.002
  rms), while the conditions' modes differ by 0.036 rms. That difference is the size of the instruction mode's error to the
  demo chunk (0.044). The 9 core conditions form 2.3 clusters at tolerance 0.05 and 1.2 at 0.15. The ratio is large
  because the within-condition spread is nearly zero, not because the modes are well separated.
- **Attribution fails: part prompts do not move the mode toward their object.**
  - The named-object part points at the named object 0.42 of the time (chance 0.32, 1.3x). The null prompt does exactly the
    same (0.42-0.44), and the instruction does 0.46.
  - The distractor parts point at their own objects at chance (0.29 vs 0.29, 0.25 vs 0.23).
  - The goal part reaches 2.4x chance (0.49 vs 0.21), but the null prompt reaches 2.3x (0.47). The scene, not the part,
    sets the direction.
  - Splicing changes nothing: plain 0.44 vs bare 0.42 for the object, 0.51 vs 0.49 for the goal.
- **Transfer fails.** Across 10 init states, the named-object part's endpoint correlates with the named object's position at
  a median r of 0.06-0.07 over 11 tasks; the go bar is 0.5. The displacement against hand-to-object correlates at -0.05. The
  instruction itself gives 0.05, so a 10-step first chunk does not track the object's position on this policy at any
  condition.
- **The precision profile is uninformative.** The log10 precision is about 2.5 for every kind, dimension group and chunk
  half, and 46% of extents sit at the 1e-4 floor.
- **Hierarchy:** the median symmetric KL to the parent is 263 along u and 324 along condition granularity. Neither is
  calibrated yet.

**P3: relation R0 over part fields, leave-task-out over the 40 training tasks.** The table gives the share of the
instruction-vs-null field gap explained on held-out tasks, pooled over 200 observations.

| part variant | R0 | R1 | equal weights | Comp-1 | Comp-3 | bootstrap stability (median / 5th pct) |
|---|---|---|---|---|---|---|
| plain (unspliced) | **0.51** | 0.50 | 0.34 | 0.55 | 0.59 | 0.86 / 0.73 |
| bare, spliced | 0.31 | 0.30 | 0.16 | 0.55 | 0.59 | 0.87 / 0.63 |
| template, spliced | 0.34 | 0.32 | 0.15 | 0.55 | 0.59 | 0.90 / 0.66 |

- **Ceiling.** The same instruction's field evaluated in another batch explains 0.996 of the gap, so numerical noise does not
  limit these numbers.
- **By level and suite (plain).**
  - R0 rises with u: 0.42 at u = 0.1 and 0.67 at u = 0.7.
  - By suite: libero_object 0.81, libero_goal 0.74, libero_10 0.56, libero_spatial 0.10.
  - libero_spatial's instructions differ only by a spatial clause ("next to the cookie box") that no part prompt carries.
  - Without libero_spatial, R0 explains 0.73, against 0.53 for equal weights and 0.60 for Comp-3. This exclusion is post hoc.
- **R1 adds nothing** (-0.01 to -0.02). The context terms overfit.
- **Readout (plain, weights pre / post grasp).**
  - The object role combines as AND on position (0.87 / 0.76) and on rotation (0.89 / 0.74).
  - The goal role switches on after the grasp: position 0.36 / 0.76, rotation THEN (0.35 / 0.76). On the spliced variants
    the goal's position is also THEN (0.18 / 0.79).
  - The gripper is a fractional OR-share for both roles.
  - The verb role is off after the grasp.
  - This matches the h1_donate reading (object fixes the approach, goal fixes the carry), with one difference: here the
    object also carries rotation.
- **Splicing hurts R0** (0.31 vs 0.51). The observation-free replace mode moves part fields away from the regime the
  relation can use.

**Go / kill reading (development, pi05_libero).**
- P2: **kill as specified.**
  - Separation is met only in a degenerate sense.
  - Attribution (1.3x) and transfer (r 0.07) fail. The decisive control: the null prompt points at the same objects as the
    part prompts.
  - On this checkpoint, part prompts do not give fillers distinct from the observation's own default motion.
- P3: **below the bar on all tasks** (0.51 < 0.70). The templates are stable (0.86 ≥ 0.8), and R0 beats equal-weight
  superposition by 0.17.
  - R0 trails Comp-3 overall (0.51 vs 0.59). Comp-3 ranks by the demo's flow loss and retrieves near-duplicate training
    instructions, so it is privileged here.
  - R0 passes 0.70 only on the 30 non-spatial tasks (0.73), and that subset was chosen after seeing the results.
- What would change the reading on our split checkpoint:
  - a part prompt whose mode departs from null (the attribution control above);
  - parts that carry the spatial relation;
  - an R0 that beats Comp-3 when no near-duplicate training instruction exists.
