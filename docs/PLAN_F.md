# Plan F: a hierarchical store of reusable prototypes, composed without supervision, as an alternative to fine-tuning (user direction, 2026-10-21)

**The goal (user).**
- A hierarchical store of reusable prototypes.
- A way to combine them without supervision so that OOD tasks are solved efficiently, and continually.
- Leverage the compositionality literature and Zekun Wang's work.
- Baselines are set aside while the framework is iterated. The one reference kept is the per-task LoRA, the "traditional fine-tuning" this replaces.

## What we build on

**The lab's theory:**
- *Diffusion Models and Concept Formation* (Wang, Singaravadivelan, MacLellan, ACS 2026): the modes of a diffusion model's noisy marginals form a hierarchy that corresponds to a Cobweb tree of Gaussian prototypes. The noise level sets the abstraction, and a basic level emerges at intermediate noise.
- *Test-time compositional generalization via prototype/concept discovery* (Wang, Gupta, Zhu, MacLellan, [arXiv 2605.07078](https://arxiv.org/abs/2605.07078)): prototypes are found by score ascent, one query selects them by submodular per-dimension coverage, and they are composed as a product of experts (PoE) guiding the composable-diffusion sampler. A LoRA distils the PoE. Its stated gap: "a mechanism to detect, consolidate and reuse genuinely new primitives". That gap is this plan's step 4.
- Favero et al., ICML 2025 ([arXiv 2502.12089](https://arxiv.org/abs/2502.12089)): diffusion models learn composition rules hierarchically, low-level features first, with polynomial sample complexity per level. Hierarchy is the natural organisation of what a diffusion model learns.

**What this repo measured** (docs/RESULTS.md, docs/INTERPRETATION.md):
1. *Prototypes in action space carry no "where".* Wang's PoE ported to motion modes failed its pilot (3 / 60). A product of motion prototypes held over a phase is an average motion (q3_pilot).
2. *The policy holds one mode per observation* (q1_options), so score ascent in action space finds nothing to compose.
3. *What composes is a conditional field re-evaluated at the current observation.*
   - Pointers to the policy's field under a training instruction win at G1 / G2 on five policies.
   - Rank-4 LoRA prototypes donate where and when (h1_donate).
   - An 18.4 KB key/value token fitted from successes acts alone: 10 / 40 vs 2 (q6_main).
   - Composition lives in the key/value split (mechanism study).
4. *The unsupervised scheme is weak where it names units from a fixed vocabulary* (the 40 training instructions).
   - Likelihood names segments but does not cut them (U1).
   - The soft blend donates about 4% (D1) and does not beat the policy in closed loop (Z5).
   - It has no creation and no hierarchy.

**Design consequence.** Discover prototypes in the policy's **conditioning space**, as key/value codes, not in action space. A prototype is then a conditional field: wherever it is used, it is re-evaluated at the current observation. That keeps "where", and composition happens where the mechanism study found it.

## The framework

**1. Units without labels.**
- Cut every training demonstration at its contact events (gripper changes, as now).
- Learn an **amortised unit encoder** E(segment) → z ∈ R^d (d ≈ 32-64) and a **decoder** D(z) → a KV token. Both are trained through the frozen policy, by the flow-matching loss of the segment's actions.
- The instruction is **removed** (empty prompt). The code must carry what the segment does, so no word supervises it.
- This is a VAE posterior through a frozen flow policy. Its prior is the store (step 2).

**2. The hierarchical store.**
- A Cobweb tree over the codes z of all training segments: incremental, category utility, Gaussian concepts. The diffusion-to-Cobweb correspondence above says this is the natural form.
- Every node's mean decodes to a token, so every level is an executable prototype. Coarse nodes are generic motions ("approach an object"); leaves are specific ones ("approach the black bowl in kitchen 3").
- The basic level is the level with the largest held-out category utility, as in the ACS paper.

**3. Composition from one demonstration (unsupervised).**
- Encode each segment of the new task's demonstration: z = E(segment). The demo is the query, used once and never executed.
- Route z best-first down the tree. Reuse the deepest node whose likelihood covers z (incorporate).
- Within a segment, when no single node covers z, select a few nodes by Wang's submodular per-dimension coverage. Combine them by appending their tokens, which the attention composes, or by a PoE of their fields per action group (blend.py), whichever wins offline.
- At run time, segments advance at the policy's own contact events, as in program.py.
- Optional: refine z by a few gradient steps of the flow loss. This is low-dimensional, so it should be fast (link to docs/PLAN_CONV.md).

**4. Creation, consolidation, continual.**
- A segment whose code no node covers (category utility favours a new child, or its flow loss is above the calibrated threshold) **creates** a new node.
  - First as a new code under its best parent.
  - Only if the decoder cannot express it, as a residual rank-4 LoRA leaf attached to that node.
- Created nodes are reusable by later tasks.
- Cobweb's merge and split operators consolidate the store as tasks stream in.
- The frozen policy and the old nodes are never changed, so there is no damage by construction.

## Probes, cheapest decisive first

| | question | kill / go |
|---|---|---|
| F0 | Capacity: can one KV token per segment, fitted directly with the instruction removed, reproduce the instruction-conditioned field on that segment? (10 training tasks, held-out demo frames; flow loss vs the policy under its instruction, and vs an empty prompt with no token) | kill if the token closes < 50% of the gap between the empty prompt and the instruction |
| F1 | Amortised units: does E/D trained on all 40 training tasks' segments (no labels) give codes whose decoded tokens (a) match F0's per-segment tokens on held-out demos, and (b) execute in closed loop? Training tasks, the code sequence of a held-out demo, switched at the policy's own contact events, against the instruction-conditioned policy and an empty prompt. | go if the codes reach at least 70% of the instruction policy's success on training tasks |
| F2 | Hierarchy: the Cobweb tree over training codes. Basic level by held-out category utility. Do node codes decode to coherent behaviour at every level (rollouts of node means)? Are objects, goals and verbs recovered? (Labels used as diagnostics only.) | go if the basic-level nodes execute and a node is shared across tasks |
| F3 | One-demo OOD, graded G1-G4: encode, route, compose, optionally refine, create where needed. Arms: the empty prompt; the instruction alone; the supervised pointers; the unsupervised codes (with and without refinement and creation); the per-task LoRA as the fine-tuning reference. Practice is charged if used. | the main read, pre-registered |
| F4 | Continual: a stream over all new tasks. Tree growth, reuse rate, bytes per task, preservation; the cost to learn each task against its arrival index. | |

**Controls built into F1 and F3** (from docs/DESIGN_REVIEW.md):
- a shuffled code assignment;
- a zero code (the observation alone, which tests the LIBERO position shortcut);
- a nearest-training-segment code (retrieval, no composition);
- a library that leaves the target task out (the codes never saw the task).

**Policy:** pi0.5 first, since token folding exists for it (lifelong_ood/policy/token_fold.py). The framework needs only a conditioning cache the action expert attends to, which holds for pi0, SmolVLA and GR00T as well.

**Rough cost:** F0 about 3 A40-h; F1 about 10 (training plus rollouts); F2 about 10; F3 about 60; F4 about 40. Everything runs on SLURM.

## Files (planned)

`lifelong_ood/policy/units.py` (encoder, decoder, segment dataset over cached prefixes), `lifelong_ood/policy/store.py` (Cobweb tree over codes: insert, route, category utility, merge and split), `experiments/f0_token_capacity.py`, `f1_units.py`, `f2_tree.py`, `f3_compose.py`, `f4_stream.py`, with pre-registrations in `experiments/prereg/f*.py`. Note: the `f0_` / `f1_` prefixes are taken by older scripts (experiments/README.md); use `pf0_` ... `pf4_`.
