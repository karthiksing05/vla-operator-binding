# The framework: symbolic composition over diffusion prototypes (v2, 2026-10-21)

**Goal (user).** A hierarchical store of reusable prototypes, combined without supervision, that learns OOD tasks
efficiently and continually, as an alternative to fine-tuning. Composition should work like a symbolic framework (TRELLIS)
while drawing its generalisation from diffusion prototype extraction (the ICLR-27 prototype-discovery paper). The
experimental design that tests it is docs/DESIGN.md.

## Sources and what each contributes

| source | contributes |
|---|---|
| TRELLIS (Singaravadivelan & Langley, ACS 2026) | the *symbolic* half: **primitives and composites (chunks)**; every element described by its **content** (its parts) and its **context** (its surroundings); two linked Cobweb taxonomies; **parsing** bottom-up with a recognition threshold; **generation** top-down from a recombination pool ranked by context; learning = Cobweb's incremental incorporate / create / merge / split, with no separate chunk learner |
| Prototype discovery (Wang et al., ICLR 2027 submission; [arXiv 2605.07078](https://arxiv.org/abs/2605.07078)) | the *continuous* half: a prototype is a **mode of a noisy marginal** with a local Gaussian; the **noise level sets abstraction**, so one datum yields a hierarchy of prototypes; one OOD query **selects** prototypes by submodular coverage; selected prototypes are combined by a **closed-form product of experts** that guides the sampler |
| Diffusion models and concept formation (Wang, Singaravadivelan & MacLellan, ACS 2026) | the bridge: the mode hierarchy of a diffusion model **is** a Cobweb tree of Gaussian prototypes, with a basic level at intermediate noise; so diffusion discovery can initialise a Cobweb store (the diffusion-concept-trees repo distils the mode tree into a Cobweb tree) |
| Favero et al., ICML 2025 ([arXiv 2502.12089](https://arxiv.org/abs/2502.12089)) | diffusion models learn composition rules level by level, so a hierarchy over learned units is the natural organisation |
| This repo's measurements | prototypes must live in the policy's **conditioning space** and be re-evaluated at the current observation: action-space prototypes carry no "where" (q3_pilot 3 / 60); conditional fields compose (pointers, LoRA prototypes, KV tokens); composition is carried by the key / value split |

## The theory, in TRELLIS's postulate form

**Representation**
- **R1.** An *experience* is a demonstration, a sequence of observation-action frames. Its elements are related by *then*
  (temporal order) and *with* (concurrent factors within one step).
- **R2.** A *primitive* is a **prototype**: a Gaussian over codes z in the policy's conditioning space. Decoded and attended
  by the frozen flow policy, a code becomes a conditional action field v(a | o, z), re-evaluated at every observation.
- **R3.** A *composite* (chunk) is an ordered pair of elements under a relation. Under *then*, the parts execute in
  sequence. Under *with*, they execute at once, as a product of experts of their fields.
- **R4.** Every element has a **content** description (its parts, as references into the context taxonomy, plus a
  complexity tag) and a **context** description (the observation features at its boundaries and its neighbouring
  elements).

**Organisation**
- **O1.** A **context taxonomy** (Cobweb, Gaussian concepts) organises elements by where they occur. It answers "which unit
  fits here".
- **O2.** A **content taxonomy** organises composites by their parts. It holds the rewrite rules, e.g. "approach X then
  grasp X then carry to Y then release".
- **O3.** Every node of either taxonomy is a prototype at its own level of abstraction. Near the root are generic units
  (approach anything); at the leaves are specific ones (approach the black bowl). This is the noise-level hierarchy of the
  diffusion picture.

**Performance**
- **P1. Discover.**
  - Training demonstrations are cut into frames.
  - An amortised unit encoder E maps windows to codes, and a decoder D maps codes to the policy's conditioning (KV tokens
    on frozen VLAs; a code input on a policy we train ourselves).
  - E and D are trained through the policy's flow loss with no labels.
  - Prototypes are the modes of the code density at several noise levels (score ascent on a small diffusion prior over
    codes, as in Wang et al.). They seed the context taxonomy, by the concept-formation correspondence.
- **P2. Parse** (recognition, one demonstration, unsupervised).
  - Start from the frontier of the demo's primitive windows.
  - Repeatedly propose chunks over adjacent elements. Sort each candidate's content and context through the two
    taxonomies. Admit it if a non-root ancestor clears the recognition threshold. Commit the best-scoring candidate.
  - Segmentation therefore emerges from recognition. It is not fixed at contact events.
  - Within a window that no single prototype covers, select prototypes by submodular per-dimension coverage of its code
    and combine them by a product of experts: a *with*-composite. This is Wang's mechanism, placed inside the parse.
- **P3. Generate** (execute, closed loop).
  - Expand the parse's root top-down. At each step, choose the next expansion from the content node's recombination pool,
    ranked by the context taxonomy against the **current observation**.
  - The leaves are prototypes whose decoded fields drive the frozen policy.
  - Composites recombine familiar parts in new orders and new concurrent mixtures: the "creativity" of TRELLIS generation.
- **P4. Halt.**
  - Parsing halts when nothing is recognised. The unrecognised remainder marks what is **new**.
  - Generation halts when every element is a primitive.

**Learning (no new mechanism, as in TRELLIS)**
- **L1.** Parsed chunks and units are incorporated into the taxonomies by Cobweb: incorporate, create, merge, split.
  Learning is incremental, unsupervised and interleaved with performance, so learning a task is a parse plus a store
  update, with no gradient step on the policy.
- **L2. Creation.** A remainder that no prototype covers becomes a new primitive.
  - First as a new code: a few gradient steps of the flow loss in the low-dimensional z, through the frozen decoder.
  - Only if no code can express it (flow loss above the calibrated threshold) as a residual rank-4 adapter leaf attached to
    that node.
  - This is the "detect, consolidate and reuse genuinely new primitives" that Wang et al. leave for future work.
- **L3. Preservation.** The policy and existing codes never change, so an old task can only be affected through routing.
  That is measured (docs/DESIGN.md C4), not assumed.

## Why this should be efficient, and what could break it

- **Efficiency.** A new task costs one parse (forward passes) plus, at most, a few steps in a 32-64-dimensional code.
  Fine-tuning costs hundreds of steps in millions of parameters. Hierarchy lets a new task reuse an abstract node when no
  specific one fits.
- **Generalisation.** The symbolic structure (which units, in which order) is recombined exactly. The continuous leaves
  generalise by product of experts and by the frozen policy's re-evaluation at each new observation.

**Assumptions, each tested first** (cheapest decisive probe):

| # | assumption | probe |
|---|---|---|
| A1 | codes in conditioning space can carry a unit | plan F, F0 / F1 (running on pi0.5) |
| A2 | the code density has a mode hierarchy with a usable basic level | modes vs noise level; held-out category utility per level |
| A3 | a PoE of two code Gaussians decodes to behaviour that combines both (one part from each) | on CompoSuite: object from one unit, objective from another |
| A4 | parsing recovers unit boundaries better than contact events | parse boundaries vs the true sub-task changes (labels as diagnostics only) |
| A5 | generation, ranked by context against the current observation, executes in closed loop | rollouts of parsed training demos, with no instruction |
| A6 | creation of a code (then a leaf) closes the gap on a new part with few steps | G-new cells on CompoSuite |

## Relation to the earlier pieces

- Supervised pointers, LoRA prototypes and the soft blend are special cases or oracles:
  - a pointer is a code fixed to a training instruction;
  - the blend is a *with*-composite with fixed segmentation.
- The baselines and controls are listed in docs/DESIGN.md section 4.
- Plan F's probes F0 / F1 test assumption A1. The rest of plan F is superseded by this doc.
