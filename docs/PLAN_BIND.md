# Plan BIND: compositional operators, holistic binding (approved by the user 2026-10-26)

**Working title:** "Compositional Operators, Holistic Binding: Where Zero-Shot Composition Breaks in Vision-Language-Action Models,
and How to Repair It".

**Background docs:**
- the evidence base: docs/ATLAS.md, docs/ATLAS_PAPER.md, docs/PROBES_DEEP.md;
- the literature and novelty checks: docs/LIT_REGION.md, docs/NOVELTY_CHECK.md, docs/NOVELTY_DEEP.md.

## 1. Theory

A task is a set of roles with fillers, ℓ = {(object, f1), (goal, f2)}. A binding b assigns each role to a scene entity or region
with pose p_b. The flow VLA behaves as a mixture over bindings:

  π(a | o, ℓ) = Σ_b P(b | o, ℓ) · O(a | o, p_b)

- **Operators O are compositional.** They are causal, pose-equivariant (up to a gain) in the object frame, and shared across
  fillers. Evidence: D4 (approach), D5 (release), OF D1 (object frame 0.62 vs world -0.07).
- **Binding P is holistic and exemplar-like.** It behaves as P(b | k(o, ℓ)), where k is the nearest training (scene, instruction).
  It does not factorise as Π_i P(b_i | f_i, o). Evidence:
  - FM: binding errors are flat in compound divergence and worst in new scenes;
  - R1: the empty-instruction replay;
  - R3c: whole-image vs region splice;
  - R2 / R8: extracted codes carry their source context.
- **Binding is a latent committed at the grasp** (R7: coverage@16 0.53 pre-grasp, 0.20 during the carry).
- **Binding resolution:** type > instance > relational sub-region. C5 is the relational sub-region case.

**Positioning.**
- This is the formal and mechanistic refinement of "instruction-action binding" (2609.39971), which has no operator/binding split,
  no per-modality localisation, no commitment timing and a training-only fix (ECT).
- It complements ReGuide (2609.38616), which factors *success* into binding x competence and fixes errors by external pose override.
- C2 runs the frame control that 2603.19233 left unrun.

## 2. Interpretability experiments (pre-register each before it is read)

| id | prediction | data / cost | falsified if |
|---|---|---|---|
| **E1** | (a) Exemplar binding: in a zs binding failure, the wrong filler is the binding the scene's training tasks use most | training lexicon + FM / M1 failure types on the 29 dev tasks; CPU | the match rate for the most frequent training binding is not above a frequency-matched chance baseline |
| **E2** | P is measurable and predictive: the mass on the correct binding, estimated from early own-policy samples, predicts zs success across tasks (calibrated) | R7 sandbox code on the dev tasks; a few A40-h | rank correlation with zs success not > 0, or calibration slope CI excludes 1 by a wide margin (bars set at pre-registration) |
| **E3** | Operators are object-frame; a rigid whole-scene transform moves the approach endpoint with the scene; injected programs lock to the workspace (2603.19233 reconciled) | D4 tooling; a few A40-h | world-frame fit >= object-frame fit without injection |
| **E4** | (b) Relational depth: at matched scene familiarity, error rises from type to instance to relational sub-region | dev tasks coded by referring-expression depth; maybe authored tasks | no monotone trend |
| **E5** | (c) Dissociation: supplying b (oracle first) removes binding errors but leaves motion errors | closed loop, oracle b, then detector or pointing model | binding errors not reduced, or motion errors reduced as much |

E1 and E2 run first; they are the cheapest ways to falsify the theory. E3-E5 follow.

## 3. Solution: repair P without touching O

Novelty criteria from NOVELTY_DEEP:
- act only on P;
- no policy fine-tuning, which distinguishes it from ECT and CRAFT;
- no external arm override, which distinguishes it from ReGuide;
- act before the grasp;
- cover relational sub-regions.

| direction | idea | status |
|---|---|---|
| **S1 (preferred): factorised binder** | a small module, extracted ONCE per model from existing training data (no new demos, no policy weight change), computes the binding role by role from each role's word plus the model's own vision prototypes, and writes it into the deep image-token code (where R3c shows whole-image changes move binding, +0.44) | design after E1-E2 |
| S2: operator retargeting | keep the policy's binding and, using the object-frame equivariance (C2), retarget the bound slot to the correct referent; gain calibrated; the action still comes from the policy | design after E3 |
| S3: exemplar-aware gate | prediction (a) says in advance which wrong binding a scene will produce, and gates S1 / S2 without a learned verifier | needs E1 |

**Baselines:**
- ReGuide (perception and oracle);
- ECT-style counterfactual training;
- CRAFT;
- CAG / CAG-VA;
- IGAR;
- TAG;
- VERITAS-style selection;
- the one-demo token (ceiling).

**Open risk.** Relational sub-regions need a region-level representation, and the policy's own vision prototypes do not resolve
one today (SigLIP anchor within 5 cm only 0.07-0.16).

## 4. Rules (unchanged)

- SLURM only, on overcap or tail-lab with the exclude list. Nothing on the login node except edits, git and job control.
- Pre-register before any read. Audits stay at 0 / 0. Local commits of own files only.
- Never open split_sealed.json. Demos are fitted or scored only, never executed.

## 1.1 Refinement after E1 / E1b / E2 / E2b (2026-10-26; docs/PLAN_E12.md)

The data support a ROLE-SPECIFIC version of the binding claim.

- **Goal role: exemplar-like, scene-keyed.**
  - On zs goal errors, the wrong goal is the region that the SCENE's training tasks bind most often: +0.48 [+0.13, +0.87] above a
    frequency-matched baseline.
  - Examples: :29 the cabinet top (18 of 19), the binding of the same scene's training task "put the black bowl on top of the
    cabinet"; :40 the shelf top.
  - The goal binding is committed by the grasp (R7) and is visible at the settled first release (E2b: :29, :40, :16 reproduce the
    zs regions).
  - So P(b_goal | o, l) ~ P(b_goal | k(scene)) holds for the goal role, at region resolution.
- **Object role: NOT explained by scene exemplars, by instruction similarity, by type-level visual similarity, by exposure, or by
  source slot.** E1 gives -0.55, and E1b gives no account above baseline (2 effective tasks). The wrong-object errors look like
  instance-level confusions, e.g. :58 ketchup -> tomato sauce. E4 (type > instance resolution) is the place to test that.
- **P is measurable from early own-policy samples** (E2b: rank-predicts zs success, rho +0.76 [+0.44, +0.98]). The decomposition
  success = P(correct b) x competence is calibrated (slope 1.00), and it separates binding failures (:29, :40, :16, :38, :58, :83)
  from competence failures (:45, :32).
- **Pooled prediction (a) as first stated is falsified** (E1). The theory keeps (a) for the goal role only. The object role needs
  its own account before S3 (the exemplar-aware gate) can cover it.
