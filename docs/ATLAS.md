# Failure atlas: compositional zero-shot in a flow VLA (synthesis, 2026-10-25)

Policy: pi05_l90c@5000, our pi0.5 trained on the LIBERO-90-C training cells only, so held-out combinations are unseen at the
weights. Dev set: 29 held-out tasks (15 recombinations of held parts, 14 with a never-seen part), 20 fresh start states each.
The sealed third is untouched. Every number below is ledger-backed in the commit cited.

## 1. How zero-shot fails

| finding | number | source |
|---|---|---|
| zs solves recombinations, never new parts | 180 / 300 recombination, 0 / 280 part | M1, bdb324b1 |
| failures are mostly binding errors | 215 of 339 typed failures are wrong-object or wrong-goal | FM, b7fb0644 |
| binding errors do not grow with compound divergence; they are worst in new scenes | flat across CD 0.1 / 0.3 / 0.5 | FM |
| the scene alone drives demonstrated behaviour (vision shortcut) | empty instruction completes a same-scene training task in 0.466 [0.313, 0.626] of dev episodes; its own task in 99 / 200 training-cell episodes | R1, fe79413a |
| zs failures are misbindings, not replays | only 38 / 400 zs failures complete a demonstrated task, nearly all from two tasks | R1 |
| the right object is usually not among the frozen policy's own options | named object is a null-field mode in 0.534 of pre-grasp observations, 0.19 on wrong-object episodes | R2a, fd474579 |
| binding leaves the word positions by layer 2 and lives in the image tokens | restoring word positions recovers at most 0.25 from layer 2 on | L0b, docs/PLAN_LANG.md |
| action-space part modes are not readable | attribution about 1.8x, transfer about 0, R0 at most 0.60 | P2/P3, 5e65bc83 |
| the approach IS object-frame on training tasks, but the part prompt does not carry it | leave-scene-out object-frame transfer pi05_libero 0.62 (world -0.07, control -0.03; pre-reg GO) but null prompt also 0.62; pi05_l90c part 0.23 / instr 0.41 (fail); yaw adds nothing; P2's ~0 was its first-frame data (its own statistic pooled on approach frames 0.66-0.79) | OF D1, docs/PLAN_OF.md |
| the vision tower finds the named object even where zs misbinds; the words do not | SigLIP prototype anchor hit 0.62 on wrong-object vs 0.68 on successes (chance 0.04; gate passes); object-word attention (layers 3-12) 0.22-0.46 vs 0.65-0.89; identity leave-scene-out 0.80 / 0.65 (gate 0.8 fails narrowly), drops 0.1 far from the usual position | OF D2/D3, docs/PLAN_OF.md |

**Reading.** The policy reads the instruction holistically, keyed on the scene. By layer 2 the instruction has been written
into the scene's image tokens with the wrong filler, and the action expert reads the misbinding from there.

## 2. What does not repair it (no target demo, no weight change)

| intervention | result | why | source |
|---|---|---|---|
| CAG-VA, training-free language guidance | w* 1.5: 173 vs 180; no w beats zs; damage -0.06 on training cells | amplifies the wrong binding; high w turns wrong-goal into no-commit | R1 |
| select among frozen null modes by instruction/null ratio | ratio loses to the policy 10:28 on wrong-object episodes | coverage, not ranking, is the bottleneck (ranking given coverage 0.93) | R2a |
| recombine stored training sentences (TLI-style) | compose_sel 150 vs zs 180 | imports source objects | M1, FM F5 |
| bank of part tokens fitted on training demos, oracle retrieval | 181 vs zs 170 vs ceiling 223 (closure 0.21, bar 0.5); no damage | tokens fit their own demos; held-out gain about 0 (3 of 16 positive) | R2, 4eff744a |
| same bank with a contrastive minimal-pair objective | held-out own-part gain positive for 3 of 16 | specificity only by damaging partners; the unit, not the loss, fails | R2c, 5bdaaf16 |
| early-layer attention steering to the oracle object region | wrong-object hit 0.077 -> 0.077 (bar +0.20); counterfactual target +0.001 | the misbinding already lives in the image tokens; where early queries look does not rewrite it | R3b, 12a9c42e |
| splice the model's own image K/V at the object's region from a same-scene training instruction (oracle source) | counterfactual hit -0.005 (bar +0.13); whole-image splice +0.44, full prompt +0.50 | binding is a distributed scene code in the deep image K/V, not local to the object's patches; no wrong-object case has a same-scene source (0 / 163) | R3c, docs/PLAN_R3C.md |
| guide the policy's own flow sampler toward a target pose (endpoint x0-projection with the OF D1 operator; oracle targets in the stage-0 test, vision anchor never reached) | all-step guidance moves chunks to an arbitrary object (+0.38 to +0.80; bar +0.13) but raises the flow-loss outlier share by +0.29 to +0.80 (bar 0.10); on-manifold settings move it by at most +0.06 | the action expert treats an unbound target as out of distribution; its last denoising steps pull the chunk back to the scene-bound choice | R4-bind, docs/PLAN_R4.md |
| reuse role tokens from earlier tasks with no fit | 8 vs zs 13 of 260 | per-task parts are not reusable words | L4, 75deff22 |

## 3. What does repair it (one demo of the target task)

| arm | recombination / 300 | part / 280 | bytes | fit |
|---|---|---|---|---|
| zs | 180 | 0 | 0 | none |
| one-demo K/V token | 234 | 117 | 73,728 | 123 s |
| one-demo LoRA r16, 1000 steps | 238 | 149 | 18.7 MB | 145 s |

- The token removes binding errors: 215 -> 34 (k = 1) -> 9 (k = 5); it saturates at k = 5 (token 446, LoRA 485 of 580), and
  the remaining gap is motion on new parts (FM).
- Routed by instruction text it damages a same-text training cell (-0.075); keyed by scene it does not (M1, L4).

## 4. One explanation for every failure

The policy does not hold object and goal as separable pieces. It reads the instruction holistically against the scene and
writes the result into a scene-wide code in the deep image tokens (layers 10-17; R3c: whole-image splice +0.44 of the full
prompt's +0.50, region splice -0.005). Each repair fails for a consequence of this:
- **no local handle:** attention steering (R3b) and region splicing (R3c) are inert;
- **no correct option inside the frozen model:** the named object is rarely a null mode (R2a) and guidance scales the wrong
  binding (R1);
- **no transferable part:** tokens fitted on training demos capture those demos, not the part (R2, R2c, L4);
- **no own correct state to borrow:** wrong-object failures occur exactly where no training cell binds that object in that
  scene (0 / 163, R3c).

The one-demo token works because it is fitted on the target task: it supplies the scene-specific binding directly. That is
demonstrated adaptation, not composition.

## 5. Open choice (for the user)

The demo-free recovery claim of Plan R is not supported on this policy. Candidate papers: (a) this atlas plus the one-demo
token as the cheap fix; (b) the atlas as a negative result on demo-free compositional recovery; (c) change training so that
binding becomes local, then retest demo-free composition (tens of A40-h).
