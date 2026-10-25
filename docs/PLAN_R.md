# Plan R: test-time recovery of compositional failures on a frozen VLA (reframe, 2026-10-24)

**Claim.** A frozen, off-the-shelf flow VLA that fails an undemonstrated combination can be recovered at test time:
- with no new demonstrations and no weight change;
- by retrieving parts from a bank extracted post hoc from the same frozen model, and composing them at the prefix K/V;
- with candidates searched and verified, and the corrector gated on predicted failure.

The prototype bank is the mechanism. The claim is test-time recovery.

**Why the gap exists** (user's novelty review, 2026-10-24; competitor details to be verified, section 7):
- Structure built in at training time is taken: AtomicVLA, SkillNet, OrthoSkillVLA, SkillMemo, HiMoE-VLA.
- REFACTOR-VLA extracts a library but retrains the decoder.
- CRAFT trains skill queries with a contrastive loss.
- The SAE line extracts from frozen models but loses closed-loop success when hooked.
- Nobody closes the loop: frozen pi0.5 -> extracted bank -> composition -> task success.
- Velocity-space composition is standard (RL²-VLA, CAG-VA), so composing is not the novelty; extraction plus search is.

## 1. What our evidence already says

| finding | source | consequence for R |
|---|---|---|
| zs failures are mostly binding, scene-driven, flat in CD | FM, b7fb0644 | the primitives exist; the instruction does not bind (the vision shortcut). Recovery by conditioning is plausible |
| action-space part modes are not readable | P2/P3, 5e65bc83 | compose at K/V conditioning, not at velocity |
| binding leaves the word positions by layer 2 | L0b | K/V tokens must be early-layer or full-prefix |
| a 74 KB K/V token from one demo removes binding errors, = LoRA on recombinations | M1, FM | the K/V level can carry the fix; the task is to get it WITHOUT the target demo |
| recombining stored sentences hurts (imports source objects) | M1, FM F5 | naive composition must be gated and verified |
| always-on composition hurts tasks zs already solves | G0, FM F5 | gate on a failure detector (matches RL²-VLA) |
| practice-selected sources beat hand-picked pairs | G10 | verified search works; R replaces practice with a cheaper verifier |

## 2. The method (frozen policy, no target demo)

1. **Bank extraction, post hoc, training demos only.**
   - Segment each training demo at gripper events: approach/grasp is the object role, carry/release the goal role.
   - Fit role K/V tokens, each shared across every training demo with that part, through the frozen VLA. No VLA weight
     changes.
   - Gate entry with REFACTOR-VLA's template: an MDL gain AND return preservation. A part token must keep the success of the
     training tasks it came from when served in place of their instruction.
   - Junk check: reject tokens whose effect is episode-specific (high variance across the demos that share the part).
2. **Hierarchical store.** Task > role (object / goal) > instance (qualifier), keyed by scene + words. Bytes grow with
   parts.
3. **Failure gate.** Run zs. Invoke recovery only when a failure detector fires. Candidates:
   - the policy's own flow-loss novelty (I6: AUC 1.0 on new concepts);
   - action-chunk variance;
   - a SAFE-style probe.
4. **Search over compositions, verified.**
   - Enumerate bank candidates for each role of the parsed instruction, plus composition coefficients and the slot
     (prefix / role position / pre- vs post-grasp schedule).
   - Score each candidate with a verifier, choosing the cheapest that works:
     - the frozen VLA's own consistency;
     - a VGAS-style value / geometric critic on action chunks;
     - a few practice rollouts (charged).
   - Execute the best candidate.
5. **Create only when nothing in the bank passes.** That is the demonstrated case, labelled as such: one demo yields one new
   part token, which joins the bank.

## 3. Composition-space comparison (de-risks 2)

The same bank and the same tasks, composed at three levels:
- **prefix K/V:** tokens / role slots;
- **velocity:** CFG-style or weighted, as in RL²-VLA and CAG-VA;
- **weight space:** task-vector or LoRA arithmetic (heed "Suppression sticks, locality is fragile").

Head to head on success, damage and bytes. Prediction from P2/P3 and M1: K/V wins.

## 4. Benchmarks

- **CRAFT's PICK-PLACE and PICK-PLACE-PRESS** (LIBERO-derived, demonstrated / undemonstrated splits; if released). Bar:
  CRAFT's 84.2 / 64.3 / 72.1 undemonstrated on pi0 / pi0.5 / GR00T, which needs training. A frozen, no-demo method near it
  is the headline.
- **LIBERO-90-C held-out recombinations** (ours; dev now, sealed once at the end).
- **Li's OOD suites; LIBERO-PRO's task split.**
- **Policies:** pi0.5 first, then pi0 / SmolVLA / X-VLA / GR00T for the "base model matters" axis.

## 5. Baselines

| baseline | role |
|---|---|
| zs | floor |
| **CAG-VA** (vision-only policy + CFG on language) | MANDATORY: the cheapest fix for a vision shortcut. If it gets most of the gain, that is the paper's biggest problem |
| Li TLI / our L2 composition | sentence-level composition |
| ReGuide | training-free frozen-pi0.5 rebinding |
| RL²-VLA-style gated velocity steering | composition with a trained second policy |
| CRAFT (their table; rerun on our split if needed) | training-time compositional SOTA |
| per-task token / per-task LoRA (one demo) | demonstrated ceilings, not competitors |

## 6. Failure atlas (the diagnosis half of the paper)

FM's typing, plus:
- the empty-instruction probe;
- "which demonstrated task did it execute instead" (does the episode complete a training combination that shares the object
  or the target).

Run across the policies, with the mechanism (L0b binding map, P2/P3) as the explanation. Systematicity tests in Fodor's
sense:
- X-on-W from X-on-Y + Z-on-W;
- the swap;
- substitutivity;
- productivity.

## 7. Probes, cheapest decisive first

| | question | go / kill |
|---|---|---|
| R0 | verify the competitor claims in this plan (REFACTOR-VLA, CRAFT benchmark release, CAG-VA, TTT-VLA, the task-vector audit, VGAS, RL²-VLA) | the related-work table is checked against sources |
| R1 | CAG-VA on pi05_l90c, on the 15 LIBERO-90-C recombination dev tasks + empty-instruction probe | measures the cheap-fix bar |
| R2 | bank from training demos only; oracle retrieval (correct object + goal tokens) zero-shot on the same 15 tasks | composed tokens close >= 50% of the zs (180) -> demo-token (234) gap; else kill R |
| R3 | + failure gate + verified search (unsupervised retrieval) | beats zs and CAG-VA, no damage on zs successes |
| R4 | composition-space comparison | K/V vs velocity vs weight |
| R5 | CRAFT benchmark + more policies; sealed read | pre-registered |

L4 (running) tests role tokens and stored-token reuse with one demo per arrival: a first read on whether part tokens are
reusable at all.

## 8. Corrections after R0 (docs/RELATED_R.md, ec7b776b)

- **ReGuide (2609.38616) already closes the frozen-pi0.5 loop with a training-demo bank.** Its bank is external and geometric:
  pose statistics, a grounding module and scripted motion. Our open ground is:
  - a bank from the model's own prefix K/V / action modes, composed inside its conditioning, with no grounding module;
  - verified, gated search;
  - the composition-space comparison.
  ReGuide is a MANDATORY baseline: perception variant headline, oracle variant marked privileged. Its pi0.5 numbers are
  39.0 -> 94.5 (oracle) and 87.5 (perception).
- **CRAFT's benchmarks and code are not released.** R5 instead authors a CRAFT-style colour-factorial cell (DESIGN 3.1 ii).
  CRAFT's undemonstrated order is pi0.5 / pi0 / GR00T = 84.2 / 64.3 / 72.1. CAG-VA in their table is 66.5 / 6.8 / 4.9.
- **CAG-VA:** Fang et al. 2602.17659, code at github.com/yuffish/LIBERO-CF (LIBERO-CF).
  - Its training-free variant drops the language to the same VLA; run it first.
  - The VA variant trains a language-free policy.
- **Verifiers:** VGAS's critic needs 5 target demos, so it is not free. REFACTOR-VLA's return gate is inert on LIBERO.
  Our gate is task success of the training tasks, which LIBERO provides.
- **Dropped or corrected:** LatentSkill is an LLM-agent paper. TTT-VLA retrains pi0.5. The SAE "hooking costs success"
  finding holds for OpenVLA, not pi0.5.

## 9. Verdicts R1, R2a, R2 (2026-10-24)

| probe | commit | result |
|---|---|---|
| R1 CAG-VA (training-free) | fe79413a | does not repair recombinations: w* 1.5 gives 173 vs zs 180 of 300; no w beats zs; damage -0.06 on training cells. The empty instruction completes a demonstrated task in 0.47 of episodes (vision shortcut real), yet only 38/400 zs failures replay one: zs mis-binds, it does not replay |
| R2a null-field modes | fd474579 | the named object is a null mode in 0.53 of pre-grasp observations (0.19 on zs wrong-object episodes); ranking given coverage is fine (0.93). Selection from the frozen field cannot fix binding |
| R2 training-demo role bank, oracle retrieval | 42822d08, 4eff744a | NO-GO: closes 0.21 of the zs -> demo-token gap (bar 0.5); bank 181 vs zs 170 vs ceiling 223 of 300; no damage. 9 of 16 tokens lower flow loss only on their own fit demos (held-out gain ~0): flow-matched part tokens memorise demo residual |
| R2c contrastive minimal-pair objective (offline stage 1) | e40fb257, (this commit) | STAGE 1 FAILS, no rollouts: held-out gain on its own part > 0 for 3 of 16 tokens (bar 8; R2 3 of 16 on the same split); specificity > 0 for 13 of 16, bought by raising partner loss (-2.6% to -17.2%), not by transfer. The extraction objective is not the bottleneck (docs/PLAN_R2C.md) |
| R3b attention-logit binding steering, oracle region (offline stage 1) | 68d20d3d, (this commit) | STAGE 1 FAILS, no rollouts: +b (2-8) on early-layer attention logits from object/goal-word or action-expert queries to the oracle region's image tokens is inert. Chosen expert L0-1 b 8: wrong-object named hit 0.077 -> 0.077 (bar +0.20), wrong-goal 0.131 -> 0.133; even a counterfactual target gains +0.001. Re-weighting where early queries look does not rewrite binding (docs/PLAN_R3B.md) |
| R3c region-wise K/V splicing from same-scene training instructions, oracle sources (offline stage 0) | (this commit) | STAGE 0 FAILS, no test, no rollouts: splicing the image K/V at the object's region (dilation 0/1, layers from 0/2/4) from a source naming a counterfactual object moves its hit by -0.005 (bar +0.13); splicing the WHOLE image moves it +0.44 (full prompt +0.50). Binding lives in the deep image K/V as a distributed scene code, not at the object's patches. Coverage: no wrong-object observation has a same-scene source (0/163) (docs/PLAN_R3C.md) |

**Consequence.** R as specified (flow-matched post-hoc part tokens in the K/V cache) is killed. The cheap fixes (language
guidance, selection among frozen modes) are also closed, which strengthens the atlas half: the zs failure is a binding error
inside the conditioning that neither the frozen field nor a guidance rule exposes.
