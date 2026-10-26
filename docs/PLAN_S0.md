# PLAN_SOLVE S0: binding adapter + oracle binding, the dissociation test (pre-registered 2026-10-26, before the adapter is trained)

From PLAN_SOLVE sections 4 and 7: the user approved a once-trained adapter and about 20 A40-h.

Ledger: m3_prereg tag `ps0`; read row `s0_adapter` tag read. split_sealed.json is never opened. Demos are FITTED (flow loss) and
SET in the simulator for rendering and masks. No demo is executed.

## 1. Adapter (experiments/s0_adapter.py)

**Locus: R3c's.** An additive edit of the agentview image tokens' cached keys / values at prefix layers 10-17. The frozen action
expert reads that cache.

**Form.** The edit is scene-wide:
- each image position's edit depends on its own (k, v), the two role masks at that position, and each role's mask-pooled (k, v);
- each layer has its own 2-layer MLP (3 x 512 + 2 -> 256 -> 512); 8.4 MB fp16 in total;
- the edit is defined as f(binding) - f(no binding), so zero masks give an exactly zero edit.

**Masks.** Oracle role masks over the 16 x 16 agentview tokens:
- object role: the named instance's segmentation;
- goal role: the own target's region box (S1's rule) or its body segmentation.

## 2. Training (experiments/s0_data.py, s0_acts.py, s0_train.py)

**Data.**
- 56 training cells with an In / On goal and a matched demo episode (the D1 / D4 / D5 match files).
- Every 3rd frame: the matched simulator state is SET and rendered as the policy sees it.
- The masks come from the sim; the target is the demo's 10-step action chunk.
- 2268 training frames. 595 frames (frame % 15 == 0) are held for the loss analysis.

**Examples.**
- **own:** the cell's prompt + its oracle masks -> its chunk.
- **cf (counterfactual):** a same-scene training cell's prompt, which names a different binding (libero_90 / 10 / goal scenes
  only), + THIS cell's masks -> its chunk. Half of each batch is cf where a partner exists.

**Loss and settings.**
- The policy's own flow-matching loss (openpi's).
- The policy is frozen. The prefix is computed without gradient; the gradient reaches the adapter through the action expert.
- AdamW 3e-4, wd 0.01, batch 16, 3000 steps, seed 0.
- Checkpoints every 250 steps (overcap, --requeue).

**Loss analysis** (held-out frames, fixed noise; descriptive). Mean flow loss for:
- own prompt: base, adapter, adapter with zero masks;
- cf prompt: base, adapter + own masks, adapter + token-permuted masks;
- adapter + the PARTNER's masks, which agree with the prompt and disagree with the demo.

The adapter is working if adapter + own masks beats base on cf prompts, and both the permuted and the partner masks raise the loss.

## 3. Closed-loop test (experiments/s0_serve.py, s0_sandbox.py, s0_pack.sbatch)

E2b's sandbox, unchanged in states, seeds, budget and labels:
- dev: 24 tasks x fresh_m1 states 0-4 x 8 seeds = 960 runs per arm, 400 steps, replan 5;
- masks are computed live from the sim at every query.

**Arms:**
- **zs:** no masks, the same server.
- **oracle:** the named object + own target.
- **wrong:** the first other movable that is not the target, plus the first other scene candidate. Does the policy follow the mask?
- **zero:** parity, on 24 units.
- **training-cell damage:** 10 training cells (:13, :17, :25, :26, :42, :43, :46, :53, :63, :65) x 5 fresh sampler states
  (seed 9400000 + k) x 8 seeds, zs vs oracle.

**Per-run outcome** (E2b's labels):
- **success.**
- **binding error:** a failure in which a movable other than the named one is bound, OR the named object's first release
  satisfies In / On for a non-own candidate.
- **motion failure:** every other failure.

## 4. Bars (pre-declared; PASS iff all three)

- **B1 binding:** the oracle binding-error rate is <= 0.5 x zs's (point estimate), AND the paired task-bootstrap difference has a
  95% CI upper bound < 0.
- **B2 dissociation:** the motion-failure rate's relative reduction (zs - oracle) / zs is <= half of B1's relative reduction.
- **B3 damage:** training-cell success with oracle masks is >= zs - 0.05 (point estimate).

**Reported:**
- the wrong-mask follow rates vs zs (binding the wrong-mask object; releasing at the wrong-mask goal);
- zero-mask parity;
- success oracle - zs;
- per-task tables.

**Pre-stated expectation.** Seventy percent of zs's dev failures are binding failures, concentrated on :29 / :40 / :16 / :38 / :58
/ :83 (E2b). If the adapter delivers a binding, those fall and :45 / :32 stay (motion). A null is possible: R1-R10 found no
working delivery site without new information. The adapter is the first module that adds the information.

## 5. Cost (planned)

- Training: about 1.1 A40-h.
- Rollouts: 3 full dev arms + zero subset + damage, about 4-6 A40-h on overcap.
- Total: under 10 A40-h.
