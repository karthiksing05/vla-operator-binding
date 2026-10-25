# Probe R3b: steer binding at its site with an attention-logit bias (written 2026-10-25, before any steered chunk was scored)

**Question.** zs on pi05_l90c@5000 fails the dev tasks mainly by binding: in R2a, wrong goal 152 and wrong object 56 of 580
episodes (docs/PLAN_R2A.md section 7). L0b (docs/PLAN_LANG.md) found that binding leaves the word positions by layer 2 and lives in
the image tokens. R1 (CAG-VA), R2a (null-mode selection), R2 and R2c (part tokens fitted on training demos) all failed to repair it
(docs/PLAN_R.md section 9). R3b asks whether binding can be steered at its site with no demo, no weight change and no new token:
bias the early-layer attention logits so the instruction's object or goal words, and/or the action expert's queries, attend to the
image tokens that cover the named object or goal region.

Code: `experiments/rs3_steer.py`. Pre-registration: `experiments/prereg/pr3b_steer.py` (ledger `m3_prereg`, tag `pr3b_steer`).
Ledger rows: `rs3_choice` (the configuration chosen on the tuning pool, recorded before any test observation is run) and `rs3_read`
(stage 1). Payloads: `results/rs3/` (split.json, choice.json and read.json are tracked; regions/, tune/, test/ and scratch/ stay local).

## 1. The intervention

- **Hook.** An opt-in forward pre-hook on every self-attention module of the VLM (prefix pass) and of the action expert (denoising
  pass). It adds +b to the additive 4-D attention mask from the chosen query rows to the chosen key columns, in the chosen layers.
  With no configuration it returns nothing, so the policy is unchanged. It lives in `rs3_steer.Steer`; no shared file is edited.
- **Sites.**
  - `word`: in the prefix pass, the query rows are the instruction tokens of the object phrase's content words (pre-grasp) or the
    goal phrase's content words (post-grasp). The parse is primitives.steps_of / step_key (M1's parse), and the words are mapped to
    sentencepiece tokens of the pi0-format prompt (BOS + text + "\n"; asserted equal to the policy's own tokens).
  - `expert`: in every denoising step, the query rows are the 10 action tokens of the action expert. Their keys at these layers are
    the prefix K/V cache plus the suffix.
  - `both`: the two at once.
- **Key columns.** The agentview image tokens (the first 256 prefix positions; 16 x 16 SigLIP patches of the 224 input) covering the
  target region.
- **Region v1 (ORACLE, privileged as ReGuide-oracle).** The simulator's segmentation at each observation: the task's start state,
  every movable object set to its recorded pose, the robot's geoms hidden, agentview at 256 in the client's flipped orientation
  (libero_client.seg_maps). A token is in an object's region if at least 5% of its pixels show the object. Fallbacks, used only
  when the mask is empty:
  - a disk of 24 px (1.5 tokens) around the object's projected position;
  - for a goal region (a BDDL site such as `basket_1_contain_region`), the parent object's mask inside that disk around the site's
    projected position, else the disk.
  The projection's orientation is chosen per task from the 8 axis flips/swaps by the median distance between projected body
  centres and mask centroids. The same transform wins on every task (asserted, median error < 30 px; body origins sit off
  the mask centroid by 1-19 px, next best transform 31-70 px).
  Fixture joints (a drawer's opening) stay at the start state; they are not recorded.
- **Region v2** (the frozen model's own word -> patch attention under the instruction vs the null prompt) runs only if v1 passes.

## 2. Stage 1 (offline, on R2a's stored observations)

At each observation, the policy draws 12 chunks with and without steering from the same noise (seed 11, R2a's `policy_sample`
seed; parity with R2a's stored draws is checked). R2a's readout scores each chunk: the integrated 10-step path, the first
candidate along its heading inside a 5 cm corridor, else the best cosine; < 2 cm points at nothing. A chunk's hit is pointing at
the named object (pre-grasp) or the own target (post-grasp). The observation's value is the share of its 12 chunks that hit.

**Sets** (episode types only, results/rs3/split.json; R2a's index and typing):
- tuning pool: the middle pre-grasp observation of every zs episode that neither succeeded nor was typed wrong_object (types
  wrong_goal, motion, no_commit, fixture);
- test:
  - `wo_pre`: every pre-grasp observation of the zs wrong_object episodes;
  - `su_pre`: every pre-grasp observation of the zs successes;
  - `wg_post`: the post-grasp observation of the zs wrong_goal episodes;
  - `su_post`: the post-grasp observation of the zs successes.

The tuning and test sets share no episode on the object role. The wrong_goal episodes give pre-grasp observations to the tuning
pool and post-grasp observations to `wg_post`, which are disjoint observations of a different role (disclosed).

**Grid** (27 configurations): site {word, expert, both} x layers {0-1, 0-3, 0-7} x b {2, 4, 8}.

**Selection rule** (pre-registered, tuning pool only). For each configuration:
- `g_named` = mean over observations of (steered-to-named hit share - unsteered hit share);
- `g_cf` = the same with a counterfactual target: one non-named movable object per observation, chosen by a hash of its key.
  The unsteered policy is not drawn to it, so `g_cf` measures how far the bias can move the action against the policy's own pull,
  which is the wrong-object situation;
- `still` = the rise in the share of chunks that move < 2 cm.

A configuration is admissible if `g_named >= -0.02` and `still <= +0.10`. The chosen one maximises `g_cf` among the admissible
ones. Ties within 0.005 go to the smaller b, then fewer layers, then site order word < expert < both. If none is admissible, the
maximiser of `g_named` is chosen. The same configuration serves the goal role (goal words, own-target region). The choice is
recorded (ledger `rs3_choice`) before any test observation is run.

**Arms on the test sets:**
- `u11`: unsteered, seed 11 (= R2a's policy_sample);
- `u21`: unsteered, seed 21, the readout's own noise floor;
- `s`: the chosen configuration toward the named object / own target, seed 11;
- `s_cf`: the chosen configuration toward the counterfactual target, a controllability read;
- `sweep_b*`: the chosen site and layers at the other two strengths (descriptive);
- R2a's arms on the same observations (ratio_top, coverage, instr / CAG modes, null mass, random, policy_exec), recomputed from
  its stored modes.

**Hypotheses.** The unit is the episode (mean over its observations). Paired differences s - u11 use the two-level task-cluster
bootstrap (stats.bootstrap_diff, B 10,000, seed 0). Holm runs over H1-H4 at alpha 0.05.
- **H1 (object, the gate).** On `wo_pre`, s - u11 >= 0.20 (point estimate), the 95% CI excludes 0, and the one-sided bootstrap p
  (share of differences <= 0) passes Holm.
- **H2 (object, no harm, the gate).** On `su_pre`, s - u11 >= -0.05: the one-sided p = the share of bootstrap differences <= -0.05
  passes Holm.
- **H3 (goal).** On `wg_post`, as H1.
- **H4 (goal, no harm).** On `su_post`, as H2.

**Stage-1 gate: H1 and H2 pass.** H3 / H4 decide whether the goal-phase steering joins stage 2. If the gate fails: stop, record,
commit.

Readout noise, reported with every rate: R2a's executed-window validity (0.645 own-bound agreement on pre-grasp, chance 0.379;
0.795 post-grasp, chance 0.272) and the u21 - u11 floor.

## 3. Stage 2 (only if stage 1 passes)

Closed loop on the 15 recombination dev tasks x 20 fresh_rb2 states, paired with R2's zs 170 and ceiling 223 of 300 (the pairing
verified from R2's roll files), plus damage on R2's 8 damage cells; fm_types typing; Holm over at most 4 hypotheses. GO: the
steered policy closes >= 0.5 of the zs -> ceiling gap. Its serving code is written only if stage 1 passes.

## 4. Cost (planned)

Regions: one EGL job. Tuning: about 250 observations x 55 arms of 12 draws, 4 A40 shards. Test: about 860 observations x 6 arms.

## 5. Result, stage 1 (2026-10-25; ledger `rs3_choice` choice, `rs3_read` stage1; results/rs3/choice.json, read.json)

**Choice on the tuning pool** (250 observations; unsteered named-object hit 0.715). The bias barely moves the action anywhere on
the grid:
- best g_cf +0.008 (expert, layers 0-3, b 8); every configuration lies within -0.006 to +0.008;
- g_named lies within -0.052 to +0.004. The four excluded configurations (word or both, layers 0-3 / 0-7, b 8) only lower it.
- The tie rule picked **expert, layers 0-1, b 8** (g_cf +0.006, g_named -0.002, still +0.001).
- Smoke scale: b 8 on the expert's layers 0-7 moves the unnormalised chunk by 0.027 on average; the word site at layers 0-1 moves it
  by 0.001-0.003.

**Test** (episode means, task-cluster bootstrap 95% CI; hit share of 12 chunks at the named object / own target):

| set | u11 (= R2a policy_sample) | s (steered) | s - u11 | noise floor u21 - u11 | s_cf - u11 at the cf target | R2a ratio_top / coverage |
|---|---|---|---|---|---|---|
| wo_pre (55 ep, 7 tasks) | 0.077 | 0.077 | -0.000 [-0.006, +0.007] | -0.005 [-0.020, +0.016] | +0.001 [-0.005, +0.007] | 0.073 / 0.194 |
| su_pre (132 ep, 11 tasks) | 0.606 | 0.603 | -0.004 [-0.009, +0.001] | -0.029 [-0.060, +0.004] | +0.001 | 0.540 / 0.558 |
| wg_post (144 ep, 14 tasks) | 0.131 | 0.133 | +0.002 [-0.002, +0.007] | -0.017 [-0.047, +0.004] | 0.000 | 0.056 / 0.076 |
| su_post (133 ep, 11 tasks) | 0.780 | 0.781 | +0.001 [-0.001, +0.005] | -0.019 [-0.041, -0.002] | 0.000 | 0.812 / 0.835 |

- The b sweep at the chosen site and layers (b 2 / 4) gives the same rates to within 0.002 on every set.
- Parity: u11 equals R2a's stored policy_sample draws exactly (max difference 0).
- Readout noise: R2a's validity is 0.645 pre-grasp (chance 0.379) and 0.795 post-grasp (chance 0.272). A second unsteered seed
  moves the rates by up to 0.03, which is ten times the steering effect.

**Pre-registered verdicts** (Holm over H1-H4):
- **H1 FAILS:** +0.000 against the bar of +0.20; p 0.52.
- **H2 passes**, but trivially: p 0.0 at margin -0.05.
- **H3 FAILS:** +0.002; p 0.26.
- **H4 passes**, again trivially.
- **STAGE-1 GATE FAILS. Stage 2 was not run.**

**Reading.**
1. **The intervention is inert, not mis-aimed.** The counterfactual arm is the test of whether an early-layer logit bias can move
   the action at all. Even toward an arbitrary object it gains nothing: +0.001 on wo_pre, +0.006 at best on the tuning pool.
   H2 and H4 pass only because nothing changes.
2. **The bias reaches the attention but not the action.** A +8 logit bias multiplies a region's attention weight by about 3,000.
   Even so, in the expert's early layers it shifts chunks by about 0.03, and on the word queries at layers 0-1 by about 0.002.
3. **This agrees with L0b.** The action expert reads binding from the image tokens' content across the deep layers (image K/V at
   layer 14 carries the most), not from where early queries look. The word's effect has already been written into the image tokens
   by layer 2. Re-weighting early attention toward the right patches does not rewrite which object those tokens encode as the
   instructed one.
4. Not tested: biases above 8, layers beyond 7, and bias on the image-token queries or the deep image K/V. Probing them would mean
   reopening the grid after a failed test. Region v2 was gated on v1 passing, so it was not built.

**Disclosures.**
- The projection-calibration assertion was relaxed twice before any policy ran (< 16 px, then "same transform everywhere, < 30 px").
  The projection only serves the fallback disks and the goal-site regions. The object regions are segmentation masks.
- Hand check: in libero_90:38, the right moka pot's region also holds a 3-4 token strip under the stove.
- M1's parse reads libero_90:16's object phrase as "black bowl at the front on the black bowl", so its object-word rows include the
  second bowl's words.
- The `still` column for the R2a arms in read.json is a placeholder 0 (R2a's arms carry no per-chunk motion); ignore it.
- The wrong_goal episodes give pre-grasp observations to the tuning pool and post-grasp observations to wg_post.
- During development, one python heredoc was attempted on the login node. The harness blocked it before it ran.
