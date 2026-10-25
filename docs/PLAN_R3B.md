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
