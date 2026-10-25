# Probe R2: a bank of role tokens from training demos only, oracle retrieval (written 2026-10-24, before any R2 number)

**Question** (docs/PLAN_R.md section 7, row R2). Can role tokens extracted post hoc from TRAINING demos only, through the frozen
pi05_l90c@5000, recover LIBERO-90-C's 15 recombination dev tasks when retrieval is an oracle (the parse picks the object-role and
goal-role tokens)? No target-task demo is used anywhere.

**Go / kill** (PLAN_R table). The composed bank tokens close at least 50% of the gap between zs and the demo-token ceiling (M1: zs
180 -> create 234 of 300 on these 15 tasks). Otherwise R is killed in this form.

**Why this design.**
- L4 (docs/PLAN_L4.md section 8): a role token fitted on one task's demo is not reusable (reuse 8 vs zs 13 of 260). It memorises
  that task's binding.
- R2a (docs/PLAN_R2A.md section 7): the frozen field does not afford the right binding where zs fails, so the repair must be
  created in the conditioning.
- R2 creates the repair from training data, pooled across every training demo that shares the part. If a reusable part exists in
  the conditioning, pooling should find it; one-task memorisation cannot survive pooling over texts and scenes.

Code: `experiments/rb2_bank.py`. Pre-registration: `experiments/prereg/pr2_bank.py` (ledger `m3_prereg`, tag `pr2_bank`).
Ledger rows: `rb2_fit` (per token), `rb2_read`. Payloads: `results/rb2/` (plan.json, fresh_check.json, read.json and types/ are
tracked; fits/, specs/, roll/, packs/, smoke/ are ignored).

## 1. The bank unit: a role token per part key

- **Token**: L4's role token, a token_fold.KVToken of n = 2 entries per layer (36,864 bytes fp16). It is appended to the
  instruction's cache in one phase of the episode:
  - O while the policy's own gripper is open (approach);
  - G while it carries (held_ops._Grasp: carrying after 3 closed queries, until the gripper opens).
- **Part keys**, from L4's parse (`l4_store.roles`, M1's content words, qualifiers kept):
  - `O|<object content words>`: the verb is dropped, so the key names what the approach goes to. "open / close the top drawer"
    and "put X in the top drawer and close it" share `O|cabinet drawer top`.
  - `G|<prep class>|<goal content words>`: on / on top of / onto -> on; in / into / inside -> in; other prepositions literal.
  - `G|manip:<verb>|<object content words>`: an articulation step's goal role (L4's item; open and close stay distinct).
  - Keys match by exact equality of the word sets.
  - This rule was chosen from the parse alone (experiments/rb2_bank.py `--inspect`, a print of which training texts hold each
    dev item). With L4's verb-keyed items, 5 of the 15 tasks had an uncovered object role. No outcome was looked at.
- **Training data**: pi05_l90c's own training set (`lol/libero_l90c`: 73 texts of the 76 training cells, about 50 demos each).
  A key's sources are every training text whose parse holds it, pooled across scenes. Scene-keyed tokens are not run here:
  the 3 texts shared across scenes cannot be split by scene in this dataset without a scene map, so the variant is not cheap.
- **Fit subset** (fixed in plan.json; cost bound):
  - 40 demos per key, spread evenly over its sources: max(3, ceil(40 / sources)) per source, evenly spaced over the source's
    episodes, excluding its last 2;
  - every frame of the key's role phase in those demos (L4.phases, the _Grasp rule run offline on the demo's gripper command),
    subsampled evenly to at most 1,024 frames;
  - each frame is conditioned on its own training instruction.
- **Fit**: L4's `fit_roles` with M1's settings (flow matching, Adam lr 1e-2, batch 32, 1000 steps, seed 13).
  - Init: frame 0's language entries + 1% noise (G's noise seed + 1).
  - Each key is fitted alone: only its own token is attached on its frames. In a two-step training text, the other step's
    token is not attached, while serving attaches every token of the request's role. This mismatch is disclosed, not fixed.
  - A key whose pooled phase holds fewer than 50 frames gets no token.
- **The bank** (plan.json): 17 tokens, 10 O and 7 G, 626,688 bytes.
  - Not fittable: `G|manip:close|bottom cabinet drawer` (its only source never carries).
  - No source: `O|black bowl left`, `G|in|middle` (task 16's parse reads its goal as "in the middle"; L4 disclosure),
    `G|on|cabinet drawer top`, `G|manip:open|cabinet drawer top`.
- **Oracle retrieval** per dev task: the tokens of every key its parse holds. Every one of the 15 tasks gets at least one token.
  - 4 tasks get O only: 16, 22, 7, 11.
  - 1 task gets G only: 60.
  - Task 1 lacks its "on top of it" goal.
- **Composition** is L4's concatenation: all of a request's O tokens in the open phase, all of its G tokens in the carrying phase.

## 2. Entry gate and junk check (PLAN_R 2.1)

- **Return preservation (the gate).** A greedy set cover of the 17 keys by training cells that have a 20-trial parity baseline
  (most uncovered keys first; ties: lowest sha1("rb2gate:" + cell)). It gives 10 gate cells:
  - libero_90:5, 13, 55, 41, 25, 88, 37, 15 and 6;
  - libero_object:5.
  Each gate cell is served with its instruction plus every bank token whose key its parse holds, on stored init states 0-19,
  noise seed 0. This is the parity read's protocol (results/l90c/parity_pi05_l90c_5000_*).
  - **Tolerance (pre-registered):** a gate cell passes if its success is at least parity - 0.15 (3 of 20 episodes).
  - The keys assigned to a failing cell are dropped from the **admitted bank**. The bank arms and the damage read use the
    admitted bank only.
  - Serving "in place of" the instruction is not run. The bank is served WITH the instruction, as on the dev tasks.
- **Junk check (descriptive; flags, does not drop).** Per demo, the flow-loss gain = loss without the token minus loss with it,
  on the demo's role-phase frames (2 fixed draws per frame). It is reported on:
  - the fit demos;
  - up to 8 held-out demos of the same training texts (each source's last 2 episodes, never fitted).
  Statistics: mean, sd, CV and the share of demos with a positive gain. A token is flagged as junk if under half of its held-out
  demos gain.

## 3. Arms (15 recombination dev tasks x 20 NEW start states, noise seed 13, paired)

| arm | served | demo used |
|---|---|---|
| zs | the instruction | none |
| bank | instruction + the oracle O tokens (open phase) + G tokens (carrying phase) of the admitted bank | training demos only |
| bank_O | instruction + the oracle O tokens only | training demos only |
| bank_G | instruction + the oracle G tokens only | training demos only |
| ceiling | instruction + M1's whole-task create token (results/m1/fits/*/create.json, n 4, fitted on the task's own demo) | the target demo (M1's fit; the demonstrated ceiling) |

- Arms with an identical served form share one rollout. For example, bank_G of an O-only task is zs.
- **States**: NEW `results/robot/fresh_rb2`, LIBERO's sampler at seed BASE 7500000 + k, 20 per task.
  - No logged base is within 1000 of it. The logged bases include M1 5700000, L4 6100000 / 6300000, R1 6700000 and R2a 7100000.
  - `--fresh-check` asserts every state is > 1e-6 from every state of every other fresh set (including fresh_r1 and fresh_r2a).
- **Damage read**: 8 further training cells that hold a bank key and have a parity baseline, chosen by lowest
  sha1("rb2dmg:" + cell), served with the admitted bank routed by parts. Each is compared with its parity read:
  - libero_spatial:6 and 7;
  - libero_90:70, 72, 52 and 42;
  - libero_goal:8;
  - libero_object:7.
- **Failure typing**: every dev episode of every arm is typed by experiments/fm_types.py (`type_episode`, with the scene's start
  states taken from fresh_rb2). The types are no_commit, wrong_object, wrong_goal, motion and fixture. Binding errors are
  wrong_object + wrong_goal.
- Recording as M1 (object poses, bound object). Demos are only fitted and scored, never executed or replayed. Dev-task demos are
  not read by any bank arm.

## 4. Hypotheses (pre-registered; Holm over H1-H4 at alpha 0.05)

- **H1, the go criterion: gap closure >= 0.5.**
  - The contrast is D = bank - 0.5 zs - 0.5 ceiling: the pooled mean over the 300 paired episodes, which pair by start state
    across the three arms.
  - The test is D > 0, with a one-sided two-level cluster bootstrap (tasks, then states; B 10,000, seed 0). p is the share of
    bootstrap D <= 0.
  - The point closure is (bank - zs) / (ceiling - zs) on fresh_rb2, with a bootstrap percentile CI.
  - If ceiling <= zs on fresh_rb2, H1 is not testable and fails.
  - Also reported: the closure against M1's reference gap (234 - 180 on fresh_m1).
  - **GO = H1 passes.**
- **H2.** bank > zs: more successes, and the exact two-sided sign test on discordant pairs.
- **H3, binding errors drop.** Per paired episode, the indicator of a binding error (wrong_object or wrong_goal). bank has fewer
  than zs, by the exact two-sided sign test on discordant pairs.
- **H4, no damage.** On the 8 damage cells, the routed admitted bank is non-inferior to parity at margin 0.05. One-sided
  cluster-bootstrap p = the share of differences <= -0.05.

**Secondary (descriptive).**
- Success per arm, overall and by bin.
- bank_O and bank_G vs zs, which say which role carries any gain.
- bank vs ceiling.
- Failure-type counts per arm.
- The gate table, junk statistics, bytes, and fit A40-seconds.

## 5. Settings and cost (planned)

- Policy: pi05_l90c main/5000 PyTorch, frozen; served as in M1 (plain Euler, 10 steps).
- Fits: 17 jobs on A40s. Each builds a CPU-held PrefixBank of 1,024 frames (about 27 GB) and runs 1000 steps.
- Rollouts: about 90 instances x 20 episodes (dev arms, 10 gate cells, 8 damage cells), at 4 instances per A40 pack, on overcap.

## 6. Disclosure (before the pre-registration row)

- Seen before this plan:
  - M1's 15-task totals (zs 180, create 234), which define the go criterion;
  - M1, FM, L4 and R2a results as published in their docs.
- `--inspect` printed the parse of the 15 tasks and the training texts holding each item (no outcome).
- `--plan` printed the bank, gate and damage tables (no outcome).
- A fit smoke (RB2_SMOKE=1: 20 steps on `G|in|tray`, results/rb2/smoke, no ledger row, no rollout) ran before the
  pre-registration to check the pipeline. It is excluded from every number.
- The fit job takes `--kid` (a hash of the key), because the GPU sbatch script splits SCRIPT on spaces. This is a pipeline
  change only.
