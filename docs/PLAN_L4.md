# Plan L4: a continual store of language prototypes (draft, 2026-10-22)

**Why this stage.** L2 (docs/PLAN_LANG.md, "L2 results") showed that a frozen VLA can be repaired for a new task by
*composing stored language states* (text latents of training sentences, word states), chosen from one demo, with no
gradient and no practice. On pi0.5 this beat zero-shot (91 vs 63 / 160) and matched Li's hand-picked pairs (84). That is
the end goal's core operation: an unsupervised combination of stored prototypes. L4 turns it into the full loop:
a hierarchical store that grows, is reused, and never damages the policy.

## 1. The store (hierarchical by granularity)

| level | prototype | source | what it carries (L0 trace) |
|---|---|---|---|
| sentence | text latent of a whole training or repaired sentence | training prompts; every accepted repair | the whole task: approach, carry, release |
| phrase / role | the object phrase or goal phrase state, keyed by role | spans of stored sentences | where the motion points while approaching (object) or carrying (goal) |
| word | a single word's layer-0..2 state | spans of stored sentences | the referent; L0 shows binding is decided at layers 0-1 |

Every prototype is executable: putting it in a prompt slot and re-evaluating the frozen policy at the current observation
gives a live action field. Prototypes are keys into the policy's own field, not stored actions.

## 2. Composition (the relation)

The operators measured in L2, used as the composition relation, cheapest first:
- **interpolate** two sentence prototypes (TLI, weight lambda);
- **transplant** a phrase or word prototype into a role of the new sentence (layers 1-k);
- **schedule** roles in time (object before the grasp, goal after; the P3 AND / THEN readout).

Selection is unsupervised: no labels, no rewards, no practice. One demo scores candidates by flow loss under a rule chosen
by L2b (the criterion study: demo-0 loss, split-demo validation, minimal-edit MDL penalty, zero-shot guard). The rule is
fixed once on dev tasks and then frozen.

## 3. The continual loop

For each arriving task (instruction + one demo):
1. **Compose.** Search candidates over the *current* store (training prototypes plus everything added so far).
2. **Accept or keep zero-shot**, by the frozen rule.
3. **Store.** An accepted composition becomes a new sentence prototype, and its role spans become new phrase prototypes,
   keyed by their words. Nothing in the policy changes, so old tasks cannot be damaged.
4. **Create only when composition fails** (L3): one inverted token for the failing word (about 2 KB). The rank-1 edit is
   dropped (L0b: no compact binding site).

## 4. What L4 must show

| claim | measure | pass |
|---|---|---|
| reuse | fraction of later tasks whose chosen candidate uses a prototype added by an earlier task; causal check by removing it | > 0, and removal lowers success |
| forward transfer | success of a task when it arrives late vs early in the stream (two mirrored orders) | late > early |
| no damage | training-task success with the full store | change >= -0.01 (expected 0, since the policy is frozen and the prompt for old tasks is unchanged) |
| cost | GPU-seconds and bytes per arrival vs per-task LoRA | <= 1/100 of LoRA bytes |

**Stream design.** Li's 20 OOD tasks plus LIBERO-90-C held-out cells (once pi05_l90c exists), in two mirrored random orders.
Order pairs so that a task sharing a part (an object or a goal) with a later task arrives first. That makes reuse possible,
and the mirrored order measures forward transfer.

## 5. Order of work

1. L2 confirm read (running): does the frozen l2sel rule hold on the 12 held-out tasks on pi0.5?
2. L2b (running): which offline criterion predicts closed-loop success, especially on pi0?
3. Pre-register L4 with the L2b rule, on pi0.5 first. L4 only adds the store-update step; it needs no new operators.
4. pi0 joins once a rule is found that does not collapse on pi0 in closed loop.

## 6. How it meets the end goal

- **hierarchical store:** sentence > phrase > word prototypes, all executable.
- **unsupervised composition:** one demo, flow-loss selection, no labels or rewards.
- **OOD efficiency:** no gradient, about the cost of scoring candidates on one demo.
- **continual:** the store grows by accepted compositions, the policy is frozen.
- **interpretable:** every repair names its source sentences, roles and layer span.

**Relation to plan v4.** Plan v4's action-space prototypes (modes under part prompts) await pi05_l90c. If they become
readable there, they enter this store as a fourth, action-level layer.

## 7. L4 as run: a continual store of created role tokens (written 2026-10-24, before any L4 number)

**Why the design changed.** M1 and FM (docs/PLAN_M1.md section 6, docs/PLAN_FM.md section 4) replaced the draft above on pi05_l90c.
Composing stored training sentences hurts (150 vs zs 180 of 580). One created 74 KB input-side KV token removes almost all binding
errors. An instruction-keyed token damages an old task with the same text (-0.075 routed), so keys must hold the scene. The unit of
the store is now a created token, and L4 asks whether created tokens are reusable and composable: does the store grow by parts?

Code: `experiments/l4_store.py`; serving through `lt:<spec>` with the additive `role_tokens` field (lifelong_ood/lang/transplant.py).
Pre-registration: `experiments/prereg/pl4_store.py` (ledger `m3_prereg`, tag `pl4_store`). Ledger rows: `l4_fit` (per fit),
`roll_record` rows of the packs (`l4_eval`, `l4_prac`, `l4_damage`), `l4_read`. Payloads: `results/l4/`.

### 7.1 The part: a role token (cheapest design that could work)

- **What it is.** A KV token (token_fold.KVToken) of n = 2 entries per layer (36,864 bytes fp16), appended to the instruction's
  cache in ONE phase of the episode. An O + G pair has exactly the bytes of M1's whole-task token (n = 4).
  - **O** is attached while the policy's own gripper is open (approach). FM: the wrong-object errors arise here.
  - **G** is attached while it carries (held_ops._Grasp: carrying after 3 closed queries, until the gripper opens). FM: the
    wrong-goal errors arise here.
- **Fit.** Both role tokens of a task are fitted JOINTLY on its one demo: every demo frame runs under its phase's token. The
  phase comes from the same _Grasp rule run offline on the demo's gripper command. M1's settings otherwise: flow matching, Adam
  lr 1e-2, batch 32, 1000 steps, seed 13; init from frame 0's language entries + 1% noise (G's noise seed + 1). A role whose phase
  holds fewer than 10 demo frames gets no token (the drawer-closing and drawer-opening demos never carry).
- **Keys** (the part a token stands for), per step of the instruction (M1's parse; qualifiers kept as content words):
  - O item: `O | verb | object words` (head + landmark).
  - G item: `G | preposition | goal words`. An articulation step has no goal, so its G item is `G | manip:verb | object words`.
  - A token's key is (scene, role, the set of items it was fitted for). It applies to any request in the same scene whose role
    items contain that set.
- **Composition** is concatenation: every stored token whose key is contained in a request's role items is appended in that
  role's phase. Example: 45 ("turn on the stove and put the frying pan on it") can take 44's stove tokens and 40's frying-pan
  O token together.

### 7.2 Arms (one demo per task; demos only fit and score, never executed or replayed)

| arm | served | new work |
|---|---|---|
| zs | the instruction | none |
| whole | M1's whole-task token (results/m1/fits, n 4, 1000 steps) | none here (M1's fit) |
| lora1000 | M1's per-task LoRA r16 (18.7 MB) | none here (M1's fit) |
| scratch | the task's own O + G pair, 1000 steps (snapshots 50 / 200 for the canonical targets) | one joint fit |
| reuse@e | earlier task e's stored role tokens whose key is contained in this task's items; nothing fitted | none |
| complete@e | reuse@e's tokens frozen + new tokens for the uncovered items (each new token keyed by the items it covers), 1000 steps | one fit, only the new tokens learn |
| causal@e | complete@e with the reused tokens removed | none |
| init@e | the task's pair initialised from e's pair, 200 steps (snapshot 50) | one short fit |
| cross@e | a same-text task e of ANOTHER scene: e's whole scratch pair (secondary: is the scene key needed?) | none |

**The reuse table** (results/l4/plan.json, from the parse; frozen before any fit):
- 25 within-scene reuse pairs. A pair e -> l means a role token of e has a key contained in l's role items.
- 22 pairs have headroom: l's M1 zs is at most 15 of 20 (fresh_m1, known before this plan).
- The **canonical partner** of each of the 13 headroom targets is the partner sharing the most items (ties: lowest
  sha1("l4partner:e>l")): 1<-0, 29<-32, 32<-29, 58<-59, 73<-75, 75<-73, 77<-80, 78<-79, 79<-78, 80<-77, 82<-83, 83<-82,
  45<-44.
- reuse runs on all 25 pairs; complete runs on the 22 headroom pairs; causal, init and the scratch snapshots run on the 13
  canonical pairs.
- 12 cross-scene same-text pairs: 7<->11; 73<->78; 75, 80, 83 among each other; 79<->82.

### 7.3 The stream (two mirrored orders)

- **Orders.** Order A sorts the 29 dev tasks by sha1("l4order:" + task); order B is A reversed.
  A = 80 61 44 1 40 75 82 36 58 11 83 79 50 60 38 45 59 76 7 9 16 77 14 29 32 22 73 78 0.
- **Arrival of task t** (instruction + one demo; the store starts empty; the policy stays frozen):
  1. Route by parts: every stored token of t's scene whose key is contained in t's role items -> the cover.
  2. The guard: zs and the cover each run 5 practice episodes, from a separate new practice set (fresh_l4p) on the new task. The
     policy's own rollouts are charged; the demo is never executed. The best (ties: zs, then the cover) passes if it succeeds
     in at least 3 of 5: serve it and fit nothing.
  3. Otherwise create:
     - COMPLETE if the cover is non-empty, at least as good as zs on practice, and leaves fit-able items uncovered;
     - else the SCRATCH pair.
     The fitted object runs 5 practice episodes.
  4. Serve the best of {zs, cover, fitted} on practice (ties in that order). Only a chosen fitted object stores tokens.
     - New tokens are stored as parts under (scene, role, the items each covers).
     - A key already held stays with its owner, and the new token goes into t's exception.
     - If routing t by parts would now serve something other than the choice, the choice is stored as an exception for
       (scene, instruction).
- **Readouts per order:**
  - success: eval rollouts of every served choice, 29 x 20;
  - choices and fits;
  - store bytes vs parts after each arrival;
  - GPU-seconds per arrival: practice episodes' wall seconds + the fit's A40 seconds from demo arrival, charged as if each
    practice episode held a GPU alone; against M1's per-task LoRA fit seconds.
- **Forward transfer.** A task is late in the order where more of its partners (pairs above) arrive before it. 16 tasks differ
  between the orders.
  FWT = its success when late minus its success when early. Episodes are paired by start state, pooled over the 16 tasks.
- **No damage.** At the end of each order, route by the final store:
  - every earlier dev task, rolled only where the routed object differs from its arrival choice (else the change is 0 by
    construction);
  - M1's 10 training cells, on stored init states 0-19, seed 0, against the parity read; rolled only where routing gives a token.
  - Parts can reach training cells: 41 (frying pan in KITCHEN_SCENE9), 4 (the top drawer in KITCHEN_SCENE10), 62 (the tray in
    LIVING_ROOM_SCENE4), 52 (the basket), 23 (the bottom drawer) and 37 (the white bowl).
  - The measure is the mean change over the 29 dev tasks and over the 10 cells.

### 7.4 Settings

- Policy: pi05_l90c main/5000 PyTorch, frozen; served as in M1 (plain Euler, 10 steps).
- Evaluation states: NEW `results/robot/fresh_l4`, LIBERO's sampler at seed base 6100000 + k, 20 per task.
- Practice states: NEW `fresh_l4p`, base 6300000, 5 per task.
- Neither base is within 1000 of any base in slurm/*.out; `--fresh-check` asserts every state is > 1e-6 from every other fresh set.
- Noise seed 13. 20 episodes per (task, served object), paired over start states. Recording as M1 (object poses, bound object).
- The whole-task token and LoRA come from M1's fits on the same demo; only their rollouts are new (on fresh_l4).
- FM's k = 5 arm is not run (not cheap enough for this study).

### 7.5 Hypotheses (pre-registered; Holm over H1-H5 at alpha 0.05)

- **H1, parts are as good as wholes.** scratch (the O + G pair) is non-inferior to whole (M1's token, same bytes) at margin 0.05
  on the 29 tasks. One-sided bootstrap p = the share of two-level cluster-bootstrap differences <= -0.05 (B 10,000).
- **H2, zero-shot reuse.** On the 13 canonical targets, reuse@partner > zs: more successes, exact two-sided sign test on the
  discordant pairs.
- **H3, the reused part carries the repair (causal).** On the 13 canonical targets, complete@partner > causal@partner (the reused
  tokens removed): more successes and the sign test.
- **H4, transfer of the fit.** On the 13 canonical targets, init@partner at 200 steps > scratch at 200 steps: more successes and
  the sign test.
- **H5, the continual stream.** It passes only if all three hold:
  - FWT > 0, pooled over both orders' late-vs-early pairs (more successes and the sign test);
  - the no-damage gate: in both orders, the mean routed change is >= -0.01 on the dev tasks and on the 10 training cells;
  - the store's final bytes are <= 1/100 of 29 per-task LoRAs, in both orders.

**Secondary, descriptive.**
- Success per arm, overall and combination / part.
- complete vs scratch (non-inferiority at 0.05, unadjusted).
- reuse vs zs on every pair, including the ceiling targets 59, 60 and 61, where reuse is a damage test.
- The cross-scene same-text reuse.
- Budget-to-threshold: the smallest snapshot reaching scratch@1000's pooled success on the canonical targets, for scratch
  (50 / 200 / 1000) and init (50 / 200).
- The stream's choices, fits, bytes-vs-parts curve, GPU-s per arrival vs LoRA, and exceptions.

**Disclosure.**
- M1's and FM's per-task success on fresh_m1 was read before this plan. It sets the headroom rule (M1 zs <= 15).
- No pi05_l90c rollout on fresh_l4 or fresh_l4p exists before L4.
- The serving smoke (2 episodes, stored init states) is excluded from every number.
- On 2026-10-24 one command that began with `python3` was refused on the login node before it ran (no python executed).

### 7.6 Cost (planned)

- Fits: 29 scratch, 22 complete, 13 init (about 64 A40 fits of 1-3 minutes, plus model loads).
- Rollouts: about 240 evaluation instances x 20 = 4,800 episodes, and about 100 practice instances x 5. Stream-only objects come
  on top, and the stream may need a few rounds.
- About 18 A40-h at 4 instances per A40.

## 8. Result (2026-10-24; ledger `l4_read` tag read, results/l4/read.json)

29 dev tasks x 20 fresh_l4 states (580 paired episodes per arm), pi05_l90c@5000, noise seed 13. 64 fits planned and 1 extra
(the stream's composite cover for 45), all completed. No code changed after the pre-registration row.

### 8.1 Parts vs wholes (fresh_l4)

| arm | success / 580 | combination (15) / 300 | part (14) / 280 | bytes per task |
|---|---|---|---|---|
| zs | 184 | 183 | 1 | 0 |
| whole (M1 token, n 4) | 333 | 221 | 112 | 73,728 |
| scratch (O + G role pair, n 2 each) | 349 | 232 | 117 | 73,728 |
| lora1000 (M1) | 375 | 230 | 145 | 18,710,528 |

**H1 FAILS (narrowly).** scratch - whole = +0.028 [-0.045, +0.100], discordant 77 to 61. Non-inferiority p 0.0196 against the Holm
threshold 0.0125. Splitting the token by phase costs nothing: the pair is level with or above the whole token at the same bytes.

### 8.2 Reuse and transfer (13 canonical headroom targets, 260 paired episodes)

| arm | success / 260 |
|---|---|
| zs | 13 |
| reuse (the partner's tokens, no fit) | 8 |
| causal (complete minus the reused tokens) | 65 |
| complete (reused frozen + new tokens) | 103 |
| scratch 50 / 200 / 1000 steps | 7 / 88 / 124 |
| init from the partner, 50 / 200 steps | 68 / 99 |

- **H2 FAILS.** reuse 8 vs zs 13, discordant 7 to 12, diff -0.019 [-0.112, +0.058].
  - A stored part never repairs a new task zero-shot: 0 of 20 on 11 of the 13 targets.
  - On the zs-ceiling targets it damages: 61 goes from 20 to 3 under 60's tray G token, and 60 from 19 to 6 under 61's. 59 stays
    at 19 under 58's.
- **H3 PASSES** (sign p 6.3e-5 against the threshold 0.01). complete 103 vs causal 65, discordant 63 to 25. The cluster bootstrap
  still includes 0: diff +0.146 [-0.004, +0.300].
  - The effect is in 5 targets: 29 (9 vs 0), 73 (10 vs 0), 45 (10 vs 0), 1 (16 vs 5), 58 (17 vs 11).
  - It reverses on 77, 79 and 80 (causal above complete).
  - Reading: the new tokens are fitted together with the reused ones and come to depend on them. This shows co-adaptation. It
    does not show that the reused part alone carries the skill (H2 says it does not).
- **Growth by parts costs success.** complete 103 vs scratch 124, diff -0.081 [-0.231, +0.077], non-inferiority p 0.67
  (secondary). Over all 22 headroom pairs, complete beats scratch where the target's own scratch pair is weak: 45 (16 and 10 vs
  7), 58 (17 vs 10), 82 (12 vs 5), 1 (16 vs 14). It loses or ties on most book tasks (e.g. 80: 1 to 3 vs 7; 78: 1 to 7 vs 10).
- **H4 FAILS.** init@200 99 vs scratch@200 88, diff +0.042 [-0.077, +0.158], sign p 0.25.
  - Init does speed the start: init@50 68 vs scratch@50 7.
  - Budget-to-threshold (scratch@1000's 124): scratch needs 1000 steps; init reaches it at neither 50 nor 200.
- **Cross-scene, same text** (secondary): another scene's whole pair mostly fails (0 to 7 of 20 on 9 of 12 pairs). Two pairs
  transfer: 11 -> 7 (16 vs scratch 0) and 82 -> 79 (14 vs 15). The scene key is mostly needed.

### 8.3 The stream (two mirrored orders)

| | order A | order B |
|---|---|---|
| success / 580 (served choices) | 313 | 338 |
| choices: zs / reuse / complete / scratch | 13 / 1 / 3 / 12 | 11 / 0 / 5 / 13 |
| fits run: complete / scratch | 7 / 13 | 6 / 14 |
| parts stored / exceptions | 26 / 4 | 30 / 1 |
| final store bytes | 958,536 | 1,105,938 |
| vs 29 per-task LoRAs (542.6 MB) | 1/566 | 1/491 |
| vs 29 whole tokens (2.14 MB) | 0.45 | 0.52 |
| mean A40-s per arrival: fit only / with practice | 96 / 514 | 98 / 498 |
| routed damage: dev tasks / 10 training cells | 0.000 / -0.045 | -0.014 / -0.045 |

- Reference (same states): per-task LoRA 375 at 145 A40-s of fit per task; scratch for every task 349; zs 184.
- **H5 FAILS on every condition but bytes.**
  - FWT is negative: late 133 vs early 182 of 320, discordant 16 to 65, diff -0.153 [-0.269, -0.031].
    - Late arrivals take the complete path or fall back to zs when the 5-episode guard rejects everything. Examples: 78 1 vs 10,
      80 0 vs 7, 83 0 vs 7, 45 0 vs 7, 29 9 vs 20.
    - Only 82 gains (12 vs 5).
  - Damage:
    - training cell 41 ("put the frying pan on top of the cabinet") takes 40's frying-pan O token by part routing and drops
      by 0.45, in both orders;
    - in order B, dev task 45 drops by 0.35 once 40's and 44's later parts reach it, and 59 by 0.05.
- **The guard is the weak link.** 5 practice episodes cost about 400 of the 500 A40-s per arrival and still pick poorly. Several
  arrivals fitted a token, scored 0 of 5 on practice, and kept zs (A: 83, 45, 77, 32, 7; B: 80, 7). Those tasks were left at
  zs's level (0 to 1 of 20). Meanwhile 1-of-5 practice scores accepted scratch pairs that reached 4 to 14 of 20.

### 8.4 Reading

1. **A created token splits cleanly by role.** The O + G pair fitted jointly by phase matches the whole token (349 vs 333) at the
   same bytes. The phase structure (approach vs carry) is a valid factorisation of what the token repairs.
2. **The split parts are not reusable on their own.** An object token from task A does nothing for task B with the same object
   in the same scene (zero-shot 8 vs 13), and a goal token from another task damages tasks zs already solves. The token is a
   joint repair of this task's binding, not a lexical item for "book" or "tray".
3. **Composition by concatenation works only as a fitting scaffold.** Freezing a reused part and fitting the rest gives tokens
   that depend on the part (H3). They are worse than fitting from scratch (103 vs 124), and the init transfer is too small to
   test as positive (H4).
4. **The continual store is cheap but not yet safe or forward-transferring.** 1/500 of LoRA bytes and two thirds of LoRA's fit
   time. But part routing leaks onto a training cell (-0.45 on cell 41), and late arrivals do worse than early ones. Keys by
   (scene, role words) are too coarse for created tokens: the binding they repair is task-specific.
5. **For the end goal:** the token remains the unit that works per task (M1, FM, here). Growing a hierarchical store by parts
   needs one of two things: parts fitted to be reusable (e.g. an object token fitted across several tasks that share the
   object, with an anchor on tasks that do not), or routing that checks a part against the task (a stronger guard than 5
   practice episodes) before it applies.

**Disclosures.**
- No code changed after the pre-registration row.
- The serving smoke (L4_SMOKE, 2 episodes from the stored init states of task 45, 20-step tokens) wrote one ledger row
  (`l4_smoke`). It is excluded from every number.
- The stream ran in rounds: a driver re-ran `--launch-stream` every 10 minutes, and each round decided arrivals until a missing
  practice, fit or eval. Every decision used practice results only. The evaluation rollouts of served choices were read only by
  `--read`.
- The stream's decisions are computed from a table of served objects (each a fixed token set on the frozen policy), so they are
  identical to a sequential run. The practice states are the same 5 per task for every candidate.
- Task 16's parse reads its goal as "in the middle" (the rightmost preposition). No other dev task shares a part with it, so no
  pair depends on it.
- On 2026-10-24 one shell command beginning with `python3` was refused on the login node before it ran.
