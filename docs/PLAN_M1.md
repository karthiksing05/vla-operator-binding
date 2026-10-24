# Plan M1: compose, create, or fine-tune? (LIBERO-90-C dev, pi05_l90c@5000; written 2026-10-24, before any M1 number)

**Question.** A new task arrives with its instruction and ONE demo. Three ways to get it working without touching the policy
weights for old tasks:
- **compose** stored language prototypes (pi05_l90c's own training-cell instructions) by text-latent interpolation (TLI);
- **create** a new prototype: a small KV token fitted on the demo through the frozen VLA;
- **fine-tune** a per-task LoRA on the demo (the standard baseline).

Which is best per byte, per GPU-second and per unit of damage? This is the comparison docs/PLAN_L4.md's continual store needs:
L4 composes first and creates only when composition fails (L3 item 1, token inversion; Gal et al. 2208.01618). Action-space part
prototypes failed on this checkpoint (results/l90c/STATUS.md, 2026-10-24 08:55), so the store holds language prototypes only.

Code: `experiments/m1_compose_create.py`. Pre-registration: `experiments/prereg/pm1_compose_create.py` (ledger `m3_prereg`, tag
`pm1_compose_create`), recorded before any fit, rollout or read. Ledger rows: `m1_store`, `m1_fit` (per task), `m1_main` /
`m1_damage` (rollout packs, via roll_record), `m1_read`. Payloads: `results/m1/`.

## 1. Set-up

| item | choice |
|---|---|
| policy | pi05_l90c main/5000, PyTorch export `checkpoints_l90c/pytorch/pi05_l90c_5000`, frozen; served as in c_parity (MODE ours = plain Euler, 10 steps) |
| tasks | the 29 DEV held-out tasks of results/l90c/split.json (bins 0.1 / 0.3 / 0.5: 5 each; part set: 14). split_sealed.json is never opened; every task passes `l90c.check_task` |
| the demo | the task's own-scene IPEC episode 0 (`l90c.task_episodes(tid)[0]`, a LIBERO teleop demo). Used only to score and to fit; never executed or replayed |
| start states | NEW set `results/robot/fresh_m1`: LIBERO's own sampler at seed BASE 5700000 + k, 20 per task (no logged fresh-states job used a base within 1000 of it; `--fresh-check` asserts every state is > 1e-6 from every earlier libero_90 fresh set) |
| episodes | 20 per (task, arm), noise seed 13, paired over start states; arms whose served form is identical share one rollout |
| hardware | every fit on one A40 (gres a40:1), one task per job, model load excluded from the timings |

## 2. Arms (the same one demo for every arm)

| arm | what is served | per-task work after the demo arrives |
|---|---|---|
| zs | the instruction | none |
| compose_cov | the instruction + TLI between two training instructions picked by **demo-free role coverage**, lambda 24 | parse + retrieval (CPU, milliseconds); the latents are already in the store |
| compose_sel | **demo-scored**: the demo's flow loss picks among zs and {blind cover pair, scene-keyed cover pair} x lambda {24, 12, 36}; zs is kept unless a composition scores strictly lower | 36 demo frames x 4 fixed draws per candidate (pl2's DemoScorer, K 4, seed 1) |
| create | the instruction + a KV token (token_fold.KVToken, 4 entries per layer, 73.7 KB fp16) appended to the prompt's cache | prefix cache of every demo frame + 1000 Adam steps (lr 1e-2, batch 32), only the token learns |
| cc (compose-or-create) | compose_sel's conditioning + a token fitted ON TOP of it (each frame's prefix under its query's TLI coefficient), initialised from the composed prompt's own language entries; equals create where compose_sel kept zs | compose_sel + the same token fit |
| lora50 / lora200 / lora1000 | per-task LoRA r16 on every Linear of the action expert (r3_fewshot's one-demo recipe: lr 1e-4, batch 32, openpi's flow loss) | ONE 1000-step run, snapshots at steps 50 and 200: the fit curve |

**The store** (built once, before any task arrives; charged separately): the TLI latent of each of pi05_l90c's 73 training-cell
instructions, identified with Li's recipe (`tli.identify`: layer-input residual at the first 9 instruction tokens, frames 0-119 of
the first 20 demos) by pi05_l90c on its own training data (`lol/libero_l90c`). 663,552 bytes each in fp16 (48.4 MB in all).

**The coverage rule** (pl2c's, generalised; fixed in code before any number):
- parse every step (`bind_hooks.split_conj`, "it" resolved) into (V, verb) by `primitives.step_key`, the object phrase split at its
  first relation word into head (O) and landmark (L), and the goal phrase (G: preposition + target). Articulation steps ("close
  the top drawer of the cabinet") have no G; their object phrase is the fixture.
- content words drop transplant.STOP **minus the instance qualifiers {top, front, back, left, right, middle}**. pl2c dropped them,
  but on LIBERO-90-C they carry held-out compounds (e.g. t:wooden_cabinet:top) and the caddy's compartments.
- pool: every training instruction except one identical to the new one (two dev texts equal a training text: dev 0 = cell 28,
  dev 50 = cell 46). Greedy same-role coverage, ties by total cover, same frame, word Jaccard, lower index; 2 sources; t1 = the
  object-leaning pick (Li's ramp gives t1 the approach).
- scene key (L2c reading; L2d rule): the same coverage restricted to training instructions of the task's own scene, if it holds
  at least 2 (else blind). The STUDY_SCENE1-3 and KITCHEN_SCENE8 tasks fall back to blind.

The frozen design table (pairs, scenes, damage cells) is `results/m1/plan.json` (`--plan`), sha256 in the pre-registration.

## 3. Measures

- **Success** per arm: 29 x 20 episodes; rate with the two-level task-cluster bootstrap (lifelong_ood/stats.bootstrap_rate);
  paired comparisons with the cluster-bootstrap difference (stats.bootstrap_diff, B 10,000) and the exact sign test on
  discordant pairs. By bin (0.1 / 0.3 / 0.5 / part).
- **GPU-seconds from demo arrival to ready**, per arm, on an A40: demo decode + (prefix cache) + scoring / fitting, synchronised.
  LoRA's are the elapsed time to each snapshot. The store build is reported separately.
- **Bytes stored per task**: token 73.7 KB; LoRA r16 (adapt.save, fp16); compose stores a pointer (two prototype ids and lambda);
  the served spec materialises the difference T1 - T2 (663 KB), recorded as well.
- **Budget-to-threshold**: the smallest LoRA budget in {50, 200, 1000} whose pooled success reaches zs, compose_sel, create and cc,
  and its GPU-seconds.
- **Damage** on 10 training cells (stored init states 0-19, seed 0, paired with the parity read of pi05_l90c@5000):
  - *routed* (the store as served: an arm's repair is used when the instruction matches the stored task's): the policy is frozen,
    so only training cells whose instruction equals a dev task's can change. Both such cells are in the set (28 <- dev 0,
    46 <- dev 50) and are rolled with compose_sel, create, cc and lora1000 of their dev task. The other 8 cells are the base
    policy exactly (change 0 by construction).
  - *unrouted* leaks: on all 10 cells, the paired dev task's lora1000 always on, and its create token appended to the OLD prompt.
  - cells: the two collision cells, then one training cell per remaining dev scene (lowest sha1 rank), each paired with its
    scene's dev task of lowest sha1 rank: 28/0, 46/50, 17/14, 52/50, 41/45, 37/36, 6/7, 4/0, 23/22, 62/61.

## 4. Hypotheses (pre-registered; Holm over H1-H3 at alpha 0.05)

- **H1** compose_sel > zs: more successes and the exact two-sided sign test on discordant pairs.
- **H2** create is non-inferior to lora1000 at margin 0.05 (one-sided bootstrap p = the share of bootstrap differences <= -0.05)
  AND its bytes are <= 1/100 of the LoRA's (deterministic).
- **H3** cc is non-inferior to create at margin 0.05 (same test).
- **H4** (an estimate gate, outside Holm) the routed damage of compose_sel and of create, mean change over the 10 cells, is >= -0.01.

Secondary, descriptive: compose_cov vs zs; every arm vs zs; cc vs create and vs lora1000 (sign test); the LoRA curve; unrouted
damage; GPU-s; bytes; the selection counts (zs / blind / scene).

**Disclosure.** No pi05_l90c rollout on any dev task exists before M1. The P2 / P3 field probes on pi05_l90c read offline
observations of 14 dev tasks (no rollout, no success). The serving smoke (2 episodes, the stored init states, not the fresh set)
is excluded from every number.

## 5. Cost (planned)

Store build about 2 A40-h; fits about 29 x 12 min = 6 A40-h; rollouts about 29 x 7 unique arms x 20 + 560 damage episodes, about
4,600 episodes at 4 per A40, about 15 A40-h.

## 6. Result (2026-10-24; ledger `m1_read` tag read, results/m1/read.json)

29 dev tasks x 20 fresh states (580 paired episodes per arm), pi05_l90c@5000, noise seed 13.

| arm | success / 580 | combination tasks (15) / 300 | part tasks (14) / 280 | fit s (A40) | bytes |
|---|---|---|---|---|---|
| zs | 180 | 180 | 0 | 0 | 0 |
| compose_cov (demo-free) | 132 | 132 | 0 | 0.05 | 0 |
| compose_sel (demo-scored, zs guard) | 150 | 150 | 0 | 15 | 0 |
| create (KV token) | 351 | 234 | 117 | 123 | 73,728 |
| cc (compose then token) | 350 | 232 | 118 | 137 | 73,728 |
| lora50 / lora200 / lora1000 | 226 / 369 / 387 | 145 / 234 / 238 | 38 / 135 / 149 | 17 / 37 / 145 | 18,710,528 |

By bin, the combination tasks (zs / compose_sel / create / lora1000): CD 0.1: 80 / 60 / 86 / 93. CD 0.3: 43 / 32 / 75 / 84.
CD 0.5: 57 / 58 / 73 / 61.

**Pre-registered verdicts (Holm over H1-H3): none pass.**
- **H1 FAILS, in the wrong direction.** compose_sel 150 vs zs 180 (discordant 8 to 38), diff -0.052 [-0.112, +0.000]. The
  demo's flow loss chose a composition on 18 of 29 tasks (blind 14, scene 4), and those compositions lowered success. This
  repeats L2b: demo flow loss is a weak selector.
- **H2 FAILS.** create 351 vs lora1000 387, diff -0.062 [-0.157, +0.031], non-inferiority p 0.61. The bytes condition holds
  (ratio 1/254). On the 15 combination tasks create is level with LoRA: 234 vs 238 (lora200 also 234). The gap is on new
  parts: 117 vs 149.
- **H3 FAILS** (p 0.049 against the Holm threshold 0.017). cc 350 vs create 351, diff -0.002 [-0.059, +0.057]. Composing first
  adds nothing.
- **H4 FAILS for create.** Routed damage is compose_sel -0.005 and create -0.075 (cc -0.14, lora1000 -0.025), measured on the
  instruction-collision cells: an instruction-keyed token is applied to an old task with the same text.
  - Unrouted leaks: lora -0.475, token -0.55.
  - So repairs must be keyed by more than the instruction (scene key).

**Reading.**
1. Composing training sentences in the text channel does not repair pi05_l90c's held-out combinations; it hurts them. Its
   own policy reads combinations at CD 0.5 as well as at 0.1 (zs 57 vs 80 of 100), and every one of the 14 part tasks is
   0 for zs and for both compose arms. As predicted, no composition covers a new part.
2. A single 74 KB input-side token (no weight change) matches a 18.7 MB LoRA on recombinations, at about equal fit time
   (123 s vs 145 s). It trails on new parts, where the motion itself is new.
3. The repair that works lives in the language channel's input (a created prompt token), not in a recombination of stored
   sentences.

**Disclosures.**
- The script changed after the pre-registration row: a smoke-only M1_FORCE switch, compose_cov added to smoke-serve, and
  the recording flags. No rule, arm or test changed.
- The read first crashed because `stats.bootstrap_diff` did not return the bootstrap samples its docstring promised.
  Fixed with an opt-in `return_samples` flag; the test is unchanged.
