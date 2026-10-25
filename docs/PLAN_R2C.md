# Probe R2c: a contrastive minimal-pair objective for R2's role tokens (written 2026-10-24, before any R2c number)

**Question.** R2 (docs/PLAN_R2.md section 7) found that role tokens flow-matched on pooled training demos lower the flow loss only on
their own fit demos: the held-out gain is about zero and 9 of 16 tokens are flagged as junk. Under oracle retrieval they closed only
0.21 of the zs -> demo-token gap on the 15 recombination dev tasks. The hypothesis is that the objective lets a token absorb
demo-specific residual. R2c changes ONLY the extraction objective, so that a token has to carry the part: it must lower the loss on
demos that have the part AND must not lower (it should raise) the loss on training demos of the same scene whose role holds a
different filler.

Code: `experiments/rc2_contrast.py`. Pre-registration: `experiments/prereg/pr2c_contrast.py` (ledger `m3_prereg`, tag
`pr2c_contrast`). Ledger rows: `rc2_fit` (per token), `rc2_stage1`. Payloads: `results/rc2/` (plan.json and stage1.json tracked;
fits/ ignored).

## 1. What stays R2's

- The 16 part keys, the key rule and each key's sources (`results/rb2/plan.json`, hashed in the pre-registration row).
- Each key's fit demos and fit frames (40 demos, at most 1,024 role-phase frames, each frame under its own training instruction).
- The token unit (KVToken, n = 2 entries per layer, 36,864 bytes fp16), its init (frame 0's language entries + 1% noise, G's noise
  seed + 1), and the frozen pi05_l90c@5000.
- The fit settings: Adam lr 1e-2, batch 32, 1000 steps, seed 13. The p-side batch draws use R2's exact streams (rng / generator /
  torch seed 13), so the p batches are R2's; the contrast batch has its own seed-14 streams.
- Serving and everything downstream (stage 2) reuse `experiments/rb2_bank.py`.

## 2. The objective

Per step, R2's batch of 32 p frames plus a batch of 32 role-phase frames from contrast-partner demos (each under its own
instruction):

    L = FM(tok | p) / Lbar0_p  +  LAMBDA * relu(MARGIN - (FM(tok | c) - FM(none | c)) / FM(none | c))

- `Lbar0_p` is the no-token flow loss on the fit frames (fixed draws, computed once before the fit). It puts the p term in relative
  units.
- `FM(none | c)` is the no-token loss on the SAME contrast batch and draw (no gradient). The contrast term is therefore a paired
  relative change, and its noise does not swamp a 1-2% effect.
- **LAMBDA = 1.** The two halves of a minimal pair weigh the same, which is the definition of a minimal-pair contrast. Both terms are
  in relative units, so 1 is the neutral weight.
- **MARGIN = 0.02.** A token that carries "go to X" should make a demo that goes to Y less likely, so it must raise the contrast
  loss, not merely leave it alone. The hinge stops pushing once the rise reaches 2% of the no-token loss. That is the low end of
  the relative gain R2's tokens reached on their own fit demos (1.7% to 7.5%). The contrast requirement is then of the same size
  as what the fit term already obtains, and the hinge cannot dominate. This number was read from R2's published junk check (disclosed).

## 3. Minimal pairs (training data only; `--plan`, no outcome)

A partner of key p (role r) is a training text that does NOT hold p, holds another filler in role r, and shares a scene (the
factors' scene of its training cells) with a source text of p.
- **Tier A:** the partner's other-role key set equals the source's, so only role r's filler differs (a minimal pair).
- **Tier B:** same scene, role r's filler differs, the other role may differ too. Used only when a key has no tier-A partner.
- **None:** no partner exists. The key keeps R2's pooled token (same objective, data and seed, not refitted). It is reported
  separately and counts as failing the specificity criterion.

Contrast fit set: 40 partner demos spread over the partners (max(3, ceil(40 / partners)) per text, evenly spaced, held-out
excluded), their role-phase frames subsampled to at most 1,024: R2's p-side budget.

Coverage (results/rc2/plan.json):

| tier | n | keys |
|---|---|---|
| A | 7 | `G|on|cabinet shelf` (vs "under the cabinet shelf"), `G|on|plate` (vs bowl on the stove / on top of the cabinet), `O|alphabet soup`, `O|bowl white`, `O|chocolate pudding`, `O|frying pan`, `O|sauce tomato` |
| B | 6 | `G|manip:close|cabinet drawer top`, `O|black bowl`, `O|black bowl front`, `O|black bowl middle`, `O|bottom cabinet drawer`, `O|cabinet drawer top` |
| none | 3 | `G|in|basket`, `G|in|tray`, `G|in|cabinet drawer top`: every training text in their scenes has the same goal |

## 4. Held-out split and the stage-1 gate

- **Split** (fixed, seeded by sha1 of the key): 16 held-out p demos from the source texts' episodes R2 never fitted (round robin over
  sources), and 16 held-out partner demos that R2c never fits. At most 24 role-phase frames per demo; 4 fixed draws per frame.
  - `O|black bowl front` has only 2 unfitted p demos (its single source has few episodes beyond R2's 40). This is disclosed, not
    padded.
- **Both tokens** (R2's, loaded from results/rb2/fits, and R2c's) are scored on identical frames and draws, from their fp16 files.
- **Per token:** gain_p = mean over held-out p demos of (loss without - loss with) / loss without; gain_c = the same on held-out
  partner demos; spec = gain_p - gain_c.
- **STAGE 1 GATE (pre-registered):** gain_p > 0 for >= 8 of 16 R2c tokens AND spec > 0 for >= 8 of 16 R2c tokens (the 3 no-partner
  keys fail spec by rule). If it fails: STOP, record, commit, no rollouts.
- Descriptive: the same numbers for R2's tokens on the identical split, the count of keys where R2c beats R2, the training curve
  (fit loss, contrast rise, hinge).

## 5. Stage 2 (only if stage 1 passes)

- Arm `bank_rc2`: the instruction + oracle O tokens (open phase) + G tokens (carrying phase) of the R2c bank. These are the same
  oracle keys, phase schedule and serving as R2's `bank`.
- 15 recombination dev tasks x 20 fresh_rb2 states, noise seed 13 (R2's states and seed, so the episodes pair with R2's zs 170 /
  bank 181 / ceiling 223 rows; the pairing is verified from the rollout settings before the read). Also the damage read on R2's 8
  damage cells (stored init states 0-19, seed 0, vs parity). Failure typing with fm_types.
- Hypotheses (Holm over H1-H4, alpha 0.05):
  - **H1 (GO):** closure >= 0.5. D = bank_rc2 - 0.5 zs - 0.5 ceiling > 0, one-sided two-level cluster bootstrap.
  - **H2:** bank_rc2 > zs, exact sign test on discordant pairs.
  - **H3:** bank_rc2 > R2's bank, exact sign test on discordant pairs.
  - **H4:** no damage. Non-inferior to parity at margin 0.05.
- Stage-2 serving code is written only if the gate passes, and an amendment row is recorded before any stage-2 rollout.

## 6. Rules and disclosure (before the pre-registration row)

- No dev-task demo is read anywhere; partners and sources are training texts of lol/libero_l90c. results/l90c/split_sealed.json is
  never opened. Demos are only fitted and scored, never executed or replayed.
- Seen before this plan: R2's published results (section 7 of PLAN_R2), which motivated the probe and set MARGIN; the rc2 `--plan`
  output (partners, tiers, counts; no loss, no outcome).
- Deviations from R2's settings: none in the fit. The held-out scoring uses 4 draws (R2's junk check used 2) and 16 + 16 demos (R2's
  used at most 8 p demos). R2's tokens are rescored on the new split, so the comparison uses the same draws and frames.
- A pipeline smoke (RC2_SMOKE=1: 20 steps on `O|frying pan`, results/rc2/smoke, 32 held-out frames, no ledger row, held-out scores
  written but not printed or read) ran before the pre-registration row. It is excluded from every number. Its training log showed the
  contrast batch's loss rising by 2.7% at step 20 (hinge 0): the init token already raises the partners' loss, so the hinge binds only
  if training drifts the token toward lowering it. LAMBDA and MARGIN were not changed after this.

## 7. Result: stage 1 FAILS, R2c stops (2026-10-25; ledger `rc2_stage1`, results/rc2/stage1.json; pre-registration m3_prereg `pr2c_contrast`, e40fb257)

16 fits (13 contrastive; 3 no-partner keys keep R2's token), each scored with R2's token on the identical held-out frames and draws.
8,946 A40-s in all (bank build + 1000 steps + scoring). No code changed after the pre-registration row. No rollout ran.

Held-out relative flow-loss gain, in % (gain_p on p demos, gain_c on partner demos, spec = gain_p - gain_c):

| key | tier | R2 gain_p | R2 gain_c | R2 spec | R2c gain_p | R2c gain_c | R2c spec |
|---|---|---|---|---|---|---|---|
| `G|in|basket` | none | -5.79 | | | -5.79 | | |
| `G|in|cabinet drawer top` | none | -1.27 | | | -1.27 | | |
| `G|in|tray` | none | -1.43 | | | -1.43 | | |
| `G|manip:close|cabinet drawer top` | B | -2.64 | -2.68 | +0.04 | -2.33 | -13.09 | +10.76 |
| `G|on|cabinet shelf` | A | -0.54 | -1.07 | +0.53 | -1.45 | -6.32 | +4.87 |
| `G|on|plate` | A | -4.64 | -1.78 | -2.87 | -4.16 | -4.53 | +0.37 |
| `O|alphabet soup` | A | -0.36 | +0.75 | -1.11 | -1.58 | -2.58 | +1.00 |
| `O|black bowl` | B | +0.26 | -0.95 | +1.21 | +0.45 | -6.37 | +6.82 |
| `O|black bowl front` | B | -4.98 | -1.38 | -3.60 | -4.64 | -6.67 | +2.03 |
| `O|black bowl middle` | B | -0.70 | -0.60 | -0.09 | -0.81 | -6.73 | +5.93 |
| `O|bottom cabinet drawer` | B | +0.63 | -0.75 | +1.38 | +0.44 | -7.48 | +7.92 |
| `O|bowl white` | A | -0.47 | -1.49 | +1.03 | -0.97 | -17.15 | +16.18 |
| `O|cabinet drawer top` | B | -2.10 | -3.28 | +1.18 | -2.29 | -7.77 | +5.49 |
| `O|chocolate pudding` | A | -1.85 | -3.76 | +1.92 | -2.20 | -4.64 | +2.44 |
| `O|frying pan` | A | -0.08 | -0.08 | +0.00 | -0.64 | -11.97 | +11.33 |
| `O|sauce tomato` | A | +1.44 | -1.40 | +2.84 | +0.31 | -4.01 | +4.32 |

**Gate verdict: FAIL.**
- gain_p > 0 on 3 of 16 R2c tokens (bar 8): `O|black bowl`, `O|bottom cabinet drawer`, `O|sauce tomato`. The median gain_p is -1.44%
  (R2's -0.98%).
- spec > 0 on 13 of 16 (bar 8; every partnered key). R2's tokens reach 9 of 16 on the same split.
- R2c beats R2 on gain_p for 4 of 16 keys and on spec for 13 of 16.
- Per the pre-registration, stage 2 does not run: no rollout, no H1-H4.

**Reading.**
1. **The contrastive objective buys specificity by damage, not by transfer.** Every partnered R2c token raises the partners' held-out
   loss (gain_c -2.6% to -17.2%). Its gain on held-out demos of its own part stays negative for 10 of 13, no better than R2's.
   The token becomes something that hurts other fillers. It does not become something that helps unseen demos of its filler.
2. **The hinge was mostly slack.** The contrast loss rose 4% to 24% by step 1000 with the hinge at 0. It was active briefly on 5
   keys near step 200, and at init the token already raised the partners' loss (smoke, section 6). So the p term alone drives a
   token away from the partners. The objective here differs from R2's in only that small pushback. This means R2's failure is not
   a token that helps everything in the scene. A token that lowers the loss on held-out demos of its own part does not exist in
   this unit for most parts, with or without the contrast.
3. **On a larger held-out split, R2's own tokens are worse than R2's junk check showed.** gain_p > 0 holds for 3 of 16, against R2's
   9-of-16 junk flag on at most 8 held-out demos. `O|black bowl` (+3.5% in R2's check) is +0.26% here.
4. **For PLAN_R.** The extraction objective is not the bottleneck. Two objectives (pooled flow matching, and the same plus a
   minimal-pair contrast) both give tokens that do not lower the loss on unseen demos of their part. Pi05_l90c already fits its
   training demos, so a 2-entry K/V token fitted there has no transferable residual to carry. The bank half of R needs a different
   unit or a different signal (for example the policy's own features at the binding site, L0b), not a different loss on the same
   unit.

**Disclosures.**
- No code changed after the pre-registration row. The ledger rows carry dirty=true: other agents' uncommitted files are in the tree.
  The rc2 files were committed at e40fb257 before the row.
- The 3 no-partner keys were scored, not refitted. Their R2c token is R2's file, so their R2 and R2c numbers are identical by
  construction.
- `O|black bowl front` has only 2 held-out p demos (section 4).
- The held-out scoring uses 4 draws and 16 + 16 demos (R2's junk check used 2 draws and at most 8 demos). R2's tokens were rescored
  on this split.
- The pipeline smoke (section 6) ran before the row and is excluded.
