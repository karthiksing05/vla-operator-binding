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
