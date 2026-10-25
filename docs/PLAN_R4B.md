# Probe R4b: R4-bind with a non-circular on-manifold judge (written 2026-10-25, before any R4b statistic was computed)

Follows R4-bind (docs/PLAN_R4.md, experiments/rb4_bind.py, results/rb4/stage0.json; commits 43b97d14 and e8a45e38).
Code: `experiments/rb4b_bind.py`. Pre-registration: `experiments/prereg/pr4b_gate.py` (ledger `m3_prereg`, tag `pr4b_gate`).
Ledger rows: `rb4b_stage0` (stage 0) and, only if stage 0 passes, `rb4b_read` (stage 1). Payloads: `results/rb4b/` (sources.json and
stage0.json are tracked; judge/ and test/ stay local).

## 1. Why R4b exists

R4-bind's stage 0 judged a guided chunk "on-manifold" by its flow-matching loss under the policy's CURRENT conditioning: the
scene plus the instruction. The atlas (docs/ATLAS.md) shows that this conditioning is misbound. A correct approach to an object
the policy did not bind is high-loss under a misbinding by definition, so the check penalises the intended fix. It is circular.
All-step x0-projection guidance did move chunks strongly (g_cf +0.38 to +0.80 toward a counterfactual object), with flow-loss
outlier rises of +0.29 to +0.80 under the current conditioning (bar 0.10).

R4b keeps EVERYTHING else of R4-bind fixed: the guidance form, the 12-configuration grid, the tuning pool (R3b's 250
observations), the counterfactual target Y (rs3_steer.cf_pick), the noise floor, the movability bar g_cf >= 0.130, the
secondary bars, the admissibility rule and the tie rule. It replaces only the on-manifold judge.

**Disclosed:** the R4-bind stage-0 numbers under the old judge are already known (docs/PLAN_R4.md section 8; movability, still,
action range, g_named and the old flow-loss outlier rise for every configuration). Movability cannot change, because R4b
re-judges the same stored chunks. Only the flow-loss check is new.

## 2. The new judge (declared before any R4b number)

For each observation and each judged arm, the target is Y (the `|cf` arm, the movability arm that R4-bind's gate used).
- **Primary: a conditioning that binds the target.** The prompt is replaced by a same-scene training instruction that names Y in
  the object role, on the same images and state. The source follows R3c's oracle rule (rs4_splice.sources, the "training-cell
  parse"): a `train_l90` cell of the target's scene, with a different instruction, whose BDDL In/On goal atom moves Y. If several
  qualify, the one with the lowest sha1 of `rs4src:{task}:O:{Y}:{cell}` is used. For R3c's named objects this rule reproduces
  results/rs4/sources.json exactly (asserted as a cross-check). No outcome enters the choice.
- **Reference distribution:** the policy's own UNGUIDED chunks under that same binding conditioning: 12 chunks at seed 11 (ref11)
  and 12 at seed 21 (ref21), sampled by the policy's own Euler sampler (step for step R4-bind's u11 sampler).
- **Flow loss:** R4-bind's estimator unchanged (t in {0.1, 0.2, 0.3, 0.5} x 4 shared noises, seed 5), with the binding prefix.
- **Outlier:** a chunk whose loss exceeds the mean + 2 sd of ref11's 12 losses at that observation. The check statistic is the
  episode-mean outlier share of the guided `|cf` chunks minus that of ref21 (task-cluster bootstrap CI reported). This mirrors
  R4-bind (threshold from one own-sample seed, rise over a second seed).
- **Fallback where no binding source exists: the null-prompt field.** The same construction with the empty prompt "" (the
  scene alone, R2a's null field): references at seeds 11 / 21 under the null prompt, the loss under the null prompt.
- **Gate statistic:** per observation, the primary judge where Y has a binding source and the null fallback elsewhere, pooled over
  the tuning pool. The coverage of each judge is reported, and so is the rise within each group.
- **Secondary (as R4-bind, bars unchanged):** the |normalised action| > 1 share (the offline joint-limit stand-in) rises by
  <= 0.05 over u11, and the still-chunk (< 2 cm) share rises by <= 0.10 over u11.
- **Descriptive only:** the null judge on every observation; the `|named` arm under the binding of the named object (R3c's source
  for the named object nearest the hand, else another named object's, else the null prompt); the naive post-hoc edit under the
  new judge; u11 / u21 under the new judge; the share of the binding references' own chunks that head to Y.

**Bars (as R4-bind):** outlier rise <= 0.10, out-of-range rise <= 0.05, still rise <= 0.10.
Admissible = movable (g_cf >= 0.130) and all three bars and g_named >= -0.02 and still(named) <= +0.10.
**Choice rule (unchanged):** the admissible configuration with the largest g_cf; ties within 0.005 go to the smaller w, then
`early` before `all`, then `d1` before `dir`. Recorded as `rb4b_stage0` before any test observation is run.
Stage 0 passes if some configuration is admissible. If none is: stop, record, commit, no test run.

## 3. Reuse of R4-bind's chunks

R4-bind saved every tuning arm's 12 chunks (results/rb4/tune/*.npz) in unnormalised action units. The policy's output transform
is affine per entry on the 70 real entries (quantile normalisation, no clipping), so the normalised rows are recovered exactly
by its inverse. Asserted per observation: the round trip (re-unnormalise the inverted rows) is within 1e-4 action units of the
stored chunk. Also asserted on 2 observations per shard: the stored instruction-prefix flow losses are reproduced from the
inverted rows. Movability, still, action range and g_named are recomputed from the stored chunks and must equal R4-bind's
stage0.json exactly (asserted). No guided chunk is regenerated.

## 4. Stage 1 (only if stage 0 passes; R4-bind's pre-registered stage 1, unchanged except the judge)

R4-bind section 4 verbatim: R3b's test sets (`wo_pre`, `su_pre`, `wg_post`, `su_post`), arms u11, u21, s_vis (vision anchor:
SigLIP prototypes plus depth; depth rendered at the episode START state because the R2a recordings are deleted), s_orc (oracle
anchor, a privileged upper bound), s_cf (controllability). The goal-anchor localisation is reported first.
- H1: on `wo_pre`, s_vis - u11 named-object hit >= 0.20, 95% CI excludes 0, one-sided p (share <= 0) passes Holm.
- H2: on `su_pre`, the drop is <= 0.05 (p = share of bootstrap differences <= -0.05 passes Holm).
- H3: on `wg_post`, as H1 with the goal-role anchor. H4: on `su_post`, as H2.
- Holm over H1-H4 at 0.05; stage-1 gate = H1 and H2. Episode unit, two-level task-cluster bootstrap B 10,000 seed 0.
- Per-arm manifold read in stage 1 uses the new judge (binding of the arm's target, else null), descriptively.
The stage-1 code is a port of rb4_bind's `--test` / `--read` with the judge swapped; it is written only if stage 0 passes.
Stage 2 (closed loop) is out of scope: if stage 1 passes, stop and hand back.

## 5. Seen before this pre-registration (no R4b statistic)

- R4-bind's full stage-0 table under the old judge (section 1).
- The binding-source coverage (results/rb4b/sources.json; no policy output): see section 6.

## 6. Binding-source coverage (results/rb4b/sources.json; computed before the pre-registration, no policy output)

- Tuning pool: Y has a binding source in **32 of 250** observations (32 episodes, 3 tasks: libero_90:16, :32, :40). The other
  218 are judged by the null-prompt fallback. So the primary judge covers 13% of the gate pool; both groups are reported.
- The named object has a binding source in 66 of 250 tuning observations (descriptive `|named` check).
- Test sets (stage 1, descriptive manifold read only): named-object source wo_pre 0 of 163, su_pre 220 of 394, wg_post 1 of 144,
  su_post 93 of 133 (R3c's counts); Y source wo_pre 17, su_pre 148, wg_post 15, su_post 28.
- The rule reproduces R3c's named-object sources on all 1,084 tuning + test observations (0 disagreements).
- Inventory: all 250 R4-bind tuning files are on disk, so the chunks are reused (section 3); no R4-bind test file exists.

## 7. Disclosures (development; appended after the pre-registration row, whose doc sha256 covers sections 1-6)

- During development one `python3 -` heredoc (an empty placeholder) was issued on the login node. The harness blocked it before
  it ran. All computation ran on compute nodes.
- The probe judges the `|cf` arm only for the gate, as R4-bind did; the `|named` arm under its own binding is descriptive.
- Coverage of the primary judge is small (32 of 250); the gate is pooled with the null fallback as declared, and each group is
  reported separately.

## 8. Result, stage 0 (2026-10-25; ledger `rb4b_stage0`; results/rb4b/stage0.json)

The stored chunks were reused: the inversion round trip is at most 1.4e-7 action units over all 250 observations. On 8 checked
observations the stored instruction-prefix flow losses are reproduced to within 0.0066 (u11 within 2e-4; the guided arms differ
more, which is consistent with batch-composition numerics in the bf16 model). Movability, still, action range and g_named,
recomputed from the stored chunks, equal R4-bind's exactly (asserted). The bar is g_cf >= 0.130.

| config | g_cf | outlier rise, NEW judge [95% CI] | binding judge only (32 obs) / null only (218) | OLD judge | still cf | movable | on-manifold |
|---|---|---|---|---|---|---|---|
| d1 / early / 0.25 | +0.017 | +0.260 [+0.124, +0.419] | +0.786 / +0.182 | +0.053 | +0.022 | no | no |
| d1 / early / 0.5 | +0.054 | +0.261 [+0.130, +0.418] | +0.773 / +0.186 | +0.069 | +0.043 | no | no |
| d1 / early / 1 | +0.158 | +0.310 [+0.191, +0.453] | +0.758 / +0.244 | +0.169 | +0.071 | yes | no |
| d1 / all / 0.25 | +0.429 | +0.392 [+0.276, +0.533] | +0.768 / +0.337 | +0.327 | +0.015 | yes | no |
| d1 / all / 0.5 | +0.740 | +0.616 [+0.471, +0.753] | +0.786 / +0.591 | +0.653 | -0.104 | yes | no |
| d1 / all / 1 | +0.782 | +0.706 [+0.567, +0.835] | +0.823 / +0.689 | +0.798 | -0.132 | yes | no |
| dir / early / 0.25 | +0.022 | +0.258 [+0.125, +0.416] | +0.781 / +0.182 | +0.056 | +0.020 | no | no |
| dir / early / 0.5 | +0.061 | +0.256 [+0.127, +0.411] | +0.753 / +0.183 | +0.067 | +0.047 | no | no |
| dir / early / 1 | +0.169 | +0.285 [+0.163, +0.437] | +0.742 / +0.218 | +0.166 | +0.106 | yes | no |
| dir / all / 0.25 | +0.375 | +0.334 [+0.215, +0.483] | +0.734 / +0.275 | +0.294 | +0.110 | yes | no |
| dir / all / 0.5 | +0.658 | +0.483 [+0.348, +0.630] | +0.745 / +0.445 | +0.581 | -0.007 | yes | no |
| dir / all / 1 | +0.802 | +0.594 [+0.467, +0.721] | +0.784 / +0.566 | +0.699 | -0.148 | yes | no |
| naive post-hoc edit, d1 / dir | - | +0.763 / +0.671 | - | +0.860 / +0.775 | - | - | - |

The action-range rise is at most +0.005 everywhere (never binds). g_named and still(named) are unchanged from R4-bind.

**STAGE 0 FAILS under the new judge too.** No configuration passes the outlier bar of 0.10: the smallest rise is +0.256. No
configuration was chosen, no test observation was run, and stage 1 (H1-H4, anchor localisation) was not run.

**Reading.**
1. **The pre-registered new judge has a floor of its own.** Under it, the UNGUIDED instruction chunks (u11) already rise by
   +0.269 over the binding/null references (u21 +0.259; mean FM of u11 1.37x the reference's). Any chunk sampled under the
   instruction differs from chunks sampled under another prompt, so the judge marks the policy's own unguided output as off-manifold.
   That floor is why even the weakest guidance (early, w 0.25, g_cf +0.017) scores +0.26. The judge removed the circularity
   R4-bind had and introduced a prompt-shift floor instead; neither judge isolates "a natural approach to Y".
2. **On the 32 observations with a binding source, every configuration is an outlier in about 0.73-0.82 of chunks.** The binding
   prompt itself does bind Y: its own unguided chunks head to Y in 0.62 of chunks (u11 0.10; the null prompt's 0.23). So under a
   prompt that truly binds Y, the guided chunks do NOT look like that prompt's own approaches to Y. The coverage is 3 tasks, so this is
   thin evidence, but it points the same way as R4-bind: guidance of the action map does not produce the chunk the model would
   produce when it binds Y.
3. **Descriptive, NOT pre-registered (for the user's decision only).** Measured relative to u11 under the same new judge instead of
   ref21 (rise minus u11's +0.269, same episodes), d1/early/1 sits at about +0.04 with g_cf +0.158 and still +0.071, which would
   pass every bar; d1/all/0.25 sits at about +0.12. dir/early/1 (about +0.02) and dir/all/0.25 (about +0.07) fail on still
   (+0.106 / +0.110). This u11-referenced variant was not declared and does not change the verdict.
4. The scale of g_cf versus outlier rise keeps the R4-bind shape: all-step guidance moves strongly and costs strongly under
   either judge; early guidance is mostly pulled back.

**Disclosures.**
- The old-judge stage-0 numbers were known before the pre-registration (section 1). The coverage was computed before it.
- The primary judge covers 32 of 250 gate observations (3 tasks); 218 use the null fallback, as declared.
- The instruction-FM reproduction check differs by up to 0.0066 on guided arms (u11 2e-4); the new judge's guided and reference
  losses are computed in one batch per observation, so this does not enter the comparison.
- The `python3 -` login-node slip of section 7. Audits after the read: audit_numbers 0 discrepancies over 2,064 checks; audit_coverage 0 uncovered numerals.
