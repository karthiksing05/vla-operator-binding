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
