"""PRE-REGISTRATION of probe R4 (docs/PLAN_R4.md; code experiments/rb4_bind.py), written 2026-10-25 AFTER the prep (D1 operator fit
on training frames, prototypes incl. fixtures, named-type coverage), the depth geometry (calibration only) and a 2-observation GPU
smoke (parity, affine action map, w = 1 exactness, normalisation; never scored by the readout), and BEFORE any guided chunk is scored
(asserted below: no results/rb4/tune or test files, no stage0.json, no rb4_stage0 / rb4_read row).

Guidance: endpoint x0-projection inside pi05_l90c's own Euler sampler: x0_hat = x_t - t v; x0_hat += w (d* - D(x0_hat)) spread over the
H xy action steps (D = integrated xy displacement, affine); v = (x_t - x0_hat') / t.  d* = D1's operator lam (c - h) + b (`d1`) or the
policy's own speed along c - h (`dir`).  Grid: target {d1, dir} x schedule {early (first 5 of 10 steps), all} x w {0.25, 0.5, 1}.

  Stage 0 (tuning pool = R3b's, ORACLE cf / named poses): movable if g_cf >= 0.10 + max(0.03, |u21 - u11 at cf|); on-manifold on the cf
  arm: flow-loss outlier share (FM > mean + 2 sd of u11's 12) rise over u21 <= 0.10, |normalised action| > 1 share rise <= 0.05,
  still (< 2 cm) rise <= 0.10; admissible = movable & on-manifold & g_named >= -0.02 & still(named) <= 0.10; none -> stop.
  Choice: max g_cf among admissible; ties 0.005 -> smaller w, early before all, d1 before dir; recorded as rb4_stage0 first.
  Stage 1 (R3b's test sets; target from the VISION anchor = argmax SigLIP token vs the named types' training prototypes, back-projected
  with depth; oracle-anchor arm = privileged upper bound): H1 wo_pre s_vis - u11 >= 0.20, CI excludes 0, one-sided p (<= 0) passes Holm;
  H2 su_pre drop <= 0.05 (p = share <= -0.05); H3 wg_post as H1 (goal-role anchor; its localisation is reported first); H4 su_post as
  H2.  Covered observations; episode unit; two-level task-cluster bootstrap B 10,000 seed 0; Holm over H1-H4 at 0.05; gate = H1 & H2.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "rb4")
assert not glob.glob(os.path.join(OUT, "tune", "*.npz")), "a tuning file already exists"
assert not glob.glob(os.path.join(OUT, "test", "*.npz")), "a test file already exists"
assert not os.path.exists(os.path.join(OUT, "stage0.json")), "a stage-0 read already exists"
assert not ledger.load("rb4_stage0") and not ledger.load("rb4_read"), "an R4 row already exists"
op = json.load(open(os.path.join(OUT, "operator.json"), encoding="utf-8"))
geo = {}
for f in sorted(glob.glob(os.path.join(OUT, "geom", "_report_*.json"))):
    geo.update(json.load(open(f, encoding="utf-8")))
med = sorted(v["median_err_m"]["rs3"] for v in geo.values())
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/rb4_bind.py"), ("doc", "docs/PLAN_R4.md"),
                                               ("operator", "results/rb4/operator.json"), ("named_types", "results/rb4/named_types.json"),
                                               ("split", "results/rs3/split.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
SMOKE = ("parity max|u11 - bk.sample| 5.9e-8 and max|u11 - R2a draws| 0; affine action-map error 9.5e-8; d1/all/w1 endpoint = d* "
         "to 1e-8 m; quantile normalisation (use_quantiles=True), u11 share |x| > 1 0.026 / 0.043; FM u11 0.089 / 0.111 vs u21 "
         "0.086 / 0.114; mean |arm - u11| early w0.25 0.013-0.015, all w1 0.13-0.17, posthoc 0.14-0.17, second seed 0.07-0.08. "
         "Deviations: the rollout recordings are deleted, so depth is rendered at each episode's start state; the calibration bound "
         "was relaxed from 3 cm to 5 cm after libero_90:40 (3.4 cm) and :1 (3.6 cm) failed it (all 23 tasks 1.7-3.6 cm, other "
         "orientations 5-46 cm).  HARD-RULE slips during development: one placeholder `python3 -` was issued on the login node "
         "and blocked by the harness before it ran.")
DISCLOSURE = (
    "seen before this row: R3b's, R3c's, R2a's and OF's published results (u11 on these observations is known: wo_pre 0.077, "
    "su_pre 0.606, wg_post 0.131, su_post 0.780; OF D3 object anchor 0.62 / 0.68); the D1 operator fit on training frames "
    f"(lam {op['lam']:.4f}, b {[round(x, 4) for x in op['b']]}, R^2 {op['r2']:.3f}); prototype coverage of named types (pre 807 / 807, "
    f"post 183 / 277); depth calibration on {len(geo)} tasks (median xy error of region token points vs candidate positions "
    f"{med[0]:.4f} to {med[-1]:.4f} m, R3b's orientation on every task); GPU smoke on 2 tuning observations: {SMOKE}")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr4_bind", seed=11,
    metrics={"guidance": "endpoint x0-projection in the policy's own Euler sampler; d* = D1 operator (d1) or own speed along c - h (dir)",
             "grid": "target {d1, dir} x schedule {early: first 5 of 10 steps, all} x w {0.25, 0.5, 1.0}",
             "stage0": "tuning pool (R3b's 250): movable if g_cf >= 0.10 + max(0.03, |u21 - u11 at cf|) toward R3b's cf ORACLE pose; "
                       "on-manifold on the cf arm: FM-outlier share rise over u21 <= 0.10 (outlier = FM > mean + 2 sd of u11's 12; "
                       "t {0.1, 0.2, 0.3, 0.5} x 4 noises seed 5), |normalised action| > 1 share rise <= 0.05, still rise <= 0.10; "
                       "admissible = movable & on-manifold & g_named >= -0.02 & still(named) <= 0.10; none admissible -> stop",
             "choice": "max g_cf among admissible; ties 0.005 -> smaller w, early before all, d1 before dir; rb4_stage0 before any test",
             "anchor": "argmax token of the cosine of SigLIP agentview tokens to the named types' training prototypes (objects + "
                       "fixtures); depth back-projection (robot hidden), token foreground-half median point; oracle anchor = "
                       "named candidate position (privileged upper bound)",
             "H1": "wo_pre: s_vis - u11 named hit >= 0.20 (episode means, covered obs), 95% CI excludes 0, one-sided p (share <= 0) "
                   "passes Holm",
             "H2": "su_pre: s_vis - u11 >= -0.05; p = share of bootstrap diffs <= -0.05 passes Holm",
             "H3": "wg_post: as H1 on the own target with the goal-role vision anchor (localisation su_post vs wg_post reported first)",
             "H4": "su_post: as H2", "family": "Holm over H1-H4 at 0.05; stage-1 gate = H1 and H2; H3/H4 decide the goal phase",
             "stage2": "only if stage 1 passes: 15 recombination dev tasks x 20 fresh_rb2 states paired with R2's zs 170 / ceiling 223; "
                       "damage on R2's 8 cells; fm_types; GO = closure >= 0.5 of the zs -> ceiling gap; Holm <= 4",
             "readout": "R2a pointed(), 12 chunks seed 11; u21 noise floor", "operator": op, "depth_calibration": geo,
             "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any guided chunk scored, any tuning statistic, any test observation run"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr4_bind.py",
    command="pre-registration: R4 vision-anchored flow guidance, stage 0 + stage-1 H1-H4, after prep/geometry/smoke, before tuning"))
