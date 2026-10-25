"""PRE-REGISTRATION of probe R3c (docs/PLAN_R3C.md; code experiments/rs4_splice.py), written 2026-10-25 AFTER the source table
(results/rs4/sources.json: oracle same-scene sources from BDDL goal atoms; coverage, no outcome) and a 2-observation GPU smoke
(identity splice == u11 exactly, parity with R2a's draws, movement magnitudes; never scored), and BEFORE any spliced chunk is scored
by the readout (asserted below: no results/rs4/tune or test files, no stage0.json, no rs4_stage0 / rs4_read row).

Region-wise K/V splicing: in the target instruction's prefix cache, layers >= L in {0, 2, 4}, the K/V at the named object's (pre-grasp)
or own target's (post-grasp) oracle region tokens (dilation 0 / 1) are replaced by those computed under a same-scene training
instruction that moves that object (A) or places on that target (B).  Whole-image splicing (all 768 image positions) and the full
source prompt are controls.

  Stage 0 (tuning pool = R3b's): some REGION config has g_cf >= 0.10 + max(0.03, |u21 - u11 at cf|), else stop.
  Choice: region configs passing stage 0 with g_named >= -0.02 and still <= +0.10; max g_cf; ties 0.005 -> fewer layers, dilation 0.
  H1 wo_pre: s - u11 >= 0.20, 95% CI excludes 0, one-sided bootstrap p (<= 0) passes Holm.   H2 su_pre: s - u11 >= -0.05 (p <= -0.05).
  H3 wg_post: as H1.   H4 su_post: as H2.   Covered observations (a source exists); a set with no covered episode -> p = 1, fails.
  Episode unit; two-level task-cluster bootstrap B 10,000 seed 0; Holm over H1-H4 at 0.05.  Stage-1 gate = H1 and H2.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "rs4")
assert not glob.glob(os.path.join(OUT, "tune", "*.npz")), "a tuning file already exists"
assert not glob.glob(os.path.join(OUT, "test", "*.npz")), "a test file already exists"
assert not os.path.exists(os.path.join(OUT, "stage0.json")), "a stage-0 read already exists"
assert not ledger.load("rs4_stage0") and not ledger.load("rs4_read"), "an R3c row already exists"
src = json.load(open(os.path.join(OUT, "sources.json"), encoding="utf-8"))
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/rs4_splice.py"), ("doc", "docs/PLAN_R3C.md"),
                                               ("sources", "results/rs4/sources.json"), ("split", "results/rs3/split.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
cov = src["coverage_recombination"]
DISCLOSURE = (
    "seen before this row: R3b's and R2a's published results (u11 on these observations is known); the source table: of the 15 "
    f"recombination tasks {cov['A_all']} have an object source A for every moved object, {cov['B_all']} a target source B, "
    f"{cov['both']} both (4 are pure articulation tasks with neither); observation coverage {json.dumps(src['counts'])}. "
    "In particular wo_pre has NO covered observation (its 7 tasks, libero_90:38/58/73/77/79/80/83, have no same-scene training cell "
    "moving the named object, even at the type level) and wg_post has 1, so H1 and H3 cannot pass as specified; recorded here "
    "before any read.  GPU smoke (results/rs4/scratch, libero_90:14 t8 and :16 t0 pre-grasp): identity splice max diff 0, u11 "
    "parity with R2a draft_instr 0, cache 18 layers x 968 positions; mean |arm - u11| region splices 0.0005-0.011, whole-image "
    "0.009-0.10, full source prompt 0.03-0.11, a second seed 0.07-0.08 (no readout computed).  During development one empty "
    "python3 heredoc was run on the login node (a placeholder command, no input; a HARD-RULE violation, disclosed).")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr3c_splice", seed=11,
    metrics={"stage0": "tuning pool: some region config has g_cf >= 0.10 + max(0.03, |u21 - u11 at cf|) (point estimate); else stop",
             "choice": "region configs passing stage 0 with g_named >= -0.02, still <= +0.10; max g_cf; ties 0.005 -> fewer spliced "
                       "layers (>=4, >=2, all), dilation 0; recorded as rs4_stage0 before any test observation is run",
             "H1": "wo_pre: spliced - unsteered named-object hit >= 0.20 (episode means, covered obs), 95% CI excludes 0, one-sided p "
                   "(share <= 0) passes Holm; no covered episode -> fails",
             "H2": "su_pre: spliced - unsteered >= -0.05; p = share of bootstrap diffs <= -0.05 passes Holm",
             "H3": "wg_post: as H1 on the own target (W's region from B)", "H4": "su_post: as H2",
             "family": "Holm over H1-H4 at 0.05; stage-1 gate = H1 and H2; H3/H4 decide the goal phase in stage 2",
             "grid": "layers {all, >=2, >=4} x {region d0, region d1, whole image}; whole image and full prompt are controls",
             "sources": "ORACLE: same-scene libero_90 training cells whose BDDL In/On atom moves X (A) / places on W (B); sha1 tie rule",
             "readout": "R2a pointed(), 12 chunks seed 11; u21 noise floor", "coverage": cov, "counts": src["counts"],
             "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any spliced chunk scored, any tuning statistic, any test observation run"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr3c_splice.py",
    command="pre-registration: R3c region-wise K/V splicing, stage 0 + stage-1 H1-H4, after sources/smoke, before tuning"))
