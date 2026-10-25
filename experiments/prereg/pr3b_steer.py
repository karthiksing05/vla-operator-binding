"""PRE-REGISTRATION of probe R3b (docs/PLAN_R3B.md; code experiments/rs3_steer.py), written 2026-10-25 AFTER the oracle region maps,
their hand check, the split by episode type and a 2-observation GPU smoke (parity of the unsteered arm with R2a's stored draws and
that a steered arm moves the chunk; results/rs3/scratch/smoke, never scored), and BEFORE any steered chunk is scored by the readout
(asserted below: no results/rs3/tune or test files, no choice.json, no rs3_choice / rs3_read row).

At R2a's stored observations of zs pi05_l90c@5000 (29 dev tasks, fresh_r2a), 12 chunks per arm from the same noise (seed 11 = R2a's
policy_sample), with an additive attention-logit bias +b from the object-word (pre-grasp) / goal-word (post-grasp) queries of the
prefix pass ("word"), and/or the action expert's queries ("expert"), to the agentview tokens of the ORACLE region (simulator
segmentation at the recorded poses), in layers {0-1, 0-3, 0-7}, b {2, 4, 8}.  The config is chosen on the tuning pool (middle
pre-grasp observation of zs episodes typed wrong_goal / motion / no_commit / fixture) by max g_cf (counterfactual controllability)
subject to g_named >= -0.02 and still <= +0.10; ties 0.005 -> smaller b, fewer layers, word < expert < both.

  H1  wo_pre (zs wrong_object, pre-grasp): s - u11 >= 0.20, 95% CI excludes 0, one-sided bootstrap p (<= 0) passes Holm.
  H2  su_pre (zs success, pre-grasp): s - u11 >= -0.05, p = share of bootstrap diffs <= -0.05 passes Holm.
  H3  wg_post (zs wrong_goal, post-grasp): as H1.          H4  su_post (zs success, post-grasp): as H2.
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

OUT = os.path.join(ROOT, "results", "rs3")
assert not glob.glob(os.path.join(OUT, "tune", "*.npz")), "a tuning file already exists"
assert not glob.glob(os.path.join(OUT, "test", "*.npz")), "a test file already exists"
assert not os.path.exists(os.path.join(OUT, "choice.json")), "a choice already exists"
assert not ledger.load("rs3_choice") and not ledger.load("rs3_read"), "an R3b row already exists"
sp = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/rs3_steer.py"), ("doc", "docs/PLAN_R3B.md"),
                                               ("split", "results/rs3/split.json"), ("r2a_read", "results/r2a/read.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
DISCLOSURE = ("seen before this row: R2a's published results (docs/PLAN_R2A.md, read.json; so u11 = R2a policy_sample on these "
              "observations is known: 0.08 on zs wrong-object, 0.61 on zs successes, pre-grasp); the oracle region job (projection "
              "orientation (0,0,1) on every task, median body-origin vs mask-centroid error 1.3-19.4 px; the calibration assertion was "
              "relaxed twice from < 16 px to 'same transform everywhere, < 30 px' after tasks 73 and 77 failed it at 17-19 px); a hand "
              "check of 12 random overlays (orientation and object identity right; the right moka pot's region in libero_90:38 also "
              "holds a 3-4 token strip under the stove); the split counts (tune 250; wo_pre 163, su_pre 394, wg_post 144, su_post 133); "
              "a 2-observation GPU smoke on tuning-pool observations (libero_90:14 t8, :16 t0; results/rs3/scratch/smoke): parity "
              "max |u11 - R2a draft_instr| = 0 and mean |steered - u11| of 0.001-0.003 (word / both) and 0.027 (expert L0-7 b8), "
              "no pointing computed.  Parse note: M1's step_key reads libero_90:16 ('stack the black bowl at the front on the black "
              "bowl in the middle') as object 'black bowl at the front on the black bowl', goal 'middle', so its object-word rows "
              "include the second bowl's words (kept as M1's parse, disclosed).  A python heredoc was attempted on the login node "
              "once during development and was blocked by the harness before running.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr3b_steer", seed=11,
    metrics={"H1": "wo_pre: steered - unsteered named-object hit share >= 0.20 (episode means), 95% CI excludes 0, one-sided "
                   "bootstrap p (share <= 0) passes Holm",
             "H2": "su_pre: steered - unsteered >= -0.05; one-sided p = share of bootstrap diffs <= -0.05 passes Holm",
             "H3": "wg_post: as H1 on the own target (goal words, own-target region)",
             "H4": "su_post: as H2 on the own target",
             "family": "Holm over H1-H4 at alpha 0.05; stage-1 gate = H1 and H2; H3/H4 decide the goal phase in stage 2",
             "selection": "grid site {word, expert, both} x layers {0-1, 0-3, 0-7} x b {2, 4, 8} on the tuning pool; admissible if "
                          "g_named >= -0.02 and still <= +0.10; max g_cf; ties 0.005 -> smaller b, fewer layers, word<expert<both; "
                          "none admissible -> max g_named; recorded as rs3_choice before any test observation is run",
             "region": "ORACLE: simulator segmentation, movable objects at recorded poses, robot hidden, agentview 16x16 tokens, "
                       ">= 5% pixel coverage; disk 24 px fallbacks; goal sites = parent mask inside the disk",
             "readout": "R2a's pointed(): integrated 10-step path, first candidate along the heading in a 5 cm corridor else best "
                        "cosine, < 2 cm none; observation value = share of 12 chunks (seed 11) at the target",
             "comparators": "u21 (unsteered seed 21, noise floor), s_cf (counterfactual target), sweep over b, R2a arms recomputed",
             "readout_noise": "R2a validity 0.645 pre-grasp (chance 0.379), 0.795 post-grasp (chance 0.272)",
             "n": {"tune": sp["n_tune"], "test": sp["n_test"]},
             "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any steered chunk scored, any tuning statistic, any test observation run"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr3b_steer.py",
    command="pre-registration: R3b attention-logit binding steering, stage-1 H1-H4, after regions/split/smoke, before tuning"))
