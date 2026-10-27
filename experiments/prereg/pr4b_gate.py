"""PRE-REGISTRATION of probe R4b (docs/PLAN_R4B.md; code experiments/rb4b_bind.py), written 2026-10-25 AFTER R4-bind's stage 0
(docs/PLAN_R4.md section 8: the old-judge numbers are KNOWN) and after the binding-source table (results/rb4b/sources.json, no policy
output), and BEFORE any R4b statistic (asserted below: no results/rb4b/judge or test file, no rb4b stage0.json, no rb4b_stage0 /
rb4b_read row).

Why: R4-bind judged guided chunks by their flow loss under the CURRENT (misbound) conditioning, so a correct approach to an unbound
object is penalised by construction (circular).  R4b keeps the guidance, grid, tuning pool, cf target, noise floor, movability bar
(g_cf >= 0.130), secondary bars, admissibility and tie rule, and replaces only that judge:
  primary  = FM under a same-scene training instruction that names the target Y in the object role (R3c's oracle source rule; lowest
             sha1 of rs4src:{task}:O:{Y}:{cell}); reference = the policy's own unguided chunks under that binding prompt; outlier = FM >
             mean + 2 sd of the seed-11 reference's 12; rise = guided |cf share - seed-21 reference share (episode means);
  fallback = the same with the null prompt "" where Y has no binding source;  gate statistic pooled (primary where covered).
  bars: outlier rise <= 0.10, |normalised action| > 1 rise <= 0.05, still rise <= 0.10; admissible = movable & bars & g_named >= -0.02
  & still(named) <= 0.10; choice max g_cf, ties 0.005 -> smaller w, early, d1.  None admissible -> stop.
  Stage 1 only if stage 0 passes: R4-bind's H1-H4 unchanged (vision anchor; oracle arm = privileged upper bound; goal-anchor
  localisation first; Holm over H1-H4; gate H1 & H2); stage 2 out of scope.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "rb4b")
assert not glob.glob(os.path.join(OUT, "judge", "*.npz")), "a judge file already exists"
assert not glob.glob(os.path.join(OUT, "test", "*.npz")), "a test file already exists"
assert not os.path.exists(os.path.join(OUT, "stage0.json")), "an R4b stage-0 read already exists"
assert not ledger.load("rb4b_stage0") and not ledger.load("rb4b_read"), "an R4b row already exists"
src = json.load(open(os.path.join(OUT, "sources.json"), encoding="utf-8"))
old = json.load(open(os.path.join(ROOT, "results", "rb4", "stage0.json"), encoding="utf-8"))
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/rb4b_bind.py"), ("doc", "docs/PLAN_R4B.md"),
                                               ("rb4_script", "experiments/rb4_bind.py"), ("sources", "results/rb4b/sources.json"),
                                               ("rb4_stage0", "results/rb4/stage0.json"), ("split", "results/rs3/split.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
DISCLOSURE = (
    "seen before this row: R4-bind's complete stage-0 table under the OLD judge (results/rb4/stage0.json: g_cf, g_named, still, "
    "action range, old FM-outlier rise for all 12 configs; stage 0 failed, no config chosen), and R3b/R3c/R2a/OF published results. "
    f"Binding-source coverage (no policy output): tuning Y {src['counts']['tune']['bind_cf']} / {src['counts']['tune']['n']} "
    f"(tasks {src['counts']['tune']['bind_cf_tasks']}), named {src['counts']['tune']['bind_named']} / 250; test named sources "
    + json.dumps({k: v["bind_named"] for k, v in src['counts'].items() if k != 'tune'})
    + f"; rule reproduces R3c's named sources ({src['rs4_crosscheck']['agree']} agree, {len(src['rs4_crosscheck']['disagree'])} "
    "disagree). Movability is fixed by the stored chunks (recomputed, asserted equal to R4-bind's). HARD-RULE slip during "
    "development: one `python3 -` heredoc was issued on the login node and blocked by the harness before it ran.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr4b_gate", seed=11,
    metrics={"why": "R4-bind's on-manifold judge (FM under the current, misbound conditioning) penalises a correct approach to an "
                    "unbound object by construction (circular); R4b replaces only that judge",
             "fixed_from_rb4": "guidance form, 12-config grid, tuning pool (R3b's 250), cf target Y (rs3 cf_pick), noise floor, "
                               f"movability bar g_cf >= {old['threshold']:.3f}, secondary bars, admissibility, tie rule, stage-1 H1-H4",
             "judge_primary": "FM (t {0.1,0.2,0.3,0.5} x 4 noises seed 5) of the guided |cf chunk under a same-scene train_l90 "
                              "instruction whose BDDL In/On goal moves Y (R3c oracle rule, lowest sha1 rs4src:{task}:O:{Y}:{cell}); "
                              "reference = own unguided chunks under that prompt, seeds 11 / 21; outlier = FM > mean + 2 sd of "
                              "ref11's 12; rise = guided share - ref21 share, episode means, task-cluster bootstrap",
             "judge_fallback": "the same under the null prompt '' where Y has no binding source; gate statistic pooled",
             "bars": "outlier rise <= 0.10; |normalised action| > 1 rise <= 0.05; still (< 2 cm) rise <= 0.10",
             "admissible": "movable & bars & g_named >= -0.02 & still(named) <= 0.10",
             "choice": "max g_cf among admissible; ties 0.005 -> smaller w, early before all, d1 before dir; rb4b_stage0 first",
             "reuse": "R4-bind's stored tuning chunks inverted through the affine output transform (round trip < 1e-4 asserted; "
                      "instruction FM reproduced on 2 obs per shard); no guided chunk regenerated",
             "stage1": "only if stage 0 passes: R4-bind's pre-registered stage 1 unchanged except the judge (vision anchor = SigLIP "
                       "prototypes + depth at the episode start state; oracle-anchor arm = privileged upper bound; goal-anchor "
                       "localisation first; H1 wo_pre rise >= 0.20 CI excl 0; H2 su_pre drop <= 0.05; H3 wg_post as H1; H4 su_post "
                       "as H2; Holm over H1-H4 at 0.05; gate H1 & H2); stage 2 out of scope",
             "coverage": src["counts"], "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any R4b judge flow loss, any R4b stage-0 statistic, any test observation"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr4b_gate.py",
    command="pre-registration: R4b non-circular on-manifold judge for R4-bind's stage 0 (+ stage 1 if it passes)"))
