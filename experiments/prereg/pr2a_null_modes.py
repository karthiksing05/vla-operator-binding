"""PRE-REGISTRATION of probe R2a (docs/PLAN_R2A.md; code experiments/r2a_null_modes.py), written 2026-10-24 AFTER the zs rollouts,
the episode typing and the readout validation (results/r2a/validate.json: the policy's executed windows against the object it
bound), and BEFORE any mode, likelihood ratio or aggregate of an arm (asserted below: no results/r2a/modes/*.npz, no r2a_read row).

At pre-grasp observations of new zs rollouts of pi05_l90c@5000 (29 dev tasks, fresh_r2a states BASE 7100000, 20 per task, noise
seed 13), the null-prompt field is ascended from 48 shared starts (12 null drafts at u 0.1 / 0.3 / 0.55, 12 x 0.7 eps at u 0.7;
Adam 80 iterations, lr 0.02, Tweedie x0), the modes are clustered (greedy rms 0.15), and each option is ranked by
Delta = L_FM(a | o, null) - L_FM(a | o, instr) (t in 0.1..0.9, K 8 shared noises).  The readout: the integrated 10-step path's
horizontal direction picks the candidate (cosine), < 2 cm -> none.  Unit = episode (mean over its observations); rates by the
two-level task-cluster bootstrap (B 10,000, seed 0).

  H1  coverage (some null option points at the named object) pooled over the 22 object-role tasks >= 0.8 (an estimate gate).
  H2  on zs episodes typed wrong_object: ratio_top > policy_sample (the instructed policy's own 12 samples at the observation);
      one-sided exact sign test on episodes whose means differ; < 10 episodes -> not testable, does not pass.
  H3  on zs successful episodes of the object-role tasks: ratio_top - policy_sample >= -0.05; one-sided bootstrap p = share of
      bootstrap differences <= -0.05.
  Holm over H2-H3 at alpha 0.05.  Closed-loop smoke (exploratory) only if H1 and H2 pass.

Disclosure (filled in when run): see the `disclosure` metric."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "r2a")
assert not glob.glob(os.path.join(OUT, "modes", "*.npz")), "a mode file already exists"
assert not ledger.load("r2a_read"), "an R2a read already exists"
assert os.path.exists(os.path.join(OUT, "validate.json")), "run the readout validation first"
val = json.load(open(os.path.join(OUT, "validate.json"), encoding="utf-8"))
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/r2a_null_modes.py"), ("doc", "docs/PLAN_R2A.md"),
                                               ("validate", "results/r2a/validate.json"), ("m1_plan", "results/m1/plan.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
DISCLOSURE = ("seen before this row: the zs rollouts of 28 of 29 tasks (libero_90:16 still rolling; its observations join the mode "
              "stage when it finishes) and their episode types (validation episode_types); the readout validation on 28 tasks; a hand "
              "check of libero_90:14 trial 13 (printed geometry), which showed max-cosine picking the cabinet BEHIND the plate, so the "
              "readout gained the first-along-heading rule (5 cm corridor) before the validation aggregate; a 2-observation GPU smoke "
              "of the mode stage (results/r2a/scratch/smoke_modes, post-grasp observations of libero_90:14, never read as data): it "
              "showed 63 s per observation and 46 of 48 null starts in one cluster, after which only the null modes and drafts are "
              "scored and the post-grasp role keeps 1 observation per episode; no pre-grasp mode, no ratio of a pre-grasp observation "
              "and no arm aggregate was computed")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr2a_null_modes", seed=13,
    metrics={"H1": "coverage of the named object by the null-field options, pooled over the 22 object-role tasks (episode means, "
                   "task-cluster bootstrap) >= 0.8 (estimate gate)",
             "H2": "zs wrong_object episodes: ratio_top > policy_sample; one-sided exact sign test on discordant episodes; < 10 "
                   "episodes -> not testable",
             "H3": "zs successful object-role episodes: ratio_top - policy_sample >= -0.05; one-sided bootstrap p (share <= -0.05)",
             "family": "Holm over H2-H3 at alpha 0.05; H1 an estimate gate; closed loop only if H1 and H2 pass",
             "estimator": "Delta = L_FM(a|o,null) - L_FM(a|o,instr), velocity loss, t 0.1..0.9 uniform, K 8 shared noises (seed 5); "
                          "Diffusion-Classifier ELBO difference",
             "modes": "null field; 12 null drafts x u {0.1, 0.3, 0.55} + 12 x 0.7 eps at u 0.7; Adam 80 it lr 0.02; Tweedie x0; "
                      "greedy rms 0.15 clusters; option score = mean member Delta; also instr and CAG (w 2) fields from the same starts",
             "readout": "10-step path integrated with the pp2_dev_l90c scale; argmax horizontal cosine over candidates; < 2 cm -> none; "
                        "cluster object = member majority",
             "comparators": "policy_sample, policy_exec, instr_lik_top, instr_mode_top, cag_mode_top, null_mass_top, random",
             "secondary": "every arm by bin / kind / scene / instance / zs type; goal role (post-grasp) tables; eps-weighted ratio; "
                          "wrong-object cards",
             "validation_seen": {k: val[k] for k in ("pre", "post", "episode_types", "n_obs") if k in val},
             "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any mode, likelihood ratio or arm aggregate"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr2a_null_modes.py",
    command="pre-registration: R2a null-field modes + instruction/null ratio, H1-H3, after readout validation, before modes"))
