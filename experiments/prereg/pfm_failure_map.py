"""PRE-REGISTRATION of plan FM, the failure map (docs/PLAN_FM.md; code experiments/fm_types.py, experiments/fm_fewshot.py), written
2026-10-24 BEFORE any typed aggregate of M1's episodes (no results/fm/types/*.json exists) and before any stage-B fit or rollout
(asserted below).

Stage A types every failed episode of M1's dev rollouts (pi05_l90c@5000, 29 dev tasks x 20 fresh_m1 states, noise seed 13) as
fixture / no_commit / wrong_object (subtype wrong_instance) / wrong_goal / motion, from the recorded object poses evaluated with
the task's own BDDL predicates in a CPU LIBERO env (settle 10 control steps; carried >= 0.10 m; half way = 0.5).  Stage B adds
LoRA from k = 5 and 10 demos (budgets 200 and 1000) and the KV token from k = 5, at M1's settings, on M1's tasks / states / seed.

  F1  zs binding-error rate (wrong object + wrong goal) rises with task CD on the 11 combination pick-and-place tasks: Spearman
      rho > 0, one-sided permutation p (100,000, seed 0).
  F2  (a) lora1000 (k = 1) wrong-object episodes start farther from the demo's grasp point (within-task permutation of the labels,
      20,000, one-sided) AND (b) lora1000 wrong-object episodes at k = 1 > at k = 10 (paired, one-sided exact sign test);
      p = max(pa, pb).
  F3  zs typed failures on the 4 instance tasks (14, 16, 60, 38): share of instance-role binding errors > 0.5 (two-level cluster
      bootstrap, B 10,000, seed 0; p = share of bootstrap shares <= 0.5); fewer than 10 typed failures -> not testable, p = 1.
  F4  part pick-and-place tasks (13): motion share of typed failures > 0.5 for create AND lora1000 (cluster bootstrap; p = max).
  F5  compose_sel's induced failures (zs success -> compose_sel failure) on the composed pick-and-place tasks: share of binding
      errors > 0.5 (cluster bootstrap); fewer than 10 -> not testable, p = 1.
  Holm over F1-F5 at alpha 0.05.  Repair readings and the k-curve: pre-specified, descriptive (docs/PLAN_FM.md section 3).

Disclosure.  M1's per-task success (results/m1/read.json) was read before this plan.  The typer was checked by hand on 16
episodes, whose types were seen: libero_90:14 zs trials 0-2 (successes; own atom holds after settling); libero_90:60 zs trial 18
wrong_goal (on the other bowl); libero_90:40 zs trials 0, 1 wrong_goal (bottom shelf); libero_90:16 zs trials 0, 1 wrong_goal
(on the third bowl); libero_90:73 create trials 0, 2 motion; libero_90:1 lora1000 trial 4 wrong_object, trial 6 motion
(in_region); libero_90:38 create trials 0, 1 no_commit; libero_90:76 lora1000 trials 0, 2 motion.  The demos' grasp points
(results/fm/demo_grasp.json) were computed from the demos alone.  On 2026-10-24 one `python3 --version` ran on the login node
(no experiment code)."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "fm")
assert not glob.glob(os.path.join(OUT, "types", "*.json")), "a typed aggregate input already exists"
assert not glob.glob(os.path.join(OUT, "fits", "*", "*", "fit.json")), "a stage-B fit already exists"
assert not glob.glob(os.path.join(OUT, "roll", "*")), "a stage-B rollout already exists"
assert not ledger.load("fm_read"), "an FM read already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("typer", "experiments/fm_types.py"), ("fewshot", "experiments/fm_fewshot.py"),
                                               ("doc", "docs/PLAN_FM.md"), ("demo_grasp", "results/fm/demo_grasp.json"),
                                               ("m1_plan", "results/m1/plan.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(sha, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pfm_failure_map", seed=13,
    metrics={"F1": "zs binding-error rate rises with task CD (11 combination pick-and-place tasks): Spearman rho > 0, one-sided "
                   "permutation p (100000)",
             "F2": "(a) lora1000 k=1 wrong-object episodes start farther from the demo grasp point (within-task label permutation, "
                   "20000, one-sided) AND (b) lora1000 wrong-object k=1 > k=10 (paired one-sided exact sign test); p = max",
             "F3": "zs typed failures on instance tasks 14/16/60/38: instance-role binding errors > 0.5 (cluster bootstrap); "
                   "< 10 failures -> not testable",
             "F4": "13 part pick-and-place tasks: motion share of typed failures > 0.5 for create AND lora1000 (cluster bootstrap; "
                   "p = max)",
             "F5": "compose_sel induced failures (zs success -> compose_sel failure, composed pick-and-place tasks): binding-error "
                   "share > 0.5 (cluster bootstrap); < 10 -> not testable",
             "family": "Holm over F1-F5 at alpha 0.05",
             "types": "fixture / no_commit / wrong_object (wrong_instance) / wrong_goal (other_region | carried_elsewhere) / motion "
                      "(toward_target | not_carried | in_region); settle 10 steps, carried >= 0.10 m, half way 0.5",
             "stage_b": dict(arms=["lora_k5_200", "lora_k5_1000", "lora_k10_200", "lora_k10_1000", "create_k5"], demos="own-scene "
                             "IPEC episodes 0..k-1 (nesting M1's episode 0)", settings="M1's (r16, lr 1e-4, batch 32; token n4 lr "
                             "1e-2 1000 steps)", states="fresh_m1, 20 per task, noise seed 13, paired with M1"),
             "secondary": "type shares by arm x axis (bin, kind, scene_familiar, instance); success-vs-k curve; create_k5 vs "
                          "lora_k5_1000; logistic slope of success on CD per arm; composition cards",
             "disclosure": "M1 per-task success read before; 16 hand-checked episodes' types seen (listed in the script docstring); "
                           "one python3 --version on the login node",
             "files_sha256": sha, "written_before": "any typed aggregate, stage-B fit or rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pfm_failure_map.py",
    command="pre-registration: failure map F1-F5, before typing aggregates and stage-B fits/rollouts"))
