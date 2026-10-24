"""PRE-REGISTRATION of plan L4, a continual store of created role tokens (docs/PLAN_L4.md section 7; code experiments/l4_store.py),
written 2026-10-24 BEFORE any L4 fit, rollout or read (asserted below).  pi05_l90c main/5000 frozen; 29 LIBERO-90-C dev tasks; one
demo per task (own-scene IPEC episode 0, never executed or replayed); new evaluation states fresh_l4 (base 6100000, 20 per task) and
practice states fresh_l4p (base 6300000, 5 per task); noise seed 13.

The part: a ROLE TOKEN (KVToken, n 2 per layer, 36,864 bytes), O attached while the policy's gripper is open, G while it carries
(held_ops._Grasp); a task's pair fitted jointly on its demo (each frame under its phase's token), 1000 steps, lr 1e-2, batch 32.
Keys (scene, role, items); a token applies to any same-scene request whose role items contain its key; composition = concatenation.

  H1  scratch role pair non-inferior to M1's whole-task token (same bytes) at margin 0.05, 29 tasks (one-sided cluster-bootstrap p).
  H2  reuse@canonical partner > zs on the 13 canonical headroom targets (more successes + exact two-sided sign test).
  H3  complete@partner > causal@partner (reused tokens removed), 13 targets (same test).
  H4  init@partner (200 steps) > scratch at 200 steps, 13 targets (same test).
  H5  stream: FWT > 0 (late vs early arrival, 16 tasks, both mirrored orders pooled; same test) AND routed damage >= -0.01 on the
      dev tasks and on M1's 10 training cells in both orders AND final store bytes <= 1/100 of 29 per-task LoRAs.
  Holm over H1-H5 at alpha 0.05.  Stream guard: 5 practice episodes per candidate, pass >= 3 of 5, ties zs < cover < fitted.

Disclosure.  M1's and FM's per-task success on fresh_m1 was read before this plan (it sets the headroom rule, M1 zs <= 15 of 20).
On 2026-10-24 one shell command beginning with `python3` was refused on the login node before running (no python executed)."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "l4")
assert not glob.glob(os.path.join(OUT, "fits", "*", "*", "fit.json")), "an L4 fit already exists"
assert not glob.glob(os.path.join(OUT, "roll", "*")), "an L4 rollout already exists"
assert not ledger.load("l4_read"), "an L4 read already exists"
assert not ledger.load("l4_fit"), "an L4 fit row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("code", "experiments/l4_store.py"), ("doc", "docs/PLAN_L4.md"),
                                               ("plan", "results/l4/plan.json"), ("serve", "lifelong_ood/lang/transplant.py"),
                                               ("m1_plan", "results/m1/plan.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(sha, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pl4_store", seed=13,
    metrics={"H1": "scratch O+G role pair (n 2 each) non-inferior to M1 whole-task token (n 4) at margin 0.05 on 29 dev tasks; "
                   "one-sided p = share of two-level cluster-bootstrap differences <= -0.05 (B 10000)",
             "H2": "reuse@canonical partner > zs on the 13 canonical headroom targets: more successes + exact two-sided sign test",
             "H3": "complete@partner > causal@partner (reused tokens removed) on the 13 targets: same test",
             "H4": "init@partner at 200 steps > scratch at 200 steps on the 13 targets: same test",
             "H5": "stream FWT > 0 (late vs early, 16 tasks, orders A/B pooled; same test) AND routed damage >= -0.01 (dev tasks and "
                   "M1's 10 training cells, both orders) AND final store bytes <= 1/100 of 29 per-task LoRAs",
             "family": "Holm over H1-H5 at alpha 0.05",
             "stream_rule": "route by parts (scene, role, items contained); practice zs + cover 5 eps each (fresh_l4p); best >= 3/5 "
                            "passes (ties zs < cover); else complete (cover frozen, new tokens for uncovered items) if the cover is "
                            "non-empty and >= zs on practice, else scratch; practice it; serve the best (ties zs < cover < fitted); "
                            "store new tokens as parts (held keys -> exception); exception when part routing differs from the choice",
             "canonical": plan["canonical"], "orders": plan["orders"], "fwt_tasks": sorted(plan["fwt"]),
             "pairs": len(plan["pairs"]), "headroom_pairs": sum(p["headroom"] for p in plan["pairs"]),
             "settings": plan["settings"],
             "secondary": "per-arm success (combination / part); complete vs scratch non-inferiority (unadjusted); reuse vs zs on all "
                          "pairs; cross-scene same-text reuse; budget-to-threshold (scratch 50/200/1000, init 50/200); stream "
                          "choices, fits, bytes-vs-parts, GPU-s per arrival vs LoRA, exceptions",
             "disclosure": "M1 and FM per-task success on fresh_m1 read before (headroom rule); one python3-prefixed command refused "
                           "on the login node before running",
             "files_sha256": sha, "written_before": "any L4 fit, rollout or read"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl4_store.py",
    command="pre-registration: L4 continual store of created role tokens, H1-H5, before any fit / rollout / read"))
