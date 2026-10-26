"""PRE-REGISTRATION ROWS of PLAN_BIND E3 and E4 (docs/PLAN_E34.md), written 2026-10-26 AFTER E4's coding dry run (codes and trial
counts only; E2b's measures were already read) and E3's plan + a one-observation mechanics smoke (results/bind/e3_smoke; no endpoint
computed) with exact injection parity, BEFORE either read (asserted)."""
import glob
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "bind")
assert not os.path.exists(os.path.join(OUT, "e3.json")) and not os.path.exists(os.path.join(OUT, "e4.json"))
assert not glob.glob(os.path.join(OUT, "e3", "*.npz")), "E3 rollouts exist"
assert not ledger.load("bind_e3") and not ledger.load("bind_e4")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


common = dict(bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
              script="experiments/prereg/pbind_e34.py")
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pbind_e4", seed=0, command="pre-registration: PLAN_BIND E4",
                    metrics={"status": "PRE-REGISTERED PLAN_BIND E4 (re-analysis of read E2b measures; coding + model fixed first)",
                             "coding": "object 0 type / 1 instance (same-type instances or same CAT category); goal 0 / 1 instance object "
                                       "/ 2 relational sub-region (body with >= 2 goal regions, or a table region relative to a body)",
                             "model": "binomial GLM logit(err) ~ depth + scene_familiar + role, IRLS; task-cluster bootstrap B 2000",
                             "falsified_if": "b_depth <= 0 or 95% CI lo <= 0", "power": "about 45 rows, 24 tasks; depth 2 confounded "
                             "with new caddy scenes; weak", "files_sha256": sha(["experiments/bind_e4.py", "results/bind/e2b.json",
                                                                                  "results/bind/e1_scene.json", "docs/PLAN_E34.md"]),
                             "written_before": "the E4 fit"}, **common))
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pbind_e3", seed=0, command="pre-registration: PLAN_BIND E3",
                    metrics={"status": "PRE-REGISTERED PLAN_BIND E3 (frames)", "obs": "54 D4 training approach states, first-chunk "
                             "endpoints, K 4", "conditions": "obj +-5/+-10 cm; rig +-5 cm; yaw +-10 deg; inj (expert layers 16+17 outputs "
                             "from a different-scene source) base + obj 5 cm; inj16",
                             "H1": "mean(SS_world - SS_obj) CI lo > 0 and obj 5 cm gain CI lo > 0.5",
                             "H2": "inj16-17 gain at 5 cm CI hi < 0.5 and mean(dist_target - dist_delta_lock) CI lo > 0",
                             "H3": "obj 10 cm gain CI lo > 0.5",
                             "files_sha256": sha(["experiments/bind_e3.py", "experiments/bind_e3_serve.py", "experiments/bind_e3_pack.sbatch",
                                                  "results/bind/e3_plan.json", "docs/PLAN_E34.md", "experiments/d5_serve.py",
                                                  "experiments/d5_goal.py"]),
                             "written_before": "any E3 rollout"}, **common))
