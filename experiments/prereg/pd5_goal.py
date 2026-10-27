"""PRE-REGISTRATION of probe D5 (docs/PLAN_D5.md; code experiments/d5_goal.py + d5_read.py + d5_serve.py + d5_pack.sbatch), written
2026-10-25 AFTER the observation index (results/d5/index.json), the sandbox parity check (results/d5/parity.json) and a one-observation
sandbox smoke per set (results/d5/sb_smoke: mechanics and timing only -- per-box release / success / steps and validity reasons were
printed; no readout point, J, bound body or hypothesis statistic was computed), and BEFORE any D5 statistic (asserted below: no
non-smoke sandbox output, no read.json, no d5_read row).

Readout (PRIMARY): from the carry state, a sandboxed simulator copy with ONE body displaced +-5 cm in x and in y (D4's validity rules)
is rolled forward with the frozen policy queried every 5 steps on the renders of that world (shared noise sets A1, A2); the hand's xy
at the first executed open command or at task success, whichever first (24 queries max), mean over A1 / A2; J_Z central differences
(2x2 xy block); bound body = argmax ||J_Z||_F over the bodies with all 4 perturbations valid.
G1  pi05_l90c training carry observations: the target body is the bound body in >= 0.8 AND the CI lower bound > mean chance;
    p = one-sided two-level task-cluster bootstrap P*(hit - chance <= 0), B 10000.
G2  the same observations: d4_read.h2 on the primary readout (leave-type-out R^2 >= 0.5 and > the filler-shuffled control);
    p = task-cluster bootstrap P*(R^2 - R^2_ctrl <= 0), B 2000.
G3  zs dev wrong-goal observations whose actual body differs from the own target (alt body, or the body nearest the zs release within
    10 cm for carried-elsewhere): bound body == actual body in >= 0.7; p = two-level bootstrap P*(rate_actual - rate_target <= 0).
Family: Holm over G1, G2, G3 at 0.05.  Secondary readouts (+-10 cm, xyz J, chunk mean, chunk end, object at release), the operator
table, sandbox fidelity, the region readout and the goal-role vision-anchor localisation are descriptive.
Stage B (R6 closed-loop smoke) runs only if G1 passes.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "d5")
assert not glob.glob(os.path.join(OUT, "sb", "*", "*.npz")), "non-smoke sandbox output already exists"
assert not os.path.exists(os.path.join(OUT, "read.json")), "read.json already exists"
assert not ledger.load("d5_read"), "a d5_read row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/d5_goal.py"), ("read", "experiments/d5_read.py"),
                                               ("serve", "experiments/d5_serve.py"), ("pack", "experiments/d5_pack.sbatch"),
                                               ("doc", "docs/PLAN_D5.md"), ("index", "results/d5/index.json"),
                                               ("parity", "results/d5/parity.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
ix = json.load(open(files["index"], encoding="utf-8"))
par = json.load(open(files["parity"], encoding="utf-8"))
counts = dict(train=len(ix["train"]), train_tasks=len({o["tid"] for o in ix["train"]}), dev=len(ix["dev"]),
              dev_wrong_goal=sum(o["type"] == "wrong_goal" for o in ix["dev"]), dev_success=sum(bool(o["success"]) for o in ix["dev"]))
print(json.dumps(dict(sha=sha, counts=counts, parity=par["rows"]), indent=1))
DISCLOSURE = (
    "seen before this row: every published result up to R5s (docs/ATLAS.md); D4's dev index episode types and the typer's alternative "
    "regions per wrong-goal episode (used to define G3's actual body and to see that :29 / :40 errors are same-body regions); D4's "
    "training carry index and render metadata (bodies per observation); the parity check (restored recorded state + recorded zs actions "
    "for 40 steps: eef within 0.4-2.7 mm of the recording on 5 of 6 states, 8.6 cm on a contact-chaotic :40 state; interleaved vs "
    "uninterrupted sandboxes equal to <= 2e-9 on 4 of 6, 4e-4 and 0.17 (the same :40 state) on 2 -- not bit-exact, disclosed); the "
    "batched-vs-single sampler parity (max |diff| 1.9e-3 to 2.7e-3 action units); a one-observation smoke per set: training "
    "tr_libero_10_1_e12_carry68 (7 bodies, 68 sandboxes, all released, 145 s) and dev dv_libero_90_1_t6_post125 (4 bodies, 3 rejected "
    "for penetration at both magnitudes, 17 sandboxes, every one ended by task success before any open command -- this is why the "
    "readout is release-OR-success, fixed after the smoke and before any readout was computed).  No readout point, J, bound body or "
    "hypothesis statistic was computed.  During development an empty python heredoc was issued on the login node; the harness denied "
    "it and it did not run.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pd5_goal", seed=0,
    metrics={"G1": "l90c training carry obs: target body == argmax ||J_xy||_F body (release-or-success hand xy, +-5 cm, noise A1/A2 "
                   "mean) in >= 0.8 and CI lo > chance; p = P*(hit - chance <= 0), two-level task-cluster bootstrap B 10000",
             "G2": "same obs: d4_read.h2 on the primary readout: leave-type-out R^2 >= 0.5 and > filler-shuffled control; "
                   "p = P*(R^2 - R^2_ctrl <= 0), B 2000",
             "G3": "zs dev wrong-goal obs with an actual body != own target (alt body; carried-elsewhere: nearest body to the zs release "
                   "within 10 cm): bound == actual in >= 0.7 (one obs per episode); p = P*(actual - target <= 0), two-level bootstrap",
             "family": "Holm over G1, G2, G3 at 0.05",
             "descriptive": "secondary readouts (+-10 cm A1; xyz J; first-chunk mean; first-chunk end; held object at release) with "
                            "the same G1-G3; validity per magnitude; release share; noise floor (B1/B2 vs A1/A2); operator table per "
                            "target type; dev success / recombination wrong-goal bound == target; sandbox fidelity (zs seeds vs the "
                            "recorded zs release); region readout; goal-role SigLIP anchor localisation (D3 region hit, 5 cm, median)",
             "readout": "mags 0.05 (primary) / 0.10; perts +-x +-y; validity: D4's (pen growth <= 2 mm, 50-step pinned settle within "
                        "5 mm); sandbox: full integration state + gripper ramp + OSC goal restored; replan 5, Q_MAX 24; noise seeds "
                        "set base + query index (A1 11e6, A2 12e6, B1 21e6, B2 22e6)",
             "stage_b_rule": "R6 exploratory closed-loop smoke only if G1 passes",
             "n": counts, "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any D5 sandbox output beyond the one-observation smokes, any readout / J / bound body / statistic"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pd5_goal.py",
    command="pre-registration: probe D5 (goal-role binding operator at the release point: G1 validity, G2 systematicity, G3 diagnosis)"))
