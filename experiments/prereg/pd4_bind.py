"""PRE-REGISTRATION of probe D4 (docs/PLAN_D4.md; code experiments/d4_bind.py + experiments/d4_read.py), written 2026-10-25 AFTER the
training indexes, the fresh_d4 start states, a 2-observation render smoke (validity self-test) and a 2-observation GPU smoke
(results/d4/render/smoke_*, results/d4/gpu/smoke_*: runtime and batched-vs-single parity only; no endpoint, operator or bound body
computed), and BEFORE any D4 statistic (asserted below: no non-smoke policy output, no read.json, no d4_read row).

Readout: one body at a time displaced +-2 cm in x and y (validity: penetration growth <= 2 mm; 0.1 s pinned-robot settle, the body
within 5 mm of base + delta, every other movable body within 5 mm of its base end); the frozen policy's chunks on every render
with the same 8 noises (seed 11), a second-seed base pair (seed 21) as the noise floor; endpoint e = h + scale * sum of xyz
actions, mean over noises; J_Z central differences (2x2 xy block primary), bound body = argmax ||J_Z||_F over scored bodies (all 4
perturbations valid).
H1  pi05_l90c training observations, approach (named object) and place (target body) roles: filler == bound body in >= 0.8 AND
    CI lower bound > mean chance (1 / scored bodies), each role; p = max over roles of the one-sided two-level task-cluster bootstrap
    P*(hit - chance <= 0) (intersection-union).
H2  pi05_l90c training observations, each role: leave-type-out prediction of the filler's 4 endpoint responses from the mean
    filler operator of the other types, R^2 (through the origin, pooled) >= 0.5 and > the filler-shuffled control (operator pooled
    from a hash-chosen non-filler scored body); p = max over roles of the task-cluster bootstrap P*(R^2 - R^2_ctrl <= 0).
H3a pi05_l90c zs dev, wrong-object episodes (pre-grasp): the bound body is the grasped object in >= 0.7 (episode means); p = two-level
    bootstrap P*(rate_grasped - rate_named <= 0).
H3b the same on wrong-goal `other_region` episodes (post-grasp): the body actually placed on / in vs the own target.
Family: Holm over H1, H2, H3a, H3b at 0.05.  H4 (nested variance category > type > instance), the operator table, the bank's bytes,
pi05_libero's H1 / H2 and the composition read are descriptive.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "d4")
for d in glob.glob(os.path.join(OUT, "gpu", "*")):
    if not os.path.basename(d).startswith("smoke_"):
        assert not glob.glob(os.path.join(d, "*.npz")), f"policy output already exists: {d}"
assert not os.path.exists(os.path.join(OUT, "read.json")), "read.json already exists"
assert not ledger.load("d4_read"), "a d4_read row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/d4_bind.py"), ("read", "experiments/d4_read.py"),
                                               ("doc", "docs/PLAN_D4.md"), ("index_l90c", "results/d4/index_train_l90c.json"),
                                               ("index_libero", "results/d4/index_train_libero.json"),
                                               ("client", "experiments/libero_client.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
counts = {}
for m in ("l90c", "libero"):
    ix = json.load(open(files[f"index_{m}"], encoding="utf-8"))["tasks"]
    counts[m] = dict(tasks=len(ix), pre=sum(len(x["pre"]) for x in ix), carry_candidates=sum(len(x["carry_cand"]) for x in ix),
                     renders=len(glob.glob(os.path.join(OUT, "render", f"train_{m}", "*.npz"))))
counts["dev_renders"] = len(glob.glob(os.path.join(OUT, "render", "dev", "*.npz")))
print(json.dumps(dict(sha=sha, counts=counts), indent=1))
DISCLOSURE = ("seen before this row: OF D1-D3, R2a, R3b, R3c, R4, R4b published results (docs/ATLAS.md); R2a's episode types per dev "
              "task (used to choose the dev task subset: R2a has no wrong-object episode on the 15 recombination tasks, so H3a runs "
              "on the 7 part tasks where R2a saw them); the training index logs (task lists, frame counts, target bodies); a "
              "2-observation render smoke (all 32 perturbations valid; a self-test that drops one object onto another is caught by "
              "the penetration rule, 16.6 mm) and a 2-observation GPU smoke (6-7 s per observation; batched vs single-observation "
              "sampler parity max |diff| 1.6e-3 to 2.1e-3 in action units); no endpoint, operator, bound body or hypothesis "
              "statistic was computed.  The validity rule and every threshold were fixed before the smoke.  At the time of this "
              "row the training renders may be complete and the dev rollouts may be running; their counts are recorded above.  "
              "The brief's type-shuffled control is implemented as a filler-shuffled control (docs/PLAN_D4.md section 4); the "
              "type-label shuffle is reported as a descriptive reference.  No yaw step (OF D1: yaw adds nothing).  During "
              "development one python heredoc was issued on the login node; the harness blocked it before it ran.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pd4_bind", seed=0,
    metrics={"H1": "l90c training obs, approach + place: filler == argmax ||J||_F body in >= 0.8 and CI lo > chance, each role; "
                   "p = max over roles of P*(hit - chance <= 0), two-level task-cluster bootstrap B 10000",
             "H2": "l90c training obs, each role: leave-type-out R^2 (through origin, pooled) of the filler's 4 endpoint responses "
                   "from the other types' mean filler operator >= 0.5 and > filler-shuffled control; p = max over roles of "
                   "task-cluster bootstrap P*(R^2 - R^2_ctrl <= 0), B 2000",
             "H3a": "l90c zs dev wrong-object episodes (pre-grasp): bound body == grasped object in >= 0.7 (episode means); "
                    "p = P*(rate_grasped - rate_named <= 0), two-level bootstrap",
             "H3b": "l90c zs dev wrong-goal other_region episodes (post-grasp): bound body == the alt body placed on / in, as H3a",
             "family": "Holm over H1, H2, H3a, H3b at 0.05",
             "descriptive": "H4 nested sums of squares (category > type > instance > observation) of the filler J; per-type operator "
                            "table; bank bytes (16 B / operator: J 3x2 + offset 2 float16, plus keys); pi05_libero H1/H2; noise floor "
                            "share; dev success readout; composition read (bank operator + oracle pose -> e_hat)",
             "readout": "delta 0.02 m, perts +-x +-y; validity pen growth <= 2 mm, 50-step pinned settle within 5 mm; K 8 noises seed "
                        "11, floor pair seed 21; endpoint = h + scale * sum xyz actions, mean over noises; J central differences",
             "n": counts, "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any D4 policy output beyond the 2-observation smoke, any operator / bound body / hypothesis statistic"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pd4_bind.py",
    command="pre-registration: probe D4 (binding operators: H1 readout validity, H2 systematicity, H3a/b diagnosis; H4 descriptive)"))
