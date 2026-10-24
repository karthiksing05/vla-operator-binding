"""PRE-REGISTRATION of plan M1 (docs/PLAN_M1.md; code experiments/m1_compose_create.py), written 2026-10-24 after the frozen design
table (results/m1/plan.json: dev tasks, cover pairs, scene pairs, damage cells; no demo loss, no fit, no rollout) and BEFORE any
fit, rollout or read (asserted below: no results/m1/fits/*/fit.json and no results/m1/roll/* exists).

Question.  On LIBERO-90-C's 29 DEV held-out tasks, with pi05_l90c@5000 frozen and ONE demo per task, how do composing stored
language prototypes (TLI of pi05_l90c's training-cell instructions), creating a KV token, and composing-then-creating compare
with per-task LoRA, in success, GPU-seconds from demo arrival, bytes and damage?

Arms (every one gets the same demo; 20 fresh start states per task, fresh_m1 BASE 5700000, noise seed 13, paired):
  zs, compose_cov (demo-free coverage pair, lambda 24), compose_sel (demo flow loss over zs + {blind, scene} pairs x {24, 12, 36};
  ties and non-improvement keep zs), create (KV token n 4, 1000 steps, lr 1e-2, batch 32), cc (the token on top of compose_sel;
  = create where compose_sel kept zs), lora50 / lora200 / lora1000 (r16, lr 1e-4, batch 32, one 1000-step run with snapshots).
Damage: 10 training cells (results/m1/plan.json "damage"), stored init states 0-19, seed 0, paired with the parity read of
pi05_l90c@5000; routed = the store as served (only the 2 instruction-collision cells can change); unrouted = the paired dev
task's LoRA always on / token appended to the old prompt.

  H1  compose_sel > zs: more successes, exact two-sided sign test on discordant pairs.
  H2  create non-inferior to lora1000, margin 0.05: one-sided two-level cluster-bootstrap p (share of B = 10,000 bootstrap
      differences <= -0.05, stats.bootstrap_diff seed 0), AND token bytes <= lora bytes / 100.
  H3  cc non-inferior to create, margin 0.05, same test.
  Holm over H1-H3 at alpha 0.05 (H1's p is 1 when compose_sel does not have more successes).
  H4  (estimate gate, outside Holm) routed damage, mean success change over the 10 cells vs the parity base, >= -0.01 for both
      compose_sel and create.
  Secondary (descriptive): compose_cov vs zs, every arm vs zs, cc vs create / lora1000, the LoRA fit curve and budget-to-threshold
  (smallest LoRA budget whose pooled success reaches zs / compose_sel / create / cc), unrouted damage, GPU-s, bytes, by bin.

Disclosure.  No pi05_l90c rollout on any dev task exists before M1; the P2 / P3 field probes read offline observations of 14 dev
tasks (no success).  A serving smoke (2 episodes on the dev task's STORED init states, not fresh_m1) checks the served modes and is
excluded from every number.  The coverage rule keeps the instance qualifiers {top, front, back, left, right, middle} as content
words (pl2c dropped them); decided from the dev tasks' parses, before any number."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "m1")
assert not glob.glob(os.path.join(OUT, "fits", "*", "fit.json")), "an M1 fit already exists"
assert not glob.glob(os.path.join(OUT, "roll", "*")), "an M1 rollout already exists"
assert not ledger.load("m1_read"), "an M1 read already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("plan.json", "results/m1/plan.json"), ("script", "experiments/m1_compose_create.py"),
                                               ("doc", "docs/PLAN_M1.md"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan.json"], encoding="utf-8"))
assert len(plan["tasks"]) == 29 and len(plan["damage"]) == 10
frozen = {k: dict(bin=v["bin"], text=v["text"], blind=[v["blind"]["t1"], v["blind"]["t2"]],
                  scene=None if v["scene_cover"] is None else [v["scene_cover"]["t1"], v["scene_cover"]["t2"]],
                  demo_episode=v["demo_episode"]) for k, v in plan["tasks"].items()}
print(json.dumps(frozen, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pm1_compose_create", seed=13,
    metrics={"arms": plan["arms"], "frozen_tasks": frozen, "damage_cells": plan["damage"],
             "H1": "compose_sel > zs: more successes and exact two-sided sign test on discordant pairs",
             "H2": "create - lora1000 non-inferior at margin 0.05 (one-sided cluster-bootstrap p, B 10000) AND token bytes <= lora/100",
             "H3": "cc - create non-inferior at margin 0.05 (one-sided cluster-bootstrap p)",
             "family": "Holm over H1-H3 at alpha 0.05",
             "H4": "routed damage (mean change over the 10 cells vs parity base) >= -0.01 for compose_sel and create; estimate gate",
             "secondary": "compose_cov vs zs; every arm vs zs; cc vs create/lora1000; LoRA curve + budget-to-threshold; unrouted "
                          "damage; GPU-s; bytes; by bin",
             "states": "fresh_m1 (BASE 5700000, 20 per task), noise seed 13, paired; damage on stored init states 0-19, seed 0",
             "settings": dict(tok_n=4, tok_steps=1000, tok_lr=1e-2, batch=32, lora_rank=16, lora_lr=1e-4, lora_snaps=[50, 200, 1000],
                              lams=[24, 12, 36], lam_cov=24, scorer="pl2 DemoScorer K 4 seed 1 36 frames"),
             "disclosure": "no pi05_l90c dev rollout before M1; P2/P3 offline observations of 14 dev tasks; serving smoke on stored "
                           "init states excluded; qualifiers kept as content words",
             "files_sha256": sha, "written_before": "any M1 fit, rollout or read"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pm1_compose_create.py",
    command="pre-registration: M1 compose / create / fine-tune, design frozen, before any fit or rollout"))
