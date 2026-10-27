"""PRE-REGISTRATION of probe R2 (docs/PLAN_R2.md; code experiments/rb2_bank.py), written 2026-10-24 BEFORE any R2 fit, rollout or read
(asserted below).  pi05_l90c main/5000 frozen; the 15 LIBERO-90-C recombination dev tasks; NO target-task demo in any bank arm.

A bank of 17 role tokens (KVToken n 2 per layer, 36,864 bytes each; O while the gripper is open, G while it carries), one per part key
(O|object words; G|prep class|goal words; G|manip:verb|object words), each fitted by flow matching through the frozen VLA on the
role-phase frames of 40 training demos (pooled across every training text of lol/libero_l90c whose parse holds the key, <= 1024
frames; 1000 steps, Adam lr 1e-2, batch 32, seed 13).  Oracle retrieval: the tokens of the keys a dev task's parse holds.

Gate: 10 training cells (set cover of the keys), served with their instruction + their keys' tokens on stored init states 0-19,
seed 0; a cell passes if success >= parity - 0.15; keys of failing cells leave the admitted bank.  Junk check descriptive.
Arms on fresh_rb2 (BASE 7500000, 20 states per task, noise seed 13): zs, bank, bank_O, bank_G, ceiling (M1's create token).

  H1  (GO) gap closure >= 0.5: D = bank - 0.5 zs - 0.5 ceiling > 0 over the 300 paired episodes; one-sided two-level cluster
      bootstrap p = share of D* <= 0 (B 10000, seed 0); not testable (fails) if ceiling <= zs on fresh_rb2.
  H2  bank > zs: more successes + exact two-sided sign test on discordant pairs.
  H3  binding errors (fm_types wrong_object + wrong_goal) fewer under bank than zs: exact two-sided sign test on discordant pairs.
  H4  8 damage cells, admitted bank routed by parts vs parity: non-inferior at margin 0.05 (one-sided cluster-bootstrap p).
  Holm over H1-H4 at alpha 0.05.  GO = H1 passes.

Disclosure: see the `disclosure` metric."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "rb2")
assert not glob.glob(os.path.join(OUT, "fits", "*", "fit.json")), "an R2 fit already exists"
assert not glob.glob(os.path.join(OUT, "roll", "*")), "an R2 rollout already exists"
assert not ledger.load("rb2_fit"), "an R2 fit row already exists"
assert not ledger.load("rb2_read"), "an R2 read already exists"
assert not [r for r in ledger.load("m3_prereg") if r.get("tag") == "pr2_bank"], "pr2_bank already recorded"
assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
files = {n: os.path.join(ROOT, p) for n, p in (("code", "experiments/rb2_bank.py"), ("doc", "docs/PLAN_R2.md"),
                                               ("plan", "results/rb2/plan.json"), ("fresh_check", "results/rb2/fresh_check.json"),
                                               ("serve", "lifelong_ood/lang/transplant.py"), ("l4", "experiments/l4_store.py"),
                                               ("typer", "experiments/fm_types.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(sha, indent=1))
DISCLOSURE = ("seen before this row: M1's 15-task totals (zs 180, create 234; they define the go criterion) and the M1 / FM / L4 / R2a "
              "results in their docs; --inspect (the parse of the 15 tasks and which training texts hold each item: it set the key "
              "rule, verb dropped for O and prepositions classed for G, because L4's verb-keyed items left 5 of 15 object roles "
              "uncovered); --plan (bank, gate, damage tables); a fit smoke (RB2_SMOKE=1, 20 steps on G|in|tray, results/rb2/smoke, "
              "no ledger row, no rollout) to check the pipeline, excluded from every number; the fit job takes --kid (a key hash) "
              "because the GPU sbatch splits SCRIPT on spaces (pipeline only).  No R2 rollout exists and no pi05_l90c rollout on "
              "fresh_rb2 exists.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr2_bank", seed=13,
    metrics={"H1": "GO: D = bank - 0.5 zs - 0.5 ceiling > 0 (300 paired episodes, 15 recombination tasks x 20 fresh_rb2 states); "
                   "one-sided two-level cluster bootstrap p = share of D* <= 0 (B 10000, seed 0); not testable (fails) if ceiling <= zs",
             "H2": "bank > zs: more successes + exact two-sided sign test on discordant pairs",
             "H3": "binding errors (wrong_object + wrong_goal, fm_types) fewer under bank than zs: exact two-sided sign test on discordant pairs",
             "H4": "8 damage cells, admitted bank routed by parts vs parity (stored init 0-19, seed 0): non-inferior at margin 0.05, "
                   "one-sided cluster-bootstrap p",
             "family": "Holm over H1-H4 at alpha 0.05; GO = H1 passes",
             "gate": "10 gate cells (set cover), served with instruction + their keys' bank tokens; pass if success >= parity - 0.15; "
                     "keys of failing cells leave the admitted bank (used by bank / bank_O / bank_G and the damage read)",
             "junk": "per-demo flow-loss gain (without - with token) on fit demos and held-out demos; flag if held-out positive share < 0.5; "
                     "descriptive only",
             "arms": "zs, bank (oracle O + G), bank_O, bank_G, ceiling (M1 create token, the demonstrated ceiling)",
             "bank": {k: dict(role=e["role"], sources=len(e["sources"]), demos=len(e["fit_demos"]), frames=len(e["fit_frames"]))
                      for k, e in plan["bank"].items() if e["fitted"]},
             "oracle": {t: v["oracle"] for t, v in plan["dev"].items()},
             "gate_cells": plan["gate"], "damage_cells": plan["damage"], "settings": plan["settings"],
             "secondary": "per-arm success overall / by bin; bank_O and bank_G vs zs; bank vs ceiling; failure-type counts per arm; gate "
                          "table; junk statistics; bytes; fit A40-s",
             "disclosure": DISCLOSURE, "files_sha256": sha, "written_before": "any R2 fit, rollout or read"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr2_bank.py",
    command="pre-registration: R2 training-demo bank of role tokens, oracle retrieval, H1-H4, before any fit / rollout / read"))
