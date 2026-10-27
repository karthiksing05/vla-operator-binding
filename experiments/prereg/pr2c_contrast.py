"""PRE-REGISTRATION of probe R2c (docs/PLAN_R2C.md; code experiments/rc2_contrast.py), written 2026-10-24 BEFORE any R2c fit, held-out
score, rollout or read (asserted below).  pi05_l90c main/5000 frozen; R2's 16 part keys, key rule, fit demos and fit settings; only the
extraction objective changes, to a contrastive minimal-pair objective on TRAINING data only.

  L = FM(tok | p) / Lbar0_p + LAMBDA * relu(MARGIN - (FM(tok | c) - FM(none | c)) / FM(none | c)),  LAMBDA 1, MARGIN 0.02;
  c = role-phase frames of partner demos (training texts in a shared scene whose role filler differs; tier A = only that filler
  differs, tier B = same scene, used only without tier A; no partner = R2's pooled token kept, fails specificity).

STAGE 1 (offline, decisive).  Per token, on a fixed seeded held-out split (16 p demos never fitted by R2, 16 partner demos never fitted
here; <= 24 role-phase frames each; 4 fixed draws per frame; both tokens scored on identical frames and draws):
  gain_p = mean over held-out p demos of the relative flow-loss gain (loss without - loss with) / loss without;
  gain_c = the same on held-out partner demos;  spec = gain_p - gain_c.
  GATE: gain_p > 0 for >= 8 of 16 R2c tokens AND spec > 0 for >= 8 of 16 R2c tokens.  Fail -> STOP (no rollouts).
  R2's tokens are scored on the identical split for comparison (descriptive).

STAGE 2 (only if stage 1 passes): bank_rc2 (oracle O + G, L4 phase schedule, instruction kept) on the 15 recombination dev tasks x 20
fresh_rb2 states, noise seed 13, paired with R2's zs / bank / ceiling rows; damage on R2's 8 damage cells.  Holm over:
  H1 (GO) closure >= 0.5: D = bank_rc2 - 0.5 zs - 0.5 ceiling > 0, one-sided two-level cluster bootstrap (B 10000, seed 0);
  H2 bank_rc2 > zs, exact two-sided sign test on discordant pairs (directional);
  H3 bank_rc2 > R2's bank, exact two-sided sign test on discordant pairs (directional);
  H4 no damage: non-inferior to parity at margin 0.05 on the 8 damage cells (one-sided cluster bootstrap).
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "rc2")
assert not glob.glob(os.path.join(OUT, "fits", "*", "fit.json")), "an R2c fit already exists"
assert not glob.glob(os.path.join(OUT, "fits", "*", "token.pt")), "an R2c token already exists"
assert not os.path.exists(os.path.join(OUT, "stage1.json")), "a stage-1 read already exists"
assert not glob.glob(os.path.join(OUT, "roll", "*")), "an R2c rollout already exists"
assert not ledger.load("rc2_fit"), "an R2c fit row already exists"
assert not ledger.load("rc2_stage1"), "an R2c stage-1 row already exists"
assert not [r for r in ledger.load("m3_prereg") if r.get("tag") == "pr2c_contrast"], "pr2c_contrast already recorded"
files = {n: os.path.join(ROOT, p) for n, p in (("code", "experiments/rc2_contrast.py"), ("doc", "docs/PLAN_R2C.md"),
                                               ("plan", "results/rc2/plan.json"), ("rb2_code", "experiments/rb2_bank.py"),
                                               ("rb2_plan", "results/rb2/plan.json"), ("l4", "experiments/l4_store.py"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
plan = json.load(open(files["plan"], encoding="utf-8"))
print(json.dumps(sha, indent=1))
DISCLOSURE = ("seen before this row: R2's published results (docs/PLAN_R2.md section 7: totals zs 170 / bank 181 / ceiling 223, the "
              "junk check's fit gains 1.7-7.5% relative and held-out gains ~0, 9 of 16 flagged), which motivated the probe and set "
              "MARGIN 0.02 at the low end of R2's fit-demo gains; --plan of rc2 (partner texts, tiers, demo and frame counts per key; "
              "no loss, no outcome); a pipeline smoke (RC2_SMOKE=1, 20 steps on O|frying pan, results/rc2/smoke, no ledger row; held-out scores not printed or read; its log showed the contrast loss rising 2.7% at step 20 with the hinge at 0; LAMBDA / MARGIN unchanged).  No R2c fit, token, held-out score or rollout exists.  Stage-2 serving code is not written yet: "
              "it will reuse rb2_bank.py's serve / roll / type / read pipeline unchanged except the arm, and an amendment row will "
              "be recorded before any stage-2 rollout.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pr2c_contrast", seed=13,
    metrics={"objective": "L = FM(tok|p)/Lbar0_p + 1.0 * relu(0.02 - (FM(tok|c) - FM(none|c))/FM(none|c)); p batch = R2's (same fit "
                          "frames, sampling seed 13), contrast batch 32 partner role-phase frames on its own seed-14 streams",
             "stage1_gate": "gain_p > 0 for >= 8 of 16 R2c tokens AND spec = gain_p - gain_c > 0 for >= 8 of 16 (no-partner keys fail "
                            "spec); held-out 16 + 16 demos per key, <= 24 frames each, 4 draws; fail -> stop, no rollouts",
             "stage1_compare": "R2's tokens scored on the identical held-out frames and draws (descriptive)",
             "H1": "stage 2 GO: D = bank_rc2 - 0.5 zs - 0.5 ceiling > 0 over 300 paired fresh_rb2 episodes, one-sided two-level cluster "
                   "bootstrap (B 10000, seed 0)",
             "H2": "bank_rc2 > zs, exact two-sided sign test on discordant pairs",
             "H3": "bank_rc2 > R2's bank, exact two-sided sign test on discordant pairs",
             "H4": "R2's 8 damage cells, routed bank_rc2 vs parity, non-inferior at margin 0.05 (one-sided cluster bootstrap)",
             "family": "Holm over H1-H4 at alpha 0.05 (stage 2 only)",
             "keys": {k: dict(role=e["role"], tier=e["tier"], partners=len(e["partners"]), fit_c_demos=len(e["fit_c"]),
                              fit_c_frames=len(e["fit_c_frames"]), hold_p=len(e["hold_p"]), hold_c=len(e["hold_c"]))
                      for k, e in plan["keys"].items()},
             "settings": plan["settings"], "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any R2c fit, held-out score, rollout or read"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pr2c_contrast.py",
    command="pre-registration: R2c contrastive minimal-pair role-token bank, stage-1 offline gate, stage-2 H1-H4"))
