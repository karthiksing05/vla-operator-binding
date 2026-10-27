"""PRE-REGISTRATION of the L2b selection rule (plan L, docs/PLAN_LANG.md; exploratory study experiments/pl2b_criteria.py), written
2026-10-22 after the L2b dev read (ledger pl2b_criteria tag read) and BEFORE the L2b agent read any pi0.5 confirm closed-loop number
(results/lang/pl2/confirm_read.json, ledger pl2_confirm, results/robot/pl2_conf_*.json were not opened).

Disclosure.  The lead's message announcing the confirm read stated four confirm reference totals (l2sel 128/240, tli_sel 143,
tli_li 169, zs 93) and that the missing zero-shot fallback hurt confirm spatial:0 and spatial:3.  The rule below was chosen on
the dev table only.  That hint points toward the zero-shot guard (criterion d), and this row does NOT pick d: on dev, d's
leave-one-task-out total (118/320) is below plain demo-0 loss (130/320).

Rule (b_inter, two-fold interleaved split-demo validation; no tuned constant):
  - Split demo 0's scored frames (L2's 36, stride round(T/36)) into even and odd positions.
  - For each fold (fit on even, score on odd; fit on odd, score on even), run L2's whole demo selection on the fit half for
    every operator: sources, layer span, template, TLI pair and lambda.  This is pl2_compose.select_task, unchanged.
    Each operator's flow loss is then measured on the other half (same 4 draws per frame for every arm), as
    log(loss / zs loss).
  - Criterion per operator = the mean of that log ratio over the two folds; zs = 0.
  - Pick the argmin over M = (zs, word, tmpl, tli_sel, role).  Ties go to the first in that order, so zs can win.
  - Deploy the picked operator's configuration as L2 fitted it on ALL of demo 0 (results/lang/pl2/offline/pi05_<task>.json,
    its specs / tli arm).  Every pick is therefore an arm already rolled in the confirm read.
Dev evidence (8 tasks x 2 policies x 20 episodes; ledger pl2b_criteria tag read):
  - Totals:
    - b_inter: 135/320 (pi0.5 96/160, pi0 39/160), top-1 hit 12/16;
    - l2sel (demo-0 loss): 130/320 (91 / 39), hit 11/16;
    - MDL c, LOTO w: 130;
    - guard d, LOTO m: 118;
    - demo-1 loss (two demos): 110;
    - time-split b_time: 68.
  - Within-task Spearman over M: b_inter +0.32 vs demo-0 loss +0.19.
  - Caveat: the dev gain over l2sel is one task (pi0.5 goal:2, tmpl over tli_sel by 0.0002 in the criterion).
Read (offline, no new rollouts): the rule's confirm success = the sum over the 12 confirm tasks of the rolled success of its
pick (pl2_conf arms), against zs, l2sel, tli_sel and tli_li on the same 240 paired episodes.  Exploratory (one number, no test
pre-stated beyond the comparison).  The picks are fixed in this row (sha256 of the split files they come from)."""
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from lifelong_ood import ledger
import pl2b_criteria as B
import pl2_compose as PC

conf = PC.split()["confirm"]
files = {f"{k}|{x}": os.path.join(B.OUTB, "split", f"pi05_{PC.sh(k)}_{x}.json") for k in conf for x in B.RULE_FOLDS}
picks, vals = {}, {}
for k in conf:
    picks[k], vals[k] = B.rule_b_inter("pi05_libero", k)
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
print(json.dumps(picks, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="pl2b_rule", seed=13,
    metrics={"rule": "b_inter: 2-fold interleaved split-demo validation over M=(zs, word, tmpl, tli_sel, role); argmin of mean "
                     "held-out log(loss/zs loss); ties to the first in M; deploy the operator's full-demo-0 L2 configuration",
             "constants": "none (no penalty weight, no margin)",
             "chosen_on": "dev only (ledger pl2b_criteria tag read, 8 dev tasks x {pi0.5, pi0})",
             "dev_totals": {"b_inter": 135, "l2sel": 130, "c_mdl_loto": 130, "d_guard_loto": 118, "e_demo1": 110, "b_time": 68,
                            "tli_sel_always": 121, "zs_always": 74, "oracle_M": 148, "of": 320},
             "disclosure": "the lead's message gave confirm totals (l2sel 128, tli_sel 143, tli_li 169, zs 93 of 240) and said the missing "
                           "zs fallback hurt spatial:0 / spatial:3; no confirm file was opened; the guard d was NOT chosen (dev LOTO 118 < 130)",
             "confirm_tasks": conf, "backbone": "pi05_libero",
             "frozen_picks": picks, "criterion_values": vals, "split_files_sha256": sha,
             "read": "offline: sum over confirm tasks of the rolled pl2_conf success of the pick; compare zs, l2sel, tli_sel, tli_li",
             "written_before": "the L2b agent opened any confirm closed-loop result"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl2b_rule.py",
    command="pre-registration: L2b selection rule b_inter, frozen from dev, before the confirm read"))
