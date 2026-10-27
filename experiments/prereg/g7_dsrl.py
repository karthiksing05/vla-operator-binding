"""Pre-registration of plan G's read G7 (ledger m3_prereg tag g7_dsrl), written before any DSRL training or scored episode of g7
(experiments/g7_dsrl.py).  DSRL (arXiv 2506.15799; lifelong_ood/policy/dsrl.py, the authors' pi0.5 LIBERO configuration) on E3's
30 tasks: dsrlz steers the frozen pi0.5 (no demonstration), dsrls the task's scene-correct one-demo LoRA (29 tasks); 100 practice
episodes each on stored states 0-29 (seed 13); then the acceptance test on stored states 0-9 (10 episodes) and E3's fresh test
states (set b, 20 per task, seed 13), paired with E3's recorded arms."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g7_dsrl", seed=13,
    metrics={"arms": {"dsrlz": "DSRL on the frozen policy, 100 practice episodes, no demonstration",
                      "dsrls": "DSRL on the per-task one-demo LoRA, 100 practice episodes"},
             "hypotheses": {"H1": "E3's kept library is non-inferior to dsrlz with acceptance (task-level bootstrap 95% lower bound "
                                  "of library minus dsrlz+acc above -0.05)",
                            "H2": "E3's kept library is non-inferior to dsrls with acceptance (same criterion)",
                            "descriptive": "each DSRL arm vs zero-shot; dsrls+acc vs LoRA+acc; DSRL practice success in its first "
                                           "10 and last 20 training episodes"},
             "context": "E3 read; DSRL on a LoRA base was a token-era baseline (archive/2026-10-program/README_demo_rows.md); a "
                        "three-episode dsrlz training smoke (seed 99) preceded this registration",
             "written_before": "any DSRL training or scored episode of g7"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g7_dsrl.py", command="pre-registration of plan G's read G7 (DSRL)"))
